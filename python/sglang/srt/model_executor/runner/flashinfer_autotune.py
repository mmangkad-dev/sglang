# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

import contextlib
import datetime
import functools
import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_schedule,
    get_spec,
    max_prefill_buffer_tokens,
)
from sglang.srt.utils import empty_context, log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.base_runner import BaseRunner

logger = logging.getLogger(__name__)

FLASHINFER_AUTOTUNE_WORKAROUND_SKIPS = frozenset()


def get_flashinfer_autotune_skip_ops(model_runner: ModelRunner) -> set[str]:
    skip_ops = set(get_exec().kernel.flashinfer_autotune_skip_ops or ())
    skip_ops.update(FLASHINFER_AUTOTUNE_WORKAROUND_SKIPS)
    return skip_ops


def should_run_flashinfer_autotune(
    model_runner: ModelRunner, *, for_speculative_draft: bool = False
) -> bool:
    """Check if flashinfer autotune should be run."""
    mr = model_runner
    if mr.device != "cuda":
        return False
    if get_exec().kernel.disable_flashinfer_autotune:
        return False
    if get_exec().deterministic.enable_deterministic_inference:
        # Tuned configs are per problem shape, so the reduction order would follow
        # the batch shape.
        return False

    if for_speculative_draft:
        backend_str = (
            get_spec().speculative_moe_runner_backend
            or get_exec().moe.moe_runner_backend
        )
        a2a_backend_str = (
            get_spec().speculative_moe_a2a_backend or get_exec().moe.moe_a2a_backend
        )
    else:
        backend_str = get_exec().moe.moe_runner_backend
        a2a_backend_str = get_exec().moe.moe_a2a_backend

    # Autotune can run before the MoE backend globals are initialized, so read
    # the configured backends -- the draft leaves (`get_spec()`) or the target
    # leaves (`get_exec().moe`) above. CuteDSL v1 bypasses MoeRunner, and its
    # dummy dispatch can exceed DeepEP low-latency's token limit.
    if backend_str == "flashinfer_cutedsl" and a2a_backend_str == "deepep":
        return False

    # TODO smor- support other cases for flashinfer autotune, such as, mamba backend

    moe_needs_autotune = backend_str in [
        "flashinfer_trtllm",
        "flashinfer_trtllm_routed",
        "flashinfer_mxfp4",
        "flashinfer_cutedsl",
        "flashinfer_cutlass",
    ]

    from sglang.srt.layers.quantization.fp4_utils import (
        get_fp4_gemm_runner_backend,
    )

    model_quantization = mr.model_config.quantization
    model_uses_fp4 = model_quantization in (
        "modelopt_fp4",
        "modelopt_mixed",
    )
    fp4_gemm_needs_autotune = model_uses_fp4 and (
        get_fp4_gemm_runner_backend().is_flashinfer_cutlass()
        or get_fp4_gemm_runner_backend().is_flashinfer_cutedsl()
    )

    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_per_tensor_fp8_supported,
        resolve_mxfp8_dense_gemm_backend,
    )

    if model_quantization == "mxfp8":
        fp8_gemm_needs_autotune = resolve_mxfp8_dense_gemm_backend().is_flashinfer()
    elif model_quantization in ("modelopt", "modelopt_fp8", "modelopt_mixed"):
        fp8_gemm_needs_autotune = flashinfer_per_tensor_fp8_supported()
    else:
        fp8_gemm_needs_autotune = False

    if not (moe_needs_autotune or fp4_gemm_needs_autotune or fp8_gemm_needs_autotune):
        return False

    if torch.cuda.get_device_capability()[0] < 9:
        return False

    if mr.spec_algorithm.is_speculative():
        return mr.is_draft_worker if for_speculative_draft else not mr.is_draft_worker

    return True


def _autotune_tactic_sync_group(
    tp_group: GroupCoordinator,
) -> Optional[torch.distributed.ProcessGroup]:
    """CPU group over the ranks that must agree on the tuned tactics.

    Per-rank timing noise alone makes each rank's ``argmin`` pick a different
    tactic for the same shape. FlashInfer all-reduces the timings over this
    group so every rank minimizes over the same numbers. TP is the scope: those
    ranks run the same dummy forward, and PP stages are already separate groups.
    """
    if tp_group.world_size <= 1:
        return None
    # The CPU group keeps the reduction of these scalars off the profiled stream.
    return tp_group.cpu_group


@contextlib.contextmanager
def _autotune_process_group(group: Optional[torch.distributed.ProcessGroup]):
    """Set FlashInfer's timing-reduction group, restoring the previous one after."""
    from flashinfer.autotuner import (
        get_autotune_process_group,
        set_autotune_process_group,
    )

    previous = get_autotune_process_group()
    set_autotune_process_group(group)
    try:
        yield
    finally:
        set_autotune_process_group(previous)


def _autotune_measurement_policy():
    """How FlashInfer should measure candidates, matched to how sglang serves.

    Whether per-call *host* cost counts is the one decision: a CUDA graph pays
    it once at capture, eager execution pays it every call, and the two rank
    host-heavy candidates very differently. The policy is part of the store's
    environment identity, so entries measured one way are never served the
    other way.
    """
    from flashinfer import MeasurementPolicy

    # Constructed even when fully default (its manifest contribution is then
    # empty) so an unusable value fails loudly here rather than at the first
    # profile.
    return MeasurementPolicy(
        execution_mode=envs.SGLANG_FLASHINFER_AUTOTUNE_MEASURE.get(),
        cold_l2=envs.SGLANG_FLASHINFER_AUTOTUNE_COLD_L2.get(),
    )


@functools.lru_cache(maxsize=None)
def _autotune_store_root(reuse_cache: bool) -> Optional[Path]:
    """Where FlashInfer's managed autotune store lives -- placement only.

    Identity lives below this directory and belongs to FlashInfer: schema
    version, an environment hash over the FlashInfer / CUDA / cuBLAS / cuDNN /
    GPU / measurement-policy manifest, one entry file per tuned operation. So
    no choice of root can mix incompatible entries, and nothing about the
    model, parallelism or skip-op set belongs in it.

    ``None`` defers to FlashInfer's own root, which is what setting
    ``FLASHINFER_AUTOTUNE_CACHE_DIR`` asks for. Cached because the no-reuse
    root is timestamped and every context in a process must resolve to the
    same store, or the finalize reload cannot see an earlier one's winners.
    """
    if os.getenv("FLASHINFER_AUTOTUNE_CACHE_DIR"):
        return None
    base = Path(envs.SGLANG_CACHE_DIR.get()).expanduser() / "flashinfer" / "autotune"
    if reuse_cache:
        return base
    # Reuse off: still publish, just into a run-scoped store, so the result
    # stays inspectable and the finalize reload below has canonical entries to
    # re-read. Per-process, since ranks tune independently into it.
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / "runs" / f"{stamp}_{os.getpid()}"


def _managed_autotune_store(root: Optional[Path], manifest: dict[str, str]):
    """FlashInfer's store object, used only to address it (log / digest /
    clear); tuning reads and writes go through ``autotune_v2``."""
    from flashinfer.autotune_cache import ManagedAutotuneCache

    return ManagedAutotuneCache(manifest=manifest, root=root)


def _autotune_store_digest(store) -> str:
    """Hash of the tuned entries this rank would serve from *store*.

    Covers the tactics, not just which operations are present: ranks holding
    the same keys with different winners still have to converge.
    """
    digest = hashlib.sha256()
    try:
        paths = sorted(store.entries_dir.glob("*.json"))
    except OSError:
        return ""
    for path in paths:
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        digest.update(path.name.encode())
        digest.update(payload)
    return digest.hexdigest()


def _converge_autotune_store(store, group: torch.distributed.ProcessGroup) -> None:
    """Enter tuning with the same tuned entries on every rank, or with none.

    A cache hit skips a profile, and profiling is collective (FlashInfer
    all-reduces the candidate timings over *group*), so ranks that disagree
    about what is already tuned desync the reduction and hang. Ranks sharing a
    filesystem read one store and agree by construction; ranks on separate
    nodes hold independent stores, so discard diverging entries and tune from
    scratch -- the run republishes identical tactics everywhere, which makes
    this self-healing rather than repeated at every start.

    Discarding is scoped to this environment's entries. A process warming up
    outside *group*, such as another PP stage on the same node, may lose an
    entry it just published and re-tune that operation.
    """
    digests: list[str] = [""] * torch.distributed.get_world_size(group)
    torch.distributed.all_gather_object(
        digests, _autotune_store_digest(store), group=group
    )
    if len(set(digests)) == 1:
        return
    log_info_on_rank0(
        logger,
        "FlashInfer autotune: per-rank managed stores disagree, discarding "
        "their entries and tuning from scratch so all ranks agree on the "
        "tactics.",
    )
    try:
        for path in store.entries_dir.glob("*.json"):
            path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning(
            "FlashInfer autotune: could not clear %s: %s", store.entries_dir, e
        )
    # Tuning publishes as it goes, so no rank may still be deleting once its
    # peers have started.
    torch.distributed.barrier(group=group)


@contextlib.contextmanager
def flashinfer_autotune_context(model_runner: ModelRunner, *, run_lm_head: bool):
    from flashinfer import autotune_v2, autotune_v2_reload
    from flashinfer.autotuner import _collect_metadata

    mr = model_runner
    sync_group = _autotune_tactic_sync_group(mr.tp_group)
    reuse_cache = envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.get()
    policy = _autotune_measurement_policy()
    cache_root = _autotune_store_root(reuse_cache)
    manifest = {**_collect_metadata(), **policy.manifest_fields()}
    store = _managed_autotune_store(cache_root, manifest)
    if reuse_cache:
        if sync_group is not None:
            _converge_autotune_store(store, sync_group)
        logger.info("Running FlashInfer autotune with cache: %s", store.env_dir)
    else:
        # A run-scoped store starts empty on every rank, so there is nothing
        # for the convergence gate above to reconcile.
        logger.info(
            "Running FlashInfer autotune (cache reuse DISABLED via "
            "SGLANG_FLASHINFER_AUTOTUNE_CACHE=0); writing fresh result to: %s",
            store.env_dir,
        )

    # Run warmup on the non-default stream to avoid NCCL 2.29+ cudaMemcpyBatchAsync
    # calls on default stream (unsupported by CUDA) when --enable-symm-mem is used.
    mr.forward_stream.wait_stream(torch.cuda.current_stream())
    with torch.get_device_module(mr.device).stream(mr.forward_stream):
        from sglang.srt.layers.logits_processor import autotune_dummy_run_mode

        skip_ops = get_flashinfer_autotune_skip_ops(mr)
        with (
            _autotune_process_group(sync_group),
            autotune_v2(
                mode="tune",
                cache_root=cache_root,
                measurement_policy=policy,
                skip_ops=skip_ops,
            ),
            autotune_dummy_run_mode(run_lm_head=run_lm_head),
        ):
            yield
        if sync_group is not None:
            # Finalize step: once every rank has stopped tuning, drop the
            # locally measured winners and re-read the store, so all ranks
            # serve the byte-identical tactics a restart would serve from
            # disk.
            torch.distributed.barrier(group=sync_group)
            autotune_v2_reload()
    torch.cuda.current_stream().wait_stream(mr.forward_stream)
    logger.info("FlashInfer autotune completed.")


def run_flashinfer_autotune_forward(
    model_runner: ModelRunner, forward_fn: Callable[[], None], *, run_lm_head: bool
) -> None:
    """Run flashinfer autotune forward."""
    with flashinfer_autotune_context(model_runner, run_lm_head=run_lm_head):
        forward_fn()


def maybe_flashinfer_autotune_speculative_draft(
    runner: BaseRunner,
    forward_fn: Callable[[], None],
    *,
    post_warmup_hook: Optional[Callable[[], None]] = None,
    run_lm_head: bool = True,
) -> None:
    """Run speculative draft flashinfer autotune."""
    mr = runner.model_runner
    phase_key = f"{runner.__class__.__module__}.{runner.__class__.__qualname__}"
    tuned_phases = getattr(mr, "_flashinfer_spec_draft_autotuned_phases", None)
    if tuned_phases is None:
        tuned_phases = set()
        mr._flashinfer_spec_draft_autotuned_phases = tuned_phases
    if phase_key in tuned_phases:
        return
    if (
        not mr.spec_algorithm.is_speculative()
        or not mr.is_draft_worker
        or not should_run_flashinfer_autotune(mr, for_speculative_draft=True)
    ):
        return

    def run_and_reset():
        forward_fn()
        if post_warmup_hook is not None:
            post_warmup_hook()

    run_flashinfer_autotune_forward(mr, run_and_reset, run_lm_head=run_lm_head)
    tuned_phases.add(phase_key)


def maybe_flashinfer_autotune_extend(
    runner: BaseRunner, *, decode_num_tokens: int
) -> None:
    """Also autotune kernels at the prefill token ceiling.

    The decode-shaped autotune only covers token counts up to the decode
    batch size, so larger prefill/extend batches fall outside the tuned
    buckets and run flashinfer's default heuristic — which can be far
    slower than the tuned tactic (e.g. trtllm-gen fp4 MoE is ~30% slower
    untuned at >=8k tokens on sm100). One extra forward at the largest
    per-rank extend token count tunes all buckets up to it.
    """
    mr = runner.model_runner
    # Prefer the per-rank scheduler buffer while preserving the legacy ceiling
    # when chunked prefill is disabled.
    num_tokens = max_prefill_buffer_tokens() or get_schedule().max_prefill_tokens
    if num_tokens <= (decode_num_tokens or 0):
        return  # decode-shaped autotune already covered these buckets
    # DSpark's dummy forward is TARGET_VERIFY-shaped and misses large prefill GEMMs.
    prefill_autotune = getattr(mr.model, "autotune_prefill_kernels", None)
    wants_prefill_autotune = getattr(mr.model, "wants_prefill_autotune", None)
    if wants_prefill_autotune is not None and not wants_prefill_autotune():
        # Entering the autotune context loads / saves the tactic cache and syncs
        # ranks, so a model that has nothing to tune must decline before it.
        prefill_autotune = None
    if prefill_autotune is not None and mr.is_generation and not mr.is_draft_worker:
        with flashinfer_autotune_context(mr, run_lm_head=False):
            tuned = prefill_autotune(num_tokens, dtype=mr.dtype)
        if tuned:
            return

    if not envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND.get():
        return
    is_pd_prefill_target = (
        get_disagg().disaggregation_mode == "prefill" and not mr.is_draft_worker
    )
    if not mr.is_generation or (
        mr.spec_algorithm.is_speculative() and not is_pd_prefill_target
    ):
        # Ordinary speculative runners force TARGET_VERIFY; PD prefill targets
        # have no draft-side state and preserve the requested EXTEND mode.
        return
    # Multimodal generation wrappers can still run this text-only EXTEND dummy;
    # an incompatible model should fail the explicit opt-in visibly.

    if mr.attn_backend.extend_dummy_seqs_capped_by_req_pool:
        pool_size = mr.req_to_token_pool.size
        num_tokens_per_req = (num_tokens + pool_size - 1) // pool_size
    else:
        # Packed dummies tune measurably worse tactics for the same token
        # bucket, so pack only where the backend would otherwise crash. None
        # (not 1) keeps the backend's own seq_len_fill_value in _dummy_run.
        num_tokens_per_req = None
    per_req = num_tokens_per_req or 1
    batch_size = (num_tokens + per_req - 1) // per_req
    num_tokens = batch_size * per_req

    buffers = runner._alloc_dummy_decode_buffers(
        batch_size,
        num_tokens_per_req=per_req,
        allocate_logits_buffer=False,
    )
    canary_run_ctx = (
        c.with_active_single_forward_manager(0)
        if (c := mr.canary_manager) is not None
        else empty_context()
    )

    forward_fn = functools.partial(
        runner._dummy_run,
        batch_size=batch_size,
        buffers=buffers,
        run_ctx=canary_run_ctx,
        forward_mode_override=ForwardMode.EXTEND,
        extend_num_tokens_per_req=num_tokens_per_req,
    )

    log_info_on_rank0(
        logger,
        f"FlashInfer autotune: extra EXTEND pass at {num_tokens} tokens "
        f"({batch_size} seqs x {per_req} tokens).",
    )
    try:
        run_flashinfer_autotune_forward(mr, forward_fn, run_lm_head=False)
    except torch.OutOfMemoryError:
        if _autotune_tactic_sync_group(mr.tp_group) is not None:
            # Tuning is collective: this rank has stopped reducing while its
            # peers wait on the next tactic, so skipping the pass would hang
            # them. Fail instead of degrading alone.
            raise
        # The pass is an optimization; without headroom for the extend-shaped
        # forward, fall back to untuned extend buckets instead of failing.
        log_info_on_rank0(
            logger,
            "FlashInfer extend autotune skipped: not enough free memory "
            f"for a {num_tokens}-token dummy forward.",
        )
    finally:
        # release dummy buffers before capture measures free memory
        del forward_fn, buffers
        torch.cuda.empty_cache()
