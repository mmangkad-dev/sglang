"""DeepSeek V4 trtllm-gen sparse MLA backend for SM100/SM103.

Decode and varlen prefill use a uniform 512-dim FP8 KV cache. Shared metadata
construction preserves the base backend's CUDA-graph replay semantics.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import (
    SWA_WINDOW,
    DeepseekV4AttnBackend,
    DeepseekV4MultiStepBackend,
)
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_resources,
    get_schedule,
    get_spec,
    max_prefill_buffer_tokens,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import DSV4AttnMetadata
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Shared zero-initialized workspace managed by the persistent-buffer lifecycle.
_TRTLLM_GEN_WORKSPACE_SIZE_MB = 128


def _get_trtllm_workspace_buffer(device: torch.device) -> torch.Tensor:
    from sglang.srt.runtime_context import get_buffer

    return get_buffer(
        "trtllm_dsv4_zero_workspace",
        lambda: torch.zeros(
            _TRTLLM_GEN_WORKSPACE_SIZE_MB * 1024 * 1024,
            dtype=torch.int8,
            device=device,
        ),
    )


def _trtllm_query_row_capacity() -> int:
    """Bound query rows across prefill chunks and speculative decode batches.

    The DSv4 hook rejects the backend when chunked prefill is disabled, so the
    prefill chunk bound is always finite here.
    """
    schedule = get_schedule()
    rows = max(
        schedule.max_prefill_tokens or 0,
        max_prefill_buffer_tokens(),
    )
    spec = get_spec()
    rows_per_req = (
        (spec.speculative_num_draft_tokens or 1)
        if spec.speculative_algorithm is not None
        else 1
    )
    rows = max(rows, (schedule.max_running_requests or 0) * rows_per_req)
    return max(rows, 1)


# FlashInfer's counter allocator is patched once per process. The hook holds no
# reference to any owner -- it resolves the current one from the resources slot
# on each call -- so an engine rebuild drops its owner and buffers without
# leaving a retained closure or stacking a second layer of dispatch.
_ALLOCATOR_HOOK = None
_FLASHINFER_COUNTER_ALLOCATOR = None


def _flashinfer_counter_allocator():
    """FlashInfer's own allocator, memoized before any interception."""
    global _FLASHINFER_COUNTER_ALLOCATOR
    if _FLASHINFER_COUNTER_ALLOCATOR is None:
        import flashinfer.mla._core as fi_core

        _FLASHINFER_COUNTER_ALLOCATOR = (
            fi_core._get_trtllm_gen_multi_ctas_kv_counter_buffer
        )
    return _FLASHINFER_COUNTER_ALLOCATOR


def _install_counter_allocator_hook() -> None:
    """Route DSv4 counter allocations to the current owner, once per process.

    Idempotent by identity, not by a flag: a caller that restores FlashInfer's
    attribute (a test tearing down its patch) gets the hook reinstalled rather
    than silently losing it to a stale "already installed" bool. Only ever one
    dispatch object exists, so reinstalling cannot stack a second layer.
    """
    global _ALLOCATOR_HOOK
    import flashinfer.mla._core as fi_core

    if fi_core._get_trtllm_gen_multi_ctas_kv_counter_buffer is _ALLOCATOR_HOOK:
        return
    original = _flashinfer_counter_allocator()

    if _ALLOCATOR_HOOK is None:

        def dispatch(batch_size, num_qo_heads, sm_count, device):
            owner = get_resources().trtllm_dsv4_kv_counter
            if owner is None or not owner.launch_in_flight:
                # Not a DSv4 launch (or no engine owns one): leave FlashInfer's
                # per-caller allocation alone so concurrent MLA callers in the
                # process keep their own counters.
                return original(batch_size, num_qo_heads, sm_count, device)
            return owner.buffer_for(
                num_qo_heads=num_qo_heads, sm_count=sm_count, device=device
            )

        _ALLOCATOR_HOOK = dispatch

    fi_core._get_trtllm_gen_multi_ctas_kv_counter_buffer = _ALLOCATOR_HOOK


class TrtllmKvCounterOwner:
    """Owns the trtllm-gen sparse-MLA multi-CTA KV counter buffers.

    FlashInfer sizes that private buffer from the ``batch_size`` it is handed
    and allocates a fresh one per call. Two things make that unsuitable here.
    Varlen prefill is under-sized: ``trtllm_batch_decode_sparse_mla_dsv4``
    passes the *request* count while the VarSeq launcher reshapes
    ``sparse_indices`` to one row per query token and indexes the counters by
    query row, so a chunk needs ``sum_q`` rows. And a fresh zeroed buffer per
    call is wasted work on the decode path -- the kernel resets the counters
    itself, and inside a captured graph the memset replays every step.

    So every DSv4 launch is bracketed by ``dsv4_launch()`` and served one
    persistent buffer per counter geometry, sized by the configured row bound;
    unrelated MLA callers keep FlashInfer's own allocation. Buffers and the
    bound live on a ``get_resources()`` slot, so ``reset_context()`` drops them
    with the engine that sized them.
    """

    def __init__(self) -> None:
        # Capacity in query rows: sum_q for prefill, requests x draft tokens
        # for decode.
        self.capacity_rows: int = 0
        # (device, num_qo_heads, sm_count) -> buffer, each sized for
        # _allocated_rows. Keyed because a draft runner may launch a different
        # head count than its target, and rekeying must not thrash one slot.
        self._buffers: dict = {}
        self._allocated_rows: int = 0
        self._launch_depth: int = 0

    # --- capacity ---------------------------------------------------------

    def reserve(self, capacity_rows: int) -> None:
        """Raise the row bound, regrowing already-allocated buffers to match."""
        if capacity_rows > self.capacity_rows:
            self.capacity_rows = capacity_rows
            if self._buffers and self._allocated_rows < self.capacity_rows:
                # A later runner (a draft backend, a second engine) raised the
                # bound after the buffers existed. Regrow now, while we are
                # outside graph capture, so no launch can pass check() against
                # a bound the allocation does not cover.
                for key in list(self._buffers):
                    self._allocate(*key)
        _install_counter_allocator_hook()

    def check(self, num_rows: int) -> None:
        """Reject a launch the counter buffers cannot cover.

        A plain exception, not an assert: an over-capacity launch scribbles
        past the buffer, so this must fire even under ``python -O``. Once
        buffers exist, the bound that matters is the one they were sized for --
        not the one that was merely requested.
        """
        bound = self._allocated_rows if self._buffers else self.capacity_rows
        if num_rows > bound:
            raise RuntimeError(
                f"trtllm-gen launch with {num_rows} query rows exceeds the "
                f"persistent multi-CTA counter capacity of {bound} rows, "
                "derived from --max-prefill-tokens / --chunked-prefill-size "
                "and --max-running-requests x speculative draft tokens; "
                "lower --chunked-prefill-size."
            )

    # --- launches ---------------------------------------------------------

    @property
    def launch_in_flight(self) -> bool:
        return self._launch_depth > 0

    @contextmanager
    def dsv4_launch(self, num_rows: int):
        """Bracket a DSv4 launch so the hook serves it a persistent buffer.

        Every DSv4 launch is bracketed, decode included: whether the launch
        exceeds FlashInfer's own sizing decides the *capacity* it needs, not
        whether it may reuse a buffer.
        """
        self.check(num_rows)
        self._launch_depth += 1
        try:
            yield
        finally:
            self._launch_depth -= 1

    # --- buffers ----------------------------------------------------------

    def _allocate(
        self, device: torch.device, num_qo_heads: int, sm_count: int
    ) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the trtllm-gen multi-CTA counter buffer must be created "
                "outside cuda-graph capture; it is normally allocated during "
                "eager warmup"
            )
        rows = max(self.capacity_rows, 1)
        # The kernel resets the counters to zero at the end of every launch, so
        # a zero-initialized buffer stays reusable across launches without
        # re-zeroing (FlashInfer documents this on the original allocator).
        buf = _flashinfer_counter_allocator()(rows, num_qo_heads, sm_count, device)
        self._buffers[(device, num_qo_heads, sm_count)] = buf
        self._allocated_rows = rows
        logger.info(
            "trtllm-gen multi-CTA counters: %d bytes for %d query rows, "
            "%d heads on %s; reused across DSv4 launches.",
            buf.numel(),
            rows,
            num_qo_heads,
            device,
        )
        return buf

    def buffer_for(
        self, *, num_qo_heads: int, sm_count: int, device: torch.device
    ) -> torch.Tensor:
        """Get-or-grow the persistent buffer for one launch's counter geometry."""
        buf = self._buffers.get((device, num_qo_heads, sm_count))
        if buf is None or self._allocated_rows < self.capacity_rows:
            return self._allocate(device, num_qo_heads, sm_count)
        return buf


def get_trtllm_kv_counter_owner() -> TrtllmKvCounterOwner:
    """The process-level counter-buffer owner, on its resources slot."""
    resources = get_resources()
    owner = resources.trtllm_dsv4_kv_counter
    if owner is None:
        owner = TrtllmKvCounterOwner()
        resources.trtllm_dsv4_kv_counter = owner
    return owner


class DeepseekV4TrtllmAttnBackend(DeepseekV4AttnBackend):
    """DSV4 attention through the trtllm-gen sparse MLA kernel."""

    trtllm_attn: bool = True

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        self.trtllm_kv_counters = get_trtllm_kv_counter_owner()
        self.trtllm_kv_counters.reserve(_trtllm_query_row_capacity())
        super().__init__(
            model_runner,
            skip_prefill=skip_prefill,
            speculative_step_id=speculative_step_id,
            topk=topk,
            speculative_num_steps=speculative_num_steps,
        )
        assert self.token_to_kv_pool.uniform_fp8, (
            "the trtllm backend requires the uniform-FP8 DSv4 KV pool."
        )
        assert not envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get(), (
            "--dsv4-attn-backend trtllm does not support "
            "SGLANG_OPT_USE_ONLINE_COMPRESS yet."
        )
        # CP round-robin reindexing breaks VarSeq's per-request query packing.
        assert get_parallel().attn_cp_size == 1, (
            "--dsv4-attn-backend trtllm does not support "
            "context parallelism (attn_cp_size > 1) yet."
        )
        self.trtllm_workspace_buffer = _get_trtllm_workspace_buffer(self.device)

    def _forward_trtllm(
        self,
        *,
        q: torch.Tensor,
        layer: RadixAttention,
        compress_ratio: Literal[0, 4, 128],
        core_attn_metadata: DSV4AttnMetadata,
        forward_batch: ForwardBatch,
        attn_sink: torch.Tensor,
        swa_page_indices: torch.Tensor,
        extra_indices: Optional[torch.Tensor],
        extra_topk_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert attn_sink is not None
        if (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            return self._forward_trtllm_decode(
                q=q,
                layer=layer,
                compress_ratio=compress_ratio,
                core_attn_metadata=core_attn_metadata,
                attn_sink=attn_sink,
                swa_page_indices=swa_page_indices,
                extra_indices=extra_indices,
                extra_topk_lengths=extra_topk_lengths,
            )
        assert forward_batch.forward_mode.is_extend_without_speculative(), (
            "uniform-FP8 pool cannot be read by the packed FlashMLA "
            f"kernels; unsupported forward mode "
            f"{forward_batch.forward_mode} under "
            "--dsv4-attn-backend trtllm"
        )
        return self._forward_trtllm_prefill(
            q=q,
            layer=layer,
            compress_ratio=compress_ratio,
            forward_batch=forward_batch,
            attn_sink=attn_sink,
            swa_page_indices=swa_page_indices,
            extra_indices=extra_indices,
            extra_topk_lengths=extra_topk_lengths,
        )

    def _get_trtllm_bmm_scales(self, layer: RadixAttention) -> Tuple[float, float]:
        """Return host scales; KV uses the store path's fixed unit scale.

        Tensor scales corrupt split-KV reduction on FlashInfer < 0.6.13.
        """

        assert layer.k_scale_float is None or layer.k_scale_float == 1.0, (
            "--dsv4-attn-backend trtllm stores KV with a "
            "fixed per-tensor scale of 1.0; a non-unit checkpoint kv-cache "
            f"scale (k_scale_float={layer.k_scale_float}) is not supported yet."
        )
        return (self.softmax_scale, 1.0)

    def _trtllm_kv_cache_views(
        self, layer_id: int, compress_ratio: Literal[0, 4, 128]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return HND views of the uniform-FP8 SWA and compressed pools.

        SWA-only layers pass the SWA pool as the required compressed tensor;
        ``sparse_topk_lens`` masks that region.
        """

        token_to_kv_pool = self.token_to_kv_pool
        swa_buf = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)
        swa_page_size = token_to_kv_pool.swa_kv_pool.page_size
        swa_kv_cache = swa_buf.view(swa_buf.shape[0], 1, swa_page_size, 512)
        if compress_ratio == 0:
            compressed_kv_cache = swa_kv_cache
        else:
            extra_buf = token_to_kv_pool.get_extra_key_buffer(layer_id)
            extra_page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
            compressed_kv_cache = extra_buf.view(
                extra_buf.shape[0], 1, extra_page_size, 512
            )
        return swa_kv_cache, compressed_kv_cache

    def _forward_trtllm_decode(
        self,
        *,
        q: torch.Tensor,
        layer: RadixAttention,
        compress_ratio: Literal[0, 4, 128],
        core_attn_metadata: DSV4AttnMetadata,
        attn_sink: torch.Tensor,
        swa_page_indices: torch.Tensor,
        extra_indices: Optional[torch.Tensor],
        extra_topk_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run sparse MLA decode with preallocated metadata tables."""

        from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

        bs, num_heads, head_dim = q.shape
        assert head_dim == 512

        # Draft-extend metadata predates DP MAX_LEN padding (#27091). Run only
        # its covered rows and leave the discarded padding output finite.
        n_meta_rows = core_attn_metadata.seq_lens_casual.shape[0]
        out_pad_tail = None
        if n_meta_rows < bs:
            out_pad_tail = torch.zeros(
                (bs, num_heads, 512), dtype=torch.bfloat16, device=q.device
            )
            q = q[:n_meta_rows]
            swa_page_indices = swa_page_indices[:n_meta_rows]
            if extra_indices is not None:
                extra_indices = extra_indices[:n_meta_rows]
            if extra_topk_lengths is not None:
                extra_topk_lengths = extra_topk_lengths[:n_meta_rows]
            bs = n_meta_rows

        # Only the c4 tail and lens vary by layer; other table data is prebuilt.
        assert swa_page_indices.shape == (bs, SWA_WINDOW)
        if compress_ratio == 0:
            # Use the metadata view backed by 64-row-aligned storage because
            # the VarSeq kernel reads table rows to the tile boundary.
            sparse_indices = core_attn_metadata.swa_page_indices
            sparse_topk_lens = core_attn_metadata.trtllm_swa_lens
        elif compress_ratio == 128:
            sparse_indices = core_attn_metadata.trtllm_c128_indices
            sparse_topk_lens = core_attn_metadata.trtllm_c128_lens
        else:
            sparse_indices = core_attn_metadata.trtllm_c4_indices
            sparse_topk_lens = core_attn_metadata.trtllm_c4_lens
        assert sparse_indices is not None and sparse_topk_lens is not None, (
            "trtllm decode requires metadata built with "
            "init_trtllm_sparse_buffers (decode-mode DSV4AttnMetadata)"
        )
        if sparse_indices.shape[0] != bs:
            assert sparse_indices.shape[0] > bs, f"{sparse_indices.shape=} {bs=}"
            sparse_indices = sparse_indices[:bs]
        if sparse_topk_lens.shape[0] != bs:
            assert sparse_topk_lens.shape[0] > bs, f"{sparse_topk_lens.shape=}"
            sparse_topk_lens = sparse_topk_lens[:bs]

        if compress_ratio == 4:
            assert extra_indices is not None and extra_topk_lengths is not None
            width = extra_indices.shape[-1]
            assert SWA_WINDOW + width == sparse_indices.shape[1], (
                f"{width=} {sparse_indices.shape=}"
            )
            sparse_indices[:, SWA_WINDOW:].copy_(extra_indices)
            # Lens include all 128 SWA slots; seq_lens controls their validity.
            sparse_topk_lens.copy_(extra_topk_lengths)
            sparse_topk_lens.add_(SWA_WINDOW)

        swa_kv_cache, compressed_kv_cache = self._trtllm_kv_cache_views(
            layer.layer_id, compress_ratio
        )

        # RoPE is already applied; the unit scale makes this a plain e4m3 cast.
        q_fp8 = q.to(torch.float8_e4m3fn).view(bs, 1, num_heads, 512)

        bmm1_scale, bmm2_scale = self._get_trtllm_bmm_scales(layer)

        seq_lens = core_attn_metadata.seq_lens_casual
        if seq_lens.shape[0] != bs:
            assert seq_lens.shape[0] > bs, f"{seq_lens.shape=} {bs=}"
            seq_lens = seq_lens[:bs]
        assert attn_sink.dtype == torch.float32
        assert self.trtllm_workspace_buffer is not None

        # Bracketed like prefill: decode needs no extra rows, but it must still
        # reuse the persistent counters rather than take a fresh zeroed buffer
        # per layer (whose memset would replay inside the captured graph).
        with self.trtllm_kv_counters.dsv4_launch(bs):
            out = trtllm_batch_decode_sparse_mla_dsv4(
                query=q_fp8,
                swa_kv_cache=swa_kv_cache,
                workspace_buffer=self.trtllm_workspace_buffer,
                sparse_indices=sparse_indices,
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=sparse_topk_lens,
                seq_lens=seq_lens,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=attn_sink,
                kv_layout="HND",
            )
        if out_pad_tail is not None:
            out_pad_tail[:bs] = out.view(bs, num_heads, 512)
            return out_pad_tail
        return out.view(bs, num_heads, 512)

    def _forward_trtllm_prefill(
        self,
        *,
        q: torch.Tensor,
        layer: RadixAttention,
        compress_ratio: Literal[0, 4, 128],
        forward_batch: ForwardBatch,
        attn_sink: torch.Tensor,
        swa_page_indices: torch.Tensor,
        extra_indices: Optional[torch.Tensor],
        extra_topk_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Drive the decode kernel as varlen prefill with one table row per token.

        ``seq_lens`` includes cached prefixes; the kernel derives causal SWA
        validity from it. This path runs eagerly.
        """

        from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

        assert q.ndim == 3, f"{q.shape=}"
        num_qo_padded, num_heads, head_dim = q.shape
        assert head_dim == 512

        # Build VarSeq metadata from the same extend lengths as the sparse tables.
        core = self.forward_metadata.core_attn_metadata
        if core.trtllm_prefill_qmeta is None:
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            assert extend_seq_lens_cpu is not None and len(extend_seq_lens_cpu) > 0
            batch_size = len(extend_seq_lens_cpu)
            cum_lens = [0] * (batch_size + 1)
            for i, extend_len in enumerate(extend_seq_lens_cpu):
                cum_lens[i + 1] = cum_lens[i] + int(extend_len)
            sum_q = cum_lens[-1]
            max_q_len = max(int(x) for x in extend_seq_lens_cpu)
            assert 0 < sum_q <= num_qo_padded, f"{sum_q=} {num_qo_padded=}"
            seq_lens_i32 = forward_batch.seq_lens.to(torch.int32)
            assert seq_lens_i32.shape == (batch_size,), f"{seq_lens_i32.shape=}"
            core.trtllm_prefill_qmeta = (
                self._move_to_device(cum_lens),
                max_q_len,
                sum_q,
                seq_lens_i32,
            )
        cum_seq_lens_q, max_q_len, sum_q, seq_lens = core.trtllm_prefill_qmeta

        # Cache layer-invariant table parts per chunk. The VarSeq kernel reads
        # rows to a 64-token boundary, so views need inert, tile-aligned parents.
        sum_q_pad = (sum_q + 63) // 64 * 64

        def _tile_padded_pf(fill, src=None, width=None):
            shape = (sum_q_pad,) if width is None else (sum_q_pad, width)
            buf = torch.full(shape, fill, **self.cuda_int32_kwargs)
            if src is not None:
                buf[:sum_q].copy_(src)
            return buf[:sum_q]

        # swa_page_indices and the topk lengths come off the metadata, so every
        # table below is layer-invariant within the chunk and is built once.
        if core.trtllm_prefill_swa_indices is None:
            core.trtllm_prefill_swa_indices = _tile_padded_pf(
                -1, swa_page_indices[:sum_q], width=SWA_WINDOW
            )
        swa_indices = core.trtllm_prefill_swa_indices
        assert swa_indices.shape == (sum_q, SWA_WINDOW), f"{swa_indices.shape=}"
        if extra_indices is None:
            sparse_indices = swa_indices
            if core.trtllm_prefill_swa_lens is None:
                core.trtllm_prefill_swa_lens = _tile_padded_pf(SWA_WINDOW)
            sparse_topk_lens = core.trtllm_prefill_swa_lens
        elif compress_ratio == 128:
            if core.trtllm_prefill_c128 is None:
                width = extra_indices.shape[-1]
                assert width % 4 == 0, f"{width=}"
                table = _tile_padded_pf(-1, width=SWA_WINDOW + width)
                table[:, :SWA_WINDOW].copy_(swa_indices)
                table[:, SWA_WINDOW:].copy_(extra_indices[:sum_q])
                assert extra_topk_lengths is not None
                lens = _tile_padded_pf(
                    SWA_WINDOW,
                    extra_topk_lengths[:sum_q].to(torch.int32) + SWA_WINDOW,
                )
                core.trtllm_prefill_c128 = (table, lens)
            sparse_indices, sparse_topk_lens = core.trtllm_prefill_c128
        else:
            assert extra_topk_lengths is not None
            width = extra_indices.shape[-1]
            # _pad_last_dim keeps the combined c4 capacity divisible by four.
            assert width % 4 == 0, f"{width=}"
            if core.trtllm_prefill_c4_indices is None:
                core.trtllm_prefill_c4_indices = _tile_padded_pf(
                    -1, width=SWA_WINDOW + width
                )
                core.trtllm_prefill_c4_indices[:, :SWA_WINDOW].copy_(swa_indices)
                # Lens include 128 SWA slots; VarSeq metadata controls validity.
                core.trtllm_prefill_c4_lens = _tile_padded_pf(
                    SWA_WINDOW,
                    extra_topk_lengths[:sum_q].to(torch.int32) + SWA_WINDOW,
                )
            sparse_indices = core.trtllm_prefill_c4_indices
            sparse_topk_lens = core.trtllm_prefill_c4_lens
            assert sparse_indices.shape == (
                sum_q,
                SWA_WINDOW + width,
            ), f"{sparse_indices.shape=} {width=}"
            # The indexer rewrites c4_sparse_page_indices per layer, so only
            # the compressed tail is refilled here.
            sparse_indices[:, SWA_WINDOW:].copy_(extra_indices[:sum_q])

        # RoPE is already applied; the unit scale makes this a plain e4m3 cast.
        q_fp8 = q[:sum_q].to(torch.float8_e4m3fn)

        swa_kv_cache, compressed_kv_cache = self._trtllm_kv_cache_views(
            layer.layer_id, compress_ratio
        )
        bmm1_scale, bmm2_scale = self._get_trtllm_bmm_scales(layer)
        assert attn_sink.dtype == torch.float32
        assert self.trtllm_workspace_buffer is not None

        out_padded = None
        out_arg = None
        if num_qo_padded != sum_q:
            # Run only real tokens and keep discarded padding rows finite.
            out_padded = torch.zeros(
                (num_qo_padded, num_heads, 512),
                dtype=torch.bfloat16,
                device=q.device,
            )
            out_arg = out_padded[:sum_q]

        # sum_q rows, above the request count flashinfer would size from.
        with self.trtllm_kv_counters.dsv4_launch(sum_q):
            out = trtllm_batch_decode_sparse_mla_dsv4(
                query=q_fp8,
                swa_kv_cache=swa_kv_cache,
                workspace_buffer=self.trtllm_workspace_buffer,
                sparse_indices=sparse_indices,
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=sparse_topk_lens,
                seq_lens=seq_lens,
                out=out_arg,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=attn_sink,
                kv_layout="HND",
                cum_seq_lens_q=cum_seq_lens_q,
                max_q_len=max_q_len,
            )
        return out_padded if out_padded is not None else out


class DeepseekV4TrtllmMultiStepBackend(
    DeepseekV4MultiStepBackend, DeepseekV4TrtllmAttnBackend
):
    """Multi-step draft wrapper whose per-step backends are trtllm.

    The wrapper is itself a backend (it inherits the whole metadata surface),
    so it must be initialized as a *trtllm* one: the MRO places
    DeepseekV4TrtllmAttnBackend after DeepseekV4MultiStepBackend, which is what
    makes the base class's plain ``super().__init__(model_runner)`` reserve the
    counter rows and lease the workspace buffer. Reordering these bases, or
    giving DeepseekV4MultiStepBackend an explicit
    ``DeepseekV4AttnBackend.__init__`` call, silently drops both --
    ``__init__`` asserts the trtllm state landed so that cannot pass quietly.
    """

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(
            model_runner, topk=topk, speculative_num_steps=speculative_num_steps
        )
        assert self.trtllm_workspace_buffer is not None, (
            "DeepseekV4TrtllmMultiStepBackend must run "
            "DeepseekV4TrtllmAttnBackend.__init__; check the base order"
        )

    def _make_step_backend(
        self, model_runner: ModelRunner, step_id: int
    ) -> DeepseekV4AttnBackend:
        return DeepseekV4TrtllmAttnBackend(
            model_runner,
            speculative_step_id=step_id,
            topk=self.topk,
            speculative_num_steps=self.speculative_num_steps,
        )


def is_dsv4_trtllm_attn_enabled() -> bool:
    return get_exec().kernel.dsv4_attn_backend == "trtllm"


def create_deepseek_v4_attn_backend(
    model_runner: ModelRunner, **kwargs
) -> DeepseekV4AttnBackend:
    """Construct the DSV4 backend matching --dsv4-attn-backend."""
    cls = (
        DeepseekV4TrtllmAttnBackend
        if is_dsv4_trtllm_attn_enabled()
        else DeepseekV4AttnBackend
    )
    return cls(model_runner, **kwargs)


def create_deepseek_v4_multistep_backend(
    model_runner: ModelRunner, topk: int, speculative_num_steps: int
) -> DeepseekV4MultiStepBackend:
    cls = (
        DeepseekV4TrtllmMultiStepBackend
        if is_dsv4_trtllm_attn_enabled()
        else DeepseekV4MultiStepBackend
    )
    return cls(model_runner, topk=topk, speculative_num_steps=speculative_num_steps)
