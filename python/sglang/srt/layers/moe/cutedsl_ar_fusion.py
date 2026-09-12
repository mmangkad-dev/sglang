"""FlashInfer MNNVL CuTe DSL AllReduce fusion, shared across architectures.

Two patterns share one workspace, both consumed at the next layer's input
RMSNorm: AR + residual + RMSNorm, and the same with the MoE finalize and the
shared-expert add folded in when the runner hands back a MoeFinalizeHandoff.
"""

from __future__ import annotations

import functools
import logging
from typing import Optional

import msgspec
import torch

from sglang.srt.arg_groups.overrides import cutedsl_moe_max_num_tokens
from sglang.srt.layers.communicator import (
    CommunicateWithAllReduceAndLayerNormFn,
    LayerCommunicator,
    ScatterMode,
    get_attn_tp_context,
)
from sglang.srt.layers.communicator_mhc import (
    MHCCommunicateSummableTensorPairFn,
    MHCCommunicateWithAllReduceAndLayerNormFn,
    MHCLayerCommunicator,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_parallel,
    get_spec,
)

logger = logging.getLogger(__name__)


def fused_norm_gamma(layernorm: torch.nn.Module) -> Optional[torch.Tensor]:
    """The gamma the fused RMSNorm reads, or None for a norm it cannot serve.

    The kernel wants the multiplier as applied, so GemmaRMSNorm hands over its
    pre-folded w + 1. None is how the eligibility predicates decline a layer.
    """
    if isinstance(layernorm, GemmaRMSNorm):
        return layernorm.gemma_weight
    if isinstance(layernorm, RMSNorm) and layernorm.has_weight:
        return layernorm.weight
    return None


def is_supported_forward_mode(forward_mode: ForwardMode) -> bool:
    return forward_mode in (
        ForwardMode.DECODE,
        ForwardMode.EXTEND,
        ForwardMode.TARGET_VERIFY,
    )


def resolve_max_m(*, server_args, max_running_requests: int | None) -> int:
    """Use framework token bounds as the workspace-capacity source of truth."""
    decode_config = get_exec().graph.cuda_graph_config.decode
    prefill_config = get_exec().graph.cuda_graph_config.prefill
    candidates = [
        cutedsl_moe_max_num_tokens(server_args),
        max_running_requests,
        decode_config.max_bs,
        prefill_config.max_bs,
        *(decode_config.bs or []),
        *(prefill_config.bs or []),
    ]
    positive = [
        int(value) for value in candidates if value is not None and int(value) > 0
    ]
    if not positive:
        raise RuntimeError("framework reported no positive fusion workspace M bound")
    return max(positive)


class MoeFinalizeHandoff(msgspec.Struct, frozen=True):
    """Unfinalized routed output plus the separately gated shared contribution."""

    routed_output: torch.Tensor
    expert_weights: torch.Tensor
    permuted_indices: torch.Tensor
    gated_shared_output: torch.Tensor
    m: int

    @classmethod
    def from_flashinfer(
        cls,
        deferred_output,
        *,
        gated_shared_output: torch.Tensor,
        m: int,
    ) -> MoeFinalizeHandoff:
        top_k = int(deferred_output.top_k)
        return cls(
            routed_output=deferred_output.gemm2_out.view(
                -1, deferred_output.gemm2_out.shape[-1]
            ),
            expert_weights=deferred_output.expert_weights.view(-1, top_k)[:m],
            permuted_indices=deferred_output.expanded_idx_to_permuted_idx.view(
                -1, top_k
            )[:m],
            gated_shared_output=gated_shared_output,
            m=int(m),
        )


class CuteDSLFusionService:
    """A lightweight model handle for the process-local FlashInfer workspace."""

    def __init__(
        self,
        *,
        hidden_size: int,
        top_k: int,
        rms_epsilon: float,
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.top_k = int(top_k)
        self.rms_epsilon = float(rms_epsilon)
        self.max_m: int | None = None
        self._workspace = None

    def prepare(self, *, max_m: int) -> None:
        if self._workspace is not None:
            assert self.max_m is not None
            if int(max_m) > self.max_m:
                raise RuntimeError(
                    f"fusion workspace is already prepared for M_max={self.max_m}; "
                    f"refusing M_max={max_m}"
                )
            return
        from sglang.srt.layers.flashinfer_mnnvl_cutedsl import (
            get_flashinfer_mnnvl_cutedsl_ar_fusion,
        )

        workspace = get_flashinfer_mnnvl_cutedsl_ar_fusion(
            hidden_size=self.hidden_size,
            top_k=self.top_k,
            max_m=int(max_m),
            rms_epsilon=self.rms_epsilon,
            # fused_norm_gamma() already returns the multiplier as applied.
            weight_bias=0.0,
        )
        self._workspace = workspace
        self.max_m = workspace.max_m

    def supports(self, m: int) -> bool:
        """False before prepare(), so it doubles as the readiness check."""
        return self._workspace is not None and self._workspace.supports(m)

    def finalize(
        self,
        handoff: MoeFinalizeHandoff,
        residual: torch.Tensor,
        gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._workspace is not None
        return self._workspace.moe_finalize_all_reduce_rms_norm(
            routed_output=handoff.routed_output,
            expert_weights=handoff.expert_weights,
            permuted_indices=handoff.permuted_indices,
            gated_shared_output=handoff.gated_shared_output,
            residual=residual,
            gamma=gamma,
        )

    def all_reduce_residual_rms_norm(
        self,
        local_contribution: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._workspace is not None
        return self._workspace.all_reduce_residual_rms_norm(
            local_contribution=local_contribution,
            residual=residual,
            gamma=gamma,
        )


class CuteDSLFusionLayerCommunicator(LayerCommunicator):
    """The only communicator that runs the CuTe DSL fused patterns."""

    fusion_service: CuteDSLFusionService | None = None

    # Recorded by install_cutedsl_fusion(): this layer's runner can defer AND
    # something downstream consumes the handoff.
    may_defer_moe_finalize: bool = False

    # Whether the NEXT layer absorbs a plain all-reduce; unlike the flag above
    # this excludes the last layer, whose final norm performs no all-reduce.
    successor_absorbs_all_reduce: bool = False

    # This layer's MoE adds a replicated contribution after its own reduction,
    # so moving that reduction to the next layer would scale it by tp_size.
    owes_local_reduction: bool = False

    def prepare_attn(
        self,
        hidden_states,
        residual,
        forward_batch,
        quant_format: str = "",
        post_residual_addition=None,
    ):
        if isinstance(hidden_states, MoeFinalizeHandoff):
            if not self._should_use_finalize(forward_batch, hidden_states.m):
                raise RuntimeError(
                    "received deferred MoE output on an ineligible path "
                    f"(M={hidden_states.m}, mode={forward_batch.forward_mode})"
                )
            if residual is None:
                raise RuntimeError("deferred MoE finalize requires residual input")
            gamma = fused_norm_gamma(self.input_layernorm)
            if gamma is None:
                raise RuntimeError(
                    "deferred MoE finalize requires a fusable RMSNorm flavour"
                )
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            assert self.fusion_service is not None
            hidden_states, residual = self.fusion_service.finalize(
                handoff=hidden_states, residual=residual, gamma=gamma
            )
            return self._finish_prepare_attn(hidden_states, residual, forward_batch)

        if (
            residual is not None
            and hasattr(hidden_states, "_sglang_needs_allreduce_fusion")
            and hidden_states._sglang_needs_allreduce_fusion
            and self._can_consume_post_moe_all_reduce(
                forward_batch, int(hidden_states.shape[0])
            )
        ):
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            assert self.fusion_service is not None
            hidden_states, residual = self.fusion_service.all_reduce_residual_rms_norm(
                local_contribution=hidden_states,
                residual=residual,
                gamma=fused_norm_gamma(self.input_layernorm),
            )
            return self._finish_prepare_attn(hidden_states, residual, forward_batch)

        return super().prepare_attn(
            hidden_states,
            residual,
            forward_batch,
            quant_format=quant_format,
            post_residual_addition=post_residual_addition,
        )

    def prepare_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        cache=None,
    ):
        if cache is not None:
            self._context.cache = cache
        if self._should_use_all_reduce_rms_norm(
            forward_batch, int(hidden_states.shape[0]), residual
        ):
            assert self.fusion_service is not None and residual is not None
            return self.fusion_service.all_reduce_residual_rms_norm(
                local_contribution=hidden_states,
                residual=residual,
                gamma=fused_norm_gamma(self.post_attention_layernorm),
            )
        return super().prepare_mlp(hidden_states, residual, forward_batch, cache=cache)

    def _should_use_all_reduce_rms_norm(
        self,
        forward_batch: ForwardBatch,
        m: int,
        residual: Optional[torch.Tensor],
    ) -> bool:
        communicate_fn = self._communicate_with_all_reduce_and_layer_norm_fn
        if isinstance(communicate_fn, functools.partial):
            norm_fn = communicate_fn.func
            residual_input_mode = communicate_fn.keywords.get("residual_input_mode")
        else:
            norm_fn = communicate_fn
            residual_input_mode = None
        parallel = get_parallel()
        return (
            self._common_eligible(forward_batch, m)
            and residual is not None
            and fused_norm_gamma(self.post_attention_layernorm) is not None
            and norm_fn
            is CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual
            and residual_input_mode is ScatterMode.TP_ATTN_FULL
            and self._context.attn_dp_size == 1
            and parallel.attn_tp_size == parallel.tp_size
            and not get_exec().comm.enable_quant_communications
        )

    def _should_use_finalize(self, forward_batch: ForwardBatch, m: int) -> bool:
        """Consuming a handoff is a property of the fused kernel and the
        topology, not of the MoE runner that produced it."""
        parallel = get_parallel()
        return (
            self._common_eligible(forward_batch, m)
            and self.layer_scatter_modes.mlp_mode is not ScatterMode.SCATTERED
            and parallel.moe_ep_size == 1
        )

    def _can_consume_post_moe_all_reduce(
        self, forward_batch: ForwardBatch, m: int
    ) -> bool:
        """Incoming: may this layer's input norm absorb the all-reduce its
        predecessor skipped. Independent of this layer's own successor."""
        return (
            self._common_eligible(forward_batch, m)
            and fused_norm_gamma(self.input_layernorm) is not None
            and not get_exec().comm.enable_quant_communications
        )

    def _can_absorb_post_moe_all_reduce(
        self, forward_batch: ForwardBatch, m: int
    ) -> bool:
        """Outgoing: may this layer skip its own all-reduce because the next
        one absorbs it. Refused when it owes a local reduction."""
        return (
            self.successor_absorbs_all_reduce
            and not self.owes_local_reduction
            and self._can_consume_post_moe_all_reduce(forward_batch, m)
        )

    def should_defer_moe_finalize(
        self, forward_batch: ForwardBatch, m: int | None = None
    ) -> bool:
        """Deferring skips the post-experts all-reduce on the promise of a
        handoff, so a layer with no consumer must not defer."""
        if not self.may_defer_moe_finalize:
            return False
        if m is None:
            m = int(forward_batch.input_ids.shape[0])
        return self._should_use_finalize(forward_batch, m)

    def _common_eligible(self, forward_batch: ForwardBatch, m: int) -> bool:
        parallel = get_parallel()
        return bool(
            self.fusion_service is not None
            and is_supported_forward_mode(forward_batch.forward_mode)
            and self.fusion_service.supports(m)
            and not is_dp_attention_enabled()
            and parallel.attn_cp_size == 1
            and not get_attn_tp_context().input_scattered
            and get_moe_a2a_backend().is_none()
            and self._context.tp_size > 1
            # Skipping the post-experts reduction drops both the EP and the TP
            # leg; one fused collective cannot restore both.
            and not (parallel.moe_ep_size > 1 and parallel.moe_tp_size > 1)
        )

    def should_fuse_mlp_allreduce_with_next_layer(
        self, forward_batch: ForwardBatch
    ) -> bool:
        m = int(forward_batch.input_ids.shape[0])
        if self.should_defer_moe_finalize(forward_batch, m):
            return True
        # A runner that returns a plain tensor can still hand its all-reduce
        # to the next layer's input norm.
        if self._can_absorb_post_moe_all_reduce(forward_batch, m):
            return True
        return super().should_fuse_mlp_allreduce_with_next_layer(forward_batch)


def install_cutedsl_fusion(
    layers,
    *,
    hidden_size: int,
    top_k: int,
    rms_epsilon: float,
    can_defer_finalize,
    requires_local_reduction=lambda layer: False,
    final_norm_consumes_handoff: bool = False,
    label: str,
) -> CuteDSLFusionService | None:
    """Give every fusion-enabled layer one shared workspace handle, or None.

    The workspace compiles per (hidden_size, top_k, rms_epsilon), which every MoE
    layer of a model shares. Every entry of ``layers`` must carry a
    ``layer_communicator``, so a PP-padded list is sliced to the local range
    first. Set ``final_norm_consumes_handoff`` only when the model's final norm
    closes out the last layer's handoff, and ``requires_local_reduction`` for a
    layer whose MoE adds a replicated output after its own all-reduce.
    """
    fusion_layers = [
        layer
        for layer in layers
        if isinstance(layer.layer_communicator, CuteDSLFusionLayerCommunicator)
    ]
    if not fusion_layers:
        return None

    service = CuteDSLFusionService(
        hidden_size=hidden_size,
        top_k=top_k,
        rms_epsilon=rms_epsilon,
    )
    for index, layer in enumerate(layers):
        communicator = layer.layer_communicator
        if not isinstance(communicator, CuteDSLFusionLayerCommunicator):
            continue
        successor = layers[index + 1] if index + 1 < len(layers) else None
        if successor is None:
            has_consumer = final_norm_consumes_handoff
        else:
            has_consumer = isinstance(
                successor.layer_communicator, CuteDSLFusionLayerCommunicator
            )
        communicator.fusion_service = service
        communicator.may_defer_moe_finalize = (
            bool(can_defer_finalize(layer)) and has_consumer
        )
        communicator.successor_absorbs_all_reduce = successor is not None and (
            isinstance(successor.layer_communicator, CuteDSLFusionLayerCommunicator)
        )
        communicator.owes_local_reduction = bool(requires_local_reduction(layer))
    logger.info(
        "Installed one %s FlashInfer MNNVL CuTe DSL fusion handle for %d of %d layers "
        "(%d can defer the MoE finalize)",
        label,
        len(fusion_layers),
        len(layers),
        sum(layer.layer_communicator.may_defer_moe_finalize for layer in fusion_layers),
    )
    return service


def prepare_cutedsl_fusion(
    service: CuteDSLFusionService | None,
    *,
    server_args,
    max_running_requests: int | None,
    label: str,
) -> None:
    """Compile the workspace before graph capture; a no-op without a handle."""
    if service is None:
        return
    if get_disagg().enable_pdmux:
        raise RuntimeError(
            "FlashInfer MNNVL CuTe DSL fusion does not support concurrent PDMux "
            "streams sharing one mutable workspace"
        )
    service.prepare(
        max_m=resolve_max_m(
            server_args=server_args, max_running_requests=max_running_requests
        )
    )
    logger.info(
        "Prepared %s FlashInfer MNNVL CuTe DSL fusion workspace for M_max=%d",
        label,
        service.max_m,
    )


# ---------------------------------------------------------------------------
# The bare collective: models whose residual stream is not a plain add
# ---------------------------------------------------------------------------

# Decode and its speculative equivalent only. Admitting EXTEND measured neutral
# per forward and cost 9 ms of mean TTFT on GLM-5.3-Flash at 8192-token
# prefills, over 12 runs across 3 server processes.
_BARE_SUPPORTED_FORWARD_MODES = frozenset(
    (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY)
)


def resolve_decode_max_m(*, max_running_requests: int | None) -> int:
    """Largest token count the bare path may serve, from the decode bounds.

    Prefill bounds are excluded because the supported forward modes are. An
    under-estimate costs the optimization and nothing else: supports() declines
    and the ordinary path runs.
    """
    decode_config = get_exec().graph.cuda_graph_config.decode
    spec = get_spec()
    tokens_per_request = (
        (spec.speculative_num_draft_tokens or 1) if spec.speculative_algorithm else 1
    )
    requests = [
        int(value)
        for value in (
            max_running_requests,
            decode_config.max_bs,
            *(decode_config.bs or []),
        )
        if value is not None and int(value) > 0
    ]
    if not requests:
        raise RuntimeError("framework reported no positive decode request bound")
    return max(requests) * tokens_per_request


class CuteDSLBareAllReduceService:
    """A model handle for the process-local, residual-free workspace."""

    def __init__(self, *, hidden_size: int, top_k: int, rms_epsilon: float) -> None:
        self.hidden_size = int(hidden_size)
        self.top_k = int(top_k)
        self.rms_epsilon = float(rms_epsilon)
        self.max_m: int | None = None
        self._workspace = None
        self._gamma: torch.Tensor | None = None
        self._norm_scratch: torch.Tensor | None = None

    def prepare(self, *, max_m: int) -> None:
        """False from supports() before this runs, so it doubles as readiness."""
        if self._workspace is not None:
            assert self.max_m is not None
            if int(max_m) > self.max_m:
                raise RuntimeError(
                    f"fusion workspace is already prepared for M_max={self.max_m}; "
                    f"refusing M_max={max_m}"
                )
            return
        from sglang.srt.layers.flashinfer_mnnvl_cutedsl import (
            get_flashinfer_mnnvl_cutedsl_ar_fusion,
        )

        workspace = get_flashinfer_mnnvl_cutedsl_ar_fusion(
            hidden_size=self.hidden_size,
            top_k=self.top_k,
            max_m=int(max_m),
            rms_epsilon=self.rms_epsilon,
            weight_bias=0.0,
            fuse_residual=False,
        )
        self._workspace = workspace
        self.max_m = workspace.max_m
        # The compiled kernel always writes a normalized output the model never
        # reads; a unit gamma keeps that half well-defined without a weight, and
        # one buffer absorbs it for every layer and both boundaries.
        self._gamma = torch.ones(
            self.hidden_size, dtype=torch.bfloat16, device=workspace.device
        )
        self._norm_scratch = torch.empty(
            (self.max_m, self.hidden_size),
            dtype=torch.bfloat16,
            device=workspace.device,
        )

    def supports(self, m: int) -> bool:
        return self._workspace is not None and self._workspace.supports(m)

    def all_reduce(self, local_contribution: torch.Tensor) -> torch.Tensor:
        m = int(local_contribution.shape[0])
        assert self._workspace is not None and self._gamma is not None
        assert self._norm_scratch is not None
        return self._workspace.all_reduce(
            local_contribution=local_contribution,
            gamma=self._gamma,
            norm_scratch=self._norm_scratch[:m],
        )


class CuteDSLBareAllReduceMHCLayerCommunicator(MHCLayerCommunicator):
    """mHC layers drive the collective without the fused residual add.

    Both compiled patterns end in ``residual + x`` then RMSNorm, and mHC gives
    neither boundary that shape: ``hc_post`` mixes the reduced value into
    ``hc_mult`` residual streams with per-token weights, and ``hc_ffn_pre``
    normalizes only after its own mix. The model reads the cross-rank sum and
    combines it as it always did.
    """

    fusion_service: CuteDSLBareAllReduceService | None = None

    # This layer's MoE adds a replicated contribution after its own reduction,
    # so moving that reduction to postprocess_layer would scale it by tp_size.
    # Only the MLP boundary is affected; the attention one still fuses.
    owes_local_reduction: bool = False

    def _post_init_communicate(self):
        super()._post_init_communicate()
        # Everything but the forward mode, M, and the scattered-input flag is
        # frozen once the communicate callables are chosen.
        parallel = get_parallel()
        communicate_fn = self._communicate_with_all_reduce_and_layer_norm_fn
        if isinstance(communicate_fn, functools.partial):
            norm_fn = communicate_fn.func
            residual_input_mode = communicate_fn.keywords.get("residual_input_mode")
        else:
            norm_fn = communicate_fn
            residual_input_mode = None
        shape_eligible = (
            not is_dp_attention_enabled()
            and self._context.attn_dp_size == 1
            and self._context.tp_size > 1
            and parallel.attn_tp_size == parallel.tp_size
            and parallel.attn_cp_size == 1
            and get_moe_a2a_backend().is_none()
            and not get_exec().comm.enable_quant_communications
        )
        self._attn_output_eligible = (
            shape_eligible
            and norm_fn
            is MHCCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual
            and residual_input_mode is ScatterMode.TP_ATTN_FULL
        )
        self._mlp_output_eligible = (
            shape_eligible
            and parallel.moe_ep_size == 1
            and self._communicate_summable_tensor_pair_fn
            is MHCCommunicateSummableTensorPairFn._trivial
        )
        # Published by should_defer_mlp_allreduce and consumed by
        # postprocess_layer, so one verdict drives both the MLP's skip and the
        # reduction that replaces it. One bool holds only because a layer runs to
        # completion between the two: two-batch overlap decomposes them into
        # separate operations and interleaves microbatches through one
        # communicator, so a GLM-5-Next TBO strategy must make this
        # per-microbatch.
        self._mlp_allreduce_deferred = False

    def prepare_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        cache=None,
    ):
        if cache is not None:
            self._context.cache = cache
        m = int(hidden_states.shape[0])
        if self._attn_output_eligible and self._forward_eligible(forward_batch, m):
            assert self.fusion_service is not None
            hidden_states = self.fusion_service.all_reduce(hidden_states)
            return self.mhc.attn_to_mlp(
                hidden_states, residual, out_norm=self.post_attention_layernorm
            )
        return super().prepare_mlp(hidden_states, residual, forward_batch, cache=cache)

    def should_defer_mlp_allreduce(self, forward_batch: ForwardBatch) -> bool:
        if self.owes_local_reduction:
            self._mlp_allreduce_deferred = False
            return False
        m = int(forward_batch.input_ids.shape[0])
        self._mlp_allreduce_deferred = self._mlp_output_eligible and (
            self._forward_eligible(forward_batch, m)
        )
        return self._mlp_allreduce_deferred

    def postprocess_layer(self, hidden_states, residual, forward_batch):
        if self._mlp_allreduce_deferred:
            self._mlp_allreduce_deferred = False
            assert self.fusion_service is not None
            hidden_states = self.fusion_service.all_reduce(hidden_states)
        return super().postprocess_layer(hidden_states, residual, forward_batch)

    def _forward_eligible(self, forward_batch: ForwardBatch, m: int) -> bool:
        return bool(
            self.fusion_service is not None
            and forward_batch.forward_mode in _BARE_SUPPORTED_FORWARD_MODES
            and not get_attn_tp_context().input_scattered
            and self.fusion_service.supports(m)
        )


def install_cutedsl_bare_all_reduce(
    layers,
    *,
    hidden_size: int,
    top_k: int,
    rms_epsilon: float,
    requires_local_reduction=lambda layer: False,
    label: str,
) -> CuteDSLBareAllReduceService | None:
    """Give every bare-collective layer one shared workspace handle, or None.

    Pass ``requires_local_reduction`` for a layer whose MoE adds a replicated
    output after its own all-reduce.
    """
    handles = [
        layer
        for layer in layers
        if isinstance(
            layer.layer_communicator, CuteDSLBareAllReduceMHCLayerCommunicator
        )
    ]
    if not handles:
        return None
    service = CuteDSLBareAllReduceService(
        hidden_size=hidden_size, top_k=top_k, rms_epsilon=rms_epsilon
    )
    for layer in handles:
        layer.layer_communicator.fusion_service = service
        layer.layer_communicator.owes_local_reduction = bool(
            requires_local_reduction(layer)
        )
    logger.info(
        "Installed one %s FlashInfer MNNVL CuTe DSL bare-collective handle for "
        "%d of %d layers",
        label,
        len(handles),
        len(layers),
    )
    return service


def prepare_cutedsl_bare_all_reduce(
    service: CuteDSLBareAllReduceService | None,
    *,
    max_running_requests: int | None,
    label: str,
) -> None:
    """Compile the workspace before graph capture; a no-op without a handle."""
    if service is None:
        return
    if get_disagg().enable_pdmux:
        raise RuntimeError(
            "FlashInfer MNNVL CuTe DSL fusion does not support concurrent PDMux "
            "streams sharing one mutable workspace"
        )
    service.prepare(
        max_m=resolve_decode_max_m(max_running_requests=max_running_requests)
    )
    logger.info(
        "Prepared %s FlashInfer MNNVL CuTe DSL bare-collective workspace for M_max=%d",
        label,
        service.max_m,
    )
