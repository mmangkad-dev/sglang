"""FlashInfer MNNVL CuTe DSL AllReduce fusion, shared across architectures.

Two patterns share one workspace, both consumed at the next layer's input
RMSNorm: AR + residual + RMSNorm, and the same with the MoE finalize and the
shared-expert add folded in when the runner hands back a MoeFinalizeHandoff.

A model whose residual update is not ``residual + x`` builds the workspace
without the residual add (``folds_residual_norm=False``) and takes the bare
reduced row; see ``cutedsl_ar_fusion_mhc`` for the mHC family.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch

from sglang.srt.arg_groups.overrides import cutedsl_moe_max_num_tokens
from sglang.srt.layers.communicator import (
    CommunicateWithAllReduceAndLayerNormFn,
    LayerCommunicator,
    ScatterMode,
    get_attn_tp_context,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_flags,
    get_parallel,
    get_spec,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

LayerPredicate = Callable[[torch.nn.Module], bool]

logger = logging.getLogger(__name__)


def fused_norm_gamma(layernorm: torch.nn.Module) -> Optional[torch.Tensor]:
    """The multiplier as applied -- GemmaRMSNorm's pre-folded w + 1. None declines."""
    if isinstance(layernorm, GemmaRMSNorm):
        return layernorm.gemma_weight
    if isinstance(layernorm, RMSNorm) and layernorm.has_weight:
        return layernorm.weight
    return None


def is_supported_forward_mode(
    forward_mode: ForwardMode, *, admits_extend: bool = True
) -> bool:
    if forward_mode is ForwardMode.EXTEND:
        return admits_extend
    return forward_mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY)


def _decode_tokens_per_request() -> int:
    spec = get_spec()
    if not spec.speculative_algorithm:
        return 1
    return spec.speculative_num_draft_tokens or 1


def resolve_max_m(
    *,
    server_args: ServerArgs,
    max_running_requests: int | None,
    admits_extend: bool = True,
) -> int:
    decode_config = get_exec().graph.cuda_graph_config.decode
    prefill_config = get_exec().graph.cuda_graph_config.prefill
    requests = [max_running_requests, decode_config.max_bs, *(decode_config.bs or [])]
    if admits_extend:
        candidates = [
            cutedsl_moe_max_num_tokens(server_args),
            prefill_config.max_bs,
            *(prefill_config.bs or []),
            *requests,
        ]
    else:
        # A decode-shaped workspace still has to cover TARGET_VERIFY, which puts
        # bs * draft tokens through one forward.
        tokens_per_request = _decode_tokens_per_request()
        candidates = [
            None if value is None else int(value) * tokens_per_request
            for value in requests
        ]
    positive = [
        int(value) for value in candidates if value is not None and int(value) > 0
    ]
    if not positive:
        raise RuntimeError("framework reported no positive fusion workspace M bound")
    return max(positive)


# Stays a dataclass against .claude/rules/no-dataclasses.md: both producers build
# this under fullgraph=True, and Dynamo cannot construct a msgspec.Struct.
@dataclass(frozen=True)
class MoeFinalizeHandoff:
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
    def __init__(
        self,
        *,
        hidden_size: int,
        top_k: int,
        rms_epsilon: float,
        folds_residual_norm: bool = True,
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.top_k = int(top_k)
        self.rms_epsilon = float(rms_epsilon)
        self.folds_residual_norm = bool(folds_residual_norm)
        # A non-folding workspace still computes and writes a [M, hidden] norm
        # it discards. That is noise at decode M and the dominant cost at a
        # prefill chunk, so such a workspace is decode-shaped.
        self.admits_extend = self.folds_residual_norm
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
            add_residual=self.folds_residual_norm,
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

    def finalize_reduce(self, handoff: MoeFinalizeHandoff) -> torch.Tensor:
        assert self._workspace is not None
        return self._workspace.moe_finalize_all_reduce(
            routed_output=handoff.routed_output,
            expert_weights=handoff.expert_weights,
            permuted_indices=handoff.permuted_indices,
            gated_shared_output=handoff.gated_shared_output,
        )

    def reduce(self, local_contribution: torch.Tensor) -> torch.Tensor:
        assert self._workspace is not None
        return self._workspace.all_reduce(local_contribution)


def fusion_is_eligible(
    *,
    service: CuteDSLFusionService | None,
    forward_batch: ForwardBatch,
    m: int,
    mlp_mode: ScatterMode,
    tp_size: int,
) -> bool:
    """The parallelism and shape guards every fused pattern shares."""
    parallel = get_parallel()
    return bool(
        service is not None
        and is_supported_forward_mode(
            forward_batch.forward_mode, admits_extend=service.admits_extend
        )
        and service.supports(m)
        and not is_dp_attention_enabled()
        and parallel.attn_cp_size == 1
        and not get_attn_tp_context().input_scattered
        and get_moe_a2a_backend().is_none()
        and tp_size > 1
        # Both branches of should_fuse_mlp_allreduce_with_next_layer() answer
        # before delegating to the base, so its guards -- moe-cp allgather,
        # MOE_FULL and SCATTERED -- are restated by requiring FULL here.
        and mlp_mode is ScatterMode.FULL
        # Skipping the post-experts reduction drops both the EP and the TP
        # leg; one fused collective cannot restore both.
        and not (parallel.moe_ep_size > 1 and parallel.moe_tp_size > 1)
    )


def finalize_is_eligible(
    *,
    service: CuteDSLFusionService | None,
    forward_batch: ForwardBatch,
    m: int,
    mlp_mode: ScatterMode,
    tp_size: int,
) -> bool:
    """Deferring hands the experts' reduction to one TP-wide collective."""
    return (
        fusion_is_eligible(
            service=service,
            forward_batch=forward_batch,
            m=m,
            mlp_mode=mlp_mode,
            tp_size=tp_size,
        )
        and get_parallel().moe_ep_size == 1
    )


def reduces_over_the_plain_tp_group(
    communicate_fn,
    *,
    expected_gather_fn,
    attn_dp_size: int,
) -> bool:
    """Whether prepare_mlp's step is a bare all-reduce the fusion can stand in for.

    The other branches scatter, gather for DP, or take the input-scattered path,
    and a plain reduction would drop their redistribution.
    """
    if isinstance(communicate_fn, functools.partial):
        norm_fn = communicate_fn.func
        residual_input_mode = communicate_fn.keywords.get("residual_input_mode")
    else:
        norm_fn = communicate_fn
        residual_input_mode = None
    parallel = get_parallel()
    return (
        norm_fn is expected_gather_fn
        and residual_input_mode is ScatterMode.TP_ATTN_FULL
        and attn_dp_size == 1
        and parallel.attn_tp_size == parallel.tp_size
        and not get_exec().comm.enable_quant_communications
    )


class CuteDSLFusionLayerCommunicator(LayerCommunicator):
    fusion_service: CuteDSLFusionService | None = None

    # This layer's runner can defer AND something downstream consumes it.
    may_defer_moe_finalize: bool = False

    # Unlike the flag above, excludes the last layer: its final norm all-reduces
    # nothing.
    successor_absorbs_all_reduce: bool = False

    # Its MoE adds a replicated contribution after its own reduction, which
    # moving that reduction onward would scale by tp_size.
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
        return (
            self._common_eligible(forward_batch, m)
            and residual is not None
            and fused_norm_gamma(self.post_attention_layernorm) is not None
            and reduces_over_the_plain_tp_group(
                self._communicate_with_all_reduce_and_layer_norm_fn,
                expected_gather_fn=(
                    CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual
                ),
                attn_dp_size=self._context.attn_dp_size,
            )
        )

    def _should_use_finalize(self, forward_batch: ForwardBatch, m: int) -> bool:
        return (
            self._common_eligible(forward_batch, m) and get_parallel().moe_ep_size == 1
        )

    def _can_consume_post_moe_all_reduce(
        self, forward_batch: ForwardBatch, m: int
    ) -> bool:
        """Incoming, and independent of this layer's own successor."""
        return (
            self._common_eligible(forward_batch, m)
            and fused_norm_gamma(self.input_layernorm) is not None
            and not get_exec().comm.enable_quant_communications
        )

    def _can_absorb_post_moe_all_reduce(
        self, forward_batch: ForwardBatch, m: int
    ) -> bool:
        """Outgoing: skip our own all-reduce because the next layer absorbs it."""
        return (
            self.successor_absorbs_all_reduce
            and not self.owes_local_reduction
            and self._can_consume_post_moe_all_reduce(forward_batch, m)
        )

    def should_defer_moe_finalize(
        self, forward_batch: ForwardBatch, m: int | None = None
    ) -> bool:
        """Deferring skips the post-experts all-reduce on the promise of a handoff."""
        if not self.may_defer_moe_finalize:
            return False
        if m is None:
            m = int(forward_batch.input_ids.shape[0])
        return self._should_use_finalize(forward_batch, m)

    def _common_eligible(self, forward_batch: ForwardBatch, m: int) -> bool:
        return fusion_is_eligible(
            service=self.fusion_service,
            forward_batch=forward_batch,
            m=m,
            mlp_mode=self.layer_scatter_modes.mlp_mode,
            tp_size=self._context.tp_size,
        )

    def should_fuse_mlp_allreduce_with_next_layer(
        self, forward_batch: ForwardBatch
    ) -> bool:
        m = int(forward_batch.input_ids.shape[0])
        if self.should_defer_moe_finalize(forward_batch, m):
            return True
        if self._can_absorb_post_moe_all_reduce(forward_batch, m):
            return True
        return super().should_fuse_mlp_allreduce_with_next_layer(forward_batch)


def fusion_communicator_types() -> tuple[type, ...]:
    """Every communicator the installers may attach a service to."""
    from sglang.srt.layers.moe.cutedsl_ar_fusion_mhc import (
        CuteDSLFusionMHCLayerCommunicator,
    )

    return (CuteDSLFusionLayerCommunicator, CuteDSLFusionMHCLayerCommunicator)


def model_installs_cutedsl_fusion(model: torch.nn.Module) -> bool:
    # Most modules carry no ``layer_communicator``, so ``__dict__.get`` stands
    # in for a defensive ``getattr`` over a heterogeneous module tree.
    types = fusion_communicator_types()
    return any(
        isinstance(module.__dict__.get("layer_communicator"), types)
        for module in model.modules()
    )


def build_cutedsl_fusion_service(
    fusion_layers: Sequence[torch.nn.Module],
    *,
    hidden_size: int,
    top_k: int,
    rms_epsilon: float,
    folds_residual_norm: bool,
    label: str,
) -> CuteDSLFusionService | None:
    """The one workspace handle a model shares, or None when it must not build one."""
    if get_flags().moe.in_speculative_scope:
        # A draft is built in the target's process, and each workspace
        # rendezvouses its own NVLS region, so a second one is refused outright.
        return None
    if not fusion_layers:
        return None

    if folds_residual_norm:
        # The kernel validates only the epsilon the service passes, so a family
        # with a per-layer value would silently normalize with the wrong one.
        for layer in fusion_layers:
            for norm in (
                layer.layer_communicator.input_layernorm,
                layer.layer_communicator.post_attention_layernorm,
            ):
                if fused_norm_gamma(norm) is None:
                    continue
                if float(norm.variance_epsilon) != float(rms_epsilon):
                    raise RuntimeError(
                        f"{label} CuTe DSL fusion compiles one workspace for "
                        f"rms_epsilon={rms_epsilon}, but a fused norm uses "
                        f"{norm.variance_epsilon}"
                    )

    return CuteDSLFusionService(
        hidden_size=hidden_size,
        top_k=top_k,
        rms_epsilon=rms_epsilon,
        folds_residual_norm=folds_residual_norm,
    )


def install_cutedsl_fusion(
    layers: Sequence[torch.nn.Module],
    *,
    hidden_size: int,
    top_k: int,
    rms_epsilon: float,
    can_defer_finalize: LayerPredicate,
    requires_local_reduction: LayerPredicate | None = None,
    final_norm_consumes_handoff: bool = False,
    label: str,
) -> CuteDSLFusionService | None:
    """One shared workspace handle per fusion-enabled layer, or None.

    Every entry of ``layers`` must carry a ``layer_communicator``.
    """
    fusion_layers = [
        layer
        for layer in layers
        if isinstance(layer.layer_communicator, CuteDSLFusionLayerCommunicator)
    ]
    service = build_cutedsl_fusion_service(
        fusion_layers,
        hidden_size=hidden_size,
        top_k=top_k,
        rms_epsilon=rms_epsilon,
        folds_residual_norm=True,
        label=label,
    )
    if service is None:
        return None
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
        communicator.owes_local_reduction = (
            requires_local_reduction is not None and requires_local_reduction(layer)
        )
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
    server_args: ServerArgs,
    max_running_requests: int | None,
    label: str,
) -> None:
    if service is None:
        return
    if get_disagg().enable_pdmux:
        raise RuntimeError(
            "FlashInfer MNNVL CuTe DSL fusion does not support concurrent PDMux "
            "streams sharing one mutable workspace"
        )
    service.prepare(
        max_m=resolve_max_m(
            server_args=server_args,
            max_running_requests=max_running_requests,
            admits_extend=service.admits_extend,
        )
    )
    logger.info(
        "Prepared %s FlashInfer MNNVL CuTe DSL fusion workspace for M_max=%d",
        label,
        service.max_m,
    )
