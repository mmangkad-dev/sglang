"""CuTe DSL AllReduce fusion for the mHC (multi-head hyper-connection) family.

mHC carries hc_mult residual streams and replaces ``residual + x`` with
``hc_post``, a learned per-token mix, and the next norm's input with
``hc_pre``'s learned collapse. FlashInfer's CuTe DSL backend implements two
patterns, both ending in ``residual_in + AR(x)`` then RMSNorm, so neither
epilogue can carry mHC -- and pushing the per-stream scale into the collective
would multiply its traffic by hc_mult.

So this family takes the collective alone: the workspace is built without the
residual add, and the fusion delivers the reduced row that ``hc_post``
consumes. The MoE finalize and the shared-expert add still fold into the
collective, and the same layer's ``postprocess_layer`` consumes the handoff
rather than the next layer's input norm.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch

from sglang.srt.layers.communicator_mhc import (
    MHCCommunicateWithAllReduceAndLayerNormFn,
    MHCLayerCommunicator,
)
from sglang.srt.layers.moe.cutedsl_ar_fusion import (
    CuteDSLFusionService,
    LayerPredicate,
    MoeFinalizeHandoff,
    build_cutedsl_fusion_service,
    finalize_is_eligible,
    fusion_is_eligible,
    reduces_over_the_plain_tp_group,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class CuteDSLFusionMHCLayerCommunicator(MHCLayerCommunicator):
    fusion_service: CuteDSLFusionService | None = None

    # This layer's runner can defer, and postprocess_layer consumes it.
    may_defer_moe_finalize: bool = False

    def prepare_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        cache=None,
    ):
        if cache is not None:
            self._context.cache = cache
        if self._should_reduce_attn_output(forward_batch, int(hidden_states.shape[0])):
            assert self.fusion_service is not None
            hidden_states = self.fusion_service.reduce(hidden_states)
            return self.mhc.attn_to_mlp(
                hidden_states, residual, out_norm=self.post_attention_layernorm
            )
        return super().prepare_mlp(hidden_states, residual, forward_batch, cache=cache)

    def postprocess_layer(self, hidden_states, residual, forward_batch: ForwardBatch):
        if isinstance(hidden_states, MoeFinalizeHandoff):
            if not self._should_use_finalize(forward_batch, hidden_states.m):
                raise RuntimeError(
                    "received deferred MoE output on an ineligible path "
                    f"(M={hidden_states.m}, mode={forward_batch.forward_mode})"
                )
            assert self.fusion_service is not None
            hidden_states = self.fusion_service.finalize_reduce(hidden_states)
        return super().postprocess_layer(hidden_states, residual, forward_batch)

    def should_defer_moe_finalize(
        self, forward_batch: ForwardBatch, m: int | None = None
    ) -> bool:
        """Deferring skips the post-experts all-reduce on the promise of a handoff."""
        if not self.may_defer_moe_finalize:
            return False
        if m is None:
            m = int(forward_batch.input_ids.shape[0])
        return self._should_use_finalize(forward_batch, m)

    def _should_use_finalize(self, forward_batch: ForwardBatch, m: int) -> bool:
        return finalize_is_eligible(
            service=self.fusion_service,
            forward_batch=forward_batch,
            m=m,
            mlp_mode=self.layer_scatter_modes.mlp_mode,
            tp_size=self._context.tp_size,
        )

    def _should_reduce_attn_output(self, forward_batch: ForwardBatch, m: int) -> bool:
        """Whether the fused collective may stand in for the post-attention AR."""
        return fusion_is_eligible(
            service=self.fusion_service,
            forward_batch=forward_batch,
            m=m,
            mlp_mode=self.layer_scatter_modes.mlp_mode,
            tp_size=self._context.tp_size,
        ) and reduces_over_the_plain_tp_group(
            self._communicate_with_all_reduce_and_layer_norm_fn,
            expected_gather_fn=(
                MHCCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual
            ),
            attn_dp_size=self._context.attn_dp_size,
        )


def install_cutedsl_mhc_fusion(
    layers: Sequence[torch.nn.Module],
    *,
    hidden_size: int,
    top_k: int,
    rms_epsilon: float,
    can_defer_finalize: LayerPredicate,
    label: str,
) -> Optional[CuteDSLFusionService]:
    """One shared workspace handle per fusion-enabled mHC layer, or None.

    Every entry of ``layers`` must carry a ``layer_communicator``.
    """
    fusion_layers = [
        layer
        for layer in layers
        if isinstance(layer.layer_communicator, CuteDSLFusionMHCLayerCommunicator)
    ]
    service = build_cutedsl_fusion_service(
        fusion_layers,
        hidden_size=hidden_size,
        top_k=top_k,
        rms_epsilon=rms_epsilon,
        folds_residual_norm=False,
        label=label,
    )
    if service is None:
        return None

    for layer in fusion_layers:
        communicator = layer.layer_communicator
        communicator.fusion_service = service
        # hc_post runs in this layer's postprocess, so the handoff never crosses
        # a layer boundary and no successor has to consume it.
        communicator.may_defer_moe_finalize = bool(can_defer_finalize(layer))
    logger.info(
        "Installed one %s FlashInfer MNNVL CuTe DSL fusion handle for %d of %d layers "
        "(%d can defer the MoE finalize)",
        label,
        len(fusion_layers),
        len(layers),
        sum(layer.layer_communicator.may_defer_moe_finalize for layer in fusion_layers),
    )
    return service
