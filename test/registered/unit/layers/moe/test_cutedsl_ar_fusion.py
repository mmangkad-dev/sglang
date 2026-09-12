from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.communicator import LayerCommunicator
from sglang.srt.layers.flashinfer_mnnvl_cutedsl import (
    FlashInferMNNVLCuteDSLARFusion,
    _with_early_finalize_shared_load,
)
from sglang.srt.layers.moe.cutedsl_ar_fusion import (
    CuteDSLFusionLayerCommunicator,
    MoeFinalizeHandoff,
    is_supported_forward_mode,
    resolve_max_m,
)
from sglang.srt.model_executor.cuda_graph_config import (
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.qwen3_5_text import Qwen3_5ForCausalLM
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


@dataclass(frozen=True)
class _TestPreset:
    load_shared_expert_before_pdl: bool = False


@dataclass(frozen=True)
class _TestTarget:
    preset: object


@dataclass(frozen=True)
class _TestRoutes:
    targets: tuple[_TestTarget, ...]


@dataclass(frozen=True)
class _TestProfile:
    finalize_routes: _TestRoutes


@dataclass(frozen=True)
class _TestConfig:
    profiles: tuple[_TestProfile, ...]


@pytest.mark.parametrize(
    ("forward_mode", "expected"),
    [
        (ForwardMode.DECODE, True),
        (ForwardMode.EXTEND, True),
        (ForwardMode.IDLE, False),
        (ForwardMode.TARGET_VERIFY, True),
        (ForwardMode.DRAFT_EXTEND_V2, False),
    ],
)
def test_supported_forward_modes(forward_mode, expected):
    assert is_supported_forward_mode(forward_mode) is expected


@patch(
    "sglang.srt.layers.moe.cutedsl_ar_fusion.cutedsl_moe_max_num_tokens",
    return_value=8192,
)
def test_framework_capacity_is_maximum_of_all_sources(_cutedsl_moe_max_num_tokens):
    # The graph bounds are a bag leaf, so the test states them by publishing.
    reset_context()
    publish(
        ServerArgs(
            model_path="dummy",
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(max_bs=512, bs=[1, 64, 256]),
                prefill=PhaseConfig(max_bs=4096, bs=[1024, 2048, 4096]),
            ),
        ),
        role="test",
    )
    assert (
        resolve_max_m(server_args=SimpleNamespace(), max_running_requests=2048) == 8192
    )


def test_deferred_handoff_reuses_producer_storage():
    m, top_k, hidden_size = 3, 10, 16
    gemm2_out = torch.empty(m * top_k + 4, hidden_size, dtype=torch.bfloat16)
    expert_weights = torch.empty(m, top_k, dtype=torch.bfloat16)
    permuted_indices = torch.empty(m, top_k, dtype=torch.int32)
    gated_shared_output = torch.empty(m, hidden_size, dtype=torch.bfloat16)
    deferred = SimpleNamespace(
        gemm2_out=gemm2_out,
        expert_weights=expert_weights,
        expanded_idx_to_permuted_idx=permuted_indices,
        top_k=top_k,
    )

    handoff = MoeFinalizeHandoff.from_flashinfer(
        deferred,
        gated_shared_output=gated_shared_output,
        m=m,
    )

    assert handoff.routed_output.data_ptr() == gemm2_out.data_ptr()
    assert handoff.expert_weights.data_ptr() == expert_weights.data_ptr()
    assert handoff.permuted_indices.data_ptr() == permuted_indices.data_ptr()
    assert handoff.gated_shared_output is gated_shared_output


def test_qwen_workspace_config_enables_only_supported_finalize_presets():
    untouched_preset = object()
    default_config = _TestConfig(
        profiles=(
            _TestProfile(
                finalize_routes=_TestRoutes(
                    targets=(
                        _TestTarget(_TestPreset()),
                        _TestTarget(untouched_preset),
                    )
                )
            ),
        )
    )

    qwen_config = _with_early_finalize_shared_load(default_config)

    assert qwen_config is not default_config
    assert (
        default_config.profiles[0]
        .finalize_routes.targets[0]
        .preset.load_shared_expert_before_pdl
        is False
    )
    assert (
        qwen_config.profiles[0]
        .finalize_routes.targets[0]
        .preset.load_shared_expert_before_pdl
        is True
    )
    assert qwen_config.profiles[0].finalize_routes.targets[1].preset is untouched_preset


def test_wrapper_calls_only_the_stable_unified_api():
    calls = []
    wrapper = object.__new__(FlashInferMNNVLCuteDSLARFusion)
    wrapper.hidden_size = 8
    wrapper.top_k = 2
    wrapper.max_m = 4
    wrapper.rms_epsilon = 1e-5
    wrapper.weight_bias = 0.0
    wrapper.fuse_residual = True
    wrapper.device = torch.device("cpu")
    wrapper.workspace = object()
    wrapper.supports = lambda m: True
    wrapper._patterns = SimpleNamespace(
        kARResidualRMSNorm=1,
        kMoEFinalizeARResidualRMSNorm=7,
    )
    wrapper._allreduce_fusion = lambda **kwargs: calls.append(kwargs)

    routed_output = torch.empty(8, 8, dtype=torch.bfloat16)
    expert_weights = torch.empty(4, 2, dtype=torch.bfloat16)
    permuted_indices = torch.empty(4, 2, dtype=torch.int32)
    gated_shared_output = torch.empty(4, 8, dtype=torch.bfloat16)
    residual = torch.empty(4, 8, dtype=torch.bfloat16)
    gamma = torch.empty(8, dtype=torch.bfloat16)
    wrapper.moe_finalize_all_reduce_rms_norm(
        routed_output=routed_output,
        expert_weights=expert_weights,
        permuted_indices=permuted_indices,
        gated_shared_output=gated_shared_output,
        residual=residual,
        gamma=gamma,
    )

    assert calls[0]["launch_with_pdl"] is True
    assert "routed_scaling_factor" not in calls[0]

    wrapper.all_reduce_residual_rms_norm(
        local_contribution=residual,
        residual=residual,
        gamma=gamma,
    )

    assert calls[1]["pattern"] == 1
    assert calls[1]["launch_with_pdl"] is True
    assert "routed_scaling_factor" not in calls[1]
    assert "expanded_idx_to_permuted_idx" not in calls[1]


def test_text_entry_wrapper_delegates_pre_capture_prepare():
    calls = []
    runner = object()
    wrapper = SimpleNamespace(
        model=SimpleNamespace(
            prepare_before_cuda_graph_capture=lambda value: calls.append(value)
        )
    )

    Qwen3_5ForCausalLM.prepare_before_cuda_graph_capture(wrapper, runner)

    assert calls == [runner]


def _eligible_communicator(*, successor: bool):
    """A CuteDSLFusionLayerCommunicator stub whose only varying input is whether
    a successor exists to absorb this layer's outgoing all-reduce."""
    comm = CuteDSLFusionLayerCommunicator.__new__(CuteDSLFusionLayerCommunicator)
    comm.successor_absorbs_all_reduce = successor
    comm.input_layernorm = SimpleNamespace()
    return comm


def test_last_layer_prepare_attn_consumes_the_pending_all_reduce():
    """A layer with no successor must still run the fused collective on a
    tensor its predecessor tagged, or that reduction is silently dropped."""
    last = _eligible_communicator(successor=False)
    last.fusion_service = SimpleNamespace(
        all_reduce_residual_rms_norm=(
            lambda *, local_contribution, residual, gamma: (
                local_contribution + 1,
                residual + 1,
            )
        )
    )
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    hidden_states = torch.zeros(8, 8)
    hidden_states._sglang_needs_allreduce_fusion = True
    residual = torch.zeros(8, 8)

    finished = []
    with (
        patch.object(
            CuteDSLFusionLayerCommunicator, "_common_eligible", return_value=True
        ),
        patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.fused_norm_gamma",
            return_value=torch.empty(8),
        ),
        patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.get_exec",
            return_value=SimpleNamespace(
                comm=SimpleNamespace(enable_quant_communications=False)
            ),
        ),
        patch.object(
            CuteDSLFusionLayerCommunicator,
            "_finish_prepare_attn",
            lambda self, h, r, fb: finished.append((h, r)) or (h, r),
        ),
        patch.object(
            LayerCommunicator,
            "prepare_attn",
            lambda *a, **k: pytest.fail(
                "the last layer fell through to the unfused path, dropping the "
                "reduction its predecessor skipped"
            ),
        ),
    ):
        out_hidden, out_residual = last.prepare_attn(
            hidden_states, residual, forward_batch
        )

    assert len(finished) == 1
    assert torch.equal(out_hidden, torch.ones(8, 8))
    assert torch.equal(out_residual, torch.ones(8, 8))


def test_last_layer_still_declines_to_skip_its_own_all_reduce():
    """The other half of the split: consuming is owed to it, deferring is not."""
    last = _eligible_communicator(successor=False)
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    with (
        patch.object(
            CuteDSLFusionLayerCommunicator, "_common_eligible", return_value=True
        ),
        patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.fused_norm_gamma",
            return_value=torch.empty(8),
        ),
        patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.get_exec",
            return_value=SimpleNamespace(
                comm=SimpleNamespace(enable_quant_communications=False)
            ),
        ),
    ):
        assert last._can_consume_post_moe_all_reduce(forward_batch, 8) is True
        assert last._can_absorb_post_moe_all_reduce(forward_batch, 8) is False


def test_hybrid_ep_tp_is_refused_like_the_base_communicator():
    """Hybrid EP+TP must stay refused: skipping the post-experts reduction
    drops both legs and one fused collective cannot restore them."""
    comm = _eligible_communicator(successor=True)
    comm.fusion_service = SimpleNamespace(supports=lambda m: True)
    comm._context = SimpleNamespace(tp_size=4, attn_dp_size=1)
    comm.layer_scatter_modes = SimpleNamespace(mlp_mode=None)
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    def parallel(*, ep, moe_tp):
        return SimpleNamespace(
            moe_ep_size=ep,
            moe_tp_size=moe_tp,
            attn_cp_size=1,
            tp_size=4,
            attn_tp_size=4,
        )

    with (
        patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.is_dp_attention_enabled",
            return_value=False,
        ),
        patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.get_attn_tp_context",
            return_value=SimpleNamespace(input_scattered=False),
        ),
        patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.get_moe_a2a_backend",
            return_value=SimpleNamespace(is_none=lambda: True),
        ),
    ):
        with patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.get_parallel",
            return_value=parallel(ep=2, moe_tp=2),
        ):
            assert comm._common_eligible(forward_batch, 8) is False
        with patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.get_parallel",
            return_value=parallel(ep=1, moe_tp=4),
        ):
            assert comm._common_eligible(forward_batch, 8) is True


def test_a_replicated_shared_expert_producer_keeps_its_own_all_reduce():
    """A TP1 shared expert is added after the layer's own reduction, so handing
    that reduction to the next layer would scale the shared output by tp_size."""
    producer = _eligible_communicator(successor=True)
    producer.owes_local_reduction = True
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE, input_ids=torch.zeros(8)
    )

    with (
        patch.object(
            CuteDSLFusionLayerCommunicator, "_common_eligible", return_value=True
        ),
        patch.object(
            CuteDSLFusionLayerCommunicator,
            "should_defer_moe_finalize",
            return_value=False,
        ),
        patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.fused_norm_gamma",
            return_value=torch.empty(8),
        ),
        patch(
            "sglang.srt.layers.moe.cutedsl_ar_fusion.get_exec",
            return_value=SimpleNamespace(
                comm=SimpleNamespace(enable_quant_communications=False)
            ),
        ),
        patch.object(
            LayerCommunicator,
            "should_fuse_mlp_allreduce_with_next_layer",
            return_value=False,
        ),
    ):
        # The publisher of fuse_mlp_allreduce, which is what skips the reduction.
        assert (
            producer.should_fuse_mlp_allreduce_with_next_layer(forward_batch) is False
        )
        assert producer._can_absorb_post_moe_all_reduce(forward_batch, 8) is False
        # Consuming what a predecessor skipped stays independently eligible.
        assert producer._can_consume_post_moe_all_reduce(forward_batch, 8) is True


def test_install_records_which_producers_owe_a_local_reduction():
    """install_cutedsl_fusion must carry the model's predicate onto the layer,
    or a replicated shared expert silently keeps the unsafe fusion."""
    from sglang.srt.layers.moe.cutedsl_ar_fusion import install_cutedsl_fusion

    layers = [
        SimpleNamespace(
            layer_communicator=CuteDSLFusionLayerCommunicator.__new__(
                CuteDSLFusionLayerCommunicator
            ),
            replicated=replicated,
        )
        for replicated in (True, False)
    ]
    install_cutedsl_fusion(
        layers,
        hidden_size=8,
        top_k=2,
        rms_epsilon=1e-6,
        can_defer_finalize=lambda layer: not layer.replicated,
        requires_local_reduction=lambda layer: layer.replicated,
        label="test",
    )

    assert layers[0].layer_communicator.owes_local_reduction is True
    assert layers[1].layer_communicator.owes_local_reduction is False


def _call_dual_stream_op(fusion, hidden_states, *, fuse_mlp_allreduce=True):
    """Redispatching to the CUDA key runs the real registered implementation
    and its schema, while the stubbed MoE keeps the tensors on CPU."""
    from sglang.srt.models.deepseek_v2 import (  # noqa: F401  (registers the op)
        dsv2_flashinfer_moe_dual_stream_graph,
    )

    op = torch.ops.sglang.dsv2_flashinfer_moe_dual_stream_graph.default
    cuda_key = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    with patch(
        "sglang.srt.models.deepseek_v2.get_tc_piecewise_forward_context",
        return_value=SimpleNamespace(moe_fusions={0: fusion}),
    ):
        return op.redispatch(cuda_key, hidden_states, 0, fuse_mlp_allreduce, False)


class _DeferRecordingMoE:
    """Returns a handoff whenever the deferral reaches it, as the real MoE does."""

    def __init__(self):
        self.seen_defer = None

    def forward_normal_dual_stream(self, hidden_states):
        from sglang.srt.runtime_context import get_forward

        self.seen_defer = get_forward().defer_moe_finalize
        if self.seen_defer:
            return object()
        return hidden_states + 1


def test_dual_stream_op_pins_the_deferral_off_under_a_deferring_caller():
    """A deferring caller must not make the op hand a handoff back through its
    Tensor schema; the dispatcher raises "Unable to cast ... to Tensor"."""
    from sglang.srt.runtime_context import get_forward

    reset_context()
    fusion = _DeferRecordingMoE()
    hidden_states = torch.zeros(4, 8)

    with get_forward().scoped(defer_moe_finalize=True):
        out = _call_dual_stream_op(fusion, hidden_states)
        # The pin is scoped to the op; the caller's own flag survives it.
        assert get_forward().defer_moe_finalize is True

    assert fusion.seen_defer is False
    assert isinstance(out, torch.Tensor)
    assert torch.equal(out, hidden_states + 1)


def test_dual_stream_op_still_republishes_its_operand_flags():
    """Pinning the deferral must not disturb the two flags the op republishes
    from its scalar operands."""
    seen = {}

    class _FlagReader:
        def forward_normal_dual_stream(self, hidden_states):
            from sglang.srt.runtime_context import get_forward

            flags = get_forward()
            seen["fuse"] = flags.fuse_mlp_allreduce
            seen["scatter"] = flags.mlp_reduce_scatter
            return hidden_states

    reset_context()
    _call_dual_stream_op(_FlagReader(), torch.zeros(4, 8))

    assert seen["fuse"] is True
    assert seen["scatter"] is False


def test_bare_workspace_refuses_the_fused_patterns():
    """A workspace compiled without the residual add cannot serve the fused
    patterns; the model reads the cross-rank sum and combines it itself."""
    wrapper = object.__new__(FlashInferMNNVLCuteDSLARFusion)
    wrapper.fuse_residual = False

    with pytest.raises(RuntimeError, match="without the residual add"):
        wrapper.all_reduce_residual_rms_norm(
            local_contribution=torch.empty(2, 8, dtype=torch.bfloat16),
            residual=torch.empty(2, 8, dtype=torch.bfloat16),
            gamma=torch.empty(8, dtype=torch.bfloat16),
        )


def test_fused_workspace_refuses_the_bare_collective():
    """The converse: reading the pre-norm output of a residual-adding workspace
    would return sum + residual, not the sum."""
    wrapper = object.__new__(FlashInferMNNVLCuteDSLARFusion)
    wrapper.fuse_residual = True

    with pytest.raises(RuntimeError, match="with the residual add"):
        wrapper.all_reduce(
            local_contribution=torch.empty(2, 8, dtype=torch.bfloat16),
            gamma=torch.empty(8, dtype=torch.bfloat16),
            norm_scratch=torch.empty(2, 8, dtype=torch.bfloat16),
        )


def test_bare_collective_reads_the_pre_norm_output():
    """The sum leaves through residual_out; norm_out is scratch the model never
    reads. Swapping them would silently return normalized values."""
    calls = {}
    wrapper = object.__new__(FlashInferMNNVLCuteDSLARFusion)
    wrapper.fuse_residual = False
    wrapper.rms_epsilon = 1e-5
    wrapper.weight_bias = 0.0
    wrapper.workspace = object()
    wrapper.supports = lambda m: True
    wrapper._patterns = SimpleNamespace(kARResidualRMSNorm=1)
    wrapper._allreduce_fusion = lambda **kw: calls.update(kw)

    local = torch.empty(2, 8, dtype=torch.bfloat16)
    scratch = torch.empty(2, 8, dtype=torch.bfloat16)
    out = wrapper.all_reduce(
        local_contribution=local,
        gamma=torch.empty(8, dtype=torch.bfloat16),
        norm_scratch=scratch,
    )

    assert calls["residual_out"] is out
    assert calls["norm_out"] is scratch
    assert calls["pattern"] == 1


def test_decode_max_m_scales_by_the_draft_token_count():
    """Speculative decode submits num_draft_tokens rows per request, and the
    workspace is compiled for a fixed M_max: under-sizing it silently drops the
    optimization at every eligible forward."""
    from sglang.srt.layers.moe.cutedsl_ar_fusion import resolve_decode_max_m

    reset_context()
    publish(
        ServerArgs(
            model_path="dummy",
            speculative_algorithm="EAGLE",
            speculative_num_draft_tokens=6,
            speculative_num_steps=5,
            speculative_eagle_topk=1,
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(max_bs=64, bs=[1, 16, 64]),
                prefill=PhaseConfig(max_bs=4096, bs=[4096]),
            ),
        ),
        role="test",
    )

    # Prefill bounds are excluded; 64 requests x 6 draft tokens.
    assert resolve_decode_max_m(max_running_requests=64) == 384


def _bare_communicator(*, attn_ok=True, mlp_ok=True, owes_local_reduction=False):
    """A CuteDSLBareAllReduceMHCLayerCommunicator with its static eligibility
    already resolved, so a case varies only what it means to vary."""
    from sglang.srt.layers.moe.cutedsl_ar_fusion import (
        CuteDSLBareAllReduceMHCLayerCommunicator,
    )

    comm = CuteDSLBareAllReduceMHCLayerCommunicator.__new__(
        CuteDSLBareAllReduceMHCLayerCommunicator
    )
    comm._attn_output_eligible = attn_ok
    comm._mlp_output_eligible = mlp_ok
    comm._mlp_allreduce_deferred = False
    comm.owes_local_reduction = owes_local_reduction
    comm.post_attention_layernorm = object()
    reduced = []
    comm.fusion_service = SimpleNamespace(
        supports=lambda m: True,
        all_reduce=lambda x: reduced.append(x) or (x * 2),
    )
    comm._reduced = reduced
    return comm


def _decode_batch(m=8):
    return SimpleNamespace(
        forward_mode=ForwardMode.DECODE, input_ids=torch.zeros(m, dtype=torch.int32)
    )


def _no_scattered_input():
    return patch(
        "sglang.srt.layers.moe.cutedsl_ar_fusion.get_attn_tp_context",
        return_value=SimpleNamespace(input_scattered=False),
    )


def test_bare_communicator_reduces_the_attention_output_then_combines_with_mhc():
    """The attention boundary: the collective replaces the ordinary all-reduce,
    and mHC still owns the combination. Dropping either would leave the
    guard-only tests green."""
    comm = _bare_communicator()
    combined = []
    comm.mhc = SimpleNamespace(
        attn_to_mlp=lambda h, r, out_norm: combined.append((h, r, out_norm)) or (h, r)
    )
    hidden, residual = torch.ones(8, 8), torch.zeros(8, 8)

    with _no_scattered_input():
        out_hidden, _ = comm.prepare_mlp(hidden, residual, _decode_batch())

    assert len(comm._reduced) == 1 and comm._reduced[0] is hidden
    assert torch.equal(out_hidden, hidden * 2)
    # mHC combined the reduced value, and against its own norm.
    assert len(combined) == 1
    assert combined[0][2] is comm.post_attention_layernorm


def test_bare_communicator_falls_back_when_the_forward_is_ineligible():
    """EXTEND is excluded, so the ordinary mHC path must run untouched."""
    from sglang.srt.layers.communicator_mhc import MHCLayerCommunicator

    comm = _bare_communicator()
    extend = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND, input_ids=torch.zeros(8, dtype=torch.int32)
    )

    with (
        _no_scattered_input(),
        patch.object(
            MHCLayerCommunicator, "prepare_mlp", return_value=("fallback", None)
        ) as ordinary,
    ):
        out, _ = comm.prepare_mlp(torch.ones(8, 8), torch.zeros(8, 8), extend)

    assert out == "fallback" and ordinary.called
    assert comm._reduced == []


def test_bare_communicator_reduces_the_mlp_output_it_deferred():
    """The MLP boundary is two halves: should_defer_mlp_allreduce suppresses the
    MoE's reduction, postprocess_layer performs it. Keeping only the first
    silently drops the reduction."""
    from sglang.srt.layers.communicator_mhc import MHCLayerCommunicator

    comm = _bare_communicator()
    hidden = torch.ones(8, 8)

    with _no_scattered_input():
        assert comm.should_defer_mlp_allreduce(_decode_batch()) is True
        with patch.object(
            MHCLayerCommunicator,
            "postprocess_layer",
            side_effect=lambda h, r, fb: (h, r),
        ):
            out, _ = comm.postprocess_layer(hidden, None, _decode_batch())

    assert len(comm._reduced) == 1
    assert torch.equal(out, hidden * 2)
    # The verdict is consumed, so the next forward re-decides.
    assert comm._mlp_allreduce_deferred is False


def test_bare_communicator_does_not_reduce_an_mlp_output_it_never_deferred():
    """postprocess_layer must not reduce when the MoE already did."""
    from sglang.srt.layers.communicator_mhc import MHCLayerCommunicator

    comm = _bare_communicator(mlp_ok=False)

    with _no_scattered_input():
        assert comm.should_defer_mlp_allreduce(_decode_batch()) is False
        with patch.object(
            MHCLayerCommunicator,
            "postprocess_layer",
            side_effect=lambda h, r, fb: (h, r),
        ):
            comm.postprocess_layer(torch.ones(8, 8), None, _decode_batch())

    assert comm._reduced == []


def test_a_replicated_shared_expert_producer_keeps_its_mlp_reduction():
    """A TP1 shared expert is added after the MoE's own all-reduce, so deferring
    that reduction to postprocess_layer would scale it by tp_size. The attention
    boundary is unaffected and stays eligible."""
    comm = _bare_communicator(owes_local_reduction=True)
    comm.mhc = SimpleNamespace(attn_to_mlp=lambda h, r, out_norm: (h, r))

    with _no_scattered_input():
        assert comm.should_defer_mlp_allreduce(_decode_batch()) is False
        assert comm._mlp_allreduce_deferred is False
        comm.prepare_mlp(torch.ones(8, 8), torch.zeros(8, 8), _decode_batch())

    # The attention-output reduction still ran.
    assert len(comm._reduced) == 1


def test_install_records_which_bare_producers_owe_a_local_reduction():
    """install must carry the model's predicate onto the layer, or a replicated
    shared expert silently keeps the unsafe deferral."""
    from sglang.srt.layers.moe.cutedsl_ar_fusion import (
        CuteDSLBareAllReduceMHCLayerCommunicator,
        install_cutedsl_bare_all_reduce,
    )

    layers = [
        SimpleNamespace(
            layer_communicator=CuteDSLBareAllReduceMHCLayerCommunicator.__new__(
                CuteDSLBareAllReduceMHCLayerCommunicator
            ),
            replicated=replicated,
        )
        for replicated in (True, False)
    ]
    install_cutedsl_bare_all_reduce(
        layers,
        hidden_size=8,
        top_k=2,
        rms_epsilon=1e-6,
        requires_local_reduction=lambda layer: layer.replicated,
        label="test",
    )

    assert layers[0].layer_communicator.owes_local_reduction is True
    assert layers[1].layer_communicator.owes_local_reduction is False


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
