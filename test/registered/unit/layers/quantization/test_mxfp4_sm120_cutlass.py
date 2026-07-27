"""SM120 FlashInfer MXFP8-by-MXFP4 MoE integration test."""

from __future__ import annotations

import builtins
import importlib
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")


def _random_weights(num_experts: int, hidden: int, intermediate: int):
    generator = torch.Generator(device="cuda").manual_seed(0)
    w13 = torch.randint(
        -128,
        128,
        (num_experts, 2 * intermediate, hidden // 2),
        dtype=torch.int8,
        device="cuda",
        generator=generator,
    )
    w2 = torch.randint(
        -128,
        128,
        (num_experts, hidden, intermediate // 2),
        dtype=torch.int8,
        device="cuda",
        generator=generator,
    )
    w13_scale_u8 = torch.randint(
        125,
        130,
        (num_experts, 2 * intermediate, hidden // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w2_scale_u8 = torch.randint(
        125,
        130,
        (num_experts, hidden, intermediate // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    return (
        w13,
        w2,
        w13_scale_u8.view(torch.float8_e8m0fnu),
        w2_scale_u8.view(torch.float8_e8m0fnu),
    )


def test_cutlass_adapter_import_does_not_require_flashinfer(monkeypatch):
    module_name = "sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe"
    # Load the package before blocking FlashInfer so this test isolates the
    # adapter import exercised by non-CUDA backends.
    importlib.import_module("sglang.srt.layers.quantization")
    cached_module = sys.modules.pop(module_name, None)
    real_import = builtins.__import__

    def import_without_flashinfer(name, *args, **kwargs):
        if name == "flashinfer" or name.startswith("flashinfer."):
            raise ModuleNotFoundError("No module named 'flashinfer'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_flashinfer)
    try:
        module = importlib.import_module(module_name)
        assert hasattr(module, "Mxfp4FlashinferCutlassMoEMethod")
    finally:
        sys.modules.pop(module_name, None)
        if cached_module is not None:
            sys.modules[module_name] = cached_module


def test_dsv4_sm120_load_contract(monkeypatch):
    import sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe as adapter_module

    monkeypatch.setattr(adapter_module, "is_sm120_supported", lambda: True)

    captured = {}

    class _Fp8Method:
        def create_weights(self, *args, **kwargs):
            captured.update(kwargs)

    method = adapter_module.Mxfp4FlashinferCutlassMoEMethod(_Fp8Method(), "test")
    method.create_weights(
        SimpleNamespace(),
        num_experts=4,
        hidden_size=256,
        intermediate_size_per_partition=256,
        params_dtype=torch.bfloat16,
    )

    assert method.load_up_proj_weight_first
    assert captured["fp4_scale_dtype"] == torch.float8_e8m0fnu


def test_gpt_oss_sm120_requires_flashinfer_cutlass_support(monkeypatch):
    import sglang.srt.layers.quantization.mxfp4 as mxfp4_module
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    monkeypatch.setattr(
        mxfp4_module,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.FLASHINFER_MXFP4,
    )
    monkeypatch.setattr(
        mxfp4_module,
        "get_server_args",
        lambda: SimpleNamespace(flashinfer_mxfp4_moe_precision="default"),
    )
    monkeypatch.setattr(mxfp4_module, "is_sm100_supported", lambda: False)
    monkeypatch.setattr(mxfp4_module, "is_sm120_supported", lambda: True)
    monkeypatch.setattr(mxfp4_module, "_FI_HAS_SM120_CUTLASS_MXFP4", False)

    with pytest.raises(RuntimeError, match="MXFP8-by-MXFP4 support"):
        mxfp4_module.Mxfp4MoEMethod("test")


def test_gpt_oss_cutlass_runner_state_is_dwdp_rebindable():
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_local_experts = 2
            self.w13_weight = torch.nn.Parameter(torch.zeros(2, 1), requires_grad=False)

    layer = _Layer()
    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    method._fi_kernel = "cutlass_sm120"
    config = MoeRunnerConfig(
        gemm1_alpha=1.5,
        gemm1_clamp_limit=None,
        swiglu_limit=9.0,
    )
    method._initialize_cutlass_runner_state(layer, config)

    per_expert = dict(FusedMoE.named_per_expert_tensors(layer, 2))
    assert "mxfp4_weight_global_scale" in per_expert
    assert "swiglu_alpha" in per_expert
    assert "swiglu_beta" in per_expert
    assert "swiglu_limit" in per_expert
    assert torch.equal(layer.swiglu_alpha, torch.full((2,), 1.5))
    assert torch.equal(layer.swiglu_limit, torch.full((2,), 9.0))

    full_scale = torch.ones(4)
    FusedMoE.replace_expert_tensor(layer, "mxfp4_weight_global_scale", full_scale)
    assert isinstance(layer.mxfp4_weight_global_scale, torch.nn.Parameter)
    assert layer.mxfp4_weight_global_scale.shape == (4,)

    sm90_layer = _Layer()
    method._fi_kernel = "cutlass_sm90"
    sm90_config = MoeRunnerConfig(
        gemm1_alpha=1.25,
        gemm1_clamp_limit=8.0,
        swiglu_limit=9.0,
    )
    method._initialize_cutlass_runner_state(sm90_layer, sm90_config)
    assert sm90_layer.mxfp4_weight_global_scale is None
    assert torch.equal(sm90_layer.swiglu_alpha, torch.full((2,), 1.25))
    assert torch.equal(sm90_layer.swiglu_limit, torch.full((2,), 8.0))


def test_gpt_oss_cutlass_forwards_nondefault_topology():
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

    captured = {}

    class _Runner:
        def run(self, dispatch_output, quant_info):
            captured["quant_info"] = quant_info
            return dispatch_output

    tensor = torch.empty(1)
    layer = SimpleNamespace(
        w13_weight=tensor,
        w2_weight=tensor,
        w13_weight_scale=tensor,
        w2_weight_scale=tensor,
        mxfp4_weight_global_scale=tensor,
        w13_weight_bias=tensor,
        w2_weight_bias=tensor,
        swiglu_alpha=tensor,
        swiglu_beta=tensor,
        swiglu_limit=tensor,
        moe_tp_size=2,
        moe_tp_rank=1,
        moe_ep_size=4,
        moe_ep_rank=3,
    )
    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    method._fi_kernel = "cutlass_sm120"
    method._padded_hidden = 128
    method.runner = _Runner()
    dispatch_output = object()

    assert method._apply_cutlass(layer, dispatch_output) is dispatch_output
    quant_info = captured["quant_info"]
    assert (
        quant_info.moe_tp_size,
        quant_info.moe_tp_rank,
        quant_info.moe_ep_size,
        quant_info.moe_ep_rank,
    ) == (2, 1, 4, 3)
    assert quant_info.use_mxfp8_act_scaling


def test_dsv4_sm120_matches_direct_flashinfer(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120 required")
    pytest.importorskip("flashinfer.fused_moe")

    from flashinfer import block_scale_interleave, mxfp8_quantize
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass as runner_module
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
        Mxfp4FlashinferCutlassMoEMethod,
    )

    monkeypatch.setattr(
        runner_module, "use_symmetric_memory", lambda *args, **kwargs: nullcontext()
    )
    monkeypatch.setattr(runner_module, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(runner_module, "get_tp_group", lambda: None)

    num_experts, hidden, intermediate = 4, 256, 256
    w13, w2, w13_scale, w2_scale = _random_weights(num_experts, hidden, intermediate)
    w1, w3 = w13.chunk(2, dim=1)
    w1_scale, w3_scale = w13_scale.chunk(2, dim=1)
    # Simulate FusedMoE's ``load_up_proj_weight_first`` loader contract.
    w31 = torch.cat((w3, w1), dim=1)
    w31_scale = torch.cat(
        (w3_scale.view(torch.uint8), w1_scale.view(torch.uint8)),
        dim=1,
    ).view(torch.float8_e8m0fnu)
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(w31.clone(), requires_grad=False),
        w2_weight=torch.nn.Parameter(w2.clone(), requires_grad=False),
        w13_weight_scale_inv=torch.nn.Parameter(w31_scale.clone(), requires_grad=False),
        w2_weight_scale_inv=torch.nn.Parameter(w2_scale.clone(), requires_grad=False),
        num_local_experts=num_experts,
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
    )

    method = Mxfp4FlashinferCutlassMoEMethod(
        SimpleNamespace(process_weights_after_loading=lambda layer: None), "test"
    )
    config = MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=num_experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        top_k=2,
        activation="silu",
        is_gated=True,
        swiglu_limit=10,
    )
    method.create_moe_runner(layer, config)

    w13_parameter = layer.w13_weight
    w2_parameter = layer.w2_weight
    w13_scale_parameter = layer.w13_weight_scale_inv
    w2_scale_parameter = layer.w2_weight_scale_inv
    method.process_weights_after_loading(layer)

    expected_w13_scale = block_scale_interleave(w31_scale.view(torch.uint8)).reshape_as(
        w31_scale
    )
    expected_w2_scale = block_scale_interleave(w2_scale.view(torch.uint8)).reshape_as(
        w2_scale
    )
    assert layer.w13_weight is w13_parameter
    assert layer.w2_weight is w2_parameter
    assert layer.w13_weight_scale_inv is w13_scale_parameter
    assert layer.w2_weight_scale_inv is w2_scale_parameter
    assert torch.equal(layer.w13_weight_scale_inv.view(torch.uint8), expected_w13_scale)
    assert torch.equal(layer.w2_weight_scale_inv.view(torch.uint8), expected_w2_scale)

    generator = torch.Generator(device="cuda").manual_seed(1)
    x = (
        torch.randn(
            8,
            hidden,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.1
    )
    logits = torch.randn(
        8,
        num_experts,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), 2, dim=-1)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    topk = StandardTopKOutput(topk_weights, topk_ids.to(torch.int32), logits)
    dispatch_output = StandardDispatchOutput(x, None, topk)

    actual = method.apply(layer, dispatch_output).hidden_states

    x_quant, x_scale = mxfp8_quantize(
        x,
        is_sf_swizzled_layout=True,
        alignment=32,
    )
    global_scale = torch.ones(num_experts, dtype=torch.float32, device="cuda")
    swiglu_limit = torch.full((num_experts,), 10.0, dtype=torch.float32, device="cuda")
    expected = torch.empty_like(x)
    cutlass_fused_moe(
        input=x_quant,
        token_selected_experts=topk_ids.to(torch.int32),
        token_final_scales=topk_weights,
        fc1_expert_weights=layer.w13_weight.view(torch.int64),
        fc2_expert_weights=layer.w2_weight.view(torch.int64),
        output_dtype=torch.bfloat16,
        quant_scales=[
            layer.w13_weight_scale_inv.view(torch.int32),
            global_scale,
            layer.w2_weight_scale_inv.view(torch.int32),
            global_scale,
        ],
        input_sf=x_scale,
        # Compare the adapter's implicit defaults against the old explicit
        # alpha=1/beta=0 representation.
        swiglu_alpha=torch.ones(num_experts, dtype=torch.float32, device="cuda"),
        swiglu_beta=torch.zeros(num_experts, dtype=torch.float32, device="cuda"),
        swiglu_limit=swiglu_limit,
        use_mxfp8_act_scaling=True,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=8,
        output=expected,
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("hidden,intermediate", [(160, 160), (256, 256)])
def test_gpt_oss_sm120_matches_direct_flashinfer(monkeypatch, hidden, intermediate):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120 required")
    pytest.importorskip("flashinfer.fused_moe")

    from flashinfer import block_scale_interleave, mxfp8_quantize
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass as runner_module
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

    monkeypatch.setattr(
        runner_module, "use_symmetric_memory", lambda *args, **kwargs: nullcontext()
    )
    monkeypatch.setattr(runner_module, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(runner_module, "get_tp_group", lambda: None)

    num_experts = 2
    padded_hidden = padded_intermediate = 256
    generator = torch.Generator(device="cuda").manual_seed(2)
    w13 = torch.randint(
        0,
        128,
        (num_experts, 2 * intermediate, hidden // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w2 = torch.randint(
        0,
        128,
        (num_experts, hidden, intermediate // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w13_scale = torch.full(
        (num_experts, 2 * intermediate, hidden // 32),
        125,
        dtype=torch.uint8,
        device="cuda",
    )
    w2_scale = torch.full(
        (num_experts, hidden, intermediate // 32),
        125,
        dtype=torch.uint8,
        device="cuda",
    )
    w13_bias = torch.randn(
        num_experts,
        2 * intermediate,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    w2_bias = torch.randn(
        num_experts,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(w13.clone(), requires_grad=False),
        w2_weight=torch.nn.Parameter(w2.clone(), requires_grad=False),
        w13_weight_scale=torch.nn.Parameter(w13_scale.clone(), requires_grad=False),
        w2_weight_scale=torch.nn.Parameter(w2_scale.clone(), requires_grad=False),
        w13_weight_bias=torch.nn.Parameter(w13_bias.clone(), requires_grad=False),
        w2_weight_bias=torch.nn.Parameter(w2_bias.clone(), requires_grad=False),
        num_local_experts=num_experts,
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
    )
    config = MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=num_experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        top_k=1,
        activation="silu",
        is_gated=True,
        gemm1_alpha=1.702,
        gemm1_clamp_limit=7.0,
    )
    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    method._fi_kernel = "cutlass_sm120"
    method._padded_hidden = padded_hidden
    method._padded_intermediate = padded_intermediate
    method.moe_runner_config = config
    method._initialize_cutlass_runner_state(layer, config)
    original_w2_weight = layer.w2_weight
    original_w2_bias = layer.w2_weight_bias
    method._process_weights_for_sm120_cutlass(layer)
    method.runner = MoeRunner(MoeRunnerBackend.FLASHINFER_MXFP4, config)

    if hidden == padded_hidden and intermediate == padded_intermediate:
        assert layer.w2_weight is original_w2_weight
        assert layer.w2_weight_bias is original_w2_bias

    expected_w13 = torch.zeros(
        num_experts,
        2 * padded_intermediate,
        padded_hidden // 2,
        dtype=torch.uint8,
        device="cuda",
    )
    expected_w13[:, :intermediate, : hidden // 2] = w13[:, 1::2]
    expected_w13[
        :, padded_intermediate : padded_intermediate + intermediate, : hidden // 2
    ] = w13[:, 0::2]

    expected_w13_scale = torch.zeros(
        num_experts,
        2 * padded_intermediate,
        padded_hidden // 32,
        dtype=torch.uint8,
        device="cuda",
    )
    expected_w13_scale[:, :intermediate, : hidden // 32] = w13_scale[:, 1::2]
    expected_w13_scale[
        :,
        padded_intermediate : padded_intermediate + intermediate,
        : hidden // 32,
    ] = w13_scale[:, 0::2]
    expected_w13_scale = block_scale_interleave(expected_w13_scale).reshape_as(
        expected_w13_scale
    )

    expected_w2 = torch.zeros(
        num_experts,
        padded_hidden,
        padded_intermediate // 2,
        dtype=torch.uint8,
        device="cuda",
    )
    expected_w2[:, :hidden, : intermediate // 2] = w2
    expected_w2_scale = torch.zeros(
        num_experts,
        padded_hidden,
        padded_intermediate // 32,
        dtype=torch.uint8,
        device="cuda",
    )
    expected_w2_scale[:, :hidden, : intermediate // 32] = w2_scale
    expected_w2_scale = block_scale_interleave(expected_w2_scale).reshape_as(
        expected_w2_scale
    )

    expected_w13_bias = torch.zeros(
        num_experts,
        2 * padded_intermediate,
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected_w13_bias[:, :intermediate] = w13_bias[:, 1::2]
    expected_w13_bias[:, padded_intermediate : padded_intermediate + intermediate] = (
        w13_bias[:, 0::2]
    )
    expected_w2_bias = torch.zeros(
        num_experts,
        padded_hidden,
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected_w2_bias[:, :hidden] = w2_bias

    x = torch.randn(4, hidden, dtype=torch.bfloat16, device="cuda", generator=generator)
    logits = torch.randn(
        4, num_experts, dtype=torch.float32, device="cuda", generator=generator
    )
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), 1, dim=-1)
    dispatch_output = StandardDispatchOutput(
        x,
        None,
        StandardTopKOutput(topk_weights, topk_ids.to(torch.int32), logits),
    )
    actual = method._apply_cutlass(layer, dispatch_output).hidden_states

    x_padded = torch.nn.functional.pad(x, (0, padded_hidden - hidden))
    x_quant, x_scale = mxfp8_quantize(
        x_padded,
        is_sf_swizzled_layout=True,
        alignment=32,
    )
    expected_padded = torch.empty(
        x.shape[0],
        padded_hidden,
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected_global_scale = torch.ones(num_experts, dtype=torch.float32, device="cuda")
    cutlass_fused_moe(
        input=x_quant,
        token_selected_experts=topk_ids.to(torch.int32),
        token_final_scales=topk_weights,
        fc1_expert_weights=expected_w13.view(torch.int64),
        fc2_expert_weights=expected_w2.view(torch.int64),
        output_dtype=torch.bfloat16,
        quant_scales=[
            expected_w13_scale.view(torch.int32),
            expected_global_scale,
            expected_w2_scale.view(torch.int32),
            expected_global_scale,
        ],
        input_sf=x_scale,
        fc1_expert_biases=expected_w13_bias,
        fc2_expert_biases=expected_w2_bias,
        swiglu_alpha=torch.full(
            (num_experts,), 1.702, dtype=torch.float32, device="cuda"
        ),
        swiglu_beta=torch.ones(num_experts, dtype=torch.float32, device="cuda"),
        swiglu_limit=torch.full(
            (num_experts,), 7.0, dtype=torch.float32, device="cuda"
        ),
        use_mxfp8_act_scaling=True,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=4,
        output=expected_padded,
    )
    expected = expected_padded[:, :hidden].contiguous()

    assert torch.equal(actual, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
