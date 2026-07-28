"""CPU tests for the MXFP4 DeepGEMM weight layout."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.moe_runner import deep_gemm as deep_gemm_runner
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    DeepGemmRunnerInput,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLDispatchOutput,
    DeepEPNormalDispatchOutput,
)
from sglang.srt.layers.quantization.mxfp4 import (
    Mxfp4MoEMethod,
    _configure_deepep_mxfp4_fp8_dispatcher,
    _get_deep_gemm_mxfp4_padded_hidden_size,
    _validate_deepep_mxfp4_fp8_dispatch,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _parameter(data):
    return nn.Parameter(data, requires_grad=False)


class TestMxfp4DeepGemmLayout(CustomTestCase):
    def test_runner_creation_does_not_require_initialized_dispatcher(self):
        runner_backend = SimpleNamespace(
            is_auto=lambda: False,
            is_aiter=lambda: False,
            is_triton_kernels=lambda: False,
            is_triton=lambda: False,
            is_marlin=lambda: False,
            is_deep_gemm=lambda: True,
            is_flashinfer_mxfp4=lambda: False,
        )
        a2a_backend = SimpleNamespace(is_deepep=lambda: True)
        server_args = SimpleNamespace(deepep_dispatcher_output_dtype="fp8")
        method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
        layer = nn.Module()

        with (
            patch(
                "sglang.srt.layers.quantization.mxfp4.get_moe_runner_backend",
                return_value=runner_backend,
            ),
            patch(
                "sglang.srt.layers.quantization.mxfp4.get_moe_a2a_backend",
                return_value=a2a_backend,
            ),
            patch(
                "sglang.srt.layers.moe.utils.get_server_args",
                return_value=server_args,
            ),
            patch("sglang.srt.layers.quantization.mxfp4.MoeRunner") as moe_runner,
        ):
            method.create_moe_runner(layer, SimpleNamespace())

        moe_runner.assert_called_once()
        self.assertNotIn("dispatcher", layer.__dict__)

    def test_bf16_dispatcher_override_is_rejected_at_configuration(self):
        class _Dispatcher:
            quant_config = None

            def set_quant_config(self, quant_config):
                self.quant_config = quant_config

        layer = SimpleNamespace(dispatcher=_Dispatcher())
        server_args = SimpleNamespace(deepep_dispatcher_output_dtype="bf16")

        with (
            patch(
                "sglang.srt.layers.moe.utils.get_server_args",
                return_value=server_args,
            ),
            self.assertRaisesRegex(ValueError, "effective dispatcher dtype is 'bf16'"),
        ):
            _configure_deepep_mxfp4_fp8_dispatcher(layer)

    def test_dispatcher_is_configured_after_layer_initialization(self):
        class _Dispatcher:
            quant_config = None

            def set_quant_config(self, quant_config):
                self.quant_config = quant_config

        dispatcher = _Dispatcher()
        layer = SimpleNamespace(dispatcher=dispatcher)
        server_args = SimpleNamespace(deepep_dispatcher_output_dtype="fp8")

        with patch(
            "sglang.srt.layers.moe.utils.get_server_args",
            return_value=server_args,
        ):
            _configure_deepep_mxfp4_fp8_dispatcher(layer)

        self.assertEqual(dispatcher.quant_config, {"dispatcher_output_dtype": "fp8"})

    def test_deep_gemm_rejects_non_deepep_a2a_backend(self):
        runner_backend = SimpleNamespace(
            is_triton_kernels=lambda: False,
            is_flashinfer_mxfp4=lambda: False,
            is_deep_gemm=lambda: True,
            is_marlin=lambda: False,
        )
        a2a_backend = SimpleNamespace(
            is_none=lambda: False,
            is_deepep=lambda: False,
            value="mooncake",
        )

        with (
            patch(
                "sglang.srt.layers.quantization.mxfp4.get_moe_runner_backend",
                return_value=runner_backend,
            ),
            patch(
                "sglang.srt.layers.quantization.mxfp4.get_moe_a2a_backend",
                return_value=a2a_backend,
            ),
            self.assertRaisesRegex(ValueError, "only.*none or deepep"),
        ):
            Mxfp4MoEMethod("experts")

    def test_weight_creation_rejects_missing_blackwell_deepgemm_recipe(self):
        method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deep_gemm = True

        with (
            patch.object(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False),
            self.assertRaisesRegex(RuntimeError, "FP8 x FP4 recipe"),
        ):
            method.create_weights(
                nn.Module(),
                num_experts=1,
                hidden_size=2880,
                intermediate_size_per_partition=2880,
                params_dtype=torch.bfloat16,
            )

    def test_deepep_low_latency_uses_supported_hidden_size(self):
        self.assertEqual(
            _get_deep_gemm_mxfp4_padded_hidden_size(2880, use_deepep_low_latency=False),
            2944,
        )
        self.assertEqual(
            _get_deep_gemm_mxfp4_padded_hidden_size(2880, use_deepep_low_latency=True),
            3072,
        )

    def test_process_weights_deinterleaves_pads_and_converts_scales(self):
        num_experts = 1
        intermediate_size = 32
        hidden_size = 32
        padded_intermediate = 128
        padded_hidden = 128

        layer = nn.Module()
        layer.num_local_experts = num_experts

        w13_values = (
            torch.arange(
                num_experts * 2 * intermediate_size * (hidden_size // 2),
                dtype=torch.int32,
            )
            .remainder(251)
            .to(torch.uint8)
        )
        layer.w13_weight = _parameter(
            w13_values.view(num_experts, 2 * intermediate_size, hidden_size // 2)
        )
        layer.w2_weight = _parameter(
            torch.arange(
                num_experts * hidden_size * (intermediate_size // 2),
                dtype=torch.int32,
            )
            .remainder(251)
            .to(torch.uint8)
            .view(num_experts, hidden_size, intermediate_size // 2)
        )

        w13_scale = torch.full(
            (num_experts, 2 * intermediate_size, hidden_size // 32),
            126,
            dtype=torch.uint8,
        )
        w13_scale[:, 1::2] = 125
        layer.w13_weight_scale = _parameter(w13_scale)
        layer.w2_weight_scale = _parameter(
            torch.full(
                (num_experts, hidden_size, intermediate_size // 32),
                124,
                dtype=torch.uint8,
            )
        )
        layer.w13_weight_bias = _parameter(
            torch.arange(2 * intermediate_size, dtype=torch.bfloat16).view(
                num_experts, -1
            )
        )
        layer.w2_weight_bias = _parameter(
            torch.arange(hidden_size, dtype=torch.bfloat16).view(num_experts, -1)
        )

        method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
        method.intermediate_size_per_partition = padded_intermediate
        method.hidden_size = padded_hidden

        original_w13 = layer.w13_weight.detach().clone()
        original_w2 = layer.w2_weight.detach().clone()
        with patch.object(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False):
            method._process_weights_for_deep_gemm(layer)

        self.assertEqual(
            layer.w13_weight.shape,
            (num_experts, 2 * padded_intermediate, padded_hidden // 2),
        )
        self.assertEqual(
            layer.w2_weight.shape,
            (num_experts, padded_hidden, padded_intermediate // 2),
        )
        self.assertEqual(layer.w13_weight.dtype, torch.int8)
        self.assertEqual(layer.w2_weight.dtype, torch.int8)

        w13_uint8 = layer.w13_weight.view(torch.uint8)
        torch.testing.assert_close(
            w13_uint8[:, :intermediate_size, : hidden_size // 2],
            original_w13[:, 0::2],
        )
        torch.testing.assert_close(
            w13_uint8[
                :,
                padded_intermediate : padded_intermediate + intermediate_size,
                : hidden_size // 2,
            ],
            original_w13[:, 1::2],
        )
        torch.testing.assert_close(
            layer.w2_weight.view(torch.uint8)[
                :, :hidden_size, : intermediate_size // 2
            ],
            original_w2,
        )

        self.assertEqual(
            layer.w13_weight_scale.shape,
            (num_experts, 2 * padded_intermediate, padded_hidden // 32),
        )
        self.assertEqual(
            layer.w2_weight_scale.shape,
            (num_experts, padded_hidden, padded_intermediate // 32),
        )
        torch.testing.assert_close(
            layer.w13_weight_scale[:, :intermediate_size, 0],
            torch.full((num_experts, intermediate_size), 0.5),
        )
        torch.testing.assert_close(
            layer.w13_weight_scale[
                :,
                padded_intermediate : padded_intermediate + intermediate_size,
                0,
            ],
            torch.full((num_experts, intermediate_size), 0.25),
        )
        torch.testing.assert_close(
            layer.w2_weight_scale[:, :hidden_size, 0],
            torch.full((num_experts, hidden_size), 0.125),
        )
        self.assertTrue(
            torch.all(layer.w13_weight_scale[:, intermediate_size:128] == 1)
        )

        torch.testing.assert_close(
            layer.w13_weight_bias[:, :intermediate_size],
            torch.arange(0, 2 * intermediate_size, 2, dtype=torch.bfloat16).view(
                num_experts, -1
            ),
        )
        torch.testing.assert_close(
            layer.w13_weight_bias[
                :,
                padded_intermediate : padded_intermediate + intermediate_size,
            ],
            torch.arange(1, 2 * intermediate_size, 2, dtype=torch.bfloat16).view(
                num_experts, -1
            ),
        )
        torch.testing.assert_close(
            layer.w2_weight_bias[:, :hidden_size],
            torch.arange(hidden_size, dtype=torch.bfloat16).view(num_experts, -1),
        )

    def test_normal_deepep_rejects_bf16_dispatch(self):
        dispatch_output = DeepEPNormalDispatchOutput(
            hidden_states=torch.zeros(2, 128, dtype=torch.bfloat16),
            hidden_states_scale=None,
            topk_ids=torch.zeros(2, 1, dtype=torch.int64),
            topk_weights=torch.ones(2, 1),
            num_recv_tokens_per_expert=[2],
        )

        with self.assertRaisesRegex(ValueError, "requires FP8 dispatcher output"):
            _validate_deepep_mxfp4_fp8_dispatch(dispatch_output)

    def test_low_latency_deepep_rejects_bf16_dispatch(self):
        dispatch_output = DeepEPLLDispatchOutput(
            hidden_states=torch.zeros(1, 2, 128, dtype=torch.bfloat16),
            hidden_states_scale=None,
            topk_ids=torch.zeros(2, 1, dtype=torch.int64),
            topk_weights=torch.ones(2, 1),
            masked_m=torch.tensor([2], dtype=torch.int32),
            expected_m=2,
        )

        with self.assertRaisesRegex(ValueError, "requires FP8 dispatcher output"):
            _validate_deepep_mxfp4_fp8_dispatch(dispatch_output)

    def test_deepep_accepts_fp8_dispatch_with_scales(self):
        dispatch_output = DeepEPNormalDispatchOutput(
            hidden_states=torch.zeros(2, 128, dtype=torch.float8_e4m3fn),
            hidden_states_scale=torch.ones(2, 1),
            topk_ids=torch.zeros(2, 1, dtype=torch.int64),
            topk_weights=torch.ones(2, 1),
            num_recv_tokens_per_expert=[2],
        )

        _validate_deepep_mxfp4_fp8_dispatch(dispatch_output)

    def test_masked_runner_applies_both_expert_biases(self):
        num_experts, max_tokens, gate_up_size, output_size = 2, 3, 4, 5
        runner = DeepGemmRunnerCore.__new__(DeepGemmRunnerCore)
        runner.config = SimpleNamespace(
            top_k=1,
            gemm1_alpha=1.702,
            gemm1_clamp_limit=7.0,
        )
        runner.swiglu_limit = None
        runner.use_swizzle = False

        runner_input = DeepGemmRunnerInput(
            hidden_states=torch.zeros(num_experts, max_tokens, 4),
            hidden_states_scale=torch.ones(num_experts, max_tokens, 1),
            use_masked_gemm=True,
            masked_m=torch.full((num_experts,), max_tokens, dtype=torch.int32),
            expected_m=max_tokens,
        )
        w13_bias = torch.arange(num_experts * gate_up_size, dtype=torch.bfloat16).view(
            num_experts, gate_up_size
        )
        w2_bias = torch.arange(num_experts * output_size, dtype=torch.bfloat16).view(
            num_experts, output_size
        )
        quant_info = DeepGemmMoeQuantInfo(
            w13_weight=torch.zeros(num_experts, gate_up_size, 2, dtype=torch.int8),
            w2_weight=torch.zeros(num_experts, output_size, 1, dtype=torch.int8),
            use_fp8=True,
            w13_scale=torch.ones(num_experts, gate_up_size, 1),
            w2_scale=torch.ones(num_experts, output_size, 1),
            w13_bias=w13_bias,
            w2_bias=w2_bias,
            block_shape=[1, 32],
            is_fp4_experts=True,
        )
        running_state = {"hidden_states_device": torch.device("cpu")}
        activation_input = None

        def fake_grouped_gemm(_lhs, _rhs, output, *_args, **_kwargs):
            output.zero_()

        def fake_activation(gateup_output, *_args, **_kwargs):
            nonlocal activation_input
            activation_input = gateup_output.clone()
            return (
                torch.zeros(num_experts, max_tokens, gate_up_size // 2),
                torch.ones(num_experts, max_tokens, 1),
            )

        with (
            patch.object(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False),
            patch.object(deep_gemm_wrapper, "DEEPGEMM_NEED_TMA_ALIGNED_SCALES", False),
            patch.object(
                deep_gemm_wrapper,
                "grouped_gemm_nt_f8f8bf16_masked",
                side_effect=fake_grouped_gemm,
            ),
            patch.object(
                deep_gemm_runner,
                "_varlen_deep_gemm_silu_mul_quant",
                side_effect=fake_activation,
            ),
            patch.object(deep_gemm_runner, "dispose_tensor"),
            patch.object(
                deep_gemm_runner,
                "use_symmetric_memory",
                side_effect=lambda *_args, **_kwargs: nullcontext(),
            ),
            patch.object(deep_gemm_runner, "get_tp_group", return_value=None),
            patch.object(
                deep_gemm_runner, "is_allocation_symmetric", return_value=False
            ),
        ):
            output = runner._run_masked_gemm(runner_input, quant_info, running_state)

        self.assertIsNotNone(activation_input)
        torch.testing.assert_close(
            activation_input,
            w13_bias[:, None, :].expand(-1, max_tokens, -1),
        )
        torch.testing.assert_close(
            output,
            w2_bias[:, None, :].expand(-1, max_tokens, -1),
        )

    def test_contiguous_runner_applies_biases_before_and_after_activation(self):
        num_experts, num_tokens, gate_up_size, output_size = 2, 3, 4, 5
        runner = DeepGemmRunnerCore.__new__(DeepGemmRunnerCore)
        runner.config = SimpleNamespace(
            top_k=1,
            gemm1_alpha=1.702,
            gemm1_clamp_limit=7.0,
        )
        runner.swiglu_limit = None
        runner.use_swizzle = False

        m_indices = torch.tensor([0, 1, 0], dtype=torch.int32)
        runner_input = DeepGemmRunnerInput(
            hidden_states=torch.zeros(num_tokens, 4, dtype=torch.float8_e4m3fn),
            hidden_states_scale=torch.ones(num_tokens, 1),
            use_masked_gemm=False,
            m_indices=m_indices,
        )
        w13_bias = torch.arange(num_experts * gate_up_size, dtype=torch.bfloat16).view(
            num_experts, gate_up_size
        )
        w2_bias = torch.arange(num_experts * output_size, dtype=torch.bfloat16).view(
            num_experts, output_size
        )
        quant_info = DeepGemmMoeQuantInfo(
            w13_weight=torch.zeros(num_experts, gate_up_size, 2, dtype=torch.int8),
            w2_weight=torch.zeros(num_experts, output_size, 1, dtype=torch.int8),
            use_fp8=True,
            w13_scale=torch.ones(num_experts, gate_up_size, 1),
            w2_scale=torch.ones(num_experts, output_size, 1),
            w13_bias=w13_bias,
            w2_bias=w2_bias,
            block_shape=[1, 32],
            is_fp4_experts=True,
        )
        running_state = {
            "all_tokens": num_tokens,
            "hidden_states_device": torch.device("cpu"),
            "hidden_states_shape": torch.Size([num_tokens, output_size]),
        }
        activation_input = None

        def fake_grouped_gemm(_lhs, _rhs, output, *_args, **_kwargs):
            output.zero_()

        def fake_activation(gateup_output, *_args, **_kwargs):
            nonlocal activation_input
            activation_input = gateup_output.clone()
            return torch.zeros(num_tokens, gate_up_size // 2)

        def fake_quant(down_input, *_args, **_kwargs):
            return (
                down_input.to(torch.float8_e4m3fn),
                torch.ones(num_tokens, 1),
            )

        with (
            patch.object(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False),
            patch.object(deep_gemm_wrapper, "DEEPGEMM_NEED_TMA_ALIGNED_SCALES", False),
            patch.object(
                deep_gemm_wrapper,
                "grouped_gemm_nt_f8f8bf16_contig",
                side_effect=fake_grouped_gemm,
            ),
            patch(
                "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe."
                "swiglu_no_interleaved_with_alpha_and_limit",
                side_effect=fake_activation,
            ),
            patch(
                "sglang.kernels.ops.quantization.fp8_kernel."
                "sglang_per_token_group_quant_fp8",
                side_effect=fake_quant,
            ),
            patch.object(deep_gemm_runner, "dispose_tensor"),
            patch.object(
                deep_gemm_runner,
                "use_symmetric_memory",
                side_effect=lambda *_args, **_kwargs: nullcontext(),
            ),
            patch.object(deep_gemm_runner, "get_tp_group", return_value=None),
            patch.object(
                deep_gemm_runner, "is_allocation_symmetric", return_value=False
            ),
        ):
            output = runner._run_contiguous_gemm(
                runner_input, quant_info, running_state
            )

        self.assertIsNotNone(activation_input)
        torch.testing.assert_close(activation_input, w13_bias[m_indices])
        torch.testing.assert_close(output, w2_bias[m_indices])


if __name__ == "__main__":
    unittest.main()
