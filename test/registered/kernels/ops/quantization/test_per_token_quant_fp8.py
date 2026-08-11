import sys

import pytest
import torch

from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.kernels.ops.quantization.per_token_quant_fp8 import per_token_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=16, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=16, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=30, stage="nightly", runner_config="1-gpu-large")


def _run_jit(input: torch.Tensor):
    output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    scale = torch.empty((input.shape[0], 1), dtype=torch.float32, device="cuda")
    per_token_quant_fp8(input, output, scale)
    return output, scale


def _torch_reference(input: torch.Tensor):
    row_max = input.float().abs().amax(dim=1)
    scale = row_max / torch.finfo(torch.float8_e4m3fn).max
    scale_inv = torch.where(scale == 0, 0, scale.reciprocal())
    output = (input.float() * scale_inv.unsqueeze(1)).clamp(-448.0, 448.0)
    return output.to(torch.float8_e4m3fn), scale.unsqueeze(1)


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor):
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def _warp_dispatch_num_tokens() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count * 16


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dispatch", ["cta", "warp"])
@pytest.mark.parametrize("hidden_dim", [1076, 1368])
def test_per_token_quant_fp8_matches_torch(dtype, dispatch, hidden_dim):
    """The JIT kernel must match FP32 row-wise reference quantization."""
    num_tokens = 39 if dispatch == "cta" else _warp_dispatch_num_tokens()
    input = torch.rand((num_tokens, hidden_dim), dtype=dtype, device="cuda")

    actual_output, actual_scale = _run_jit(input)
    expected_output, expected_scale = _torch_reference(input)

    torch.testing.assert_close(actual_scale, expected_scale, rtol=1e-6, atol=0)
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), rtol=0.125, atol=0.0625
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dispatch", ["cta", "warp"])
def test_per_token_quant_fp8_zero_rows(dtype, dispatch):
    """Pin the legacy zero-scale behavior of each launch strategy."""
    num_tokens = 1 if dispatch == "cta" else _warp_dispatch_num_tokens()
    input = torch.zeros((num_tokens, 512), dtype=dtype, device="cuda")

    actual_output, actual_scale = _run_jit(input)

    assert torch.count_nonzero(actual_scale) == 0
    expected_value = 448.0 if dispatch == "cta" else 0.0
    assert torch.all(actual_output.float() == expected_value)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dispatch", ["cta", "warp"])
def test_per_token_quant_fp8_midpoints_match_torch(dtype, dispatch):
    """FP8 rounding ties must select the reference representable value."""
    num_tokens = 1 if dispatch == "cta" else _warp_dispatch_num_tokens()
    midpoint_values = torch.tensor(
        [448.0, 1.0625, 1.1875, 1.375, -1.0625, -1.1875, -1.375],
        dtype=dtype,
        device="cuda",
    )
    input = midpoint_values.repeat(num_tokens, 512 // midpoint_values.numel() + 1)[
        :, :512
    ].contiguous()

    actual_output, actual_scale = _run_jit(input)
    expected_output, expected_scale = _torch_reference(input)

    torch.testing.assert_close(actual_scale, expected_scale, rtol=1e-6, atol=0)
    _assert_bitwise_equal(actual_output, expected_output)


def test_scaled_fp8_quant_accepts_padded_outputs():
    """Dynamic per-token quantization supports the serving padding contract."""
    input = torch.rand((1, 512), dtype=torch.float16, device="cuda")

    output, scale = scaled_fp8_quant(
        input, num_token_padding=17, use_per_token_if_dynamic=True
    )
    expected_output, expected_scale = _run_jit(input)

    assert output.shape == (17, 512)
    assert scale.shape == (17, 1)
    _assert_bitwise_equal(output[:1], expected_output)
    _assert_bitwise_equal(scale[:1], expected_scale)


def test_per_token_quant_fp8_preserves_padded_tail():
    input = torch.rand((1, 512), dtype=torch.float16, device="cuda")
    output = torch.full((17, 512), 1.0, dtype=torch.float8_e4m3fn, device="cuda")
    scale = torch.full((17, 1), 2.0, dtype=torch.float32, device="cuda")

    per_token_quant_fp8(input, output, scale)

    assert torch.all(output[1:].float() == 1.0)
    assert torch.all(scale[1:] == 2.0)


def test_per_token_quant_fp8_rejects_unsupported_dtype():
    input = torch.ones((1, 512), dtype=torch.int32, device="cuda")
    output = torch.empty((1, 512), dtype=torch.float8_e4m3fn, device="cuda")
    scale = torch.empty((1, 1), dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="Unsupported dtype"):
        per_token_quant_fp8(input, output, scale)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
