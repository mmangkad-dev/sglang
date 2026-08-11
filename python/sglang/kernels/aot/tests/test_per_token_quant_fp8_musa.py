import pytest
import torch

from sglang.srt.utils import is_musa

pytestmark = pytest.mark.skipif(not is_musa(), reason="MUSA-only AOT coverage")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 39, 128])
@pytest.mark.parametrize("hidden_dim", [512, 1076])
def test_per_token_quant_fp8_musa_matches_torch(dtype, num_tokens, hidden_dim):
    """The retained MUSA AOT kernel must match FP32 row-wise quantization."""
    from sgl_kernel import sgl_per_token_quant_fp8

    input = torch.rand(
        (num_tokens, hidden_dim), dtype=dtype, device=torch.device("musa")
    )
    output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    scale = torch.empty((num_tokens, 1), dtype=torch.float32, device=input.device)

    sgl_per_token_quant_fp8(input, output, scale)

    row_max = input.float().abs().amax(dim=1)
    expected_scale = row_max / torch.finfo(torch.float8_e4m3fn).max
    expected_output = (input.float() / expected_scale.unsqueeze(1)).clamp(-448.0, 448.0)
    expected_output = expected_output.to(torch.float8_e4m3fn)

    torch.testing.assert_close(scale[:, 0], expected_scale, rtol=1e-6, atol=0)
    torch.testing.assert_close(output.float(), expected_output.float(), rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
