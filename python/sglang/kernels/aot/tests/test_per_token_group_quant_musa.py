import pytest
import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    per_token_group_quant_8bit,
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.utils import is_musa

pytestmark = pytest.mark.skipif(not is_musa(), reason="MUSA-only Triton coverage")


def _empty_with_same_layout(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_strided(
        tensor.shape,
        tensor.stride(),
        dtype=tensor.dtype,
        device=tensor.device,
    )


@pytest.mark.parametrize("group_size", [16, 32, 64, 128])
@pytest.mark.parametrize("column_major_scales", [False, True])
def test_per_token_group_quant_allocating_and_caller_owned_outputs(
    group_size, column_major_scales
):
    """MUSA Triton group quant must preserve layout, ownership, and output bits."""
    torch.manual_seed(0)
    input = torch.randn(
        (7, group_size * 4), dtype=torch.bfloat16, device=torch.device("musa")
    )

    expected_q, expected_s = sglang_per_token_group_quant_fp8(
        x=input,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=False,
    )

    assert expected_q.data_ptr() != input.data_ptr()
    assert expected_q.is_contiguous()
    if column_major_scales:
        assert expected_s.stride(-2) == 1
        assert expected_s.stride(-1) >= expected_s.shape[-2]
    else:
        assert expected_s.is_contiguous()

    output_q = torch.empty_like(expected_q)
    output_s = _empty_with_same_layout(expected_s)
    actual_q, actual_s = per_token_group_quant_8bit(
        x=input,
        group_size=group_size,
        dst_dtype=torch.float8_e4m3fn,
        column_major_scales=column_major_scales,
        scale_tma_aligned=False,
        output_q=output_q,
        output_s=output_s,
    )

    assert actual_q.data_ptr() == output_q.data_ptr()
    assert actual_s.data_ptr() == output_s.data_ptr()
    assert actual_s.stride() == expected_s.stride()
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
