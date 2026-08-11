from typing import Optional

import torch


def awq_dequantize(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor
) -> torch.ByteTensor:
    return torch.ops.sgl_kernel.awq_dequantize.default(qweight, scales, qzeros)


def int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    return torch.ops.sgl_kernel.int8_scaled_mm.default(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )


def fp8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    return torch.ops.sgl_kernel.fp8_scaled_mm.default(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )


def shuffle_rows(input_tensor, dst2src_map, output_tensor_shape):
    output_tensor = torch.empty(
        output_tensor_shape,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    torch.ops.sgl_kernel.shuffle_rows.default(input_tensor, dst2src_map, output_tensor)
    return output_tensor


# GPTQ kernels
def gptq_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_gptq_qzeros: torch.Tensor,
    b_gptq_scales: torch.Tensor,
    b_g_idx: torch.Tensor,
    use_shuffle: bool,
    bit: int,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.gptq_gemm(
        a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, use_shuffle, bit
    )


def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor, bit: int) -> None:
    torch.torch.ops.sgl_kernel.gptq_shuffle(q_weight, q_perm, bit)
