import torch
import triton
import triton.language as tl


@triton.jit
def _zero_padded_rows_kernel(
    tokens_ptr,
    valid_rows_ptr,
    row_width,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_rows = tl.load(valid_rows_ptr)
    mask = (row >= valid_rows) & (col < row_width)
    tl.store(tokens_ptr + row * row_width + col, 0, mask=mask)


def zero_padded_rows(tokens: torch.Tensor, valid_rows: torch.Tensor) -> None:
    """Zero rows ``valid_rows:`` using a device-resident scalar count."""
    if tokens.shape[0] == 0:
        return
    assert tokens.is_contiguous()
    assert valid_rows.numel() == 1
    row_width = tokens.numel() // tokens.shape[0]
    block_size = min(triton.next_power_of_2(row_width), 1024)
    _zero_padded_rows_kernel[(tokens.shape[0], triton.cdiv(row_width, block_size))](
        tokens,
        valid_rows,
        row_width,
        BLOCK_SIZE=block_size,
    )
