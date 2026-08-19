from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)

from .utils import make_name


@cache_once
def _jit_topk_v1_module():
    # topk (<= 1024) is a runtime argument, not a compile-time constant, so a
    # single module serves every k. Baking it in via -DSGL_TOPK used to build one
    # module per k, and since the macro fed a `constexpr` rather than a template
    # parameter every module exported identically mangled symbols -- see the
    # comment in topk_v1.cuh for how that broke the second module's launch.
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("topk_v1"),
        *args,
        cuda_files=["deepseek_v4/topk_v1.cuh"],
        cuda_wrappers=[("topk_transform", f"TopKKernel<{args}>::transform")],
    )


@cache_once
def _jit_topk_v2_module():
    # v2 is universal: topk (<= 2048) is a runtime argument, not a compile-time
    # constant, so a single module serves every k.
    return load_jit(
        make_name("topk_v2"),
        cuda_files=["deepseek_v4/topk_v2.cuh"],
        cuda_wrappers=[
            ("topk_transform", "TopKKernel::transform"),
            ("topk_transform_extend", "TopKKernel::transform_extend"),
            ("topk_plan", "TopKKernel::plan"),
        ],
    )


def topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    if is_hip_runtime():
        torch.ops.sgl_kernel.deepseek_v4_topk_transform_512(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
        )
    else:
        module = _jit_topk_v1_module()
        module.topk_transform(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
        )


# metadata is (batch+1, 2) int32: row 0 = {cluster_threshold, num_cluster_items};
# rows 1..N = {batch_id, seq_len} of items routed to the persistent cluster pool.
_PLAN_METADATA_INTS_PER_BATCH = 2


def plan_topk_v2(seq_lens: torch.Tensor, static_threshold: int = 0) -> torch.Tensor:
    """Preprocess the per-batch routing plan for :func:`topk_transform_512_v2`.

    IMPORTANT: every entry of ``seq_lens`` must be NON-NEGATIVE. The device
    kernel reads the int32 buffer as ``uint32_t``, so a negative length (e.g.
    -4 from a DP-padded / idle-companion row) reinterprets as ~4e9, poisons
    the plan, and drives the transform kernel into an illegal memory access.
    Producers of padded rows must clamp their lengths to 0 (0 selects the
    trivial all-(-1) output path, which is safe).
    """
    module = _jit_topk_v2_module()
    bs = seq_lens.shape[0]
    metadata = seq_lens.new_empty(bs + 1, _PLAN_METADATA_INTS_PER_BATCH)
    module.topk_plan(seq_lens, metadata, static_threshold)
    return metadata


def topk_transform_512_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    metadata: torch.Tensor,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Fused top-k + page-table transform (DeepSeek-V4 top-k v2 kernel).

    IMPORTANT: every entry of ``seq_lens`` must be NON-NEGATIVE, and
    ``metadata`` must come from :func:`plan_topk_v2` over the same ``seq_lens``
    values. The kernel reads lengths as ``uint32_t``: a negative entry
    reinterprets as a ~4e9-token sequence, sending the row down the cluster
    path over garbage scores and crashing with an illegal memory access
    (GLM 5.2 MTP DP-idle companion rows hit exactly this). A length of 0 is
    the valid way to express "no tokens": the row takes the trivial path and
    the output is all -1.
    """
    module = _jit_topk_v2_module()
    module.topk_transform(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        metadata,
        out_raw_indices,
    )


def topk_transform_extend_v2(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    row_starts: torch.Tensor,
    out: torch.Tensor,
    metadata: torch.Tensor,
    max_seq_len: int,
    page_size: int = 0,
    page_table: Optional[torch.Tensor] = None,
    row_to_batch: Optional[torch.Tensor] = None,
    out_offsets: Optional[torch.Tensor] = None,
) -> None:
    """Fused top-k + output transform for the extend phase (DeepSeek-V4 top-k v2).

    The extend counterpart of :func:`topk_transform_512_v2`: row ``b`` selects the
    top-k of ``scores[b, row_starts[b] : row_starts[b] + lengths[b]]`` and writes
    the transformed positions into ``out[b]``, ``-1`` padded. Exactly one output
    transform must be given:

    * paged -- ``page_table`` (the compact page-size-``page_size`` table) plus
      ``row_to_batch`` (row -> page-table row): emits
      ``page_table[row_to_batch[b], p // page_size] * page_size + p % page_size``,
      the same physical page-size-1 KV slots as sgl_kernel's
      ``fast_topk_transform_fused`` without materializing a page-size-1 table.
    * ragged -- ``out_offsets``: emits ``p + out_offsets[b]``, matching
      ``fast_topk_transform_ragged_fused``.

    ``max_seq_len`` is the kernel-selection bound and MUST be ``>= lengths.max()``
    (a too-low bound silently picks a level that cannot cover the longest row). It
    is a separate argument because an extend score matrix is as wide as the whole
    batch's concatenated KV, so its width is a far looser bound than any row's
    length -- unlike decode, where the two coincide.

    ``metadata`` must come from :func:`plan_topk_v2` over the same ``lengths``, and
    every entry of ``lengths`` must be NON-NEGATIVE (the kernel reads them as
    ``uint32_t``; see :func:`topk_transform_512_v2`). ``scores`` must be fp32 with
    unit row stride and a row stride that is a multiple of 4 (16-byte vectorized
    loads); ``row_starts`` may be arbitrary -- the kernel scalar-loads each row's
    unaligned head.
    """
    module = _jit_topk_v2_module()
    module.topk_transform_extend(
        scores,
        lengths,
        row_starts,
        out,
        metadata,
        page_size,
        max_seq_len,
        page_table,
        row_to_batch,
        out_offsets,
    )
