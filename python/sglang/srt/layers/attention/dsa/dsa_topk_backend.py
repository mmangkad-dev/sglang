from __future__ import annotations

from enum import Enum, IntEnum, auto
from typing import Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs

_FLASHINFER_TIE_BREAK_VALUES = {
    "small": 1,
    "large": 2,
}


class TopkTransformMethod(IntEnum):
    # Transform topk indices to indices to the page table (page_size = 1)
    PAGED = auto()
    # Transform topk indices to indices to ragged kv (non-paged)
    RAGGED = auto()


class DSATopKBackend(Enum):
    SGL_KERNEL = "sgl-kernel"
    TORCH = "torch"
    FLASHINFER = "flashinfer"

    def is_sgl_kernel(self) -> bool:
        return self == DSATopKBackend.SGL_KERNEL

    def is_torch(self) -> bool:
        return self == DSATopKBackend.TORCH

    def is_flashinfer(self) -> bool:
        return self == DSATopKBackend.FLASHINFER

    def should_use_topk_v2(self) -> bool:
        return self.is_sgl_kernel() and envs.SGLANG_OPT_USE_TOPK_V2.get()

    def topk_func(
        self,
        score: torch.Tensor,
        lengths: torch.Tensor,
        topk: int,
        row_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_sgl_kernel():
            from sgl_kernel import fast_topk_v2

            return fast_topk_v2(score, lengths, topk, row_starts=row_starts)
        if self.is_torch():
            return _topk_unfused(
                score,
                lengths,
                topk,
                row_starts=row_starts,
                topk_op=torch.topk,
                topk_op_kwargs={"dim": -1},
            )
        if self.is_flashinfer():
            import flashinfer

            return _topk_unfused(
                score,
                lengths,
                topk,
                row_starts=row_starts,
                topk_op=flashinfer.top_k,
                topk_op_kwargs={
                    "sorted": False,
                    "deterministic": envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
                    "tie_break": _flashinfer_tie_break_value(),
                    "dsa_graph_safe": True,
                },
            )
        raise RuntimeError(f"Unsupported {self = }.")

    def topk_transform(
        self,
        logits: torch.Tensor,
        lengths: torch.Tensor,
        topk: int,
        topk_transform_method: TopkTransformMethod,
        attn_metadata,
        cu_seqlens_q_topk: Optional[torch.Tensor] = None,
        topk_indices_offset: Optional[torch.Tensor] = None,
        row_starts: Optional[torch.Tensor] = None,
        batch_idx_list: Optional[List[int]] = None,
        force_unfused_topk: bool = False,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``out``: optional caller-owned ``(num_rows, topk)`` int32 destination.
        Only the extend v2 path honors it (it saves copying the result into the
        graph's static top-k buffer); every other path ignores it and returns a
        freshly allocated tensor, which the caller must still handle."""
        if not envs.SGLANG_DSA_FUSE_TOPK.get() or force_unfused_topk:
            return self.topk_func(logits, lengths, topk, row_starts=row_starts)

        # Decode-shaped PAGED top-k for the SGL backend (plain decode AND spec
        # verify / draft-extend, whose expanded rows match the same shape) routes
        # to the DeepSeek-V4 top-k v2 JIT kernel. It fuses top-k selection and the
        # page-table transform in one launch and consumes the indexer's own
        # page_size>=1 table directly, so no page_size=1 table is materialized.
        # Shared by DeepSeek-V3.2 and GLM DSA.
        # This is a deterministic dispatch on the work shape, not a best-effort
        # attempt: the fused-decode CUDA graph drops the page_size=1 table for
        # exactly this case (see dsa_drop_wide_page_table), so once the shape
        # matches we commit to v2 and never silently fall back to the legacy
        # page_size=1 path from here.
        if (
            self.should_use_topk_v2()
            and topk_transform_method == TopkTransformMethod.PAGED
            and row_starts is None
            and batch_idx_list is None
            and 0 < topk <= 2048
            and lengths.shape[0]
            == logits.shape[0]
            == attn_metadata.real_page_table.shape[0]
        ):
            return _topk_transform_v2_paged(logits, lengths, topk, attn_metadata)

        # Extend-shaped top-k for the SGL backend routes to the same DeepSeek-V4
        # v2 JIT kernel through its extend entry point, which adds the per-row
        # score window (row_starts) the prefill logits matrix needs. Both output
        # transforms are covered: PAGED reads the compact page-size-64 table via
        # row_to_batch (so the wide page-size-1 gather disappears) and RAGGED adds
        # topk_indices_offset. Measured 1.4-2.3x the legacy tilelang-derived
        # kernels on prefill shapes -- the scores stay register-resident and are
        # read once instead of twice.
        #
        # Unlike the decode PAGED dispatch above this is a best-effort fast path:
        # the legacy kernels remain correct for every input, so preconditions the
        # v2 kernel cannot serve (a chunked-logits call whose per-chunk row count
        # no longer matches the per-forward plan, a non-16B-aligned score stride,
        # the dummy-logits path that passes no row starts) fall through to them
        # below rather than failing.
        if self.should_use_topk_v2():
            v2_extend = _topk_transform_v2_extend(
                logits=logits,
                lengths=lengths,
                topk=topk,
                topk_transform_method=topk_transform_method,
                row_starts=row_starts,
                topk_indices_offset=topk_indices_offset,
                batch_idx_list=batch_idx_list,
                attn_metadata=attn_metadata,
                out=out,
            )
            if v2_extend is not None:
                return v2_extend

        # The legacy transforms below read attn_metadata.page_table_1 (page_size=1),
        # which is always present here: the fold only drops it for the decode case
        # dispatched to v2 above.
        assert attn_metadata.page_table_1 is not None

        if self.is_sgl_kernel():
            from sgl_kernel import (
                fast_topk_transform_fused,
                fast_topk_transform_ragged_fused,
            )

            if topk_transform_method == TopkTransformMethod.PAGED:
                page_table_size_1 = (
                    attn_metadata.page_table_1[batch_idx_list]
                    if batch_idx_list is not None
                    else attn_metadata.page_table_1
                )
                return fast_topk_transform_fused(
                    score=logits,
                    lengths=lengths,
                    page_table_size_1=page_table_size_1,
                    cu_seqlens_q=cu_seqlens_q_topk,
                    topk=topk,
                    row_starts=row_starts,
                )
            if topk_transform_method == TopkTransformMethod.RAGGED:
                if topk_indices_offset is None:
                    raise RuntimeError(
                        "RAGGED topk_transform requires topk_indices_offset; "
                        "expected extend-without-speculative metadata."
                    )
                return fast_topk_transform_ragged_fused(
                    score=logits,
                    lengths=lengths,
                    topk_indices_offset=topk_indices_offset,
                    topk=topk,
                    row_starts=row_starts,
                )
            raise RuntimeError(f"Unsupported {topk_transform_method = }.")

        if self.is_flashinfer():
            import flashinfer

            if topk_transform_method == TopkTransformMethod.PAGED:
                row_to_batch, page_table_row_starts = _build_flashinfer_paged_args(
                    attn_metadata=attn_metadata,
                    row_starts=row_starts,
                    cu_seqlens_q_topk=cu_seqlens_q_topk,
                    batch_idx_list=batch_idx_list,
                    device=logits.device,
                    num_rows=logits.shape[0],
                )
                return flashinfer.top_k_page_table_transform(
                    logits.contiguous(),
                    attn_metadata.page_table_1.contiguous(),
                    lengths.contiguous(),
                    topk,
                    row_to_batch=row_to_batch,
                    deterministic=envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
                    tie_break=_flashinfer_tie_break_value(),
                    dsa_graph_safe=True,
                    row_starts=row_starts,
                    page_table_row_starts=page_table_row_starts,
                )
            if topk_transform_method == TopkTransformMethod.RAGGED:
                if topk_indices_offset is None:
                    raise RuntimeError(
                        "RAGGED topk_transform requires topk_indices_offset; "
                        "expected extend-without-speculative metadata."
                    )
                return flashinfer.top_k_ragged_transform(
                    logits.contiguous(),
                    topk_indices_offset.contiguous(),
                    lengths.contiguous(),
                    topk,
                    deterministic=envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
                    tie_break=_flashinfer_tie_break_value(),
                    dsa_graph_safe=True,
                    row_starts=row_starts,
                )
            raise RuntimeError(f"Unsupported {topk_transform_method = }.")

        raise RuntimeError(f"Unsupported {self = } for SGLANG_DSA_FUSE_TOPK.")


def _topk_unfused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
    topk_op: Callable[..., Tuple[torch.Tensor, torch.Tensor]] = torch.topk,
    topk_op_kwargs: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    batch_size, max_score_len = score.shape
    topk_indices = score.new_full((batch_size, topk), -1, dtype=torch.int32)
    if batch_size == 0 or topk == 0 or max_score_len == 0:
        return topk_indices

    if row_starts is None:
        row_starts = torch.zeros_like(lengths, dtype=torch.int32, device=score.device)
    else:
        row_starts = row_starts.to(dtype=torch.int32, device=score.device)
    lengths = lengths.to(dtype=torch.int32, device=score.device)

    col_indices = torch.arange(max_score_len, dtype=torch.int32, device=score.device)
    col_indices = col_indices.unsqueeze(0)
    row_starts_unsqueezed = row_starts.unsqueeze(1)
    row_ends_unsqueezed = (row_starts + lengths).unsqueeze(1)
    valid_mask = (col_indices >= row_starts_unsqueezed) & (
        col_indices < row_ends_unsqueezed
    )

    masked_logits = score.masked_fill(~valid_mask, float("-inf"))
    valid_topk = min(topk, max_score_len)
    topk_kwargs = topk_op_kwargs or {}
    topk_scores, topk_col_indices = topk_op(masked_logits, valid_topk, **topk_kwargs)
    topk_local_indices = topk_col_indices.to(torch.int32) - row_starts_unsqueezed
    topk_local_indices = topk_local_indices.masked_fill(
        topk_scores == float("-inf"), -1
    )
    topk_indices[:, :valid_topk] = topk_local_indices

    return topk_indices


def _topk_transform_v2_extend(
    *,
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    topk_transform_method: TopkTransformMethod,
    row_starts: Optional[torch.Tensor],
    topk_indices_offset: Optional[torch.Tensor],
    batch_idx_list: Optional[List[int]],
    attn_metadata,
    out: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Extend-phase fused top-k + transform via the DeepSeek-V4 v2 JIT kernel.

    Returns ``(num_rows, topk)`` int32 indices -- physical page-size-1 KV slots for
    PAGED, ragged-buffer indices for RAGGED, ``-1`` padded -- identical in meaning
    to ``fast_topk_transform_fused`` / ``fast_topk_transform_ragged_fused``.
    Returns None when this call does not meet the kernel's preconditions, so the
    caller falls through to those legacy kernels.

    Every precondition is a property of shapes/strides/dtypes the caller already
    holds on the host -- no device reads -- so the check is cheap and CUDA-graph
    safe.
    """
    num_rows = logits.shape[0]
    # row_starts is the per-row start column. Its absence marks either a decode
    # shape (handled by the fold above) or the dummy-logits path, whose score
    # buffer is only `topk` columns wide and so cannot be indexed by the
    # (possibly much larger) row lengths.
    if row_starts is None or batch_idx_list is not None:
        return None
    if not (0 < topk <= 2048):
        return None
    # The indexer (DeepGEMM) emits fp32 scores with unit row stride and a
    # 16B-aligned row stride, which is exactly this kernel's ABI.
    if (
        logits.dtype != torch.float32
        or logits.stride(1) != 1
        or logits.stride(0) % 4 != 0
    ):
        return None
    if lengths.dtype != torch.int32 or row_starts.dtype != torch.int32:
        return None
    if lengths.shape[0] != num_rows or row_starts.shape[0] != num_rows:
        return None
    if not (lengths.is_contiguous() and row_starts.is_contiguous()):
        return None
    # The plan is preprocessed once per forward from dsa_seqlens_expanded, and it
    # carries a cluster_threshold plus a compacted work-list of the rows the
    # persistent cluster kernel will process -- a plan built from *different*
    # lengths makes the main kernel skip a row nobody processed and publish
    # uninitialized indices as KV slots. `lengths` can be overridden by the caller
    # (ke_offset), so require the exact tensor rather than merely a matching row
    # count. This is also what makes attn_metadata.max_seq_len_k a sound bound
    # below. The chunked-logits loop and the CP path both pass their own lengths and
    # so land on the legacy kernels.
    if lengths is not attn_metadata.dsa_seqlens_expanded:
        return None
    plan = attn_metadata.topk_v2_plan
    if plan is None or plan.shape[0] != num_rows + 1:
        return None
    # Kernel-selection bound. The score matrix is as wide as the batch's
    # concatenated KV, so its width over-estimates the longest row whenever the
    # batch holds more than one request -- which picks a needlessly wide register
    # template, and can even route a batch of short rows to the cluster path with
    # no work for it. max_seq_len_k is the host-side max over the batch's KV
    # lengths and dsa_seqlens_expanded's largest entry is exactly one request's KV
    # length, so for EXTEND the two are the same number: a tight, valid bound.
    #
    # Reject rather than clamp. A bound past the score width is a broken metadata
    # invariant, and min()-ing it would silently substitute a LOWER bound -- the one
    # direction that miscomputes the top-k (a too-small level cannot cover the
    # longest row). Falling back keeps the kernel's own range check reachable
    # instead of pre-empting it. Unreachable today: logits.shape[1] is the sum of
    # the batch's KV lengths and max_seq_len_k is their max.
    max_seq_len = attn_metadata.max_seq_len_k
    if max_seq_len <= 0 or max_seq_len > logits.shape[1]:
        return None

    kwargs = {}
    if topk_transform_method == TopkTransformMethod.PAGED:
        page_table = attn_metadata.real_page_table
        row_to_batch = attn_metadata.token_to_batch_idx
        if page_table is None or row_to_batch is None:
            return None
        if page_table.dtype != torch.int32 or row_to_batch.dtype != torch.int32:
            return None
        if page_table.stride(1) != 1 or not row_to_batch.is_contiguous():
            return None
        if row_to_batch.shape[0] != num_rows:
            return None
        kwargs = dict(
            page_size=attn_metadata.page_size,
            page_table=page_table,
            row_to_batch=row_to_batch,
        )
    elif topk_transform_method == TopkTransformMethod.RAGGED:
        if topk_indices_offset is None:
            return None
        if (
            topk_indices_offset.dtype != torch.int32
            or topk_indices_offset.shape[0] < num_rows
            or topk_indices_offset.stride(0) != 1
        ):
            return None
        kwargs = dict(out_offsets=topk_indices_offset[:num_rows])
    else:
        return None

    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_extend_v2

    # Write straight into the caller's buffer when it gave us one: under CUDA
    # graph the caller's buffer is the static top-k tensor a captured segment
    # reads, so this replaces a full-size copy (128 MB per indexer layer at a 16K
    # prefill) with nothing.
    if (
        out is not None
        and out.dtype == torch.int32
        and out.shape == (num_rows, topk)
        and out.is_contiguous()
    ):
        topk_transform_extend_v2(
            logits, lengths, row_starts, out, plan, max_seq_len, **kwargs
        )
        return out
    # Uninitialized, same output-completeness invariant as the paged helper below
    # -- skipping the -1 pre-fill avoids a full-size memset per indexer layer.
    fresh = logits.new_empty((num_rows, topk), dtype=torch.int32)
    topk_transform_extend_v2(
        logits, lengths, row_starts, fresh, plan, max_seq_len, **kwargs
    )
    return fresh


def _topk_transform_v2_paged(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    attn_metadata,
) -> torch.Tensor:
    """Fused top-k + page-table transform via the DeepSeek-V4 v2 JIT kernel.

    Returns the transformed page indices ``(num_rows, topk)`` int32 (physical
    page_size=1 KV slots, ``-1`` padded) -- identical in meaning to
    ``fast_topk_transform_fused`` / ``flashinfer.top_k_page_table_transform``.
    The kernel selects, per row, the top-k of ``logits[row, :lengths[row]]`` and
    maps each selected position ``p`` through the page table as
    ``real_page_table[row, p // page_size] * page_size + (p % page_size)``. Feeding
    it the indexer's compact ``real_page_table`` (page_size = pool page size,
    typically 64) yields the same physical slots as gathering the page_size=1
    table, without materializing that wide table.

    This is a committed contract, not a best-effort path: ``topk_transform`` routes
    here only for the decode-shaped PAGED case, and the fused-decode CUDA graph
    drops the page_size=1 table for exactly this case (see
    ``dsa_drop_wide_page_table``). The preconditions below are therefore
    invariants the caller must uphold -- they assert (raise) on violation rather
    than fall back to the slow legacy path (which may not even have a page_size=1
    table to fall back to) or silently paper over bad input (padding, recomputing
    the plan) at the cost of the performance this path exists to deliver.

    ``lengths`` entries must be NON-NEGATIVE: the kernel reads them as
    ``uint32_t``, so a negative row length (DP-padded / idle-companion rows)
    reinterprets as ~4e9 tokens and illegal-addresses. Metadata producers clamp
    padded rows to 0 (see ``fused_dsa_draft_extend_metadata`` /
    ``seqlens_expand_kernel``); 0 takes the trivial all-(-1) output path.
    """
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512_v2

    num_rows = logits.shape[0]

    # The indexer (DeepGEMM) emits fp32 scores with unit row stride and a 16B-aligned
    # row stride (a multiple of 4), which is exactly the kernel's ABI (it checks
    # score_stride % 4 == 0 with strides {S, 1}). This holds even though the scores
    # may be a padded view (stride(0) > width, so not `is_contiguous()`); assert the
    # real requirement rather than force a contiguous copy of the wide score buffer.
    assert (
        logits.dtype == torch.float32
        and logits.stride(1) == 1
        and logits.stride(0) % 4 == 0
    ), f"v2 top-k expects fp32 scores with unit row stride and 16B-aligned score_stride, got {logits.dtype=} {logits.stride()=}"
    assert 0 < topk <= 2048, f"v2 top-k supports 0 < topk <= 2048, got {topk=}"

    page_table = attn_metadata.real_page_table
    assert page_table.dtype == torch.int32
    lengths_i32 = lengths.to(torch.int32)

    # The plan is preprocessed once per forward (DSAMetadata.topk_v2_plan,
    # refreshed in-place under CUDA graph) and reused across layers. A missing or
    # mismatched plan means the caller skipped that preprocessing -- fail loudly
    # rather than silently recompute it per layer.
    plan = attn_metadata.topk_v2_plan
    assert (
        plan is not None and plan.shape[0] == num_rows + 1
    ), "topk_v2_plan must be preprocessed per forward (see DSAMetadata.topk_v2_plan)"

    page_size = attn_metadata.page_size
    # Uninitialized: the kernel covers every one of the `topk` slots of every row,
    # padding included (see the output-completeness invariant above
    # trivial_transform in topk_v2.cuh), so a -1 pre-fill is a full-size memset for
    # nothing. Pinned by test_topk_v2_writes_every_slot.
    out = logits.new_empty((num_rows, topk), dtype=torch.int32)
    topk_transform_512_v2(logits, lengths_i32, page_table, out, page_size, plan)
    return out


def _build_flashinfer_paged_args(
    attn_metadata,
    row_starts: Optional[torch.Tensor],
    cu_seqlens_q_topk: Optional[torch.Tensor],
    batch_idx_list: Optional[List[int]],
    device: torch.device,
    num_rows: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    row_to_batch = (
        torch.as_tensor(batch_idx_list, dtype=torch.int32, device=device)
        if batch_idx_list is not None
        else None
    )

    # Both dynamic mappings contain one entry per logit row. Supplying the known
    # size avoids synchronizing CUDA to infer the sum of the repeat counts.
    if (
        row_to_batch is not None
        and cu_seqlens_q_topk is not None
        and row_to_batch.shape[0] != num_rows
    ):
        q_lens = (cu_seqlens_q_topk[1:] - cu_seqlens_q_topk[:-1]).to(
            dtype=torch.int32, device=device
        )
        row_to_batch = torch.repeat_interleave(
            row_to_batch, q_lens, output_size=num_rows
        )

    if row_to_batch is None and cu_seqlens_q_topk is not None:
        # Decode-like case (one query row per batch) does not need an explicit mapping.
        # Avoid dynamic tensor construction in this branch to keep CUDA graph capture safe.
        num_batches = cu_seqlens_q_topk.shape[0] - 1
        if not (row_starts is None and num_rows == num_batches):
            q_lens = (cu_seqlens_q_topk[1:] - cu_seqlens_q_topk[:-1]).to(
                dtype=torch.int32, device=device
            )
            row_to_batch = torch.repeat_interleave(
                torch.arange(q_lens.shape[0], dtype=torch.int32, device=device),
                q_lens,
                output_size=num_rows,
            )

    if row_starts is not None and row_to_batch is None:
        raise RuntimeError(
            "PAGED topk_transform with row_starts requires cu_seqlens_q metadata."
        )

    page_table_row_starts = row_starts
    if page_table_row_starts is not None and row_to_batch is not None:
        page_table_row_starts = (
            page_table_row_starts - attn_metadata.cu_seqlens_k[:-1][row_to_batch]
        )

    return row_to_batch, page_table_row_starts


def _flashinfer_tie_break_value() -> int:
    mode = envs.SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK.get()
    if mode is None:
        return 0
    mode = mode.lower()
    if mode not in _FLASHINFER_TIE_BREAK_VALUES:
        raise RuntimeError(
            "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK must be one of "
            f"{tuple(_FLASHINFER_TIE_BREAK_VALUES)} or unset, got {mode!r}."
        )
    return _FLASHINFER_TIE_BREAK_VALUES[mode]
