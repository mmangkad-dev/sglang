"""Host-dispatch tests for the extend-phase top-k v2 fast path.

``test_topk_v2.py`` covers the kernel; this file covers the decision in front of
it -- ``DSATopKBackend.topk_transform`` -> ``_topk_transform_v2_extend`` -- which is
where the guards live that stand between a stale plan and garbage KV indices.

Two properties are checked for every case:

* **Equivalence.** v2's output must equal the legacy AOT kernel it replaces
  (``fast_topk_transform_fused`` for PAGED, ``fast_topk_transform_ragged_fused``
  for RAGGED) on the same inputs -- not merely "a plausible top-k". Score rows are
  built so no two entries in a row's window tie, which makes the top-k unique and
  the comparison exact.
* **Which kernel ran.** Each case asserts whether the v2 entry point was reached,
  so a guard that stops working shows up as a test failure rather than as a silent
  change of kernel. Every fallback case is also checked for correctness, since
  falling back must stay correct as well as slow.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import topk as topk_ops
from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE_SIZE = 64
TOPK = 2048
DEVICE = "cuda"


def _build_batch(kv_lens, extend_lens, seed=0):
    """One extend batch's worth of indexer state, laid out the way
    ``_cal_indexer_k_start_end`` and the extend metadata build produce it.

    Returns the tensors ``topk_transform`` takes plus a synthetic ``attn_metadata``
    exposing only the fields the v2 dispatch and the legacy kernels read.
    """
    torch.manual_seed(seed)
    starts = [sum(kv_lens[:r]) for r in range(len(kv_lens))]
    lengths, row_starts, row_to_batch = [], [], []
    for r, (kv, e) in enumerate(zip(kv_lens, extend_lens)):
        lengths += list(range(kv - e + 1, kv + 1))
        row_starts += [starts[r]] * e
        row_to_batch += [r] * e
    num_rows, kv_total = len(lengths), sum(kv_lens)

    # deep_gemm pads the logits row stride to a multiple of 256 floats; mimic that,
    # so `scores` is a padded non-contiguous view with a 16B-aligned row stride --
    # exactly what the indexer hands the transform, and what the v2 alignment guard
    # keys on. Build and perturb the PADDED buffer, then slice: adding to the slice
    # would materialize a fresh contiguous tensor and silently drop the padding
    # (which is how the first draft of this helper made the guard reject every case
    # whose kv_total happened not to be a multiple of 4).
    stride = (kv_total + 255) // 256 * 256
    padded = torch.randn(num_rows, stride, dtype=torch.float32, device=DEVICE)
    # Break ties: a unique per-column offset makes each row's top-k unique, so v2
    # and the legacy kernel must agree exactly rather than up to a tie swap.
    padded += torch.arange(stride, dtype=torch.float32, device=DEVICE) * (
        1.0 / (8.0 * stride)
    )
    scores = padded[:, :kv_total]
    assert scores.stride(0) % 4 == 0
    # Non-contiguous whenever the padding is non-empty (kv_total not already a
    # multiple of 256) -- the case the v2 guard has to accept on stride alone.
    assert scores.is_contiguous() == (stride == kv_total)

    lengths_t = torch.tensor(lengths, dtype=torch.int32, device=DEVICE)
    row_starts_t = torch.tensor(row_starts, dtype=torch.int32, device=DEVICE)
    row_to_batch_t = torch.tensor(row_to_batch, dtype=torch.int32, device=DEVICE)

    # A permuted page table with page-contiguous slots, so real_page_table is its
    # strided decimation exactly as _transform_table_1_to_real produces.
    max_kv = max(kv_lens)
    num_pages = (max_kv + PAGE_SIZE - 1) // PAGE_SIZE
    real_pt = (
        torch.randperm(num_pages * len(kv_lens), device=DEVICE, dtype=torch.int32)
        % 30000
    ).reshape(len(kv_lens), num_pages)
    page_table_1 = (
        (
            real_pt[:, :, None].to(torch.int64) * PAGE_SIZE
            + torch.arange(PAGE_SIZE, device=DEVICE)
        )
        .reshape(len(kv_lens), -1)[:, :max_kv]
        .to(torch.int32)
    )

    cu_seqlens_q = torch.tensor(
        [0] + [sum(extend_lens[: r + 1]) for r in range(len(extend_lens))],
        dtype=torch.int32,
        device=DEVICE,
    )

    attn_metadata = SimpleNamespace(
        topk_v2_plan=plan_topk_v2(lengths_t),
        dsa_seqlens_expanded=lengths_t,
        max_seq_len_k=max_kv,
        real_page_table=real_pt.contiguous(),
        token_to_batch_idx=row_to_batch_t,
        page_table_1=page_table_1.contiguous(),
        page_size=PAGE_SIZE,
        cu_seqlens_q=cu_seqlens_q,
        topk_indices_offset=row_starts_t,  # == ks for the RAGGED extend contract
    )
    return scores, lengths_t, row_starts_t, cu_seqlens_q, attn_metadata


def _run(method, scores, lengths, row_starts, cu_seqlens_q, attn_metadata, **kw):
    """Call the backend and report whether the v2 extend entry point was reached."""
    calls = []
    real = topk_ops.topk_transform_extend_v2

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    topk_ops.topk_transform_extend_v2 = spy
    try:
        out = DSATopKBackend.SGL_KERNEL.topk_transform(
            logits=scores,
            lengths=lengths,
            topk=TOPK,
            topk_transform_method=method,
            attn_metadata=attn_metadata,
            cu_seqlens_q_topk=cu_seqlens_q,
            topk_indices_offset=attn_metadata.topk_indices_offset,
            row_starts=row_starts,
            **kw,
        )
    finally:
        topk_ops.topk_transform_extend_v2 = real
    return out, bool(calls)


def _legacy_paged(scores, lengths, row_starts, cu_seqlens_q, attn_metadata):
    from sgl_kernel import fast_topk_transform_fused

    return fast_topk_transform_fused(
        score=scores,
        lengths=lengths,
        page_table_size_1=attn_metadata.page_table_1,
        cu_seqlens_q=cu_seqlens_q,
        topk=TOPK,
        row_starts=row_starts,
    )


def _legacy_ragged(scores, lengths, row_starts, attn_metadata):
    from sgl_kernel import fast_topk_transform_ragged_fused

    return fast_topk_transform_ragged_fused(
        score=scores,
        lengths=lengths,
        topk_indices_offset=attn_metadata.topk_indices_offset,
        topk=TOPK,
        row_starts=row_starts,
    )


def _assert_same(got, want):
    """Compare as per-row sets: both kernels emit the same selection but are not
    required to emit it in the same slot order."""
    g, w = got.cpu().tolist(), want.cpu().tolist()
    assert len(g) == len(w)
    for i, (gr, wr) in enumerate(zip(g, w)):
        gs = sorted(v for v in gr if v >= 0)
        ws = sorted(v for v in wr if v >= 0)
        assert (
            gs == ws
        ), f"row {i}: {len(gs)} vs {len(ws)} valid, first diff around {[x for x in gs[:8]]} vs {[x for x in ws[:8]]}"
        assert gr.count(-1) == wr.count(-1), f"row {i}: padding count differs"


# Batches whose row starts are deliberately not multiples of 4, and whose
# concatenated width exceeds the longest row -- the real prefill shape.
BATCHES = [
    pytest.param([4096], [4096], id="single-request"),
    pytest.param([3001, 2503], [512, 512], id="two-unaligned"),
    pytest.param([1000, 5003, 9001], [256, 256, 256], id="three-mixed"),
    pytest.param([4099] * 5, [64] * 5, id="wide-width-short-rows"),
]


@pytest.mark.parametrize("kv_lens,extend_lens", BATCHES)
@torch.inference_mode()
def test_extend_paged_matches_legacy(kv_lens, extend_lens) -> None:
    """PAGED extend: v2 (compact page-size-64 table + row_to_batch) must select the
    same physical slots as the legacy page-size-1 gather."""
    args = _build_batch(kv_lens, extend_lens, seed=len(kv_lens))
    out, used_v2 = _run(TopkTransformMethod.PAGED, *args)
    assert used_v2, "expected the v2 extend path"
    _assert_same(out, _legacy_paged(*args))


@pytest.mark.parametrize("kv_lens,extend_lens", BATCHES)
@torch.inference_mode()
def test_extend_ragged_matches_legacy(kv_lens, extend_lens) -> None:
    """RAGGED extend: v2's position + out_offset must equal the legacy kernel's."""
    args = _build_batch(kv_lens, extend_lens, seed=7 + len(kv_lens))
    out, used_v2 = _run(TopkTransformMethod.RAGGED, *args)
    assert used_v2, "expected the v2 extend path"
    _assert_same(out, _legacy_ragged(args[0], args[1], args[2], args[4]))


@torch.inference_mode()
def test_extend_writes_into_caller_buffer() -> None:
    """With `out` supplied the result must BE that buffer, not a copy of it -- the
    indexer skips its own copy by comparing data_ptr()."""
    scores, lengths, row_starts, cu_q, meta = _build_batch([3001, 2503], [512, 512])
    num_rows = scores.shape[0]
    # Pad exactly like the graph's static top-k buffer, then hand over a prefix view.
    static = torch.full((num_rows + 96, TOPK), -1, dtype=torch.int32, device=DEVICE)
    out, used_v2 = _run(
        TopkTransformMethod.PAGED,
        scores,
        lengths,
        row_starts,
        cu_q,
        meta,
        out=static[:num_rows],
    )
    assert used_v2
    assert out.data_ptr() == static.data_ptr(), "v2 did not write into `out`"
    _assert_same(out, _legacy_paged(scores, lengths, row_starts, cu_q, meta))
    # The rows past the handed-over prefix must be untouched.
    assert bool((static[num_rows:] == -1).all())


@torch.inference_mode()
def test_extend_ignores_unusable_out_buffer() -> None:
    """A caller buffer that is not an int32 (num_rows, topk) contiguous tensor must
    be ignored, not written through -- the helper allocates its own instead."""
    scores, lengths, row_starts, cu_q, meta = _build_batch([3001, 2503], [512, 512])
    wrong = torch.zeros(scores.shape[0], TOPK, dtype=torch.int64, device=DEVICE)
    out, used_v2 = _run(
        TopkTransformMethod.PAGED, scores, lengths, row_starts, cu_q, meta, out=wrong
    )
    assert used_v2
    assert out.data_ptr() != wrong.data_ptr()
    assert bool((wrong == 0).all()), "unusable out buffer was written"
    _assert_same(out, _legacy_paged(scores, lengths, row_starts, cu_q, meta))


@torch.inference_mode()
def test_extend_falls_back_on_row_count_mismatched_plan() -> None:
    """A plan whose row count does not match must not be used with these lengths:
    it carries a cluster work-list keyed to the rows it was built from."""
    scores, lengths, row_starts, cu_q, meta = _build_batch([1000, 5003], [256, 256])
    meta.topk_v2_plan = plan_topk_v2(lengths[:-4])
    out, used_v2 = _run(
        TopkTransformMethod.PAGED, scores, lengths, row_starts, cu_q, meta
    )
    assert not used_v2, "a mismatched plan must fall back"
    _assert_same(out, _legacy_paged(scores, lengths, row_starts, cu_q, meta))


@torch.inference_mode()
def test_extend_falls_back_when_lengths_are_not_the_planned_tensor() -> None:
    """Same row count but a different lengths tensor (what an ke_offset override
    produces) must fall back: the plan and the bound both come from
    dsa_seqlens_expanded, so only the identical tensor is safe."""
    scores, lengths, row_starts, cu_q, meta = _build_batch([1000, 5003], [256, 256])
    other = lengths.clone()  # equal values, different tensor
    out, used_v2 = _run(
        TopkTransformMethod.PAGED, scores, other, row_starts, cu_q, meta
    )
    assert not used_v2, "non-identical lengths must fall back"
    _assert_same(out, _legacy_paged(scores, other, row_starts, cu_q, meta))


@pytest.mark.parametrize("bad_max_seq_len_k", [0, -1, 10**9])
@torch.inference_mode()
def test_extend_falls_back_on_out_of_range_metadata_bound(bad_max_seq_len_k) -> None:
    """max_seq_len_k out of range means broken metadata, so fall back rather than
    clamp into it.

    Clamping with min(max_seq_len_k, logits.shape[1]) would look harmless but
    substitutes a LOWER bound whenever max_seq_len_k is too large -- the one
    direction that miscomputes the top-k, since a level whose fixed-unrolled loop is
    too short cannot cover the longest row. Rejecting also leaves the kernel's own
    range check reachable. Guards against re-introducing the clamp.
    """
    scores, lengths, row_starts, cu_q, meta = _build_batch([1000, 5003], [256, 256])
    meta.max_seq_len_k = bad_max_seq_len_k
    out, used_v2 = _run(
        TopkTransformMethod.PAGED, scores, lengths, row_starts, cu_q, meta
    )
    assert not used_v2
    _assert_same(out, _legacy_paged(scores, lengths, row_starts, cu_q, meta))


@torch.inference_mode()
def test_extend_falls_back_without_row_starts() -> None:
    """No row starts marks the dummy-logits path, whose score buffer is only `topk`
    columns wide and cannot be indexed by the row lengths."""
    scores, lengths, _, cu_q, meta = _build_batch([1000, 5003], [256, 256])
    out, used_v2 = _run(TopkTransformMethod.PAGED, scores, lengths, None, cu_q, meta)
    assert not used_v2
    _assert_same(out, _legacy_paged(scores, lengths, None, cu_q, meta))


@torch.inference_mode()
def test_extend_falls_back_with_batch_idx_list() -> None:
    """batch_idx_list means the page table was already row-selected for a CP shard;
    v2's row_to_batch indexes the unselected table, so it must not run."""
    scores, lengths, row_starts, cu_q, meta = _build_batch([1000, 5003], [256, 256])
    _, used_v2 = _run(
        TopkTransformMethod.PAGED,
        scores,
        lengths,
        row_starts,
        cu_q,
        meta,
        batch_idx_list=[0, 1],
    )
    assert not used_v2


@torch.inference_mode()
def test_extend_falls_back_on_unaligned_score_stride() -> None:
    """The vectorized loads need a 16B-aligned row base; deep_gemm always pads to a
    multiple of 256 floats, but the guard must hold if that ever changes."""
    scores, lengths, row_starts, cu_q, meta = _build_batch([1000, 5003], [256, 256])
    kv_total = scores.shape[1]
    width = kv_total + 1
    while width % 4 == 0:
        width += 1
    odd = torch.randn(scores.shape[0], width, dtype=torch.float32, device=DEVICE)[
        :, :kv_total
    ]
    assert odd.stride(0) % 4 != 0
    out, used_v2 = _run(TopkTransformMethod.PAGED, odd, lengths, row_starts, cu_q, meta)
    assert not used_v2
    _assert_same(out, _legacy_paged(odd, lengths, row_starts, cu_q, meta))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
