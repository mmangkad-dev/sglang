"""Correctness tests for the DeepSeek-V4 (DSA indexer) JIT top-k transform v2.

The v2 kernel selects the per-row top-k of ``scores`` (ragged ``seq_lens``) and
writes the page-table transform of the selected raw indices into the output. We
validate against ``torch.topk`` with a small tolerance for boundary ties (the
fp16 coarse histogram can swap elements of equal score).

Coverage is organized around the kernel's dispatch so every template and its
boundaries are exercised:

  template      per-row seq            reached when
  --------      ----------             ------------
  trivial       seq <= k
  Register2     k < seq <= 8192        max_seq <= 8192          (level 0)
  Register4     8192 < seq <= 16384    max_seq <= 16384         (level 1)
  Streaming     16384 < seq <= floor   max_seq > 16384, non-cluster (level 2)
  Cluster       seq > floor(=65536)    max_seq > floor and batch <= 128

and two cluster dispatch shapes: the fused small-batch kernel (batch <= 30) and
the persistent-pool + main kernel (30 < batch <= 128). Boundary seq lengths
(8192/8193, 16384/16385, 65535/65536/65537) and batch sizes (30/31, 128/129) are
included explicitly, across k in {512,1024,2048} and identity/perm page tables.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2,
    topk_transform_512_v2,
    topk_transform_extend_v2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE_SIZE = 64  # c4 page size = 256 // 4
PAGE_BITS = PAGE_SIZE.bit_length() - 1
PAGE_MASK = PAGE_SIZE - 1
MAX_PERMIT_ERROR = 5
FLOOR = 65536  # kClusterFloor

# (batch, seq) chosen to land on each template and each dispatch boundary.
FIXED_CONFIGS = [
    # --- trivial (seq <= k) ---
    (8, 256),  # trivial for every k
    (16, 1024),  # trivial for k>=1024
    # --- Register2 (level 0: max_seq <= 8192) ---
    (8, 4096),
    (8, 8192),  # reg2 upper boundary
    (128, 8192),
    (300, 8192),  # batch > 128, still level 0
    # --- Register4 (level 1: 8192 < max_seq <= 16384) ---
    (8, 8193),  # just over reg2
    (64, 16384),  # reg4 upper boundary
    (256, 16384),  # batch > 128
    # --- Streaming (level 2: max_seq > 16384, non-cluster) ---
    (8, 16385),  # just over reg4 (small batch, seq < floor => non-cluster)
    (4, 32768),
    (16, 65535),  # just under floor
    (4, 65536),  # at floor (seq == floor => non-cluster)
    (100, 65536),
    # --- Cluster, fused small-batch kernel (batch <= 30, max_seq > floor) ---
    (1, 65537),  # single row just over floor
    (2, 131072),
    (8, 98304),
    (30, 131072),  # batch == pool boundary
    # --- Cluster, persistent pool + main kernel (30 < batch <= 128) ---
    (31, 131072),  # just over small-batch
    (40, 262144),  # N > pool of 30 => round-robin
    (64, 196608),
    (128, 131072),  # cluster batch upper boundary
    # --- batch > 128 => non-cluster streaming even at long ctx ---
    (129, 131072),
    (200, 262144),
]


def _assert_topk_close(scores_cpu, ref_raw, our_raw, bs, seq_lens, k):
    """Set-compare our top-k raw indices vs torch's, tolerating equal-score ties."""
    bad = 0
    for i in range(bs):
        L = int(seq_lens[i])
        ref, our = set(ref_raw[i]), set(our_raw[i])
        more, less = our - ref, ref - our
        if more or less:
            mv = sorted(scores_cpu[i, list(more)].tolist())
            lv = sorted(scores_cpu[i, list(less)].tolist())
            if mv != lv:  # not merely a tie swap -> genuine error
                bad += len(more)
                print(
                    f"b={i} L={L} k={k}: more={list(more)[:4]} less={list(less)[:4]} mv={mv[:3]} lv={lv[:3]}"
                )
        assert len(our) == min(
            k, L
        ), f"b={i} L={L} k={k}: {len(our)} valid != {min(k, L)}"
    assert bad <= MAX_PERMIT_ERROR, f"{bad=} > {MAX_PERMIT_ERROR}"


def _make_page_table(batch, num_pages, mode, device, per_row=False):
    if mode == "identity":
        pt = torch.arange(num_pages, dtype=torch.int32, device=device)
        full = pt.unsqueeze(0).expand(batch, -1).contiguous()
        inv = pt.unsqueeze(0).expand(batch, -1).cpu()
        return full, inv
    # permutation (optionally a distinct permutation per row)
    rows = batch if per_row else 1
    full = torch.stack(
        [torch.randperm(num_pages, device=device) for _ in range(rows)]
    ).to(torch.int32)
    inv = torch.empty_like(full)
    ar = torch.arange(num_pages, dtype=torch.int32, device=device)
    for r in range(rows):
        inv[r, full[r].long()] = ar
    if not per_row:
        full = full.expand(batch, -1).contiguous()
        inv = inv.expand(batch, -1)
    return full, inv.cpu()


def _invert(out_row, inv_row):
    """Undo page_to_indices for one row's page indices (drop -1 padding)."""
    return [
        (int(inv_row[v >> PAGE_BITS]) << PAGE_BITS) | (v & PAGE_MASK)
        for v in out_row
        if v != -1
    ]


def _reference(scores, seq_lens, k):
    """torch.topk reference indices per row (trivial rows -> all positions)."""
    ref = []
    for i in range(scores.shape[0]):
        L = int(seq_lens[i])
        if L <= k:
            ref.append(list(range(L)))
        else:
            ref.append(
                torch.topk(scores[i, :L], k, sorted=False).indices.cpu().tolist()
            )
    return ref


def _run(scores, seq_lens, page_table, inv_cpu, k):
    batch = scores.shape[0]
    out = torch.full((batch, k), -1, dtype=torch.int32, device=scores.device)
    metadata = plan_topk_v2(seq_lens)
    topk_transform_512_v2(scores, seq_lens, page_table, out, PAGE_SIZE, metadata)
    torch.cuda.synchronize()
    out_cpu = out.cpu().tolist()
    return [_invert(out_cpu[i], inv_cpu[i]) for i in range(batch)]


def _run_raw(scores, seq_lens, page_table, k):
    """Run the kernel and return its optional raw (pre-transform) top-k index
    output per row, dropping -1 padding -- the selected positions themselves,
    NOT the page-table transform of them."""
    batch = scores.shape[0]
    out = torch.full((batch, k), -1, dtype=torch.int32, device=scores.device)
    raw = torch.full((batch, k), -1, dtype=torch.int32, device=scores.device)
    metadata = plan_topk_v2(seq_lens)
    topk_transform_512_v2(scores, seq_lens, page_table, out, PAGE_SIZE, metadata, raw)
    torch.cuda.synchronize()
    raw_cpu = raw.cpu().tolist()
    return [[v for v in raw_cpu[i] if v != -1] for i in range(batch)]


@pytest.mark.parametrize("page_mode", ["identity", "perm"])
@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("batch,seq", FIXED_CONFIGS)
@torch.inference_mode()
def test_topk_v2(batch: int, seq: int, k: int, page_mode: str) -> None:
    torch.manual_seed(batch * 100003 + seq * 7 + k)
    device = "cuda"
    # Pad the row stride to a multiple of 4 (16-byte vectorized load) while keeping
    # the exact seq_len -- this also exercises the scalar-tail path for odd seq.
    width = (seq + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)[:, :seq]
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(batch, num_pages, page_mode, device)

    our_raw = _run(scores, seq_lens, page_table, inv_cpu, k)
    ref_raw = _reference(scores, seq_lens, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, batch, seq_lens.cpu(), k)


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize(
    "batch,shape",
    [
        (20, "small_batch"),  # fused small-batch kernel (<= pool of 30)
        (64, "persistent"),  # persistent pool + main kernel
        (128, "persistent"),  # cluster batch boundary
    ],
)
@pytest.mark.parametrize("per_row_pt", [False, True])
@torch.inference_mode()
def test_topk_v2_ragged(batch: int, shape: str, k: int, per_row_pt: bool) -> None:
    """Ragged lengths spanning trivial..cluster in one launch, both dispatch shapes.

    ``per_row_pt`` gives each row a distinct page-table permutation, exercising
    the per-batch page_table indexing (batch_id stride) rather than a shared one.
    """
    torch.manual_seed(7777 + batch + k + int(per_row_pt))
    device = "cuda"
    seq = 262144
    scores = torch.randn(batch, seq, dtype=torch.float32, device=device)
    # span every path; guarantee at least one > floor row so cluster dispatch fires
    buckets = [max(1, k // 2), k, 4096, 12000, 40000, 65536, 98304, 262144]
    g = torch.Generator(device="cpu").manual_seed(batch + k)
    lengths = torch.tensor(
        [
            buckets[int(torch.randint(0, len(buckets), (1,), generator=g))]
            for _ in range(batch)
        ],
        dtype=torch.int32,
        device=device,
    )
    lengths[0] = max(1, k // 2)  # a trivial row
    lengths[1] = 262144  # a long (cluster) row
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(
        batch, num_pages, "perm", device, per_row=per_row_pt
    )

    our_raw = _run(scores, lengths, page_table, inv_cpu, k)
    ref_raw = _reference(scores, lengths, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, batch, lengths.cpu(), k)


@pytest.mark.parametrize("page_mode", ["identity", "perm"])
@pytest.mark.parametrize(
    "batch,seq",
    [
        (8, 256),  # trivial
        (8, 4096),  # register
        (4, 131072),  # fused small-batch cluster
        (64, 131072),  # persistent cluster + main<3> epilogue
        (256, 131072),  # non-cluster streaming
    ],
)
@torch.inference_mode()
def test_topk_v2_raw_indices(batch: int, seq: int, page_mode: str) -> None:
    """The optional raw-index output must be the pre-transform position of each
    transformed output slot (out[j] == page_to_indices(raw[j])), and -1 aligns."""
    k = 512
    torch.manual_seed(batch * 131 + seq)
    device = "cuda"
    width = (seq + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)[:, :seq]
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(batch, num_pages, page_mode, device)
    out = torch.full((batch, k), -1, dtype=torch.int32, device=device)
    raw = torch.full((batch, k), -1, dtype=torch.int32, device=device)

    metadata = plan_topk_v2(seq_lens)
    topk_transform_512_v2(scores, seq_lens, page_table, out, PAGE_SIZE, metadata, raw)
    torch.cuda.synchronize()

    out_cpu, raw_cpu = out.cpu().tolist(), raw.cpu().tolist()
    for i in range(batch):
        for j in range(k):
            o, r = out_cpu[i][j], raw_cpu[i][j]
            if o == -1:
                assert r == -1, f"b={i} j={j}: out=-1 but raw={r}"
            else:
                inv = (int(inv_cpu[i][o >> PAGE_BITS]) << PAGE_BITS) | (o & PAGE_MASK)
                assert r == inv, f"b={i} j={j}: raw={r} != inverse(out)={inv}"


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("batch,seq", FIXED_CONFIGS)
@torch.inference_mode()
def test_topk_v2_output_indices(batch: int, seq: int, k: int) -> None:
    """Validate the raw (pre-transform) index output DIRECTLY against torch.topk.

    Unlike ``test_topk_v2`` -- which checks the page-transformed output and inverts
    it through the page table -- this exercises the selected indices themselves, so
    it isolates the top-k selection from the page-table transform. A permuted page
    table is used so raw != out, catching any bug that leaks transformed page
    indices into the raw buffer. Covers every dispatch template/boundary.
    """
    torch.manual_seed(batch * 100003 + seq * 7 + k + 1)
    device = "cuda"
    width = (seq + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)[:, :seq]
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, _ = _make_page_table(batch, num_pages, "perm", device)

    our_raw = _run_raw(scores, seq_lens, page_table, k)
    ref_raw = _reference(scores, seq_lens, k)
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, batch, seq_lens.cpu(), k)


SENTINEL = -12345  # not producible by the kernel: outputs are >= 0 or exactly -1


@pytest.mark.parametrize(
    "batch,seq",
    [
        (8, 256),  # trivial (seq <= k)
        (8, 4096),  # Register2
        (64, 16384),  # Register4
        (256, 40000),  # Streaming
        (4, 131072),  # fused small-batch cluster
        (64, 131072),  # persistent cluster + main<3> epilogue
    ],
)
@torch.inference_mode()
def test_topk_v2_writes_every_slot(batch: int, seq: int) -> None:
    """Both callers allocate the output uninitialized, so every slot of every row
    must be written -- padding slots too. Pre-fill a sentinel the kernel cannot
    produce and assert none survives, on every template.

    Without this, a template that left padding slots untouched would hand
    downstream sparse attention an arbitrary int32 as a KV slot instead of a -1 it
    knows to mask.
    """
    k = 2048
    torch.manual_seed(batch * 31 + seq)
    device = "cuda"
    width = (seq + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)[:, :seq]
    # Mix in short rows so the trivial path and its padding are exercised too.
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    seq_lens[0] = 0
    seq_lens[1 % batch] = min(seq, k // 2)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, _ = _make_page_table(batch, num_pages, "perm", device)
    plan = plan_topk_v2(seq_lens)

    paged = torch.full((batch, k), SENTINEL, dtype=torch.int32, device=device)
    topk_transform_512_v2(scores, seq_lens, page_table, paged, PAGE_SIZE, plan)
    row_starts = torch.zeros(batch, dtype=torch.int32, device=device)
    ragged = torch.full((batch, k), SENTINEL, dtype=torch.int32, device=device)
    topk_transform_extend_v2(
        scores, seq_lens, row_starts, ragged, plan, seq, out_offsets=row_starts
    )
    torch.cuda.synchronize()

    for tag, buf in (("paged", paged), ("ragged", ragged)):
        left = (buf == SENTINEL).sum().item()
        assert left == 0, f"{tag}: {left} of {batch * k} slots never written"


# --- extend entry point (transform_extend) ---------------------------------
#
# Extend differs from decode in three ways the kernel has to handle: each row's
# score window starts at an arbitrary column (``row_starts``, so the window base
# can sit off the 16-byte boundary the vectorized loads need), many query rows map
# to one request's page table (``row_to_batch``), and the output transform may be
# a ragged offset add instead of a page-table gather.
#
# Row starts are the cumulative KV lengths of the preceding requests in the batch,
# so the deliberately non-multiple-of-4 lengths below are what a real batch looks
# like -- they exercise the unaligned-head path on every template.


def _extend_batch(kv_lens, extend_lens):
    """Build one extend batch: per-row window length / start / request index.

    Row i of request r covers ``scores[i, start_r : start_r + len]`` where the
    lengths ramp up to the request's KV length, exactly as the indexer's
    ``seqlens_expanded`` / ``indexer_k_start_end`` do.
    """
    starts = [sum(kv_lens[:r]) for r in range(len(kv_lens))]
    lengths, row_starts, row_to_batch = [], [], []
    for r, (kv, e) in enumerate(zip(kv_lens, extend_lens)):
        lengths += list(range(kv - e + 1, kv + 1))
        row_starts += [starts[r]] * e
        row_to_batch += [r] * e
    return lengths, row_starts, row_to_batch, sum(kv_lens)


def _window_view(scores_cpu, lengths, row_starts, batch):
    """Re-base every row on its window start so column j means window position j.

    ``_assert_topk_close`` looks up scores by the indices under test, which are
    window-relative; padding short rows with -inf keeps the stack rectangular
    without making a padded slot selectable.
    """
    widest = max(lengths)
    return torch.stack(
        [
            torch.nn.functional.pad(
                scores_cpu[i, row_starts[i] : row_starts[i] + lengths[i]],
                (0, widest - lengths[i]),
                value=float("-inf"),
            )
            for i in range(batch)
        ]
    )


def _extend_reference(scores_cpu, lengths, row_starts, k):
    ref = []
    for i, (L, st) in enumerate(zip(lengths, row_starts)):
        window = scores_cpu[i, st : st + L]
        if L <= k:
            ref.append(list(range(L)))
        else:
            ref.append(torch.topk(window, k, sorted=False).indices.tolist())
    return ref


EXTEND_CONFIGS = [
    # (kv_lens, extend_lens) -- id names the template the longest row reaches.
    pytest.param([4096], [4096], id="reg2-single-aligned"),
    pytest.param([3001, 2503], [512, 512], id="reg2-two-unaligned"),
    pytest.param([1000, 5003, 9001], [256, 256, 256], id="reg4-three-unaligned"),
    pytest.param([900, 1700], [256, 256], id="trivial-two-unaligned"),
    pytest.param([1234, 40001], [64, 64], id="streaming-unaligned"),
    pytest.param([999, 100003], [16, 16], id="cluster-unaligned"),
    pytest.param([16384], [2048], id="reg4-chunked-single"),
    # Width >> longest row, the shape a multi-request prefill batch actually has.
    # Verified perf-only, not a guard: passing scores.shape[1] as the bound instead
    # of max(lengths) is still correct here (0/512 rows differ), just slower -- a
    # too-HIGH bound only over-selects the template. These cases are coverage for
    # running Register2 against a Streaming/Cluster-width matrix; what keeps the
    # bound itself sound is the caller-side contract in
    # _topk_transform_v2_extend (see test_dsa_topk_backend_extend.py).
    pytest.param([4096] * 8, [64] * 8, id="wide-width-short-rows"),
    pytest.param([4099] * 17, [16] * 17, id="wide-width-short-rows-unaligned"),
]


@pytest.mark.parametrize("k", [1024, 2048])
@pytest.mark.parametrize("kv_lens,extend_lens", EXTEND_CONFIGS)
@torch.inference_mode()
def test_topk_v2_extend_paged(kv_lens, extend_lens, k: int) -> None:
    """Paged extend: emitted slots must equal the page-size-1 gather of the
    selected window positions through row ``row_to_batch[i]`` of the page table."""
    torch.manual_seed(sum(kv_lens) * 31 + k)
    device = "cuda"
    lengths, row_starts, row_to_batch, kv_total = _extend_batch(kv_lens, extend_lens)
    batch = len(lengths)
    width = (kv_total + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)[:, :kv_total]
    lengths_t = torch.tensor(lengths, dtype=torch.int32, device=device)
    row_starts_t = torch.tensor(row_starts, dtype=torch.int32, device=device)
    row_to_batch_t = torch.tensor(row_to_batch, dtype=torch.int32, device=device)

    num_pages = (max(kv_lens) + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(
        len(kv_lens), num_pages, "perm", device, per_row=True
    )
    out = torch.full((batch, k), -1, dtype=torch.int32, device=device)
    topk_transform_extend_v2(
        scores,
        lengths_t,
        row_starts_t,
        out,
        plan_topk_v2(lengths_t),
        max(lengths),
        page_size=PAGE_SIZE,
        page_table=page_table,
        row_to_batch=row_to_batch_t,
    )
    torch.cuda.synchronize()

    out_cpu = out.cpu().tolist()
    our_raw = [_invert(out_cpu[i], inv_cpu[row_to_batch[i]]) for i in range(batch)]
    scores_cpu = scores.cpu()
    ref_raw = _extend_reference(scores_cpu, lengths, row_starts, k)
    # Compare in window coordinates: shift the score row so column 0 is the
    # window start, matching the raw indices both sides produce.
    shifted = _window_view(scores_cpu, lengths, row_starts, batch)
    _assert_topk_close(shifted, ref_raw, our_raw, batch, lengths, k)


@pytest.mark.parametrize("k", [1024, 2048])
@pytest.mark.parametrize("kv_lens,extend_lens", EXTEND_CONFIGS)
@torch.inference_mode()
def test_topk_v2_extend_ragged(kv_lens, extend_lens, k: int) -> None:
    """Ragged extend: emitted indices must be ``window position + out_offset``.

    The indexer passes the row's own KV start as the offset, so the emitted value
    is the absolute column of the selected score -- which is what makes this
    checkable without a page table.
    """
    torch.manual_seed(sum(kv_lens) * 17 + k)
    device = "cuda"
    lengths, row_starts, _, kv_total = _extend_batch(kv_lens, extend_lens)
    batch = len(lengths)
    width = (kv_total + 3) & ~3
    scores = torch.randn(batch, width, dtype=torch.float32, device=device)[:, :kv_total]
    lengths_t = torch.tensor(lengths, dtype=torch.int32, device=device)
    row_starts_t = torch.tensor(row_starts, dtype=torch.int32, device=device)

    out = torch.full((batch, k), -1, dtype=torch.int32, device=device)
    topk_transform_extend_v2(
        scores,
        lengths_t,
        row_starts_t,
        out,
        plan_topk_v2(lengths_t),
        max(lengths),
        out_offsets=row_starts_t,
    )
    torch.cuda.synchronize()

    out_cpu = out.cpu().tolist()
    our_raw = [[v - row_starts[i] for v in out_cpu[i] if v != -1] for i in range(batch)]
    scores_cpu = scores.cpu()
    ref_raw = _extend_reference(scores_cpu, lengths, row_starts, k)
    shifted = _window_view(scores_cpu, lengths, row_starts, batch)
    _assert_topk_close(shifted, ref_raw, our_raw, batch, lengths, k)


@pytest.mark.parametrize("bad_bound", [0, 4097])
@torch.inference_mode()
def test_topk_v2_extend_rejects_out_of_range_bound(bad_bound: int) -> None:
    """``max_seq_len`` selects the template, so a bound of 0 (level 0 for any row)
    or one past the score width (a read out of bounds) must be refused, not
    silently clamped."""
    device = "cuda"
    batch, kv, k = 4, 4096, 2048
    scores = torch.randn(batch, kv, dtype=torch.float32, device=device)
    lengths = torch.full((batch,), kv, dtype=torch.int32, device=device)
    row_starts = torch.zeros(batch, dtype=torch.int32, device=device)
    out = torch.empty(batch, k, dtype=torch.int32, device=device)
    with pytest.raises(Exception):
        topk_transform_extend_v2(
            scores,
            lengths,
            row_starts,
            out,
            plan_topk_v2(lengths),
            bad_bound,
            out_offsets=row_starts,
        )


@torch.inference_mode()
def test_topk_v2_extend_requires_exactly_one_transform() -> None:
    """Supplying both output transforms, or neither, must fail loudly rather than
    silently pick one -- the two emit different index spaces."""
    device = "cuda"
    batch, kv, k = 4, 4096, 2048
    scores = torch.randn(batch, kv, dtype=torch.float32, device=device)
    lengths = torch.full((batch,), kv, dtype=torch.int32, device=device)
    row_starts = torch.zeros(batch, dtype=torch.int32, device=device)
    out = torch.empty(batch, k, dtype=torch.int32, device=device)
    plan = plan_topk_v2(lengths)
    page_table = torch.zeros(1, kv // PAGE_SIZE, dtype=torch.int32, device=device)
    row_to_batch = torch.zeros(batch, dtype=torch.int32, device=device)

    with pytest.raises(Exception):
        topk_transform_extend_v2(scores, lengths, row_starts, out, plan, kv)
    with pytest.raises(Exception):
        topk_transform_extend_v2(
            scores,
            lengths,
            row_starts,
            out,
            plan,
            kv,
            page_size=PAGE_SIZE,
            page_table=page_table,
            row_to_batch=row_to_batch,
            out_offsets=row_starts,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
