"""Correctness test for the FlashKDA prefill backend (safe-gate KDA).

Validates ``FlashKDAKernel`` (the external ``flash_kda`` fused CUTLASS kernel)
against the production Triton ``chunk_kda`` safe-gate reference. FlashKDA only
implements the safe/bounded gate, so this exercises ``lower_bound=-5``.

Requires an SM90+ GPU and the ``flash_kda`` package
(``pip install git+https://github.com/MoonshotAI/FlashKDA.git``); skips
otherwise. Mirrors ``test_kda_prefill_cutedsl.py``.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

# SM90+ single-GPU kernel-unit suite. Disabled in CI: flash_kda is not in the
# public runner image, and the module-level pytest.skip aborts non-zero under
# `python3 file.py`. Drop `disabled=` once flash_kda ships in the runner image;
# still runs locally / on internal CI where it is installed.
register_cuda_ci(
    est_time=20,
    stage="base-b",
    runner_config="1-gpu-large",
    disabled="flash_kda not in public CI runner image (only on Ant-internal PyPI)",
)

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9):
    pytest.skip(
        "FlashKDA requires CUDA SM90+ (Hopper or newer).",
        allow_module_level=True,
    )
try:
    import flash_kda  # noqa: F401
except ImportError:
    pytest.skip(
        "flash_kda not installed "
        "(pip install git+https://github.com/MoonshotAI/FlashKDA.git).",
        allow_module_level=True,
    )

from sglang.kernels.ops.attention.fla.kda import chunk_kda  # noqa: E402
from sglang.srt.layers.attention.linear.kernels.kda_flashkda import (  # noqa: E402
    FlashKDAKernel,
)

LOWER_BOUND = -5.0
H, K, V = 16, 128, 128  # FlashKDA requires K == V == 128, HV == H


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def _make_inputs(seq_lens):
    n = len(seq_lens)
    cu = torch.zeros(n + 1, device="cuda", dtype=torch.int32)
    cu[1:] = torch.tensor(seq_lens, device="cuda").cumsum(0)
    total = int(cu[-1].item())
    idx = torch.arange(n, device="cuda", dtype=torch.int32)
    return dict(
        cu=cu,
        idx=idx,
        # RAW q/k (FlashKDA L2-norms internally; chunk_kda via use_qk_l2norm).
        q=torch.randn(1, total, H, K, device="cuda", dtype=torch.bfloat16) * 0.5,
        k=torch.randn(1, total, H, K, device="cuda", dtype=torch.bfloat16) * 0.5,
        v=torch.randn(1, total, H, V, device="cuda", dtype=torch.bfloat16) * 0.5,
        g=torch.randn(1, total, H, K, device="cuda", dtype=torch.bfloat16) * 0.5,
        # post-sigmoid beta in [0.1, 0.9] (FlashKDA inverts to logits internally).
        beta=(torch.rand(1, total, H, device="cuda") * 0.8 + 0.1).to(torch.bfloat16),
        A_log=torch.randn(1, 1, H, 1, device="cuda", dtype=torch.float32) * 0.5,
        dt_bias=torch.randn(H * K, device="cuda", dtype=torch.float32) * 0.1,
        pool=torch.randn(n, H, V, K, device="cuda", dtype=torch.float32) * 0.1,
    )


def _chunk_kda_ref(d, lower_bound):
    """Triton chunk_kda reference. chunk_kda mutates g/v and the state in place,
    so feed clones; returns (output, updated_state_slots)."""
    st = d["pool"].clone()
    out = chunk_kda(
        q=d["q"].clone(),
        k=d["k"].clone(),
        v=d["v"].clone(),
        g=d["g"].clone(),
        beta=d["beta"].clone(),
        initial_state=st,
        initial_state_indices=d["idx"],
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=d["cu"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        lower_bound=lower_bound,
    )
    # chunk_kda returns the output alone unless intermediate states are asked
    # for; keep accepting the legacy (output, states) tuple.
    if isinstance(out, tuple):
        out = out[0]
    return out, st[d["idx"]]


@pytest.mark.parametrize("seq_lens", [[128], [128, 384, 512], [96] * 4])
def test_flashkda_matches_triton_safe_gate(seq_lens):
    torch.manual_seed(len(seq_lens))
    d = _make_inputs(seq_lens)

    ref_out, ref_state = _chunk_kda_ref(d, LOWER_BOUND)

    st_fk = d["pool"].clone()
    out = FlashKDAKernel().extend(
        d["q"].clone(),
        d["k"].clone(),
        d["v"].clone(),
        d["g"].clone(),
        d["beta"].clone(),
        ssm_states=st_fk,
        cache_indices=d["idx"],
        query_start_loc=d["cu"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        lower_bound=LOWER_BOUND,
        extend_seq_lens_cpu=seq_lens,
    )
    torch.cuda.synchronize()

    # Every KDA extend kernel returns a bare tensor unless intermediate states
    # were requested; the backend indexes the result directly.
    assert isinstance(out, torch.Tensor), f"extend returned {type(out).__name__}"
    assert torch.isfinite(out).all(), "FlashKDA output has non-finite values"
    assert torch.isfinite(st_fk).all(), "FlashKDA final state has non-finite values"
    # bf16 cross-implementation noise (chunk=16 CUTLASS vs chunk=64 Triton);
    # measured cos ~0.985 output / ~0.9999 state on H20-3e and B200.
    assert _cos(ref_out, out) > 0.95, f"output cos too low: {_cos(ref_out, out):.4f}"
    assert _cos(ref_state, st_fk[d["idx"]]) > 0.99, (
        f"state cos too low: {_cos(ref_state, st_fk[d['idx']]):.4f}"
    )


def test_flashkda_falls_back_without_lower_bound():
    """Unbounded gate (lower_bound=None): FlashKDA must route to the Triton
    chunk_kda fallback (it only supports the safe gate)."""
    torch.manual_seed(0)
    d = _make_inputs([256])

    ref_out, _ = _chunk_kda_ref(d, None)

    st_fk = d["pool"].clone()
    out = FlashKDAKernel().extend(
        d["q"].clone(),
        d["k"].clone(),
        d["v"].clone(),
        d["g"].clone(),
        d["beta"].clone(),
        ssm_states=st_fk,
        cache_indices=d["idx"],
        query_start_loc=d["cu"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        lower_bound=None,
        extend_seq_lens_cpu=[256],
    )
    torch.cuda.synchronize()

    assert torch.isfinite(out).all()
    # Same Triton code path as the reference -> matches closely.
    assert _cos(ref_out, out) > 0.999, f"fallback cos too low: {_cos(ref_out, out):.4f}"


def test_flashkda_spec_verify_falls_back():
    """Speculative verify / draft-extend (is_spec_decode=True) must route to the
    Triton fallback even with a safe gate and in-range seqs -- FlashKDA commits
    the recurrent state in place, which would break draft rollback."""
    torch.manual_seed(0)
    d = _make_inputs([256])
    ref_out, _ = _chunk_kda_ref(d, LOWER_BOUND)

    st_fk = d["pool"].clone()
    out = FlashKDAKernel().extend(
        d["q"].clone(),
        d["k"].clone(),
        d["v"].clone(),
        d["g"].clone(),
        d["beta"].clone(),
        ssm_states=st_fk,
        cache_indices=d["idx"],
        query_start_loc=d["cu"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        lower_bound=LOWER_BOUND,
        extend_seq_lens_cpu=[256],
        is_spec_decode=True,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(out).all()
    # Took the Triton fallback (not FlashKDA) -> matches chunk_kda closely. If
    # FlashKDA had run, the cross-impl cos would be ~0.985 and this would fail.
    assert _cos(ref_out, out) > 0.999, (
        f"spec-decode did not fall back: {_cos(ref_out, out):.4f}"
    )


@pytest.mark.parametrize(
    "seq_lens,num_heads,expect_fallback",
    [
        # Below the chunk size, and short sequences (win at any head count).
        ([32], 16, True),
        ([256], 16, False),
        ([2048], 16, False),
        ([2048], 4, False),  # short: no occupancy requirement
        # Long sequences: the gate needs seqs x heads >= 32 CTAs, where "seqs"
        # is total/longest so a batch padded with short requests does not count
        # as full. Measured speedups are in the comment on the gate itself.
        ([8192], 16, True),  # 16 CTAs -> 0.83x
        ([8192, 1024], 16, True),  # 1.1 equivalents -> 0.85x
        ([8192, 8192], 16, False),  # 32 CTAs -> 1.13x
        ([8192, 4096, 4096], 16, False),  # 32 CTAs -> 1.12x
        ([4096, 2048], 16, True),  # 1.5 equivalents, 24 CTAs
        # Same batches, different per-rank head counts: the crossover moves.
        ([8192, 8192], 8, True),  # 16 CTAs -> 0.77x, must not run FlashKDA
        ([8192] * 4, 8, False),  # 32 CTAs -> 1.06x
        ([8192] * 4, 4, True),  # 16 CTAs -> 0.82x
        ([8192] * 8, 4, False),  # 32 CTAs -> 1.14x
        ([8192], 32, False),  # 32 CTAs -> 1.07x, wins on one sequence
    ],
)
def test_flashkda_batch_fill_gate(seq_lens, num_heads, expect_fallback):
    """The long-sequence gate keys off grid occupancy -- (total tokens /
    longest sequence) x per-rank heads -- not the longest sequence alone.
    FlashKDA's cost tracks the longest sequence while Triton's tracks total
    tokens, so a well-populated batch is a FlashKDA win well past
    _FLASHKDA_SHORT_SEQ_LEN, and a thin one is a loss even at 2 sequences."""
    cu = torch.zeros(len(seq_lens) + 1, device="cuda", dtype=torch.int32)
    cu[1:] = torch.tensor(seq_lens, device="cuda").cumsum(0)

    # Both the CPU-list and the query_start_loc-derived paths must agree.
    for lens_cpu in (seq_lens, None):
        assert (
            FlashKDAKernel._should_fall_back(
                LOWER_BOUND, False, cu, lens_cpu, num_heads
            )
            is expect_fallback
        ), f"seq_lens={seq_lens} heads={num_heads} extend_seq_lens_cpu={lens_cpu}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
