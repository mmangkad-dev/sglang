"""Prefill CUDA-graph ladder sizing: the token cap and the memory reserved for it.

Two coupled defaults, both of which regressed once:

* ``_default_prefill_cuda_graph_max_bs`` decides how far the capture ladder runs.
  It raises the legacy 2048-token MLA cap only for a DSA model on the BREAKABLE
  backend -- tc_piecewise and full still capture the sparse-attention and indexer
  kernels, so the kernel-dispatch regression the cap exists to avoid still applies
  there. And ``chunked_prefill_size`` can legitimately be -1 ("chunking off"), which
  a plain ``min`` turns into ``max_bs=-1``, emptying the ladder so that
  ``PrefillCudaGraphRunner`` asserts at startup.

* ``reserve_for_graph_mb`` sizes the memory that ladder needs, and must fire on the
  same condition -- a widened ladder captured against the narrow ladder's flat
  reserve takes the shortfall out of the KV pool instead.

Both are pinned here because both failure modes are silent-at-review and expensive
at runtime, and because the reserve's coefficients are a measurement (see below)
that nothing else in the tree records executably.

    python -m pytest test/registered/unit/server_args/test_prefill_cuda_graph_sizing.py -v
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.server_args import DSA_PREFILL_CUDA_GRAPH_MAX_TOKENS, ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The default ladder's shape at the two caps that matter, from
# _generate_prefill_cuda_graph_batch_sizes. Spelled out so a change to the ladder
# that moves the reserve shows up here rather than only in a GPU run.
BUCKETS_AT_2048 = 42
BUCKETS_AT_6144 = 54
BUCKETS_AT_16384 = 74


def _args(
    *,
    use_mla: bool,
    is_dsa: bool,
    prefill_backend: str,
    chunked_prefill_size: int,
) -> ServerArgs:
    """A stand-in carrying only what the two methods under test read.

    ServerArgs' real constructor resolves a model config; this exercises the
    decision in isolation, the same way test_page_major_backend_allowlist does.
    """
    sa = ServerArgs.__new__(ServerArgs)
    object.__setattr__(sa, "chunked_prefill_size", chunked_prefill_size)
    object.__setattr__(sa, "disaggregation_mode", "null")
    object.__setattr__(
        sa,
        "cuda_graph_config",
        CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL, max_bs=512),
            prefill=PhaseConfig(backend=prefill_backend),
        ),
    )
    sa.use_mla_backend = lambda: use_mla
    sa._resolved = lambda: SimpleNamespace(enable_dp_attention=False)
    # is_deepseek_dsa() only reads the HF config; hand it one that answers.
    sa.get_model_config = lambda: SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM" if is_dsa else "Qwen3ForCausalLM"],
            index_topk=2048 if is_dsa else None,
        )
    )
    return sa


def _ladder(max_bs: int) -> list:
    return ServerArgs._generate_prefill_cuda_graph_batch_sizes(None, max_bs)


def _prefill_reserve(sa: ServerArgs, bs: list) -> float:
    """The prefill term of reserve_for_graph_mb, isolated by differencing against a
    disabled prefill backend so the decode / DP terms cancel."""
    backend = sa.cuda_graph_config.prefill.backend
    with mock.patch(
        "sglang.srt.server_args.resolved_view",
        return_value=SimpleNamespace(moe_a2a_backend="none"),
    ):
        sa.cuda_graph_config.prefill.bs = bs
        sa.cuda_graph_config.prefill.max_bs = max(bs) if bs else None
        total = sa.reserve_for_graph_mb()
        sa.cuda_graph_config.prefill.backend = Backend.DISABLED
        try:
            baseline = sa.reserve_for_graph_mb()
        finally:
            sa.cuda_graph_config.prefill.backend = backend
    return total - baseline


class TestPrefillCudaGraphMaxBs(CustomTestCase):
    def test_dsa_breakable_raises_the_cap(self):
        sa = _args(
            use_mla=True,
            is_dsa=True,
            prefill_backend=Backend.BREAKABLE,
            chunked_prefill_size=16384,
        )
        self.assertEqual(
            sa._default_prefill_cuda_graph_max_bs(), DSA_PREFILL_CUDA_GRAPH_MAX_TOKENS
        )

    def test_only_breakable_raises_the_cap(self):
        """The justification for raising it is about the backend, not the model:
        tc_piecewise and full capture the kernels the breaks keep eager, so they must
        keep the legacy MLA cap. This is the half that must stay in step with
        reserve_for_graph_mb (see TestPrefillGraphReserve)."""
        for backend in (Backend.TC_PIECEWISE, Backend.FULL, Backend.DISABLED):
            with self.subTest(backend=backend):
                sa = _args(
                    use_mla=True,
                    is_dsa=True,
                    prefill_backend=backend,
                    chunked_prefill_size=16384,
                )
                self.assertEqual(sa._default_prefill_cuda_graph_max_bs(), 2048)

    def test_non_dsa_mla_keeps_the_legacy_cap(self):
        sa = _args(
            use_mla=True,
            is_dsa=False,
            prefill_backend=Backend.BREAKABLE,
            chunked_prefill_size=16384,
        )
        self.assertEqual(sa._default_prefill_cuda_graph_max_bs(), 2048)

    def test_cap_never_exceeds_chunked_prefill_size(self):
        """A replay executes its whole bucket, so capturing past the scheduler's
        per-batch token budget would only ever pad."""
        sa = _args(
            use_mla=True,
            is_dsa=True,
            prefill_backend=Backend.BREAKABLE,
            chunked_prefill_size=4096,
        )
        self.assertEqual(sa._default_prefill_cuda_graph_max_bs(), 4096)

    def test_disabled_chunked_prefill_does_not_empty_the_ladder(self):
        """chunked_prefill_size == -1 means chunking is off (--enable-mis, multimodal
        archs that cannot chunk, or passed directly). min(-1, cap) == -1 drops every
        candidate from the ladder and PrefillCudaGraphRunner asserts on the empty
        list, so the cap alone has to be the bound."""
        for size in (-1, 0):
            with self.subTest(chunked_prefill_size=size):
                sa = _args(
                    use_mla=True,
                    is_dsa=True,
                    prefill_backend=Backend.BREAKABLE,
                    chunked_prefill_size=size,
                )
                max_bs = sa._default_prefill_cuda_graph_max_bs()
                self.assertEqual(max_bs, DSA_PREFILL_CUDA_GRAPH_MAX_TOKENS)
                self.assertTrue(_ladder(max_bs), "capture ladder came back empty")


class TestPrefillGraphReserve(CustomTestCase):
    """The widened-ladder reserve.

    Coefficients measured on GLM-5.2-NVFP4 (TP4, GB300) by differencing the
    per-bucket avail_mem the capture loop logs: ~36 MB per captured bucket for the
    graph objects (a breakable capture is ~2 segments per layer, so this tracks the
    bucket count) plus ~0.3 MB per token of the largest bucket for the shared mempool
    holding that bucket's activations. Measured totals were 2.10 GB at 42 buckets /
    2048 tokens, 3.93 GB at 54 / 6144 and 7.50 GB at 74 / 16384.

    Pinned as arithmetic rather than as prose: the fit is a three-point calibration
    on one model, so the numbers below are the record of what was measured, and a
    silent edit to either coefficient would otherwise only show up as an OOM or a
    shrunken KV pool on a machine nobody is watching.
    """

    def _sa(self, *, use_mla=True, backend=Backend.BREAKABLE):
        return _args(
            use_mla=use_mla,
            is_dsa=True,
            prefill_backend=backend,
            chunked_prefill_size=16384,
        )

    def test_narrow_ladder_keeps_the_flat_reserve(self):
        """At the legacy cap the arm must not fire -- the ladder was not widened, and
        every non-DSA MLA default lands here."""
        bs = _ladder(2048)
        self.assertEqual(len(bs), BUCKETS_AT_2048)
        self.assertEqual(_prefill_reserve(self._sa(), bs), 1.5 * 1024)

    def test_widened_ladder_is_sized_from_buckets_and_tokens(self):
        for max_bs, buckets in ((6144, BUCKETS_AT_6144), (16384, BUCKETS_AT_16384)):
            with self.subTest(max_bs=max_bs):
                bs = _ladder(max_bs)
                self.assertEqual(len(bs), buckets)
                self.assertAlmostEqual(
                    _prefill_reserve(self._sa(), bs),
                    buckets * 36 + max_bs * 0.3,
                    places=3,
                )

    def test_widened_reserve_is_floored_at_the_legacy_value(self):
        """A short-but-widened ladder must not come out below the flat reserve it
        replaces: the fit is calibrated on one model and this arm also covers a
        non-DSA MLA model whose max_bs was raised by hand."""
        bs = [2560]  # formula gives 36 + 768 = 804 MB, under the floor
        self.assertEqual(_prefill_reserve(self._sa(), bs), 1.5 * 1024)

    def test_only_breakable_uses_the_widened_reserve(self):
        """The other half of the agreement with _default_prefill_cuda_graph_max_bs:
        a backend that cannot get a widened ladder must not get its reserve either."""
        bs = _ladder(16384)
        for backend in (Backend.TC_PIECEWISE, Backend.FULL):
            with self.subTest(backend=backend):
                self.assertEqual(
                    _prefill_reserve(self._sa(backend=backend), bs), 1.5 * 1024
                )

    def test_non_mla_reserve_is_unchanged(self):
        bs = _ladder(16384)
        self.assertEqual(_prefill_reserve(self._sa(use_mla=False), bs), len(bs) * 8)

    def test_cap_and_reserve_agree(self):
        """Whenever the cap widens the ladder past 2048, the widened reserve must
        fire. Checked over the cross product rather than asserted in prose, because
        these two decisions live ~200 lines apart."""
        for backend in (Backend.BREAKABLE, Backend.TC_PIECEWISE, Backend.FULL):
            for is_dsa in (True, False):
                for chunked in (-1, 2048, 8192, 16384):
                    with self.subTest(backend=backend, is_dsa=is_dsa, chunked=chunked):
                        sa = _args(
                            use_mla=True,
                            is_dsa=is_dsa,
                            prefill_backend=backend,
                            chunked_prefill_size=chunked,
                        )
                        max_bs = sa._default_prefill_cuda_graph_max_bs()
                        bs = _ladder(max_bs)
                        widened_reserve = _prefill_reserve(sa, bs) != 1.5 * 1024
                        if max_bs > 2048:
                            self.assertTrue(
                                widened_reserve,
                                f"ladder widened to {max_bs} on the flat reserve",
                            )


if __name__ == "__main__":
    unittest.main()
