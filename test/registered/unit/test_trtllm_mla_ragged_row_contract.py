"""FlashInfer's ragged prefill only accepts ``skip_all_rows_active_check`` when
every row has positive query and KV lengths. A batch that breaks that must hand
the kernel host length mirrors instead, so it still compacts the empty rows.
Skipping the scan on such a batch is undefined behavior the outputs of the
*other* rows depend on, and no downstream fixup restores them.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ast
import inspect
import unittest
from pathlib import Path

import torch

from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLAPrefillMetadata,
)
from sglang.test.test_utils import CustomTestCase

_SKIP = "skip_all_rows_active_check"

_ATTENTION_DIR = Path(inspect.getfile(TRTLLMMLABackend)).parent

# Files that define a _run_prefill_kernel override. A new entry here has to take
# the base hook's full argument list or the prefix path fails to dispatch to it.
_PREFILL_HOOK_OWNERS = {"trtllm_mla_backend.py", "tokenspeed_mla_backend.py"}

# The arguments the chunked-prefix call site sends when a chunk may be empty.
_HOOK_CALL = dict(
    q=None,
    k=None,
    v=None,
    layer=None,
    batch_size=3,
    cum_seq_lens_q=None,
    max_q_len=257,
    seq_lens_kv=None,
    cum_seq_lens_kv=None,
    max_kv_len=512,
    is_causal=False,
    return_lse=True,
    out_buffer=None,
    o_sf_scale=-1.0,
    q_seq_lens_cpu=None,
    kv_seq_lens_cpu=None,
)


def _backend(extend_lens):
    backend = object.__new__(TRTLLMMLABackend)
    backend.forward_prefill_metadata = TRTLLMMLAPrefillMetadata(
        max_seq_len=max(extend_lens),
        cum_seq_lens=torch.empty(0),
        seq_lens=torch.empty(0),
        seq_lens_cpu=torch.tensor(extend_lens, dtype=torch.int32),
        all_rows_active=min(extend_lens) > 0,
    )
    return backend


class TestTrtllmMlaRaggedRowContract(CustomTestCase):
    def test_all_positive_rows_send_no_mirrors(self):
        # No mirrors is what lets _run_prefill_kernel skip the scan; the unsafe
        # direction (skipping when a row is inactive) is covered below.
        self.assertEqual(_backend([129, 7, 257])._row_len_mirrors(), {})

    def test_zero_kv_chunk_sends_mirrors_instead_of_skipping(self):
        kwargs = _backend([129, 7, 257])._row_len_mirrors(
            kv_lens_cpu=torch.tensor([0, 64, 512], dtype=torch.int32),
            kv_has_zero=True,
        )
        self.assertNotIn(_SKIP, kwargs)
        self.assertEqual(kwargs["kv_seq_lens_cpu"].tolist(), [0, 64, 512])
        self.assertEqual(kwargs["q_seq_lens_cpu"].tolist(), [129, 7, 257])

    def test_zero_query_row_sends_mirrors_on_the_causal_path(self):
        # kv_len == q_len there, so a zero extend length makes the row inactive
        # even though no prefix chunk is involved.
        kwargs = _backend([129, 0, 257])._row_len_mirrors()
        self.assertNotIn(_SKIP, kwargs)
        self.assertEqual(kwargs["kv_seq_lens_cpu"].tolist(), [129, 0, 257])

    def test_mirrors_are_int32_cpu_as_the_kernel_requires(self):
        kwargs = _backend([129, 7, 257])._row_len_mirrors(
            kv_lens_cpu=torch.tensor([0, 64, 512], dtype=torch.int64),
            kv_has_zero=True,
        )
        for name in ("q_seq_lens_cpu", "kv_seq_lens_cpu"):
            self.assertEqual(kwargs[name].dtype, torch.int32, name)
            self.assertEqual(kwargs[name].device.type, "cpu", name)


class TestPrefillHookDispatch(CustomTestCase):
    """The prefix path calls _run_prefill_kernel with host length mirrors, so
    every override has to accept them; a stale signature raises TypeError at
    dispatch, before any kernel runs, and only on batches with an empty row.
    """

    def test_every_override_is_accounted_for(self):
        owners = set()
        for path in _ATTENTION_DIR.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "_run_prefill_kernel"
                ):
                    owners.add(path.name)
        self.assertEqual(
            owners,
            _PREFILL_HOOK_OWNERS,
            "a _run_prefill_kernel override was added or removed; make sure it "
            "takes q_seq_lens_cpu / kv_seq_lens_cpu and update this pin",
        )

    def test_overrides_bind_the_chunked_prefix_arguments(self):
        for cls in (TRTLLMMLABackend, TokenspeedMLABackend):
            with self.subTest(cls.__name__):
                # bind() checks argument binding without running the kernel.
                inspect.signature(cls._run_prefill_kernel).bind(
                    object.__new__(cls), **_HOOK_CALL
                )


if __name__ == "__main__":
    unittest.main()
