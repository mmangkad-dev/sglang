"""Every _run_prefill_kernel override must bind the chunked-prefix arguments.

An override left on a stale signature raises TypeError at dispatch, before any
kernel runs, and only on batches that carry an empty row, so it survives both
the helper tests and an ordinary prefill.
"""

import ast
import inspect
import unittest
from pathlib import Path

from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_ATTENTION_DIR = Path(inspect.getfile(TRTLLMMLABackend)).parent

_PREFILL_HOOK_OWNERS = {"trtllm_mla_backend.py", "tokenspeed_mla_backend.py"}

# What the chunked-prefix call site passes when a chunk may carry an empty row.
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
    q_seq_lens_cpu=None,
    kv_seq_lens_cpu=None,
    all_rows_active=False,
    o_sf_scale=-1.0,
)


class TestPrefillHookDispatch(CustomTestCase):
    def test_every_override_is_accounted_for(self):
        owners = set()
        for path in _ATTENTION_DIR.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "_run_prefill_kernel"
                ):
                    owners.add(path.name)
        self.assertEqual(
            owners,
            _PREFILL_HOOK_OWNERS,
            "a _run_prefill_kernel override was added or removed; it must take "
            "q_seq_lens_cpu / kv_seq_lens_cpu / all_rows_active, then update "
            "this pin",
        )

    def test_overrides_bind_the_chunked_prefix_arguments(self):
        for cls in (TRTLLMMLABackend, TokenspeedMLABackend):
            with self.subTest(cls.__name__):
                inspect.signature(cls._run_prefill_kernel).bind(
                    object.__new__(cls), **_HOOK_CALL
                )


if __name__ == "__main__":
    unittest.main()
