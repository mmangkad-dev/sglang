"""MLA prefill is traced under `--cuda-graph-backend-prefill=tc_piecewise`, where
Dynamo rejects the non-contiguous `out=` that lands the absorbed query token-major.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    enable_tc_piecewise_cuda_graph,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    _absorbed_q_bmm,
)

# (num_tokens, num_heads); GLM-5.3-Flash at TP4 is 16 heads, 256 -> 512.
SHAPES = [
    (0, 4),  # idle batch
    (1, 4),  # single decode token: `out` view is contiguous anyway
    (4, 1),  # single head: likewise
    (5, 4),  # both > 1: only this shape trips the Dynamo rejection
    (384, 16),  # decode-sized
]
QK_NOPE_HEAD_DIM, KV_LORA_RANK = 8, 6


def _inputs(*, num_tokens: int, num_heads: int, dtype: torch.dtype):
    torch.manual_seed(0)
    q_nope = torch.randn(num_tokens, num_heads, QK_NOPE_HEAD_DIM, dtype=dtype)
    w_kc = torch.randn(num_heads, QK_NOPE_HEAD_DIM, KV_LORA_RANK, dtype=dtype)
    return q_nope, w_kc


def _reference(*, q_nope: torch.Tensor, w_kc: torch.Tensor) -> torch.Tensor:
    return torch.bmm(q_nope.transpose(0, 1), w_kc).transpose(0, 1)


class TestAbsorbedQBmm(CustomTestCase):
    def test_matches_allocating_bmm_and_is_token_major(self):
        """Untraced, the result is bit-identical to the allocating form."""
        for num_tokens, num_heads in SHAPES:
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(tokens=num_tokens, heads=num_heads, dtype=dtype):
                    q_nope, w_kc = _inputs(
                        num_tokens=num_tokens, num_heads=num_heads, dtype=dtype
                    )
                    out = _absorbed_q_bmm(q_nope=q_nope, w_kc=w_kc)
                    self.assertEqual(out.shape, (num_tokens, num_heads, KV_LORA_RANK))
                    self.assertTrue(out.is_contiguous())
                    expected = _reference(q_nope=q_nope, w_kc=w_kc)
                    self.assertTrue(torch.equal(out, expected))

    def test_compiles_while_traced(self):
        """Traced, the helper must not hand Dynamo a non-contiguous `out=`.

        Only shapes whose token AND head counts both exceed 1 catch this; the
        `out=` view is contiguous anyway when either is 1.
        """
        for num_tokens, num_heads in SHAPES:
            with self.subTest(tokens=num_tokens, heads=num_heads):
                q_nope, w_kc = _inputs(
                    num_tokens=num_tokens, num_heads=num_heads, dtype=torch.float32
                )
                expected = _reference(q_nope=q_nope, w_kc=w_kc)
                torch._dynamo.reset()
                compiled = torch.compile(
                    _absorbed_q_bmm, backend="eager", fullgraph=True
                )
                with enable_tc_piecewise_cuda_graph():
                    out = compiled(q_nope=q_nope, w_kc=w_kc)
                self.assertEqual(out.shape, expected.shape)
                self.assertTrue(torch.equal(out, expected))


if __name__ == "__main__":
    unittest.main()
