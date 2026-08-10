"""Regression tests for FlashInfer TRT-LLM routed-input carrier selection."""

import unittest

import torch

import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    _get_topk_routing_input_for_flashinfer_routed,
)
from sglang.srt.layers.moe.topk import (
    PackedTopKOutput,
    StandardTopKOutput,
    StandardTopKOutputPacked,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlashInferTrtllmRoutingInput(CustomTestCase):
    def test_preserves_packed_and_converts_standard_carriers(self):
        """Packed-only model routing must not be read as a standard carrier."""
        packed = torch.tensor([[0x10000]], dtype=torch.int32)
        router_logits = torch.empty((1, 0))
        packed_output = PackedTopKOutput(packed, router_logits)

        self.assertIs(
            _get_topk_routing_input_for_flashinfer_routed(
                packed_output, unpack_standard=True
            ),
            packed,
        )

        standard_packed_output = StandardTopKOutputPacked(
            torch.tensor([[0.5]]),
            torch.tensor([[1]], dtype=torch.int32),
            router_logits,
            packed,
        )
        self.assertIs(
            _get_topk_routing_input_for_flashinfer_routed(
                standard_packed_output, unpack_standard=True
            ),
            packed,
        )

        topk_ids = torch.tensor([[1]], dtype=torch.int32)
        topk_weights = torch.tensor([[0.5]], dtype=torch.float32)
        standard_output = StandardTopKOutput(topk_weights, topk_ids, router_logits)
        routing_ids, routing_weights = _get_topk_routing_input_for_flashinfer_routed(
            standard_output, unpack_standard=True
        )

        self.assertIs(routing_ids, topk_ids)
        self.assertEqual(routing_weights.dtype, torch.bfloat16)
        torch.testing.assert_close(routing_weights.float(), topk_weights)


if __name__ == "__main__":
    unittest.main()
