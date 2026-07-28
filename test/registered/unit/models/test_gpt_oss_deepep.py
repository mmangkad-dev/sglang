"""Unit tests for GPT-OSS DeepEP routing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.models.gpt_oss import (
    GptOssSparseMoeBlock,
    _validate_gpt_oss_deepep_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _DeepEPBackend:
    def is_deepep(self):
        return True


class _Router(nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 1, None


class _TopK(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_token_non_padded = None
        self.expert_location_dispatch_info = None
        self.empty_layer_id = None

    def forward(
        self,
        hidden_states,
        router_logits,
        *,
        num_token_non_padded,
        expert_location_dispatch_info,
    ):
        self.num_token_non_padded = num_token_non_padded
        self.expert_location_dispatch_info = expert_location_dispatch_info
        return hidden_states + router_logits

    def empty_topk_output(self, device, *, layer_id=None):
        self.empty_layer_id = layer_id
        return torch.empty(0, 2, device=device)


class _Experts(nn.Module):
    def forward(self, hidden_states, topk_output):
        return hidden_states + topk_output


class _PaddedExperts(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.quant_method = SimpleNamespace(hidden_size=hidden_size)
        self.received_shape = None

    def forward(self, hidden_states, topk_output):
        self.received_shape = hidden_states.shape
        return hidden_states


class TestGptOssDeepEPRouting(CustomTestCase):
    def test_bf16_deepep_is_rejected(self):
        with (
            patch(
                "sglang.srt.models.gpt_oss.get_moe_a2a_backend",
                return_value=_DeepEPBackend(),
            ),
            self.assertRaisesRegex(ValueError, "only MXFP4 experts"),
        ):
            _validate_gpt_oss_deepep_config(None)

    def _make_block(self, experts):
        block = GptOssSparseMoeBlock.__new__(GptOssSparseMoeBlock)
        nn.Module.__init__(block)
        block.layer_id = 3
        block.hidden_size = 2
        block.router = _Router()
        block.topk = _TopK()
        block.experts = experts
        block.expert_hidden_size = (
            experts.quant_method.hidden_size
            if isinstance(experts, _PaddedExperts)
            else block.hidden_size
        )
        return block

    def _forward(self, block, hidden_states, dispatch_info):
        forward_batch = SimpleNamespace(num_token_non_padded=1)
        with (
            patch(
                "sglang.srt.models.gpt_oss.get_moe_a2a_backend",
                return_value=_DeepEPBackend(),
            ),
            patch(
                "sglang.srt.models.gpt_oss.get_server_args",
                return_value=SimpleNamespace(dwdp_size=1),
            ),
            patch.object(
                ExpertLocationDispatchInfo,
                "init_new",
                return_value=dispatch_info,
            ),
        ):
            return block(hidden_states, forward_batch)

    def test_forward_routes_non_padding_token_count_to_deepep_topk(self):
        block = self._make_block(_Experts())
        hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        dispatch_info = object()

        output = self._forward(block, hidden_states, dispatch_info)

        self.assertEqual(block.topk.num_token_non_padded, 1)
        self.assertIs(block.topk.expert_location_dispatch_info, dispatch_info)
        torch.testing.assert_close(
            output,
            torch.tensor([[4.0, 7.0], [10.0, 13.0]]),
        )

    def test_forward_pads_for_deepep_and_trims_expert_output(self):
        experts = _PaddedExperts(hidden_size=4)
        block = self._make_block(experts)
        hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        output = self._forward(block, hidden_states, object())

        self.assertEqual(experts.received_shape, torch.Size([2, 4]))
        torch.testing.assert_close(output, hidden_states)

    def test_empty_rank_participates_in_expert_location_routing(self):
        experts = _PaddedExperts(hidden_size=4)
        block = self._make_block(experts)
        hidden_states = torch.empty(0, 2)

        output = self._forward(block, hidden_states, object())

        self.assertEqual(block.topk.empty_layer_id, block.layer_id)
        self.assertEqual(experts.received_shape, torch.Size([0, 4]))
        self.assertEqual(output.shape, torch.Size([0, 2]))


if __name__ == "__main__":
    unittest.main()
