"""Unit tests for ``flush_moe_deferred_all_reduce``.

This helper completes a post-expert TP/EP all-reduce that all-reduce fusion
deferred into the next layer's ``prepare_attn``. At a pipeline-parallel
boundary there is no next layer and the ``_sglang_needs_allreduce_fusion``
marker is lost across the PP send, so the model calls this helper right before
handing ``hidden_states`` to the next stage as a ``PPProxyTensor``.

These tests pin the helper's contract — no-op safety when the marker is
absent, correct EP-then-TP ordering and world-size gating, and marker
clearing — without launching a server or initializing a real process group.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.distributed import flush_moe_deferred_all_reduce
from sglang.test.test_utils import CustomTestCase

# The helper references its dependencies as module globals of
# communication_op, so patch them there (not at the re-export in
# sglang.srt.distributed, and not at their source in parallel_state).
_CO = "sglang.srt.distributed.communication_op"


class TestFlushMoeDeferredAllReduce(CustomTestCase):
    def _tensor_with_marker(self, *, marked: bool) -> torch.Tensor:
        t = torch.zeros(3, 4)
        if marked:
            t._sglang_needs_allreduce_fusion = True
        return t

    def test_noop_without_marker(self):
        """The common path: nothing is deferred, so the tensor is returned
        untouched and no all-reduce is issued."""
        t = self._tensor_with_marker(marked=False)

        with (
            patch(f"{_CO}.moe_expert_parallel_all_reduce") as ep,
            patch(f"{_CO}.moe_tensor_model_parallel_all_reduce") as tp,
            patch(f"{_CO}.get_moe_expert_parallel_world_size", return_value=2),
            patch(f"{_CO}.get_moe_tensor_parallel_world_size", return_value=2),
        ):
            out = flush_moe_deferred_all_reduce(t)

        self.assertIs(out, t)
        ep.assert_not_called()
        tp.assert_not_called()

    def test_none_passthrough(self):
        """A PP boundary may pass ``hidden_states=None`` (e.g. an empty/idle
        forward); the helper must not raise and must return ``None``."""
        with (
            patch(f"{_CO}.moe_expert_parallel_all_reduce") as ep,
            patch(f"{_CO}.moe_tensor_model_parallel_all_reduce") as tp,
            patch(f"{_CO}.get_moe_expert_parallel_world_size", return_value=2),
            patch(f"{_CO}.get_moe_tensor_parallel_world_size", return_value=2),
        ):
            self.assertIsNone(flush_moe_deferred_all_reduce(None))
        ep.assert_not_called()
        tp.assert_not_called()

    def test_flushes_ep_then_tp_and_clears_marker(self):
        """When the marker is set and both worlds > 1, the EP reduce runs
        first (feeding its output into the TP reduce), then the marker is
        cleared — matching the original qwen2_moe fix."""
        t = self._tensor_with_marker(marked=True)
        ep_out = torch.ones(3, 4)
        tp_out = torch.full((3, 4), 2.0)

        with (
            patch(f"{_CO}.moe_expert_parallel_all_reduce", return_value=ep_out) as ep,
            patch(
                f"{_CO}.moe_tensor_model_parallel_all_reduce", return_value=tp_out
            ) as tp,
            patch(f"{_CO}.get_moe_expert_parallel_world_size", return_value=2),
            patch(f"{_CO}.get_moe_tensor_parallel_world_size", return_value=2),
        ):
            out = flush_moe_deferred_all_reduce(t)

        self.assertIs(out, tp_out)
        # EP reduce ran on the original marked tensor; TP reduce ran on the
        # EP-reduced tensor (the return value of the EP mock).
        ep.assert_called_once_with(t)
        tp.assert_called_once_with(ep_out)
        self.assertFalse(out._sglang_needs_allreduce_fusion)

    def test_only_tp_when_ep_world_size_is_one(self):
        """EP all-reduce is skipped when ``get_moe_expert_parallel_world_size
        == 1``; the TP reduce still runs and clears the marker."""
        t = self._tensor_with_marker(marked=True)
        tp_out = torch.full((3, 4), 7.0)

        with (
            patch(f"{_CO}.moe_expert_parallel_all_reduce") as ep,
            patch(
                f"{_CO}.moe_tensor_model_parallel_all_reduce", return_value=tp_out
            ) as tp,
            patch(f"{_CO}.get_moe_expert_parallel_world_size", return_value=1),
            patch(f"{_CO}.get_moe_tensor_parallel_world_size", return_value=2),
        ):
            out = flush_moe_deferred_all_reduce(t)

        self.assertIs(out, tp_out)
        ep.assert_not_called()
        tp.assert_called_once_with(t)
        self.assertFalse(out._sglang_needs_allreduce_fusion)

    def test_only_ep_when_tp_world_size_is_one(self):
        """TP all-reduce is skipped when ``get_moe_tensor_parallel_world_size
        == 1``; the EP reduce still runs and clears the marker."""
        t = self._tensor_with_marker(marked=True)
        ep_out = torch.full((3, 4), 9.0)

        with (
            patch(f"{_CO}.moe_expert_parallel_all_reduce", return_value=ep_out) as ep,
            patch(f"{_CO}.moe_tensor_model_parallel_all_reduce") as tp,
            patch(f"{_CO}.get_moe_expert_parallel_world_size", return_value=2),
            patch(f"{_CO}.get_moe_tensor_parallel_world_size", return_value=1),
        ):
            out = flush_moe_deferred_all_reduce(t)

        self.assertIs(out, ep_out)
        ep.assert_called_once_with(t)
        tp.assert_not_called()
        self.assertFalse(out._sglang_needs_allreduce_fusion)

    def test_no_reduce_when_both_world_sizes_are_one(self):
        """Even with the marker set, neither reduce runs if neither parallel
        group exists, but the marker is still cleared so the hand-off tensor
        is clean."""
        t = self._tensor_with_marker(marked=True)

        with (
            patch(f"{_CO}.moe_expert_parallel_all_reduce") as ep,
            patch(f"{_CO}.moe_tensor_model_parallel_all_reduce") as tp,
            patch(f"{_CO}.get_moe_expert_parallel_world_size", return_value=1),
            patch(f"{_CO}.get_moe_tensor_parallel_world_size", return_value=1),
        ):
            out = flush_moe_deferred_all_reduce(t)

        self.assertIs(out, t)
        ep.assert_not_called()
        tp.assert_not_called()
        self.assertFalse(out._sglang_needs_allreduce_fusion)


if __name__ == "__main__":
    unittest.main(verbosity=3)
