"""Loader units for ModelOpt NVFP4 checkpoints with fused expert tensors.

Such checkpoints (e.g. NVFP4 GPT-OSS) keep every expert of a layer in one
tensor and share a single scalar for the per-tensor scales, and the trtllm-gen
runner pads buffers past the checkpoint's own sizes. These tests pin the three
pieces of that path that are easy to break silently: name binding, scalar
broadcast, and padded copies.
"""

import unittest

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import (
    FusedMoE,
    _copy_into_padded_expert_data,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10)


class TestFusedNvfp4ExpertMapping(CustomTestCase):
    """The mapping binds checkpoint names to parameters by exact suffix."""

    def setUp(self):
        self.mapping = FusedMoE.make_expert_params_mapping_fused_nvfp4(
            ckpt_gate_up_proj_name="gate_up_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_gate_up_proj_bias_name="gate_up_proj_bias",
            ckpt_down_proj_bias_name="down_proj_bias",
        )

    def _resolve(self, ckpt_name: str):
        """Mirror the model's matcher: first mapping whose suffix matches."""
        for param_name, weight_name, shard_id in self.mapping:
            if ckpt_name.endswith(weight_name):
                return param_name, shard_id
        return None, None

    def test_every_checkpoint_tensor_binds_to_its_own_parameter(self):
        expected = {
            "gate_up_proj": ("experts.w13_weight", "w13"),
            "gate_up_proj_bias": ("experts.w13_weight_bias", "w13"),
            "gate_up_proj_weight_scale": ("experts.w13_weight_scale", "w13"),
            "gate_up_proj_weight_scale_2": ("experts.w13_weight_scale_2", "w13"),
            "gate_up_proj_input_scale": ("experts.w13_input_scale", "w13"),
            "down_proj": ("experts.w2_weight", "w2"),
            "down_proj_bias": ("experts.w2_weight_bias", "w2"),
            "down_proj_weight_scale": ("experts.w2_weight_scale", "w2"),
            "down_proj_weight_scale_2": ("experts.w2_weight_scale_2", "w2"),
            "down_proj_input_scale": ("experts.w2_input_scale", "w2"),
        }
        for suffix, want in expected.items():
            name = f"model.layers.3.mlp.experts.{suffix}"
            self.assertEqual(self._resolve(name), want, msg=name)

    def test_longer_names_win_over_their_prefixes(self):
        # "gate_up_proj" is a prefix of every other w13 name; ordering the
        # mapping wrong would silently route a scale into the weight buffer.
        weight_names = [weight_name for _, weight_name, _ in self.mapping]
        for i, name in enumerate(weight_names):
            for other in weight_names[i + 1 :]:
                self.assertFalse(
                    other.endswith(name),
                    msg=f"{other!r} is shadowed by the earlier entry {name!r}",
                )


class TestFusedPerTensorScaleBroadcast(CustomTestCase):
    """A checkpoint-wide scalar scale fills the per-expert parameter."""

    def test_scalar_fills_two_dimensional_parameter(self):
        param = torch.nn.Parameter(torch.empty(8, 2), requires_grad=False)
        FusedMoE._load_fused_per_tensor_scale(
            None, param, torch.tensor(0.0009765625, dtype=torch.float32)
        )
        self.assertTrue(torch.all(param.data == 0.0009765625))

    def test_per_expert_vector_broadcasts_over_the_shard_dim(self):
        param = torch.nn.Parameter(torch.empty(4, 2), requires_grad=False)
        loaded = torch.tensor([1.0, 2.0, 3.0, 4.0])
        FusedMoE._load_fused_per_tensor_scale(None, param, loaded)
        torch.testing.assert_close(param.data[:, 0], loaded)
        torch.testing.assert_close(param.data[:, 1], loaded)

    def test_scalar_fills_one_dimensional_parameter(self):
        param = torch.nn.Parameter(torch.empty(5), requires_grad=False)
        FusedMoE._load_fused_per_tensor_scale(None, param, torch.tensor(0.25))
        self.assertTrue(torch.all(param.data == 0.25))


class TestCopyIntoPaddedExpertData(CustomTestCase):
    """Padded buffers take the checkpoint in their leading slice."""

    def test_padding_on_the_last_dim_keeps_the_tail_untouched(self):
        # trtllm-gen pads hidden (the K dim of w13), which is not the dim the
        # TP sharding logic narrows.
        expert_data = torch.zeros(2, 4, 6)
        loaded = torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(2, 4, 5)
        _copy_into_padded_expert_data(expert_data, loaded)
        torch.testing.assert_close(expert_data[:, :, :5], loaded)
        self.assertTrue(torch.all(expert_data[:, :, 5:] == 0))

    def test_padding_on_several_dims_at_once(self):
        expert_data = torch.zeros(2, 8, 6)
        loaded = torch.ones(2, 4, 5)
        _copy_into_padded_expert_data(expert_data, loaded)
        torch.testing.assert_close(expert_data[:, :4, :5], loaded)
        self.assertEqual(expert_data.sum().item(), loaded.sum().item())

    def test_exact_fit_copies_everything(self):
        expert_data = torch.zeros(2, 3)
        loaded = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        _copy_into_padded_expert_data(expert_data, loaded)
        torch.testing.assert_close(expert_data, loaded)

    def test_rank_mismatch_falls_back_to_broadcast(self):
        expert_data = torch.zeros(2, 3)
        _copy_into_padded_expert_data(expert_data, torch.tensor(7.0))
        self.assertTrue(torch.all(expert_data == 7.0))


if __name__ == "__main__":
    unittest.main()
