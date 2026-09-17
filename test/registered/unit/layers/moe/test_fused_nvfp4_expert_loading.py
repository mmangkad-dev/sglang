"""Loader units for ModelOpt NVFP4 checkpoints with fused expert tensors.

Such checkpoints (e.g. NVFP4 GPT-OSS) keep every expert of a layer in one
tensor and share a single scalar for the per-tensor scales, and the trtllm-gen
runner pads buffers past the checkpoint's own sizes. These tests pin the three
pieces of that path that are easy to break silently: name binding, scalar
broadcast, and padded copies.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.utils import round_up

from sglang.srt.layers.moe.fused_moe_triton.layer import (
    FusedMoE,
    _copy_into_padded_expert_data,
    _load_fused_per_tensor_scale,
    _narrow_checkpoint_to_rank,
    match_fused_expert_param,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Every expert tensor an NVFP4 GPT-OSS layer carries, and the parameter each one
# must reach. Taken from the safetensors index of mmangkad/gpt-oss-120b-nvfp4.
CHECKPOINT_TO_PARAM = {
    "gate_up_proj": "w13_weight",
    "gate_up_proj_bias": "w13_weight_bias",
    "gate_up_proj_weight_scale": "w13_weight_scale",
    "gate_up_proj_weight_scale_2": "w13_weight_scale_2",
    "gate_up_proj_input_scale": "w13_input_scale",
    "down_proj": "w2_weight",
    "down_proj_bias": "w2_weight_bias",
    "down_proj_weight_scale": "w2_weight_scale",
    "down_proj_weight_scale_2": "w2_weight_scale_2",
    "down_proj_input_scale": "w2_input_scale",
}


def _nvfp4_mapping():
    return FusedMoE.make_expert_params_mapping_fused_nvfp4(
        ckpt_gate_up_proj_name="gate_up_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_gate_up_proj_bias_name="gate_up_proj_bias",
        ckpt_down_proj_bias_name="down_proj_bias",
    )


class TestFusedNvfp4ExpertMapping(CustomTestCase):
    def setUp(self):
        self.mapping = _nvfp4_mapping()
        # The parameters ModelOptNvFp4FusedMoEMethod registers for a layer with
        # biases, which is what the mapping has to reach.
        self.params_dict = {
            f"model.layers.0.mlp.experts.{param}": object()
            for param in CHECKPOINT_TO_PARAM.values()
        }

    def test_every_checkpoint_tensor_binds_to_its_own_parameter(self):
        # A missing binding is silent: an unbound w13_input_scale keeps its
        # fill_value=1.0 default and dequantizes with the wrong scale.
        for suffix, param in CHECKPOINT_TO_PARAM.items():
            name = f"model.layers.0.mlp.experts.{suffix}"
            mapped, shard_id = match_fused_expert_param(
                name, self.mapping, self.params_dict
            )
            self.assertEqual(mapped, f"model.layers.0.mlp.experts.{param}", msg=name)
            self.assertEqual(shard_id, "w13" if param.startswith("w13") else "w2")

    def test_binding_is_one_to_one(self):
        mapped = [
            match_fused_expert_param(
                f"model.layers.0.mlp.experts.{suffix}", self.mapping, self.params_dict
            )[0]
            for suffix in CHECKPOINT_TO_PARAM
        ]
        self.assertEqual(len(set(mapped)), len(mapped))

    def test_unknown_tensor_does_not_bind(self):
        mapped, shard_id = match_fused_expert_param(
            "model.layers.0.mlp.router.weight", self.mapping, self.params_dict
        )
        self.assertIsNone(mapped)
        self.assertIsNone(shard_id)

    def test_longer_names_win_over_their_prefixes(self):
        weight_names = [weight_name for _, weight_name, _ in self.mapping]
        for i, name in enumerate(weight_names):
            for other in weight_names[i + 1 :]:
                self.assertFalse(
                    other.endswith(name),
                    msg=f"{other!r} is shadowed by the earlier entry {name!r}",
                )


class TestFusedPerTensorScaleBroadcast(CustomTestCase):
    def test_scalar_fills_two_dimensional_parameter(self):
        param = torch.nn.Parameter(torch.empty(8, 2), requires_grad=False)
        _load_fused_per_tensor_scale(param, torch.tensor(0.0009765625))
        self.assertTrue(torch.all(param.data == 0.0009765625))

    def test_per_expert_vector_broadcasts_over_the_shard_dim(self):
        param = torch.nn.Parameter(torch.empty(4, 2), requires_grad=False)
        loaded = torch.tensor([1.0, 2.0, 3.0, 4.0])
        _load_fused_per_tensor_scale(param, loaded)
        torch.testing.assert_close(param.data[:, 0], loaded)
        torch.testing.assert_close(param.data[:, 1], loaded)

    def test_scalar_fills_one_dimensional_parameter(self):
        param = torch.nn.Parameter(torch.empty(5), requires_grad=False)
        _load_fused_per_tensor_scale(param, torch.tensor(0.25))
        self.assertTrue(torch.all(param.data == 0.25))


class TestNarrowCheckpointToRank(CustomTestCase):
    """Ranks partition the checkpoint exactly: no gap, no overlap."""

    def test_ranks_tile_the_checkpoint_without_overlap(self):
        # GPT-OSS at TP=2: 2880 intermediate -> 1440 per rank, rounded to 1536
        # for trtllm-gen. Sizing the read by the padded parameter makes rank 0
        # read 3072 rows, i.e. 192 rows that belong to rank 1, and both ranks
        # then compute those experts.
        loaded = torch.arange(5760, dtype=torch.float32).reshape(1, 5760, 1)
        for tp_size in (1, 2, 4, 8):
            seen = []
            for rank in range(tp_size):
                view = _narrow_checkpoint_to_rank(loaded, 1, tp_size, rank, False)
                self.assertEqual(view.shape[1], 5760 // tp_size)
                seen.append(view[0, :, 0])
            torch.testing.assert_close(torch.cat(seen), loaded[0, :, 0])

    def test_presharded_weights_are_returned_whole(self):
        loaded = torch.zeros(1, 2880, 4)
        view = _narrow_checkpoint_to_rank(loaded, 1, 2, 1, True)
        self.assertIs(view, loaded)


class TestLoadW13ShardSlicing(CustomTestCase):
    """End-to-end slice check through _load_w13 with a padded buffer."""

    @staticmethod
    def _stub(tp_size):
        import types

        return types.SimpleNamespace(
            moe_tp_size=tp_size,
            use_padded_loading=True,
            use_presharded_weights=False,
            use_triton_kernels=False,
            moe_runner_config=types.SimpleNamespace(is_gated=True),
            quant_method=types.SimpleNamespace(load_up_proj_weight_first=False),
        )

    def test_each_rank_loads_its_own_rows_and_zeroes_the_padding(self):
        rows, padded_per_rank = 5760, {1: 2944, 2: 1536, 4: 768}
        ckpt = torch.arange(rows, dtype=torch.float32).reshape(1, rows, 1)
        for tp_size, pad in padded_per_rank.items():
            for rank in range(tp_size):
                param = torch.full((1, 2 * pad, 1), -1.0)
                FusedMoE._load_w13(
                    self._stub(tp_size),
                    expert_data=param,
                    shard_dim=1,
                    shard_id="w13",
                    loaded_weight=ckpt.clone(),
                    tp_rank=rank,
                )
                n_real = rows // tp_size
                torch.testing.assert_close(
                    param[0, :n_real, 0],
                    ckpt[0, rank * n_real : (rank + 1) * n_real, 0],
                )
                self.assertTrue(torch.all(param[0, n_real:, 0] == 0))


class TestQuantBlockAlignedPartition(CustomTestCase):
    """A rank boundary that splits a quantization block must be rejected.

    Packed NVFP4 weights hold 2 channels per byte and block scales hold
    group_size, so if a rank's channel count is not a multiple of group_size the
    two slices describe different channels -- and both still divide evenly, so
    nothing raises on its own. GPT-OSS at TP=8 is the case: 2880/8 = 360
    channels, 180 packed bytes (360 channels) but 22 scales (352 channels).
    """

    def _create_weights(self, unpadded_per_rank, tp_size):
        """Run the real create_weights far enough to hit its validation."""
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptNvFp4FusedMoEMethod,
        )

        # __init__ requires Blackwell; the validation under test does not.
        method = ModelOptNvFp4FusedMoEMethod.__new__(ModelOptNvFp4FusedMoEMethod)
        method.quant_config = SimpleNamespace(group_size=16)
        layer = SimpleNamespace(
            intermediate_size_per_partition_unpadded=unpadded_per_rank,
            moe_tp_size=tp_size,
            num_local_experts=128,
            hidden_size_unpadded=2880,
            moe_runner_config=SimpleNamespace(is_gated=True),
        )
        try:
            method.create_weights(
                layer,
                num_experts=128,
                hidden_size=2880,
                intermediate_size_per_partition=round_up(unpadded_per_rank, 128),
                params_dtype=torch.bfloat16,
            )
        except Exception as exc:  # the stub layer stops it after the validation
            return exc
        return None

    def _rejected_for_alignment(self, exc):
        return isinstance(exc, ValueError) and "divisible by" in str(exc)

    def test_unaligned_partition_is_rejected(self):
        exc = self._create_weights(2880 // 8, 8)
        self.assertTrue(
            self._rejected_for_alignment(exc),
            msg=f"TP=8 leaves 360 channels per rank and must be rejected, got {exc!r}",
        )

    def test_block_aligned_partitions_are_accepted(self):
        for tp_size in (1, 2, 4):
            exc = self._create_weights(2880 // tp_size, tp_size)
            self.assertFalse(
                self._rejected_for_alignment(exc),
                msg=f"tp={tp_size} is block-aligned and must pass validation",
            )


class TestNarrowFusedExpertsToEpRank(CustomTestCase):
    """EP ownership covers per-expert scale vectors, not just matrices."""

    @staticmethod
    def _narrow(param_name, tensor, ep_size, ep_rank, redundant=0):
        from sglang.srt.models import gpt_oss

        model = gpt_oss.GptOssForCausalLM.__new__(gpt_oss.GptOssForCausalLM)
        parallel = SimpleNamespace(moe_ep_size=ep_size, moe_ep_rank=ep_rank)
        exec_ctx = SimpleNamespace(
            moe=SimpleNamespace(ep_num_redundant_experts=redundant)
        )
        with (
            mock.patch.object(gpt_oss, "get_parallel", return_value=parallel),
            mock.patch.object(gpt_oss, "get_exec", return_value=exec_ctx),
        ):
            return model._narrow_fused_experts_to_ep_rank(param_name, tensor)

    def test_per_expert_scale_vector_is_sharded(self):
        # A weight_scale_2 stored per expert rather than as one scalar: the
        # parameter is rank-local, so it has to be sliced like the matrices.
        scales = torch.arange(128, dtype=torch.float32)
        for ep_rank in (0, 1):
            got = self._narrow("experts.w13_weight_scale_2", scales, 2, ep_rank)
            torch.testing.assert_close(got, scales[ep_rank * 64 : (ep_rank + 1) * 64])

    def test_singleton_scale_is_kept_whole(self):
        for scale in (torch.tensor(0.5), torch.tensor([0.5])):
            got = self._narrow("experts.w2_weight_scale_2", scale, 2, 1)
            torch.testing.assert_close(got, scale)

    def test_input_scales_stay_global(self):
        scales = torch.arange(128, dtype=torch.float32)
        got = self._narrow("experts.w13_input_scale", scales, 2, 1)
        torch.testing.assert_close(got, scales)

    def test_matrices_are_sharded(self):
        w = torch.arange(128 * 2, dtype=torch.float32).reshape(128, 2)
        got = self._narrow("experts.w2_weight", w, 4, 3)
        torch.testing.assert_close(got, w[96:128])

    def test_sharded_scale_vector_then_loads_into_the_rank_local_param(self):
        # The composition is the failure: an unsharded [128] reaching
        # _load_fused_per_tensor_scale reshapes into a [64, 2] parameter and
        # raises "shape '[64]' is invalid for input of size 128".
        scales = torch.arange(128, dtype=torch.float32)
        param = torch.nn.Parameter(torch.empty(64, 2), requires_grad=False)
        sharded = self._narrow("experts.w13_weight_scale_2", scales, 2, 1)
        _load_fused_per_tensor_scale(param, sharded)
        torch.testing.assert_close(param.data[:, 0], scales[64:])
        torch.testing.assert_close(param.data[:, 1], scales[64:])

    def test_expert_count_must_divide_over_ranks(self):
        w = torch.zeros(127, 2)
        with self.assertRaises(ValueError):
            self._narrow("experts.w2_weight", w, 2, 0)


class TestCopyIntoPaddedExpertData(CustomTestCase):
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
