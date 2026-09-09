"""Unit tests for the shared FP8 post-load pipeline - CPU-only, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization import fp8_postload
from sglang.srt.layers.quantization.fp8_postload import (
    process_fp8_linear_after_loading,
)
from sglang.test.test_utils import CustomTestCase

N, K = 8, 16
SHARDS = [3, 5]

FP8_MIN = torch.finfo(torch.float8_e4m3fn).min


def make_layer(*, per_channel_scale, with_input_scale=False):
    """Stand-in for the LinearBase the weight loader hands to post-load."""
    layer = torch.nn.Module()
    weight = (torch.arange(N * K, dtype=torch.float32).reshape(N, K) / 97.0).to(
        torch.float8_e4m3fn
    )
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    if per_channel_scale:
        scale = torch.linspace(0.1, 0.4, N, dtype=torch.float32).reshape(N, 1)
    else:
        # One scale per shard, the trailing one left at the sentinel that means
        # "fused in the checkpoint", which is what skips the requantize loop.
        scale = torch.tensor([0.25, FP8_MIN], dtype=torch.float32)
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    layer.logical_widths = SHARDS
    if with_input_scale:
        layer.input_scale = torch.nn.Parameter(
            torch.tensor([0.25, 0.5]), requires_grad=False
        )
    return layer


class TestChannelGranularity(CustomTestCase):
    def test_scales_pass_through_and_weight_is_transposed(self):
        layer = make_layer(per_channel_scale=True)
        before = layer.weight_scale.detach().clone()
        process_fp8_linear_after_loading(layer, granularity="channel")
        self.assertEqual(tuple(layer.weight.shape), (K, N))
        torch.testing.assert_close(layer.weight_scale, before)

    def test_committed_as_non_trainable_parameters(self):
        # torch.compile requires these to be torch.nn.Parameter.
        layer = make_layer(per_channel_scale=True)
        process_fp8_linear_after_loading(layer, granularity="channel")
        for name in ("weight", "weight_scale"):
            with self.subTest(param=name):
                value = getattr(layer, name)
                self.assertIsInstance(value, torch.nn.Parameter)
                self.assertFalse(value.requires_grad)

    def test_aiter_shuffle_keeps_n_by_k(self):
        """The shuffled layout is what the kernel indexes, so it is not transposed."""
        layer = make_layer(per_channel_scale=True)
        shuffle = mock.Mock(side_effect=lambda w, layout: w.contiguous())
        aiter_shuffle = type(torch)("aiter.ops.shuffle")
        aiter_shuffle.shuffle_weight = shuffle
        modules = {
            "aiter": type(torch)("aiter"),
            "aiter.ops": type(torch)("aiter.ops"),
            "aiter.ops.shuffle": aiter_shuffle,
        }
        with mock.patch.dict("sys.modules", modules):
            process_fp8_linear_after_loading(
                layer, granularity="channel", aiter_shuffle=True
            )
        self.assertEqual(tuple(layer.weight.shape), (N, K))
        self.assertEqual(shuffle.call_args.args[1], (16, 16))


class TestTensorGranularity(CustomTestCase):
    def test_requantized_to_one_max_scale(self):
        layer = make_layer(per_channel_scale=False)
        process_fp8_linear_after_loading(layer, granularity="tensor")
        self.assertEqual(layer.weight_scale.numel(), 1)
        self.assertAlmostEqual(layer.weight_scale.item(), 0.25, places=6)
        self.assertEqual(tuple(layer.weight.shape), (K, N))

    def test_logical_widths_reach_the_requantizer(self):
        """The shard widths decide which rows get rescaled, so they must arrive."""
        layer = make_layer(per_channel_scale=False)
        recorder = mock.Mock(
            return_value=(layer.weight_scale.max(), layer.weight.detach())
        )
        with mock.patch.object(fp8_postload, "requantize_with_max_scale", recorder):
            process_fp8_linear_after_loading(layer, granularity="tensor")
        self.assertEqual(recorder.call_args.kwargs["logical_widths"], SHARDS)


class TestRocmNormalization(CustomTestCase):
    """On ROCm the checkpoint's e4m3fn weights are reinterpreted as e4m3fnuz.

    Same bits, half the value, so the scales double. Missing either half of that
    silently halves or doubles every dequantized weight.
    """

    def _run(self, **kwargs):
        with mock.patch.object(fp8_postload, "is_fp8_fnuz", return_value=True):
            process_fp8_linear_after_loading(**kwargs)

    def test_channel_doubles_the_weight_scale(self):
        layer = make_layer(per_channel_scale=True)
        before = layer.weight_scale.detach().clone()
        self._run(layer=layer, granularity="channel")
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fnuz)
        torch.testing.assert_close(layer.weight_scale, before * 2.0)

    def test_tensor_doubles_the_weight_scale(self):
        layer = make_layer(per_channel_scale=False)
        self._run(layer=layer, granularity="tensor")
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fnuz)
        self.assertAlmostEqual(layer.weight_scale.item(), 0.5, places=6)

    def test_input_scale_is_rescaled_with_the_weight(self):
        for granularity, per_channel in (("channel", True), ("tensor", False)):
            with self.subTest(granularity=granularity):
                layer = make_layer(per_channel_scale=per_channel, with_input_scale=True)
                before = layer.input_scale.detach().clone()
                self._run(layer=layer, granularity=granularity)
                torch.testing.assert_close(layer.input_scale, before * 2.0)
                self.assertIsInstance(layer.input_scale, torch.nn.Parameter)
                self.assertFalse(layer.input_scale.requires_grad)

    def test_a_layer_without_an_input_scale_does_not_gain_one(self):
        layer = make_layer(per_channel_scale=True)
        self._run(layer=layer, granularity="channel")
        self.assertFalse(hasattr(layer, "input_scale"))


class TestNoNormalizationOffRocm(CustomTestCase):
    def test_weight_dtype_and_scales_are_left_alone(self):
        layer = make_layer(per_channel_scale=True, with_input_scale=True)
        before_scale = layer.weight_scale.detach().clone()
        before_input = layer.input_scale.detach().clone()
        with mock.patch.object(fp8_postload, "is_fp8_fnuz", return_value=False):
            process_fp8_linear_after_loading(layer, granularity="channel")
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        torch.testing.assert_close(layer.weight_scale, before_scale)
        torch.testing.assert_close(layer.input_scale, before_input)


class TestUnknownGranularity(CustomTestCase):
    def test_raises_and_names_the_value(self):
        layer = make_layer(per_channel_scale=True)
        with self.assertRaises(ValueError) as caught:
            process_fp8_linear_after_loading(layer, granularity="per_group")
        self.assertIn("per_group", str(caught.exception))


class TestCallersRouteThroughTheHelper(CustomTestCase):
    """The two migrated post-load paths must reach the shared pipeline.

    A scheme that quietly keeps its own copy still works, and stops tracking
    the shared one. Both callers import the helper by name, so the patch has to
    land in the calling module rather than in fp8_postload.
    """

    def test_compressed_tensors_w8a8_fp8(self):
        from compressed_tensors.quantization import QuantizationStrategy

        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_w8a8_fp8 as ct_module,
        )

        cases = {
            QuantizationStrategy.TENSOR: ("tensor", False),
            QuantizationStrategy.CHANNEL: ("channel", True),
        }
        for strategy, (granularity, per_channel) in cases.items():
            with self.subTest(strategy=strategy):
                scheme = object.__new__(ct_module.CompressedTensorsW8A8Fp8)
                scheme.strategy = strategy
                scheme.is_static_input_scheme = False
                layer = make_layer(per_channel_scale=per_channel)
                with mock.patch.object(
                    ct_module, "process_fp8_linear_after_loading"
                ) as helper:
                    scheme.process_weights_after_loading(layer)
                helper.assert_called_once()
                self.assertEqual(helper.call_args.kwargs["granularity"], granularity)

    def test_compressed_tensors_channel_forwards_the_aiter_decision(self):
        from compressed_tensors.quantization import QuantizationStrategy

        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_w8a8_fp8 as ct_module,
        )

        for use_aiter in (False, True):
            with self.subTest(use_aiter=use_aiter):
                scheme = object.__new__(ct_module.CompressedTensorsW8A8Fp8)
                scheme.strategy = QuantizationStrategy.CHANNEL
                scheme.is_static_input_scheme = False
                layer = make_layer(per_channel_scale=True)
                with (
                    mock.patch.object(ct_module, "_use_aiter", use_aiter),
                    (
                        mock.patch.object(ct_module, "process_fp8_linear_after_loading")
                    ) as helper,
                ):
                    scheme.process_weights_after_loading(layer)
                self.assertEqual(helper.call_args.kwargs["aiter_shuffle"], use_aiter)

    def test_w8a8_fp8_serialized_checkpoint(self):
        from sglang.srt.layers.quantization import w8a8_fp8 as w8a8_module

        method = object.__new__(w8a8_module.W8A8Fp8LinearMethod)
        method.cutlass_fp8_supported = False
        method.quantization_config = mock.Mock(is_checkpoint_fp8_serialized=True)
        layer = make_layer(per_channel_scale=True)
        with mock.patch.object(
            w8a8_module, "process_fp8_linear_after_loading"
        ) as helper:
            method.process_weights_after_loading(layer)
        helper.assert_called_once()
        self.assertEqual(helper.call_args.kwargs["granularity"], "channel")

    def test_w8a8_fp8_online_quantization_is_untouched(self):
        """The non-serialized branch quantizes from bf16 and must not route here."""
        from sglang.srt.layers.quantization import w8a8_fp8 as w8a8_module

        method = object.__new__(w8a8_module.W8A8Fp8LinearMethod)
        method.cutlass_fp8_supported = False
        method.quantization_config = mock.Mock(is_checkpoint_fp8_serialized=False)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.randn(N, K, dtype=torch.float32), requires_grad=False
        )
        layer.logical_widths = SHARDS
        with mock.patch.object(
            w8a8_module, "process_fp8_linear_after_loading"
        ) as helper:
            method.process_weights_after_loading(layer)
        helper.assert_not_called()
        self.assertIsNone(layer.input_scale)


if __name__ == "__main__":
    unittest.main()
