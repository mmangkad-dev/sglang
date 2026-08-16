"""Regression test for MXFP4 Triton-kernel sharded state exports."""

import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import safe_open

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

maybe_stub_sgl_kernel()

import sglang.srt.layers.quantization.mxfp4 as mxfp4_module  # noqa: E402
import sglang.srt.model_loader.loader as loader_module  # noqa: E402
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod  # noqa: E402
from sglang.srt.model_loader.loader import ShardedStateLoader  # noqa: E402
from sglang.srt.utils.offloader import OffloaderV1  # noqa: E402

# Weights are rebound onto the Triton wrapper storage; scales are not, since
# repointing them triggers an illegal address in the mxfp4 matmul (see
# mxfp4.py), so the swizzled scales are published as separate buffers.
_RUNTIME_WEIGHT_NAMES = ("w13_weight", "w2_weight")
_RUNTIME_SCALE_NAMES = ("w13_weight_scale_triton", "w2_weight_scale_triton")
_RUNTIME_STATE_NAMES = _RUNTIME_WEIGHT_NAMES + _RUNTIME_SCALE_NAMES
_PRESWIZZLE_SCALE_NAMES = ("w13_weight_scale", "w2_weight_scale")


def _new_method():
    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    method.use_marlin = False
    method.use_deep_gemm = False
    method.use_mega_moe = False
    method._fi_kernel = None
    method.use_flashinfer = False
    method.use_triton_kernels = True
    return method


class _TinyMxfp4Layer(torch.nn.Module):
    def __init__(self, method, initialize_to_zero=False):
        super().__init__()
        self.num_local_experts = 1
        method.create_weights(
            self,
            num_experts=1,
            hidden_size=64,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
        )
        for name, value in (
            ("w13_weight", 17),
            ("w13_weight_scale", 113),
            ("w2_weight", 18),
            ("w2_weight_scale", 114),
            ("w13_weight_bias", 3),
            ("w2_weight_bias", 4),
        ):
            getattr(self, name).data.fill_(0 if initialize_to_zero else value)


class _WrappedTensor:
    def __init__(self, data):
        self.storage = SimpleNamespace(data=data)

    @property
    def data(self):
        return self.storage.data


def _fake_swizzle(weight, scale, _num_warps):
    def wrap(tensor):
        data = tensor.detach().clone().transpose(-2, -1)
        return _WrappedTensor(data)

    return wrap(weight), object(), wrap(scale)


def _runtime_tensors(method):
    return {
        "w13_weight": method.w13_weight_triton_tensor,
        "w2_weight": method.w2_weight_triton_tensor,
        "w13_weight_scale_triton": method.w13_precision_config.b_mx_scale,
        "w2_weight_scale_triton": method.w2_precision_config.b_mx_scale,
    }


def _storage_offsets(tensor):
    """Linear storage offset of every logical element of `tensor`."""
    grids = torch.meshgrid(
        *[torch.arange(size, device=tensor.device) for size in tensor.shape],
        indexing="ij",
    )
    offsets = torch.full_like(grids[0], tensor.storage_offset())
    for grid, stride in zip(grids, tensor.stride()):
        offsets = offsets + grid * stride
    return offsets.reshape(-1)


class TestMxfp4SwizzledStorageCoverage(CustomTestCase):
    """Pin the layout property the save/load round trip depends on.

    `.contiguous()` + `copy_` only move logically reachable elements, so the
    round trip is byte-exact only while every storage byte is addressed by
    exactly one logical index. That is triton_kernels' behavior, not ours, so a
    layout bump must fail here instead of as a silent accuracy drop.
    """

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_real_swizzle_storage_is_fully_addressable(self):
        try:
            import triton_kernels  # noqa: F401
        except ImportError:
            self.skipTest("triton_kernels is not installed")

        from sglang.srt.layers.quantization.mxfp4 import _swizzle_mxfp4

        num_experts, out_features, in_features = 2, 256, 256
        weight = torch.randint(
            0,
            255,
            (num_experts, out_features, in_features // 2),
            dtype=torch.uint8,
            device="cuda",
        )
        scale = torch.randint(
            0,
            255,
            (num_experts, out_features, in_features // 32),
            dtype=torch.uint8,
            device="cuda",
        )

        swizzled_weight, _, swizzled_scale = _swizzle_mxfp4(weight, scale, 8)

        for name, wrapper in (
            ("weight", swizzled_weight),
            ("scale", swizzled_scale),
        ):
            tensor = wrapper.storage.data
            self.assertEqual(
                tensor.numel() * tensor.element_size(),
                tensor.untyped_storage().nbytes(),
                f"{name}: unreachable storage bytes, which contiguous()/copy_ "
                f"would silently drop on reload",
            )
            offsets = _storage_offsets(tensor)
            self.assertEqual(
                offsets.unique().numel(),
                tensor.numel(),
                f"{name}: swizzled layout aliases storage bytes",
            )


class TestMxfp4ShardedState(CustomTestCase):
    def test_runtime_parameters_are_not_cpu_offloaded(self):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.empty(1, device="meta"), requires_grad=False
        )
        layer.weight._sglang_keep_on_device = True

        offloader = OffloaderV1(cpu_offload_max_bytes=layer.weight.nbytes)
        self.assertIs(offloader.maybe_offload_to_cpu(layer), layer)
        self.assertEqual(layer.weight.device.type, "meta")
        self.assertEqual(offloader._cpu_offload_bytes, 0)

    def test_runtime_weight_and_scale_sharded_round_trip(self):
        triton_kernels = ModuleType("triton_kernels")
        triton_kernels.__path__ = []
        triton_matmul = ModuleType("triton_kernels.matmul")
        triton_matmul.FlexCtx = SimpleNamespace
        triton_matmul.PrecisionConfig = SimpleNamespace

        source_method = _new_method()
        destination_method = _new_method()
        source = _TinyMxfp4Layer(source_method)
        destination = _TinyMxfp4Layer(destination_method, initialize_to_zero=True)
        source_original_parameters = {
            name: getattr(source, name)
            for name in _RUNTIME_WEIGHT_NAMES + _PRESWIZZLE_SCALE_NAMES
        }
        destination_original_parameters = {
            name: getattr(destination, name)
            for name in _RUNTIME_WEIGHT_NAMES + _PRESWIZZLE_SCALE_NAMES
        }

        with patch.dict(
            sys.modules,
            {
                "triton_kernels": triton_kernels,
                "triton_kernels.matmul": triton_matmul,
            },
        ), patch.object(
            mxfp4_module, "_swizzle_mxfp4", side_effect=_fake_swizzle
        ), patch.object(
            mxfp4_module, "_use_aiter", False
        ), patch.object(
            torch.cuda, "empty_cache"
        ):
            source_method.process_weights_after_loading(source)
            destination_method.process_weights_after_loading(destination)

        source_runtime_tensors = _runtime_tensors(source_method)
        destination_runtime_tensors = _runtime_tensors(destination_method)
        source_state = source.state_dict()
        destination_state = destination.state_dict()
        for layer, original_parameters, runtime_tensors, state in (
            (
                source,
                source_original_parameters,
                source_runtime_tensors,
                source_state,
            ),
            (
                destination,
                destination_original_parameters,
                destination_runtime_tensors,
                destination_state,
            ),
        ):
            # Everything the kernels read is exported, and is the same storage
            # they read, so a sharded load lands in it.
            for name in _RUNTIME_STATE_NAMES:
                self.assertIn(name, state)
                entry = getattr(layer, name)
                runtime_tensor = runtime_tensors[name]
                self.assertFalse(entry.is_contiguous())
                self.assertEqual(
                    runtime_tensor.storage.data.data_ptr(), entry.data_ptr()
                )
                self.assertEqual(runtime_tensor.storage.data.stride(), entry.stride())

            # Rebound in place, so identity and weight-loader attrs survive.
            for name in _RUNTIME_WEIGHT_NAMES:
                parameter = getattr(layer, name)
                self.assertIs(parameter, original_parameters[name])
                self.assertTrue(parameter._sglang_keep_on_device)

            # Left completely alone: repointing these is what faults.
            runtime_scale_ptrs = {
                runtime_tensors[name].storage.data.data_ptr()
                for name in _RUNTIME_SCALE_NAMES
            }
            for name in _PRESWIZZLE_SCALE_NAMES:
                parameter = getattr(layer, name)
                self.assertIs(parameter, original_parameters[name])
                self.assertTrue(parameter.is_contiguous())
                self.assertNotIn(parameter.data_ptr(), runtime_scale_ptrs)

        for name in _RUNTIME_STATE_NAMES:
            self.assertFalse(torch.equal(source_state[name], destination_state[name]))

        with tempfile.TemporaryDirectory() as output_dir, patch.object(
            loader_module, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
        ):
            ShardedStateLoader.save_model(source, output_dir)
            checkpoint = f"{output_dir}/model-rank-0-part-0.safetensors"
            with safe_open(checkpoint, framework="pt") as handle:
                self.assertEqual(set(handle.keys()), set(source_state))
                missing = destination_state.copy()
                for name in handle.keys():  # noqa: SIM118
                    saved_tensor = handle.get_tensor(name)
                    self.assertTrue(torch.equal(saved_tensor, source_state[name]), name)
                    # This is the copy performed by ShardedStateLoader.load_model.
                    missing.pop(name).copy_(saved_tensor)
                self.assertFalse(missing)

        for name in source_state:
            self.assertTrue(
                torch.equal(destination_state[name], source_state[name]),
                name,
            )
        for name in _RUNTIME_STATE_NAMES:
            destination_entry = getattr(destination, name)
            destination_runtime_tensor = destination_runtime_tensors[name]
            self.assertFalse(destination_entry.is_contiguous())
            self.assertEqual(
                destination_runtime_tensor.storage.data.data_ptr(),
                destination_entry.data_ptr(),
            )
            self.assertEqual(
                destination_runtime_tensor.storage.data.stride(),
                destination_entry.stride(),
            )
            # The load reached the storage the kernels actually read.
            self.assertTrue(
                torch.equal(
                    destination_runtime_tensor.storage.data,
                    source_runtime_tensors[name].storage.data,
                ),
                name,
            )


if __name__ == "__main__":
    unittest.main()
