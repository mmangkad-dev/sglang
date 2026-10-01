"""Real-backend regressions for FlashInfer BF16 linear dispatch."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization import unquant
from sglang.srt.models.kimi_k3 import _k3_bf16_gemm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "Blackwell required",
)
class TestFlashInferBf16Gemm(CustomTestCase):
    def setUp(self):
        torch.manual_seed(1)
        backend = patch.object(
            unquant, "_BF16_GEMM_BACKEND", unquant.Bf16GemmBackend.CUTEDSL
        )
        backend.start()
        self.addCleanup(backend.stop)
        splitk = patch.object(unquant, "_enable_bf16_splitk_gemm", False)
        splitk.start()
        self.addCleanup(splitk.stop)

    @torch.inference_mode()
    def test_runtime_alignment_and_empty_batches(self):
        # K=510 previously reached TGV's unguarded TMA path and silently
        # corrupted almost every output element. K=512 is the aligned control.
        from flashinfer.autotuner import autotune

        for k in (510, 512):
            for has_bias in (False, True):
                layer = ReplicatedLinear(
                    k, 1024, bias=has_bias, params_dtype=torch.bfloat16
                ).cuda()
                layer.weight.normal_()
                if has_bias:
                    layer.bias.normal_()
                bias = layer.bias if has_bias else None
                for m in (0, 4):
                    with self.subTest(k=k, bias=has_bias, m=m):
                        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                        ref = x.float() @ layer.weight.float().t()
                        if bias is not None:
                            ref += bias.float()
                        for tune in (False, True):
                            with autotune(tune):
                                output = layer.quant_method.apply(layer, x, bias)
                                out = torch.empty_like(output)
                                self.assertIs(
                                    layer.quant_method.apply_into(layer, x, out, bias),
                                    out,
                                )
                            torch.testing.assert_close(
                                output, ref.bfloat16(), rtol=2e-2, atol=0.5
                            )
                            torch.testing.assert_close(
                                out, ref.bfloat16(), rtol=2e-2, atol=0.5
                            )

    @torch.inference_mode()
    def test_k3_output_dtype_and_graph_replay(self):
        from flashinfer.autotuner import autotune

        for k in (510, 512):
            weight = torch.randn(1024, k, device="cuda", dtype=torch.bfloat16)
            for dtype in (torch.bfloat16, torch.float32):
                for m in (0, 4):
                    with self.subTest(k=k, dtype=dtype, m=m):
                        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                        out = torch.empty(m, 1024, device="cuda", dtype=dtype)
                        with autotune(True):
                            self.assertIs(_k3_bf16_gemm(x, weight, out=out), out)
                        allocated = _k3_bf16_gemm(x, weight, out_dtype=dtype)
                        self.assertEqual(allocated.dtype, dtype)
                        torch.testing.assert_close(allocated, out)
                        if m:
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                _k3_bf16_gemm(x, weight, out=out)
                            x.normal_()
                            out.fill_(float("nan"))
                            graph.replay()
                        ref = (x.double() @ weight.double().t()).to(dtype)
                        torch.testing.assert_close(
                            out,
                            ref,
                            rtol=2e-2 if dtype == torch.bfloat16 else 1e-4,
                            atol=0.5 if dtype == torch.bfloat16 else 1e-3,
                        )


if __name__ == "__main__":
    unittest.main()
