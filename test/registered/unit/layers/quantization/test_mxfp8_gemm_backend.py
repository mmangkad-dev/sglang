import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.quantization import fp8_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMXFP8GemmBackend(CustomTestCase):
    def setUp(self):
        self.previous_backend = fp8_utils.FP8_GEMM_RUNNER_BACKEND

    def tearDown(self):
        fp8_utils.FP8_GEMM_RUNNER_BACKEND = self.previous_backend

    def _initialize(
        self,
        sm,
        *,
        cli_quantization=None,
        effective_quantization=None,
    ):
        server_args = SimpleNamespace(
            fp8_gemm_runner_backend="auto", quantization=cli_quantization
        )
        with (
            patch.object(fp8_utils, "is_sm100_supported", return_value=sm // 10 == 10),
            patch.object(fp8_utils, "is_sm120_supported", return_value=sm // 10 == 12),
            patch.object(fp8_utils, "is_blackwell_supported", return_value=sm >= 100),
        ):
            fp8_utils.initialize_fp8_gemm_config(
                server_args,
                effective_quantization=effective_quantization,
            )
        return fp8_utils.get_fp8_gemm_runner_backend()

    def test_checkpoint_detected_mxfp8_selects_cutedsl_on_sm100(self):
        backend = self._initialize(100, effective_quantization="mxfp8")
        self.assertIs(backend, fp8_utils.Fp8GemmRunnerBackend.FLASHINFER_CUTEDSL)

    def test_checkpoint_detected_mxfp8_selects_cutlass_on_sm110(self):
        backend = self._initialize(110, effective_quantization="mxfp8")
        self.assertIs(backend, fp8_utils.Fp8GemmRunnerBackend.FLASHINFER_CUTLASS)

    def test_checkpoint_detected_mxfp8_selects_cutlass_on_sm12x(self):
        for sm in (120, 121):
            with self.subTest(sm=sm):
                backend = self._initialize(sm, effective_quantization="mxfp8")
                self.assertIs(
                    backend, fp8_utils.Fp8GemmRunnerBackend.FLASHINFER_CUTLASS
                )

    def test_cli_mxfp8_remains_supported(self):
        backend = self._initialize(103, cli_quantization="mxfp8")
        self.assertIs(backend, fp8_utils.Fp8GemmRunnerBackend.FLASHINFER_CUTEDSL)

    def test_cutedsl_dispatches_flashinfer_mxfp8(self):
        fp8_utils.FP8_GEMM_RUNNER_BACKEND = (
            fp8_utils.Fp8GemmRunnerBackend.FLASHINFER_CUTEDSL
        )
        implementation = fp8_utils.dispatch_w8a8_mxfp8_linear()
        self.assertIs(implementation, fp8_utils.flashinfer_mxfp8_blockscaled_linear)

    def test_cutlass_dispatches_flashinfer_mxfp8(self):
        fp8_utils.FP8_GEMM_RUNNER_BACKEND = (
            fp8_utils.Fp8GemmRunnerBackend.FLASHINFER_CUTLASS
        )
        implementation = fp8_utils.dispatch_w8a8_mxfp8_linear()
        self.assertIs(implementation, fp8_utils.flashinfer_mxfp8_blockscaled_linear)


if __name__ == "__main__":
    unittest.main()
