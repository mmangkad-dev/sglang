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
        co_resident_quantizations=None,
    ):
        server_args = SimpleNamespace(
            fp8_gemm_runner_backend="auto", quantization=cli_quantization
        )
        with (
            patch.object(
                fp8_utils, "get_device_capability", return_value=divmod(sm, 10)
            ),
            patch.object(fp8_utils, "is_sm120_supported", return_value=sm // 10 == 12),
        ):
            fp8_utils.initialize_fp8_gemm_config(
                server_args,
                effective_quantization=effective_quantization,
                co_resident_quantizations=co_resident_quantizations,
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

    def test_unsupported_sm107_keeps_fallback(self):
        backend = self._initialize(107, effective_quantization="mxfp8")
        self.assertIs(backend, fp8_utils.Fp8GemmRunnerBackend.AUTO)

    def test_cli_mxfp8_remains_supported(self):
        backend = self._initialize(103, cli_quantization="mxfp8")
        self.assertIs(backend, fp8_utils.Fp8GemmRunnerBackend.FLASHINFER_CUTEDSL)

    def test_mixed_mxfp8_target_and_fp8_draft_selects_cutlass(self):
        backend = self._initialize(
            100,
            effective_quantization="mxfp8",
            co_resident_quantizations=["fp8"],
        )
        self.assertIs(backend, fp8_utils.Fp8GemmRunnerBackend.FLASHINFER_CUTLASS)

    def test_composite_block_fp8_drafts_select_cutlass(self):
        for draft_quantization in ("w4afp8", "compressed-tensors"):
            with self.subTest(draft_quantization=draft_quantization):
                backend = self._initialize(
                    103,
                    effective_quantization="mxfp8",
                    co_resident_quantizations=[draft_quantization],
                )
                self.assertIs(
                    backend, fp8_utils.Fp8GemmRunnerBackend.FLASHINFER_CUTLASS
                )

    def test_unquantized_draft_keeps_cutedsl(self):
        backend = self._initialize(
            100,
            effective_quantization="mxfp8",
            co_resident_quantizations=[None],
        )
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
