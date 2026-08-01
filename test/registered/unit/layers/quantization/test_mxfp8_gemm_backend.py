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
            patch.object(fp8_utils, "is_flashinfer_available", return_value=True),
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

    def test_cutedsl_dispatch_uses_exact_sm_contract(self):
        fp8_utils.FP8_GEMM_RUNNER_BACKEND = (
            fp8_utils.Fp8GemmRunnerBackend.FLASHINFER_CUTEDSL
        )
        with (
            patch.object(fp8_utils, "get_device_capability", return_value=(10, 3)),
            patch.object(fp8_utils, "is_flashinfer_available", return_value=True),
        ):
            implementation = fp8_utils.dispatch_w8a8_mxfp8_linear()
        self.assertIs(implementation, fp8_utils.flashinfer_mxfp8_blockscaled_linear)

        with (
            patch.object(fp8_utils, "get_device_capability", return_value=(10, 7)),
            patch.object(fp8_utils, "is_flashinfer_available", return_value=True),
            self.assertRaisesRegex(RuntimeError, "SM100/SM103"),
        ):
            fp8_utils.dispatch_w8a8_mxfp8_linear()

    def test_cutedsl_rejects_non_mxfp8_dispatch(self):
        with self.assertRaisesRegex(RuntimeError, "supports MXFP8 GEMM only"):
            fp8_utils._dispatch_explicit_backend(
                fp8_utils.Fp8GemmRunnerBackend.FLASHINFER_CUTEDSL
            )

    def test_cutlass_dispatch_uses_flashinfer_sm_contract(self):
        fp8_utils.FP8_GEMM_RUNNER_BACKEND = (
            fp8_utils.Fp8GemmRunnerBackend.FLASHINFER_CUTLASS
        )
        with (
            patch.object(fp8_utils, "get_device_capability", return_value=(11, 0)),
            patch.object(fp8_utils, "is_flashinfer_available", return_value=True),
        ):
            implementation = fp8_utils.dispatch_w8a8_mxfp8_linear()
        self.assertIs(implementation, fp8_utils.flashinfer_mxfp8_blockscaled_linear)

        with (
            patch.object(fp8_utils, "get_device_capability", return_value=(10, 7)),
            patch.object(fp8_utils, "is_flashinfer_available", return_value=True),
            self.assertRaisesRegex(RuntimeError, "SM100/SM103/SM110/SM120/SM121"),
        ):
            fp8_utils.dispatch_w8a8_mxfp8_linear()

    def test_one_batch_uses_checkpoint_detected_quantization(self):
        import sglang.benchmark.one_batch as one_batch

        server_args = SimpleNamespace()
        model_config = SimpleNamespace(quantization="mxfp8")
        with (
            patch.object(
                one_batch.ModelConfig,
                "from_server_args",
                return_value=model_config,
            ),
            patch.object(one_batch, "initialize_moe_config"),
            patch.object(one_batch, "initialize_fp8_gemm_config") as init_fp8,
            patch.object(one_batch, "initialize_fp4_gemm_config"),
        ):
            resolved = one_batch.initialize_latency_test_gemm_configs(server_args)

        self.assertIs(resolved, model_config)
        init_fp8.assert_called_once_with(server_args, effective_quantization="mxfp8")


if __name__ == "__main__":
    unittest.main()
