"""An NVFP4 fused-MoE method must keep obeying the backend its layer was built
for while the speculative contexts hold the draft's backends, and must reject an
unresolved `auto` rather than answer it alone. Kernels are covered on-device in
test_nvfp4_moe_backends.py."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=17, suite="base-a-test-cpu")

import unittest

from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    MoeRunnerBackend,
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.runtime_context import get_flags, override_platform
from sglang.test.test_utils import CustomTestCase

# What the draft runs while the target's layers are still live.
SPECULATIVE_RUNNER_BACKEND = MoeRunnerBackend.TRITON
SPECULATIVE_A2A_BACKEND = MoeA2ABackend.NONE


def _method(*, runner_backend: MoeRunnerBackend, a2a_backend: MoeA2ABackend):
    with get_flags().moe.override(
        runner_backend=runner_backend,
        a2a_backend=a2a_backend,
        speculative_runner_backend=SPECULATIVE_RUNNER_BACKEND,
        speculative_a2a_backend=SPECULATIVE_A2A_BACKEND,
    ):
        return ModelOptNvFp4FusedMoEMethod(
            ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16)
        )


class TestNvFp4MoeRunnerBackendOwnership(CustomTestCase):
    def setUp(self):
        self._platform(is_cuda=True, is_blackwell=True, device_capability=(10, 0))

    def _platform(self, **facts):
        platform = override_platform(**facts)
        platform.install()
        self.addCleanup(platform.restore)

    def test_cutlass_survives_the_speculative_swap(self):
        method = _method(
            runner_backend=MoeRunnerBackend.FLASHINFER_CUTLASS,
            a2a_backend=MoeA2ABackend.NONE,
        )

        with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.assertTrue(method.enable_flashinfer_cutlass_moe)
            self.assertFalse(method.enable_flashinfer_trtllm_moe)
            self.assertFalse(method.enable_flashinfer_cutedsl_moe)
            self.assertEqual(
                method.moe_runner_backend, MoeRunnerBackend.FLASHINFER_CUTLASS
            )

    def test_trtllm_routed_survives_the_speculative_swap(self):
        method = _method(
            runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
            a2a_backend=MoeA2ABackend.NONE,
        )

        with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.assertTrue(method.enable_flashinfer_trtllm_moe)

    def test_cutedsl_variant_survives_the_speculative_swap(self):
        """The v1/v2 answer picks the weight layout at load time and the kernel
        at forward time; a live A2A read flips it between the two."""
        method = _method(
            runner_backend=MoeRunnerBackend.FLASHINFER_CUTEDSL,
            a2a_backend=MoeA2ABackend.DEEPEP,
        )

        with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.assertTrue(method._is_cutedsl_v1_deepep)
            self.assertFalse(method._is_cutedsl_v2_standard)

    def test_cutedsl_standard_variant_is_unchanged(self):
        method = _method(
            runner_backend=MoeRunnerBackend.FLASHINFER_CUTEDSL,
            a2a_backend=MoeA2ABackend.NONE,
        )

        with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.assertTrue(method._is_cutedsl_v2_standard)
            self.assertFalse(method._is_cutedsl_v1_deepep)

    def test_auto_is_rejected_rather_than_resolved_here(self):
        """FusedMoE reads the same setting for the w1/w3 shard swap, so a
        backend answered only here loads gate and up exchanged."""
        with self.assertRaisesRegex(ValueError, "--moe-runner-backend"):
            _method(
                runner_backend=MoeRunnerBackend.AUTO, a2a_backend=MoeA2ABackend.NONE
            )

    def test_auto_still_takes_the_marlin_fallback_before_blackwell(self):
        """Marlin is the one answer this method may give itself: no FusedMoE
        switch keys off it, and NVFP4 has no other pre-Blackwell path."""
        self._platform(is_cuda=True, is_blackwell=False, device_capability=(9, 0))

        method = _method(
            runner_backend=MoeRunnerBackend.AUTO, a2a_backend=MoeA2ABackend.NONE
        )

        self.assertEqual(method.moe_runner_backend, MoeRunnerBackend.MARLIN)


if __name__ == "__main__":
    unittest.main()
