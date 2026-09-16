"""CPU unit tests for which MoE backend an NVFP4 fused-MoE method obeys.

ModelOptNvFp4FusedMoEMethod allocates parameters, preps weights, builds its
MoeRunner and dispatches kernels for one backend. That backend is the one the
layer was built for, not whatever the process-wide setting happens to be at the
time: the speculative contexts swap the MoE and A2A backends to the draft's
around draft work that also runs the target's layers, so a method that re-reads
them there picks a kernel its weights were never prepared for.

These cases stay on CPU; the kernels are covered on-device in
test_nvfp4_moe_backends.py.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

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
DRAFT_RUNNER_BACKEND = MoeRunnerBackend.TRITON
DRAFT_A2A_BACKEND = MoeA2ABackend.NONE


def _method(runner_backend: MoeRunnerBackend, a2a_backend: MoeA2ABackend):
    with get_flags().moe.override(
        runner_backend=runner_backend,
        a2a_backend=a2a_backend,
        speculative_runner_backend=DRAFT_RUNNER_BACKEND,
        speculative_a2a_backend=DRAFT_A2A_BACKEND,
    ):
        return ModelOptNvFp4FusedMoEMethod(
            ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16)
        )


class TestNvFp4MoeRunnerBackendOwnership(CustomTestCase):
    def setUp(self):
        platform = override_platform(is_cuda=True, is_blackwell=True)
        platform.install()
        self.addCleanup(platform.restore)

    def test_cutlass_survives_the_speculative_swap(self):
        method = _method(MoeRunnerBackend.FLASHINFER_CUTLASS, MoeA2ABackend.NONE)

        with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.assertTrue(method.enable_flashinfer_cutlass_moe)
            self.assertFalse(method.enable_flashinfer_trtllm_moe)
            self.assertFalse(method.enable_flashinfer_cutedsl_moe)
            self.assertEqual(
                method.moe_runner_backend, MoeRunnerBackend.FLASHINFER_CUTLASS
            )

    def test_trtllm_routed_survives_the_speculative_swap(self):
        method = _method(MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED, MoeA2ABackend.NONE)

        with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.assertTrue(method.enable_flashinfer_trtllm_moe)

    def test_cutedsl_variant_survives_the_speculative_swap(self):
        """The v1/v2 answer picks the weight layout at load time and the kernel
        at forward time; reading the live A2A backend flips it between them."""
        method = _method(MoeRunnerBackend.FLASHINFER_CUTEDSL, MoeA2ABackend.DEEPEP)

        with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.assertTrue(method._is_cutedsl_v1_deepep)
            self.assertFalse(method._is_cutedsl_v2_standard)

    def test_cutedsl_standard_variant_is_unchanged(self):
        method = _method(MoeRunnerBackend.FLASHINFER_CUTEDSL, MoeA2ABackend.NONE)

        with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.assertTrue(method._is_cutedsl_v2_standard)
            self.assertFalse(method._is_cutedsl_v1_deepep)

    def test_auto_is_rejected_rather_than_resolved_here(self):
        """`auto` must be resolved before the layers are built: FusedMoE reads
        the same setting for the w1/w3 shard swap, so a backend picked only
        here loads the experts with gate and up exchanged."""
        with self.assertRaisesRegex(ValueError, "explicit --moe-runner-backend"):
            _method(MoeRunnerBackend.AUTO, MoeA2ABackend.NONE)


if __name__ == "__main__":
    unittest.main()
