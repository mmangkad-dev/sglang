"""B200 GPT-OSS MXFP4 coverage for DeepGEMM with DeepEP AUTO dispatch.

The evaluation performs prompt prefill through DeepEP normal dispatch and
generation through DeepEP low-latency dispatch, including CUDA Graph decode.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.gpt_oss_common import BaseTestGptOss

register_cuda_ci(est_time=300, stage="base-c", runner_config="deepep-4-gpu-b200")


class TestGptOss4GpuMxfp4DeepEP(BaseTestGptOss):
    def test_mxfp4_120b_deepep_auto(self):
        self.run_test(
            model_variant="120b",
            quantization="mxfp4",
            expected_score_of_reasoning_effort={
                "low": 0.58,
            },
            other_args=[
                "--tp",
                "4",
                "--enable-prefill-cp",
                "--attn-cp-size",
                "4",
                "--cp-strategy",
                "zigzag",
                "--ep",
                "4",
                "--moe-runner-backend",
                "deep_gemm",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "auto",
                "--deepep-dispatcher-output-dtype",
                "fp8",
                "--cuda-graph-max-bs-decode",
                "200",
            ],
        )


if __name__ == "__main__":
    unittest.main()
