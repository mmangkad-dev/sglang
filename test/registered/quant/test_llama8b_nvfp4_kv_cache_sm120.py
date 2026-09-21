import unittest

from sglang.srt.utils.common import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(
    est_time=300,
    stage="extra-a",
    runner_config="1-gpu-small",
)


@unittest.skipUnless(
    is_sm120_supported(), "requires at least 1 SM120 GPU with CUDA 12.8+"
)
class TestLlama8BNVFP4KVCacheSM120(GSM8KMixin, DefaultServerBase):
    """Llama-3.1-8B-Instruct-NVFP4 with NVFP4 KV cache on SM120."""

    model = "nvidia/Llama-3.1-8B-Instruct-NVFP4"
    # Measured over 13 greedy runs on one RTX 5090: mean 0.62077, stdev 0.00895.
    # 0.59 is 3.4 stdev below the mean; do not re-derive it from a single run.
    gsm8k_accuracy_thres = 0.59
    gsm8k_num_questions = 1319
    gsm8k_num_threads = 200

    other_args = [
        "--quantization",
        "modelopt_fp4",
        "--kv-cache-dtype",
        "nvfp4",
        "--prefill-attention-backend",
        "flashinfer",
        "--decode-attention-backend",
        "trtllm_mha",
    ]


if __name__ == "__main__":
    unittest.main()
