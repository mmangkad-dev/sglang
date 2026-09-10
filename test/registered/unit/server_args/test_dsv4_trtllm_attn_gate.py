# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""`--dsv4-attn-backend trtllm` rejects what the uniform-FP8 path cannot serve.

Two of these gates stand between a launch and silent KV corruption: PD
disaggregation would let a peer read the 584-byte packed layout out of the
512-byte uniform-FP8 pool (the handshake compares only kv_cache_dtype), and
disabled chunked prefill leaves the trtllm-gen counter buffer with no finite
row bound. They must raise, and they must raise as ValueError so `python -O`
cannot strip them.

    python -m pytest test/registered/unit/server_args/test_dsv4_trtllm_attn_gate.py -v
"""

import unittest
import unittest.mock
from types import SimpleNamespace

from sglang.srt.arg_groups.deepseek_v4_hook import _validate_deepseek_v4_trtllm_attn
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SUPPORTED = {
    "device": "cuda",
    "kv_cache_dtype": "fp8_e4m3",
    "enable_hisparse": False,
    "attn_cp_size": 1,
    "dcp_size": 1,
    "enable_prefill_cp": False,
    "disaggregation_mode": "null",
    "chunked_prefill_size": 4096,
}


def _validate(**overrides):
    """Run the gate over a supported config with `overrides` applied."""
    cfg = SimpleNamespace(**{**_SUPPORTED, **overrides})
    with unittest.mock.patch(
        "sglang.srt.utils.common.is_sm100_supported", return_value=True
    ):
        _validate_deepseek_v4_trtllm_attn(cfg)


class TestDSV4TrtllmAttnGate(unittest.TestCase):
    def test_supported_config_passes(self):
        """An inverted condition here would reject every trtllm launch."""
        _validate()

    def test_pd_disaggregation_is_refused(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError) as ctx:
                    _validate(disaggregation_mode=mode)
                self.assertIn("PD disaggregation", str(ctx.exception))

    def test_disabled_chunked_prefill_is_refused(self):
        for size in (None, 0, -1):
            with self.subTest(chunked_prefill_size=size):
                with self.assertRaises(ValueError) as ctx:
                    _validate(chunked_prefill_size=size)
                self.assertIn("chunked prefill", str(ctx.exception))

    def test_non_fp8_kv_cache_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            _validate(kv_cache_dtype="bf16")
        self.assertIn("fp8_e4m3", str(ctx.exception))
        # "auto" is materialized as fp8_e4m3 later in the pipeline.
        _validate(kv_cache_dtype="auto")

    def test_context_parallelism_is_refused(self):
        for field in ("attn_cp_size", "dcp_size"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError) as ctx:
                    _validate(**{field: 2})
                self.assertIn("context parallelism", str(ctx.exception))
        with self.assertRaises(ValueError):
            _validate(enable_prefill_cp=True)

    def test_hisparse_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            _validate(enable_hisparse=True)
        self.assertIn("enable_hisparse", str(ctx.exception))

    def test_non_blackwell_is_refused(self):
        cfg = SimpleNamespace(**_SUPPORTED)
        with unittest.mock.patch(
            "sglang.srt.utils.common.is_sm100_supported", return_value=False
        ):
            with self.assertRaises(ValueError) as ctx:
                _validate_deepseek_v4_trtllm_attn(cfg)
        self.assertIn("SM100/SM103", str(ctx.exception))
        with self.assertRaises(ValueError):
            _validate(device="cpu")


if __name__ == "__main__":
    unittest.main()
