import unittest
from array import array

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_req(**kwargs) -> Req:
    sampling_params = SamplingParams(max_new_tokens=1)
    sampling_params.normalize(None)
    return Req(
        rid="req",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2]),
        sampling_params=sampling_params,
        vocab_size=128,
        **kwargs,
    )


class TestFakeTransferRadixInsertSkip(unittest.TestCase):
    """A decode-side fake-transfer request must never be radix-inserted:
    its non-matched input positions were never written.
    Prefill-side fake-transfer and real-transfer decode requests stay insertable."""

    def test_decode_fake_bootstrap_request_skips_radix_insert(self):
        req = _make_req(
            bootstrap_host=FAKE_BOOTSTRAP_HOST,
            disagg_mode=DisaggregationMode.DECODE,
        )
        self.assertTrue(req.skip_radix_cache_insert)

    def test_decode_fake_backend_request_skips_radix_insert(self):
        with get_context().override_server_args(disaggregation_transfer_backend="fake"):
            req = _make_req(
                bootstrap_host=None,
                disagg_mode=DisaggregationMode.DECODE,
            )
        self.assertTrue(req.skip_radix_cache_insert)

    def test_prefill_fake_bootstrap_request_stays_insertable(self):
        req = _make_req(
            bootstrap_host=FAKE_BOOTSTRAP_HOST,
            disagg_mode=DisaggregationMode.PREFILL,
        )
        self.assertFalse(req.skip_radix_cache_insert)

    def test_decode_real_bootstrap_request_stays_insertable(self):
        req = _make_req(
            bootstrap_host="10.0.0.1",
            disagg_mode=DisaggregationMode.DECODE,
        )
        self.assertFalse(req.skip_radix_cache_insert)


if __name__ == "__main__":
    unittest.main()
