import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.model_executor.runner import prefill_cuda_graph_runner as prefill_mod
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPrefillCudaGraphPadding(CustomTestCase):
    def _make_runner(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner.enable_lora = False
        runner._capture_chunked_prefix = False
        runner.prefill_backend_name = Backend.TC_PIECEWISE
        runner.has_mha_companion_layers = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.capture_num_tokens = [4, 16]
        runner.max_num_tokens = 16
        return runner

    def _make_forward_batch(self, num_tokens):
        return SimpleNamespace(
            batch_size=1,
            input_embeds=None,
            replace_embeds=None,
            mm_inputs=None,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_tokens_cpu=None,
            return_logprob=False,
            input_ids=list(range(num_tokens)),
            extend_prefix_lens_cpu=[0],
        )

    def test_rejects_more_than_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertFalse(runner.can_run_graph(self._make_forward_batch(5)))

    def test_accepts_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertTrue(runner.can_run_graph(self._make_forward_batch(8)))

    def test_replay_snapshot_uses_padded_token_count(self):
        runner = self._make_runner()
        runner.use_captured_attn_metadata = False
        attn_backend = mock.Mock()
        runner.model_runner = SimpleNamespace(attn_backend=attn_backend)
        forward_batch = self._make_forward_batch(8)
        static_forward_batch = self._make_forward_batch(16)

        runner._prepare_forward_metadata_for_replay(
            forward_batch,
            static_forward_batch,
            num_tokens=16,
        )

        attn_backend.init_forward_metadata.assert_called_once_with(forward_batch)
        attn_backend.prepare_prefill_shared_read_snapshot.assert_called_once_with(
            forward_batch, num_qo_tokens=16
        )


class TestPrefillCaptureBucketAttnTpAlignment(CustomTestCase):
    """Capture buckets must divide evenly across the attention TP group.

    When attn_tp scatters hidden states, every rank needs an equal shard.
    Bucket 28 on attn_tp_size=8 splits [4, 4, 4, 4, 3, 3, 3, 3] and the
    mismatched reduce-scatter hangs capture, which is reachable once breakable
    prefill CUDA graphs are enabled for MLA models.
    """

    DEFAULT_BUCKETS = [4, 8, 12, 16, 20, 24, 28, 32, 48, 64]

    def _align(self, buckets, attn_tp_size, gathered_buffer):
        with mock.patch.object(
            prefill_mod, "require_gathered_buffer", return_value=gathered_buffer
        ):
            return prefill_mod._align_capture_num_tokens_to_attn_tp(
                list(buckets), attn_tp_size
            )

    def test_gathered_buffer_rounds_buckets_up_to_attn_tp(self):
        aligned = self._align(self.DEFAULT_BUCKETS, 8, True)
        self.assertEqual(aligned, [8, 16, 24, 32, 48, 64])
        self.assertTrue(all(n % 8 == 0 for n in aligned))

    def test_buckets_untouched_without_gathered_buffer(self):
        self.assertEqual(
            self._align(self.DEFAULT_BUCKETS, 8, False), self.DEFAULT_BUCKETS
        )

    def test_attn_tp_size_one_is_a_no_op(self):
        self.assertEqual(
            self._align(self.DEFAULT_BUCKETS, 1, True), self.DEFAULT_BUCKETS
        )

    def test_explicit_unaligned_bucket_is_rounded_not_dropped(self):
        # --cuda-graph-bs-prefill 28 must still capture a usable bucket.
        self.assertEqual(self._align([28], 8, True), [32])


if __name__ == "__main__":
    unittest.main()
