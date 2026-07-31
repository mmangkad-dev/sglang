import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.batch_overlap.two_batch_overlap as tbo
from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestTboUnpaddedCounts(CustomTestCase):
    def test_filter_batch_clears_unpadded_global_counts(self):
        batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=8,
            input_ids=torch.zeros(8, dtype=torch.long),
            positions=torch.zeros(8, dtype=torch.long),
            out_cache_loc=torch.zeros(8, dtype=torch.long),
            req_pool_indices=torch.zeros(8, dtype=torch.long),
            seq_lens=torch.ones(8, dtype=torch.int32),
            seq_lens_cpu=torch.ones(8, dtype=torch.int32),
            seq_lens_sum=8,
            global_num_tokens_unpadded_gpu=torch.tensor([4, 8], dtype=torch.int32),
        )
        fake_args = SimpleNamespace(moe_dense_tp_size=None, attention_backend="fa3")

        with get_parallel().override(attn_tp_size=1), patch.object(
            tbo, "get_server_args", lambda: fake_args
        ):
            child = TboForwardBatchPreparer.filter_batch(
                batch,
                start_token_index=0,
                end_token_index=4,
                start_seq_index=0,
                end_seq_index=4,
                out_num_token_non_padded=torch.tensor(4),
            )

        self.assertIsNone(child.global_num_tokens_unpadded_gpu)


if __name__ == "__main__":
    unittest.main()
