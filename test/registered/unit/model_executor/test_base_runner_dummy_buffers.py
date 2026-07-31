import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.model_executor.runner.base_runner as base_runner
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.base_runner import BaseRunner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _TestRunner(BaseRunner):
    def can_run_graph(self, forward_batch):
        return False

    def load_batch(self, forward_batch, **kwargs):
        return None

    def execute(self, forward_batch, **kwargs):
        return None


class _SpecAlgorithm:
    def is_speculative(self):
        return False

    def is_eagle(self):
        return False

    def is_standalone(self):
        return False

    def is_dflash_family(self):
        return False

    def is_ngram(self):
        return False


class _Model:
    _can_torch_compile = True

    def __init__(self):
        self.forward_batch = None

    def forward(self, input_ids, positions, forward_batch):
        self.forward_batch = forward_batch


class TestBaseRunnerDummyBuffers(CustomTestCase):
    def test_allocated_dummy_buffers_run_through_dummy_forward(self):
        model = _Model()
        attn_backend = SimpleNamespace(
            get_cuda_graph_seq_len_fill_value=lambda: 1,
            init_forward_metadata=lambda forward_batch: None,
        )
        server_args = SimpleNamespace(
            enable_return_hidden_states=False,
            pp_size=1,
            dp_size=1,
            enable_lora=False,
        )
        model_runner = SimpleNamespace(
            device=torch.device("cpu"),
            is_generation=True,
            is_draft_worker=False,
            spec_algorithm=_SpecAlgorithm(),
            server_args=server_args,
            model_config=SimpleNamespace(hidden_size=8),
            dtype=torch.bfloat16,
            attn_backend=attn_backend,
            model=model,
            prepare_dummy_forward_batch=lambda forward_batch: forward_batch,
            tp_group=SimpleNamespace(barrier=lambda: None),
        )
        runner = _TestRunner.__new__(_TestRunner)
        runner.model_runner = model_runner
        buffers = base_runner._allocate_decode_buffers(
            device=torch.device("cpu"),
            max_bs=2,
            max_num_token=2,
            hidden_size=8,
            vocab_size=16,
            dtype=torch.bfloat16,
            dp_size=1,
            pp_size=1,
            is_encoder_decoder=False,
            require_mlp_tp_gather=False,
            seq_len_fill_value=1,
            encoder_len_fill_value=0,
            num_tokens_per_req=1,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=False,
        )
        flags = SimpleNamespace(capture=SimpleNamespace(enable_torch_compile=False))

        with (
            patch.object(base_runner, "get_flags", lambda: flags),
            patch.object(base_runner, "require_mlp_tp_gather", lambda args: False),
            patch.object(base_runner, "require_attn_tp_gather", lambda args: False),
            patch.object(base_runner, "require_gathered_buffer", lambda args: False),
        ):
            runner._dummy_run(
                batch_size=2,
                forward_mode_override=ForwardMode.DECODE,
                buffers=buffers,
            )

        self.assertEqual(buffers.global_num_tokens_unpadded_gpu.shape, (1,))
        self.assertIs(
            model.forward_batch.global_num_tokens_unpadded_gpu,
            buffers.global_num_tokens_unpadded_gpu,
        )


if __name__ == "__main__":
    unittest.main()
