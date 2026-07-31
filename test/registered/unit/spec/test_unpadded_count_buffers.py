import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.speculative.eagle_draft_cuda_graph_runner as eagle_draft
import sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner as eagle_extend
import sglang.srt.speculative.frozen_kv_mtp_cuda_graph_runner as frozen_kv
import sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner as multi_layer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StopAtIntegrationPoint(Exception):
    pass


def _capture_buffers():
    buffers = MagicMock()
    buffers.input_ids = torch.zeros(1, dtype=torch.int64)
    buffers.positions = torch.zeros(1, dtype=torch.int64)
    buffers.seq_lens = torch.ones(1, dtype=torch.int32)
    buffers.seq_lens_cpu = torch.ones(1, dtype=torch.int32)
    buffers.hidden_states = torch.zeros(1, 2)
    buffers.global_num_tokens_gpu = torch.zeros(1, dtype=torch.int32)
    buffers.global_num_tokens_unpadded_gpu = torch.zeros(1, dtype=torch.int32)
    buffers.global_num_tokens_for_logprob_gpu = torch.zeros(1, dtype=torch.int32)
    buffers.rids_int = None
    buffers.bootstrap_room_ids_int = None
    buffers.draft_probs = None
    buffers.dsa_seed_topk = None
    buffers.dsa_seed_topk_capture = None
    buffers.temperatures = None
    return buffers


def _configure_capture_runner(runner, buffers):
    runner.buffers = buffers
    runner.captured_req_width = 1
    runner.speculative_num_steps = 1
    runner.require_mlp_tp_gather = False
    runner.require_attn_tp_gather = True
    runner.attn_dp_size = 1
    runner.dp_size = 1
    runner.extend_seq_lens_cpu = [1]
    runner.temperatures = torch.ones(1, 1)
    runner.forward_mode = MagicMock()
    runner.num_front_tokens = 0
    runner.prune_draft_extend_logits = False
    runner.topk = 1
    runner.model_runner = SimpleNamespace(
        model_config=SimpleNamespace(vocab_size=8),
        spec_algorithm=SimpleNamespace(is_standalone=lambda: False),
    )


def _draft_capture_case():
    buffers = _capture_buffers()
    runner = eagle_draft.EAGLEDraftCudaGraphRunner.__new__(
        eagle_draft.EAGLEDraftCudaGraphRunner
    )
    _configure_capture_runner(runner, buffers)
    return (
        eagle_draft,
        runner,
        lambda: runner.capture_one_shape(1, MagicMock()),
        buffers,
    )


def _extend_capture_case():
    buffers = _capture_buffers()
    runner = eagle_extend.EAGLEDraftExtendCudaGraphRunner.__new__(
        eagle_extend.EAGLEDraftExtendCudaGraphRunner
    )
    _configure_capture_runner(runner, buffers)
    return (
        eagle_extend,
        runner,
        lambda: runner.capture_one_shape(1, MagicMock()),
        buffers,
    )


def _frozen_capture_case():
    buffers = _capture_buffers()
    runner = frozen_kv.FrozenKVMTPCudaGraphRunner.__new__(
        frozen_kv.FrozenKVMTPCudaGraphRunner
    )
    _configure_capture_runner(runner, buffers)
    return frozen_kv, runner, lambda: runner.capture_one_shape(1, MagicMock()), buffers


def _multi_layer_capture_case():
    buffers = _capture_buffers()
    runner = multi_layer.MultiLayerEagleDraftExtendCudaGraphRunner.__new__(
        multi_layer.MultiLayerEagleDraftExtendCudaGraphRunner
    )
    _configure_capture_runner(runner, buffers)
    return multi_layer, runner, lambda: runner.get_forward_batch(1), buffers


CAPTURE_CASES = {
    "eagle_draft": _draft_capture_case,
    "eagle_extend": _extend_capture_case,
    "multi_layer": _multi_layer_capture_case,
    "frozen_kv": _frozen_capture_case,
}


def _base_replay_runner(runner_type):
    runner = runner_type.__new__(runner_type)
    buffers = MagicMock()
    buffers.global_num_tokens_gpu = torch.zeros(2, dtype=torch.int32)
    buffers.global_num_tokens_unpadded_gpu = torch.zeros(2, dtype=torch.int32)
    buffers.global_num_tokens_for_logprob_gpu = torch.zeros(2, dtype=torch.int32)
    runner.buffers = buffers
    runner.require_gathered_buffer = True
    runner.require_mlp_tp_gather = False
    runner.captured_req_width = 1
    runner.capture_bs = [1]
    runner.seq_len_fill_value = 1
    runner._pad_to_bucket = lambda size, buckets: size
    runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
    return runner, buffers


def _forward_batch():
    return SimpleNamespace(
        batch_size=1,
        input_ids=torch.zeros(1, dtype=torch.int64),
        seq_lens=torch.ones(1, dtype=torch.int32),
        seq_lens_cpu=None,
        seq_lens_sum=1,
        extend_seq_lens=None,
        extend_seq_lens_cpu=None,
        out_cache_loc=torch.zeros(1, dtype=torch.int64),
        positions=torch.zeros(1, dtype=torch.int64),
        mrope_positions=None,
        req_pool_indices=torch.zeros(1, dtype=torch.int32),
        rids_int=None,
        bootstrap_room_ids_int=None,
        sampling_info=None,
        global_num_tokens_unpadded_gpu=torch.tensor([1, 8], dtype=torch.int32),
        spec_info=SimpleNamespace(
            topk_p=torch.ones(1, 1),
            topk_index=torch.zeros(1, 1, dtype=torch.int64),
            draft_probs=None,
            hidden_states=torch.zeros(1, 2),
            dsa_topk_indices=None,
            num_correct_drafts=None,
            num_accept_tokens=None,
            bonus_tokens=torch.zeros(1, dtype=torch.int64),
        ),
    )


def _draft_replay_case():
    runner, buffers = _base_replay_runner(eagle_draft.EAGLEDraftCudaGraphRunner)
    runner.speculative_num_steps = 1
    runner.model_runner = SimpleNamespace(
        model_config=SimpleNamespace(vocab_size=8),
        server_args=SimpleNamespace(speculative_use_rejection_sampling=False),
    )
    return eagle_draft, runner, runner.execute, buffers


def _extend_replay_case():
    runner, buffers = _base_replay_runner(eagle_extend.EAGLEDraftExtendCudaGraphRunner)
    runner.model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_eagle=lambda: True)
    )
    return eagle_extend, runner, runner.execute, buffers


def _frozen_replay_case():
    runner, buffers = _base_replay_runner(frozen_kv.FrozenKVMTPCudaGraphRunner)
    runner.topk = 1
    return frozen_kv, runner, runner.execute, buffers


def _multi_layer_replay_case():
    runner = multi_layer.MultiLayerEagleMultiStepDraftExtendCudaGraphRunner.__new__(
        multi_layer.MultiLayerEagleMultiStepDraftExtendCudaGraphRunner
    )
    buffers = MagicMock()
    buffers.hidden_states = torch.zeros(1, 2)
    buffers.temperatures = None
    buffers.global_num_tokens_gpu = torch.zeros(2, dtype=torch.int32)
    buffers.global_num_tokens_unpadded_gpu = torch.zeros(2, dtype=torch.int32)
    buffers.global_num_tokens_for_logprob_gpu = torch.zeros(2, dtype=torch.int32)
    runner.buffers = buffers
    runner.require_gathered_buffer = True
    runner.require_mlp_tp_gather = False
    runner.capture_bs = [1]
    runner.captured_req_width = 1
    runner.num_front_tokens = 0
    runner.seq_len_fill_value = 1
    runner.get_runner = lambda step: SimpleNamespace(
        _pad_to_bucket=lambda size, buckets: size
    )
    return multi_layer, runner, runner.prepare, buffers


REPLAY_CASES = {
    "eagle_draft": _draft_replay_case,
    "eagle_extend": _extend_replay_case,
    "multi_layer": _multi_layer_replay_case,
    "frozen_kv": _frozen_replay_case,
}


class TestSpecializedUnpaddedCountIntegration(CustomTestCase):
    def test_capture_batch_references_static_unpadded_buffer(self):
        for name, case_factory in CAPTURE_CASES.items():
            with self.subTest(runner=name):
                module, _, invoke, buffers = case_factory()

                def capture_forward_batch(**kwargs):
                    self.assertIs(
                        kwargs["global_num_tokens_unpadded_gpu"],
                        buffers.global_num_tokens_unpadded_gpu,
                    )
                    raise _StopAtIntegrationPoint

                with patch.object(module, "ForwardBatch", capture_forward_batch):
                    with self.assertRaises(_StopAtIntegrationPoint):
                        invoke()

    def test_replay_copies_live_unpadded_counts(self):
        for name, case_factory in REPLAY_CASES.items():
            with self.subTest(runner=name):
                module, _, invoke, buffers = case_factory()
                forward_batch = _forward_batch()

                def copy_and_stop(actual_buffers, actual_batch):
                    self.assertIs(actual_buffers, buffers)
                    self.assertIs(actual_batch, forward_batch)
                    actual_buffers.global_num_tokens_unpadded_gpu.copy_(
                        actual_batch.global_num_tokens_unpadded_gpu
                    )
                    raise _StopAtIntegrationPoint

                patches = [
                    patch.object(
                        module,
                        "refresh_global_num_tokens_unpadded",
                        copy_and_stop,
                    )
                ]
                if hasattr(module, "_grouped_foreach_copy_"):
                    patches.append(
                        patch.object(
                            module, "_grouped_foreach_copy_", lambda *args: None
                        )
                    )
                if module is eagle_draft:
                    patches.extend(
                        [
                            patch.object(
                                module, "maybe_detect_nan", lambda *args: None
                            ),
                            patch.object(
                                module, "maybe_detect_oob", lambda *args: None
                            ),
                        ]
                    )
                if module is multi_layer:
                    patches.append(
                        patch.object(
                            module,
                            "fill_draft_extend_prepare_buffers",
                            lambda *args: None,
                        )
                    )

                with self.assertRaises(_StopAtIntegrationPoint):
                    with patches[0]:
                        with ExitStack() as stack:
                            for active_patch in patches[1:]:
                                stack.enter_context(active_patch)
                            invoke(forward_batch)

                self.assertTrue(
                    torch.equal(
                        buffers.global_num_tokens_unpadded_gpu,
                        forward_batch.global_num_tokens_unpadded_gpu,
                    )
                )


if __name__ == "__main__":
    unittest.main()
