import dataclasses
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.input_buffers import (
    refresh_global_num_tokens_unpadded,
)
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EagleDraftInputBuffers,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EagleDraftExtendInputBuffers,
)
from sglang.srt.speculative.frozen_kv_mtp_cuda_graph_runner import (
    FrozenKVMTPInputBuffers,
)
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleDraftExtendInputBuffers,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


SPECIALIZED_BUFFER_TYPES = [
    EagleDraftInputBuffers,
    EagleDraftExtendInputBuffers,
    MultiLayerEagleDraftExtendInputBuffers,
    FrozenKVMTPInputBuffers,
]


@pytest.mark.parametrize("buffer_type", SPECIALIZED_BUFFER_TYPES)
def test_specialized_graph_buffer_refreshes_unpadded_counts(buffer_type):
    assert "global_num_tokens_unpadded_gpu" in {
        field.name for field in dataclasses.fields(buffer_type)
    }
    buffers = buffer_type.__new__(buffer_type)
    buffers.global_num_tokens_unpadded_gpu = torch.zeros(2, dtype=torch.int32)
    forward_batch = SimpleNamespace(
        global_num_tokens_unpadded_gpu=torch.tensor([1, 8], dtype=torch.int32)
    )

    refresh_global_num_tokens_unpadded(buffers, forward_batch)

    assert torch.equal(
        buffers.global_num_tokens_unpadded_gpu,
        forward_batch.global_num_tokens_unpadded_gpu,
    )
