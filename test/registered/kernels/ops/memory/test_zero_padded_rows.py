import pytest
import torch

from sglang.kernels.ops.memory.zero_padded_rows import zero_padded_rows
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zero_padded_rows_reads_updated_count_on_graph_replay():
    tokens = torch.empty(8, 257, device="cuda", dtype=torch.bfloat16)
    valid_rows = torch.tensor([8], device="cuda", dtype=torch.int32)

    zero_padded_rows(tokens, valid_rows)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        zero_padded_rows(tokens, valid_rows)

    tokens.fill_(float("nan"))
    tokens[0].fill_(1)
    valid_rows.fill_(1)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.all(tokens[0] == 1)
    assert torch.count_nonzero(tokens[1:]).item() == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
