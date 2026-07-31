import socket

import pytest
import torch

from sglang.kernels.ops.memory.zero_padded_rows import zero_padded_rows
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="2-gpu-large")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


def _run_uneven_gather(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        max_rows, width = 8, 257
        local_tokens = torch.empty(
            max_rows, width, device=f"cuda:{rank}", dtype=torch.bfloat16
        )
        local_num_tokens = torch.tensor(
            [max_rows], device=f"cuda:{rank}", dtype=torch.int32
        )

        # Capture with the padded count, then replay with unequal live counts.
        zero_padded_rows(local_tokens, local_num_tokens)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            zero_padded_rows(local_tokens, local_num_tokens)

        real_rows = 1 if rank == 0 else max_rows
        local_tokens.fill_(float("nan"))
        local_tokens[:real_rows].fill_(rank + 1)
        local_num_tokens.fill_(real_rows)
        graph.replay()

        gathered = torch.empty(
            world_size * max_rows,
            width,
            device=f"cuda:{rank}",
            dtype=torch.bfloat16,
        )
        torch.distributed.all_gather_into_tensor(gathered, local_tokens)

        assert torch.all(gathered[0] == 1)
        assert torch.count_nonzero(gathered[1:max_rows]).item() == 0
        assert torch.all(gathered[max_rows:] == 2)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA GPUs required")
def test_uneven_dp_counts_zero_padding_before_graph_replay_gather():
    torch.multiprocessing.spawn(
        _run_uneven_gather,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
