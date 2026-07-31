import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import sglang.srt.layers.dp_attention as dp_attention
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
        global_num_tokens_unpadded_gpu = torch.tensor(
            [1, max_rows], device=f"cuda:{rank}", dtype=torch.int32
        )
        real_rows = 1 if rank == 0 else max_rows
        local_tokens.fill_(float("nan"))
        local_tokens[:real_rows].fill_(rank + 1)

        gathered = torch.empty(
            world_size * max_rows,
            width,
            device=f"cuda:{rank}",
            dtype=torch.bfloat16,
        )

        class _Group:
            @staticmethod
            def all_gather_into_tensor(output, input_):
                torch.distributed.all_gather_into_tensor(output, input_)

        forward_batch = SimpleNamespace(
            global_num_tokens_unpadded_gpu=global_num_tokens_unpadded_gpu
        )
        with (
            patch.object(
                dp_attention, "get_attn_tensor_model_parallel_world_size", lambda: 1
            ),
            patch.object(dp_attention, "get_attention_dp_rank", lambda: rank),
            patch.object(dp_attention, "world_dp_gather_enabled", lambda: False),
            patch.object(dp_attention, "get_tp_group", lambda: _Group()),
        ):
            dp_attention._dp_gather_via_all_gather(
                gathered, local_tokens, forward_batch, is_partial=True
            )

        assert torch.all(gathered[0] == 1)
        assert torch.count_nonzero(gathered[1:max_rows]).item() == 0
        assert torch.all(gathered[max_rows:] == 2)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA GPUs required")
def test_uneven_dp_counts_zero_padding_in_production_gather():
    torch.multiprocessing.spawn(
        _run_uneven_gather,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
