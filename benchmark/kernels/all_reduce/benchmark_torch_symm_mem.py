"""Tune SGLang all-reduce selection on NVIDIA symmetric-memory systems.

This benchmark compares the current SGLang implementations that can service an
out-of-place BF16 all-reduce:

* SGLang custom all-reduce
* ordinary PyNCCL
* NCCL symmetric memory (``ncclMemAlloc`` + registered symmetric windows)
* PyTorch symmetric-memory multimem
* PyTorch symmetric-memory two-shot

Both eager execution and CUDA-graph replay are supported. Results are reduced
with MAX across ranks, because a collective completes at the pace of its
slowest rank, and can be written to JSON for threshold analysis.

Examples:

    torchrun --standalone --nproc-per-node=8 \
      benchmark/kernels/all_reduce/benchmark_torch_symm_mem.py \
      --output-json sm90-tp8.json

    torchrun --standalone --nproc-per-node=4 \
      benchmark/kernels/all_reduce/benchmark_torch_symm_mem.py \
      --sizes-kib 16 32 64 128 256 512 1024 2048 4096 8192 \
                  16384 32768 65536 131072 \
      --modes eager graph --output-json sm90-tp4.json

NCCL symmetric memory requires a sufficiently recent NCCL/CUDA/PyTorch stack.
The benchmark sets the standard NCCL NVLS/CUMEM environment defaults before
NCCL initialization; explicit user values are preserved.

The benchmark disables ``SGLANG_OPT_USE_INKLING_CUSTOM_AR`` by default.
Inkling mode requires a published ``ServerArgs`` runtime context, which this
low-level communicator benchmark intentionally does not construct. An explicit
environment value is still preserved for debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, ContextManager, Optional

# These must be set before importing/initializing NCCL.
os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")
os.environ.setdefault("NCCL_NVLS_ENABLE", "1")
# This option defaults to enabled in SGLang, but cannot work in this standalone
# benchmark because no ServerArgs runtime context is published.
os.environ.setdefault("SGLANG_OPT_USE_INKLING_CUSTOM_AR", "0")

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)

DEFAULT_SIZES_KIB = [
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
]


@dataclass
class Measurement:
    backend: str
    mode: str
    size_bytes: int
    latency_ms: Optional[float]
    available: bool
    error: Optional[str] = None


@dataclass
class Backend:
    name: str
    run: Callable[[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]
    allocate: Callable[[int], tuple[torch.Tensor, torch.Tensor]]
    enabled: Callable[[torch.Tensor], bool] = lambda _tensor: True
    capture_context: Callable[[], ContextManager] = nullcontext
    close: Callable[[], None] = lambda: None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark current SGLang all-reduce implementations."
    )
    parser.add_argument(
        "--sizes-kib",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES_KIB,
        help="BF16 message sizes in KiB (default: 16 KiB through 128 MiB).",
    )
    parser.add_argument(
        "--modes",
        choices=("eager", "graph"),
        nargs="+",
        default=("eager", "graph"),
        help="Execution modes to benchmark.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument(
        "--ops-per-trial",
        type=int,
        default=10,
        help="Collectives per timed trial; amortizes host timing overhead.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=(
            "custom",
            "pynccl",
            "nccl-symm",
            "torch-multimem",
            "torch-two-shot",
        ),
        default=(
            "custom",
            "pynccl",
            "nccl-symm",
            "torch-multimem",
            "torch-two-shot",
        ),
    )
    parser.add_argument(
        "--output-json", type=Path, help="Write metadata and measurements to JSON."
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip the rank-dependent correctness check (not recommended).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if min(args.sizes_kib) <= 0:
        parser.error("--sizes-kib values must be positive")
    if args.warmup < 0 or args.trials <= 0 or args.ops_per_trial <= 0:
        parser.error(
            "warmup must be nonnegative; trials/ops-per-trial must be positive"
        )
    return args


def rank_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def initialize() -> tuple[int, int, torch.device]:
    rank, world_size, local_rank = rank_info()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires NVIDIA CUDA GPUs")
    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    # A CUDA TP group creates PyNCCL by default. The module-level
    # NCCL_CUMEM_ENABLE/NCCL_NVLS_ENABLE defaults and the explicit
    # SymmetricMemoryContext below enable the NCCL symmetric-memory path.
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def regular_allocate(
    size_bytes: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    numel = size_bytes // torch.bfloat16.itemsize
    return (
        torch.empty(numel, dtype=torch.bfloat16, device=device),
        torch.empty(numel, dtype=torch.bfloat16, device=device),
    )


def make_backends(
    requested: list[str],
    device: torch.device,
    max_size: int,
) -> tuple[list[Backend], list[str]]:
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        SymmetricMemoryContext,
    )
    from sglang.srt.distributed.device_communicators.torch_symm_mem import (
        TorchSymmMemCommunicator,
    )

    tp = get_tp_group()
    cpu_group = tp.cpu_group
    pynccl = tp.pynccl_comm
    backends: list[Backend] = []
    setup_errors: list[str] = []

    def add(name: str, factory: Callable[[], Backend]) -> None:
        if name not in requested:
            return
        try:
            backends.append(factory())
        except Exception as exc:
            setup_errors.append(f"{name}: {type(exc).__name__}: {exc}")

    def custom_backend() -> Backend:
        # Reuse the communicator constructed by GroupCoordinator so this is
        # exactly the custom-AR implementation and workspace sizing selected by
        # the current SGLang configuration.
        comm = tp.ca_comm
        if comm is None or comm.disabled:
            raise RuntimeError("communicator disabled for this topology")
        return Backend(
            name="custom",
            run=lambda inp, _out: comm.custom_all_reduce(inp),
            allocate=lambda size: regular_allocate(size, device),
            enabled=comm.should_custom_ar,
            capture_context=comm.capture,
        )

    def pynccl_backend() -> Backend:
        if pynccl is None or not pynccl.available:
            raise RuntimeError("PyNCCL communicator unavailable")

        def run(inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            with pynccl.change_state(enable=True):
                result = pynccl.outplace_all_reduce(inp, out)
            return result

        return Backend(
            name="pynccl",
            run=run,
            allocate=lambda size: regular_allocate(size, device),
        )

    def nccl_symm_backend() -> Backend:
        if pynccl is None or not pynccl.available:
            raise RuntimeError("PyNCCL communicator unavailable")

        def allocate(size: int) -> tuple[torch.Tensor, torch.Tensor]:
            # Allocation order and sizes must be identical on every rank.
            with SymmetricMemoryContext(tp):
                return regular_allocate(size, device)

        def run(inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            with pynccl.change_state(enable=True):
                result = pynccl.outplace_all_reduce(inp, out)
            return result

        return Backend(name="nccl-symm", run=run, allocate=allocate)

    def torch_backend(name: str, force_multimem: bool) -> Backend:
        # Eligibility uses a strict ``input_size < max_size`` check. Add one
        # byte so the largest requested (even-byte BF16) payload is admitted;
        # integer division still allocates exactly max_size bytes.
        comm = TorchSymmMemCommunicator(
            group=cpu_group,
            device=device,
            force_multimem=force_multimem,
            max_size_override=max_size + 1,
        )
        if comm.disabled:
            raise RuntimeError("communicator disabled or NVLink multicast unavailable")
        return Backend(
            name=name,
            run=lambda inp, out: comm.all_reduce(inp, out=out),
            allocate=lambda size: regular_allocate(size, device),
            enabled=comm.should_torch_symm_mem_allreduce,
        )

    add("custom", custom_backend)
    add("pynccl", pynccl_backend)
    add("nccl-symm", nccl_symm_backend)
    add("torch-multimem", lambda: torch_backend("torch-multimem", True))
    add("torch-two-shot", lambda: torch_backend("torch-two-shot", False))
    return backends, setup_errors


def validate(
    backend: Backend,
    inp: torch.Tensor,
    out: torch.Tensor,
    rank: int,
    world_size: int,
) -> None:
    inp.fill_(rank + 1)
    out.fill_(float("nan"))
    result = backend.run(inp, out)
    if result is None:
        raise RuntimeError("backend returned None")
    torch.cuda.synchronize()
    expected = world_size * (world_size + 1) // 2
    torch.testing.assert_close(
        result,
        torch.full_like(result, expected),
        rtol=0,
        atol=0,
    )


def capture_graph(
    backend: Backend,
    inputs_and_outputs: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    inp, out = inputs_and_outputs[0]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            backend.run(inp, out)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    graph_out = out
    with backend.capture_context():
        with torch.cuda.graph(graph, stream=stream):
            # Custom AR registers graph inputs per collective. Use distinct
            # static input buffers, as a serving graph does across layers,
            # rather than registering the same pointer repeatedly.
            for inp, out in inputs_and_outputs:
                result = backend.run(inp, out)
                if result is not None:
                    graph_out = result
    torch.cuda.current_stream().wait_stream(stream)
    return graph, graph_out


def timed_trials(
    operation: Callable[[], None],
    warmup: int,
    trials: int,
    ops_per_trial: int,
    device: torch.device,
) -> tuple[float, list[float]]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    dist.barrier()

    samples: list[float] = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        local_ms = start.elapsed_time(end) / ops_per_trial
        value = torch.tensor(local_ms, dtype=torch.float64, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        samples.append(value.item())
    return statistics.median(samples), samples


def measure(
    backend: Backend,
    mode: str,
    size_bytes: int,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> Measurement:
    try:
        inp, out = backend.allocate(size_bytes)
        inp.fill_(rank + 1)
        if not backend.enabled(inp):
            return Measurement(
                backend.name, mode, size_bytes, None, False, "ineligible"
            )
        if not args.skip_correctness:
            validate(backend, inp, out, rank, world_size)
            inp.fill_(rank + 1)

        if mode == "graph":
            inputs_and_outputs = [(inp, out)]
            for _ in range(args.ops_per_trial - 1):
                graph_inp, graph_out = backend.allocate(size_bytes)
                graph_inp.fill_(rank + 1)
                inputs_and_outputs.append((graph_inp, graph_out))
            graph, graph_out = capture_graph(backend, inputs_and_outputs)
            if not args.skip_correctness:
                graph.replay()
                torch.cuda.synchronize()
                expected = world_size * (world_size + 1) // 2
                torch.testing.assert_close(
                    graph_out,
                    torch.full_like(graph_out, expected),
                    rtol=0,
                    atol=0,
                )
            operation = graph.replay
        else:

            def operation() -> None:
                for _ in range(args.ops_per_trial):
                    backend.run(inp, out)

        median_ms, samples = timed_trials(
            operation,
            args.warmup,
            args.trials,
            args.ops_per_trial,
            device,
        )
        if args.verbose and rank == 0:
            print(
                f"  samples: min={min(samples):.4f} "
                f"median={median_ms:.4f} max={max(samples):.4f} ms"
            )
        return Measurement(backend.name, mode, size_bytes, median_ms, True)
    except Exception as exc:
        return Measurement(
            backend.name,
            mode,
            size_bytes,
            None,
            False,
            f"{type(exc).__name__}: {exc}",
        )


def metadata(world_size: int, device: torch.device) -> dict:
    props = torch.cuda.get_device_properties(device)
    return {
        "world_size": world_size,
        "device_name": props.name,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nccl_version": list(torch.cuda.nccl.version()),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "NCCL_CUMEM_ENABLE",
                "NCCL_NVLS_ENABLE",
                "NCCL_DEBUG",
                "SGLANG_OPT_USE_INKLING_CUSTOM_AR",
            )
        },
    }


def print_table(measurements: list[Measurement], sizes: list[int]) -> None:
    columns = sorted({(m.backend, m.mode) for m in measurements})
    lookup = {(m.size_bytes, m.backend, m.mode): m for m in measurements}
    headings = ["size"] + [f"{backend}/{mode}" for backend, mode in columns]
    widths = [max(10, len(heading)) for heading in headings]
    print("  ".join(h.ljust(w) for h, w in zip(headings, widths)))
    print("  ".join("-" * w for w in widths))
    for size in sizes:
        row = [f"{size / 1024:.0f} KiB"]
        for backend, mode in columns:
            item = lookup.get((size, backend, mode))
            if item is None or not item.available:
                row.append("N/A")
            else:
                row.append(f"{item.latency_ms:.4f} ms")
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


def print_crossover_summary(measurements: list[Measurement]) -> None:
    """Report raw crossover candidates without silently choosing a config.

    Torch's algorithm is a per-(architecture, world-size) choice, while the
    production fallback can vary by message size. We therefore show both Torch
    variants against the fastest measured non-Torch SGLang path and leave the
    final safety margin to the operator.
    """

    modes = sorted({item.mode for item in measurements})
    for mode in modes:
        by_size: dict[int, dict[str, float]] = {}
        for item in measurements:
            if item.mode == mode and item.available and item.latency_ms is not None:
                by_size.setdefault(item.size_bytes, {})[item.backend] = item.latency_ms

        print(f"\nCrossover candidates ({mode}):")
        print("  size       best Torch         best non-Torch      Torch/non-Torch")
        for size, values in sorted(by_size.items()):
            torch_values = {
                name: value
                for name, value in values.items()
                if name in ("torch-multimem", "torch-two-shot")
            }
            fallback_values = {
                name: value
                for name, value in values.items()
                if name in ("custom", "pynccl", "nccl-symm")
            }
            if not torch_values or not fallback_values:
                continue
            torch_name = min(torch_values, key=torch_values.get)
            fallback_name = min(fallback_values, key=fallback_values.get)
            ratio = torch_values[torch_name] / fallback_values[fallback_name]
            print(
                f"  {size / 1024:>7.0f} KiB  "
                f"{torch_name:<17} {torch_values[torch_name]:>8.4f} ms  "
                f"{fallback_name:<12} {fallback_values[fallback_name]:>8.4f} ms  "
                f"{ratio:>7.3f}x"
            )


def main() -> int:
    args = parse_args()
    rank = 0
    try:
        rank, world_size, device = initialize()
        sizes = sorted(set(size * 1024 for size in args.sizes_kib))
        backends, setup_errors = make_backends(args.backends, device, max(sizes))
        run_metadata = metadata(world_size, device)

        if rank == 0:
            print(json.dumps(run_metadata, indent=2))
            for error in setup_errors:
                print(f"Backend unavailable: {error}", file=sys.stderr)

        measurements: list[Measurement] = []
        for size in sizes:
            for backend in backends:
                for mode in args.modes:
                    dist.barrier()
                    result = measure(
                        backend, mode, size, args, rank, world_size, device
                    )
                    measurements.append(result)
                    if rank == 0:
                        value = (
                            f"{result.latency_ms:.4f} ms"
                            if result.available
                            else f"N/A ({result.error})"
                        )
                        print(
                            f"{size / 1024:>8.0f} KiB  "
                            f"{backend.name:<16} {mode:<5} {value}"
                        )

        if rank == 0:
            print()
            print_table(measurements, sizes)
            print_crossover_summary(measurements)
            if args.output_json:
                payload = {
                    "metadata": run_metadata,
                    "configuration": {
                        "sizes_kib": args.sizes_kib,
                        "modes": args.modes,
                        "warmup": args.warmup,
                        "trials": args.trials,
                        "ops_per_trial": args.ops_per_trial,
                        "requested_backends": args.backends,
                        "setup_errors": setup_errors,
                    },
                    "measurements": [asdict(item) for item in measurements],
                }
                args.output_json.parent.mkdir(parents=True, exist_ok=True)
                args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
                print(f"Wrote {args.output_json}")
        for backend in backends:
            backend.close()
        if not any(item.available for item in measurements):
            if rank == 0:
                print(
                    "ERROR: no backend produced a valid measurement",
                    file=sys.stderr,
                )
            return 2
        return 0
    finally:
        if dist.is_initialized():
            dist.barrier()
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    raise SystemExit(main())
