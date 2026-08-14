import inspect
import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed import (
    get_attn_tp_group,
    get_moe_ep_group,
    get_moe_tp_group,
    get_tp_group,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.utils import (
    ceil_align,
    get_cuda_driver_bindings,
    is_flashinfer_available,
    is_sm90_supported,
    is_sm100_supported,
)
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)

# FlashInfer allreduce fusion: set when flashinfer is available (see block below)
_flashinfer_comm = None
_TorchDistBackend = None
_mnnvl_comm_backend = None
_create_allreduce_fusion_workspace = None
_flashinfer_allreduce_unavailable = False
_flashinfer_create_workspace_supports_group = False
_flashinfer_create_workspace_supports_comm_backend = False
_flashinfer_allreduce_supports_trigger_completion = False


def _mnnvl_supported(is_multi_node: bool) -> bool:
    """Whether the mnnvl backend is usable on the current system."""
    if is_sm100_supported():
        return True
    return is_sm90_supported() and not is_multi_node


def _resolve_backend(backend: str, is_multi_node: bool = False) -> str:
    """Resolve the requested FlashInfer allreduce fusion backend."""
    if not (is_sm90_supported() or is_sm100_supported()):
        raise ValueError(
            "FlashInfer allreduce fusion requires SM90 or SM10X NVIDIA GPUs."
        )

    if backend == "auto":
        if is_multi_node:
            if is_sm100_supported():
                return "mnnvl"
            raise ValueError(
                "FlashInfer allreduce fusion does not support multi-node on "
                "non-Blackwell systems."
            )
        if is_sm100_supported():
            return "mnnvl"
        return "trtllm"

    if backend == "trtllm" and is_multi_node:
        raise ValueError(
            "FlashInfer allreduce fusion trtllm backend supports single-node only."
        )

    if backend == "mnnvl" and not _mnnvl_supported(is_multi_node):
        raise ValueError(
            "FlashInfer allreduce fusion mnnvl backend requires a Blackwell "
            "system, or SM90 single-node."
        )
    return backend


def resolve_flashinfer_allreduce_fusion_backend(server_args) -> Optional[str]:
    backend = getattr(server_args, "flashinfer_allreduce_fusion_backend", None)
    if backend is None:
        return None
    is_multi_node = getattr(server_args, "nnodes", 1) > 1
    return _resolve_backend(backend, is_multi_node)


if is_flashinfer_available():
    try:
        import flashinfer.comm as comm

        if hasattr(comm, "allreduce_fusion") and hasattr(
            comm, "create_allreduce_fusion_workspace"
        ):
            _flashinfer_comm = comm
            _create_allreduce_fusion_workspace = comm.create_allreduce_fusion_workspace
            workspace_params = inspect.signature(
                comm.create_allreduce_fusion_workspace
            ).parameters
            allreduce_params = inspect.signature(comm.allreduce_fusion).parameters
            _flashinfer_create_workspace_supports_group = "group" in workspace_params
            _flashinfer_create_workspace_supports_comm_backend = (
                "comm_backend" in workspace_params
            )
            _flashinfer_allreduce_supports_trigger_completion = (
                "trigger_completion_at_end" in allreduce_params
            )
        else:
            _flashinfer_allreduce_unavailable = True
            logger.warning(
                "flashinfer.comm unified allreduce_fusion API is not available, "
                "falling back to standard implementation"
            )
    except (ImportError, AttributeError) as e:
        _flashinfer_allreduce_unavailable = True
        logger.warning(
            "flashinfer.comm allreduce_fusion API is not available (%s), "
            "falling back to standard implementation",
            e,
        )

    try:
        from flashinfer.comm.mnnvl import TorchDistBackend

        class _FixedTorchDistBackend(TorchDistBackend):
            """Workaround for FlashInfer TorchDistBackend issues.

            1. bcast fix: TorchDistBackend.bcast passes the in-group rank
               directly as `src` to broadcast_object_list, which expects a
               global rank.
            2. Graph-capture fix: initialize with NCCL device_group (so
               the backend derives correct device_idx / GPU mapping), but
               broadcast via GLOO cpu_group (to avoid NCCL collectives
               that interfere with CUDA graph capture).
            """

            def __init__(self, device_group, cpu_group):
                super().__init__(group=device_group)
                self._cpu_group = cpu_group

            def bcast(self, data, root):
                import torch.distributed as dist

                group_ranks = dist.get_process_group_ranks(self._cpu_group)
                global_root = group_ranks[root]
                object_list = [data]
                dist.broadcast_object_list(
                    object_list, src=global_root, group=self._cpu_group
                )
                return object_list[0]

        _TorchDistBackend = _FixedTorchDistBackend
    except ImportError:
        logger.debug(
            "flashinfer.comm.mnnvl.TorchDistBackend is not available, "
            "allreduce fusion will use the default process group"
        )

    try:
        from flashinfer.comm.mnnvl import CommBackend

        class TorchDistributedCommBackend(CommBackend):
            """
            Use torch distributed instead of MPI to set up flashinfer MNNVL
            workspaces during initialization.
            """

            def __init__(self, group: ProcessGroup):
                self._group = group

            def Get_rank(self) -> int:
                return self._group.rank()

            def Get_size(self) -> int:
                return self._group.size()

            def allgather(self, data: int):
                gathered = [None] * self.Get_size()
                dist.all_gather_object(gathered, data, group=self._group)
                return gathered

            def bcast(self, data, root: int = 0):
                """Broadcast a picklable Python object from root to all ranks."""
                obj_list = [data]
                dist.broadcast_object_list(obj_list, src=root, group=self._group)
                return obj_list[0]

            def barrier(self):
                dist.barrier(group=self._group)

            def Split(self, color: int, key: int):
                # No need to split; we already use the proper group.
                return self._group

        _mnnvl_comm_backend = TorchDistributedCommBackend
    except ImportError:
        _mnnvl_comm_backend = None


# FlashInfer allreduce fusion backend support matrix for
# --flashinfer-allreduce-fusion-backend:
#
#   Backend   | SM103 | SM100 | SM90        | Single-Node | Multi-Node |
#   --------- | ----- | ----- | ----------- | ----------- | ---------- |
#   trtllm    | Yes   | Yes   | Yes         | Yes         | No         |
#   mnnvl     | Yes   | Yes   | Single-node | Yes         | Blackwell  |
#
# FlashInfer allreduce fusion requires SM90 or SM10X. auto resolves to mnnvl
# on Blackwell (SM100/SM103) systems (single- and multi-node) and to trtllm on
# SM90 single-node systems. SM90 multi-node and non-SM90/SM10X configurations
# are rejected. Either mnnvl or trtllm can be requested explicitly on
# single-node systems, and mnnvl additionally on Blackwell multi-node.


def is_flashinfer_allreduce_unavailable() -> bool:
    return _flashinfer_allreduce_unavailable


def _make_flashinfer_workspace_allocation_prop(cuda_driver):
    from flashinfer.comm.mnnvl import is_mnnvl_fabric_supported

    handle_types = cuda_driver.CUmemAllocationHandleType
    if is_mnnvl_fabric_supported(torch.cuda.current_device()):
        handle_type = handle_types.CU_MEM_HANDLE_TYPE_FABRIC
    else:
        handle_type = handle_types.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

    prop = cuda_driver.CUmemAllocationProp()
    prop.requestedHandleTypes = handle_type
    prop.type = cuda_driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location = cuda_driver.CUmemLocation()
    prop.location.type = cuda_driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = torch.cuda.current_device()
    prop.allocFlags.gpuDirectRDMACapable = 1
    return prop


def _flashinfer_trtllm_workspace_allocation_sizes(
    cuda_driver,
    prop,
    world_size: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
) -> list[int]:
    """Mirror FlashInfer TRTLLM SymmDeviceMemory local allocation sizes."""
    elem_size = 4 if dtype == torch.float32 else 2
    buffer_size = world_size * max_token_num * hidden_dim * 2
    flag_size = world_size * 256 * 4

    max_comm_size = 2147483647 & ~((1 << 21) - 1)
    lamport_comm_size = min(
        world_size * max_token_num * hidden_dim * elem_size,
        max_comm_size,
    )
    lamport_buffer_size = lamport_comm_size * 3

    # trtllm_create_ipc_workspace_for_all_reduce_fusion rounds each logical
    # buffer to 2 MiB before passing it to SymmDeviceMemory.
    buffer_sizes = (
        ceil_align(size, 1 << 21)
        for size in (buffer_size, flag_size, lamport_buffer_size)
    )

    signal_pad_size = 2048
    allocation_sizes = []
    for buffer_size in buffer_sizes:
        err, alloc_granularity = cuda_driver.cuMemGetAllocationGranularity(
            prop,
            cuda_driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        )
        if err != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(
                "cuMemGetAllocationGranularity failed for FlashInfer "
                f"workspace preflight: {err}"
            )

        allocation_size = ceil_align(buffer_size + signal_pad_size, alloc_granularity)

        mc_prop = cuda_driver.CUmulticastObjectProp()
        mc_prop.numDevices = world_size
        mc_prop.size = allocation_size
        mc_prop.handleTypes = prop.requestedHandleTypes

        err, mc_granularity = cuda_driver.cuMulticastGetGranularity(
            mc_prop,
            cuda_driver.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_RECOMMENDED,
        )
        if err != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(
                "cuMulticastGetGranularity failed for FlashInfer "
                f"workspace preflight: {err}"
            )

        allocation_size = ceil_align(allocation_size, mc_granularity)
        allocation_sizes.append(allocation_size)
    return allocation_sizes


# TWOSHOT splits each lamport buffer into two stages and addresses the second at
# buffer_size_bytes / 2, so that half-offset is only 16B-aligned -- what the
# kernel's vector accesses need -- when the buffer itself is 32B-aligned.
_MNNVL_TWOSHOT_STAGE_ALIGNMENT = 32
# FlashInfer rounds the multicast allocation up to a fixed granularity and then
# takes buffer_size_bytes = floor(alloc / 3) // 16 * 16, which lands on a
# 32B-misaligned size for roughly every third granularity step. Asking for more
# tokens moves to the next step, so a couple of retries always finds a usable
# one; grow by a quarter so each retry clears a step outright.
_MNNVL_ALIGNMENT_RETRIES = 3


def _mnnvl_twoshot_stage_is_aligned(workspace) -> bool:
    """Whether ``workspace`` can run the MNNVL TWOSHOT kernel without faulting.

    Non-MNNVL backends size their buffers differently and are unaffected.
    """
    if workspace.backend != "mnnvl":
        return True
    return workspace.buffer_size_bytes % _MNNVL_TWOSHOT_STAGE_ALIGNMENT == 0


def _create_workspace_aligned_for_twoshot(create_kw: dict) -> Tuple[object, int]:
    """Create a workspace whose TWOSHOT stage offset is aligned.

    Returns ``(workspace, max_token_num)``, where ``max_token_num`` is the size
    actually allocated -- retries grow it past a misaligned granularity step.

    Without this, an unlucky (max_token_num, hidden_dim) pair produces a buffer
    whose second TWOSHOT stage starts on a 16B-misaligned address and the kernel
    kills the process with cudaErrorMisalignedAddress on the first
    TWOSHOT-sized input. ONESHOT (tiny inputs) is unaffected, so warmup can
    sail past it. hidden_dim=7168 at max_token_num=16384 reproduces it.

    Every rank requests the same sizes and the allocator is deterministic, so
    all ranks walk the same retry sequence and stay in lockstep through the
    handle exchanges inside each attempt. If that ever stops holding, the symptom
    is a hang inside the handle exchange rather than an error, because the ranks
    would be on different attempts of the loop.
    """
    max_token_num = create_kw["max_token_num"]
    previous_buffer_size = None
    for attempt in range(_MNNVL_ALIGNMENT_RETRIES):
        workspace = _create_allreduce_fusion_workspace(
            **{**create_kw, "max_token_num": max_token_num}
        )
        if _mnnvl_twoshot_stage_is_aligned(workspace):
            if attempt:
                # Not free: a larger buffer costs proportionally more lamport
                # clearing on every all-reduce that uses it.
                logger.warning(
                    "FlashInfer MNNVL workspace grew from the requested "
                    "max_token_num=%d to %d to reach a %dB-aligned TWOSHOT "
                    "stage offset; the larger buffer makes every all-reduce "
                    "through it somewhat slower.",
                    create_kw["max_token_num"],
                    max_token_num,
                    _MNNVL_TWOSHOT_STAGE_ALIGNMENT,
                )
            return workspace, max_token_num

        buffer_size = workspace.buffer_size_bytes
        # A quarter more tokens normally clears a granularity step. When it does
        # not -- the step is coarse relative to the request -- stepping again by a
        # quarter would burn a retry on the identical buffer, so double instead.
        if buffer_size == previous_buffer_size:
            grown = max_token_num * 2
        else:
            grown = -(-max_token_num * 5 // 4)
        logger.warning(
            "FlashInfer MNNVL workspace for max_token_num=%d has a "
            "%dB-misaligned lamport buffer (%d bytes); retrying with "
            "max_token_num=%d (attempt %d/%d).",
            max_token_num,
            _MNNVL_TWOSHOT_STAGE_ALIGNMENT,
            buffer_size,
            grown,
            attempt + 1,
            _MNNVL_ALIGNMENT_RETRIES,
        )
        workspace.destroy()
        previous_buffer_size = buffer_size
        max_token_num = grown

    # initialize() catches this and disables fusion for the process, which is the
    # intended outcome: falling back to an ordinary all-reduce is correct, just
    # slower. Nothing is required of the operator.
    raise RuntimeError(
        "Could not allocate a FlashInfer MNNVL workspace whose TWOSHOT stage "
        f"offset is {_MNNVL_TWOSHOT_STAGE_ALIGNMENT}B-aligned after "
        f"{_MNNVL_ALIGNMENT_RETRIES} attempts; disabling allreduce fusion. "
        "The trtllm backend sizes its workspace differently and is unaffected."
    )


def _probe_cumem_create_sequence(cuda_driver, allocation_sizes, prop) -> bool:
    handles = []
    try:
        for allocation_size in allocation_sizes:
            err, handle = cuda_driver.cuMemCreate(allocation_size, prop, 0)
            if err != cuda_driver.CUresult.CUDA_SUCCESS:
                return False
            handles.append(handle)
        return True
    finally:
        for handle in reversed(handles):
            cuda_driver.cuMemRelease(handle)


def _preflight_check_workspace_memory(
    world_size: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    cpu_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> bool:
    """Collectively decide whether to enter FlashInfer workspace creation.

    FlashInfer TRTLLM workspaces allocate several SymmDeviceMemory buffers and
    then exchange handles across ranks. If one rank fails local cuMemCreate and
    exits while peers enter handle exchange, peers can hang until the watchdog
    aborts. Probe the same handle type and allocation sequence first, then vote
    on a CPU group so all ranks proceed or skip together.
    """
    import torch.distributed as dist

    group = cpu_group
    if group is None:
        tp_group = get_tp_group()
        if tp_group.world_size <= 1:
            return True
        group = tp_group.cpu_group

    allocation_sizes = []
    try:
        cuda_driver = get_cuda_driver_bindings()
        prop = _make_flashinfer_workspace_allocation_prop(cuda_driver)
        allocation_sizes = _flashinfer_trtllm_workspace_allocation_sizes(
            cuda_driver,
            prop,
            world_size,
            max_token_num,
            hidden_dim,
            dtype,
        )
        local_ok = _probe_cumem_create_sequence(cuda_driver, allocation_sizes, prop)
    except Exception as e:
        logger.warning(
            "FlashInfer workspace preflight probe failed (%s). "
            "Skipping allreduce fusion.",
            e,
        )
        local_ok = False

    flag = torch.tensor([1 if local_ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.BAND, group=group)

    logger.debug(
        "FlashInfer workspace preflight [rank %s]: probe=%.2f GB, "
        "local_probe=%s, vote=%s",
        dist.get_rank(group=group),
        sum(allocation_sizes) / 1e9,
        "OK" if local_ok else "FAIL",
        "PROCEED" if flag.item() == 1 else "SKIP",
    )
    if flag.item() == 0:
        logger.warning(
            "FlashInfer workspace preflight: cuMemCreate probe failed on at "
            "least one rank. Skipping allreduce fusion to avoid cross-rank "
            "desync inside the flashinfer collective."
        )
        return False
    return True


class FlashInferWorkspaceManager:
    """
    Manages FlashInfer's unified allreduce workspace.
    Supports trtllm and mnnvl backends via create_allreduce_fusion_workspace().
    """

    def __init__(self):
        self.workspace = None
        self.world_size = None
        self.rank = None
        self.group = None
        self.max_token_num = None
        self.hidden_dim = None
        self.dtype = None
        self.initialized = False
        # Track max sizes ever requested so the workspace only grows (fewer recreates)
        self._max_token_num_seen: Optional[int] = None
        self._max_hidden_dim_seen: Optional[int] = None
        self._logged_init = False
        self._workspace_size_check_kwarg = None
        self._workspace_size_check_strategy_type = None
        self._workspace_reinitialization_allowed = True
        # Largest token count the allocated buffer can actually service, probed
        # once at creation. Allocator rounding usually makes this far larger than
        # the requested max_token_num.
        self.max_supported_token_num: Optional[int] = None

    def _freeze(self) -> None:
        """Keep device addresses stable for CUDA graphs captured with this workspace."""
        self._workspace_reinitialization_allowed = False

    def _probe_max_supported_token_num(
        self, hidden_dim: int, dtype: torch.dtype
    ) -> int:
        """Binary-search the largest row count the buffer covers.

        The backends express capacity in bytes (MNNVL) or in
        max_token_num * hidden_dim (trtllm); both requirements are monotone in
        the row count, so a search over is_buffer_size_sufficient() finds the
        exact boundary without duplicating either formula. Runs once per
        allocation, so the per-forward gate is a plain integer compare.
        """

        def covers(token_num: int) -> bool:
            return self.is_buffer_size_sufficient(
                token_num=token_num, hidden_dim=hidden_dim, dtype=dtype
            )

        if not covers(1):
            return 0
        # Double until it stops covering, keeping `low` on the last value that
        # did, then bisect the open interval (low, high). Generous ceiling: real
        # prefill token budgets sit far below it.
        ceiling = 1 << 22
        low, high = 1, 2
        while high <= ceiling and covers(high):
            low, high = high, high * 2
        while high - low > 1:
            mid = (low + high) // 2
            if covers(mid):
                low = mid
            else:
                high = mid
        return low

    def _configure_workspace_size_check(self):
        """Cache the backend-specific size-check API for this workspace."""
        size_check_params = inspect.signature(
            self.workspace.is_buffer_size_sufficient
        ).parameters
        if "use_oneshot" in size_check_params:
            self._workspace_size_check_kwarg = "use_oneshot"
            self._workspace_size_check_strategy_type = None
        elif "strategy" in size_check_params:
            strategy_default = size_check_params["strategy"].default
            self._workspace_size_check_kwarg = "strategy"
            self._workspace_size_check_strategy_type = type(strategy_default)
        else:
            self._workspace_size_check_kwarg = None
            self._workspace_size_check_strategy_type = None

    def initialize(
        self,
        world_size: int,
        rank: int,
        max_token_num: int,
        hidden_dim: int,
        backend: str = "auto",
        group: Optional[ProcessGroup] = None,
        use_fp32_lamport: bool = False,
        dtype: Optional[torch.dtype] = None,
        use_oneshot: Optional[bool] = None,
        device_group: Optional["torch.distributed.ProcessGroup"] = None,
        cpu_group: Optional["torch.distributed.ProcessGroup"] = None,
    ):
        """Initialize workspace using FlashInfer's unified API."""
        global _flashinfer_allreduce_unavailable

        # Reuse existing workspace if it already covers this problem size
        if (
            self.initialized
            and self.world_size == world_size
            and self.is_buffer_size_sufficient(
                token_num=max_token_num,
                hidden_dim=hidden_dim,
                dtype=dtype or torch.bfloat16,
                use_oneshot=use_oneshot,
            )
        ):
            return

        # Captured graphs retain the workspace's device addresses. Once frozen,
        # an unsupported larger eager shape must fall back instead of replacing
        # the allocation underneath those graphs.
        if not self._workspace_reinitialization_allowed:
            return

        # Track the high-water mark so allocations only grow before capture.
        self._max_token_num_seen = max(max_token_num, self._max_token_num_seen or 0)
        self._max_hidden_dim_seen = max(hidden_dim, self._max_hidden_dim_seen or 0)

        # Same world_size but buffer too small: free old workspace before creating new
        if self.initialized and self.world_size == world_size:
            self.cleanup()

        if _flashinfer_comm is None or _create_allreduce_fusion_workspace is None:
            logger.warning(
                "FlashInfer comm not available, skipping workspace initialization"
            )
            return

        self.cleanup()

        if not _preflight_check_workspace_memory(
            world_size=world_size,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            cpu_group=cpu_group,
        ):
            _flashinfer_allreduce_unavailable = True
            self.workspace = None
            self.initialized = False
            return

        # Determine GPUs per node for MNNVL topology detection
        gpus_per_node = None
        node_pg = cpu_group if cpu_group is not None else group
        if node_pg is not None:
            gpus_per_node = sum(in_the_same_node_as(node_pg, source_rank=0))
        comm_backend = None
        if (
            _TorchDistBackend is not None
            and device_group is not None
            and cpu_group is not None
        ):
            comm_backend = _TorchDistBackend(
                device_group=device_group, cpu_group=cpu_group
            )
        elif _mnnvl_comm_backend is not None and group is not None:
            comm_backend = _mnnvl_comm_backend(group)

        try:
            requested_token_num = max(max_token_num, self._max_token_num_seen or 0)
            alloc_hidden_dim = max(hidden_dim, self._max_hidden_dim_seen or 0)
            create_kw = dict(
                backend=backend,
                world_size=world_size,
                rank=rank,
                max_token_num=requested_token_num,
                hidden_dim=alloc_hidden_dim,
                dtype=dtype or torch.bfloat16,
                gpus_per_node=gpus_per_node,
            )
            if (
                _flashinfer_create_workspace_supports_comm_backend
                and comm_backend is not None
            ):
                create_kw["comm_backend"] = comm_backend
            if _flashinfer_create_workspace_supports_group:
                # Pin the symmetric-memory rendezvous to the actual
                # subgroup. Without this, flashinfer >=0.6.10 falls back
                # to WORLD and TP/EP/CP subgroup peers get addressed
                # incorrectly (kernel hangs in cuda-graph warmup).
                create_kw["group"] = device_group
            if use_oneshot is not None:
                create_kw["force_oneshot_support"] = bool(use_oneshot)
            if use_fp32_lamport:
                create_kw["use_fp32_lamport"] = True
            self.workspace, alloc_token_num = _create_workspace_aligned_for_twoshot(
                create_kw
            )
            # An alignment retry allocates more than was asked for; record what
            # was actually taken so a later re-init cannot shrink below it.
            self._max_token_num_seen = max(
                alloc_token_num, self._max_token_num_seen or 0
            )
            self._configure_workspace_size_check()
            self.world_size = world_size
            self.rank = rank
            self.group = (device_group, cpu_group)
            self.max_token_num = alloc_token_num
            self.hidden_dim = alloc_hidden_dim
            self.dtype = dtype or torch.bfloat16
            self.initialized = True
            self.max_supported_token_num = self._probe_max_supported_token_num(
                hidden_dim=self.hidden_dim, dtype=self.dtype
            )

            backend_name = getattr(self.workspace, "backend", "unknown")
            if not self._logged_init:
                logger.info(
                    f"FlashInfer AllReduce Fusion enabled and workspace initialized: "
                    f"backend={backend_name}, rank={rank}, world_size={world_size}, "
                    f"max_token_num={self.max_token_num}, hidden_dim={self.hidden_dim}"
                )
                self._logged_init = True
            else:
                logger.debug(
                    f"FlashInfer workspace re-initialized: backend={backend_name}, "
                    f"rank={rank}, world_size={world_size}"
                )
        except Exception as e:
            _flashinfer_allreduce_unavailable = True
            logger.warning(
                f"Failed to initialize FlashInfer workspace (backend={backend}): {e}. "
                "Disabling flashinfer allreduce fusion permanently."
            )
            self.workspace = None
            self._workspace_size_check_kwarg = None
            self._workspace_size_check_strategy_type = None
            self.initialized = False
            return

    def is_buffer_size_sufficient(
        self,
        token_num: int,
        hidden_dim: int,
        dtype: torch.dtype,
        use_oneshot: Optional[bool] = None,
    ) -> bool:
        if not self.initialized or self.workspace is None:
            return False
        try:
            check_kw = dict(
                tp_size=self.world_size,
                num_tokens=token_num,
                hidden_dim=hidden_dim,
                dtype=dtype,
            )
            if self._workspace_size_check_kwarg == "use_oneshot":
                check_kw["use_oneshot"] = use_oneshot
            elif (
                self._workspace_size_check_kwarg == "strategy"
                and use_oneshot is not None
            ):
                # FlashInfer's MNNVL workspace expresses the same choice with
                # an enum-valued `strategy` argument rather than `use_oneshot`.
                check_kw["strategy"] = getattr(
                    self._workspace_size_check_strategy_type,
                    "ONESHOT" if use_oneshot else "TWOSHOT",
                )
            return self.workspace.is_buffer_size_sufficient(**check_kw)
        except Exception as e:
            logger.debug(f"FlashInfer workspace size check failed: {e}")
            # Fallback: some backends may not implement is_buffer_size_sufficient;
            # reuse if within our allocated dimensions.
            if (
                self.max_token_num is not None
                and self.hidden_dim is not None
                and token_num <= self.max_token_num
                and hidden_dim <= self.hidden_dim
            ):
                return True
            return False

    def cleanup(self):
        """Clean up workspace."""
        if self.workspace is not None:
            try:
                if hasattr(self.workspace, "destroy"):
                    self.workspace.destroy()
            except Exception as e:
                logger.warning(f"Failed to cleanup FlashInfer workspace: {e}")
            finally:
                self.workspace = None
                self._workspace_size_check_kwarg = None
                self._workspace_size_check_strategy_type = None
                self.initialized = False
                self.world_size = None
                self.rank = None
                self.group = None
                self.max_token_num = None
                self.hidden_dim = None
                self.dtype = None
                self.max_supported_token_num = None
                self._logged_init = False


def _get_workspace_manager(use_attn_tp_group: bool) -> FlashInferWorkspaceManager:
    """The per-group fusion workspace manager; the instances live on
    ``ctx.resources`` (one per comm group, created lazily)."""
    from sglang.srt.runtime_context import get_resources

    buffers = get_resources().buffers
    name = (
        "flashinfer_fusion_attn_tp_workspace"
        if use_attn_tp_group
        else "flashinfer_fusion_moe_tp_workspace"
    )
    manager = buffers.get(name)
    if manager is None:
        manager = FlashInferWorkspaceManager()
        buffers[name] = manager
    return manager


def _sync_allreduce_unavailable_across_tp():
    """Synchronize _flashinfer_allreduce_unavailable across all TP ranks.

    If workspace initialization fails on any rank, all ranks must agree to
    disable fusion. Otherwise ranks diverge during CUDA graph capture: some
    use FlashInfer fusion (skipping custom allreduce), others fall back to
    standard allreduce (calling register_buffer collectives), causing a hang
    in register_graph_buffers.
    """
    global _flashinfer_allreduce_unavailable
    try:
        import torch.distributed as dist

        tp_group = get_tp_group()
        if tp_group.world_size <= 1:
            return
        flag = torch.tensor(
            [1 if _flashinfer_allreduce_unavailable else 0],
            dtype=torch.int32,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=tp_group.cpu_group)
        if flag.item() > 0 and not _flashinfer_allreduce_unavailable:
            _flashinfer_allreduce_unavailable = True
            logger.warning(
                "FlashInfer allreduce fusion disabled globally because "
                "workspace initialization failed on at least one rank."
            )
    except Exception as e:
        logger.debug(f"Failed to sync flashinfer unavailable flag: {e}")


def _fusion_group(use_attn_tp_group: bool):
    """The (world_size, rank, coordinator) the fusion workspace rendezvouses on.

    Shared by workspace creation and by the capacity predicate so the two cannot
    disagree about which peers a workspace belongs to.
    """
    parallel = get_parallel()
    if use_attn_tp_group:
        return parallel.attn_tp_size, parallel.attn_tp_rank, get_attn_tp_group()
    if parallel.moe_ep_size > 1:
        return parallel.moe_ep_size, parallel.moe_ep_rank, get_moe_ep_group()
    return parallel.moe_tp_size, parallel.moe_tp_rank, get_moe_tp_group()


def fusion_workspace_covers(num_tokens: int, *, use_attn_tp_group: bool) -> bool:
    """Whether the fusion workspace can service ``num_tokens`` rows.

    Compares against a token count probed once at workspace creation
    (``max_supported_token_num``) rather than calling into FlashInfer per
    forward: the answer must be a pure function of static state so it stays
    usable while Dynamo traces, and so the fused and non-fused decisions taken at
    different call sites for the same batch always agree.
    """
    manager = _get_workspace_manager(use_attn_tp_group)
    return (
        manager.max_supported_token_num is not None
        and 0 < num_tokens <= manager.max_supported_token_num
    )


def can_use_flashinfer_allreduce_fusion(
    input_tensor: torch.Tensor, *, use_attn_tp_group: bool
) -> bool:
    """Whether ``flashinfer_allreduce_residual_rmsnorm`` can service this shape.

    Split out from the kernel call for the same reason as
    ``can_use_flashinfer_allreduce``: the custom op is opaque to Dynamo and its
    schema promises two tensors, so a shape-dependent ``return None, None``
    inside it is invisible at trace time. The fake impl hands the tracer real
    tensors, the ``is not None`` test folds to True, the fallback is compiled
    out, and the op then raises ``expected Tensor()`` at runtime. Deciding out
    here keeps the op on paths that always produce tensors.

    Every check is rank-invariant: a rank that declines while its peers enter
    the kernel mismatches and hangs. The unavailable flag and the workspace are
    cross-rank synced at init; the rest are pure functions of group identity and
    of tensor metadata, which is identical on every rank of the group.
    """
    if _flashinfer_allreduce_unavailable or _flashinfer_comm is None:
        return False

    if not supports_collective_topology(use_attn_tp_group=use_attn_tp_group):
        return False

    if input_tensor.ndim != 2 or not input_tensor.is_contiguous():
        return False

    manager = _get_workspace_manager(use_attn_tp_group)
    if not manager.initialized or manager.workspace is None:
        return False

    # The workspace is only usable on the peers it was rendezvoused with. Under
    # hybrid EP+TP the EP and MoE-TP groups have equal world size but pair
    # different ranks, so a mismatch reduces over the wrong peers and silently
    # returns garbage. Freezing means initialize() can no longer self-heal this,
    # so it has to be checked here.
    world_size, rank, coordinator = _fusion_group(use_attn_tp_group)
    if world_size <= 1:
        return False
    if (
        manager.world_size != world_size
        or manager.rank != rank
        or manager.group != (coordinator.device_group, coordinator.cpu_group)
    ):
        return False

    # The workspace was allocated for one (hidden_dim, dtype); a second runner in
    # the process with a different pair must not reuse it.
    if manager.hidden_dim != input_tensor.shape[-1] or manager.dtype != (
        input_tensor.dtype
    ):
        return False

    # Size check stays last: it reads the token dim, which is symbolic under
    # Dynamo, so statically-off configs short-circuit before reaching it (same
    # ordering rule as apply_flashinfer_allreduce_fusion).
    return fusion_workspace_covers(
        input_tensor.shape[0], use_attn_tp_group=use_attn_tp_group
    )


def ensure_workspace_initialized(
    max_token_num: int = 2048,
    hidden_dim: int = 4096,
    use_fp32_lamport: bool = False,
    dtype: Optional[torch.dtype] = None,
    token_num: Optional[int] = None,
    use_oneshot: Optional[bool] = None,
    use_attn_tp_group: bool = True,
):
    """Ensure workspace is initialized."""
    if _flashinfer_allreduce_unavailable:
        return False

    if not is_flashinfer_available() or _flashinfer_comm is None:
        return False

    world_size, rank, coordinator = _fusion_group(use_attn_tp_group)

    # Always pass the coordinator's groups: flashinfer >=0.6.10 reads the
    # rendezvous group from `group=...` (falling back to WORLD when None),
    # so leaving it None silently rendezvouses on WORLD and the kernel ends
    # up addressing the wrong peers in TP/EP/CP subgroup setups.
    device_group = coordinator.device_group
    cpu_group = coordinator.cpu_group

    if world_size <= 1:
        return False

    workspace_manager = _get_workspace_manager(use_attn_tp_group)
    token_num = token_num or max_token_num
    group_key = (device_group, cpu_group)
    effective_dtype = dtype or torch.bfloat16
    server_args = get_server_args()
    backend = resolve_flashinfer_allreduce_fusion_backend(server_args)
    if backend is None:
        return False

    if (
        not workspace_manager.initialized
        or workspace_manager.world_size != world_size
        or workspace_manager.rank != rank
        or workspace_manager.group != group_key
        or not workspace_manager.is_buffer_size_sufficient(
            token_num=token_num,
            hidden_dim=hidden_dim,
            dtype=effective_dtype,
            use_oneshot=use_oneshot,
        )
    ):
        workspace_manager.initialize(
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            backend=backend,
            group=cpu_group,
            use_fp32_lamport=use_fp32_lamport,
            dtype=dtype,
            use_oneshot=use_oneshot,
            device_group=device_group,
            cpu_group=cpu_group,
        )

        _sync_allreduce_unavailable_across_tp()

    # Whether the workspace *covers* this shape is decided by
    # can_use_flashinfer_allreduce_fusion(), in plain Python outside the custom
    # op. Reporting it from here would put a shape-dependent verdict inside the
    # op, which Dynamo cannot see past (see that function's docstring).
    return workspace_manager.initialized


def fake_flashinfer_allreduce_residual_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    max_token_num: int = 16384,
    use_oneshot: Optional[bool] = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
    use_attn_tp_group: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)
    return norm_out, residual_out


def supports_collective_topology(*, use_attn_tp_group: bool) -> bool:
    """Whether one fused all-reduce covers every reduction this site needs.

    The attention-TP site always reduces over a single group. The MoE site does
    too, unless expert and MoE-tensor parallelism are both on: that reduction
    spans _MOE_EP and then _MOE_TP, two disjoint groups, and FlashInfer fuses
    exactly one collective.
    """
    if use_attn_tp_group:
        return True
    parallel = get_parallel()
    return not (parallel.moe_ep_size > 1 and parallel.moe_tp_size > 1)


@register_custom_op(
    mutates_args=["input_tensor", "residual", "weight"],
    fake_impl=fake_flashinfer_allreduce_residual_rmsnorm,
)
def flashinfer_allreduce_residual_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    max_token_num: int = 2048,
    use_oneshot: Optional[bool] = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
    use_attn_tp_group: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use FlashInfer's unified fused allreduce + residual + RMS norm operation.
    Automatically selects between trtllm and mnnvl backends based on topology
    and hardware (controlled by --flashinfer-allreduce-fusion-backend).

    Args:
        input_tensor: Input tensor that needs allreduce
        residual: Residual tensor
        weight: RMS norm weight
        eps: RMS norm epsilon
        max_token_num: Maximum token number
        use_oneshot: Whether to use oneshot mode
        trigger_completion_at_end: Whether to trigger completion at end
        fp32_acc: Whether to use fp32 precision
        use_attn_tp_group: If True, use attention TP group; otherwise use MoE TP group

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (norm_output, residual_output)
    """
    if not is_flashinfer_available() or _flashinfer_comm is None:
        logger.debug(
            "FlashInfer not available, falling back to standard implementation"
        )
        return None, None

    if use_attn_tp_group:
        world_size = get_parallel().attn_tp_size
    else:
        # FlashInfer can fuse one collective. Hybrid expert + MoE tensor
        # parallelism requires EP followed by MoE-TP, so defer both to the
        # ordinary fallback instead of silently omitting the second reduction.
        if not supports_collective_topology(use_attn_tp_group=False):
            return None, None
        if get_parallel().moe_ep_size > 1:
            world_size = get_parallel().moe_ep_size
        else:
            world_size = get_parallel().moe_tp_size

    if world_size <= 1:
        logger.debug("Single GPU, no need for allreduce fusion")
        return None, None

    assert input_tensor.shape[0] <= max_token_num
    if (
        not input_tensor.is_contiguous()
        or not residual.is_contiguous()
        or not weight.is_contiguous()
    ):
        logger.debug("Non-contiguous tensors, skipping FlashInfer allreduce fusion")
        return None, None

    if not ensure_workspace_initialized(
        max_token_num=max_token_num,
        hidden_dim=input_tensor.shape[-1],
        use_fp32_lamport=(input_tensor.dtype == torch.float32),
        dtype=input_tensor.dtype,
        token_num=input_tensor.shape[0],
        use_oneshot=use_oneshot,
        use_attn_tp_group=use_attn_tp_group,
    ):
        logger.debug("FlashInfer workspace not available")
        return None, None

    workspace_manager = _get_workspace_manager(use_attn_tp_group)
    if workspace_manager.workspace is None:
        logger.debug("FlashInfer workspace is None")
        return None, None

    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)

    kwargs = dict(
        input=input_tensor,
        workspace=workspace_manager.workspace,
        pattern=_flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
        launch_with_pdl=True,
        residual_out=residual_out,
        norm_out=norm_out,
        residual_in=residual,
        rms_gamma=weight,
        rms_eps=eps,
        use_oneshot=use_oneshot,
        fp32_acc=fp32_acc,
    )
    if _flashinfer_allreduce_supports_trigger_completion:
        kwargs["trigger_completion_at_end"] = trigger_completion_at_end
    _flashinfer_comm.allreduce_fusion(**kwargs)

    return norm_out, residual_out


def can_use_flashinfer_allreduce(
    input_: torch.Tensor,
    *,
    use_attn_tp_group: bool,
    expected_world_size: int,
    expected_group_key: Tuple[Optional[ProcessGroup], Optional[ProcessGroup]],
) -> bool:
    """Whether ``flashinfer_allreduce`` can service this all-reduce.

    Split out from the kernel call so the decision happens in plain Python,
    outside the custom op: the op is opaque to Dynamo and has to return a
    tensor, so it cannot carry a data-dependent fallback of its own.

    ``expected_world_size`` / ``expected_group_key`` describe the calling group;
    the workspace is only usable when it was rendezvoused on exactly those peers.

    Every check here is rank-invariant by construction, and must stay that way:
    a rank that quietly falls back to NCCL while its peers enter the kernel
    mismatches and hangs. The unavailable flag and workspace initialization are
    cross-rank synced at init time (``_sync_allreduce_unavailable_across_tp``);
    the rest are pure functions of the group identity and of tensor metadata,
    which is identical on every rank of the group.
    """
    if _flashinfer_allreduce_unavailable or _flashinfer_comm is None:
        return False

    if input_.ndim != 2 or not input_.is_contiguous():
        return False

    workspace_manager = _get_workspace_manager(use_attn_tp_group)
    if not workspace_manager.initialized or workspace_manager.workspace is None:
        return False

    # The two workspaces are keyed by attention-TP vs MoE, but the MoE one
    # rendezvouses on either the EP or the MoE-TP group depending on topology.
    # Under hybrid EP+TP those groups have equal world size but pair different
    # ranks, so a mismatch here reduces across the wrong peers and silently
    # produces garbage rather than failing. Require an exact match.
    if (
        workspace_manager.world_size != expected_world_size
        or workspace_manager.group != expected_group_key
    ):
        return False

    # Size checks stay last: they read the token dim, which is symbolic under
    # Dynamo, so statically-off configs must short-circuit before reaching them
    # (same ordering rule as apply_flashinfer_allreduce_fusion).
    token_num, hidden_dim = input_.shape
    if torch.compiler.is_compiling():
        # Don't call into the flashinfer workspace object while tracing. The
        # workspace was allocated for (max_token_num, hidden_dim, dtype) and
        # vetted by is_buffer_size_sufficient() at init; the requirement is
        # monotone in token_num/hidden_dim, so staying within the allocation
        # (including dtype) is a conservative stand-in here.
        return (
            workspace_manager.max_token_num is not None
            and workspace_manager.hidden_dim is not None
            and workspace_manager.dtype is not None
            and token_num <= workspace_manager.max_token_num
            and hidden_dim <= workspace_manager.hidden_dim
            and workspace_manager.dtype == input_.dtype
        )

    return workspace_manager.is_buffer_size_sufficient(
        token_num=token_num,
        hidden_dim=hidden_dim,
        dtype=input_.dtype,
    )


def flashinfer_allreduce(
    input_: torch.Tensor,
    *,
    use_attn_tp_group: bool,
) -> torch.Tensor:
    """Allreduce-only FlashInfer kAllReduce.

    Assumes ``can_use_flashinfer_allreduce`` returned True for this call; there
    is no fallback here. Kernel errors are deliberately not caught -- swallowing
    one would put this rank on NCCL while its peers stay in the kernel, which
    mismatch-hangs instead of failing.
    """
    workspace_manager = _get_workspace_manager(use_attn_tp_group)

    output = torch.empty_like(input_)
    kwargs = dict(
        input=input_,
        workspace=workspace_manager.workspace,
        pattern=_flashinfer_comm.AllReduceFusionPattern.kAllReduce,
        launch_with_pdl=True,
        fp32_acc=False,
        output=output,
    )
    if _flashinfer_allreduce_supports_trigger_completion:
        kwargs["trigger_completion_at_end"] = False
    _flashinfer_comm.allreduce_fusion(**kwargs)
    return output


def pre_initialize_workspaces(
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    use_oneshot: Optional[bool] = None,
):
    """Pre-initialize flashinfer workspaces before CUDA graph capture.

    This must be called before graph capture to avoid collective operations
    (broadcasts, barriers) inside the graph capture context, which can
    deadlock with custom_all_reduce.register_graph_buffers.
    """
    if _flashinfer_allreduce_unavailable or _flashinfer_comm is None:
        return

    # Initialize MoE workspace
    ensure_workspace_initialized(
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
        use_oneshot=use_oneshot,
        use_attn_tp_group=False,
    )

    # Initialize attention workspace
    ensure_workspace_initialized(
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
        use_oneshot=use_oneshot,
        use_attn_tp_group=True,
    )


def freeze_flashinfer_workspaces() -> None:
    """Prevent graph-captured workspace addresses from being replaced.

    Only freezes managers that already hold an allocation. Freezing an empty one
    would pin it empty for the process, and _get_workspace_manager() creates on
    demand, so freezing unconditionally could invent a manager just to disable
    it.

    Note the managers live on the process-global resource table while this runs
    per ModelRunner, so with speculative decoding the target freezes the
    workspace the draft will later share. That is fine as long as the draft fits
    inside it -- EAGLE/MTP drafts consume the target's hidden states and so carry
    the same hidden_dim, and their verify shapes are covered by the decode floor.
    A draft with a larger hidden_dim would find the workspace frozen and run
    unfused (correct, not faster).
    """
    from sglang.srt.runtime_context import get_resources

    buffers = get_resources().buffers
    for name in (
        "flashinfer_fusion_attn_tp_workspace",
        "flashinfer_fusion_moe_tp_workspace",
    ):
        manager = buffers.get(name)
        if manager is not None and manager.initialized:
            manager._freeze()


def cleanup_flashinfer_workspace():
    from sglang.srt.runtime_context import get_resources

    buffers = get_resources().buffers
    for name in (
        "flashinfer_fusion_attn_tp_workspace",
        "flashinfer_fusion_moe_tp_workspace",
    ):
        manager = buffers.get(name)
        if manager is not None:
            manager.cleanup()
