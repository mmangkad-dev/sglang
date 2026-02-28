import torch.distributed as dist

from sglang.srt.utils import is_flashinfer_available

if is_flashinfer_available():
    try:
        from flashinfer.comm.mnnvl import CommBackend
    except ImportError:
        CommBackend = object  # type: ignore[assignment,misc]
else:

    class CommBackend:
        """Placeholder base class when flashinfer is not available."""

        pass


def create_mnnvl_comm_backend(group: dist.ProcessGroup):
    """Create a mnnvl comm backend backed by torch.distributed process group."""
    try:
        from flashinfer.comm.mnnvl import TorchDistBackend

        return TorchDistBackend(group=group)
    except Exception:
        pass

    class TorchDistributedCommBackend(CommBackend):
        def __init__(self, group_: dist.ProcessGroup):
            self._group = group_

        def Get_rank(self) -> int:
            return self._group.rank()

        def Get_size(self) -> int:
            return self._group.size()

        def allgather(self, data: int):
            gathered = [None] * self.Get_size()
            dist.all_gather_object(gathered, data, group=self._group)
            return gathered

        def bcast(self, data, root: int = 0):
            obj_list = [data]
            dist.broadcast_object_list(obj_list, src=root, group=self._group)
            return obj_list[0]

        def Split(self, color: int, key: int):
            return self

        def barrier(self):
            dist.barrier(group=self._group)

    return TorchDistributedCommBackend(group)
