import logging
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.flashinfer_utils import (
    create_mnnvl_comm_backend,
)

logger = logging.getLogger(__name__)

_flashinfer_comm = None
_flashinfer_ar_available = False
try:
    import flashinfer.comm as flashinfer_comm

    if hasattr(flashinfer_comm, "allreduce_fusion") and hasattr(
        flashinfer_comm, "create_allreduce_fusion_workspace"
    ):
        _flashinfer_comm = flashinfer_comm
        _flashinfer_ar_available = True
except ImportError:
    pass

MiB = 1024 * 1024

# Max size of the communicated tensor by world size and GPU capability.
# Adopted from vLLM thresholds.
# TODO(mmangkad): Tune these thresholds for SGLang, since optimal values may
# differ from vLLM based on runtime/scheduling behavior.
_FI_ALLREDUCE_MAX_SIZE_MB: dict[int, dict[int, float]] = {
    90: {
        2: 64,
        4: 2,
        8: 0.5,
    },
    100: {
        2: 64,
        4: 32,
        8: 1,
    },
}


def _get_device_capability() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


class FlashInferAllReduce:
    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        backend: str = "auto",
    ):
        self.disabled = True
        self.workspace = None
        self.max_num_tokens = 0
        self.max_workspace_size = None
        self.hidden_dim = None
        self.dtype = None

        if not _flashinfer_ar_available or _flashinfer_comm is None:
            logger.info(
                "FlashInfer allreduce disabled: flashinfer comm API unavailable."
            )
            return

        if not torch.cuda.is_available():
            logger.info("FlashInfer allreduce disabled: CUDA is unavailable.")
            return

        self.group = group
        self.world_size = dist.get_world_size(group=self.group)
        self.rank = dist.get_rank(group=self.group)
        self.device = device
        self.backend = backend

        if self.world_size == 1:
            return

        capability = _get_device_capability()
        self.max_workspace_size = _FI_ALLREDUCE_MAX_SIZE_MB.get(capability, {}).get(
            self.world_size
        )
        if self.max_workspace_size is None:
            logger.warning(
                "FlashInfer allreduce disabled: unsupported world_size=%d for SM=%s.",
                self.world_size,
                str(capability),
            )
            return

        self.max_workspace_size = int(self.max_workspace_size * MiB)
        self.disabled = False

    def _max_tokens_for(self, hidden_dim: int, dtype: torch.dtype) -> int:
        assert self.max_workspace_size is not None
        if hidden_dim <= 0:
            return 0
        element_size = torch._utils._element_size(dtype)
        return self.max_workspace_size // (hidden_dim * element_size)

    def _can_fit_workspace(
        self,
        num_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
    ) -> bool:
        if self.max_workspace_size is None:
            return False
        max_tokens = self._max_tokens_for(hidden_dim, dtype)
        return max_tokens > 0 and num_tokens <= max_tokens

    def _create_workspace(
        self,
        max_token_num: int,
        hidden_dim: int,
        dtype: torch.dtype,
    ) -> bool:
        assert _flashinfer_comm is not None

        workspace_kwargs = dict(
            backend=self.backend,
            world_size=self.world_size,
            rank=self.rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
        )

        if self.backend in ("auto", "mnnvl"):
            comm_backend = create_mnnvl_comm_backend(self.group)
            if comm_backend is not None:
                workspace_kwargs["comm_backend"] = comm_backend

        self.workspace = _flashinfer_comm.create_allreduce_fusion_workspace(
            **workspace_kwargs
        )
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        return self.workspace is not None

    def _ensure_workspace(
        self,
        num_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
    ) -> bool:
        if self.workspace is not None:
            if self.hidden_dim == hidden_dim and self.dtype == dtype:
                try:
                    if self.workspace.is_buffer_size_sufficient(
                        tp_size=self.world_size,
                        num_tokens=num_tokens,
                        hidden_dim=hidden_dim,
                        dtype=dtype,
                    ):
                        return True
                except Exception as e:
                    logger.debug(
                        "FlashInfer workspace size check failed; recreating workspace: %s",
                        e,
                    )
            self.destroy()

        max_tokens = self._max_tokens_for(hidden_dim, dtype)
        if max_tokens <= 0 or num_tokens > max_tokens:
            return False

        self.max_num_tokens = max_tokens
        try:
            return self._create_workspace(
                max_token_num=max_tokens,
                hidden_dim=hidden_dim,
                dtype=dtype,
            )
        except Exception as e:
            logger.warning(
                "Failed to initialize FlashInfer allreduce workspace: %s. "
                "Disabling FlashInfer allreduce.",
                e,
            )
            self.disabled = True
            self.workspace = None
            return False

    def should_use_fi_ar(self, input_tensor: torch.Tensor) -> bool:
        if self.disabled:
            return False

        if not input_tensor.is_cuda or not input_tensor.is_contiguous():
            return False

        if len(input_tensor.shape) != 2:
            return False

        num_tokens, hidden_dim = input_tensor.shape
        return self._can_fit_workspace(num_tokens, hidden_dim, input_tensor.dtype)

    def all_reduce(self, input_tensor: torch.Tensor) -> torch.Tensor:
        assert _flashinfer_comm is not None
        if not self.should_use_fi_ar(input_tensor):
            raise RuntimeError(
                "FlashInfer allreduce was invoked for an unsupported tensor shape."
            )

        num_tokens, hidden_dim = input_tensor.shape
        if not self._ensure_workspace(num_tokens, hidden_dim, input_tensor.dtype):
            raise RuntimeError("FlashInfer allreduce workspace initialization failed.")

        try:
            return _flashinfer_comm.allreduce_fusion(
                input=input_tensor,
                workspace=self.workspace,
                pattern=_flashinfer_comm.AllReduceFusionPattern.kAllReduce,
            )
        except Exception as e:
            logger.warning(
                "FlashInfer allreduce execution failed: %s. "
                "Disabling FlashInfer allreduce.",
                e,
            )
            self.destroy()
            self.disabled = True
            raise RuntimeError("FlashInfer allreduce execution failed.") from e

    def destroy(self):
        if self.workspace is not None:
            try:
                self.workspace.destroy()
            except Exception as e:
                logger.debug("Failed to destroy FlashInfer workspace: %s", e)
            self.workspace = None
            self.hidden_dim = None
            self.dtype = None
