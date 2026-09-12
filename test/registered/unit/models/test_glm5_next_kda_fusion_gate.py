import os
import socket
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.glm5_next import (
    Glm5NextForConditionalGeneration,
    Glm5NextLinearAttention,
    _fused_qkvbfg_is_unquantized,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PREFIX = "model.layers.0.linear_attn"
# The six projections fused_qkvbfg_a_proj and fused_fg_b_proj are built from.
_KDA_PROJECTIONS = [
    f"{_PREFIX}.{name}"
    for name in (
        "q_proj",
        "k_proj",
        "v_proj",
        "b_proj",
        "f_a_proj",
        "g_a_proj",
        "f_b_proj",
        "g_b_proj",
    )
]


def _fp8_config(skipped):
    return Fp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "modules_to_not_convert": skipped,
            "packed_modules_mapping": (
                Glm5NextForConditionalGeneration.packed_modules_mapping
            ),
        }
    )


def test_unquantized_checkpoint_fuses_without_the_env_gate():
    with envs.SGLANG_OPT_GLM5_NEXT_FUSE_KDA_QKVBFG.override(False):
        assert _fused_qkvbfg_is_unquantized(quant_config=None, prefix=_PREFIX)


@pytest.mark.parametrize("gate", [False, True])
def test_quantized_kda_projections_never_fuse(gate):
    """Fusing genuinely quantized projections would feed the merged GEMM weights
    in the wrong format, so the gate alone must not be enough to enable it."""
    with envs.SGLANG_OPT_GLM5_NEXT_FUSE_KDA_QKVBFG.override(gate):
        assert not _fused_qkvbfg_is_unquantized(
            quant_config=_fp8_config([]), prefix=_PREFIX
        )


@pytest.mark.parametrize(
    "quantized",
    [
        pytest.param(["f_a_proj"], id="qkvbfg-group"),
        pytest.param(["f_b_proj"], id="fg-b-group"),
    ],
)
def test_mixed_precision_group_declines_instead_of_raising(quantized):
    """A fused group whose projections disagree on precision must fall back to
    the unfused path. Probing the fused name raises there instead."""
    skipped = [p for p in _KDA_PROJECTIONS if p.rsplit(".", 1)[1] not in quantized]
    with envs.SGLANG_OPT_GLM5_NEXT_FUSE_KDA_QKVBFG.override(True):
        assert not _fused_qkvbfg_is_unquantized(
            quant_config=_fp8_config(skipped), prefix=_PREFIX
        )


def test_fp8_checkpoint_that_skips_kda_fuses_only_when_gated():
    """GLM-5.3-Flash ships fp8 with every linear-attention projection in
    modules_to_not_convert, so a non-None quant_config does not imply these
    layers are quantized."""
    config = _fp8_config(_KDA_PROJECTIONS)
    with envs.SGLANG_OPT_GLM5_NEXT_FUSE_KDA_QKVBFG.override(True):
        assert _fused_qkvbfg_is_unquantized(quant_config=config, prefix=_PREFIX)
    with envs.SGLANG_OPT_GLM5_NEXT_FUSE_KDA_QKVBFG.override(False):
        assert not _fused_qkvbfg_is_unquantized(quant_config=config, prefix=_PREFIX)


@pytest.fixture(scope="module")
def gloo_world():
    """a one-rank cpu process group, so the column-parallel projections can build"""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    initialize_model_parallel(tensor_model_parallel_size=1, backend="gloo")
    yield
    destroy_model_parallel()
    destroy_distributed_environment()


def test_fused_projections_share_the_runtime_dtype(gloo_world):
    """The loader can override the checkpoint's declared dtype. Both fused
    projections must land on the runtime one, or the first forward hits
    'expected scalar type Half but found BFloat16'."""
    config = SimpleNamespace(
        dtype=torch.bfloat16,
        torch_dtype=torch.bfloat16,
        linear_attn_config={
            "short_conv_kernel_size": 4,
            "num_heads": 4,
            "head_dim": 16,
        },
    )
    with (
        get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0),
        set_default_torch_dtype(torch.float16),
    ):
        layer = Glm5NextLinearAttention(
            layer_idx=0,
            hidden_size=64,
            config=config,
            quant_config=None,
            prefix=_PREFIX,
        )
    assert layer.do_fuse_qkvbfg
    assert layer.fused_qkvbfg_a_proj.params_dtype == torch.float16
    assert layer.fused_fg_b_proj.weight.dtype == torch.float16


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
