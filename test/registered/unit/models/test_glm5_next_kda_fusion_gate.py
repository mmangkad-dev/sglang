import sys

import pytest

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.models.glm5_next import (
    Glm5NextForConditionalGeneration,
    _fused_qkvbfg_is_unquantized,
)
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


def test_fp8_checkpoint_that_skips_kda_fuses_only_when_gated():
    """GLM-5.3-Flash ships fp8 with every linear-attention projection in
    modules_to_not_convert, so a non-None quant_config does not imply these
    layers are quantized."""
    config = _fp8_config(_KDA_PROJECTIONS)
    with envs.SGLANG_OPT_GLM5_NEXT_FUSE_KDA_QKVBFG.override(True):
        assert _fused_qkvbfg_is_unquantized(quant_config=config, prefix=_PREFIX)
    with envs.SGLANG_OPT_GLM5_NEXT_FUSE_KDA_QKVBFG.override(False):
        assert not _fused_qkvbfg_is_unquantized(quant_config=config, prefix=_PREFIX)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
