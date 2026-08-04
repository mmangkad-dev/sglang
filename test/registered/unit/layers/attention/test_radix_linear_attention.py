"""CPU tests for shared ``RadixLinearAttention`` state contracts."""

from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_plain_radix_linear_attention_has_unfused_kda_handoff_state():
    """KimiLinear uses this plain layer with the shared KDA backend."""
    layer = RadixLinearAttention(
        layer_id=0,
        num_q_heads=2,
        num_k_heads=2,
        num_v_heads=2,
        head_q_dim=4,
        head_k_dim=4,
        head_v_dim=4,
    )

    # These are the direct reads in KDA decode and target-verify. Their default
    # values select the generic unfused implementation.
    assert layer._k3_fused_decode_args is None
    assert layer._k3_onorm_gate is None
    assert layer._k3_onorm_consumed is False
