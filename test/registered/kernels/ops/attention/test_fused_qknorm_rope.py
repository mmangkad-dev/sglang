import pytest
import torch

from sglang.kernels.ops.attention.fused_qknorm_rope import fused_qk_norm_rope
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@torch.inference_mode()
def _reference(
    qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    eps,
    q_weight,
    k_weight,
    base,
    is_neox,
    position_ids,
    partial_rotary_factor,
):
    q_size = num_heads_q * head_dim
    k_size = num_heads_k * head_dim
    v_size = num_heads_v * head_dim
    assert qkv.shape[1] == q_size + k_size + v_size

    q = qkv[:, :q_size]
    k = qkv[:, q_size : q_size + k_size]
    v = qkv[:, q_size + k_size :]

    q_norm = RMSNorm(hidden_size=head_dim, eps=eps).to(qkv.device, qkv.dtype)
    k_norm = RMSNorm(hidden_size=head_dim, eps=eps).to(qkv.device, qkv.dtype)
    q_norm.weight.copy_(q_weight)
    k_norm.weight.copy_(k_weight)
    q = q_norm(q.reshape(-1, head_dim)).view_as(q)
    k = k_norm(k.reshape(-1, head_dim)).view_as(k)

    rotary_emb = get_rope(
        head_dim,
        rotary_dim=head_dim,
        max_position=8192,
        base=base,
        is_neox_style=is_neox,
        rope_scaling=None,
        dual_chunk_attention_config=None,
        partial_rotary_factor=partial_rotary_factor,
    ).to(qkv.device)
    q, k = rotary_emb(position_ids, q, k, fused_set_kv_buffer_arg=None)
    return torch.cat((q, k, v), dim=1)


@pytest.mark.parametrize(
    "head_dim,num_heads,num_tokens,is_neox,partial_rotary_factor",
    [
        (64, (16, 8, 8), 1, False, 1.0),
        (64, (12, 1, 1), 3, True, 0.5),
        (128, (32, 8, 8), 8, False, 0.5),
        (128, (40, 8, 8), 32, True, 1.0),
    ],
)
@torch.inference_mode()
def test_fused_qk_norm_rope_matches_reference(
    head_dim, num_heads, num_tokens, is_neox, partial_rotary_factor
):
    """Guard Q/K/V partitioning, RoPE layout, and partial rotary dimensions."""
    num_heads_q, num_heads_k, num_heads_v = num_heads
    hidden_size = sum(num_heads) * head_dim
    torch.manual_seed(0)
    qkv = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    expected_input = qkv.clone()
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda") + 100
    q_weight = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda") * 5.0
    k_weight = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda") * 5.0
    eps = 1e-5
    base = 10000.0

    fused_qk_norm_rope(
        qkv=qkv,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        eps=eps,
        q_weight=q_weight,
        k_weight=k_weight,
        base=base,
        is_neox=is_neox,
        position_ids=position_ids,
        factor=1.0,
        low=0.0,
        high=0.0,
        attention_factor=1.0,
        rotary_dim=int(head_dim * partial_rotary_factor),
    )
    with get_context().override_server_args():
        expected = _reference(
            expected_input,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_dim,
            eps,
            q_weight,
            k_weight,
            base,
            is_neox,
            position_ids,
            partial_rotary_factor,
        )

    torch.testing.assert_close(qkv, expected, rtol=5e-2, atol=1e-1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
