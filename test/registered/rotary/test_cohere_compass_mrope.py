import unittest
from unittest.mock import patch

import torch
from transformers.models.cohere_compass import CohereCompassTextConfig

from sglang.kernels.spec import KernelBackend
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.models.cohere_compass import get_text_rotary_emb
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
# Runs the triton serving kernel, which the CPU suite skips.
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

# CohereLabs/North-Micro-Vision-Instruct's sliding-window RoPE.
HEAD_DIM = 128
ROPE_THETA = 50000
MROPE_SECTION = [24, 20, 20]
MAX_POSITIONS = 64


def make_config() -> CohereCompassTextConfig:
    return CohereCompassTextConfig(
        hidden_size=HEAD_DIM * 2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        num_hidden_layers=2,
        max_position_embeddings=MAX_POSITIONS,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": ROPE_THETA,
                "mrope_section": MROPE_SECTION,
                "mrope_interleaved": True,
            },
            "full_attention": None,
        },
    )


def reference_rotate(positions, q, k):
    """transformers 5.16 CohereCompassRotaryEmbedding + apply_rotary_pos_emb.

    The checkpoint pins transformers 5.16; 5.17 changed the lane layout and
    regressed the model, so the installed module cannot serve as reference.
    """
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float) / HEAD_DIM)
    )
    freqs = positions[..., None].float() * inv_freq  # (3, n, head_dim // 2)
    # apply_interleaved_mrope: start from t, then overwrite every third lane.
    freqs_t = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        idx = slice(offset, MROPE_SECTION[dim] * 3, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    emb = torch.cat((freqs_t, freqs_t), dim=-1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]

    def rotate(x):
        x = x.view(x.shape[0], -1, HEAD_DIM)
        half = torch.cat((-x[..., HEAD_DIM // 2 :], x[..., : HEAD_DIM // 2]), -1)
        return (x * cos + half * sin).flatten(1)

    return rotate(q), rotate(k)


def text_image_text_positions() -> torch.Tensor:
    """(t, h, w) rows for 5 text tokens, a 1x3x4 image grid, then 3 text tokens."""
    text = torch.arange(5).expand(3, -1)
    h, w = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")
    image = torch.stack([torch.zeros(12, dtype=torch.long), h.flatten(), w.flatten()])
    tail = (torch.arange(3) + 9).expand(3, -1)
    return torch.cat([text, image + 5, tail], dim=1)


class TestCohereCompassMRope(CustomTestCase):
    def setUp(self):
        cpu_patch = patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
        cpu_patch.start()
        self.addCleanup(cpu_patch.stop)
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        torch.manual_seed(0)
        self.rope = get_text_rotary_emb(make_config())

    def cases(self):
        yield KernelBackend.TORCH, "cpu"
        if torch.cuda.is_available():
            yield KernelBackend.TRITON, "cuda"

    def rotate(self, positions, q, k, backend, device):
        q_out, k_out = self.rope.to(device)(
            positions.to(device), q.to(device), k.to(device), backend=backend
        )
        return q_out.cpu(), k_out.cpu()

    def test_text_tokens_rotate_as_standard_rope(self):
        """With t == h == w, interleaved M-RoPE reduces to plain 1-D RoPE; a
        layout that permutes the rotary lanes (transformers 5.17) breaks this
        for text-only prompts too."""
        positions = torch.arange(MAX_POSITIONS - 1)
        q = torch.randn(positions.numel(), 2 * HEAD_DIM)
        k = torch.randn(positions.numel(), HEAD_DIM)
        plain = get_rope(
            HEAD_DIM, HEAD_DIM, MAX_POSITIONS, ROPE_THETA, True, None, torch.float32
        )
        cos, sin = plain.cos_sin_cache[positions].float().chunk(2, dim=-1)
        q_ref = apply_rotary_emb(q.view(-1, 2, HEAD_DIM), cos, sin, True).flatten(1)
        k_ref = apply_rotary_emb(k.view(-1, 1, HEAD_DIM), cos, sin, True).flatten(1)
        for backend, device in self.cases():
            with self.subTest(backend=backend.value):
                q_out, k_out = self.rotate(
                    positions.expand(3, -1), q.clone(), k.clone(), backend, device
                )
                torch.testing.assert_close(q_out, q_ref, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(k_out, k_ref, atol=1e-5, rtol=1e-5)

    def test_image_tokens_follow_reference_interleaving(self):
        positions = text_image_text_positions()
        q = torch.randn(positions.shape[1], 2 * HEAD_DIM)
        k = torch.randn(positions.shape[1], HEAD_DIM)
        q_ref, k_ref = reference_rotate(positions, q, k)
        for backend, device in self.cases():
            with self.subTest(backend=backend.value):
                q_out, k_out = self.rotate(
                    positions, q.clone(), k.clone(), backend, device
                )
                torch.testing.assert_close(q_out, q_ref, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(k_out, k_ref, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
