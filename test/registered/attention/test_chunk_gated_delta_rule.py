import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from sglang.srt.layers.attention.linear import gdn_backend
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.mem_cache.layout.page_major import (
    build_page_major_mamba_views,
    mamba_entry_bytes,
)
from sglang.srt.utils import get_device, is_hip
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cuda_ci,
    register_xpu_ci,
)

register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=11, stage="stage-b", runner_config="1-gpu-large-amd")
register_xpu_ci(est_time=900, suite="stage-b-test-1-gpu-xpu")


@unittest.skipIf(
    not (torch.cuda.is_available() or torch.xpu.is_available()),
    "Test requires CUDA or XPU",
)
class TestChunkGatedDeltaRule(unittest.TestCase):
    """Test chunk_gated_delta_rule against token-by-token fused_recurrent reference."""

    ATOL = 2e-2
    RTOL = 1e-2

    def _run_reference(self, pool_init, cache_indices, q, k, v, g, beta):
        """Per-batch token-by-token reference using fused_recurrent_gated_delta_rule.

        initial_state shape: [N, H, V, K] (native layout on this branch).
        """
        B = cache_indices.shape[0]
        T_per_seq = q.shape[1] // B
        pool = pool_init.clone()
        h_cur = pool[cache_indices].contiguous().clone()

        o_list = []
        for b in range(B):
            sl = slice(b * T_per_seq, (b + 1) * T_per_seq)
            o_b, h_b = fused_recurrent_gated_delta_rule(
                q=q[0, sl].unsqueeze(0),
                k=k[0, sl].unsqueeze(0),
                v=v[0, sl].unsqueeze(0),
                g=g[0, sl].unsqueeze(0),
                beta=beta[0, sl].unsqueeze(0),
                initial_state=h_cur[b : b + 1],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            o_list.append(o_b)
            h_cur[b] = h_b[0]

        pool[cache_indices] = h_cur
        return torch.cat(o_list, dim=1), pool

    def _run_chunk(
        self,
        pool_init,
        cache_indices,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        *,
        clone_pool=True,
    ):
        """Run chunk_gated_delta_rule with native [V, K] pool."""
        pool = pool_init.clone() if clone_pool else pool_init
        o, _, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=pool,
            initial_state_indices=cache_indices,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        return o, pool

    def _make_inputs(self, *, B, T_per_seq, H, K, V, seed=42):
        device = get_device()
        dtype = torch.bfloat16
        T = B * T_per_seq
        torch.manual_seed(seed)
        q = torch.randn(1, T, H, K, dtype=dtype, device=device)
        k = torch.randn(1, T, H, K, dtype=dtype, device=device)
        v = torch.randn(1, T, H, V, dtype=dtype, device=device)
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, T, H, dtype=dtype, device=device)
        )
        beta = torch.sigmoid(torch.randn(1, T, H, dtype=dtype, device=device))
        cu_seqlens = torch.arange(0, T + 1, T_per_seq, dtype=torch.long, device=device)
        return q, k, v, g, beta, cu_seqlens

    def _make_page_major_pool(self, *, layers, slots, H, K, V):
        device = get_device()
        conv_shapes = [(2, 3)]
        dtype = torch.float32
        entry_bytes = mamba_entry_bytes(
            layer_num=layers,
            conv_state_shapes=conv_shapes,
            conv_dtype=dtype,
            temporal_state_shape=(H, V, K),
            temporal_dtype=dtype,
        )
        raw = torch.zeros(slots * entry_bytes, dtype=torch.uint8, device=device)
        conv_views, temporal = build_page_major_mamba_views(
            raw,
            layer_num=layers,
            conv_state_shapes=conv_shapes,
            conv_dtype=dtype,
            temporal_state_shape=(H, V, K),
            temporal_dtype=dtype,
            max_slots=slots,
        )
        return conv_views, temporal

    def _check_shape(
        self, B, T_per_seq, H, K, V, pool_size, sequential_indices=False, seed=42
    ):
        """Run correctness check for one (B, T_per_seq, H, K, V, pool_size) config."""
        device = get_device()
        dtype = torch.bfloat16
        T = B * T_per_seq

        torch.manual_seed(seed)

        if sequential_indices:
            cache_indices = torch.arange(B, dtype=torch.int32, device=device)
        else:
            perm = torch.randperm(pool_size, device=device)[:B]
            cache_indices = perm.to(torch.int32)

        pool_init = (
            torch.randn(pool_size, H, V, K, dtype=torch.float32, device=device) * 0.1
        )
        cu_seqlens = torch.zeros(B + 1, dtype=torch.long, device=device)
        cu_seqlens[1:] = (
            torch.arange(1, B + 1, dtype=torch.long, device=device) * T_per_seq
        )

        q = torch.randn(1, T, H, K, dtype=dtype, device=device)
        k = torch.randn(1, T, H, K, dtype=dtype, device=device)
        v = torch.randn(1, T, H, V, dtype=dtype, device=device)
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, T, H, dtype=dtype, device=device)
        )
        beta = torch.sigmoid(torch.randn(1, T, H, dtype=dtype, device=device))

        o_ref, pool_ref = self._run_reference(
            pool_init, cache_indices, q, k, v, g, beta
        )
        o_new, pool_new = self._run_chunk(
            pool_init, cache_indices, q, k, v, g, beta, cu_seqlens
        )

        self.assertTrue(
            torch.allclose(
                o_ref.float(), o_new.float(), atol=self.ATOL, rtol=self.RTOL
            ),
            f"Output mismatch: max_diff="
            f"{(o_ref.float() - o_new.float()).abs().max().item():.2e}",
        )

        ref_slots = pool_ref[cache_indices].contiguous()
        new_slots = pool_new[cache_indices].contiguous()
        self.assertTrue(
            torch.allclose(
                ref_slots.float(), new_slots.float(), atol=self.ATOL, rtol=self.RTOL
            ),
            f"State mismatch: max_diff="
            f"{(ref_slots.float() - new_slots.float()).abs().max().item():.2e}",
        )

    # ------------------------------------------------------------------
    # Production-style configs (Qwen3-Next)
    # ------------------------------------------------------------------
    def test_production_nt1(self):
        self._check_shape(B=4, T_per_seq=64, H=16, K=128, V=128, pool_size=32)

    def test_production_nt2(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=128, V=128, pool_size=32)

    def test_production_nt4(self):
        self._check_shape(B=4, T_per_seq=256, H=16, K=128, V=128, pool_size=32)

    # ------------------------------------------------------------------
    # Batch size sweep
    # ------------------------------------------------------------------
    def test_batch_1(self):
        self._check_shape(B=1, T_per_seq=128, H=16, K=128, V=128, pool_size=32)

    def test_batch_2(self):
        self._check_shape(B=2, T_per_seq=128, H=16, K=128, V=128, pool_size=32)

    def test_batch_8(self):
        self._check_shape(B=8, T_per_seq=128, H=16, K=128, V=128, pool_size=64)

    def test_batch_16(self):
        self._check_shape(B=16, T_per_seq=64, H=16, K=128, V=128, pool_size=128)

    def test_batch_32(self):
        self._check_shape(B=32, T_per_seq=32, H=16, K=128, V=128, pool_size=256)

    # ------------------------------------------------------------------
    # Head count sweep
    # ------------------------------------------------------------------
    def test_heads_4(self):
        self._check_shape(B=4, T_per_seq=128, H=4, K=128, V=128, pool_size=32)

    def test_heads_8(self):
        self._check_shape(B=4, T_per_seq=128, H=8, K=128, V=128, pool_size=32)

    def test_heads_32(self):
        self._check_shape(B=4, T_per_seq=128, H=32, K=128, V=128, pool_size=32)

    def test_heads_64(self):
        self._check_shape(B=4, T_per_seq=128, H=64, K=128, V=128, pool_size=32)

    # ------------------------------------------------------------------
    # K != V  (exercises that [V,K] != [K,V] byte-order matters)
    # ------------------------------------------------------------------
    def test_dim_64x64(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=64, V=64, pool_size=32)

    def test_dim_k_lt_v(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=64, V=128, pool_size=32)

    def test_dim_k_gt_v(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=128, V=64, pool_size=32)

    @unittest.skipIf(
        is_hip(),
        "K=V=256 head dim exceeds the FLA chunk triton kernel's shared-memory "
        "budget on ROCm (out-of-resource at launch); smaller head dims pass.",
    )
    def test_dim_256x256(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=256, V=256, pool_size=32)

    # ------------------------------------------------------------------
    # Short sequences (T < chunk_size=64)
    # ------------------------------------------------------------------
    def test_seqlen_1(self):
        self._check_shape(B=4, T_per_seq=1, H=16, K=128, V=128, pool_size=32)

    def test_seqlen_7(self):
        self._check_shape(B=4, T_per_seq=7, H=16, K=128, V=128, pool_size=32)

    def test_seqlen_16(self):
        self._check_shape(B=4, T_per_seq=16, H=16, K=128, V=128, pool_size=32)

    def test_seqlen_32(self):
        self._check_shape(B=4, T_per_seq=32, H=16, K=128, V=128, pool_size=32)

    # ------------------------------------------------------------------
    # Multi-chunk and large pool
    # ------------------------------------------------------------------
    def test_multi_chunk_nt8(self):
        self._check_shape(B=4, T_per_seq=512, H=16, K=128, V=128, pool_size=32)

    def test_large_pool(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=128, V=128, pool_size=512)

    # ------------------------------------------------------------------
    # Long prompts (many chunks; regression test for cross-chunk errors)
    # ------------------------------------------------------------------
    def test_long_prompt(self):
        for B, T_per_seq in [(1, 1024), (1, 1536), (1, 2048), (2, 1024)]:
            with self.subTest(T_per_seq=T_per_seq):
                self._check_shape(
                    B=B, T_per_seq=T_per_seq, H=16, K=128, V=128, pool_size=32
                )

    # ------------------------------------------------------------------
    # Combined stress
    # ------------------------------------------------------------------
    def test_stress(self):
        self._check_shape(B=32, T_per_seq=128, H=32, K=128, V=128, pool_size=256)

    # ------------------------------------------------------------------
    # Sequential-index variants (pool_size == B, indices = 0..B-1)
    # ------------------------------------------------------------------
    def test_seq_idx_b4(self):
        self._check_shape(
            B=4,
            T_per_seq=128,
            H=16,
            K=128,
            V=128,
            pool_size=4,
            sequential_indices=True,
        )

    def test_seq_idx_b8(self):
        self._check_shape(
            B=8,
            T_per_seq=128,
            H=16,
            K=128,
            V=128,
            pool_size=8,
            sequential_indices=True,
        )

    def test_seq_idx_h32(self):
        self._check_shape(
            B=4,
            T_per_seq=128,
            H=32,
            K=128,
            V=128,
            pool_size=4,
            sequential_indices=True,
        )

    def test_seq_idx_h64(self):
        self._check_shape(
            B=4,
            T_per_seq=128,
            H=64,
            K=128,
            V=128,
            pool_size=4,
            sequential_indices=True,
        )

    def test_seq_idx_stress(self):
        self._check_shape(
            B=32,
            T_per_seq=128,
            H=32,
            K=128,
            V=128,
            pool_size=32,
            sequential_indices=True,
        )

    def test_page_major_pool_is_updated_in_place(self):
        """A strided page-major state pool must retain the prefill update."""
        B, T_per_seq, H, K, V = 2, 16, 2, 64, 64
        _conv, temporal = self._make_page_major_pool(layers=2, slots=5, H=H, K=K, V=V)
        page_pool = temporal[1]
        page_pool.normal_(mean=0.0, std=0.1)
        initial = page_pool.contiguous().clone()
        indices = torch.tensor([4, 1], dtype=torch.int32, device=get_device())
        inputs = self._make_inputs(B=B, T_per_seq=T_per_seq, H=H, K=K, V=V)

        ref_output, ref_pool = self._run_chunk(initial, indices, *inputs)
        output, _ = self._run_chunk(page_pool, indices, *inputs, clone_pool=False)

        torch.testing.assert_close(output, ref_output, atol=self.ATOL, rtol=self.RTOL)
        torch.testing.assert_close(
            page_pool[indices.long()],
            ref_pool[indices.long()],
            atol=self.ATOL,
            rtol=self.RTOL,
        )
        self.assertGreater(
            (page_pool[indices.long()] - initial[indices.long()]).abs().max().item(),
            0.0,
        )

    def test_negative_padding_index_uses_zero_state_without_pool_write(self):
        """A padded -1 row must neither address nor modify surrounding pool rows."""
        B, T_per_seq, H, K, V = 2, 128, 2, 64, 64
        conv_views, temporal = self._make_page_major_pool(
            layers=2, slots=5, H=H, K=K, V=V
        )
        conv_views[0].fill_(17.0)
        temporal.normal_(mean=0.0, std=0.1)
        page_pool = temporal[1]
        before_conv = conv_views[0].clone()
        before_temporal = temporal.clone()
        indices = torch.tensor([2, -1], dtype=torch.int32, device=get_device())
        inputs = self._make_inputs(B=B, T_per_seq=T_per_seq, H=H, K=K, V=V, seed=123)

        output, _ = self._run_chunk(page_pool, indices, *inputs, clone_pool=False)

        reference_pool = before_temporal[1].contiguous().clone()
        reference_pool[3].zero_()
        reference_indices = torch.tensor([2, 3], dtype=torch.int32, device=get_device())
        reference_output, _ = self._run_chunk(
            reference_pool, reference_indices, *inputs
        )
        torch.testing.assert_close(
            output, reference_output, atol=self.ATOL, rtol=self.RTOL
        )
        torch.testing.assert_close(conv_views[0], before_conv)
        torch.testing.assert_close(temporal[0], before_temporal[0])
        for slot in (0, 1, 3, 4):
            torch.testing.assert_close(page_pool[slot], before_temporal[1, slot])

    def test_rejects_noncontiguous_inner_state_dimensions(self):
        """Only the pool-slot stride may differ from contiguous state layout."""
        B, T_per_seq, H, K, V = 1, 7, 2, 64, 64
        state = torch.empty(3, H, K, V, device=get_device()).transpose(-1, -2)
        indices = torch.tensor([1], dtype=torch.int32, device=get_device())
        inputs = self._make_inputs(B=B, T_per_seq=T_per_seq, H=H, K=K, V=V)

        with self.assertRaisesRegex(ValueError, "contiguous.*H, V, K"):
            self._run_chunk(state, indices, *inputs, clone_pool=False)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA causal convolution required")
    def test_conv_padding_gather_scatter_does_not_touch_last_pool_row(self):
        """GDN prefill must not scatter a padded conv row into physical slot -1."""
        device = get_device()
        layers, slots, dim, width = 2, 5, 8, 4
        entry_bytes = mamba_entry_bytes(
            layer_num=layers,
            conv_state_shapes=[(dim, width - 1)],
            conv_dtype=torch.bfloat16,
            temporal_state_shape=(1, 1, 1),
            temporal_dtype=torch.float32,
        )
        raw = torch.zeros(slots * entry_bytes, dtype=torch.uint8, device=device)
        conv_views, _ = build_page_major_mamba_views(
            raw,
            layer_num=layers,
            conv_state_shapes=[(dim, width - 1)],
            conv_dtype=torch.bfloat16,
            temporal_state_shape=(1, 1, 1),
            temporal_dtype=torch.float32,
            max_slots=slots,
        )
        conv_pool = conv_views[0][1]
        conv_pool.normal_()
        before = conv_pool.clone()
        state_indices = torch.tensor([2, -1], dtype=torch.int32, device=device)
        ssm_pool = torch.zeros(slots, 1, 4, 2, device=device)
        cache_params = SimpleNamespace(conv=[conv_pool], temporal=ssm_pool)
        backend = object.__new__(GDNAttnBackend)
        backend.req_to_token_pool = SimpleNamespace(
            mamba2_layer_cache=lambda _layer_id: cache_params
        )
        backend.forward_metadata = SimpleNamespace(
            query_start_loc=torch.tensor([0, 8, 16], dtype=torch.int32, device=device),
            mamba_cache_indices=state_indices,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            retrieve_parent_token=None,
            has_mamba_track_mask=False,
            state_checkpoint_cu_starts=None,
            num_state_checkpoints=None,
            state_checkpoint_every_n_tokens=None,
        )
        backend.kernel_dispatcher = MagicMock()
        backend.kernel_dispatcher.extend.return_value = (
            torch.empty(1, 16, 1, 4, device=device),
            None,
            None,
        )
        layer = SimpleNamespace(
            layer_id=1,
            conv_weights=torch.randn(dim, width, dtype=torch.bfloat16, device=device),
            bias=None,
            activation="silu",
            q_dim=2,
            k_dim=2,
            v_dim=4,
            num_q_heads=1,
            num_k_heads=1,
            num_v_heads=1,
            head_q_dim=2,
            head_k_dim=2,
            head_v_dim=4,
            A_log=torch.empty(1, device=device),
            dt_bias=torch.empty(1, device=device),
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_target_verify=lambda: False),
            extend_prefix_lens=torch.tensor([1, 0], device=device),
            extend_seq_lens_cpu=torch.tensor([8, 8]),
        )
        mixed_qkv = torch.randn(16, dim, dtype=torch.bfloat16, device=device)
        gating = torch.empty(16, 1, dtype=torch.bfloat16, device=device)

        with (
            patch.object(gdn_backend, "is_cuda", return_value=False),
            patch.object(gdn_backend, "is_hip", return_value=False),
            patch.object(
                gdn_backend,
                "fused_gdn_gating",
                return_value=(gating, gating),
            ),
        ):
            backend.forward_extend(layer, forward_batch, mixed_qkv, gating, gating)

        for slot in (0, 1, 3, 4):
            torch.testing.assert_close(conv_pool[slot], before[slot])
        self.assertGreater((conv_pool[2] - before[2]).abs().max().item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA Triton frontend required")
    def test_xpu_kernel_frontend_accepts_fp32_state_and_bf16_scratch(self):
        """XPU source must not merge FP32 pool and BF16 scratch pointer types."""
        from sglang.srt.hardware_backend.xpu.kernels.fla.chunk_delta_h import (
            chunk_gated_delta_rule_fwd_h,
        )

        device = get_device()
        T, H, K, V = 128, 2, 64, 64
        k = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16)
        w = torch.randn_like(k)
        u = torch.randn(1, T, H, V, device=device, dtype=torch.bfloat16)
        g = torch.randn(1, T, H, device=device, dtype=torch.float32)
        state = torch.randn(3, H, V, K, device=device, dtype=torch.float32)
        indices = torch.tensor([1], device=device, dtype=torch.int32)
        cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int64)

        h, _ = chunk_gated_delta_rule_fwd_h(
            k,
            w,
            u,
            g=g,
            initial_state=state,
            initial_state_indices=indices,
            cu_seqlens=cu_seqlens,
        )
        torch.cuda.synchronize()

        self.assertEqual(h.dtype, torch.bfloat16)
        self.assertEqual(state.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
