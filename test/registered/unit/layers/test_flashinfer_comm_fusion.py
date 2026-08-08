import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import flashinfer_comm_fusion as fusion
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-c", runner_config="4-gpu-h100")
register_cuda_ci(est_time=30, stage="base-c", runner_config="4-gpu-b200")
register_cuda_ci(est_time=30, stage="base-c", runner_config="4-gpu-gb300")


class _FakeWorkspace:
    def __init__(self, backend, world_size):
        self.backend = backend
        self.world_size = world_size

    def is_buffer_size_sufficient(self, **_kwargs):
        return True


class _FakeFlashInferComm:
    class AllReduceFusionPattern:
        kARResidualRMSNorm = object()

    def __init__(self):
        self.calls = []

    def create_allreduce_fusion_workspace(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeWorkspace(kwargs["backend"], kwargs["world_size"])

    def allreduce_fusion(
        self,
        *,
        input,
        workspace,
        residual_out,
        norm_out,
        residual_in,
        rms_gamma,
        rms_eps,
        **_kwargs,
    ):
        allreduced = input * workspace.world_size
        expected_residual = allreduced + residual_in
        variance = expected_residual.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        expected_norm = (
            expected_residual.to(torch.float32)
            * torch.rsqrt(variance + rms_eps)
            * rms_gamma.to(torch.float32)
        ).to(input.dtype)
        residual_out.copy_(expected_residual)
        norm_out.copy_(expected_norm)


def _torch_allreduce_residual_rmsnorm_baseline(
    input_tensor, residual, weight, world_size, eps
):
    allreduced = input_tensor * world_size
    residual_out = allreduced + residual
    variance = residual_out.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    norm_out = (
        residual_out.to(torch.float32)
        * torch.rsqrt(variance + eps)
        * weight.to(torch.float32)
    ).to(input_tensor.dtype)
    return norm_out, residual_out


class TestFlashInferCommFusion(CustomTestCase):
    def test_frozen_workspace_is_not_replaced_for_larger_shape(self):
        """A post-capture eager shape must not invalidate captured device pointers."""
        manager = fusion.FlashInferWorkspaceManager()
        workspace = _FakeWorkspace("trtllm", world_size=4)
        workspace.is_buffer_size_sufficient = lambda **kwargs: (
            kwargs["num_tokens"] <= 2048
        )
        manager.workspace = workspace
        manager.initialized = True
        manager.world_size = 4
        manager.max_token_num = 2048
        manager.hidden_dim = 4096
        manager._freeze()

        with patch.object(manager, "cleanup") as cleanup:
            manager.initialize(
                world_size=4,
                rank=0,
                max_token_num=4096,
                hidden_dim=4096,
                dtype=torch.bfloat16,
            )

        cleanup.assert_not_called()
        self.assertIs(manager.workspace, workspace)
        self.assertFalse(
            manager.is_buffer_size_sufficient(
                token_num=4096,
                hidden_dim=4096,
                dtype=torch.bfloat16,
            )
        )

    def test_failed_fusion_falls_back_to_allreduce_before_norm(self):
        """Workspace rejection must not silently omit the required collective."""
        from sglang.srt.layers import layernorm

        class _FakeNorm:
            variance_epsilon = 1e-6

            def forward(self, x, residual, post_residual_addition):
                self.inputs = (x, residual, post_residual_addition)
                return x + residual

        norm = _FakeNorm()
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        residual = torch.ones_like(x)
        weight = torch.ones(4)

        for use_attn_tp_group, moe_ep_size, moe_tp_size, collective_names in (
            (True, 1, 2, ("attention_tensor_model_parallel_all_reduce",)),
            (False, 2, 1, ("moe_expert_parallel_all_reduce",)),
            (False, 1, 2, ("moe_tensor_model_parallel_all_reduce",)),
            (
                False,
                2,
                2,
                (
                    "moe_expert_parallel_all_reduce",
                    "moe_tensor_model_parallel_all_reduce",
                ),
            ),
        ):
            collective_calls = []

            def all_reduce(tensor, *, name):
                collective_calls.append(name)
                return tensor * 2

            with (
                self.subTest(
                    use_attn_tp_group=use_attn_tp_group,
                    moe_ep_size=moe_ep_size,
                ),
                patch.object(layernorm, "_use_aiter", False),
                patch.object(
                    fusion,
                    "flashinfer_allreduce_residual_rmsnorm",
                    return_value=(None, None),
                ),
                patch(
                    "sglang.srt.distributed.attention_tensor_model_parallel_all_reduce",
                    side_effect=lambda tensor: all_reduce(tensor, name="attention"),
                ),
                patch(
                    "sglang.srt.distributed.moe_expert_parallel_all_reduce",
                    side_effect=lambda tensor: all_reduce(tensor, name="ep"),
                ),
                patch(
                    "sglang.srt.distributed.moe_tensor_model_parallel_all_reduce",
                    side_effect=lambda tensor: all_reduce(tensor, name="moe_tp"),
                ),
                get_parallel().override(
                    attn_tp_size=2,
                    moe_ep_size=moe_ep_size,
                    moe_tp_size=moe_tp_size,
                ),
            ):
                output = layernorm._forward_with_allreduce_fusion(
                    norm_module=norm,
                    x=x,
                    residual=residual,
                    post_residual_addition=None,
                    weight=weight,
                    use_attn_tp_group=use_attn_tp_group,
                )

            expected_calls = {
                "attention_tensor_model_parallel_all_reduce": "attention",
                "moe_expert_parallel_all_reduce": "ep",
                "moe_tensor_model_parallel_all_reduce": "moe_tp",
            }
            self.assertEqual(
                collective_calls, [expected_calls[name] for name in collective_names]
            )
            expected_x = x * (2 ** len(collective_names))
            torch.testing.assert_close(norm.inputs[0], expected_x)
            self.assertIsNone(norm.inputs[2])
            torch.testing.assert_close(output, expected_x + residual)

    def test_aiter_rejection_falls_back_to_full_tp(self):
        """AITER rejection must retain the full TP collective it attempted to fuse."""
        from sglang.srt.layers import layernorm

        norm = types.SimpleNamespace(
            variance_epsilon=1e-6,
            forward=lambda x, residual, _post: (x, residual),
        )
        x = torch.ones(2, 4)
        residual = torch.ones_like(x)

        with (
            patch.object(layernorm, "_use_aiter", True),
            patch(
                "sglang.srt.distributed.tensor_model_parallel_fused_allreduce_rmsnorm",
                return_value=None,
            ),
            patch(
                "sglang.srt.distributed.tensor_model_parallel_all_reduce",
                side_effect=lambda tensor: tensor * 8,
            ) as tp_all_reduce,
            patch(
                "sglang.srt.distributed.moe_expert_parallel_all_reduce"
            ) as ep_all_reduce,
            get_parallel().override(
                tp_size=8,
                attn_tp_size=1,
                moe_ep_size=1,
                moe_tp_size=1,
            ),
        ):
            output, _ = layernorm._forward_with_allreduce_fusion(
                norm_module=norm,
                x=x,
                residual=residual,
                post_residual_addition=None,
                weight=torch.ones(4),
                use_attn_tp_group=False,
            )

        tp_all_reduce.assert_called_once_with(x)
        ep_all_reduce.assert_not_called()
        torch.testing.assert_close(output, x * 8)

    def test_aux_capture_owns_snapshot_before_fallback_norm_mutation(self):
        """Aux capture must survive a later in-place mutation of the residual."""
        from sglang.srt.layers.communicator import LayerCommunicator

        class _Accumulator:
            copies_on_append = False

            def __init__(self):
                self.values = []

            def append(self, value):
                self.values.append(value)

        communicator = LayerCommunicator.__new__(LayerCommunicator)
        communicator.prepare_attn = lambda hidden, residual, *_args, **_kwargs: (
            hidden,
            residual,
        )
        communicator._communicate_simple_fn = (
            lambda *, hidden_states, **_kwargs: hidden_states
        )
        communicator._context = None

        residual = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        expected = residual.clone()
        accumulator = _Accumulator()
        _, returned_residual = communicator.prepare_attn_and_capture_last_layer_outputs(
            hidden_states=torch.zeros_like(residual),
            residual=residual,
            forward_batch=None,
            captured_last_layer_outputs=accumulator,
        )

        returned_residual.add_(100)
        torch.testing.assert_close(accumulator.values[0], expected)
        self.assertIsNot(accumulator.values[0], returned_residual)

    def test_hybrid_moe_parallelism_rejects_single_collective_fusion(self):
        """Hybrid EP+MoE-TP cannot use a fusion that performs only one reduction."""
        with get_parallel().override(moe_ep_size=2, moe_tp_size=2):
            self.assertFalse(
                fusion._supports_collective_topology(use_attn_tp_group=False)
            )
            self.assertTrue(
                fusion._supports_collective_topology(use_attn_tp_group=True)
            )

    def test_auto_backend_resolves_by_arch(self):
        single_node = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="auto", nnodes=1
        )
        multi_node = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="auto", nnodes=2
        )

        # Blackwell: mnnvl on both single-node and multi-node.
        with patch.object(fusion, "is_sm100_supported", return_value=True):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(single_node),
                "mnnvl",
            )
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node), "mnnvl"
            )

        # SM90: auto uses trtllm on single-node, multi-node is unsupported.
        with (
            patch.object(fusion, "is_sm100_supported", return_value=False),
            patch.object(fusion, "is_sm90_supported", return_value=True),
        ):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(single_node),
                "trtllm",
            )
            with self.assertRaises(ValueError):
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node)

        # Architectures outside SM90/SM10X are unsupported. Both pre-SM90
        # and post-SM10X devices (e.g. SM120) must fail closed.
        for arch in ("pre_sm90", "post_sm10x"):
            with (
                self.subTest(arch=arch),
                patch.object(fusion, "is_sm100_supported", return_value=False),
                patch.object(fusion, "is_sm90_supported", return_value=False),
            ):
                with self.assertRaises(ValueError):
                    fusion.resolve_flashinfer_allreduce_fusion_backend(single_node)
                with self.assertRaises(ValueError):
                    fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node)

    def test_explicit_backend_validation(self):
        single_node_mnnvl = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="mnnvl", nnodes=1
        )
        multi_node_mnnvl = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="mnnvl", nnodes=2
        )
        single_node_trtllm = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="trtllm", nnodes=1
        )
        multi_node_trtllm = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="trtllm", nnodes=2
        )

        with (
            patch.object(fusion, "is_sm100_supported", return_value=False),
            patch.object(fusion, "is_sm90_supported", return_value=True),
        ):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(single_node_mnnvl),
                "mnnvl",
            )
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(single_node_trtllm),
                "trtllm",
            )
            with self.assertRaises(ValueError):
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node_mnnvl)
            with self.assertRaises(ValueError):
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node_trtllm)

        with patch.object(fusion, "is_sm100_supported", return_value=True):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node_mnnvl),
                "mnnvl",
            )
            with self.assertRaises(ValueError):
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node_trtllm)

        for arch in ("pre_sm90", "post_sm10x"):
            with (
                self.subTest(arch=arch),
                patch.object(fusion, "is_sm100_supported", return_value=False),
                patch.object(fusion, "is_sm90_supported", return_value=False),
            ):
                for args in (
                    single_node_mnnvl,
                    multi_node_mnnvl,
                    single_node_trtllm,
                    multi_node_trtllm,
                ):
                    with self.subTest(backend=args.flashinfer_allreduce_fusion_backend):
                        with self.assertRaises(ValueError):
                            fusion.resolve_flashinfer_allreduce_fusion_backend(args)

    def test_allreduce_fusion_backends_match_torch_baseline(self):
        fake_comm = _FakeFlashInferComm()
        original_comm = fusion._flashinfer_comm
        original_create = fusion._create_allreduce_fusion_workspace
        original_unavailable = fusion._flashinfer_allreduce_unavailable
        from sglang.srt.runtime_context import get_resources

        buffers = get_resources().buffers
        manager_key = "flashinfer_fusion_attn_tp_workspace"
        original_manager = buffers.get(manager_key)
        try:
            fusion._flashinfer_comm = fake_comm
            fusion._create_allreduce_fusion_workspace = (
                fake_comm.create_allreduce_fusion_workspace
            )
            fusion._flashinfer_allreduce_unavailable = False

            for backend in ("trtllm", "mnnvl"):
                with self.subTest(backend=backend):
                    world_size = 4
                    manager = fusion.FlashInferWorkspaceManager()
                    manager.workspace = _FakeWorkspace(backend, world_size)
                    manager.initialized = True
                    buffers[manager_key] = manager
                    if not torch.cuda.is_available():
                        self.skipTest("FlashInfer allreduce custom op is CUDA-only")
                    device = torch.device("cuda")
                    torch.manual_seed(0)
                    input_tensor = torch.randn(4, 8, dtype=torch.float32, device=device)
                    residual = torch.randn(4, 8, dtype=torch.float32, device=device)
                    weight = torch.randn(8, dtype=torch.float32, device=device)
                    eps = 1e-6

                    expected_norm, expected_residual = (
                        _torch_allreduce_residual_rmsnorm_baseline(
                            input_tensor, residual, weight, world_size, eps
                        )
                    )

                    with (
                        patch.object(
                            fusion, "is_flashinfer_available", return_value=True
                        ),
                        get_parallel().override(attn_tp_size=world_size),
                        patch.object(
                            fusion, "ensure_workspace_initialized", return_value=True
                        ),
                    ):
                        norm_out, residual_out = (
                            fusion.flashinfer_allreduce_residual_rmsnorm(
                                input_tensor=input_tensor,
                                residual=residual,
                                weight=weight,
                                eps=eps,
                                max_token_num=8,
                            )
                        )

                    torch.testing.assert_close(norm_out, expected_norm)
                    torch.testing.assert_close(residual_out, expected_residual)
        finally:
            fusion._flashinfer_comm = original_comm
            fusion._create_allreduce_fusion_workspace = original_create
            if original_manager is None:
                buffers.pop(manager_key, None)
            else:
                buffers[manager_key] = original_manager
            fusion._flashinfer_allreduce_unavailable = original_unavailable


if __name__ == "__main__":
    unittest.main()
