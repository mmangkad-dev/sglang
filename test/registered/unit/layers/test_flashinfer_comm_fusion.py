import contextlib
import math
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import flashinfer_comm_fusion as fusion
from sglang.srt.runtime_context import get_context, get_parallel
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
        kAllReduce = object()
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
        pattern,
        output=None,
        residual_out=None,
        norm_out=None,
        residual_in=None,
        rms_gamma=None,
        rms_eps=None,
        **_kwargs,
    ):
        if pattern is self.AllReduceFusionPattern.kAllReduce:
            allreduced = input * workspace.world_size
            if output is None:
                return allreduced
            output.copy_(allreduced)
            return output

        if pattern is not self.AllReduceFusionPattern.kARResidualRMSNorm:
            raise ValueError(f"Unexpected pattern: {pattern}")

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
    def test_alignment_retry_finds_a_usable_granularity_step(self):
        """A 32B-misaligned lamport buffer faults the TWOSHOT kernel at runtime."""

        class _Workspace:
            def __init__(self, backend, buffer_size_bytes):
                self.backend = backend
                self.buffer_size_bytes = buffer_size_bytes
                self.destroyed = False

            def destroy(self):
                self.destroyed = True

        BAD, GOOD = 536870224, 715827200
        for name, backend, sizes, expected_requests in (
            ("already aligned", "mnnvl", [GOOD], [16384]),
            ("quarter step clears it", "mnnvl", [BAD, GOOD], [16384, 20480]),
            # +25% landed on the same buffer, so the next attempt must double.
            ("doubles when stuck", "mnnvl", [BAD, BAD, GOOD], [16384, 20480, 40960]),
            # trtllm sizes its buffer differently; the check must not apply.
            ("trtllm exempt", "trtllm", [BAD], [16384]),
        ):
            created = []

            def create(**kwargs):
                created.append(_Workspace(backend, sizes[len(created)]))
                self.assertEqual(
                    kwargs["max_token_num"], expected_requests[len(created) - 1]
                )
                return created[-1]

            with (
                self.subTest(name),
                patch.object(fusion, "_create_allreduce_fusion_workspace", create),
            ):
                workspace, max_token_num = fusion._create_workspace_aligned_for_twoshot(
                    {"max_token_num": 16384}
                )

            self.assertEqual(len(created), len(expected_requests))
            self.assertEqual(max_token_num, expected_requests[-1])
            self.assertIs(workspace, created[-1])
            # Rejected attempts must release their multicast memory.
            self.assertTrue(all(w.destroyed for w in created[:-1]))
            self.assertFalse(created[-1].destroyed)

        # Exhausting the retries must fail rather than return a faulting workspace.
        with patch.object(
            fusion,
            "_create_allreduce_fusion_workspace",
            lambda **_kw: _Workspace("mnnvl", BAD),
        ):
            with self.assertRaises(RuntimeError):
                fusion._create_workspace_aligned_for_twoshot({"max_token_num": 16384})

    def test_capacity_gate_declines_what_the_workspace_cannot_serve(self):
        """Coverage, group identity and (hidden_dim, dtype) all gate the fused op."""
        manager = fusion.FlashInferWorkspaceManager()
        workspace = _FakeWorkspace("mnnvl", world_size=4)
        workspace.buffer_size_bytes = 32
        workspace.is_buffer_size_sufficient = lambda **kw: kw["num_tokens"] <= 6240
        manager.workspace = workspace
        manager.initialized = True
        manager.world_size = 4
        manager.rank = 0
        manager.hidden_dim = 7168
        manager.dtype = torch.bfloat16
        manager.group = (None, None)
        manager.max_supported_token_num = manager._probe_max_supported_token_num(
            hidden_dim=7168, dtype=torch.bfloat16
        )
        # Probed from the buffer, not from the requested max_token_num.
        self.assertEqual(manager.max_supported_token_num, 6240)

        group = types.SimpleNamespace(device_group=None, cpu_group=None)
        bf16 = torch.bfloat16
        for name, tensor, group_key, expected in (
            ("covered", torch.zeros(6240, 7168, dtype=bf16), (None, None), True),
            ("one row over", torch.zeros(6241, 7168, dtype=bf16), (None, None), False),
            (
                "other hidden_dim",
                torch.zeros(64, 4096, dtype=bf16),
                (None, None),
                False,
            ),
            (
                "other dtype",
                torch.zeros(64, 7168, dtype=torch.float16),
                (None, None),
                False,
            ),
            # Freezing removed initialize()'s ability to self-heal a group change.
            ("other group", torch.zeros(64, 7168, dtype=bf16), ("a", "b"), False),
        ):
            manager.group = group_key
            with (
                self.subTest(name),
                patch.object(fusion, "_get_workspace_manager", return_value=manager),
                patch.object(fusion, "_fusion_group", return_value=(4, 0, group)),
                get_parallel().override(moe_ep_size=1, moe_tp_size=4),
            ):
                self.assertEqual(
                    fusion.can_use_flashinfer_allreduce_fusion(
                        tensor, use_attn_tp_group=False
                    ),
                    expected,
                )

    def test_coverage_probe_reports_a_contiguous_bound(self):
        """MNNVL coverage holes below the one-shot threshold must not be skipped.

        ONESHOT needs tp copies of the payload where TWOSHOT needs 2, so for
        tp > 2 the requirement drops at the switch and a small buffer hides a band
        of unsupported rows that bisection alone would step over.
        """
        hidden_dim, tp, elem = 2880, 4, 2
        one_shot_threshold = 1024 * 1024

        def probe(buffer_size_bytes):
            def required(token_num):
                payload = token_num * hidden_dim * tp * elem
                if payload <= one_shot_threshold:  # ONESHOT: tp copies
                    return payload
                return 2 * math.ceil(token_num / tp) * tp * hidden_dim * elem

            manager = fusion.FlashInferWorkspaceManager()
            manager.workspace = _FakeWorkspace("mnnvl", world_size=tp)
            manager.workspace.is_buffer_size_sufficient = (
                lambda **kw: required(kw["num_tokens"]) <= buffer_size_bytes
            )
            manager.initialized = True
            manager.world_size = tp
            return manager._probe_max_supported_token_num(
                hidden_dim=hidden_dim, dtype=torch.bfloat16
            )

        # 768 KiB: rows 1-34 fit, 35-45 do not, 46-68 fit again. Bound is 34.
        self.assertEqual(probe(768 * 1024), 34)
        # The real floor is ~170x the threshold, so the switch is inside the
        # covered region and the bound is exact.
        self.assertEqual(probe(178956288), 15532)

    def test_frozen_workspace_is_not_replaced_for_larger_shape(self):
        """A post-capture eager shape must not invalidate captured device pointers."""
        manager = fusion.FlashInferWorkspaceManager()
        workspace = _FakeWorkspace("trtllm", world_size=4)
        workspace.is_buffer_size_sufficient = lambda **kw: kw["num_tokens"] <= 2048
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

    def test_gate_has_no_batch_size_cap(self):
        """A reintroduced cap would silently undo this change."""
        from sglang.srt.layers import communicator as communicator_module

        with (
            patch.object(communicator_module, "_is_sm100_supported", True),
            patch.object(communicator_module, "_is_flashinfer_available", True),
            patch.object(
                communicator_module, "is_dp_attention_enabled", return_value=False
            ),
            patch.object(
                communicator_module,
                "is_flashinfer_allreduce_unavailable",
                return_value=False,
            ),
            get_context().override_server_args(
                flashinfer_allreduce_fusion_backend="auto"
            ),
        ):
            gate = communicator_module.apply_flashinfer_allreduce_fusion
            # 2048 was the old cap: both sides of it must now behave alike.
            self.assertTrue(gate(2048))
            self.assertTrue(gate(2049))
            self.assertFalse(gate(0))

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

        for use_attn_tp_group, moe_ep_size, moe_tp_size, expected in (
            (True, 1, 2, ["attention"]),
            (False, 2, 1, ["ep"]),
            (False, 1, 2, ["moe_tp"]),
            (False, 2, 2, ["ep", "moe_tp"]),
        ):
            calls = []

            def all_reduce(tensor, *, name):
                calls.append(name)
                return tensor * 2

            with (
                self.subTest(use_attn_tp_group=use_attn_tp_group, ep=moe_ep_size),
                patch.object(layernorm, "_use_aiter", False),
                patch.object(
                    fusion,
                    "can_use_flashinfer_allreduce_fusion",
                    return_value=False,
                ),
                patch(
                    "sglang.srt.distributed.attention_tensor_model_parallel_all_reduce",
                    side_effect=lambda t: all_reduce(t, name="attention"),
                ),
                patch(
                    "sglang.srt.distributed.moe_expert_parallel_all_reduce",
                    side_effect=lambda t: all_reduce(t, name="ep"),
                ),
                patch(
                    "sglang.srt.distributed.moe_tensor_model_parallel_all_reduce",
                    side_effect=lambda t: all_reduce(t, name="moe_tp"),
                ),
                get_parallel().override(
                    attn_tp_size=2, moe_ep_size=moe_ep_size, moe_tp_size=moe_tp_size
                ),
            ):
                output = layernorm._forward_with_allreduce_fusion(
                    norm_module=norm,
                    x=x,
                    residual=residual,
                    post_residual_addition=None,
                    weight=torch.ones(4),
                    use_attn_tp_group=use_attn_tp_group,
                )

            self.assertEqual(calls, expected)
            expected_x = x * (2 ** len(expected))
            torch.testing.assert_close(norm.inputs[0], expected_x)
            # post_residual_addition was already folded into residual upstream;
            # passing it again would double-apply it.
            self.assertIsNone(norm.inputs[2])
            torch.testing.assert_close(output, expected_x + residual)

    def test_aiter_rejection_falls_back_to_full_tp(self):
        """AITER fuses over the whole TP group, so its fallback must too."""
        from sglang.srt.layers import layernorm

        norm = types.SimpleNamespace(
            variance_epsilon=1e-6,
            forward=lambda x, residual, _post: (x, residual),
        )
        x = torch.ones(2, 4)

        with (
            patch.object(layernorm, "_use_aiter", True),
            patch(
                "sglang.srt.distributed.tensor_model_parallel_fused_allreduce_rmsnorm",
                return_value=None,
            ),
            patch(
                "sglang.srt.distributed.tensor_model_parallel_all_reduce",
                side_effect=lambda t: t * 8,
            ) as tp_all_reduce,
            patch("sglang.srt.distributed.moe_expert_parallel_all_reduce") as ep,
            get_parallel().override(
                tp_size=8, attn_tp_size=1, moe_ep_size=1, moe_tp_size=1
            ),
        ):
            output, _ = layernorm._forward_with_allreduce_fusion(
                norm_module=norm,
                x=x,
                residual=torch.ones_like(x),
                post_residual_addition=None,
                weight=torch.ones(4),
                use_attn_tp_group=False,
            )

        tp_all_reduce.assert_called_once_with(x)
        ep.assert_not_called()
        torch.testing.assert_close(output, x * 8)

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


_GROUP_KEY = ("device_group", "cpu_group")
_OTHER_GROUP_KEY = ("other_device_group", "other_cpu_group")


class TestFlashInferAllReduceOnly(CustomTestCase):
    def _make_manager(self, world_size, group_key=_GROUP_KEY):
        manager = fusion.FlashInferWorkspaceManager()
        manager.workspace = _FakeWorkspace(None, world_size)
        manager.initialized = True
        manager.world_size = world_size
        manager.group = group_key
        manager.max_token_num = 2048
        manager.hidden_dim = 4096
        manager.dtype = torch.float32
        return manager

    @contextlib.contextmanager
    def _patched_attn_workspace(self, manager):
        from sglang.srt.runtime_context import get_resources

        buffers = get_resources().buffers
        manager_key = "flashinfer_fusion_attn_tp_workspace"
        original_manager = buffers.get(manager_key)
        original_comm = fusion._flashinfer_comm
        original_unavailable = fusion._flashinfer_allreduce_unavailable

        buffers[manager_key] = manager
        fusion._flashinfer_comm = _FakeFlashInferComm()
        fusion._flashinfer_allreduce_unavailable = False
        try:
            yield
        finally:
            fusion._flashinfer_comm = original_comm
            fusion._flashinfer_allreduce_unavailable = original_unavailable
            if original_manager is None:
                buffers.pop(manager_key, None)
            else:
                buffers[manager_key] = original_manager

    def _can_use(self, input_, world_size=4, group_key=_GROUP_KEY):
        return fusion.can_use_flashinfer_allreduce(
            input_,
            use_attn_tp_group=True,
            expected_world_size=world_size,
            expected_group_key=group_key,
        )

    def test_allreduce_output_equals_input_times_world_size(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for flashinfer custom op")
        world_size = 4
        with self._patched_attn_workspace(self._make_manager(world_size)):
            input_ = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
            expected = input_ * world_size

            with get_parallel().override(attn_tp_size=world_size):
                self.assertTrue(self._can_use(input_, world_size=world_size))
                result = fusion.flashinfer_allreduce(input_, use_attn_tp_group=True)

            torch.testing.assert_close(result, expected)

    def test_shape_guard_rejects_non_2d(self):
        with self._patched_attn_workspace(self._make_manager(4)):
            self.assertFalse(self._can_use(torch.randn(16)))
            self.assertFalse(self._can_use(torch.randn(2, 8, 16)))

    def test_shape_guard_rejects_non_contiguous(self):
        with self._patched_attn_workspace(self._make_manager(4)):
            non_contiguous = torch.randn(16, 8).t()
            self.assertFalse(non_contiguous.is_contiguous())
            self.assertFalse(self._can_use(non_contiguous))

    def test_rejects_when_unavailable(self):
        original_unavailable = fusion._flashinfer_allreduce_unavailable
        try:
            fusion._flashinfer_allreduce_unavailable = True
            self.assertFalse(self._can_use(torch.randn(8, 16)))
        finally:
            fusion._flashinfer_allreduce_unavailable = original_unavailable

    def test_rejects_when_workspace_uninitialized(self):
        with self._patched_attn_workspace(fusion.FlashInferWorkspaceManager()):
            with get_parallel().override(attn_tp_size=4):
                self.assertFalse(self._can_use(torch.randn(8, 16)))

    def test_rejects_when_workspace_group_differs(self):
        """A workspace rendezvoused on other peers must not be reused.

        Under hybrid EP+TP (e.g. tp=4, ep=2) the MoE-TP and MoE-EP groups have
        the same world size but pair different ranks, so a workspace built for
        one silently reduces across the wrong peers when used by the other --
        wrong output rather than a crash.
        """
        with self._patched_attn_workspace(self._make_manager(2)):
            self.assertFalse(
                self._can_use(
                    torch.randn(8, 16), world_size=2, group_key=_OTHER_GROUP_KEY
                )
            )

    def test_rejects_when_workspace_world_size_differs(self):
        with self._patched_attn_workspace(self._make_manager(4)):
            self.assertFalse(self._can_use(torch.randn(8, 16), world_size=2))

    def test_rejects_when_token_num_exceeds_workspace_capacity(self):
        """Under Dynamo the capacity check replaces is_buffer_size_sufficient().

        _FakeWorkspace.is_buffer_size_sufficient() always says yes, so this only
        passes if the compiling branch consults the manager's own allocation.
        """
        manager = self._make_manager(4)
        manager.max_token_num = 8
        with self._patched_attn_workspace(manager):
            with patch.object(torch.compiler, "is_compiling", return_value=True):
                self.assertTrue(self._can_use(torch.randn(8, 16)))
                self.assertFalse(self._can_use(torch.randn(9, 16)))

    def test_rejects_when_hidden_dim_exceeds_workspace_capacity(self):
        manager = self._make_manager(4)
        manager.hidden_dim = 16
        with self._patched_attn_workspace(manager):
            with patch.object(torch.compiler, "is_compiling", return_value=True):
                self.assertTrue(self._can_use(torch.randn(8, 16)))
                self.assertFalse(self._can_use(torch.randn(8, 17)))

    def test_rejects_when_dtype_mismatches_workspace(self):
        manager = self._make_manager(4)
        manager.dtype = torch.bfloat16
        with self._patched_attn_workspace(manager):
            with patch.object(torch.compiler, "is_compiling", return_value=True):
                self.assertTrue(self._can_use(torch.randn(8, 16, dtype=torch.bfloat16)))
                self.assertFalse(self._can_use(torch.randn(8, 16, dtype=torch.float32)))


class _FakeGroupCoordinator:
    def __init__(self, world_size):
        self.world_size = world_size
        self._fi_workspace_hint = None


class TestTagGroupsForFlashInferAllReduceOnly(CustomTestCase):
    """The MoE workspace rendezvouses on the EP group when moe_ep_size > 1 and
    on the MoE-TP group otherwise, so only that one group may be tagged."""

    def _tag(self, *, attn_tp, moe_ep, moe_tp):
        from sglang.srt.distributed import parallel_state as ps

        with (
            patch.object(ps, "_ENABLE_FLASHINFER_ALLREDUCE_ONLY", True),
            patch.object(ps, "_ATTN_TP", attn_tp),
            patch.object(ps, "_MOE_EP", moe_ep),
            patch.object(ps, "_MOE_TP", moe_tp),
        ):
            ps._tag_groups_for_flashinfer_allreduce_only()

    def test_hybrid_ep_tp_tags_only_the_ep_group(self):
        attn_tp = _FakeGroupCoordinator(4)
        moe_ep = _FakeGroupCoordinator(2)
        moe_tp = _FakeGroupCoordinator(2)

        self._tag(attn_tp=attn_tp, moe_ep=moe_ep, moe_tp=moe_tp)

        self.assertEqual(attn_tp._fi_workspace_hint, "attn_tp")
        self.assertEqual(moe_ep._fi_workspace_hint, "moe")
        self.assertIsNone(moe_tp._fi_workspace_hint)

    def test_pure_moe_tp_tags_only_the_moe_tp_group(self):
        attn_tp = _FakeGroupCoordinator(4)
        moe_ep = _FakeGroupCoordinator(1)
        moe_tp = _FakeGroupCoordinator(4)

        self._tag(attn_tp=attn_tp, moe_ep=moe_ep, moe_tp=moe_tp)

        self.assertEqual(moe_tp._fi_workspace_hint, "moe")
        self.assertIsNone(moe_ep._fi_workspace_hint)

    def test_shared_coordinator_prefers_attn_tp(self):
        # tp=4, ep=4: _ATTN_TP is _MOE_EP is _TP. Either workspace spans the
        # same peers, but the choice must be deterministic.
        shared = _FakeGroupCoordinator(4)
        moe_tp = _FakeGroupCoordinator(1)

        self._tag(attn_tp=shared, moe_ep=shared, moe_tp=moe_tp)

        self.assertEqual(shared._fi_workspace_hint, "attn_tp")
        self.assertIsNone(moe_tp._fi_workspace_hint)


if __name__ == "__main__":
    unittest.main()
