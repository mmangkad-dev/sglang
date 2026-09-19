"""FlashInfer autotune must reach the same tactics on every TP rank.

Without a cross-rank reduction each rank's ``argmin`` follows local timing noise
(measured: 20/20 tuned MoE shapes diverged across 4 ranks on gpt-oss-120b). The
reduction holds only if ranks also enter tuning with the same tuned entries, so
these cover that gate, the digest it decides on, and where the managed store the
entries live in is placed.
"""

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=57, suite="base-a-test-cpu")
register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")

import multiprocessing
import os
import tempfile
import traceback
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

from sglang.srt.model_executor.runner import flashinfer_autotune as autotune
from sglang.srt.model_executor.runner.flashinfer_autotune import (
    _autotune_measurement_policy,
    _autotune_store_digest,
    _autotune_store_root,
    _autotune_tactic_sync_group,
    _converge_autotune_store,
    _managed_autotune_store,
)
from sglang.test.test_utils import CustomTestCase, find_available_port

ENV = {"flashinfer_version": "0.7.0rc3", "gpu": "NVIDIA GB300"}


def _publish(store, key: str, tactic) -> None:
    """Put one tuned winner in *store*, the way FlashInfer would."""
    store.publish(key, "TestRunner", tactic)


def _gate_worker(rank, world_size, master_port, root, entries, writer):
    """Run the entry gate on one rank; report whether its entries survived."""
    try:
        os.environ.update(
            RANK=str(rank),
            WORLD_SIZE=str(world_size),
            MASTER_ADDR="localhost",
            MASTER_PORT=str(master_port),
        )
        store = _managed_autotune_store(Path(root), ENV)
        for key, tactic in entries.items():
            _publish(store, key, tactic)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        _converge_autotune_store(store, dist.group.WORLD)
        writer.send(("ok", sorted(p.name for p in store.entries_dir.glob("*.json"))))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        writer.send(("error", f"{e}"))
    finally:
        writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


class TestAutotuneTacticSyncGroup(CustomTestCase):
    def test_single_rank_has_nobody_to_agree_with(self):
        # A 1-rank group would add a collective per tactic for no agreement.
        tp_group = SimpleNamespace(world_size=1, cpu_group=object())
        self.assertIsNone(_autotune_tactic_sync_group(tp_group))


class TestAutotuneStoreDigest(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _store(self, name: str, env=ENV):
        return _managed_autotune_store(self.dir / name, env)

    def test_stores_with_nothing_tuned_digest_alike(self):
        # An absent store and an empty one must not read as different caches.
        absent, empty = self._store("absent"), self._store("empty")
        empty.publish("op", "TestRunner", 7)
        for path in empty.entries_dir.glob("*.json"):
            path.unlink()
        self.assertEqual(_autotune_store_digest(absent), _autotune_store_digest(empty))

    def test_same_tactics_digest_alike(self):
        rank0, rank1 = self._store("rank0"), self._store("rank1")
        for store in (rank0, rank1):
            _publish(store, "op_a", 7)
            _publish(store, "op_b", 3)
        self.assertEqual(_autotune_store_digest(rank0), _autotune_store_digest(rank1))

    def test_same_operations_with_different_winners_diverge(self):
        # Matching keys are not enough: the tactics themselves must agree.
        rank0, rank1 = self._store("rank0"), self._store("rank1")
        _publish(rank0, "op", 7)
        _publish(rank1, "op", 8)
        self.assertNotEqual(
            _autotune_store_digest(rank0), _autotune_store_digest(rank1)
        )

    def test_missing_operation_diverges(self):
        rank0, rank1 = self._store("rank0"), self._store("rank1")
        _publish(rank0, "op_a", 7)
        _publish(rank1, "op_a", 7)
        _publish(rank1, "op_b", 3)
        self.assertNotEqual(
            _autotune_store_digest(rank0), _autotune_store_digest(rank1)
        )

    def test_environment_is_part_of_what_a_rank_would_serve(self):
        # A drifted environment hashes to another directory, so that rank
        # serves nothing even though the root is the same.
        tuned = self._store("shared")
        _publish(tuned, "op", 7)
        drifted = self._store("shared", {**ENV, "gpu": "NVIDIA B200"})
        self.assertNotEqual(
            _autotune_store_digest(tuned), _autotune_store_digest(drifted)
        )


class TestConvergeAutotuneStore(CustomTestCase):
    """The gate itself, over a real gloo group and real stores."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _run_gate(self, per_rank_entries) -> list:
        world_size = len(per_rank_entries)
        port = find_available_port(23456)
        ctx = multiprocessing.get_context("spawn")
        procs, readers = [], []
        for rank, entries in enumerate(per_rank_entries):
            reader, writer = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_gate_worker,
                args=(
                    rank,
                    world_size,
                    port,
                    str(self.dir / f"rank{rank}"),
                    entries,
                    writer,
                ),
            )
            proc.start()
            writer.close()
            procs.append(proc)
            readers.append(reader)
        results = [r.recv() for r in readers]
        for proc in procs:
            proc.join(timeout=120)
        for status, value in results:
            self.assertEqual(status, "ok", msg=value)
        return [value for _, value in results]

    def test_matching_stores_are_kept(self):
        entries = {"op_a": 7, "op_b": 3}
        kept = self._run_gate([entries, entries])
        self.assertEqual(len(kept[0]), 2)
        self.assertEqual(kept[0], kept[1])

    def test_diverged_stores_are_emptied_on_every_rank(self):
        # A rank that kept its entries would skip profiles its peer still runs.
        self.assertEqual(self._run_gate([{"op": 7}, {"op": 8}]), [[], []])

    def test_a_store_with_extra_entries_empties_both(self):
        self.assertEqual(
            self._run_gate([{"op_a": 7}, {"op_a": 7, "op_b": 3}]), [[], []]
        )


class TestAutotuneStoreRoot(CustomTestCase):
    """Placement of the managed store; FlashInfer owns everything below it."""

    def setUp(self):
        _autotune_store_root.cache_clear()
        self.addCleanup(_autotune_store_root.cache_clear)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_reuse_root_is_stable_across_contexts(self):
        with autotune.envs.SGLANG_CACHE_DIR.override(self.tmp.name):
            root = _autotune_store_root(True)
            self.assertEqual(root, _autotune_store_root(True))
            self.assertEqual(root, Path(self.tmp.name) / "flashinfer" / "autotune")

    def test_fresh_run_root_is_scoped_but_stable_within_the_process(self):
        # Every context in a process must publish to one store, or the
        # finalize reload cannot see an earlier context's winners.
        with autotune.envs.SGLANG_CACHE_DIR.override(self.tmp.name):
            root = _autotune_store_root(False)
            self.assertEqual(root, _autotune_store_root(False))
            self.assertNotEqual(root, _autotune_store_root(True))
            self.assertEqual(root.parent.name, "runs")

    def test_flashinfer_override_wins(self):
        # An explicit FlashInfer root asks for FlashInfer's own placement.
        with (
            autotune.envs.SGLANG_CACHE_DIR.override(self.tmp.name),
            patch.dict(os.environ, {"FLASHINFER_AUTOTUNE_CACHE_DIR": self.tmp.name}),
        ):
            self.assertIsNone(_autotune_store_root(True))


class TestAutotuneMeasurementPolicy(CustomTestCase):
    """Measurement mode is store identity, not just a profiling detail."""

    def test_default_policy_adds_nothing_to_the_store_identity(self):
        self.assertEqual(_autotune_measurement_policy().manifest_fields(), {})

    def test_execution_mode_reaches_the_manifest(self):
        with autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_MEASURE.override("cuda_graph"):
            policy = _autotune_measurement_policy()
        self.assertTrue(policy.use_cuda_graph)
        self.assertEqual(
            policy.manifest_fields(), {"measure_execution_mode": "cuda_graph"}
        )

    def test_cold_l2_reaches_the_manifest(self):
        with autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_COLD_L2.override(True):
            self.assertEqual(
                _autotune_measurement_policy().manifest_fields(),
                {"measure_cold_l2": "True"},
            )

    def test_an_unusable_mode_fails_at_warmup(self):
        with autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_MEASURE.override("graph"):
            with self.assertRaises(ValueError):
                _autotune_measurement_policy()


class TestModelPrefillAutotune(CustomTestCase):
    """Model kernel warmup must cover prefill without a speculative dummy batch."""

    def setUp(self):
        self.hook = Mock(return_value=1)
        self.mr = SimpleNamespace(
            model=SimpleNamespace(autotune_prefill_kernels=self.hook),
            is_generation=True,
            is_draft_worker=False,
            dtype=torch.bfloat16,
        )
        self.runner = SimpleNamespace(model_runner=self.mr)
        # No dummy-buffer or attention APIs: this path must not build a
        # TARGET_VERIFY batch or mutate request/KV state.
        for target, kwargs in (
            ("max_prefill_buffer_tokens", {"return_value": 65536}),
            (
                "flashinfer_autotune_context",
                {"side_effect": lambda *a, **k: nullcontext()},
            ),
        ):
            p = patch.object(autotune, target, **kwargs)
            setattr(self, target, p.start())
            self.addCleanup(p.stop)
        p = patch.object(
            autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND, "get", return_value=False
        )
        p.start()
        self.addCleanup(p.stop)

    def test_declining_model_never_enters_the_autotune_context(self):
        self.mr.model.wants_prefill_autotune = lambda: False
        autotune.maybe_flashinfer_autotune_extend(self.runner, decode_num_tokens=384)
        self.hook.assert_not_called()
        self.flashinfer_autotune_context.assert_not_called()

    def test_extend_pass_is_opt_in(self):
        # A draft worker keeps its own warmup; a model without the hook opts out.
        for draft, has_hook in ((True, True), (False, False)):
            with self.subTest(draft=draft, has_hook=has_hook):
                self.mr.is_draft_worker = draft
                if not has_hook:
                    del self.mr.model.autotune_prefill_kernels
                autotune.maybe_flashinfer_autotune_extend(
                    self.runner, decode_num_tokens=384
                )
                self.hook.assert_not_called()
                self.flashinfer_autotune_context.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "FlashInfer requires CUDA")
class TestAutotuneStoreLifecycle(CustomTestCase):
    """Tuned entries outlive each warmup phase and the context they were
    tuned in; serving runs outside any context and must still find them."""

    def setUp(self):
        from flashinfer.autotuner import AutoTuner

        self.tuner = AutoTuner.get()
        self.tuner.clear_cache()
        self.addCleanup(self.tuner.clear_cache)
        _autotune_store_root.cache_clear()
        self.addCleanup(_autotune_store_root.cache_clear)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runner = SimpleNamespace(
            device="cuda",
            forward_stream=torch.cuda.Stream(),
            tp_group=SimpleNamespace(world_size=1),
        )
        p = patch.object(
            autotune, "get_flashinfer_autotune_skip_ops", return_value=set()
        )
        p.start()
        self.addCleanup(p.stop)

    def _store(self, reuse_cache: bool):
        from flashinfer.autotuner import _collect_metadata

        return _managed_autotune_store(
            _autotune_store_root(reuse_cache), _collect_metadata()
        )

    def test_target_tactics_survive_the_draft_pass_and_stay_attached(self):
        with (
            autotune.envs.SGLANG_CACHE_DIR.override(self.tmp.name),
            autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(True),
        ):
            store = self._store(True)
            _publish(store, "target_prefill", 7)
            with autotune.flashinfer_autotune_context(self.runner, run_lm_head=False):
                pass
            # The draft worker tunes its own operations; a per-entry store has
            # no reason to touch the target's.
            with autotune.flashinfer_autotune_context(self.runner, run_lm_head=False):
                _publish(store, "draft_decode", 3)
            self.assertEqual(store.lookup("target_prefill"), ("TestRunner", 7))
            self.assertEqual(store.lookup("draft_decode"), ("TestRunner", 3))
            # Serving happens with no context open, so the store has to remain
            # the process ambient after warmup exits.
            self.assertEqual(self.tuner._managed_cache.env_dir, store.env_dir)

    def test_cache_reuse_off_tunes_into_a_run_scoped_store(self):
        with autotune.envs.SGLANG_CACHE_DIR.override(self.tmp.name):
            reused = self._store(True)
            _publish(reused, "target_prefill", 7)
            with (
                autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(False),
                autotune.flashinfer_autotune_context(self.runner, run_lm_head=False),
            ):
                pass
            fresh = self._store(False)
            self.assertNotEqual(fresh.env_dir, reused.env_dir)
            self.assertIsNone(fresh.lookup("target_prefill"))
            self.assertEqual(self.tuner._managed_cache.env_dir, fresh.env_dir)


if __name__ == "__main__":
    unittest.main()
