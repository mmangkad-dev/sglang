"""Unit tests for precision_baseline_store — no server, no model loading, no HF network."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from sglang.test import precision_baseline_store as hfs
from sglang.test.test_utils import CustomTestCase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> hfs.HfStoreConfig:
    return hfs.HfStoreConfig(repo="test/repo", revision="main")


def _make_rows(n: int, *, model: str = "org/model", base_index: int = 0) -> list[dict]:
    return [
        {
            "model": model,
            "run_path": f"org__model/2025/01/{i:02d}/run-abc123{i}",
            "date": f"2025-01-{i + base_index:02d}",
            "push_index": (i + base_index) * 1000,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHfStoreConfig(CustomTestCase):
    def test_from_env_reads_required_var(self):
        with patch.dict(
            os.environ, {"SGLANG_PRECISION_HF_REPO": "my/repo"}, clear=False
        ):
            cfg = hfs.HfStoreConfig.from_env()
        self.assertEqual(cfg.repo, "my/repo")
        self.assertEqual(cfg.revision, "main")

    def test_from_env_reads_optional_revision(self):
        with patch.dict(
            os.environ,
            {
                "SGLANG_PRECISION_HF_REPO": "my/repo",
                "SGLANG_PRECISION_HF_REVISION": "dev",
            },
            clear=False,
        ):
            cfg = hfs.HfStoreConfig.from_env()
        self.assertEqual(cfg.revision, "dev")

    def test_from_env_reads_read_only_mode(self):
        with patch.dict(
            os.environ,
            {
                "SGLANG_PRECISION_HF_REPO": "my/repo",
                "SGLANG_PRECISION_HF_READ_ONLY": "1",
            },
            clear=False,
        ):
            cfg = hfs.HfStoreConfig.from_env()
        self.assertTrue(cfg.read_only)

    def test_from_env_raises_when_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                hfs.HfStoreConfig.from_env()


class TestSanitizeModelName(CustomTestCase):
    def test_slashes_and_spaces(self):
        self.assertEqual(hfs._sanitize_model_name("org/model name"), "org__model_name")

    def test_no_changes_needed(self):
        self.assertEqual(hfs._sanitize_model_name("simple"), "simple")


class TestRowRecencyKey(CustomTestCase):
    def test_uses_explicit_push_index(self):
        row = {"push_index": 100}
        self.assertEqual(hfs._row_recency_key(row, 5), (100, 5))

    def test_falls_back_to_index(self):
        row = {}
        self.assertEqual(hfs._row_recency_key(row, 5), (-1, 5))

    def test_invalid_push_index(self):
        row = {"push_index": "bad"}
        self.assertEqual(hfs._row_recency_key(row, 5), (-1, 5))

    def test_none_push_index(self):
        row = {"push_index": None}
        self.assertEqual(hfs._row_recency_key(row, 5), (-1, 5))


def _row(run_path, idx, *, label="passed", runner=None, trust=None, **extra):
    row = {"model": "org/m", "run_path": run_path, "push_index": idx}
    if label is not None:
        row["pass_label"] = label
    if runner is not None:
        row["runner_name"] = runner
    if trust is not None:
        row["trust"] = trust
    row.update(extra)
    return row


def _paths(refs):
    return [r.run_path for r in refs]


class TestRowTrust(CustomTestCase):
    def test_explicit_trust_wins(self):
        self.assertEqual(
            hfs.row_trust({"pass_label": "passed", "trust": "unconfirmed"}),
            hfs.TRUST_UNCONFIRMED,
        )

    def test_legacy_passed_is_confirmed(self):
        self.assertEqual(hfs.row_trust({"pass_label": "passed"}), hfs.TRUST_CONFIRMED)

    def test_legacy_established_is_unconfirmed(self):
        # A forced refresh was never compared against anything.
        self.assertEqual(
            hfs.row_trust({"pass_label": "baseline_established"}),
            hfs.TRUST_UNCONFIRMED,
        )

    def test_missing_label_is_unconfirmed(self):
        self.assertEqual(hfs.row_trust({}), hfs.TRUST_UNCONFIRMED)


class TestSelectBaselines(CustomTestCase):
    def test_picks_latest_confirmed(self):
        rows = _make_rows(3)
        for r in rows:
            r["pass_label"] = "passed"
        refs = hfs._select_baselines(rows, model="org/model")
        self.assertEqual(_paths(refs), [rows[-1]["run_path"]])
        self.assertTrue(refs[0].confirmed)

    def test_filters_by_model(self):
        rows = [
            {"model": "a/model", "run_path": "a", "push_index": 1},
            {"model": "b/model", "run_path": "b", "push_index": 2},
        ]
        self.assertEqual(_paths(hfs._select_baselines(rows, model="a/model")), ["a"])

    def test_filters_by_capture_signature(self):
        rows = [
            _row("old", 1, capture_signature="abc123"),
            _row("new", 2, capture_signature="def456"),
        ]
        refs = hfs._select_baselines(rows, model="org/m", capture_signature="def456")
        self.assertEqual(_paths(refs), ["new"])

    def test_returns_empty_on_empty(self):
        self.assertEqual(hfs._select_baselines([], model="org/m"), [])

    def test_skips_rows_without_run_path(self):
        rows = [{"model": "org/m", "push_index": 1}, _row("good", 2)]
        self.assertEqual(_paths(hfs._select_baselines(rows, model="org/m")), ["good"])

    def test_returns_empty_when_signature_mismatch(self):
        rows = [_row("old", 1, capture_signature="abc123")]
        self.assertEqual(
            hfs._select_baselines(rows, model="org/m", capture_signature="zzz"), []
        )

    def test_prefers_older_passed_over_newer_failed(self):
        # A failed run must not shadow an older good baseline, or a persistent
        # regression is masked after one night.
        rows = [_row("good", 1), _row("bad", 2, label="failed")]
        self.assertEqual(_paths(hfs._select_baselines(rows, model="org/m")), ["good"])

    def test_prefers_baseline_established_over_newer_failed(self):
        rows = [
            _row("seed", 1, label="baseline_established"),
            _row("bad", 2, label="failed"),
        ]
        refs = hfs._select_baselines(rows, model="org/m")
        self.assertEqual(_paths(refs), ["seed"])
        self.assertFalse(refs[0].confirmed)

    def test_falls_back_to_failed_when_only_failed(self):
        rows = [_row("bad1", 1, label="failed"), _row("bad2", 2, label="failed")]
        refs = hfs._select_baselines(rows, model="org/m")
        self.assertEqual(_paths(refs), ["bad2"])
        self.assertFalse(refs[0].confirmed)

    def test_missing_pass_label_treated_as_usable(self):
        # Legacy rows without pass_label stay usable as baselines.
        rows = [_row("legacy", 1, label=None), _row("bad", 2, label="failed")]
        self.assertEqual(_paths(hfs._select_baselines(rows, model="org/m")), ["legacy"])

    def test_newer_unconfirmed_candidate_is_compared_before_confirmed(self):
        rows = [
            _row("good", 1, runner="A"),
            _row("forced", 2, label="baseline_established", runner="B"),
        ]
        refs = hfs._select_baselines(rows, model="org/m")
        self.assertEqual(_paths(refs), ["forced", "good"])
        self.assertEqual([r.confirmed for r in refs], [False, True])

    def test_older_unconfirmed_candidate_is_dropped(self):
        rows = [
            _row("forced", 1, label="baseline_established", runner="B"),
            _row("good", 2, runner="A", trust="confirmed"),
        ]
        self.assertEqual(_paths(hfs._select_baselines(rows, model="org/m")), ["good"])

    def test_keeps_newest_candidate_per_runner(self):
        rows = [
            _row("good", 1, runner="A"),
            _row("b1", 2, label="baseline_established", runner="B"),
            _row("c1", 3, label="baseline_established", runner="C"),
            _row("b2", 4, runner="B", trust="unconfirmed"),
        ]
        refs = hfs._select_baselines(rows, model="org/m")
        self.assertEqual(_paths(refs), ["b2", "c1", "good"])

    def test_caps_unconfirmed_candidates(self):
        rows = [_row("good", 0, runner="A")] + [
            _row(f"c{i}", i, label="baseline_established", runner=f"R{i}")
            for i in range(1, 6)
        ]
        refs = hfs._select_baselines(rows, model="org/m", max_unconfirmed=2)
        self.assertEqual(_paths(refs), ["c5", "c4", "good"])

    def test_carries_runner_name(self):
        rows = [_row("good", 1, runner="runner-good-a")]
        self.assertEqual(
            hfs._select_baselines(rows, model="org/m")[0].runner_name,
            "runner-good-a",
        )


def _ref(path, *, runner, confirmed):
    return hfs.BaselineRef(run_path=path, runner_name=runner, confirmed=confirmed)


def _cmp(ref, *, passed, exact=False, summary="s"):
    return hfs.Comparison(baseline=ref, passed=passed, exact=exact, summary=summary)


class TestDecideVerdict(CustomTestCase):
    def setUp(self):
        self.good = _ref("good", runner="A", confirmed=True)
        self.cand = _ref("cand", runner="B", confirmed=False)

    def test_exact_match_on_other_runner_confirms(self):
        v = hfs.decide_verdict(
            [_cmp(self.cand, passed=True, exact=True), _cmp(self.good, passed=False)],
            runner_name="C",
        )
        self.assertTrue(v.passed)
        self.assertEqual(v.trust, hfs.TRUST_CONFIRMED)
        self.assertEqual(v.confirms, self.cand)
        self.assertEqual(v.baseline, self.cand)

    def test_same_runner_cannot_confirm_itself(self):
        v = hfs.decide_verdict(
            [_cmp(self.cand, passed=True, exact=True), _cmp(self.good, passed=False)],
            runner_name="B",
        )
        self.assertTrue(v.passed)
        self.assertEqual(v.trust, hfs.TRUST_UNCONFIRMED)
        self.assertIsNone(v.confirms)

    def test_unknown_runner_cannot_confirm(self):
        v = hfs.decide_verdict(
            [_cmp(self.cand, passed=True, exact=True)], runner_name=None
        )
        self.assertEqual(v.trust, hfs.TRUST_UNCONFIRMED)
        unknown = _ref("cand", runner=None, confirmed=False)
        v = hfs.decide_verdict(
            [_cmp(unknown, passed=True, exact=True)], runner_name="C"
        )
        self.assertEqual(v.trust, hfs.TRUST_UNCONFIRMED)

    def test_inexact_pass_on_other_runner_does_not_confirm(self):
        # Within threshold is not agreement: good runners reproduce exactly.
        v = hfs.decide_verdict(
            [_cmp(self.cand, passed=True, exact=False), _cmp(self.good, passed=False)],
            runner_name="C",
        )
        self.assertTrue(v.passed)
        self.assertEqual(v.trust, hfs.TRUST_UNCONFIRMED)

    def test_confirmed_pass_supersedes_unreproduced_candidate(self):
        v = hfs.decide_verdict(
            [_cmp(self.cand, passed=False), _cmp(self.good, passed=True, exact=True)],
            runner_name="A",
        )
        self.assertTrue(v.passed)
        self.assertEqual(v.trust, hfs.TRUST_CONFIRMED)
        self.assertEqual(v.baseline, self.good)
        self.assertIn("superseded unconfirmed baseline cand (runner B)", v.detail)

    def test_plain_confirmed_pass(self):
        v = hfs.decide_verdict(
            [_cmp(self.good, passed=True, exact=True)], runner_name="A"
        )
        self.assertEqual(
            (v.passed, v.trust, v.baseline), (True, "confirmed", self.good)
        )

    def test_nothing_matches_fails_and_names_runners(self):
        v = hfs.decide_verdict(
            [
                _cmp(self.cand, passed=False, summary="rel_diff=0.5"),
                _cmp(self.good, passed=False, summary="rel_diff=0.1"),
            ],
            runner_name="C",
        )
        self.assertFalse(v.passed)
        self.assertIsNone(v.trust)
        self.assertEqual(v.baseline, self.good)
        self.assertIn("vs unconfirmed baseline cand (runner B): rel_diff=0.5", v.detail)
        self.assertIn("vs confirmed baseline good (runner A): rel_diff=0.1", v.detail)
        self.assertIn("this run on runner C", v.detail)

    def test_failure_flags_runner_that_never_confirmed(self):
        history = hfs.RunnerHistory("bad", 0, 3, ("good-a", "good-b"))
        v = hfs.decide_verdict(
            [_cmp(self.good, passed=False)], runner_name="bad", history=history
        )
        self.assertIn("suspect the machine", v.detail)
        self.assertIn("3 earlier failure(s)", v.detail)

    def test_failure_does_not_blame_runner_with_confirmed_history(self):
        history = hfs.RunnerHistory("good-a", 5, 1, ("good-b",))
        v = hfs.decide_verdict(
            [_cmp(self.good, passed=False)], runner_name="good-a", history=history
        )
        self.assertNotIn("suspect the machine", v.detail)


class TestRunnerHistory(CustomTestCase):
    def test_counts_per_runner(self):
        rows = [
            _row("p1", 1, runner="good-a", trust="confirmed"),
            _row("p2", 2, runner="good-b", trust="confirmed"),
            _row("f1", 3, label="failed", runner="bad"),
            _row("f2", 4, label="failed", runner="bad"),
            _row("e1", 5, label="baseline_established", runner="bad"),
            _row("u1", 6, runner="bad", trust="unconfirmed"),
        ]
        h = hfs._runner_history(
            rows, model="org/m", capture_signature=None, runner_name="bad"
        )
        self.assertEqual((h.num_confirmed, h.num_failed), (0, 2))
        self.assertEqual(h.confirmed_elsewhere, ("good-a", "good-b"))


class TestPoisonedBaseline(CustomTestCase):
    """A forced refresh lands on a runner whose numerics diverge; every other
    runner would then fail against it with the same rel_diff."""

    def setUp(self):
        self.rows = [
            _row("run-good", 1, runner=None),  # legacy good-runner pass
            _row("run-bad-failed", 2, label="failed", runner="runner-bad"),
            _row(
                "run-bad-forced",
                3,
                label="baseline_established",
                runner="runner-bad",
                trust="unconfirmed",
            ),
        ]

    def test_good_runner_still_compares_against_pre_poison_baseline(self):
        refs = hfs._select_baselines(self.rows, model="org/m")
        self.assertEqual(_paths(refs), ["run-bad-forced", "run-good"])

    def test_good_runner_matching_old_baseline_heals_the_store(self):
        refs = hfs._select_baselines(self.rows, model="org/m")
        bad, good = refs
        v = hfs.decide_verdict(
            [_cmp(bad, passed=False), _cmp(good, passed=True, exact=True)],
            runner_name="runner-good-a",
        )
        self.assertTrue(v.passed)
        self.assertEqual(v.trust, hfs.TRUST_CONFIRMED)
        healed = self.rows + [
            _row("run-next", 4, runner="runner-good-a", trust=v.trust)
        ]
        self.assertEqual(
            _paths(hfs._select_baselines(healed, model="org/m")), ["run-next"]
        )

    def test_bad_runner_agreeing_with_itself_never_confirms(self):
        refs = hfs._select_baselines(self.rows, model="org/m")
        bad, good = refs
        v = hfs.decide_verdict(
            [_cmp(bad, passed=True, exact=True), _cmp(good, passed=False)],
            runner_name="runner-bad",
        )
        self.assertTrue(v.passed)
        self.assertEqual(v.trust, hfs.TRUST_UNCONFIRMED)
        # The good baseline stays in every later plan.
        rows = self.rows + [
            _row("run-bad-again", 4, runner="runner-bad", trust=v.trust)
        ]
        self.assertEqual(
            _paths(hfs._select_baselines(rows, model="org/m")),
            ["run-bad-again", "run-good"],
        )


class TestAllocateRunPath(CustomTestCase):
    def test_fresh_path(self):
        self.assertEqual(
            hfs._allocate_run_path([], "m/run-abc", {}), ("m/run-abc", False)
        )

    def test_reuses_for_same_producer(self):
        rows = [{"run_path": "m/run-abc", "runner_name": "A", "pass_label": "passed"}]
        meta = {"runner_name": "A", "pass_label": "passed"}
        self.assertEqual(
            hfs._allocate_run_path(rows, "m/run-abc", meta), ("m/run-abc", True)
        )

    def test_other_runner_gets_its_own_path(self):
        # Otherwise a good runner's row would point at a bad runner's tensors.
        rows = [
            {
                "run_path": "m/run-abc",
                "runner_name": "A",
                "pass_label": "baseline_established",
            },
            {"run_path": "m/run-abc-2", "runner_name": "C", "pass_label": "passed"},
        ]
        meta = {"runner_name": "B", "pass_label": "passed", "trust": "confirmed"}
        self.assertEqual(
            hfs._allocate_run_path(rows, "m/run-abc", meta), ("m/run-abc-3", False)
        )

    def test_same_runner_different_label_gets_its_own_path(self):
        rows = [{"run_path": "m/run-abc", "runner_name": "A", "pass_label": "failed"}]
        meta = {"runner_name": "A", "pass_label": "passed", "trust": "confirmed"}
        self.assertEqual(
            hfs._allocate_run_path(rows, "m/run-abc", meta), ("m/run-abc-2", False)
        )


class TestReadManifest(CustomTestCase):
    @patch("sglang.test.precision_baseline_store.hf_hub_download")
    def test_parses_valid_manifest(self, mock_download):
        content = (
            '{"model":"a","run_path":"p1","push_index":1}\n'
            '{"model":"b","run_path":"p2","push_index":2}\n'
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        try:
            tmp.write(content)
            tmp.close()
            mock_download.return_value = tmp.name
            rows, text = hfs._read_manifest(_make_config())
        finally:
            os.unlink(tmp.name)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["model"], "a")
        self.assertEqual(text, content)

    @patch("sglang.test.precision_baseline_store.hf_hub_download")
    def test_skips_blank_and_corrupt_lines(self, mock_download):
        content = (
            '{"model":"a","run_path":"p1"}\n\nnot-json\n{"model":"b","run_path":"p2"}\n'
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        try:
            tmp.write(content)
            tmp.close()
            mock_download.return_value = tmp.name
            rows, _ = hfs._read_manifest(_make_config())
        finally:
            os.unlink(tmp.name)
        self.assertEqual(len(rows), 2)

    @patch("sglang.test.precision_baseline_store.hf_hub_download")
    def test_returns_empty_on_not_found(self, mock_download):
        from huggingface_hub.errors import EntryNotFoundError

        mock_download.side_effect = EntryNotFoundError("not found")
        rows, text = hfs._read_manifest(_make_config())
        self.assertEqual(rows, [])
        self.assertEqual(text, "")


class TestDownloadBaseline(CustomTestCase):
    @patch("sglang.test.precision_baseline_store.snapshot_download")
    def test_downloads_and_copies_tensors(self, mock_snapshot):
        with tempfile.TemporaryDirectory() as snap_dir:
            tensors = Path(snap_dir) / "org__m/2025/01/01/run-abc/tensors"
            tensors.mkdir(parents=True)
            (tensors / "layer0.pt").write_bytes(b"\x00")
            mock_snapshot.return_value = snap_dir

            with tempfile.TemporaryDirectory() as target_root:
                target = Path(target_root) / "tensors"
                target.mkdir()
                (target / "stale.pt").write_bytes(b"\xff")
                found = hfs.download_baseline(
                    config=_make_config(),
                    run_path="org__m/2025/01/01/run-abc",
                    target_tensors_dir=target,
                )
                self.assertEqual((target / "layer0.pt").read_bytes(), b"\x00")
                self.assertFalse((target / "stale.pt").exists())
        self.assertTrue(found)
        kwargs = mock_snapshot.call_args.kwargs
        self.assertEqual(
            kwargs["allow_patterns"], ["org__m/2025/01/01/run-abc/tensors/*"]
        )

    @patch("sglang.test.precision_baseline_store.snapshot_download")
    def test_missing_tensors_returns_false(self, mock_snapshot):
        with tempfile.TemporaryDirectory() as snap_dir:
            mock_snapshot.return_value = snap_dir
            with tempfile.TemporaryDirectory() as target_root:
                target = Path(target_root) / "tensors"
                target.mkdir()
                (target / "stale.pt").write_bytes(b"\xff")
                found = hfs.download_baseline(
                    config=_make_config(), run_path="gone", target_tensors_dir=target
                )
                self.assertFalse(target.exists())
        self.assertFalse(found)


class TestPlanBaselines(CustomTestCase):
    @patch.object(hfs, "_read_manifest")
    def test_plans_from_manifest(self, mock_manifest):
        rows = [
            _row("good", 1, runner="A", capture_signature="sig"),
            _row(
                "forced",
                2,
                label="baseline_established",
                runner="B",
                capture_signature="sig",
            ),
            _row("other_sig", 3, runner="A", capture_signature="old"),
            _row("f", 4, label="failed", runner="B", capture_signature="sig"),
        ]
        mock_manifest.return_value = (rows, "")
        plan = hfs.plan_baselines(
            config=_make_config(),
            model="org/m",
            runner_name="B",
            capture_signature="sig",
        )
        self.assertEqual(_paths(plan.baselines), ["forced", "good"])
        self.assertEqual(plan.runner_history.num_failed, 1)
        self.assertEqual(plan.runner_history.confirmed_elsewhere, ("A",))

    @patch.object(hfs, "_read_manifest")
    def test_empty_manifest(self, mock_manifest):
        mock_manifest.return_value = ([], "")
        plan = hfs.plan_baselines(
            config=_make_config(), model="org/m", runner_name=None
        )
        self.assertEqual(plan.baselines, ())


class TestPushRun(CustomTestCase):
    """push_run deletes its temp manifest file in a finally block, so tests
    that inspect the manifest content must capture it via a side_effect on
    the mock upload_file *before* push_run cleans up."""

    def test_read_only_store_rejects_push_before_api_access(self):
        config = hfs.HfStoreConfig(repo="test/repo", read_only=True)
        with tempfile.TemporaryDirectory() as td, patch.object(hfs, "HfApi") as api:
            with self.assertRaisesRegex(PermissionError, "read-only"):
                hfs.push_run(
                    config=config,
                    model="org/model",
                    sglang_commit="abc1234",
                    today_tensors_dir=Path(td),
                    meta={},
                )
        api.assert_not_called()

    @staticmethod
    def _make_push_mocks(mock_manifest, mock_api_cls):
        mock_manifest.return_value = ([], "")
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        # Capture manifest text before push_run's finally block deletes it.
        captured = []
        mock_api.upload_file.side_effect = lambda *a, **kw: captured.append(
            Path(kw["path_or_fileobj"]).read_text()
        )
        return mock_api, captured

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch.object(hfs, "_read_manifest")
    def test_uploads_tensors_and_manifest(self, mock_manifest, mock_api_cls):
        mock_api, captured = self._make_push_mocks(mock_manifest, mock_api_cls)

        with tempfile.TemporaryDirectory() as tensor_dir:
            (Path(tensor_dir) / "layer0.pt").write_bytes(b"\x01")
            meta = {"tp_size": 8, "capture_signature": "abc", "hardware": "H200"}

            run_path = hfs.push_run(
                config=_make_config(),
                model="org/m",
                sglang_commit="abc1234567",
                today_tensors_dir=Path(tensor_dir),
                meta=meta,
            )

        mock_api.upload_folder.assert_called_once()
        mock_api.upload_file.assert_called_once()
        row = json.loads(captured[0].strip().splitlines()[-1])
        self.assertEqual(row["model"], "org/m")
        self.assertEqual(row["capture_signature"], "abc")
        self.assertEqual(row["tp_size"], 8)
        self.assertTrue(run_path.startswith("org__m/"))

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch.object(hfs, "_read_manifest")
    def test_skips_existing_tensors_unless_force(self, mock_manifest, mock_api_cls):
        # The run_path must match what push_run generates: model/date/sha7.
        # _today_path() returns today's date, so build the path accordingly.
        today_date, today_date_path = hfs._today_path()
        existing_run_path = f"org__m/{today_date_path}/run-abc1234"
        existing_row = {
            "model": "org/m",
            "run_path": existing_run_path,
            "date": today_date,
            "push_index": 1,
        }
        mock_manifest.return_value = ([existing_row], json.dumps(existing_row) + "\n")
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        # Capture pt file count before push_run cleans up the temp staging dir.
        captured_pt_count = []
        mock_api.upload_folder.side_effect = lambda *a, **kw: captured_pt_count.append(
            len(list(Path(kw["folder_path"]).rglob("*.pt")))
        )

        with tempfile.TemporaryDirectory() as tensor_dir:
            (Path(tensor_dir) / "layer0.pt").write_bytes(b"\x01")
            hfs.push_run(
                config=_make_config(),
                model="org/m",
                sglang_commit="abc1234567",
                today_tensors_dir=Path(tensor_dir),
                meta={"tp_size": 8},
            )

        self.assertEqual(captured_pt_count[0], 0)

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch.object(hfs, "_read_manifest")
    def test_other_runner_same_sha_uploads_to_new_path(
        self, mock_manifest, mock_api_cls
    ):
        today_date, today_date_path = hfs._today_path()
        existing_row = {
            "model": "org/m",
            "run_path": f"org__m/{today_date_path}/run-abc1234",
            "date": today_date,
            "push_index": 1,
            "runner_name": "runner-bad",
            "pass_label": "baseline_established",
            "trust": "unconfirmed",
        }
        mock_manifest.return_value = ([existing_row], json.dumps(existing_row) + "\n")
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        uploads = []
        mock_api.upload_folder.side_effect = lambda *a, **kw: uploads.append(
            (kw["path_in_repo"], len(list(Path(kw["folder_path"]).rglob("*.pt"))))
        )
        manifests = []
        mock_api.upload_file.side_effect = lambda *a, **kw: manifests.append(
            Path(kw["path_or_fileobj"]).read_text()
        )

        with tempfile.TemporaryDirectory() as tensor_dir:
            (Path(tensor_dir) / "layer0.pt").write_bytes(b"\x01")
            run_path = hfs.push_run(
                config=_make_config(),
                model="org/m",
                sglang_commit="abc1234567",
                today_tensors_dir=Path(tensor_dir),
                meta={
                    "runner_name": "runner-good-a",
                    "pass_label": "passed",
                    "trust": "confirmed",
                    "baseline_run_path": "org__m/2025/01/01/run-old",
                },
            )

        self.assertEqual(run_path, f"org__m/{today_date_path}/run-abc1234-2")
        self.assertEqual(uploads, [(run_path, 1)])
        row = json.loads(manifests[0].strip().splitlines()[-1])
        self.assertEqual(row["runner_name"], "runner-good-a")
        self.assertEqual(row["trust"], "confirmed")
        self.assertEqual(row["baseline_run_path"], "org__m/2025/01/01/run-old")

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch.object(hfs, "_read_manifest")
    def test_force_re_uploads(self, mock_manifest, mock_api_cls):
        # Use today's date so the run_path matches what push_run generates.
        today_date, today_date_path = hfs._today_path()
        existing_run_path = f"org__m/{today_date_path}/run-abc1234"
        existing_row = {
            "model": "org/m",
            "run_path": existing_run_path,
            "date": today_date,
            "push_index": 1,
        }
        mock_manifest.return_value = ([existing_row], json.dumps(existing_row) + "\n")
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        # Capture pt file count before push_run cleans up the temp staging dir.
        captured_pt_count = []
        mock_api.upload_folder.side_effect = lambda *a, **kw: captured_pt_count.append(
            len(list(Path(kw["folder_path"]).rglob("*.pt")))
        )

        with tempfile.TemporaryDirectory() as tensor_dir:
            (Path(tensor_dir) / "layer0.pt").write_bytes(b"\x01")
            hfs.push_run(
                config=_make_config(),
                model="org/m",
                sglang_commit="abc1234567",
                today_tensors_dir=Path(tensor_dir),
                meta={"tp_size": 8},
                force=True,
            )

        self.assertGreater(captured_pt_count[0], 0)

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch.object(hfs, "_read_manifest")
    def test_manifest_row_promotes_keys(self, mock_manifest, mock_api_cls):
        _mock_api, captured = self._make_push_mocks(mock_manifest, mock_api_cls)

        with tempfile.TemporaryDirectory() as tensor_dir:
            (Path(tensor_dir) / "layer0.pt").write_bytes(b"\x01")
            meta = {
                "tp_size": 4,
                "hardware": "H100",
                "capture_signature": "sig1",
                "num_layers_compared": 10,
                "num_layers_passed": 10,
                "num_layers_failed": 0,
                "max_rel_diff": 0.001,
                "ci_run_id": "12345",
                "extra_key_not_promoted": True,
            }
            hfs.push_run(
                config=_make_config(),
                model="org/m",
                sglang_commit="abc1234567",
                today_tensors_dir=Path(tensor_dir),
                meta=meta,
            )

        row = json.loads(captured[0].strip().splitlines()[-1])
        for key in hfs._MANIFEST_PROMOTE_KEYS:
            if key in meta:
                self.assertEqual(
                    row.get(key),
                    meta[key],
                    f"manifest missing promoted key: {key}",
                )
        self.assertNotIn("extra_key_not_promoted", row)

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch.object(hfs, "_read_manifest")
    def test_includes_comparator_report(self, mock_manifest, mock_api_cls):
        mock_manifest.return_value = ([], "")
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        # Capture file existence before push_run cleans up the temp staging dir.
        captured_files = []
        mock_api.upload_folder.side_effect = lambda *a, **kw: captured_files.append(
            list(Path(kw["folder_path"]).iterdir())
        )

        with tempfile.TemporaryDirectory() as tensor_dir:
            (Path(tensor_dir) / "layer0.pt").write_bytes(b"\x01")
            report_path = Path(tensor_dir) / "report.jsonl"
            report_path.write_text('{"type":"comparison_tensor"}\n')

            hfs.push_run(
                config=_make_config(),
                model="org/m",
                sglang_commit="abc1234567",
                today_tensors_dir=Path(tensor_dir),
                meta={"tp_size": 8},
                comparator_report=report_path,
            )

        staged_names = [f.name for f in captured_files[0]]
        self.assertIn("comparator_report.jsonl", staged_names)


class TestPruneOldRuns(CustomTestCase):
    @patch.object(hfs, "_read_manifest")
    def test_keeps_recent_runs(self, mock_manifest):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = [
            {"model": "org/m", "run_path": "recent", "date": today},
        ]
        mock_manifest.return_value = (rows, json.dumps(rows[0]) + "\n")
        result = hfs.prune_old_runs(config=_make_config(), keep_days=30)
        self.assertIn("recent", result["kept"])
        self.assertEqual(result["pruned"], [])

    @patch.object(hfs, "_read_manifest")
    def test_archives_one_per_week(self, mock_manifest):
        rows = [
            {"model": "org/m", "run_path": "old1", "date": "2020-01-06"},
            {"model": "org/m", "run_path": "old2", "date": "2020-01-07"},
            {"model": "org/m", "run_path": "old3", "date": "2020-01-08"},
        ]
        mock_manifest.return_value = (rows, "")
        result = hfs.prune_old_runs(
            config=_make_config(), keep_days=0, weekly_archive=True, dry_run=True
        )
        self.assertEqual(len(result["kept"]), 1)
        self.assertEqual(result["kept"][0], "old3")
        self.assertEqual(len(result["pruned"]), 2)

    @patch.object(hfs, "_read_manifest")
    def test_prune_without_archive(self, mock_manifest):
        rows = [
            {"model": "org/m", "run_path": "old1", "date": "2020-01-06"},
            {"model": "org/m", "run_path": "old2", "date": "2020-01-07"},
        ]
        mock_manifest.return_value = (rows, "")
        result = hfs.prune_old_runs(
            config=_make_config(), keep_days=0, weekly_archive=False, dry_run=True
        )
        self.assertEqual(result["kept"], [])
        self.assertEqual(len(result["pruned"]), 2)

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch.object(hfs, "_read_manifest")
    def test_dry_run_does_not_delete(self, mock_manifest, mock_api_cls):
        rows = [
            {"model": "org/m", "run_path": "old1", "date": "2020-01-06"},
        ]
        mock_manifest.return_value = (rows, "")
        mock_api_cls.return_value = MagicMock()
        hfs.prune_old_runs(config=_make_config(), keep_days=0, dry_run=True)
        mock_api_cls.return_value.upload_file.assert_not_called()
        mock_api_cls.return_value.delete_folder.assert_not_called()

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch.object(hfs, "_read_manifest")
    def test_live_mode_deletes(self, mock_manifest, mock_api_cls):
        rows = [
            {"model": "org/m", "run_path": "old1", "date": "2020-01-06"},
            {"model": "org/m", "run_path": "old2", "date": "2020-01-07"},
        ]
        mock_manifest.return_value = (rows, "")
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api

        result = hfs.prune_old_runs(
            config=_make_config(), keep_days=0, weekly_archive=True, dry_run=False
        )
        self.assertEqual(len(result["kept"]), 1)
        self.assertEqual(len(result["pruned"]), 1)
        mock_api.upload_file.assert_called_once()
        mock_api.delete_folder.assert_called_once()

    @patch.object(hfs, "_read_manifest")
    def test_filters_by_model(self, mock_manifest):
        rows = [
            {"model": "org/m1", "run_path": "m1_old", "date": "2020-01-06"},
            {"model": "org/m2", "run_path": "m2_old", "date": "2020-01-06"},
        ]
        mock_manifest.return_value = (rows, "")
        result = hfs.prune_old_runs(
            config=_make_config(),
            model="org/m1",
            keep_days=0,
            weekly_archive=False,
            dry_run=True,
        )
        self.assertIn("m2_old", result["kept"])
        self.assertIn("m1_old", result["pruned"])


class TestWithRetries(CustomTestCase):
    @patch("sglang.test.precision_baseline_store.time")
    def test_succeeds_on_first_attempt(self, mock_time):
        result = hfs._with_retries(lambda: 42, what="test")
        self.assertEqual(result, 42)
        mock_time.sleep.assert_not_called()

    @patch("sglang.test.precision_baseline_store.time")
    def test_retries_on_429(self, mock_time):
        from huggingface_hub.errors import HfHubHTTPError

        resp_429 = MagicMock()
        resp_429.status_code = 429
        exc_429 = HfHubHTTPError("rate limited", response=resp_429)

        mock_op = MagicMock(side_effect=[exc_429, "ok"])
        result = hfs._with_retries(mock_op, what="test", base_delay=0.01)
        self.assertEqual(result, "ok")
        mock_time.sleep.assert_called_once()

    @patch("sglang.test.precision_baseline_store.time")
    def test_raises_on_auth_error(self, mock_time):
        from huggingface_hub.errors import HfHubHTTPError

        resp_401 = MagicMock()
        resp_401.status_code = 401
        exc_401 = HfHubHTTPError("unauthorized", response=resp_401)

        mock_op = MagicMock(side_effect=exc_401)
        with self.assertRaises(HfHubHTTPError):
            hfs._with_retries(mock_op, what="test")
        mock_time.sleep.assert_not_called()

    @patch("sglang.test.precision_baseline_store.time")
    def test_raises_after_max_attempts(self, mock_time):
        from huggingface_hub.errors import HfHubHTTPError

        resp_500 = MagicMock()
        resp_500.status_code = 500
        exc = HfHubHTTPError("server error", response=resp_500)

        mock_op = MagicMock(side_effect=exc)
        with self.assertRaises(HfHubHTTPError):
            hfs._with_retries(mock_op, what="test", attempts=2, base_delay=0.001)
        self.assertEqual(mock_time.sleep.call_count, 1)


if __name__ == "__main__":
    unittest.main()
