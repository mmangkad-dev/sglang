"""HF dataset store for nightly precision-regression baselines.

Layout: ``<model>/<YYYY>/<MM>/<DD>/run-<sglang_sha7>[-N]/{meta.json,
comparator_report.jsonl, tensors/*.pt}``. Root ``manifest.jsonl`` has one
row per run; rows carry a ``push_index`` so fetch picks the latest push
regardless of file order (prune may rewrite the file).

Runner lineage: every row records the runner that produced it and how far it
is trusted. A row that was never compared against anything (first run or
forced refresh) is *unconfirmed*; it becomes the parent of a *confirmed* row
only once a different runner reproduces it bit-exactly. Runs compare against
the latest confirmed baseline plus any newer unconfirmed candidates, so a
runner whose numerics diverge cannot turn its own output into the reference
the rest of the pool is judged against.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TypeVar

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)


def _store_token() -> Optional[str]:
    """Write token for the baseline dataset repo.

    Deliberately not HF_TOKEN: that name already carries the runner's
    gated-model read token, so writing the store token there would shadow it
    and turn every gated model on the job into a 401.
    """
    return os.environ.get("SGLANG_PRECISION_HF_TOKEN") or None


@dataclass
class HfStoreConfig:
    repo: str
    revision: str = "main"
    read_only: bool = False

    @classmethod
    def from_env(cls) -> HfStoreConfig:
        repo = os.environ.get("SGLANG_PRECISION_HF_REPO")
        if not repo:
            raise RuntimeError(
                "SGLANG_PRECISION_HF_REPO is not set. The precision baseline "
                "store is required (there is no local-only mode); set the repo "
                "and SGLANG_PRECISION_HF_TOKEN."
            )
        revision = os.environ.get("SGLANG_PRECISION_HF_REVISION", "main")
        read_only = os.environ.get("SGLANG_PRECISION_HF_READ_ONLY", "0") == "1"
        return cls(repo=repo, revision=revision, read_only=read_only)


def _sanitize_model_name(model: str) -> str:
    return model.replace("/", "__").replace(" ", "_")


def _today_path() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d"), now.strftime("%Y/%m/%d")


def _push_index() -> int:
    return time.time_ns()


_T = TypeVar("_T")


def _with_retries(
    op: Callable[[], _T],
    *,
    what: str,
    attempts: int = 3,
    base_delay: float = 2.0,
) -> _T:
    """Exponential backoff on 429/5xx; auth/404 raise immediately."""
    last_exc: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return op()
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            transient = status is None or status == 429 or 500 <= status < 600
            if not transient or attempt == attempts:
                raise
            last_exc = e
            time.sleep(base_delay * (2 ** (attempt - 1)))
    raise RuntimeError(f"unreachable retry exit for {what}: {last_exc}")


def _row_recency_key(row: dict[str, Any], fallback_index: int) -> tuple[int, int]:
    # Fall back to file position for legacy rows that predate push_index.
    explicit = row.get("push_index")
    try:
        explicit_i = int(explicit) if explicit is not None else -1
    except (TypeError, ValueError):
        explicit_i = -1
    return (explicit_i, fallback_index)


TRUST_CONFIRMED = "confirmed"
TRUST_UNCONFIRMED = "unconfirmed"
# Newest unconfirmed candidate per runner, capped: same-runner candidates are
# the same numerics, and each extra reference costs one download + comparator.
MAX_UNCONFIRMED_BASELINES = 3


def row_trust(row: dict[str, Any]) -> str:
    trust = row.get("trust")
    if trust in (TRUST_CONFIRMED, TRUST_UNCONFIRMED):
        return trust
    # Rows predating runner lineage: a pass was checked against the previous
    # baseline, while an established (first-run or forced) row never was.
    return TRUST_CONFIRMED if row.get("pass_label") == "passed" else TRUST_UNCONFIRMED


@dataclass(frozen=True)
class BaselineRef:
    run_path: str
    runner_name: Optional[str]
    confirmed: bool

    def describe(self) -> str:
        trust = "confirmed" if self.confirmed else "unconfirmed"
        return (
            f"{trust} baseline {self.run_path} (runner {self.runner_name or 'unknown'})"
        )


@dataclass(frozen=True)
class RunnerHistory:
    runner_name: Optional[str]
    num_confirmed: int
    num_failed: int
    confirmed_elsewhere: tuple[str, ...]


@dataclass(frozen=True)
class BaselinePlan:
    # Unconfirmed candidates newest first, then the latest confirmed baseline.
    baselines: tuple[BaselineRef, ...]
    runner_history: RunnerHistory


def _matching_rows(
    rows: list[dict[str, Any]],
    *,
    model: str,
    capture_signature: Optional[str],
) -> list[dict[str, Any]]:
    # A baseline is only comparable to a target with the same capture shape, so
    # when a signature is given, mismatched (incl. legacy unsigned) rows are
    # skipped — selection then returns nothing and the caller establishes a
    # fresh one instead of erroring on incompatible tensors.
    keyed: list[tuple[tuple[int, int], dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        if row.get("model") != model:
            continue
        if (
            capture_signature is not None
            and row.get("capture_signature") != capture_signature
        ):
            continue
        if "run_path" not in row:
            continue
        keyed.append((_row_recency_key(row, idx), row))
    keyed.sort(key=lambda kv: kv[0])
    return [row for _, row in keyed]


def _to_ref(row: dict[str, Any]) -> BaselineRef:
    return BaselineRef(
        run_path=row["run_path"],
        runner_name=row.get("runner_name") or None,
        confirmed=row_trust(row) == TRUST_CONFIRMED,
    )


def _select_baselines(
    rows: list[dict[str, Any]],
    *,
    model: str,
    capture_signature: Optional[str] = None,
    max_unconfirmed: int = MAX_UNCONFIRMED_BASELINES,
) -> list[BaselineRef]:
    matching = _matching_rows(rows, model=model, capture_signature=capture_signature)
    # A failed run must not become the next comparison baseline, or a persistent
    # regression is masked: today's regressed tensors (uploaded as "failed")
    # would be selected as the reference next run. Fall back to a failed one
    # only when no usable baseline exists.
    usable = [r for r in matching if r.get("pass_label") != "failed"]
    if not usable:
        return [_to_ref(matching[-1])] if matching else []

    confirmed_idx = next(
        (
            i
            for i in range(len(usable) - 1, -1, -1)
            if row_trust(usable[i]) == TRUST_CONFIRMED
        ),
        None,
    )
    newer = usable if confirmed_idx is None else usable[confirmed_idx + 1 :]

    refs: list[BaselineRef] = []
    seen_runners: set[Optional[str]] = set()
    for row in reversed(newer):
        runner = row.get("runner_name") or None
        if runner in seen_runners:
            continue
        seen_runners.add(runner)
        refs.append(_to_ref(row))
        if len(refs) >= max_unconfirmed:
            break
    if confirmed_idx is not None:
        refs.append(_to_ref(usable[confirmed_idx]))
    return refs


def _runner_history(
    rows: list[dict[str, Any]],
    *,
    model: str,
    capture_signature: Optional[str],
    runner_name: Optional[str],
) -> RunnerHistory:
    num_confirmed = 0
    num_failed = 0
    elsewhere: set[str] = set()
    for row in _matching_rows(rows, model=model, capture_signature=capture_signature):
        runner = row.get("runner_name") or None
        is_confirmed = (
            row.get("pass_label") == "passed" and row_trust(row) == TRUST_CONFIRMED
        )
        if runner_name is not None and runner == runner_name:
            num_confirmed += is_confirmed
            num_failed += row.get("pass_label") == "failed"
        elif runner is not None and is_confirmed:
            elsewhere.add(runner)
    return RunnerHistory(
        runner_name=runner_name,
        num_confirmed=num_confirmed,
        num_failed=num_failed,
        confirmed_elsewhere=tuple(sorted(elsewhere)),
    )


def plan_baselines(
    *,
    config: HfStoreConfig,
    model: str,
    runner_name: Optional[str],
    capture_signature: Optional[str] = None,
) -> BaselinePlan:
    rows, _ = _read_manifest(config)
    return BaselinePlan(
        baselines=tuple(
            _select_baselines(rows, model=model, capture_signature=capture_signature)
        ),
        runner_history=_runner_history(
            rows,
            model=model,
            capture_signature=capture_signature,
            runner_name=runner_name,
        ),
    )


def download_baseline(
    *,
    config: HfStoreConfig,
    run_path: str,
    target_tensors_dir: Path,
) -> bool:
    # Tensors land flat in target_tensors_dir (no enclosing tensors/) so the
    # caller can treat it like a fresh dump dir.
    snapshot_root = _with_retries(
        lambda: snapshot_download(
            repo_id=config.repo,
            repo_type="dataset",
            revision=config.revision,
            allow_patterns=[f"{run_path}/tensors/*"],
            token=_store_token(),
        ),
        what="snapshot download",
    )
    src = Path(snapshot_root) / run_path / "tensors"
    if not src.exists():
        shutil.rmtree(target_tensors_dir, ignore_errors=True)
        return False

    target_tensors_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target_tensors_dir.parent) as staging_root:
        staged_tensors = Path(staging_root) / "tensors"
        shutil.copytree(src, staged_tensors)
        shutil.rmtree(target_tensors_dir, ignore_errors=True)
        staged_tensors.rename(target_tensors_dir)
    return True


@dataclass(frozen=True)
class Comparison:
    baseline: BaselineRef
    passed: bool  # every compared tensor within the diff threshold
    exact: bool  # every compared tensor bit-identical (rel_diff == 0)
    summary: str


@dataclass(frozen=True)
class Verdict:
    passed: bool
    trust: Optional[str]  # None for a failed run, which is never a baseline
    baseline: Optional[BaselineRef]  # the reference this run descends from
    confirms: Optional[BaselineRef]
    detail: str


def _on_other_runner(runner_name: Optional[str], ref: BaselineRef) -> bool:
    return (
        bool(runner_name) and bool(ref.runner_name) and runner_name != ref.runner_name
    )


def decide_verdict(
    comparisons: Sequence[Comparison],
    *,
    runner_name: Optional[str],
    history: Optional[RunnerHistory] = None,
) -> Verdict:
    """Turn one run's comparisons against a BaselinePlan into a verdict.

    Bit-exact agreement across two runners is the only thing that confirms a
    baseline; good runners reproduce each other exactly, so any looser match
    could let a divergent runner's numerics in. A run that matches nothing
    fails and never becomes a candidate: only an explicit refresh or a first
    run may propose new numerics.
    """
    unconfirmed = [c for c in comparisons if not c.baseline.confirmed]
    confirmed = next((c for c in comparisons if c.baseline.confirmed), None)

    for c in unconfirmed:
        if c.exact and _on_other_runner(runner_name, c.baseline):
            return Verdict(
                passed=True,
                trust=TRUST_CONFIRMED,
                baseline=c.baseline,
                confirms=c.baseline,
                detail=(
                    f"confirmed baseline {c.baseline.run_path} from runner "
                    f"{c.baseline.runner_name}: reproduced bit-exactly on runner "
                    f"{runner_name}"
                ),
            )

    if confirmed is not None and confirmed.passed:
        detail = f"comparison ok vs {confirmed.baseline.describe()}"
        rejected = [c for c in unconfirmed if not c.passed]
        if rejected:
            detail += "; superseded " + ", ".join(
                c.baseline.describe() for c in rejected
            )
        return Verdict(
            passed=True,
            trust=TRUST_CONFIRMED,
            baseline=confirmed.baseline,
            confirms=None,
            detail=detail,
        )

    matched = next((c for c in unconfirmed if c.passed), None)
    if matched is not None:
        return Verdict(
            passed=True,
            trust=TRUST_UNCONFIRMED,
            baseline=matched.baseline,
            confirms=None,
            detail=(
                f"comparison ok vs {matched.baseline.describe()}; it stays "
                f"unconfirmed until a runner other than "
                f"{matched.baseline.runner_name or 'the producer'} reproduces it "
                f"bit-exactly"
            ),
        )

    parts = [f"vs {c.baseline.describe()}: {c.summary}" for c in comparisons]
    if runner_name:
        parts.append(f"this run on runner {runner_name}")
    if (
        history is not None
        and history.runner_name
        and history.num_confirmed == 0
        and history.num_failed > 0
        and history.confirmed_elsewhere
    ):
        parts.append(
            f"runner {history.runner_name} has never reproduced a confirmed "
            f"baseline ({history.num_failed} earlier failure(s)) but "
            f"{', '.join(history.confirmed_elsewhere)} did: suspect the machine, "
            f"not the commit"
        )
    primary = confirmed or (comparisons[0] if comparisons else None)
    return Verdict(
        passed=False,
        trust=None,
        baseline=primary.baseline if primary else None,
        confirms=None,
        detail="; ".join(parts) if parts else "no baseline compared",
    )


def _read_manifest(config: HfStoreConfig) -> tuple[list[dict[str, Any]], str]:
    # Skip corrupt rows rather than bricking the store on a partial write.
    try:
        manifest_local = _with_retries(
            lambda: hf_hub_download(
                repo_id=config.repo,
                repo_type="dataset",
                filename="manifest.jsonl",
                revision=config.revision,
                token=_store_token(),
            ),
            what="manifest fetch",
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return [], ""

    text = Path(manifest_local).read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows, text


_MANIFEST_PROMOTE_KEYS = (
    "hardware",
    "tp_size",
    "pass_label",
    "capture_signature",
    "num_layers_compared",
    "num_layers_passed",
    "num_layers_failed",
    "max_rel_diff",
    "ci_run_id",
    "runner_name",
    "driver_version",
    "trust",
    "baseline_run_path",
    "baseline_runner_name",
    "confirms_run_path",
)


def _producer_identity(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (row.get("runner_name"), row.get("pass_label"), row.get("trust"))


def _allocate_run_path(
    existing_rows: list[dict[str, Any]], base_path: str, meta: dict[str, Any]
) -> tuple[str, bool]:
    # Tensors at a run_path are shared by every row naming it, so reuse one only
    # for the same producer (runner, label, trust). Otherwise a run on another
    # runner at the same sha on the same day would be recorded against the
    # first runner's tensors.
    identity = _producer_identity(meta)
    run_path = base_path
    suffix = 1
    while True:
        same_path = [r for r in existing_rows if r.get("run_path") == run_path]
        if not same_path:
            return run_path, False
        if any(_producer_identity(r) == identity for r in same_path):
            return run_path, True
        suffix += 1
        run_path = f"{base_path}-{suffix}"


def push_run(
    *,
    config: HfStoreConfig,
    model: str,
    sglang_commit: str,
    today_tensors_dir: Path,
    meta: dict[str, Any],
    comparator_report: Optional[Path] = None,
    force: bool = False,
) -> str:
    if config.read_only:
        raise PermissionError("precision baseline store is read-only")

    # Dedup: same model+date+sha+producer → skip tensor upload but still refresh
    # meta + comparator_report + append a new manifest row, so pass-1 baseline
    # and pass-2 stats both land. force=True re-uploads tensors too.
    api = HfApi(token=_store_token())
    date_str, date_path = _today_path()
    model_sanitized = _sanitize_model_name(model)
    sha7 = (
        sglang_commit[:7] if sglang_commit and sglang_commit != "unknown" else "no_sha"
    )
    existing_rows, existing_text = _read_manifest(config)
    run_path, tensors_already_present = _allocate_run_path(
        existing_rows, f"{model_sanitized}/{date_path}/run-{sha7}", meta
    )
    skip_tensors = tensors_already_present and not force

    with tempfile.TemporaryDirectory() as stage_dir:
        stage = Path(stage_dir)
        run_dir = stage / "run"
        run_dir.mkdir(parents=True)
        if not skip_tensors:
            tensors_out = run_dir / "tensors"
            tensors_out.mkdir()
            for fp in today_tensors_dir.iterdir():
                if fp.is_file() and fp.suffix == ".pt":
                    shutil.copy2(fp, tensors_out / fp.name)
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if comparator_report is not None and comparator_report.exists():
            shutil.copy2(comparator_report, run_dir / "comparator_report.jsonl")

        commit_msg = (
            f"refresh meta {sha7} for {model} on {date_str}"
            if skip_tensors
            else f"add run {sha7} for {model} on {date_str}"
        )
        _with_retries(
            lambda: api.upload_folder(
                repo_id=config.repo,
                repo_type="dataset",
                revision=config.revision,
                folder_path=str(run_dir),
                path_in_repo=run_path,
                commit_message=commit_msg,
            ),
            what="upload_folder",
        )

    manifest_row = {
        "date": date_str,
        "model": model,
        "run_path": run_path,
        "sglang_commit": sglang_commit,
        "push_index": _push_index(),
        **{k: meta.get(k) for k in _MANIFEST_PROMOTE_KEYS if k in meta},
    }

    new_manifest_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmp_out:
            tmp_out.write(existing_text)
            if existing_text and not existing_text.endswith("\n"):
                tmp_out.write("\n")
            tmp_out.write(json.dumps(manifest_row) + "\n")
            new_manifest_path = tmp_out.name

        _with_retries(
            lambda: api.upload_file(
                path_or_fileobj=new_manifest_path,
                path_in_repo="manifest.jsonl",
                repo_id=config.repo,
                repo_type="dataset",
                revision=config.revision,
                commit_message=f"manifest += {sha7} {date_str}",
            ),
            what="manifest upload",
        )
    finally:
        if new_manifest_path and os.path.exists(new_manifest_path):
            os.unlink(new_manifest_path)
    return run_path


def prune_old_runs(
    *,
    config: HfStoreConfig,
    model: Optional[str] = None,
    keep_days: int = 30,
    weekly_archive: bool = True,
    dry_run: bool = True,
) -> dict[str, list[str]]:
    # dry_run defaults True because model=None+keep_days=0 would wipe the
    # store. Live mode rewrites the manifest before deleting folders so a
    # mid-run failure leaves manifest pointing at the kept rows only.
    api = HfApi(token=_store_token())
    rows, _ = _read_manifest(config)
    if not rows:
        return {"kept": [], "pruned": []}

    cutoff_date = datetime.now(timezone.utc).date()

    def _row_date(row: dict[str, Any]) -> Optional[datetime]:
        try:
            return datetime.strptime(row["date"], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except (KeyError, ValueError):
            return None

    kept_rows: list[dict[str, Any]] = []
    pruned_rows: list[dict[str, Any]] = []

    by_model_week: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        if model is not None and row.get("model") != model:
            kept_rows.append(row)
            continue
        dt = _row_date(row)
        if dt is None:
            kept_rows.append(row)
            continue
        age_days = (cutoff_date - dt.date()).days
        if age_days <= keep_days:
            kept_rows.append(row)
            continue
        iso_year, iso_week, _ = dt.isocalendar()
        by_model_week.setdefault((row.get("model", ""), iso_year, iso_week), []).append(
            row
        )

    for week_rows in by_model_week.values():
        week_rows.sort(key=lambda r: r.get("date", ""))
        if weekly_archive and week_rows:
            kept_rows.append(week_rows[-1])
            pruned_rows.extend(week_rows[:-1])
        else:
            pruned_rows.extend(week_rows)

    kept_rows.sort(key=lambda r: (r.get("date", ""), r.get("model", "")))

    report = {
        "kept": [r["run_path"] for r in kept_rows if "run_path" in r],
        "pruned": [r["run_path"] for r in pruned_rows if "run_path" in r],
    }
    if dry_run or not pruned_rows:
        return report

    rewritten: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmp_out:
            for r in kept_rows:
                tmp_out.write(json.dumps(r) + "\n")
            rewritten = tmp_out.name
        _with_retries(
            lambda: api.upload_file(
                path_or_fileobj=rewritten,
                path_in_repo="manifest.jsonl",
                repo_id=config.repo,
                repo_type="dataset",
                revision=config.revision,
                commit_message=f"manifest -= {len(pruned_rows)} pruned",
            ),
            what="manifest rewrite (prune)",
        )
    finally:
        if rewritten and os.path.exists(rewritten):
            os.unlink(rewritten)

    for r in pruned_rows:
        rp = r.get("run_path")
        if not rp:
            continue
        try:
            api.delete_folder(
                repo_id=config.repo,
                repo_type="dataset",
                path_in_repo=rp,
                revision=config.revision,
                commit_message=f"prune {rp}",
            )
        except Exception:
            # Folder may already be missing; manifest no longer points at it.
            pass

    return report
