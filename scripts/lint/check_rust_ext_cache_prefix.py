#!/usr/bin/env python3
"""Check that the rust-ext cache key stays in sync across its sites.

The build workflow looks up and saves cache entries under its own prefix, the
reusable stage forwards its default, and the download action restores with its
own. These files cannot reference one another, and a mismatch makes every pool
silently fall back to source builds at install time.
"""

import ast
import re
import sys
from pathlib import Path

import yaml

BUILD_WORKFLOW = ".github/workflows/_pr-test-rust-ext-build.yml"
DOWNLOAD_ACTION = ".github/actions/download-rust-ext/action.yml"
STAGE_WORKFLOW = ".github/workflows/_pr-test-stage.yml"
RERUN_WORKFLOW = ".github/workflows/rerun-test.yml"
SEED_WORKFLOW = ".github/workflows/seed-rust-ext-cache.yml"
SLASH_HANDLER = "scripts/ci/utils/slash_command_handler.py"

RUST_BUILD_REUSABLE = "./.github/workflows/_pr-test-rust-ext-build.yml"
STAGE_REUSABLE = "./.github/workflows/_pr-test-stage.yml"
DOWNLOAD_ACTION_REUSABLE = "./.github/actions/download-rust-ext"
SLASH_PREFIX_CONSTANT = "_DEFAULT_RUST_EXT_CACHE_KEY_PREFIX"

_HASH_FILES = re.compile(r"hashFiles\(([^)]*)\)")
_QUOTED = re.compile(r"'([^']*)'")


def hashed_inputs(path: str) -> list[tuple[str, ...]]:
    """The argument tuple of every ``hashFiles(...)`` cache key in a file."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return [tuple(_QUOTED.findall(args)) for args in _HASH_FILES.findall(text)]


def load_yaml(path: str):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_python_string(path: str, name: str) -> str:
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise ValueError(f"{path} has no string assignment for {name}")


def check_rerun_source_fallback(rerun: dict) -> list[str]:
    errors = []
    downloads = []
    for job_name, job in rerun["jobs"].items():
        for step in job.get("steps", []):
            if step.get("uses") == DOWNLOAD_ACTION_REUSABLE:
                downloads.append((job_name, step))
    if not downloads:
        return [f"{RERUN_WORKFLOW} has no Rust extension download steps"]
    for job_name, step in downloads:
        if str(step.get("with", {}).get("required", "false")).lower() == "true":
            errors.append(
                f"{RERUN_WORKFLOW} {job_name} requires an evictable cross-workflow "
                "cache; reruns must retain source-build fallback"
            )
    return errors


def _needs(job: dict) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else needs


def _default_build_prefix() -> str:
    workflow = load_yaml(BUILD_WORKFLOW)
    triggers = workflow.get("on", workflow.get(True))
    return triggers["workflow_call"]["inputs"]["cache_key_prefix"]["default"]


def check_required_prebuilt_wiring() -> list[str]:
    errors = []
    default_prefix = _default_build_prefix()
    required_prefixes = set()
    seed_prefixes = set()
    for workflow_path in sorted(Path(".github/workflows").glob("*.yml")):
        path = str(workflow_path)
        jobs = load_yaml(path).get("jobs", {})
        producers_by_prefix = {}
        for name, job in jobs.items():
            if job.get("uses") != RUST_BUILD_REUSABLE:
                continue
            prefix = job.get("with", {}).get("cache_key_prefix", default_prefix)
            if prefix in producers_by_prefix:
                errors.append(
                    f"{path} has multiple Rust extension producers for {prefix}: "
                    f"{producers_by_prefix[prefix]}, {name}"
                )
                continue
            producers_by_prefix[prefix] = name
            if path == SEED_WORKFLOW:
                seed_prefixes.add(prefix)

        for consumer_name, consumer in jobs.items():
            consumer_inputs = consumer.get("with", {})
            if (
                consumer.get("uses") != STAGE_REUSABLE
                or consumer_inputs.get("require_prebuilt_rust_ext") is not True
            ):
                continue
            prefix = consumer_inputs.get("rust_ext_cache_key_prefix", default_prefix)
            required_prefixes.add(prefix)
            producer_name = producers_by_prefix.get(prefix)
            if producer_name is None:
                errors.append(
                    f"{path} {consumer_name} requires prebuilt Rust extensions "
                    f"with prefix {prefix!r} but has no matching producer"
                )
                continue
            if producer_name not in _needs(consumer):
                errors.append(
                    f"{path} {consumer_name} does not depend on {producer_name}"
                )
            expected_artifact = (
                "${{ needs." + producer_name + ".outputs.artifact_name }}"
            )
            if consumer_inputs.get("rust_ext_artifact") != expected_artifact:
                errors.append(
                    f"{path} {consumer_name}.rust_ext_artifact is "
                    f"{consumer_inputs.get('rust_ext_artifact')!r}, expected "
                    f"{expected_artifact!r}"
                )

    missing_seed_prefixes = sorted(required_prefixes - seed_prefixes)
    if missing_seed_prefixes:
        errors.append(
            f"{SEED_WORKFLOW} has no producer for required prefixes: "
            + ", ".join(missing_seed_prefixes)
        )
    return errors


def main() -> int:
    workflow = load_yaml(BUILD_WORKFLOW)
    action = load_yaml(DOWNLOAD_ACTION)
    stage = load_yaml(STAGE_WORKFLOW)
    rerun = load_yaml(RERUN_WORKFLOW)

    # yaml 1.1 parses the `on:` key as boolean True
    triggers = workflow.get("on", workflow.get(True))
    stage_triggers = stage.get("on", stage.get(True))
    save_prefix = triggers["workflow_call"]["inputs"]["cache_key_prefix"]["default"]
    restore_prefix = action["inputs"]["cache_key_prefix"]["default"]
    stage_prefix = stage_triggers["workflow_call"]["inputs"][
        "rust_ext_cache_key_prefix"
    ]["default"]
    rerun_triggers = rerun.get("on", rerun.get(True))
    rerun_prefix = rerun_triggers["workflow_dispatch"]["inputs"][
        "rust_ext_cache_key_prefix"
    ]["default"]
    slash_prefix = load_python_string(SLASH_HANDLER, SLASH_PREFIX_CONSTANT)

    if (
        len({save_prefix, restore_prefix, stage_prefix, rerun_prefix, slash_prefix})
        != 1
    ):
        print("ERROR: rust-ext cache_key_prefix defaults do not match.")
        print(f"  {BUILD_WORKFLOW} saves under:    {save_prefix}")
        print(f"  {DOWNLOAD_ACTION} restores with: {restore_prefix}")
        print(f"  {STAGE_WORKFLOW} forwards:        {stage_prefix}")
        print(f"  {RERUN_WORKFLOW} defaults to:     {rerun_prefix}")
        print(f"  {SLASH_HANDLER} dispatches:       {slash_prefix}")
        print("Bump all five together, or consumers miss the shared cache.")
        return 1

    # Adding a file to one key alone permanently misses the other's entries.
    sites = [(BUILD_WORKFLOW, inputs) for inputs in hashed_inputs(BUILD_WORKFLOW)]
    sites += [(DOWNLOAD_ACTION, inputs) for inputs in hashed_inputs(DOWNLOAD_ACTION)]

    if not sites:
        print("ERROR: no hashFiles(...) cache key found; this check is dead.")
        return 1

    if len({inputs for _, inputs in sites}) > 1:
        print("ERROR: rust-ext cache key inputs do not match.")
        for path, inputs in sites:
            print(f"  {path}: {list(inputs)}")
        print("Every lookup/save/restore site must hash the same inputs.")
        return 1

    wiring_errors = check_required_prebuilt_wiring()
    if wiring_errors:
        print("ERROR: required prebuilt Rust extension wiring is inconsistent.")
        for error in wiring_errors:
            print(f"  {error}")
        return 1

    fallback_errors = check_rerun_source_fallback(rerun)
    if fallback_errors:
        print("ERROR: rerun Rust extension source fallback is disabled.")
        for error in fallback_errors:
            print(f"  {error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
