#!/usr/bin/env python3
"""Check that the rust-ext cache key stays in sync across its sites.

The build workflow looks up and saves cache entries under its own prefix, the
reusable stage forwards its default, and the download action restores with its
own. These files cannot reference one another, and a mismatch makes every pool
silently fall back to source builds at install time.
"""

import re
import sys

import yaml

BUILD_WORKFLOW = ".github/workflows/_pr-test-rust-ext-build.yml"
DOWNLOAD_ACTION = ".github/actions/download-rust-ext/action.yml"
STAGE_WORKFLOW = ".github/workflows/_pr-test-stage.yml"
PR_WORKFLOW = ".github/workflows/pr-test.yml"
NIGHTLY_WORKFLOW = ".github/workflows/nightly-test-nvidia.yml"
SEED_WORKFLOW = ".github/workflows/seed-rust-ext-cache.yml"

AARCH64_ARTIFACT = "rust-ext-aarch64"
AARCH64_CACHE_KEY_PREFIX = "rust-ext-aarch64-cp310-cp312"
AARCH64_RUNNER = "arm-kernel-build-node"

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


def check_aarch64_wiring() -> list[str]:
    errors = []
    for path, build_job, consumer_job in (
        (PR_WORKFLOW, "rust-ext-build-arm", "base-c-test-4-gpu-gb300"),
        (NIGHTLY_WORKFLOW, "rust-ext-build-arm", "nightly-4-gpu-gb300"),
    ):
        jobs = load_yaml(path)["jobs"]
        build_inputs = jobs[build_job]["with"]
        consumer_inputs = jobs[consumer_job]["with"]
        expected_build_inputs = {
            "runs_on": AARCH64_RUNNER,
            "artifact_name": AARCH64_ARTIFACT,
            "cache_key_prefix": AARCH64_CACHE_KEY_PREFIX,
        }
        for key, expected in expected_build_inputs.items():
            if build_inputs.get(key) != expected:
                errors.append(
                    f"{path} {build_job}.{key} is {build_inputs.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if consumer_inputs.get("rust_ext_cache_key_prefix") != AARCH64_CACHE_KEY_PREFIX:
            errors.append(
                f"{path} {consumer_job}.rust_ext_cache_key_prefix is "
                f"{consumer_inputs.get('rust_ext_cache_key_prefix')!r}, "
                f"expected {AARCH64_CACHE_KEY_PREFIX!r}"
            )

    seed_inputs = load_yaml(SEED_WORKFLOW)["jobs"]["seed-aarch64"]["with"]
    for key, expected in {
        "runs_on": AARCH64_RUNNER,
        "artifact_name": AARCH64_ARTIFACT,
        "cache_key_prefix": AARCH64_CACHE_KEY_PREFIX,
    }.items():
        if seed_inputs.get(key) != expected:
            errors.append(
                f"{SEED_WORKFLOW} seed-aarch64.{key} is "
                f"{seed_inputs.get(key)!r}, expected {expected!r}"
            )
    return errors


def main() -> int:
    workflow = load_yaml(BUILD_WORKFLOW)
    action = load_yaml(DOWNLOAD_ACTION)
    stage = load_yaml(STAGE_WORKFLOW)

    # yaml 1.1 parses the `on:` key as boolean True
    triggers = workflow.get("on", workflow.get(True))
    stage_triggers = stage.get("on", stage.get(True))
    save_prefix = triggers["workflow_call"]["inputs"]["cache_key_prefix"]["default"]
    restore_prefix = action["inputs"]["cache_key_prefix"]["default"]
    stage_prefix = stage_triggers["workflow_call"]["inputs"][
        "rust_ext_cache_key_prefix"
    ]["default"]

    if len({save_prefix, restore_prefix, stage_prefix}) != 1:
        print("ERROR: rust-ext cache_key_prefix defaults do not match.")
        print(f"  {BUILD_WORKFLOW} saves under:    {save_prefix}")
        print(f"  {DOWNLOAD_ACTION} restores with: {restore_prefix}")
        print(f"  {STAGE_WORKFLOW} forwards:        {stage_prefix}")
        print("Bump all three together, or every pool falls back to source builds.")
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

    aarch64_errors = check_aarch64_wiring()
    if aarch64_errors:
        print("ERROR: aarch64 Rust extension cache wiring is inconsistent.")
        for error in aarch64_errors:
            print(f"  {error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
