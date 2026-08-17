#!/usr/bin/env python3
"""Check that the rust-ext cache key stays in sync across its sites.

The build workflow looks up and saves cache entries under its own prefix, the
reusable stage forwards its default, and the download action restores with its
own. These files cannot reference one another, and a mismatch makes every pool
silently fall back to source builds at install time.
"""

import re
import sys
from pathlib import Path

import yaml

BUILD_WORKFLOW = ".github/workflows/_pr-test-rust-ext-build.yml"
DOWNLOAD_ACTION = ".github/actions/download-rust-ext/action.yml"
STAGE_WORKFLOW = ".github/workflows/_pr-test-stage.yml"
PR_WORKFLOW = ".github/workflows/pr-test.yml"
NIGHTLY_WORKFLOW = ".github/workflows/nightly-test-nvidia.yml"
SEED_WORKFLOW = ".github/workflows/seed-rust-ext-cache.yml"
RUNNER_CONFIGS = "scripts/ci/runner_configs.yml"

AARCH64 = "aarch64"
AARCH64_CACHE_PREFIX = "rust-ext-aarch64-"
RUST_BUILD_REUSABLE = "./.github/workflows/_pr-test-rust-ext-build.yml"
STAGE_REUSABLE = "./.github/workflows/_pr-test-stage.yml"
ARM_PRODUCER_INPUTS = (
    "runs_on",
    "artifact_name",
    "cache_key_prefix",
    "build_python_310",
    "max_glibc",
)
ARM_CONSUMER_INPUTS = {
    "rust_ext_cache_key_prefix": "cache_key_prefix",
}

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


def _aarch64_runner_configs() -> set[str]:
    configs = load_yaml(RUNNER_CONFIGS)["runner_configs"]
    return {
        name
        for name, config in configs.items()
        if config.get("architecture") == AARCH64
    }


def _arm_producers(jobs: dict) -> list[tuple[str, dict]]:
    return [
        (name, job)
        for name, job in jobs.items()
        if job.get("uses") == RUST_BUILD_REUSABLE
        and job.get("with", {})
        .get("cache_key_prefix", "")
        .startswith(AARCH64_CACHE_PREFIX)
    ]


def _arm_consumers(
    jobs: dict, aarch64_runner_configs: set[str]
) -> list[tuple[str, dict]]:
    return [
        (name, job)
        for name, job in jobs.items()
        if job.get("uses") == STAGE_REUSABLE
        and job.get("with", {}).get("runner_config") in aarch64_runner_configs
    ]


def _needs(job: dict) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else needs


def check_aarch64_wiring() -> list[str]:
    errors = []
    aarch64_runner_configs = _aarch64_runner_configs()
    if not aarch64_runner_configs:
        return [f"{RUNNER_CONFIGS} defines no architecture: {AARCH64} pools"]

    canonical_path = None
    canonical_inputs = None
    producer_paths = set()
    for workflow_path in sorted(Path(".github/workflows").glob("*.yml")):
        path = str(workflow_path)
        jobs = load_yaml(path).get("jobs", {})
        producers = _arm_producers(jobs)
        consumers = _arm_consumers(jobs, aarch64_runner_configs)
        if not producers and not consumers:
            continue
        if len(producers) != 1:
            errors.append(
                f"{path} has {len(producers)} aarch64 Rust extension producers; "
                "expected exactly one when it produces or consumes aarch64 modules"
            )
            continue

        producer_name, producer = producers[0]
        producer_paths.add(path)
        producer_inputs = producer.get("with", {})
        missing = [key for key in ARM_PRODUCER_INPUTS if key not in producer_inputs]
        if missing:
            errors.append(
                f"{path} {producer_name} is missing inputs: {', '.join(missing)}"
            )
            continue

        if canonical_inputs is None:
            canonical_path = path
            canonical_inputs = producer_inputs
        else:
            for key in ARM_PRODUCER_INPUTS:
                if producer_inputs[key] != canonical_inputs[key]:
                    errors.append(
                        f"{path} {producer_name}.{key} is {producer_inputs[key]!r}, "
                        f"but {canonical_path} uses {canonical_inputs[key]!r}"
                    )

        for consumer_name, consumer in consumers:
            consumer_inputs = consumer.get("with", {})
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
            if consumer_inputs.get("require_prebuilt_rust_ext") is not True:
                errors.append(
                    f"{path} {consumer_name}.require_prebuilt_rust_ext must be true"
                )
            for consumer_key, producer_key in ARM_CONSUMER_INPUTS.items():
                if consumer_inputs.get(consumer_key) != producer_inputs[producer_key]:
                    errors.append(
                        f"{path} {consumer_name}.{consumer_key} is "
                        f"{consumer_inputs.get(consumer_key)!r}, but "
                        f"{producer_name}.{producer_key} is "
                        f"{producer_inputs[producer_key]!r}"
                    )

    required_producer_paths = {PR_WORKFLOW, NIGHTLY_WORKFLOW, SEED_WORKFLOW}
    missing_paths = sorted(required_producer_paths - producer_paths)
    if missing_paths:
        errors.append(
            "missing aarch64 Rust extension producer in: " + ", ".join(missing_paths)
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
