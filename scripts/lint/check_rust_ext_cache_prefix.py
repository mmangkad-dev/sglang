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
SEED_WORKFLOW = ".github/workflows/seed-rust-ext-cache.yml"
RUNNER_CONFIGS = "scripts/ci/runner_configs.yml"

DEFAULT_ARCHITECTURE = "x86_64"
RUST_BUILD_REUSABLE = "./.github/workflows/_pr-test-rust-ext-build.yml"
STAGE_REUSABLE = "./.github/workflows/_pr-test-stage.yml"
PRODUCER_INPUTS = (
    "architecture",
    "runs_on",
    "artifact_name",
    "cache_key_prefix",
    "build_python_310",
    "max_glibc",
)
PLATFORM_PROFILE_INPUTS = tuple(
    key for key in PRODUCER_INPUTS if key != "artifact_name"
)

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


def _producers(jobs: dict) -> list[tuple[str, dict]]:
    return [
        (name, job)
        for name, job in jobs.items()
        if job.get("uses") == RUST_BUILD_REUSABLE
    ]


def _required_consumers(jobs: dict) -> list[tuple[str, dict]]:
    return [
        (name, job)
        for name, job in jobs.items()
        if job.get("uses") == STAGE_REUSABLE
        and job.get("with", {}).get("require_prebuilt_rust_ext") is True
    ]


def _needs(job: dict) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else needs


def _producer_defaults() -> dict:
    workflow = load_yaml(BUILD_WORKFLOW)
    triggers = workflow.get("on", workflow.get(True))
    inputs = triggers["workflow_call"]["inputs"]
    return {key: inputs.get(key, {}).get("default") for key in PRODUCER_INPUTS}


def _producer_profile(job: dict, defaults: dict) -> dict:
    inputs = job.get("with", {})
    return {key: inputs.get(key, defaults[key]) for key in PRODUCER_INPUTS}


def _runner_architecture(runner_config: str, runner_configs: dict) -> str | None:
    config = runner_configs.get(runner_config)
    if config is None:
        return None
    return config.get("architecture", DEFAULT_ARCHITECTURE)


def check_platform_wiring() -> list[str]:
    errors = []
    defaults = _producer_defaults()
    runner_configs = load_yaml(RUNNER_CONFIGS)["runner_configs"]
    canonical_profiles = {}
    required_architectures = set()
    seed_architectures = set()
    for workflow_path in sorted(Path(".github/workflows").glob("*.yml")):
        path = str(workflow_path)
        jobs = load_yaml(path).get("jobs", {})
        producers = _producers(jobs)
        consumers = _required_consumers(jobs)
        if not producers and not consumers:
            continue

        producers_by_architecture = {}
        for producer_name, producer in producers:
            profile = _producer_profile(producer, defaults)
            missing = [key for key, value in profile.items() if value is None]
            if missing:
                errors.append(
                    f"{path} {producer_name} is missing inputs: {', '.join(missing)}"
                )
                continue
            architecture = profile["architecture"]
            if architecture in producers_by_architecture:
                other_name, _ = producers_by_architecture[architecture]
                errors.append(
                    f"{path} has multiple {architecture} Rust extension producers: "
                    f"{other_name}, {producer_name}"
                )
                continue
            producers_by_architecture[architecture] = (producer_name, profile)
            if path == SEED_WORKFLOW:
                seed_architectures.add(architecture)

            canonical = canonical_profiles.get(architecture)
            if canonical is None:
                canonical_profiles[architecture] = (path, producer_name, profile)
                continue
            canonical_path, canonical_name, canonical_profile = canonical
            for key in PLATFORM_PROFILE_INPUTS:
                if profile[key] != canonical_profile[key]:
                    errors.append(
                        f"{path} {producer_name}.{key} is {profile[key]!r}, but "
                        f"{canonical_path} {canonical_name}.{key} is "
                        f"{canonical_profile[key]!r}"
                    )

        for consumer_name, consumer in consumers:
            consumer_inputs = consumer.get("with", {})
            runner_config = consumer_inputs.get("runner_config")
            architecture = _runner_architecture(runner_config, runner_configs)
            if architecture is None:
                errors.append(
                    f"{path} {consumer_name} uses unknown runner_config "
                    f"{runner_config!r}"
                )
                continue
            required_architectures.add(architecture)
            producer_entry = producers_by_architecture.get(architecture)
            if producer_entry is None:
                errors.append(
                    f"{path} {consumer_name} requires prebuilt {architecture} Rust "
                    "extensions but has no matching producer"
                )
                continue
            producer_name, producer_profile = producer_entry
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
            expected_prefix = producer_profile["cache_key_prefix"]
            if consumer_inputs.get("rust_ext_cache_key_prefix") != expected_prefix:
                errors.append(
                    f"{path} {consumer_name}.rust_ext_cache_key_prefix is "
                    f"{consumer_inputs.get('rust_ext_cache_key_prefix')!r}, but "
                    f"{producer_name}.cache_key_prefix is {expected_prefix!r}"
                )

    missing_seed_architectures = sorted(required_architectures - seed_architectures)
    if missing_seed_architectures:
        errors.append(
            f"{SEED_WORKFLOW} has no producer for required architectures: "
            + ", ".join(missing_seed_architectures)
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

    platform_errors = check_platform_wiring()
    if platform_errors:
        print("ERROR: Rust extension platform wiring is inconsistent.")
        for error in platform_errors:
            print(f"  {error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
