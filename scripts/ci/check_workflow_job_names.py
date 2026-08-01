#!/usr/bin/env python3
"""Check invariants that prevent required status check bypasses.

Duplicate job names on the same commit allow a passing job in one workflow
to satisfy a required status check meant for a different workflow, bypassing
branch protection.

See: https://github.com/sgl-project/sglang/pull/20208 for an example where
pr-test-npu.yml's "pr-test-finish" job (which passed) caused GitHub to treat
the required "pr-test-finish" check (from pr-test.yml, which failed) as met.
"""

import glob
import sys
from collections import defaultdict

import yaml

# Job names used as required status checks in branch protection.
# These MUST be unique across all workflow files.
PROTECTED_JOB_NAMES = {
    "pr-test-finish",
    "lint",
}


def main() -> int:
    workflows = sorted(glob.glob(".github/workflows/*.yml"))
    job_to_files: dict[str, list[str]] = defaultdict(list)
    protected_label_workflows: list[str] = []

    for wf in workflows:
        with open(wf, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data or "jobs" not in data:
            continue
        for job in data["jobs"]:
            if job in PROTECTED_JOB_NAMES:
                job_to_files[job].append(wf)
        # PyYAML 1.1 parses the unquoted key `on` as boolean True.
        triggers = data.get("on", data.get(True, {}))
        pull_request = (
            triggers.get("pull_request", {}) if isinstance(triggers, dict) else {}
        )
        types = pull_request.get("types", []) if isinstance(pull_request, dict) else []
        if (
            any(job in PROTECTED_JOB_NAMES for job in data["jobs"])
            and "labeled" in types
        ):
            protected_label_workflows.append(wf)

    duplicates = {job: files for job, files in job_to_files.items() if len(files) > 1}

    if not duplicates and not protected_label_workflows:
        return 0

    print("ERROR: Required status check workflow invariant violated.\n")
    for job, files in sorted(duplicates.items()):
        print(f"  Job '{job}' appears in:")
        for f in files:
            print(f"    - {f}")
        print()

    if protected_label_workflows:
        print(
            "Protected-check workflows must not subscribe directly to pull_request.labeled."
        )
        print(
            "A rejected label event can cancel valid CI and publish a replacement "
            "check. Use the non-required label listener instead:\n"
        )
        for wf in protected_label_workflows:
            print(f"    - {wf}")
        print()

    print(
        "Fix duplicate jobs by renaming them; route labels through run-ci-label-listener.yml."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
