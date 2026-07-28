#!/usr/bin/env python3
"""Add the legacy DeepEP low-latency specialization used by GPT-OSS."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_ANCHOR = "        case 2560: case_macro(2560); \\\n"
_GPT_OSS_CASE = (
    "        case 3072: case_macro(3072); /* for gpt-oss */             \\\n"
)
_CASE_3072_RE = re.compile(r"^\s*case\s+3072\s*:", re.MULTILINE)


def patch_deepep_launch_header(deepep_dir: Path) -> bool:
    """Patch DeepEP's legacy launch table, returning whether it changed."""
    launch_header = deepep_dir / "csrc" / "kernels" / "launch.cuh"
    source = launch_header.read_text()

    case_3072_count = len(_CASE_3072_RE.findall(source))
    case_macro_3072_count = source.count("case_macro(3072)")
    if case_3072_count == case_macro_3072_count == 1:
        return False
    if case_3072_count != 0 or case_macro_3072_count != 0:
        raise RuntimeError(
            f"Found an incomplete or duplicated 3072 specialization in "
            f"{launch_header}: case_count={case_3072_count}, "
            f"case_macro_count={case_macro_3072_count}"
        )

    anchor_count = source.count(_ANCHOR)
    if anchor_count != 1:
        raise RuntimeError(
            f"Expected exactly one legacy DeepEP 2560 case in {launch_header}, "
            f"found {anchor_count}; refusing an ambiguous source patch"
        )

    patched_source = source.replace(_ANCHOR, _ANCHOR + _GPT_OSS_CASE)
    if (
        len(_CASE_3072_RE.findall(patched_source)) != 1
        or patched_source.count("case_macro(3072)") != 1
    ):
        raise RuntimeError(f"Failed to add exactly one GPT-OSS case to {launch_header}")
    launch_header.write_text(patched_source)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("deepep_dir", type=Path)
    args = parser.parse_args()

    changed = patch_deepep_launch_header(args.deepep_dir)
    print(
        f"{'Patched' if changed else 'Already patched'} "
        f"{args.deepep_dir / 'csrc/kernels/launch.cuh'}"
    )


if __name__ == "__main__":
    main()
