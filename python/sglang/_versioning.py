from __future__ import annotations

import pathlib
import subprocess
import sys

# SGLang release tags are cut from a side branch, so the nearest ancestor tag on
# main can lag far behind the highest published release tag.
_SGLANG_TAG_GLOB = "v*.*.*"


def _default_repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _run_git(args: list[str], cwd: pathlib.Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _is_dirty(cwd: pathlib.Path) -> bool:
    status = _run_git(["status", "--porcelain", "--untracked-files=no"], cwd)
    return bool(status)


def describe_latest_tag(repo_root: pathlib.Path | str | None = None) -> str:
    cwd = pathlib.Path(repo_root or pathlib.Path.cwd()).resolve()
    tags = _run_git(["tag", "--list", "--sort=-version:refname", _SGLANG_TAG_GLOB], cwd)
    tag = next((line.strip() for line in tags.splitlines() if line.strip()), None)
    if tag is None:
        raise RuntimeError(f"no tags found matching {_SGLANG_TAG_GLOB}")

    distance = _run_git(["rev-list", "--count", f"{tag}..HEAD"], cwd)
    node = _run_git(["rev-parse", "--short", "HEAD"], cwd)
    describe = f"{tag}-{distance}-g{node}"
    if _is_dirty(cwd):
        describe += "-dirty"
    return describe


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args == ["describe"]:
        try:
            print(describe_latest_tag(_default_repo_root()))
        except Exception as exc:  # pragma: no cover - build-time CLI fallback
            print(str(exc), file=sys.stderr)
            return 1
        return 0

    print("usage: _versioning.py describe", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
