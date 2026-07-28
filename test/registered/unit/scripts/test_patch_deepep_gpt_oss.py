"""Tests for the legacy DeepEP GPT-OSS source patch."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_PATCH_SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "ci"
    / "cuda"
    / "patch_deepep_gpt_oss.py"
)
_SPEC = importlib.util.spec_from_file_location("patch_deepep_gpt_oss", _PATCH_SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_PATCH_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PATCH_MODULE)
patch_deepep_launch_header = _PATCH_MODULE.patch_deepep_launch_header

_LEGACY_SWITCH = """\
#define SWITCH_HIDDEN(case_macro) \\
    switch (hidden) { \\
        case 2048: case_macro(2048); \\
        case 2560: case_macro(2560); \\
        case 4096: case_macro(4096); \\
        default: EP_HOST_ASSERT(false and "Unsupported hidden"); \\
    } while (false)
"""

_FORMATTED_GPT_OSS_SWITCH = """\
#define SWITCH_HIDDEN(case_macro)                           \\
    switch (hidden) {                                       \\
        case 2560:                                          \\
            case_macro(2560);                               \\
        case 3072:                                          \\
            case_macro(3072); /* for gpt-oss */             \\
        case 4096:                                          \\
            case_macro(4096);                               \\
    }                                                       \\
    while (false)
"""


class TestPatchDeepEPGptOss(CustomTestCase):
    def _write_launch_header(self, root: Path, source: str) -> Path:
        launch_header = root / "csrc" / "kernels" / "launch.cuh"
        launch_header.parent.mkdir(parents=True)
        launch_header.write_text(source)
        return launch_header

    def test_patch_is_exact_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            launch_header = self._write_launch_header(root, _LEGACY_SWITCH)

            self.assertTrue(patch_deepep_launch_header(root))
            patched_once = launch_header.read_text()
            self.assertFalse(patch_deepep_launch_header(root))

            self.assertEqual(launch_header.read_text(), patched_once)
            self.assertEqual(patched_once.count("case 3072:"), 1)
            self.assertLess(
                patched_once.index("case 2560:"),
                patched_once.index("case 3072:"),
            )
            self.assertLess(
                patched_once.index("case 3072:"),
                patched_once.index("case 4096:"),
            )

    def test_patch_rejects_an_ambiguous_anchor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_launch_header(root, _LEGACY_SWITCH + _LEGACY_SWITCH)

            with self.assertRaisesRegex(RuntimeError, "found 2"):
                patch_deepep_launch_header(root)

    def test_formatted_pr_458_source_is_already_patched(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            launch_header = self._write_launch_header(root, _FORMATTED_GPT_OSS_SWITCH)

            self.assertFalse(patch_deepep_launch_header(root))
            self.assertEqual(launch_header.read_text(), _FORMATTED_GPT_OSS_SWITCH)


if __name__ == "__main__":
    unittest.main()
