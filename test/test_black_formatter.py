# pylint: disable=all

import unittest

from pathlib import Path


try:
    import black
    import black.report
    from black.files import parse_pyproject_toml

    black_imported = True
except ImportError:
    black_imported = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_black_mode():
    pyproject_toml = PROJECT_ROOT / "pyproject.toml"
    if pyproject_toml.exists():
        config = parse_pyproject_toml(str(pyproject_toml))
    else:
        config = {}
    return black.Mode(**config)


@unittest.skipIf(black_imported is False, "Black is not installed")
class TestBlackFormatter(unittest.TestCase):
    def test_code_formatting(self):
        mode = _get_black_mode()
        excluded_dirs = ["venv", "env", ".venv", ".env", "__pycache__", ".git"]
        py_files = [
            file
            for file in PROJECT_ROOT.rglob("*.py")
            if not any(excluded_dir in file.parts for excluded_dir in excluded_dirs)
        ]

        for file_path in py_files:
            original_source = file_path.read_text(encoding="utf-8")

            with self.assertRaises(black.report.NothingChanged):
                black.format_file_contents(original_source, fast=False, mode=mode)
