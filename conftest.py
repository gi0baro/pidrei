"""Repo-root pytest config.

Only wires the opt-in blocking-fs detector (`scripts/blocking_fs_detector.py`).
Per-package fixtures live in `packages/*/tests/conftest.py`.
"""

import pathlib
import sys


sys.path.insert(0, str(pathlib.Path(__file__).parent / "scripts"))


def pytest_configure(config):
    import os

    if os.environ.get("PIDREI_FS_DETECT") != "1":
        return
    import blocking_fs_detector

    blocking_fs_detector.install()
    config._blocking_fs_detector = blocking_fs_detector


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    detector = getattr(config, "_blocking_fs_detector", None)
    if detector is None:
        return
    terminalreporter.write_sep("=", "blocking filesystem calls")
    terminalreporter.write_line(detector.format_report())
