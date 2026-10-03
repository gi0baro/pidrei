"""Shared fixtures for the mcp package tests. The clock and timer seams are
guarded by the root conftest."""

import os
import signal
import sys
import warnings
from pathlib import Path

import pytest

from pidrei_http import http
from pidrei_mcp.transports import stdio


sys.path.insert(0, str(Path(__file__).resolve().parent))
# `fake_timers` lives with the agent tests, as the pidrei suite imports it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent" / "tests"))


_SEAM_FUNCTIONS = {"client_for": http.client_for, "shared_client": http.shared_client}


@pytest.fixture(autouse=True)
def _tonio_runtime(tonio_runtime):
    """Every test pulls in the runtime, so async fixtures run for plain sync
    tests too."""


@pytest.fixture(autouse=True)
def _http_seam_guard():
    """Fail-loud restore of the HTTP seam's client lookups (see the http
    package's conftest): the default fetch goes through them."""
    yield
    for name, original in _SEAM_FUNCTIONS.items():
        if getattr(http, name) is not original:
            setattr(http, name, original)
            warnings.warn(f"test left pidrei_http.http.{name} replaced; restored", stacklevel=1)


@pytest.fixture(autouse=True)
def _live_process_groups_guard():
    """Fail-loud cleanup of stdio servers a test left running.

    A stdio transport tracks its server's process group until the server is
    reaped (the exit hook kills what is left at interpreter exit). One still
    tracked after a test is a server the test never closed: it would run on
    through every later test. The guard kills the group and warns, naming
    the polluting test.
    """
    yield
    with stdio._live_guard:
        leftover = list(stdio._live_process_groups)
        stdio._live_process_groups.clear()
    for pid in leftover:
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass
    if leftover:
        warnings.warn(f"test left stdio MCP servers running (process groups {leftover}); killed", stacklevel=1)
