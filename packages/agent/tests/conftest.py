"""Shared fixtures for the agent package tests."""

import warnings

import pytest

from pidrei_agent.harness.utils import adaptive_publisher
from pidrei_ai.utils import clock


# Captured at collection time, before any test can swap them.
_TIMER_SEAMS = (
    (adaptive_publisher, "_set_timeout", adaptive_publisher._set_timeout),
    (clock, "now_ms", clock.now_ms),
    (clock, "sleep_ms", clock.sleep_ms),
)


@pytest.fixture(autouse=True)
def _timer_seam_guard():
    """Fail-loud reset of the process-wide clock and timer seams.

    The whole suite shares one tonio runtime and one copy of these modules, so
    a test that installs `fake_timers()` (or any clock stub) and never restores
    it would hand every later test a frozen clock and a timer queue nothing
    advances. The warning names the polluting test; the reset keeps the poison
    from spreading.
    """
    yield
    for module, name, original in _TIMER_SEAMS:
        if getattr(module, name) is not original:
            setattr(module, name, original)
            warnings.warn(
                f"test left {module.__name__}.{name} swapped (a fake clock/timer was not exited); restored",
                stacklevel=1,
            )
