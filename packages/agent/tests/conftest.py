"""Shared fixtures for the agent package tests."""

import warnings

import pytest

from pidrei_agent.harness.env import local
from pidrei_agent.harness.utils import adaptive_publisher
from pidrei_ai.utils import clock


# Module attributes tests are allowed to swap, captured at collection time
# before any test can touch them: the clock/timer seams behind `fake_timers()`
# and the shell-exec knobs the slow-spill regression test narrows.
_PROCESS_SEAMS = (
    (adaptive_publisher, "_set_timeout", adaptive_publisher._set_timeout),
    (clock, "now_ms", clock.now_ms),
    (clock, "sleep_ms", clock.sleep_ms),
    (local, "SPILL_CHANNEL_SIZE", local.SPILL_CHANNEL_SIZE),
    (local, "EXIT_STDIO_GRACE_SECONDS", local.EXIT_STDIO_GRACE_SECONDS),
)


@pytest.fixture(autouse=True)
def _timer_seam_guard():
    """Fail-loud reset of the process-wide seams tests may swap.

    The whole suite shares one tonio runtime and one copy of these modules, so
    a test that installs `fake_timers()` (or any stub of these) and never
    restores it would hand every later test a frozen clock, a timer queue
    nothing advances, or a one-slot spill channel. The warning names the
    polluting test; the reset keeps the poison from spreading.
    """
    yield
    for module, name, original in _PROCESS_SEAMS:
        if getattr(module, name) is not original:
            setattr(module, name, original)
            warnings.warn(
                f"test left {module.__name__}.{name} swapped (a fake was not exited); restored",
                stacklevel=1,
            )
