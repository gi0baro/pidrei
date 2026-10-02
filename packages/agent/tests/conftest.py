"""Shared fixtures for the agent package tests."""

import warnings

import pytest

from pidrei_ai.utils import clock, timers


# Module attributes tests are allowed to swap, captured at collection time
# before any test can touch them: the clock/timer seams behind `fake_timers()`.
_PROCESS_SEAMS = (
    (timers, "set_timeout", timers.set_timeout),
    (clock, "now_ms", clock.now_ms),
    (clock, "monotonic", clock.monotonic),
    (clock, "sleep_ms", clock.sleep_ms),
)


@pytest.fixture(autouse=True)
def _tonio_runtime(tonio_runtime):
    """Every test pulls in the runtime: `clock.monotonic` reads its clock, and
    async fixtures then run for plain sync tests too."""


@pytest.fixture(autouse=True)
def _timer_seam_guard():
    """Fail-loud reset of the process-wide seams tests may swap.

    The whole suite shares one tonio runtime and one copy of these modules, so
    a test that installs `fake_timers()` (or any stub of these) and never
    restores it would hand every later test a frozen clock or a timer queue
    nothing advances. The warning names the polluting test; the reset keeps
    the poison from spreading.
    """
    yield
    for module, name, original in _PROCESS_SEAMS:
        if getattr(module, name) is not original:
            setattr(module, name, original)
            warnings.warn(
                f"test left {module.__name__}.{name} swapped (a fake was not exited); restored",
                stacklevel=1,
            )
