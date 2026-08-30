import warnings

import pytest

from pidrei_tui import terminal_image
from pidrei_tui._timers import get_ui_owner, set_ui_owner


@pytest.fixture(autouse=True)
def _ambient_ui_owner_guard():
    """Fail-loud reset of the process-wide ambient timer owner.

    The whole suite shares one tonio runtime (session fixture over a global
    singleton), so a test that registers its TUI's owner and never reaches
    `stop()` would silently capture every later test's `Timeout`/`Interval`
    into a queue nothing drains. The warning names the polluting test; the
    reset keeps the poison from spreading.
    """
    yield
    if get_ui_owner() is not None:
        set_ui_owner(None)
        warnings.warn(
            "test left pidrei_tui's ambient UI owner registered (a TuiBase was "
            "started without reaching stop()); reset to detached timers",
            stacklevel=1,
        )


@pytest.fixture(autouse=True)
def _capability_overrides_guard():
    """Fail-loud reset of the process-wide terminal capability overrides.

    `set_capability_overrides` (0.84.4) keeps a module-level override dict and
    invalidates the capability cache when it changes; a test that leaves it
    populated silently rewrites `get_capabilities()` for every later test in
    the shared runtime. The warning names the polluting test; the reset keeps
    the poison from spreading.
    """
    yield
    if terminal_image._capability_overrides:
        terminal_image.set_capability_overrides({})
        warnings.warn(
            "test left pidrei_tui's terminal capability overrides set "
            "(set_capability_overrides was not restored); reset to auto-detection",
            stacklevel=1,
        )
