import warnings

import pytest

from pidrei_tui import terminal_image
from pidrei_tui._owner import OwnerTask
from pidrei_tui._timers import get_ui_owner, set_ui_owner


@pytest.fixture(autouse=True)
def _ambient_ui_owner_guard():
    """End of each test's UI lifecycle: close the ambient owner a started TUI
    left registered, and reset the registry.

    The whole suite shares one tonio runtime (session fixture over a global
    singleton), so a registered owner that is still serving would capture
    every later test's `Timeout`/`Interval`.
    """
    yield
    owner = get_ui_owner()
    if owner is None:
        return
    set_ui_owner(None)
    if isinstance(owner, OwnerTask):  # not the ManualUiTimers fake
        owner.close()


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
