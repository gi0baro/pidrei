import warnings

import pytest

from pidrei_tui import terminal_image


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
    if terminal_image.get_capability_overrides():
        terminal_image.set_capability_overrides({})
        warnings.warn(
            "test left pidrei_tui's terminal capability overrides set "
            "(set_capability_overrides was not restored); reset to auto-detection",
            stacklevel=1,
        )
