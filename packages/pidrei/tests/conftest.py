import importlib
import os
import warnings

import pytest

from pidrei_ai.utils import clock, timers
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
def _theme_json_validator_guard():
    """Fail-loud check of the process-wide theme JSON validator.

    `set_theme_json_validator` is a set-once startup install (pi's module
    `let`); a test that installs it would make every later custom-theme load
    validate, which no test may rely on by accident. The install is left in
    place so the warning names the polluting test rather than masking it.
    """
    # The package re-exports the `theme` proxy under the submodule's name, so
    # attribute-style imports resolve to the proxy; go through the registry.
    theme_module = importlib.import_module("pidrei.modes.interactive.theme.theme")
    before = theme_module._theme_json_validator
    yield
    if theme_module._theme_json_validator is not before:
        warnings.warn(
            "test changed the theme JSON validator (set_theme_json_validator) and did not restore it",
            stacklevel=1,
        )


@pytest.fixture(autouse=True)
def _capability_overrides_guard():
    """Fail-loud reset of the process-wide terminal capability overrides.

    `set_capability_overrides` (0.84.4) keeps a module-level override dict and
    invalidates the capability cache when it changes; this suite reaches it
    through production paths (`InteractiveMode.__init__`,
    `_apply_runtime_settings`, `create_startup_tui`), so a test driving those
    with terminal settings would silently rewrite `get_capabilities()` for
    every later test in the shared runtime. The warning names the polluting
    test; the reset keeps the poison from spreading.
    """
    yield
    if terminal_image._capability_overrides:
        terminal_image.set_capability_overrides({})
        warnings.warn(
            "test left pidrei_tui's terminal capability overrides set "
            "(set_capability_overrides was not restored); reset to auto-detection",
            stacklevel=1,
        )


# The clock/timer seams `fake_timers()` swaps, captured at collection time
# before any test can touch them.
_TIMER_SEAMS = (
    (timers, "set_timeout", timers.set_timeout),
    (clock, "now_ms", clock.now_ms),
)


@pytest.fixture(autouse=True)
def _timer_seam_guard():
    """Fail-loud reset of the process-wide clock and timer seams.

    The whole suite shares one tonio runtime and one copy of these modules, so
    a test that installs `fake_timers()` (the cache-warmer tests) and never
    exits it would hand every later test a frozen clock or a timer queue nothing
    advances. The warning names the polluting test; the reset keeps the poison
    from spreading.
    """
    yield
    for module, name, original in _TIMER_SEAMS:
        if getattr(module, name) is not original:
            setattr(module, name, original)
            warnings.warn(
                f"test left {module.__name__}.{name} swapped (a fake was not exited); restored",
                stacklevel=1,
            )


def pytest_configure(config):
    # pi's model-registry tests used to really fetch pi.dev catalogs; since
    # 0.83.0 pi also runs its suite with PI_OFFLINE=1 (opt-out per test via
    # allowNetwork()). pidrei tests stay hermetic the same way: the PI_OFFLINE
    # equivalent disables the remote-catalog network path for every runtime
    # built in this suite; tests that exercise network code paths against
    # local mocks pop the variable themselves.
    os.environ["PIDREI_OFFLINE"] = "1"
