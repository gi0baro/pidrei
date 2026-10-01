import importlib
import os
import warnings

import pytest

from pidrei.core import output_guard
from pidrei_ai.utils import clock, timers
from pidrei_tui import terminal_image, utils as tui_utils


@pytest.fixture(autouse=True)
def _tonio_runtime(tonio_runtime):
    """Every test pulls in the runtime, so async fixtures run for plain sync
    tests too (tonio's plugin only handles them where `tonio_runtime` is part
    of the test's fixture closure)."""


@pytest.fixture(autouse=True, scope="session")
def _jieba_loaded():
    """Word segmentation loads jieba in the background on the first Han run;
    tests see it loaded from the start, so a Han expectation never depends on
    whether an earlier test happened to start the load."""
    tui_utils._initialize_jieba()


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
def _terminal_colors_guard():
    """Fail-loud reset of the process-wide terminal colors in the theme module.

    `set_terminal_colors`/`set_terminal_color_scheme` record what the terminal
    reported, and every theme resolves "" tokens (and the system theme its
    whole palette and light/dark appearance) from them; a test that feeds
    reports and leaves them set changes every later test's colors. The warning
    names the polluting test; the reset keeps the poison from spreading. The
    pending flag is reset silently: every `InteractiveThemeController`
    construction sets it (production behavior), so only its reset matters.
    """
    theme_module = importlib.import_module("pidrei.modes.interactive.theme.theme")
    yield
    # A timed-out query records a report of Nones: nothing reported, nothing left.
    left_colors = (
        any(value is not None for value in theme_module._terminal_colors.values())
        or theme_module._terminal_color_scheme is not None
    )
    theme_module.set_terminal_colors({})
    theme_module.set_terminal_color_scheme(None)
    if left_colors:
        warnings.warn(
            "test left the theme module's terminal colors set "
            "(set_terminal_colors/set_terminal_color_scheme were not reset); reset",
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
    if terminal_image.get_capability_overrides():
        terminal_image.set_capability_overrides({})
        warnings.warn(
            "test left pidrei_tui's terminal capability overrides set "
            "(set_capability_overrides was not restored); reset to auto-detection",
            stacklevel=1,
        )


@pytest.fixture(autouse=True)
def _output_guard_guard():
    """Fail-loud check of the process-wide stdio writer and stdout takeover.

    With the writer stopped, every stdio write in the suite goes straight to
    the (captured) fd; a test that leaves it running would queue every later
    test's writes behind a writer bound to whatever fds that test installed,
    and a leftover takeover reroutes every later `write_stdout` to stderr. The
    takeover is undone here; the writer cannot be (stopping it awaits), so the
    warning names the polluting test.
    """
    yield
    if output_guard._sender is not None:
        warnings.warn(
            "test left the output guard's writer running (start_output_writer without stop_output_writer)",
            stacklevel=1,
        )
    if output_guard.is_stdout_taken_over():
        output_guard.restore_stdout()
        warnings.warn("test left stdout taken over (take_over_stdout without restore_stdout); restored", stacklevel=1)


# The clock/timer seams `fake_timers()` swaps, captured at collection time
# before any test can touch them.
_TIMER_SEAMS = (
    (timers, "set_timeout", timers.set_timeout),
    (clock, "now_ms", clock.now_ms),
    (clock, "monotonic", clock.monotonic),
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
