import warnings

import pytest

from pidrei_tui import clock, terminal_image, utils


# The clock seams tests may swap, captured at collection time before any test
# can touch them.
_CLOCK_SEAMS = (
    (clock, "monotonic", clock.monotonic),
    (clock, "now_ms", clock.now_ms),
)


@pytest.fixture(autouse=True)
def _tonio_runtime(tonio_runtime):
    """Every test pulls in the runtime: `clock.monotonic` reads its clock, and
    async fixtures then run for plain sync tests too."""


@pytest.fixture(autouse=True)
def _clock_seam_guard():
    """Fail-loud reset of the process-wide clock seams: a test that swaps one
    and never restores it would hand every later test a frozen clock. The
    warning names the polluting test; the reset keeps it from spreading."""
    yield
    for module, name, original in _CLOCK_SEAMS:
        if getattr(module, name) is not original:
            setattr(module, name, original)
            warnings.warn(f"test left {module.__name__}.{name} swapped; restored", stacklevel=1)


@pytest.fixture(autouse=True, scope="session")
def _jieba_loaded():
    """Word segmentation loads jieba in the background on the first Han run;
    tests see it loaded from the start, so a Han expectation never depends on
    whether an earlier test happened to start the load."""
    utils._initialize_jieba()


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
