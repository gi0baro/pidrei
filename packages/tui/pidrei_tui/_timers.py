"""setTimeout/setInterval equivalents for the tui package.

No pi counterpart: pi leans on the JS event loop's global timers, whose
callbacks run on the one thread that owns all UI state. The equivalent
here is the started TUI's owner task (`OwnerTask`): `TUI.start()` registers
it with `set_ui_owner`, and from then on every `Timeout`/`Interval` fires on
that task — ordered with input handling, so a callback never overlaps a key
being processed, `cancel()` is exact (cancel and fire are ordered on the
same task), and the timer tasks are children of the TUI's scope, reaped at
`stop()` instead of ticking on after a skipped `dispose()`.

With no TUI started (tests, headless modes) timers run detached, calling
`fn` on their own task — the pre-owner behaviour. Each detached timer gets
its own single-use `OwnerTask`: a shared module-level one would be state
spanning every TUI lifetime in the process (and, under pytest's
session-scoped runtime, every test), accumulating handles and pinning
zombie tasks across runtime teardowns.

A registered owner is routed to only while it is `serving` — merely
`started` is not enough: an owner whose TUI never reached `stop()` (a
crashed flow, a test that died mid-lifecycle) still reads as started, but
its scope is consumed, and `scope.spawn` on a consumed scope silently
drops the timer coroutine — a timer that never fires and never errors.

`fn` must return an awaitable (async-only callback policy); the result is
awaited rather than dropped.
"""

from ._owner import OwnerTask, TimerHandle


_ui_owner: OwnerTask | None = None


def set_ui_owner(owner: OwnerTask | None) -> None:
    """Route new timers to `owner` (the started TUI's); `None` restores detached timers."""
    global _ui_owner
    _ui_owner = owner


def get_ui_owner() -> OwnerTask | None:
    return _ui_owner


def _owner() -> OwnerTask:
    owner = _ui_owner
    return owner if owner is not None and owner.serving else OwnerTask()


class Timeout:
    """One-shot timer: run `fn` after `delay_ms` unless cancelled first."""

    def __init__(self, delay_ms: float, fn) -> None:
        self._handle: TimerHandle = _owner().after(delay_ms, fn)

    def cancel(self) -> None:
        self._handle.cancel()


class Interval:
    """Repeating timer: run `fn` every `delay_ms` until cancelled."""

    def __init__(self, delay_ms: float, fn) -> None:
        self._handle: TimerHandle = _owner().every(delay_ms, fn)

    def cancel(self) -> None:
        self._handle.cancel()
