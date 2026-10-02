"""setTimeout/setInterval equivalents, the one timer API of every pidrei
package.

No pi counterpart: pi uses the JS event loop's global timers. Here a timer is
a task parked on its cancel event with the delay as timeout
(`Event.wait(delay)`, a runtime timer): once the delay passes, `fn` runs
unless `cancel()` set the event first. The timer knows nothing else, as
`setTimeout` does not (spec/ui-island.md, "Timers").

`fn` is synchronous, like pi's timer callbacks, and runs on the timer's own
task. Whatever it touches is the caller's to guard: a callback that mutates
UI state takes the UI state lock, as every other UI mutation does.

`cancel()` is atomic with the fire's check: a fire that sees the event set
does not run `fn`, one that sees it clear came first. pi's single thread also
guarantees nothing runs after `clearTimeout` returns; here a fire that
checked just before the cancel still runs `fn` afterwards, so a callback
whose late run matters re-checks its own state under its own guard.

A timer lives until it is cancelled (an `Interval`) or has fired (a
`Timeout`), as pi's do: its creator cancels what it no longer wants.

An exception from `fn` ends the timer (an `Interval` stops ticking) and goes
to `on_error` when one is given; without one it is dropped (pi would crash
on it as an uncaught exception).
"""

from collections.abc import Callable

import tonio.colored as tonio


type ErrorHandler = Callable[[Exception], None]


def _start(
    cancelled: tonio.Event, delay_ms: float, fn: Callable[[], None], repeat: bool, on_error: ErrorHandler | None
) -> None:
    tonio.spawn.without_tracking(_run(cancelled, delay_ms / 1000, fn, repeat, on_error))


async def _run(
    cancelled: tonio.Event, delay_s: float, fn: Callable[[], None], repeat: bool, on_error: ErrorHandler | None
) -> None:
    while True:
        await cancelled.wait(delay_s)
        if cancelled.is_set():
            return
        try:
            fn()
        except Exception as error:
            if on_error is not None:
                on_error(error)
            return
        if not repeat:
            return


class Timeout:
    """One-shot timer: run `fn` after `delay_ms` unless cancelled first."""

    __slots__ = ("_cancelled",)

    def __init__(self, delay_ms: float, fn: Callable[[], None], on_error: ErrorHandler | None = None) -> None:
        self._cancelled = tonio.Event()
        _start(self._cancelled, delay_ms, fn, False, on_error)

    def cancel(self) -> None:
        self._cancelled.set()


class Interval:
    """Repeating timer: run `fn` every `delay_ms` until cancelled."""

    __slots__ = ("_cancelled",)

    def __init__(self, delay_ms: float, fn: Callable[[], None], on_error: ErrorHandler | None = None) -> None:
        self._cancelled = tonio.Event()
        _start(self._cancelled, delay_ms, fn, True, on_error)

    def cancel(self) -> None:
        self._cancelled.set()
