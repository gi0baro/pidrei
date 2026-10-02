"""The `tui` extensions receive (spec/ui-island.md).

pidrei-only: pi hands extension factories its real `TUI`, safe on its one
thread. Here extension code runs on its own coroutines and on the UI's loops
alike, so it gets a narrowed, guarded surface instead: everything on it
either takes the UI state lock itself or needs no lock, and the raw lock is
not reachable. The components extensions build with it (our `Editor`,
`CustomEditor`, `BorderedLoader` included) hold it too, so it also provides
what those use.
"""

import inspect
from collections.abc import Awaitable

import tonio.colored as tonio

from pidrei_tui import OverlayHandle
from pidrei_utils.timers import Interval, Timeout


__all__ = ["ExtensionTui"]


class _TerminalSize:
    """The terminal's size, read live; not the terminal itself."""

    __slots__ = ("_tui",)

    def __init__(self, tui) -> None:
        self._tui = tui

    @property
    def rows(self) -> int:
        return self._tui.terminal.rows

    @property
    def columns(self) -> int:
        return self._tui.terminal.columns


class _TimerHandle:
    """What `timeout`/`interval` return. `cancel()` marks the handle under the
    UI state lock, and every fire re-checks the mark under the same lock, so
    once `cancel()` returns the callback never runs again — pi's
    `clearTimeout`/`clearInterval` — even when a fire was already waiting
    for the lock. Callable from anywhere, the callback itself included."""

    __slots__ = ("_timer", "_tui", "cancelled")

    def __init__(self, tui) -> None:
        self._tui = tui
        self._timer = None
        self.cancelled = False

    def cancel(self) -> None:
        def mark() -> None:
            self.cancelled = True

        self._tui.apply(mark)
        self._timer.cancel()


class ExtensionTui:
    """Wraps interactive mode's stable TUI reference, so it follows renderer
    switches like the reference does."""

    __slots__ = ("_terminal", "_tui")

    def __init__(self, tui) -> None:
        self._tui = tui
        self._terminal = _TerminalSize(tui)

    @property
    def terminal(self) -> _TerminalSize:
        return self._terminal

    def request_render(self, force: bool = False) -> None:
        self._tui.request_render(force)

    def report_error(self, error: BaseException) -> None:
        self._tui.report_error(error)

    def apply(self, fn):
        """Run the synchronous `fn` under the UI state lock and return its
        result; coroutines are refused (see `TuiBase.apply`)."""
        return self._tui.apply(fn)

    def spawn(self, coro) -> None:
        """Run `coro` fire-and-forget; whatever escapes it goes to
        `report_error`."""

        async def run() -> None:
            try:
                await coro
            except Exception as error:
                self._tui.report_error(error)

        tonio.spawn.without_tracking(run())

    def timeout(self, delay_ms: float, fn) -> _TimerHandle:
        """Call `fn` once after `delay_ms`, under the UI state lock. Returns
        a handle with `cancel()`."""
        return self._timer(Timeout, delay_ms, fn)

    def interval(self, delay_ms: float, fn) -> _TimerHandle:
        """Call `fn` every `delay_ms`, under the UI state lock, until
        `cancel()` (or until it raises). Returns a handle with `cancel()`."""
        return self._timer(Interval, delay_ms, fn)

    def _timer(self, kind, delay_ms: float, fn) -> _TimerHandle:
        handle = _TimerHandle(self._tui)
        handle._timer = kind(delay_ms, self._guarded(fn, handle), on_error=self._tui.report_error)
        return handle

    def _guarded(self, fn, handle: _TimerHandle):
        if inspect.iscoroutinefunction(fn):
            raise TypeError("timer callbacks run under the UI state lock and must be synchronous")

        def fire() -> None:
            # Re-checked under the lock: a fire already waiting for it when
            # `cancel()` ran does not call `fn` (the timer's own check is
            # made before it waits).
            if not handle.cancelled:
                fn()

        return lambda: self._tui.apply(fire)

    def set_focus(self, component) -> None:
        self._tui.apply(lambda: self._tui.set_focus(component))

    def show_overlay(self, component, options: dict | None = None) -> OverlayHandle:
        return guard_overlay_handle(self._tui, self._tui.apply(lambda: self._tui.show_overlay(component, options)))

    def hide_overlay(self) -> None:
        self._tui.apply(lambda: self._tui.hide_overlay())

    def finish_before_next_input(self, handle) -> None:
        self._tui.finish_before_next_input(handle)

    def stop(self) -> Awaitable[None]:
        return self._tui.stop()

    def start(self) -> Awaitable[None]:
        return self._tui.start()


def guard_overlay_handle(tui, handle: OverlayHandle) -> OverlayHandle:
    """The same handle, with every call going through the UI state lock."""

    def guard(fn):
        return lambda *args, **kwargs: tui.apply(lambda: fn(*args, **kwargs))

    return OverlayHandle(
        hide=guard(handle.hide),
        set_hidden=guard(handle.set_hidden),
        is_hidden=guard(handle.is_hidden),
        focus=guard(handle.focus),
        unfocus=guard(handle.unfocus),
        is_focused=guard(handle.is_focused),
        get_bounds=guard(handle.get_bounds),
    )
