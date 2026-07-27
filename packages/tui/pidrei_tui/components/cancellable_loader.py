"""Loader that can be cancelled with Escape (port of pi tui ``components/cancellable-loader.ts``).

Extends Loader with a cancel token for cancelling async operations. pi uses a
DOM ``AbortController``; pidrei's tui-local ``CancelToken`` mirrors the subset
of ``pidrei_ai.utils.cancel.CancelToken`` consumers rely on (``cancelled``,
``cancel()``, ``on_cancel``, ``wait``) so loader signals stay duck-type
compatible with the ai layer without a cross-package dependency.

Example::

    loader = CancellableLoader(tui, cyan, dim, "Working...")
    loader.on_abort = lambda: done(None)
    do_work(loader.signal)
"""

import threading

from tonio.colored import Event

from ..keybindings import get_keybindings
from .loader import Loader


__all__ = ["CancelToken", "CancellableLoader"]


class CancelToken:
    """Minimal cancel token (see module docstring)."""

    __slots__ = ("_callbacks", "_event", "_lock")

    def __init__(self) -> None:
        self._event = Event()
        self._lock = threading.Lock()
        self._callbacks: list = []

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        """Cancel the token. Only the first call has any effect."""
        with self._lock:
            if self._event.is_set():
                return
            callbacks, self._callbacks = self._callbacks, []
            self._event.set()
        for callback in callbacks:
            callback()

    def on_cancel(self, callback):
        """Register a cancellation callback; returns an unsubscribe function.

        If the token is already cancelled the callback is invoked immediately.
        """
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(callback)

                def unsubscribe() -> None:
                    with self._lock:
                        try:
                            self._callbacks.remove(callback)
                        except ValueError:
                            pass

                return unsubscribe
        callback()
        return lambda: None

    def wait(self, timeout: float | None = None):
        """Awaitable resolving once the token is cancelled."""
        return self._event.wait(timeout)


class CancellableLoader(Loader):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._token = CancelToken()
        # Called when user presses Escape
        self.on_abort = None

    @property
    def signal(self) -> CancelToken:
        """Token that is cancelled when user presses Escape."""
        return self._token

    @property
    def aborted(self) -> bool:
        """Whether the loader was aborted."""
        return self._token.cancelled

    async def handle_input(self, data: str) -> None:
        kb = get_keybindings()
        if kb.matches(data, "tui.select.cancel"):
            self._token.cancel()
            if self.on_abort is not None:
                self.on_abort()

    def dispose(self) -> None:
        self.stop()
