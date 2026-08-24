"""Loader that can be cancelled with Escape (port of pi tui ``components/cancellable-loader.ts``).

Extends Loader with a cancel token for cancelling async operations. pi uses a
DOM ``AbortController``; pidrei's tui-local ``CancelToken`` mirrors the caller
surface of ``pidrei_ai.utils.cancel.CancelToken`` (``cancelled``, ``reason``,
``never``, ``cancel(reason)``, ``raise_if_cancelled``, ``on_cancel``,
``wait``) so loader signals stay duck-type compatible with the ai layer
without a cross-package dependency. ``on_cancel`` callbacks receive the abort
reason, exactly like the ai token — the ai layer registers
``callback(reason)`` subscribers on caller tokens
(``pidrei_ai.utils.abort.run_cancellable``/``race_with_cancel``).
``tests/test_cancel_token_mirror.py`` in the pidrei package holds the two
classes to this contract.

Example::

    loader = CancellableLoader(tui, cyan, dim, "Working...")
    loader.on_abort = lambda: done(None)
    do_work(loader.signal)
"""

import threading

from tonio.colored import Event

from ..keybindings import get_keybindings
from .loader import Loader


__all__ = ["AbortError", "CancelToken", "CancellableLoader"]


class AbortError(Exception):
    """Default abort reason (mirror of `pidrei_ai.utils.cancel.AbortError`)."""


class CancelToken:
    """Minimal cancel token (see module docstring)."""

    __slots__ = ("_callbacks", "_event", "_lock", "_reason")

    def __init__(self) -> None:
        self._event = Event()
        self._lock = threading.Lock()
        self._reason: BaseException | None = None
        self._callbacks: list = []

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> BaseException | None:
        return self._reason

    @property
    def never(self) -> bool:
        """Mirror of the ai token's placeholder probe; a real token can fire."""
        return False

    def cancel(self, reason: BaseException | None = None) -> None:
        """Cancel the token. Only the first call has any effect."""
        with self._lock:
            if self._event.is_set():
                return
            self._reason = reason if reason is not None else AbortError("Operation was aborted")
            callbacks, self._callbacks = self._callbacks, []
            self._event.set()
        for callback in callbacks:
            callback(self._reason)

    def raise_if_cancelled(self) -> None:
        """Mirror of `AbortSignal.throwIfAborted`: raise `reason` when cancelled."""
        if self._event.is_set():
            raise self._reason  # type: ignore[misc]

    def on_cancel(self, callback):
        """Register a cancellation callback; returns an unsubscribe function.

        The callback receives the abort reason. If the token is already
        cancelled the callback is invoked immediately.
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
        callback(self._reason)
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
