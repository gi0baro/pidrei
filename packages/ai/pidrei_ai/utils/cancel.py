"""Cancellation primitives mirroring pi's `AbortSignal` semantics.

pi threads `AbortSignal` tokens through every layer and checks them
cooperatively (`signal.aborted`) after suspension points. pidrei keeps the
token as the *edge* object (pi's API shape, checked where pi checks it); the
*mechanism* behind it is tonio's structured cancellation: work that must be
interruptible runs as the child of a scope whose owner waits for either
completion or the token (`EventStream.spawn_producer`,
`utils.abort.run_cancellable`), and the token's `on_cancel` cancels that
scope. A cancelled child is unwound at its current suspension point and may
not await waiters afterwards, so async teardown belongs to the owner.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass

from tonio.colored import Event, Waiter


class AbortError(Exception):
    """Raised when an operation is aborted through a `CancelToken`."""


class CancelToken:
    """Mirror of `AbortController`/`AbortSignal`, fused into a single object.

    Unlike DOM signals, callbacks registered after cancellation fire
    immediately: on a multi-threaded runtime the DOM behavior would make
    "check `cancelled`, then register" racy against concurrent cancellation.
    """

    __slots__ = ("_callbacks", "_event", "_lock", "_reason")

    def __init__(self) -> None:
        self._event = Event()
        self._lock = threading.Lock()
        self._reason: BaseException | None = None
        self._callbacks: list[Callable[[BaseException], None]] = []

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> BaseException | None:
        return self._reason

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

    def on_cancel(self, callback: Callable[[BaseException], None]) -> Callable[[], None]:
        """Register a cancellation callback; returns an unsubscribe function.

        If the token is already cancelled the callback is invoked immediately
        (see class docstring) and the returned unsubscribe is a no-op.
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
        callback(self._reason)  # type: ignore[arg-type]
        return lambda: None

    def wait(self, timeout: float | None = None) -> Waiter:
        """Awaitable resolving once the token is cancelled (or after `timeout`)."""
        return self._event.wait(timeout)

    @property
    def never(self) -> bool:
        """True for the shared placeholder that can never fire (`NEVER_CANCELLED`)."""
        return False


class _NeverCancel(CancelToken):
    """The placeholder behind optional tokens: nobody holds it, so it never fires.

    Races and subscriptions against it are skipped outright (`race_with_cancel`,
    `combine_cancel_tokens`), which is what makes an optional token free on the
    path that never passes one.
    """

    __slots__ = ()

    @property
    def never(self) -> bool:
        return True

    def cancel(self, reason: BaseException | None = None) -> None:
        raise RuntimeError("NEVER_CANCELLED is a shared placeholder; create a CancelToken to cancel")

    def on_cancel(self, callback: Callable[[BaseException], None]) -> Callable[[], None]:
        return lambda: None


NEVER_CANCELLED = _NeverCancel()


@dataclass(slots=True)
class CombinedCancel:
    """Mirror of pi's `CombinedAbortSignal` (packages/ai/src/utils/abort-signals.ts)."""

    token: CancelToken | None
    cleanup: Callable[[], None]


def combine_cancel_tokens(*tokens: CancelToken | None) -> CombinedCancel:
    """Mirror of pi's `combineAbortSignals`: a token cancelled when any input is.

    Call `cleanup()` when done with the combined token to detach it from the
    input tokens.
    """
    active = [token for token in tokens if token is not None and not token.never]
    if not active:
        return CombinedCancel(None, lambda: None)
    if len(active) == 1:
        return CombinedCancel(active[0], lambda: None)

    combined = CancelToken()
    unsubscribes: list[Callable[[], None]] = []
    for token in active:
        unsubscribes.append(token.on_cancel(combined.cancel))
        if combined.cancelled:
            break

    def cleanup() -> None:
        for unsubscribe in unsubscribes:
            unsubscribe()

    return CombinedCancel(combined, cleanup)
