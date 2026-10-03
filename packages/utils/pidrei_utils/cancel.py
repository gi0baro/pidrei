"""Cancellation primitives mirroring pi's `AbortSignal` semantics.

pi threads `AbortSignal` tokens through every layer and checks them
cooperatively (`signal.aborted`) after suspension points. pidrei keeps the
token as the *edge* object (pi's API shape, checked where pi checks it); the
*mechanism* behind it is tonio's structured cancellation: work that must be
interruptible runs as the child of a scope whose owner waits for either
completion or the token (`EventStream.spawn_producer`, `run_cancellable`
below), and the token's `on_cancel`
cancels that scope. A cancelled child is unwound at its current suspension point and may
not await waiters afterwards, so async teardown belongs to the owner.
"""

import threading
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio
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
    def event(self) -> Event:
        """The token's Event, for composed waits (`Waiter.any(token.event, ...)`)."""
        return self._event

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


def _abort_reason(cancel: CancelToken) -> BaseException:
    reason = cancel.reason
    return reason if reason is not None else AbortError("The operation was aborted")


async def run_cancellable[T](operation: Coroutine[Any, Any, T], cancel: CancelToken | None) -> T:
    """Await `operation`; if `cancel` fires first, unwind it and raise the reason.

    The scope-owned shape (same as `EventStream.spawn_producer`): the
    operation runs as the child of a scope, the caller waits inside that
    scope for either outcome, and leaving the scope after a cancel is what
    unwinds the child at its current suspension point — a pending request
    head, a parked read, a backoff sleep. Unlike `race_with_cancel`, the
    abandoned operation does not keep running.
    """
    if cancel is None or cancel.never:
        return await operation
    if cancel.cancelled:
        operation.close()
        raise _abort_reason(cancel)

    settled = tonio.Event()
    outcome = tonio.Result()
    # Who owns `operation`: the child claims it before awaiting it; after the
    # scope, the caller closes it only if the child never did (see below).
    claim_guard = threading.Lock()
    claim = {"child": False, "abandoned": False}

    async def _child() -> None:
        try:
            with claim_guard:
                if claim["abandoned"]:
                    return
                claim["child"] = True
            outcome.store((False, await operation))
        except Exception as error:
            # Delivery is `raise payload` at the call site below: escaping
            # further would only double-report through tonio's
            # unhandled-coroutine printer on stdout. A cancel is reported as
            # the token's reason below.
            outcome.store((True, error))
        finally:
            settled.set()

    def _on_cancel(_reason: BaseException) -> None:
        scope.cancel()
        settled.set()

    async with tonio.scope() as scope:
        scope.spawn(_child())
        unsubscribe = cancel.on_cancel(_on_cancel)
        await settled.wait()
    unsubscribe()
    # A cancel landing before the child first runs aborts it unstarted, so the
    # operation is never awaited: close it (no body runs) rather than leave a
    # dropped coroutine to the garbage collector. Leaving the scope does not
    # interrupt a child already running on another worker, so a state probe
    # could close the coroutine the child is about to await: ownership is
    # settled under the claim lock instead — a child that starts late sees
    # the abandonment and leaves the operation alone.
    with claim_guard:
        claim["abandoned"] = not claim["child"]
    if claim["abandoned"]:
        operation.close()
    stored = outcome.fetch()
    if stored is None:
        raise _abort_reason(cancel)
    failed, payload = stored
    if failed:
        raise payload
    return payload
