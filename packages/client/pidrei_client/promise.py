"""One-shot promise primitives (counterpart of pi client `promise.ts`).

pi polyfills `Promise.withResolvers()`; the Python analogue is a settle-once
slot built on a tonio `Event`. Unlike a coroutine, a `Deferred` supports any
number of awaiters — every place pi shares one promise between consumers
(handshakes, pending requests, detachments, release chains) needs this, since
a Python coroutine can only be awaited once. `wait()` returns a fresh
coroutine per call; awaiting after settlement resolves immediately.
"""

from typing import Any

from tonio.colored import Event


class Deferred:
    """Settle-once value/error slot; the first `resolve`/`reject` wins."""

    __slots__ = ("_error", "_event", "_value")

    def __init__(self) -> None:
        self._event = Event()
        self._value: Any = None
        self._error: BaseException | None = None

    def resolve(self, value: Any = None) -> None:
        if self._event.is_set():
            return
        self._value = value
        self._event.set()

    def reject(self, error: BaseException) -> None:
        if self._event.is_set():
            return
        self._error = error
        self._event.set()

    async def wait(self) -> Any:
        await self._event.wait()
        if self._error is not None:
            raise self._error
        return self._value

    async def wait_silenced(self) -> None:
        """Await settlement, swallowing rejection (pi's `.catch(() => {})`)."""
        await self._event.wait()


async def _resolved_coroutine(value: Any) -> Any:
    return value


async def _rejected_coroutine(error: BaseException) -> Any:
    raise error


def resolved(value: Any = None) -> Any:
    """An already-resolved awaitable (pi's `Promise.resolve(value)`)."""
    return _resolved_coroutine(value)


def rejected(error: BaseException) -> Any:
    """An already-rejected awaitable (pi's `Promise.reject(error)`).

    pi's async methods convert synchronous throws into rejections; sync-prologue
    ports use this to surface prologue failures at await time the same way.
    """
    return _rejected_coroutine(error)
