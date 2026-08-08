"""One-shot promise primitives for the remote-session client layer.

Upstream `src/client/` has no promise.ts — `RemoteSession` uses raw promises
(the dispose signal, the memoized dispose promise, the tracked attachment
operations raced against disposal). The Python port needs an explicit slot
for them because they are shared between consumers (`dispose()` returns the
same promise on every call; a tracked attachment operation is awaited both by
its own operation and by disposal cleanup) and a coroutine can only be
awaited once.

Same flavour as the server package's promise module: `Deferred` is itself
awaitable and `resolved`/`rejected` return settled `Deferred`s rather than
coroutines, so a promise nobody awaits — the loser of pi's
`Promise.race([running, disposeSignal])` after a preempting dispose — is
inert instead of a dropped-coroutine warning.
"""

from collections.abc import Awaitable, Coroutine
from typing import Any

import tonio.colored as tonio
from tonio.colored import Event


class Deferred:
    """Settle-once value/error slot; the first `resolve`/`reject` wins.

    Awaitable any number of times, both directly and via `wait()`; safe to
    drop without awaiting.
    """

    __slots__ = ("_error", "_event", "_value")

    def __init__(self) -> None:
        self._event = Event()
        self._value: Any = None
        self._error: BaseException | None = None

    @property
    def settled(self) -> bool:
        return self._event.is_set()

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

    def __await__(self) -> Any:
        return self.wait().__await__()

    async def wait(self) -> Any:
        await self._event.wait()
        if self._error is not None:
            raise self._error
        return self._value

    async def wait_silenced(self) -> None:
        """Await settlement, swallowing rejection (pi's `.catch(() => {})`)."""
        await self._event.wait()


def resolved(value: Any = None) -> Deferred:
    """An already-resolved awaitable (pi's `Promise.resolve(value)`)."""
    deferred = Deferred()
    deferred.resolve(value)
    return deferred


def rejected(error: BaseException) -> Deferred:
    """An already-rejected awaitable (pi's `Promise.reject(error)`).

    pi's async methods convert synchronous throws into rejections; sync-prologue
    ports use this to surface prologue failures at await time the same way.
    """
    deferred = Deferred()
    deferred.reject(error)
    return deferred


def driven(coro: Coroutine[Any, Any, Any]) -> Deferred:
    """Run `coro` on the runtime and mirror its outcome in a `Deferred`.

    The JS shape where an async function body executes whether or not anyone
    awaits the returned promise.
    """
    deferred = Deferred()

    async def _drive() -> None:
        try:
            result = await coro
        except Exception as error:
            deferred.reject(error)
            return
        deferred.resolve(result)

    tonio.spawn.without_tracking(_drive())
    return deferred


async def all_settled(awaitables: list[Awaitable[Any]]) -> list[tuple[Any, BaseException | None]]:
    """Start every awaitable concurrently and await them all (`Promise.allSettled`)."""
    driven_all = [aw if isinstance(aw, Deferred) else driven(aw) for aw in awaitables]
    results: list[tuple[Any, BaseException | None]] = []
    for deferred in driven_all:
        try:
            value = await deferred
        except Exception as error:
            results.append((None, error))
            continue
        results.append((value, None))
    return results
