"""One-shot promise primitives for the server package.

Upstream pi-server has no promise.ts — its sources use raw promises
(`broadcastQueue`, `state.handshake`, `live.disposing`, memoized close/start
promises) and `test/` gets a `Deferred` from `testing/service.ts`. The Python
port needs an explicit slot for all of them, since a coroutine can only be
awaited once and pi shares these promises among several consumers.

Unlike the client package's `Deferred`, this one is itself awaitable
(`__await__`) and `resolved`/`rejected` return settled `Deferred`s rather than
coroutines: nearly every promise in pi-server is voided (`void this.foo()`),
and a dropped coroutine warns while a dropped `Deferred` is inert — the exact
semantics of dropping a JS promise whose errors are handled elsewhere.
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


async def gather(awaitables: list[Awaitable[Any]]) -> list[Any]:
    """Start every awaitable concurrently and collect results (`Promise.all`).

    Unlike `Promise.all` this settles only after every awaitable has settled,
    then raises the first (by position) failure; the fail-fast distinction is
    not observable to pi-server's call sites, which only await cleanup fan-outs.
    """
    settled = await all_settled(awaitables)
    for _, error in settled:
        if error is not None:
            raise error
    return [value for value, _ in settled]
