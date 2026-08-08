"""Mirror of pi ai src/utils/abort.ts.

pi's `raceWithAbortSignal` stops waiting on a promise while continuing to
observe it so a later rejection is always handled. A coroutine has no
independent life of its own, so the operation is spawned as a detached task
whose outcome lands in a box — the box swallows a post-abandonment failure the
same way pi's `.catch(() => {})` does.
"""

from collections.abc import Coroutine
from typing import Any

import tonio.colored as tonio

from pidrei_ai.utils.cancel import AbortError, CancelToken


def operation_cancel(cancel: CancelToken | None) -> CancelToken:
    """Create an operation-local token for public APIs whose token is optional."""
    return cancel if cancel is not None else CancelToken()


def _abort_reason(cancel: CancelToken) -> BaseException:
    reason = cancel.reason
    return reason if reason is not None else AbortError("The operation was aborted")


async def race_with_cancel[T](operation: Coroutine[Any, Any, T], cancel: CancelToken) -> T:
    """Stop waiting for an operation when its token cancels while letting the
    abandoned operation run to completion as a detached task.

    Cancellation settles the race synchronously inside `cancel()` (pi's abort
    listener), so an operation failure caused by the same cancellation can
    never win the race against the abort reason."""
    settled = tonio.Event()
    outcome: list[tuple[str, Any]] = []

    def _settle(kind: str, payload: Any) -> None:
        if outcome:
            return
        outcome.append((kind, payload))
        settled.set()

    async def _run() -> None:
        try:
            value = await operation
        except BaseException as error:
            _settle("error", error)
        else:
            _settle("value", value)

    if cancel.cancelled:
        tonio.spawn.without_tracking(_run())
        raise _abort_reason(cancel)

    unsubscribe = cancel.on_cancel(lambda _reason: _settle("abort", _abort_reason(cancel)))
    tonio.spawn.without_tracking(_run())
    try:
        await settled.wait()
    finally:
        unsubscribe()
    kind, payload = outcome[0]
    if kind == "abort" or kind == "error":
        raise payload
    return payload
