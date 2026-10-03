"""Mirror of pi ai src/utils/abort.ts.

Two ways to stop waiting on an operation when a token fires:

- `pidrei_utils.cancel.run_cancellable` unwinds the operation (tonio scope
  cancel): for requests and reads, where nothing useful happens after the
  caller walked away. It is pidrei's own and lives with the token, so the
  packages that do not depend on pidrei-ai have it too.
- `race_with_cancel` keeps it running detached, as pi's `raceWithAbortSignal`
  does — the operation is spawned as a detached task whose outcome lands in a
  box that swallows a post-abandonment failure the same way pi's
  `.catch(() => {})` does: for state mutations (credential and model-store
  writes) that must not be torn by an abort.

Both are free when the token is `None` or the shared placeholder.
"""

from collections.abc import Coroutine
from typing import Any

import tonio.colored as tonio

from pidrei_utils.cancel import NEVER_CANCELLED, AbortError, CancelToken


def operation_cancel(cancel: CancelToken | None) -> CancelToken:
    """The token for public APIs whose token is optional: the caller's, or the
    shared never-firing placeholder (so callees need no `None` branches)."""
    return cancel if cancel is not None else NEVER_CANCELLED


def _abort_reason(cancel: CancelToken) -> BaseException:
    reason = cancel.reason
    return reason if reason is not None else AbortError("The operation was aborted")


async def race_with_cancel[T](operation: Coroutine[Any, Any, T], cancel: CancelToken | None) -> T:
    """Stop waiting for an operation when its token cancels while letting the
    abandoned operation run to completion as a detached task.

    Cancellation settles the race synchronously inside `cancel()` (pi's abort
    listener), so an operation failure caused by the same cancellation can
    never win the race against the abort reason. A token that cannot fire
    awaits the operation inline: no task, no box, no subscription."""
    if cancel is None or cancel.never:
        return await operation

    settled = tonio.Event()
    outcome = tonio.Result()

    def _settle(kind: str, payload: Any) -> None:
        if settled.is_set():
            return
        outcome.store((kind, payload))
        settled.set()

    async def _run() -> None:
        try:
            value = await operation
        except Exception as error:
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
    kind, payload = outcome.fetch()
    if kind == "abort" or kind == "error":
        raise payload
    return payload
