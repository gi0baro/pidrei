"""Mirror of pi ai src/utils/abort.ts.

Two ways to stop waiting on an operation when a token fires:

- `run_cancellable` unwinds the operation (tonio scope cancel): for requests
  and reads, where nothing useful happens after the caller walked away.
- `race_with_cancel` keeps it running detached, as pi's `raceWithAbortSignal`
  does — the operation is spawned as a detached task whose outcome lands in a
  box that swallows a post-abandonment failure the same way pi's
  `.catch(() => {})` does: for state mutations (credential and model-store
  writes) that must not be torn by an abort.

Both are free when the token is `None` or the shared placeholder.
"""

import threading
from collections.abc import Coroutine
from typing import Any

import tonio.colored as tonio
from tonio.exceptions import CancelledError

from pidrei_ai.utils.cancel import NEVER_CANCELLED, AbortError, CancelToken


def operation_cancel(cancel: CancelToken | None) -> CancelToken:
    """The token for public APIs whose token is optional: the caller's, or the
    shared never-firing placeholder (so callees need no `None` branches)."""
    return cancel if cancel is not None else NEVER_CANCELLED


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
        except CancelledError:
            raise  # reported as the token's reason below
        except GeneratorExit:
            raise  # coroutine close protocol, not an operation failure
        except BaseException as error:
            # Delivery is `raise payload` at the call site below: escaping
            # further would only double-report through tonio's
            # unhandled-coroutine printer on stdout.
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
    kind, payload = outcome.fetch()
    if kind == "abort" or kind == "error":
        raise payload
    return payload
