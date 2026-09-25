"""Mirror of pi coding-agent test/model-catalog-refresh.test.ts.

pi's `refreshModelCatalogs` starts the runtime refresh synchronously; pidrei
detaches it onto the runtime, so call-count assertions wait for the fake
runtime's refresh to be reached. Callers run as spawned tasks whose outcomes
are captured (awaiting a spawn handle wraps failures in a SpawnExceptionGroup);
each caller's token reports when the caller has joined the shared refresh.
"""

import pytest
import tonio.colored as tonio

from pidrei.modes.interactive.model_catalog_refresh import (
    _model_catalog_refresh_coordinator,
    refresh_model_catalogs,
)
from pidrei_ai.registry import ModelsRefreshResult
from pidrei_ai.utils.cancel import AbortError, CancelToken


def successful_refresh() -> ModelsRefreshResult:
    return ModelsRefreshResult(aborted=False, errors={})


class FakeRuntime:
    def __init__(self) -> None:
        self.calls = []
        self.release = tonio.Event()
        self.result = successful_refresh()
        self._called = [tonio.Event(), tonio.Event()]

    async def refresh(self, options=None):
        self.calls.append(options)
        if len(self.calls) <= len(self._called):
            self._called[len(self.calls) - 1].set()
        await self.release.wait()
        return self.result

    async def until_called(self, times: int) -> None:
        called = self._called[times - 1]
        await called.wait(5)
        assert called.is_set(), f"runtime refresh reached {len(self.calls)} of {times} times"


class JoiningToken(CancelToken):
    """A caller's token that reports when the caller has joined the shared
    refresh: the coordinator counts the waiter, then subscribes to the token
    while it waits (pi's callers join synchronously; here each is a spawned
    task that joins when it first runs)."""

    __slots__ = ("joined",)

    def __init__(self) -> None:
        super().__init__()
        self.joined = tonio.Event()

    def on_cancel(self, callback):
        unsubscribe = super().on_cancel(callback)
        self.joined.set()
        return unsubscribe


async def _joined(*tokens: JoiningToken) -> None:
    for token in tokens:
        await token.joined.wait(5)
        assert token.joined.is_set(), "caller never joined the shared refresh"


async def _settled(coro):
    try:
        return "fulfilled", await coro
    except BaseException as error:
        return "rejected", error


@pytest.mark.tonio
async def test_shares_one_runtime_refresh_between_concurrent_callers():
    runtime = FakeRuntime()
    first_controller = JoiningToken()
    second_controller = JoiningToken()

    first = tonio.spawn(_settled(refresh_model_catalogs(runtime, first_controller)))
    second = tonio.spawn(_settled(refresh_model_catalogs(runtime, second_controller)))
    await _joined(first_controller, second_controller)
    await runtime.until_called(1)
    assert len(runtime.calls) == 1

    runtime.release.set()
    assert await first == ("fulfilled", runtime.result)
    assert await second == ("fulfilled", runtime.result)


@pytest.mark.tonio
async def test_keeps_the_shared_refresh_alive_when_one_caller_stops_waiting():
    runtime = FakeRuntime()
    first_controller = JoiningToken()
    second_controller = JoiningToken()
    first = tonio.spawn(_settled(refresh_model_catalogs(runtime, first_controller)))
    second = tonio.spawn(_settled(refresh_model_catalogs(runtime, second_controller)))
    # Both callers must be counted before one leaves: the last waiter leaving
    # aborts the shared refresh, which is the coordinator working as designed.
    await _joined(first_controller, second_controller)
    await runtime.until_called(1)
    assert len(runtime.calls) == 1

    first_controller.cancel()
    status, error = await first
    assert status == "rejected"
    assert isinstance(error, AbortError)
    refresh_cancel = runtime.calls[0].cancel
    assert refresh_cancel is not None and refresh_cancel.cancelled is False

    runtime.release.set()
    assert await second == ("fulfilled", runtime.result)


@pytest.mark.tonio
async def test_aborts_an_abandoned_refresh_and_allows_a_later_refresh_to_start():
    runtime = FakeRuntime()  # release never set: the refresh hangs like pi's unresolved promise
    first_controller = JoiningToken()
    first = tonio.spawn(_settled(refresh_model_catalogs(runtime, first_controller)))
    await _joined(first_controller)
    await runtime.until_called(1)
    abandoned = _model_catalog_refresh_coordinator._active_by_runtime.get(runtime)
    assert abandoned is not None

    first_controller.cancel()
    status, error = await first
    assert status == "rejected"
    assert isinstance(error, AbortError)
    await runtime.calls[0].cancel.wait(5)
    assert runtime.calls[0].cancel.cancelled is True
    # Let the abandoned shared refresh settle and clean up (pi's waitFor gets
    # this for free from microtask ordering) before joining a new caller: its
    # `done` is set after the entry is removed.
    await abandoned.done.wait(5)
    assert abandoned.done.is_set()

    second_controller = JoiningToken()
    second = tonio.spawn(_settled(refresh_model_catalogs(runtime, second_controller)))
    await _joined(second_controller)
    await runtime.until_called(2)
    second_controller.cancel()
    status, error = await second
    assert status == "rejected"
    assert isinstance(error, AbortError)
