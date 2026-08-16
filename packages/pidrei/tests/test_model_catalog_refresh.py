"""Mirror of pi coding-agent test/model-catalog-refresh.test.ts.

pi's `refreshModelCatalogs` starts the runtime refresh synchronously; pidrei
detaches it onto the runtime, so call-count assertions settle behind a short
wait. Callers run as spawned tasks whose outcomes are captured (awaiting a
spawn handle wraps failures in a SpawnExceptionGroup).
"""

import pytest
import tonio.colored as tonio

from pidrei.modes.interactive.model_catalog_refresh import refresh_model_catalogs
from pidrei_ai.registry import ModelsRefreshResult
from pidrei_ai.utils.cancel import AbortError, CancelToken


def successful_refresh() -> ModelsRefreshResult:
    return ModelsRefreshResult(aborted=False, errors={})


class FakeRuntime:
    def __init__(self) -> None:
        self.calls = []
        self.release = tonio.Event()
        self.result = successful_refresh()

    async def refresh(self, options=None):
        self.calls.append(options)
        await self.release.wait()
        return self.result


async def _settled(coro):
    try:
        return "fulfilled", await coro
    except BaseException as error:
        return "rejected", error


async def _wait_until(condition, timeout=2.0):
    waited = 0.0
    while not condition():
        await tonio.time.sleep(0.005)
        waited += 0.005
        if waited >= timeout:
            raise AssertionError("condition not reached before timeout")


@pytest.mark.tonio
async def test_shares_one_runtime_refresh_between_concurrent_callers():
    runtime = FakeRuntime()
    first_controller = CancelToken()
    second_controller = CancelToken()

    first = tonio.spawn(_settled(refresh_model_catalogs(runtime, first_controller)))
    second = tonio.spawn(_settled(refresh_model_catalogs(runtime, second_controller)))
    await _wait_until(lambda: len(runtime.calls) >= 1)
    await tonio.time.sleep(0.05)
    assert len(runtime.calls) == 1

    runtime.release.set()
    assert await first == ("fulfilled", runtime.result)
    assert await second == ("fulfilled", runtime.result)


@pytest.mark.tonio
async def test_keeps_the_shared_refresh_alive_when_one_caller_stops_waiting():
    runtime = FakeRuntime()
    first_controller = CancelToken()
    second_controller = CancelToken()
    first = tonio.spawn(_settled(refresh_model_catalogs(runtime, first_controller)))
    second = tonio.spawn(_settled(refresh_model_catalogs(runtime, second_controller)))
    await _wait_until(lambda: len(runtime.calls) >= 1)

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
    first_controller = CancelToken()
    first = tonio.spawn(_settled(refresh_model_catalogs(runtime, first_controller)))
    await _wait_until(lambda: len(runtime.calls) >= 1)

    first_controller.cancel()
    status, error = await first
    assert status == "rejected"
    assert isinstance(error, AbortError)
    await _wait_until(lambda: runtime.calls[0].cancel.cancelled is True)
    # Let the abandoned shared refresh settle and clean up (pi's waitFor gets
    # this for free from microtask ordering) before joining a new caller.
    await tonio.time.sleep(0.05)

    second_controller = CancelToken()
    second = tonio.spawn(_settled(refresh_model_catalogs(runtime, second_controller)))
    await _wait_until(lambda: len(runtime.calls) >= 2)
    second_controller.cancel()
    status, error = await second
    assert status == "rejected"
    assert isinstance(error, AbortError)
