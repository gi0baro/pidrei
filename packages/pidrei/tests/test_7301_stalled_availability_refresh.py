"""Mirror of pi coding-agent test/suite/regressions/7301-stalled-availability-refresh.test.ts.

pi parks the stalled credential list on a deferred promise and drives the
stale read as a floating promise; here the stale reader runs as a sibling
coroutine under `tonio.spawn`, gated on `tonio.Event`s. pi's
`vi.waitFor(callCount == 2)` is synchronization noise for promises resolving
between assertions — awaiting the recovery refresh covers it, and the call
count is asserted directly.
"""

import pytest
import tonio.colored as tonio

from pidrei.core.provider_composer import AuthStatus
from pidrei_ai.auth.types import ApiKeyCredential
from pidrei_ai.registry import ModelsRefreshOptions

from .harness import create_harness


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


class StalledCredentialList:
    """Stalls the next `credentials.list()` call behind a gate."""

    def __init__(self, harness):
        original_list = harness.auth_storage.list
        self.started = tonio.Event()
        self.call_count = 0
        self._gate = tonio.Event()
        self._fail_error: Exception | None = None
        self._should_stall = True

        async def stalling_list(options=None):
            entries = await original_list(options)
            self.call_count += 1
            if not self._should_stall:
                return entries
            self._should_stall = False
            self.started.set()
            await self._gate.wait()
            if self._fail_error is not None:
                raise self._fail_error
            return entries

        harness.auth_storage.list = stalling_list

    def release(self) -> None:
        self._gate.set()

    def fail(self, error: Exception) -> None:
        self._fail_error = error
        self._gate.set()


@pytest.mark.tonio
async def test_recovers_without_letting_the_stalled_refresh_overwrite_the_newer_snapshot(harnesses):
    harness = await create_harness(with_configured_auth=False)
    harnesses.append(harness)
    runtime = harness.session.model_runtime

    async def set_stale(_current):
        return ApiKeyCredential(key="stale-key")

    async def set_current(_current):
        return ApiKeyCredential(key="current-key")

    await harness.auth_storage.modify("stale-provider", set_stale)
    await runtime.refresh(ModelsRefreshOptions(allow_network=False))
    assert runtime.get_provider_auth_status("stale-provider") == AuthStatus(configured=True, source="stored")

    stalled = StalledCredentialList(harness)

    async def run_stale_read():
        await runtime.get_available()

    async def drive():
        await stalled.started.wait()

        await harness.auth_storage.delete("stale-provider")
        await harness.auth_storage.modify("current-provider", set_current)
        await runtime.refresh(ModelsRefreshOptions(allow_network=False))
        assert stalled.call_count == 2
        assert runtime.get_provider_auth_status("stale-provider") == AuthStatus(configured=False)
        assert runtime.get_provider_auth_status("current-provider") == AuthStatus(configured=True, source="stored")

        stalled.release()

    await tonio.spawn(run_stale_read(), drive())

    assert runtime.get_provider_auth_status("stale-provider") == AuthStatus(configured=False)
    assert runtime.get_provider_auth_status("current-provider") == AuthStatus(configured=True, source="stored")


@pytest.mark.tonio
async def test_does_not_let_a_stale_failure_overwrite_newer_availability_error_state(harnesses):
    harness = await create_harness(with_configured_auth=False)
    harnesses.append(harness)
    runtime = harness.session.model_runtime

    stalled = StalledCredentialList(harness)
    stale_outcome: dict = {}

    async def run_stale_read():
        try:
            await runtime.get_available()
            stale_outcome["error"] = None
        except Exception as error:
            stale_outcome["error"] = error

    async def drive():
        await stalled.started.wait()

        await runtime.refresh(ModelsRefreshOptions(allow_network=False))
        assert stalled.call_count == 2
        assert runtime.get_error() is None

        stalled.fail(Exception("stale credential list failure"))

    await tonio.spawn(run_stale_read(), drive())

    assert stale_outcome["error"] is not None
    assert "stale credential list failure" in str(stale_outcome["error"])
    assert runtime.get_error() is None
