"""Mirror of pi's provider-retry.test.ts.

pi drives the backoff with vitest fake timers; tonio has no fake clock, so the
delay-based cases use small real delays instead. Semantics under test are
identical.
"""

import pytest
import tonio.colored as tonio

from pppi_ai.utils.cancel import AbortError, CancelToken
from pppi_ai.utils.provider_retry import retry_provider_request


class ProviderError(Exception):
    def __init__(self, status: int | None, headers: dict[str, str] | None = None):
        super().__init__(f"Provider error: {status}")
        self.status = status
        self.headers = headers or {}


@pytest.mark.tonio
async def test_retries_retryable_provider_errors():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError(429, {"retry-after-ms": "50"})
        return "ok"

    result = await retry_provider_request(request, max_retries=1)
    assert result == "ok"
    assert calls == 2


@pytest.mark.tonio
async def test_does_not_retry_errors_marked_non_retryable():
    calls = 0
    error = ProviderError(429, {"x-should-retry": "false"})

    async def request():
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(ProviderError) as excinfo:
        await retry_provider_request(request, max_retries=2)
    assert excinfo.value is error
    assert calls == 1


@pytest.mark.tonio
async def test_rejects_provider_requested_retry_delay_above_the_limit():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        raise ProviderError(429, {"retry-after": "277403"})

    with pytest.raises(RuntimeError, match=r"Server requested 277403s retry delay \(max: 1s\)"):
        await retry_provider_request(request, max_retries=1, max_retry_delay_ms=1000)
    assert calls == 1


@pytest.mark.tonio
async def test_allows_disabling_the_provider_requested_retry_delay_cap():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError(429, {"retry-after": "0.05"})
        return "ok"

    result = await retry_provider_request(request, max_retries=1, max_retry_delay_ms=0)
    assert result == "ok"
    assert calls == 2


@pytest.mark.tonio
async def test_aborts_a_provider_requested_retry_delay():
    cancel = CancelToken()
    calls = 0
    requested = tonio.Event()

    async def request():
        nonlocal calls
        calls += 1
        requested.set()
        raise ProviderError(429, {"retry-after": "277403"})

    async def run():
        try:
            await retry_provider_request(request, max_retries=2, max_retry_delay_ms=0, cancel=cancel)
        except AbortError as error:
            return error
        return None

    async def abort_after_first_request():
        await requested.wait()
        await tonio.yield_now()
        cancel.cancel()

    outcome, _ = await tonio.spawn(run(), abort_after_first_request())
    assert isinstance(outcome, AbortError)
    assert calls == 1


@pytest.mark.tonio
async def test_retries_when_status_is_none():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError(None)
        return "ok"

    assert await retry_provider_request(request, max_retries=1) == "ok"
    assert calls == 2


@pytest.mark.tonio
async def test_does_not_retry_non_provider_errors():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        raise ValueError("plain error")

    with pytest.raises(ValueError, match="plain error"):
        await retry_provider_request(request, max_retries=3)
    assert calls == 1
