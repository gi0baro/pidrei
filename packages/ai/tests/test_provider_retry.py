"""Mirror of pi's provider-retry.test.ts.

pi drives the backoff with vitest fake timers; tonio has no fake clock, so the
delay-based cases use small real delays instead. Semantics under test are
identical.
"""

import pytest
import tonio.colored as tonio

from pidrei_ai.types import ProviderResponse
from pidrei_ai.utils.provider_retry import retry_provider_request
from pidrei_utils.cancel import AbortError, CancelToken


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
async def test_does_not_retry_statuses_listed_in_no_retry_statuses():
    calls = 0
    error = ProviderError(504, {"retry-after-ms": "0"})

    async def request():
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(ProviderError) as excinfo:
        await retry_provider_request(request, max_retries=2, no_retry_statuses=[504])
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


# --- provider request error responses (#12376) ---------------------------------


@pytest.mark.tonio
async def test_reports_each_failed_attempts_status_and_headers_before_retrying():
    responses: list[ProviderResponse] = []
    errors = [ProviderError(429, {"retry-after-ms": "0"}), ProviderError(503, {"retry-after-ms": "0"})]

    async def request():
        if errors:
            raise errors.pop(0)
        return "ok"

    async def on_error_response(response):
        responses.append(response)

    result = await retry_provider_request(request, max_retries=2, on_error_response=on_error_response)

    assert result == "ok"
    assert responses == [
        ProviderResponse(status=429, headers={"retry-after-ms": "0"}),
        ProviderResponse(status=503, headers={"retry-after-ms": "0"}),
    ]


@pytest.mark.tonio
async def test_reports_the_final_non_retryable_failure():
    responses: list[ProviderResponse] = []
    error = ProviderError(400, {"x-request-id": "req-1"})

    async def request():
        raise error

    async def on_error_response(response):
        responses.append(response)

    with pytest.raises(ProviderError) as excinfo:
        await retry_provider_request(request, max_retries=2, on_error_response=on_error_response)

    assert excinfo.value is error
    assert responses == [ProviderResponse(status=400, headers={"x-request-id": "req-1"})]


@pytest.mark.tonio
async def test_does_not_report_failures_without_an_http_status():
    responses: list[ProviderResponse] = []

    async def request():
        raise RuntimeError("Connection error.")

    async def on_error_response(response):
        responses.append(response)

    with pytest.raises(RuntimeError, match=r"Connection error\."):
        await retry_provider_request(request, on_error_response=on_error_response)
    assert responses == []


@pytest.mark.tonio
async def test_fails_the_request_with_the_callbacks_error_when_the_callback_throws():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        raise ProviderError(429, {"retry-after-ms": "0"})

    async def on_error_response(_response):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        await retry_provider_request(request, max_retries=2, on_error_response=on_error_response)
    assert calls == 1


@pytest.mark.tonio
async def test_does_not_report_a_failure_after_the_request_was_aborted():
    cancel = CancelToken()
    responses: list[ProviderResponse] = []

    async def request():
        cancel.cancel()
        raise ProviderError(429)

    async def on_error_response(response):
        responses.append(response)

    with pytest.raises(AbortError):
        await retry_provider_request(request, cancel=cancel, on_error_response=on_error_response)
    assert responses == []
