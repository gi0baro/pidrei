"""Mirror of pi's google-shared-retry.test.ts.

pi drives the backoff with vitest fake timers; tonio has no fake clock, so the
retry case pays one small real backoff delay instead.
"""

import pytest

from pidrei_ai.api.google_client import GoogleApiError
from pidrei_ai.api.google_shared import retry_google_request
from pidrei_ai.types import StreamOptions


def google_api_error(status: int) -> GoogleApiError:
    """Shaped like the SDK's ApiError: has `status`, but no `headers`."""
    return GoogleApiError(status, f"got status: {status}")


@pytest.mark.tonio
async def test_retries_a_headers_less_sdk_error_with_a_retryable_status():
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise google_api_error(429)
        return "ok"

    result = await retry_google_request(request, StreamOptions(max_retries=1))

    assert result == "ok"
    assert calls == 2


@pytest.mark.tonio
async def test_does_not_retry_when_max_retries_is_unset():
    calls = 0
    error = google_api_error(429)

    async def request():
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(GoogleApiError) as excinfo:
        await retry_google_request(request)

    assert excinfo.value is error
    assert calls == 1


@pytest.mark.tonio
async def test_does_not_retry_a_non_retryable_status():
    calls = 0
    error = google_api_error(400)

    async def request():
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(GoogleApiError) as excinfo:
        await retry_google_request(request, StreamOptions(max_retries=2))

    assert excinfo.value is error
    assert calls == 1
