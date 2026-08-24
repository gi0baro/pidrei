"""Mirror of pi coding-agent test/management-http.test.ts.

pi stubs global `fetch`; pidrei routes through the `utils/http.py` seam, so
these swap `shared_client` (the same pattern as `test_version_check.py`). pi's
"does not retry caller cancellation" case has no counterpart: cancellation
reaches these requests as the shared timeout budget, covered by the
budget-exhaustion case below.
"""

import pytest

import pidrei_ai.utils.http as http_module
from pidrei.utils.management_http import fetch_with_retry


class _Response:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _ScriptedClient:
    """Plays back a script of responses/exceptions and records each timeout."""

    def __init__(self, script: list):
        self._script = list(script)
        self.calls = 0
        self.timeouts: list = []

    async def get(self, _url, *, headers=None, timeout=None):
        self.calls += 1
        self.timeouts.append(timeout)
        step = self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


@pytest.fixture
def stub_http(request):
    original = http_module.shared_client
    request.addfinalizer(lambda: setattr(http_module, "shared_client", original))

    def install(script: list) -> _ScriptedClient:
        client = _ScriptedClient(script)
        http_module.shared_client = lambda: client
        return client

    return install


@pytest.mark.tonio
async def test_retries_a_transient_transport_failure(stub_http):
    ok = _Response()
    client = stub_http([Exception("request failed"), Exception("request failed"), ok])

    response = await fetch_with_retry("https://example.test")

    assert response is ok
    assert client.calls == 3


@pytest.mark.tonio
async def test_gives_up_after_the_retry_budget(stub_http):
    client = stub_http([Exception("request failed")] * 3)

    with pytest.raises(Exception, match="request failed"):
        await fetch_with_retry("https://example.test", max_retries=2)

    assert client.calls == 3


@pytest.mark.tonio
async def test_retries_an_attempt_timeout(stub_http):
    from pidrei_ai.utils.http import RequestTimeout

    ok = _Response()
    client = stub_http([RequestTimeout("Timed out"), ok])

    response = await fetch_with_retry("https://example.test", attempt_timeout_ms=4000)

    assert response is ok
    assert client.calls == 2
    # A hung attempt is abandoned, not charged against the next one: every
    # attempt gets the full per-attempt bound (pi creates a fresh
    # AbortSignal.timeout per attempt).
    assert [timeout.read for timeout in client.timeouts] == [4.0, 4.0]


@pytest.mark.tonio
async def test_retries_transient_http_responses_and_returns_the_successful_response(stub_http):
    busy = _Response(503)
    ok = _Response()
    client = stub_http([busy, ok])

    response = await fetch_with_retry("https://example.test")

    assert response is ok
    assert client.calls == 2
    assert busy.closed is True


@pytest.mark.tonio
async def test_does_not_retry_a_non_transient_http_response(stub_http):
    not_found = _Response(404)
    client = stub_http([not_found, _Response()])

    response = await fetch_with_retry("https://example.test")

    assert response is not_found
    assert client.calls == 1


@pytest.mark.tonio
async def test_shares_the_timeout_budget_across_attempts(stub_http):
    ok = _Response()
    client = stub_http([Exception("request failed"), ok])

    response = await fetch_with_retry("https://example.test", timeout_ms=1000)

    assert response is ok
    assert client.calls == 2
    # Each attempt is bounded by what is left of the one budget, never by a
    # fresh 1000ms (pi shares a single AbortSignal.timeout instead).
    assert client.timeouts[1].read < client.timeouts[0].read <= 1.0


@pytest.mark.tonio
async def test_stops_retrying_once_the_budget_is_exhausted(stub_http):
    from pidrei_ai.utils.http import RequestTimeout

    client = stub_http([Exception("request failed")] * 3)

    with pytest.raises((RequestTimeout, Exception), match="request failed|Timed out"):
        await fetch_with_retry("https://example.test", timeout_ms=0.001)

    assert client.calls <= 1
