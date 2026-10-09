"""PiDrei-only tests for the Anthropic federation token cache.

pi gets the cache from `@anthropic-ai/sdk` (`TokenCache`) and has no tests of
its own for it; here it is PiDrei code running on parallel workers, so its
policy and its coalescing are tested directly. Exchanges go through the
`oauth_http` stub, gated on Events where a test needs one in flight, and time is
the virtual wall clock the cache reads through `clock.now_ms`. Hooks on the
cache (`_start`, `_finish`, `_next_step`) observe its decisions without timing.
"""

import json
from urllib.parse import urlsplit

import pytest
import tonio.colored as tonio
from tonio.colored import fs

from pidrei_ai.api.anthropic_messages import AnthropicOptions, stream
from pidrei_ai.auth.anthropic_federation import (
    AnthropicFederationConfig,
    AnthropicWorkloadIdentityError,
    federation_token_cache,
    reset_federation_token_cache,
)
from pidrei_ai.types import Context, Model, ModelCost, UserMessage
from pidrei_http import http
from tests.anthropic_helpers import now_ms
from tests.oauth_helpers import OAuthRequest, json_response, stub_oauth_http, virtual_clock


BASE_URL = "https://api.anthropic.com"


@pytest.fixture
async def config(tmp_path):
    identity_token_file = str(tmp_path / "identity.jwt")
    await fs.Path(identity_token_file).write_text("header.payload.signature\n")
    try:
        yield AnthropicFederationConfig(
            federation_rule_id="fdrl_test",
            organization_id="org-test",
            identity_token_file=identity_token_file,
        )
    finally:
        reset_federation_token_cache()


def token_response(request_count: int):
    return json_response({"access_token": f"token-{request_count}", "expires_in": 3600})


def count_starts(cache) -> list[bool]:
    """Every exchange the cache starts, as its `advisory` flag."""
    starts: list[bool] = []
    original = cache._start

    def recording(*, advisory: bool):
        starts.append(advisory)
        return original(advisory=advisory)

    cache._start = recording
    return starts


def finish_events(cache, count: int) -> list[tonio.Event]:
    """One Event per exchange the cache finishes, set in finishing order."""
    events = [tonio.Event() for _ in range(count)]
    finished: list[object] = []
    original = cache._finish

    def recording(refresh, token, error):
        original(refresh, token, error)
        finished.append(refresh)
        events[len(finished) - 1].set()

    cache._finish = recording
    return events


def decisions_event(cache, count: int) -> tonio.Event:
    """Set once `count` callers have made their `get_token` decision."""
    decided = tonio.Event()
    seen: list[object] = []
    original = cache._next_step

    def recording():
        step = original()
        seen.append(step)
        if len(seen) >= count:
            decided.set()
        return step

    cache._next_step = recording
    return decided


@pytest.mark.tonio
async def test_reads_the_identity_token_and_exchanges_it_with_the_federation_betas(config):
    with virtual_clock(), stub_oauth_http(lambda request: token_response(1)) as calls:
        assert await federation_token_cache(f"{BASE_URL}/", config).get_token() == "token-1"

    assert calls[0].url == "https://api.anthropic.com/v1/oauth/token"
    assert calls[0].headers["anthropic-beta"] == "oauth-2025-04-20,oidc-federation-2026-04-01"
    assert calls[0].json_body == {
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "assertion": "header.payload.signature",
        "federation_rule_id": "fdrl_test",
        "organization_id": "org-test",
    }
    assert calls[0].timeout_ms == 30_000


@pytest.mark.tonio
async def test_concurrent_callers_share_one_exchange(config):
    release = tonio.Event()

    async def handler(request: OAuthRequest):
        await release.wait(5)
        return token_response(len(calls))

    with virtual_clock(), stub_oauth_http(handler) as calls:
        cache = federation_token_cache(BASE_URL, config)
        decided = decisions_event(cache, 2)
        first = tonio.spawn(cache.get_token())
        second = tonio.spawn(cache.get_token())
        await decided.wait(5)
        assert decided.is_set()
        release.set()

        assert await first == "token-1"
        assert await second == "token-1"
    assert len(calls) == 1


@pytest.mark.tonio
async def test_serves_the_cached_token_in_the_advisory_window_and_refreshes_in_the_background(config):
    with virtual_clock() as now, stub_oauth_http(lambda request: token_response(len(calls))) as calls:
        cache = federation_token_cache(BASE_URL, config)
        assert await cache.get_token() == "token-1"
        finished = finish_events(cache, 1)

        now["now"] += (3600 - 60) * 1000
        assert await cache.get_token() == "token-1"
        await finished[0].wait(5)
        assert finished[0].is_set()

        assert await cache.get_token() == "token-2"
    assert len(calls) == 2


@pytest.mark.tonio
async def test_backs_off_after_a_failed_background_refresh(config):
    def handler(request: OAuthRequest):
        if len(calls) == 1:
            return token_response(1)
        return json_response({"error": "server_error"}, status=500)

    with virtual_clock() as now, stub_oauth_http(handler) as calls:
        cache = federation_token_cache(BASE_URL, config)
        assert await cache.get_token() == "token-1"
        starts = count_starts(cache)
        finished = finish_events(cache, 2)

        now["now"] += (3600 - 60) * 1000
        assert await cache.get_token() == "token-1"
        await finished[0].wait(5)
        assert finished[0].is_set()
        assert starts == [True]

        # Inside the backoff the stale token is served with no new exchange.
        assert await cache.get_token() == "token-1"
        assert starts == [True]

        now["now"] += 5 * 1000
        assert await cache.get_token() == "token-1"
        assert starts == [True, True]
        await finished[1].wait(5)
        assert finished[1].is_set()


@pytest.mark.tonio
async def test_blocks_on_an_expiring_token_and_raises_the_redacted_exchange_failure(config):
    def handler(request: OAuthRequest):
        if len(calls) == 1:
            return token_response(1)
        return json_response(
            {"error": "invalid_grant", "error_description": "no matching rule", "assertion": "header.payload"},
            status=401,
        )

    with virtual_clock() as now, stub_oauth_http(handler) as calls:
        cache = federation_token_cache(BASE_URL, config)
        assert await cache.get_token() == "token-1"

        now["now"] += (3600 - 10) * 1000
        with pytest.raises(AnthropicWorkloadIdentityError) as raised:
            await cache.get_token()

    assert raised.value.status_code == 401
    assert str(raised.value) == (
        'Token exchange failed with status 401: {"error":"invalid_grant","error_description":"no matching rule"} '
        "Ensure your federation rule matches your identity token. If your federation rule is scoped to multiple "
        "workspaces, set the ANTHROPIC_WORKSPACE_ID environment variable, the 'workspace_id' config key, or the "
        "`workspaceId` option. View your authentication events in the Workload identity page of Claude Console "
        "for more details."
    )


@pytest.mark.tonio
async def test_invalidate_starts_a_fresh_exchange_that_a_slower_older_one_cannot_overwrite(config):
    second_entered = tonio.Event()
    release_second = tonio.Event()

    async def handler(request: OAuthRequest):
        count = len(calls)
        if count == 2:
            second_entered.set()
            await release_second.wait(5)
        return token_response(count)

    with virtual_clock() as now, stub_oauth_http(handler) as calls:
        cache = federation_token_cache(BASE_URL, config)
        assert await cache.get_token() == "token-1"
        starts = count_starts(cache)
        finished = finish_events(cache, 2)

        # A background refresh is in flight when the token is invalidated.
        now["now"] += (3600 - 60) * 1000
        assert await cache.get_token() == "token-1"
        assert starts == [True]
        # Exchanges are numbered by arrival: the forced one must arrive third.
        await second_entered.wait(5)
        assert second_entered.is_set()

        cache.invalidate()
        # The forced caller does not join the background refresh.
        assert await cache.get_token() == "token-3"
        assert starts == [True, False]

        release_second.set()
        await finished[1].wait(5)
        assert finished[1].is_set()
        assert await cache.get_token() == "token-3"


@pytest.mark.tonio
async def test_a_cancelled_caller_leaves_the_exchange_running_for_the_others(config):
    entered = tonio.Event()
    release = tonio.Event()

    async def handler(request: OAuthRequest):
        entered.set()
        await release.wait(5)
        return token_response(len(calls))

    with virtual_clock(), stub_oauth_http(handler) as calls:
        cache = federation_token_cache(BASE_URL, config)
        async with tonio.scope() as cancelled:
            cancelled.spawn(cache.get_token())
            await entered.wait(5)
            assert entered.is_set()
            cancelled.cancel()

        decided = decisions_event(cache, 1)
        joined = tonio.spawn(cache.get_token())
        await decided.wait(5)
        assert decided.is_set()
        release.set()

        assert await joined == "token-1"
    assert len(calls) == 1


class _Response:
    def __init__(self, status: int, body: bytes, content_type: str):
        self.status_code = status
        self.headers = {"content-type": content_type}
        self._body = body

    async def iter_bytes(self):
        yield self._body

    async def read(self) -> bytes:
        return self._body

    async def close(self) -> None:
        pass


def _sse_body() -> bytes:
    events = [
        {"type": "message_start", "message": {"id": "msg_test", "usage": {"input_tokens": 1, "output_tokens": 0}}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


@pytest.mark.tonio
async def test_a_401_fails_the_request_and_the_next_request_exchanges_again(config):
    exchanges: list[str] = []
    authorizations: list[str | None] = []

    class FakeClient:
        async def send(self, request):
            path = urlsplit(str(request.url)).path
            if path == "/v1/oauth/token":
                exchanges.append(path)
                body = f'{{"access_token": "token-{len(exchanges)}", "expires_in": 3600}}'.encode()
                return _Response(200, body, "application/json")
            authorizations.append(request.headers.get("authorization"))
            if len(authorizations) == 1:
                body = b'{"type":"error","error":{"type":"authentication_error","message":"invalid token"}}'
                return _Response(401, body, "application/json")
            return _Response(200, _sse_body(), "text/event-stream")

    model = Model(
        id="claude-test",
        name="Claude Test",
        api="anthropic-messages",
        provider="anthropic",
        base_url=BASE_URL,
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=100000,
        max_tokens=4096,
    )
    env = {
        "ANTHROPIC_FEDERATION_RULE_ID": config.federation_rule_id,
        "ANTHROPIC_ORGANIZATION_ID": config.organization_id,
        "ANTHROPIC_IDENTITY_TOKEN_FILE": config.identity_token_file,
    }
    context = Context(messages=[UserMessage(content="Hello", timestamp=now_ms())])

    original_client_for = http.client_for
    http.client_for = lambda _url, _env=None: FakeClient()
    try:
        failed = await stream(model, context, AnthropicOptions(env=env)).result()
        succeeded = await stream(model, context, AnthropicOptions(env=env)).result()
    finally:
        http.client_for = original_client_for

    assert failed.stop_reason == "error"
    assert failed.error_message == "401 invalid token"
    assert succeeded.stop_reason == "stop"
    assert exchanges == ["/v1/oauth/token", "/v1/oauth/token"]
    assert authorizations == ["Bearer token-1", "Bearer token-2"]


# pi keys its federation client on `(config, fetch)` by identity
# (anthropic-messages.ts: `federationClient.fetch !== fetch`), and the SDK
# runs the exchange through that fetch.


@pytest.mark.tonio
async def test_runs_the_exchange_through_the_requests_fetch_and_keys_the_cache_on_its_identity(config):
    async def custom_fetch(request, *, env=None):
        raise AssertionError("the stubbed seam answers before the fetch is reached")

    async def other_fetch(request, *, env=None):
        raise AssertionError("the stubbed seam answers before the fetch is reached")

    # The stub records a call before answering it, so `len(calls)` numbers this exchange.
    with virtual_clock(), stub_oauth_http(lambda request: token_response(len(calls))) as calls:
        first = federation_token_cache(BASE_URL, config, custom_fetch)
        assert await first.get_token() == "token-1"
        assert calls[0].fetch is custom_fetch

        # The same fetch joins the same cache: the token is served, no exchange.
        assert federation_token_cache(BASE_URL, config, custom_fetch) is first
        assert await first.get_token() == "token-1"
        assert len(calls) == 1

        # A different fetch replaces the cache, and its exchange runs through it.
        second = federation_token_cache(BASE_URL, config, other_fetch)
        assert second is not first
        assert await second.get_token() == "token-2"
        assert calls[1].fetch is other_fetch
