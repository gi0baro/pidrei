"""Mirror of pi's anthropic-federation-sdk.test.ts.

pi runs the real Anthropic SDK against a fake `fetch` to count token exchanges.
Here the real adapter and token cache run against a fake `fetch` passed as the
request option, which both the messages request and the token exchange
(`auth/oauth/http.py`) go through — one fake for both, as pi's single
`fetch`. pi's second case checks that the SDK's own credential chain
stays off; there is no SDK chain here, so it checks that header-owned auth
starts no exchange even with the federation variables in the process env.
"""

import json
from urllib.parse import urlsplit

import pytest
from tonio.colored import fs

from pidrei_ai.api.anthropic_messages import AnthropicOptions, stream
from pidrei_ai.auth.anthropic_federation import reset_federation_token_cache
from pidrei_ai.types import Context, Model, ModelCost, UserMessage
from tests.anthropic_helpers import now_ms
from tests.oauth_helpers import process_env


# https://github.com/earendil-works/pi/issues/10177


def _sse_body() -> bytes:
    events = [
        {"type": "message_start", "message": {"id": "msg_test", "usage": {"input_tokens": 1, "output_tokens": 0}}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


class _Response:
    def __init__(self, body: bytes, content_type: str):
        self.status_code = 200
        self.headers = {"content-type": content_type}
        self._body = body

    async def iter_bytes(self):
        yield self._body

    async def read(self) -> bytes:
        return self._body

    async def close(self) -> None:
        pass


class _FakeFetch:
    """pi's `createFetch`: records (path, authorization) and answers both endpoints."""

    def __init__(self):
        self.requests: list[tuple[str, str | None]] = []

    async def __call__(self, request, *, env=None) -> _Response:
        path = urlsplit(str(request.url)).path
        authorization = next(
            (value for name, value in request.headers.items() if name.lower() == "authorization"), None
        )
        self.requests.append((path, authorization))
        if path == "/v1/oauth/token":
            body = json.dumps({"access_token": "federated-token", "expires_in": 3600}).encode()
            return _Response(body, "application/json")
        return _Response(_sse_body(), "text/event-stream")


def make_context() -> Context:
    return Context(system_prompt="System prompt.", messages=[UserMessage(content="Hello", timestamp=now_ms())])


def make_model() -> Model:
    return Model(
        id="claude-test",
        name="Claude Test",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=100000,
        max_tokens=4096,
    )


@pytest.fixture
async def federation_env(tmp_path):
    identity_token_file = str(tmp_path / "identity.jwt")
    await fs.Path(identity_token_file).write_text("header.payload.signature")
    return {
        "ANTHROPIC_FEDERATION_RULE_ID": "fdrl_test",
        "ANTHROPIC_ORGANIZATION_ID": "org-test",
        "ANTHROPIC_IDENTITY_TOKEN_FILE": identity_token_file,
    }


@pytest.fixture
def fake_fetch():
    try:
        yield _FakeFetch()
    finally:
        reset_federation_token_cache()


@pytest.mark.tonio
async def test_exchanges_the_identity_token_once_across_requests(federation_env, fake_fetch):
    for _ in range(3):
        message = await stream(
            make_model(), make_context(), AnthropicOptions(env=federation_env, fetch=fake_fetch)
        ).result()
        assert message.stop_reason == "stop"

    assert len([request for request in fake_fetch.requests if request[0] == "/v1/oauth/token"]) == 1
    message_requests = [request for request in fake_fetch.requests if request[0] == "/v1/messages"]
    assert len(message_requests) == 3
    for _path, authorization in message_requests:
        assert authorization == "Bearer federated-token"


@pytest.mark.tonio
async def test_does_not_exchange_for_header_owned_auth(federation_env, fake_fetch):
    with process_env(**federation_env):
        message = await stream(
            make_model(),
            make_context(),
            AnthropicOptions(headers={"Authorization": "Bearer auth-token"}, fetch=fake_fetch),
        ).result()

    assert message.stop_reason == "stop"
    assert fake_fetch.requests == [("/v1/messages", "Bearer auth-token")]
