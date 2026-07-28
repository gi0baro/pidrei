"""Mirror of pi's anthropic-auth-token.test.ts.

pi mocks the Anthropic SDK constructor and asserts on its options
(`apiKey`/`authToken` null, `defaultHeaders`); here the same observable
request is inspected on the transport `_create_client` builds — no
`x-api-key` header stands in for "constructor got a null apiKey", the
`anthropic-beta` header carries the OAuth-shaping check, and the payload
comes from `on_payload` capture.
"""

import pytest

from pidrei_ai.api.anthropic_messages import AnthropicOptions
from pidrei_ai.auth.types import AuthResult, ModelAuth
from pidrei_ai.env_api_keys import ANTHROPIC_AUTH_TOKEN_ENV, ANTHROPIC_OAUTH_TOKEN_ENV
from pidrei_ai.providers.anthropic import anthropic_provider
from pidrei_ai.registry import create_models
from pidrei_ai.types import Context, Model, ModelCost, SimpleStreamOptions, UserMessage
from tests.anthropic_helpers import (
    PayloadCaptured,
    _recording_transport,
    capture_request,
    now_ms,
)


class _EnvContext:
    """pi's inline `ctx` literals: env lookup plus a no-op fileExists."""

    def __init__(self, env: dict[str, str]):
        self._env = env

    async def env(self, name: str) -> str | None:
        return self._env.get(name)

    async def file_exists(self, path: str) -> bool:
        return False


def make_context() -> Context:
    return Context(
        system_prompt="System prompt.",
        messages=[UserMessage(content="Hello", timestamp=now_ms())],
    )


def make_model() -> Model:
    return Model(
        id="claude-test",
        name="Claude Test",
        api="anthropic-messages",
        provider="anthropic",
        base_url="http://127.0.0.1:9",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=100000,
        max_tokens=4096,
    )


async def _capture_models_request(env: dict[str, str], options: SimpleStreamOptions) -> tuple[dict[str, str], dict]:
    """The (headers, payload) the models registry path puts on the wire.

    pi drives these cases through `createModels({authContext}).streamSimple`;
    the transport recording plus `on_payload` capture mirror the SDK mock.
    """
    captured: list[dict] = []

    async def on_payload(payload, _model):
        captured.append(payload)
        raise PayloadCaptured()

    models = create_models(auth_context=_EnvContext(env))
    models.set_provider(anthropic_provider())
    options.on_payload = on_payload

    with _recording_transport() as recorded:
        await models.stream_simple(make_model(), make_context(), options).result()

    assert recorded, "Expected the adapter to build a transport"
    assert captured, "Expected payload to be captured before request failure"
    return recorded[0], captured[0]


@pytest.mark.tonio
async def test_resolves_anthropic_auth_token_as_a_bearer_authorization_header():
    provider = anthropic_provider()
    auth = await provider.auth.api_key.resolve(
        _EnvContext(
            {
                "ANTHROPIC_AUTH_TOKEN": "auth-token",
                "ANTHROPIC_OAUTH_TOKEN": "oauth-token",
                "ANTHROPIC_API_KEY": "api-key",
            }
        ),
        None,
    )

    assert auth == AuthResult(
        auth=ModelAuth(headers={"Authorization": "Bearer auth-token"}),
        source=ANTHROPIC_AUTH_TOKEN_ENV,
    )


@pytest.mark.tonio
async def test_preserves_anthropic_oauth_token_as_oauth_shaped_api_auth():
    provider = anthropic_provider()
    auth = await provider.auth.api_key.resolve(
        _EnvContext(
            {
                "ANTHROPIC_OAUTH_TOKEN": "oauth-token",
                "ANTHROPIC_API_KEY": "api-key",
            }
        ),
        None,
    )

    assert auth == AuthResult(auth=ModelAuth(api_key="oauth-token"), source=ANTHROPIC_OAUTH_TOKEN_ENV)


@pytest.mark.tonio
async def test_uses_authorization_headers_without_oauth_mode_request_shaping():
    headers, payload = await capture_request(
        make_model(),
        AnthropicOptions(headers={"Authorization": "Bearer gateway-token"}),
        make_context(),
        default_api_key=None,
    )

    assert headers["Authorization"] == "Bearer gateway-token"
    assert "x-api-key" not in headers
    assert "oauth-2025-04-20" not in headers.get("anthropic-beta", "")
    assert payload["system"][0]["text"] == "System prompt."


@pytest.mark.tonio
async def test_threads_auth_context_anthropic_auth_token_through_request_headers():
    headers, payload = await _capture_models_request({"ANTHROPIC_AUTH_TOKEN": "ctx-token"}, SimpleStreamOptions())

    assert headers["Authorization"] == "Bearer ctx-token"
    assert "x-api-key" not in headers
    assert "oauth-2025-04-20" not in headers.get("anthropic-beta", "")
    assert payload["system"][0]["text"] == "System prompt."


@pytest.mark.tonio
async def test_preserves_oauth_request_shaping_for_anthropic_oauth_token():
    headers, _payload = await _capture_models_request(
        {"ANTHROPIC_OAUTH_TOKEN": "sk-ant-oat-test"}, SimpleStreamOptions()
    )

    assert headers["authorization"] == "Bearer sk-ant-oat-test"
    assert "x-api-key" not in headers
    assert "oauth-2025-04-20" in headers.get("anthropic-beta", "")


@pytest.mark.tonio
async def test_lets_explicit_request_headers_override_anthropic_auth_token():
    headers, _payload = await _capture_models_request(
        {"ANTHROPIC_AUTH_TOKEN": "ctx-token"},
        SimpleStreamOptions(headers={"Authorization": "Bearer explicit-token"}),
    )

    assert headers["Authorization"] == "Bearer explicit-token"
