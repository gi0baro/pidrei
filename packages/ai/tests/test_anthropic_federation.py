"""Mirror of pi's anthropic-federation.test.ts.

pi mocks the Anthropic SDK constructor and asserts on its options (`apiKey`,
`authToken`, the federation `config`, `defaultHeaders`). Here the same request
is inspected on the transport `_create_client` builds: its headers, and the
`AnthropicFederationConfig` it was given in place of the SDK's `config`. The
payload comes from `on_payload` capture, which stops the request before the
transport asks for a token, so no exchange runs.
"""

import contextlib
from dataclasses import replace

import pytest

from pidrei_ai.api import anthropic_messages
from pidrei_ai.api.anthropic_messages import AnthropicOptions, stream
from pidrei_ai.auth.anthropic_federation import AnthropicFederationConfig, reset_federation_token_cache
from pidrei_ai.auth.types import AuthResult, ModelAuth
from pidrei_ai.providers.anthropic import anthropic_provider
from pidrei_ai.registry import create_models
from pidrei_ai.types import Context, Model, ModelCost, SimpleStreamOptions, UserMessage
from pidrei_utils.cancel import CancelToken
from tests.anthropic_helpers import PayloadCaptured, now_ms


_never_aborted_cancel = CancelToken()

FEDERATION_ENV = {
    "ANTHROPIC_FEDERATION_RULE_ID": "fdrl_test",
    "ANTHROPIC_ORGANIZATION_ID": "org-test",
    "ANTHROPIC_SERVICE_ACCOUNT_ID": "svac_test",
    "ANTHROPIC_IDENTITY_TOKEN_FILE": "/tmp/identity.jwt",
}

EXPECTED_CONFIG = AnthropicFederationConfig(
    federation_rule_id="fdrl_test",
    organization_id="org-test",
    identity_token_file="/tmp/identity.jwt",
    service_account_id="svac_test",
    workspace_id=None,
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
        base_url="https://api.anthropic.com",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=100000,
        max_tokens=4096,
    )


@contextlib.contextmanager
def _recording_transport():
    """Record what each transport is built with: (headers, federation config)."""
    recorded: list[tuple[dict[str, str], AnthropicFederationConfig | None]] = []
    original = anthropic_messages._PunkreqAnthropicClient

    class RecordingClient(original):
        def __init__(self, base_url, headers, env=None, federation=None, fetch=None):
            recorded.append((dict(headers), federation))
            super().__init__(base_url, headers, env, federation=federation, fetch=fetch)

    anthropic_messages._PunkreqAnthropicClient = RecordingClient
    try:
        yield recorded
    finally:
        anthropic_messages._PunkreqAnthropicClient = original
        reset_federation_token_cache()


def _capture_payload_into(captured: list[dict]):
    async def on_payload(payload, _model):
        captured.append(payload)
        raise PayloadCaptured()

    return on_payload


async def _stream_and_record(model: Model, options: AnthropicOptions):
    captured: list[dict] = []
    options.on_payload = _capture_payload_into(captured)
    with _recording_transport() as recorded:
        message = await stream(model, make_context(), options).result()
    return recorded, captured, message


def resolve_with_env(env: dict[str, str]):
    return anthropic_provider().auth.api_key.resolve(_EnvContext(env), None, _never_aborted_cancel)


# https://github.com/earendil-works/pi/issues/10177


@pytest.mark.tonio
async def test_resolves_the_federation_variables_as_provider_env_with_no_request_auth():
    assert await resolve_with_env(FEDERATION_ENV) == AuthResult(
        auth=ModelAuth(),
        env=FEDERATION_ENV,
        source="workload identity federation",
    )


@pytest.mark.tonio
async def test_passes_anthropic_workspace_id_through_when_set():
    result = await resolve_with_env({**FEDERATION_ENV, "ANTHROPIC_WORKSPACE_ID": "wrkspc_test"})
    assert result is not None and result.env is not None
    assert result.env["ANTHROPIC_WORKSPACE_ID"] == "wrkspc_test"


@pytest.mark.tonio
async def test_is_not_configured_when_a_federation_variable_is_missing():
    partial = {key: value for key, value in FEDERATION_ENV.items() if key != "ANTHROPIC_IDENTITY_TOKEN_FILE"}
    assert await resolve_with_env(partial) is None


@pytest.mark.tonio
async def test_treats_anthropic_service_account_id_as_optional_like_the_sdk():
    partial = {key: value for key, value in FEDERATION_ENV.items() if key != "ANTHROPIC_SERVICE_ACCOUNT_ID"}
    assert await resolve_with_env(partial) == AuthResult(
        auth=ModelAuth(),
        env=partial,
        source="workload identity federation",
    )

    recorded, _captured, _message = await _stream_and_record(make_model(), AnthropicOptions(env=partial))
    assert recorded[0][1] == replace(EXPECTED_CONFIG, service_account_id=None)


@pytest.mark.tonio
async def test_keeps_api_key_and_auth_token_precedence_over_federation():
    assert await resolve_with_env({**FEDERATION_ENV, "ANTHROPIC_API_KEY": "api-key"}) == AuthResult(
        auth=ModelAuth(api_key="api-key"),
        source="ANTHROPIC_API_KEY",
    )
    assert await resolve_with_env({**FEDERATION_ENV, "ANTHROPIC_AUTH_TOKEN": "auth-token"}) == AuthResult(
        auth=ModelAuth(headers={"Authorization": "Bearer auth-token"}),
        source="ANTHROPIC_AUTH_TOKEN",
    )


@pytest.mark.tonio
async def test_hands_the_transport_a_federation_config_instead_of_a_key():
    recorded, captured, _message = await _stream_and_record(make_model(), AnthropicOptions(env=FEDERATION_ENV))

    headers, federation = recorded[0]
    assert "x-api-key" not in headers
    assert federation == EXPECTED_CONFIG
    assert not any(name.lower() == "authorization" for name in headers)
    assert "oauth-2025-04-20" not in captured[0].get("betas", [])


@pytest.mark.tonio
async def test_threads_auth_context_federation_variables_through_models():
    models = create_models(auth_context=_EnvContext(FEDERATION_ENV))
    models.set_provider(anthropic_provider())
    captured: list[dict] = []

    with _recording_transport() as recorded:
        await models.stream_simple(
            make_model(), make_context(), SimpleStreamOptions(on_payload=_capture_payload_into(captured))
        ).result()

    headers, federation = recorded[0]
    assert "x-api-key" not in headers
    assert federation == EXPECTED_CONFIG


@pytest.mark.tonio
async def test_lets_an_explicit_api_key_win_over_federation_env():
    recorded, _captured, _message = await _stream_and_record(
        make_model(), AnthropicOptions(api_key="explicit-key", env=FEDERATION_ENV)
    )

    headers, federation = recorded[0]
    assert headers["x-api-key"] == "explicit-key"
    assert federation is None


@pytest.mark.tonio
async def test_does_not_federate_other_anthropic_messages_providers():
    kimi = replace(make_model(), provider="kimi-coding", base_url="https://api.kimi.com/coding")
    recorded, _captured, message = await _stream_and_record(kimi, AnthropicOptions(env=FEDERATION_ENV))

    assert message.stop_reason == "error"
    assert recorded == []
