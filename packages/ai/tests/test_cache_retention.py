"""Mirror of pi's cache-retention.test.ts (PIDREI_CACHE_RETENTION).

pi's API-key-gated variants assert the same payloads as the fake-key ones
(`onPayload` fires before any request), so everything mirrors as unit tests.
Cases already pinned by the adapter suites (responses explicit-mode/reject,
long-retention key+24h) are not duplicated here.
"""

import os
import time
from contextlib import contextmanager
from dataclasses import replace

import pytest

from pidrei_ai.api.anthropic_messages import AnthropicOptions, stream as stream_anthropic
from pidrei_ai.api.openai_completions import (
    OpenAICompletionsOptions,
    build_params as build_completions_params,
)
from pidrei_ai.api.openai_responses import (
    OpenAIResponsesOptions,
    build_params as build_responses_params,
)
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import (
    AnthropicMessagesCompat,
    Context,
    JsonSchemaConstrainedSampling,
    OpenAICompletionsCompat,
    OpenAIResponsesCompat,
    SystemMessage,
    Tool,
    TranscriptContext,
    UserMessage,
)
from pidrei_ai.utils.transcript import normalize_context
from tests.test_openai_completions import make_model as make_completions_model
from tests.test_openai_responses import make_model as make_responses_model


class PayloadCaptured(Exception):
    pass


@contextmanager
def cache_retention_env(value: str | None):
    """In-test env handling via context manager (predates tonio 0.9.14, which
    made yield fixtures like `monkeypatch` usable in tonio-marked tests)."""
    original = os.environ.get("PIDREI_CACHE_RETENTION")
    if value is None:
        os.environ.pop("PIDREI_CACHE_RETENTION", None)
    else:
        os.environ["PIDREI_CACHE_RETENTION"] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("PIDREI_CACHE_RETENTION", None)
        else:
            os.environ["PIDREI_CACHE_RETENTION"] = original


def make_context() -> TranscriptContext:
    return normalize_context(
        Context(
            system_prompt="You are a helpful assistant.",
            messages=[UserMessage(content="Hello", timestamp=int(time.time() * 1000))],
        )
    )


async def capture_anthropic_payload(model, options: AnthropicOptions | None = None) -> dict:
    captured: list[dict] = []

    async def on_payload(payload, _model):
        captured.append(payload)
        raise PayloadCaptured()

    opts = replace(options if options is not None else AnthropicOptions(), api_key="fake-key", on_payload=on_payload)
    await stream_anthropic(model, make_context(), opts).result()
    assert captured
    return captured[0]


def haiku():
    model = get_builtin_model("anthropic", "claude-haiku-4-5")
    assert model is not None
    return model


# --- Anthropic ---------------------------------------------------------------


@pytest.mark.tonio
async def test_anthropic_default_cache_ttl_when_env_not_set():
    with cache_retention_env(None):
        payload = await capture_anthropic_payload(haiku())

    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.tonio
async def test_anthropic_1h_ttl_when_env_long():
    with cache_retention_env("long"):
        payload = await capture_anthropic_payload(haiku())

    assert payload["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


@pytest.mark.tonio
async def test_anthropic_adds_ttl_for_proxy_base_url_by_default():
    with cache_retention_env("long"):
        proxy_model = replace(haiku(), base_url="https://my-proxy.example.com/v1")
        payload = await capture_anthropic_payload(proxy_model)

    assert payload["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


@pytest.mark.tonio
async def test_anthropic_omits_ttl_when_supports_long_cache_retention_is_false():
    with cache_retention_env(None):
        proxy_model = replace(
            haiku(),
            base_url="https://my-proxy.example.com/v1",
            compat=AnthropicMessagesCompat(supports_long_cache_retention=False),
        )
        payload = await capture_anthropic_payload(proxy_model, AnthropicOptions(cache_retention="long"))

    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.tonio
async def test_anthropic_omits_cache_control_when_retention_is_none():
    with cache_retention_env(None):
        payload = await capture_anthropic_payload(haiku(), AnthropicOptions(cache_retention="none"))

    assert "cache_control" not in payload["system"][0]


@pytest.mark.tonio
async def test_anthropic_adds_cache_control_to_string_user_messages():
    with cache_retention_env(None):
        payload = await capture_anthropic_payload(haiku())

    last_message = payload["messages"][-1]
    assert isinstance(last_message["content"], list)
    last_block = last_message["content"][-1]
    assert last_block["cache_control"] == {"type": "ephemeral"}


@pytest.mark.tonio
async def test_anthropic_1h_ttl_when_cache_retention_option_is_long():
    with cache_retention_env(None):
        payload = await capture_anthropic_payload(haiku(), AnthropicOptions(cache_retention="long"))

    assert payload["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


# --- OpenAI Responses ---------------------------------------------------------


@pytest.mark.parametrize(
    "model_id", ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "gpt-6-luna", "gpt-6-sol"]
)
def test_responses_does_not_enable_cache_warming_from_the_documented_ttl_alone(model_id):
    assert get_builtin_model("openai", model_id).prompt_cache is None


def test_responses_no_prompt_cache_retention_by_default(monkeypatch):
    monkeypatch.delenv("PIDREI_CACHE_RETENTION", raising=False)
    params = build_responses_params(make_responses_model(), make_context(), OpenAIResponsesOptions())

    assert "prompt_cache_retention" not in params


def test_responses_prompt_cache_retention_when_env_long(monkeypatch):
    monkeypatch.setenv("PIDREI_CACHE_RETENTION", "long")
    params = build_responses_params(make_responses_model(), make_context(), OpenAIResponsesOptions())

    assert params["prompt_cache_retention"] == "24h"


@pytest.mark.parametrize(
    ("model_id", "retention", "cache_options"),
    [
        ("gpt-4o-mini", "24h", None),
        ("gpt-6-astra", None, {"ttl": "30m"}),
        ("gpt-6-sol", None, {"ttl": "30m"}),
        ("gpt-6-luna", None, {"ttl": "30m"}),
    ],
)
def test_responses_uses_the_supported_long_cache_field(monkeypatch, model_id, retention, cache_options):
    monkeypatch.delenv("PIDREI_CACHE_RETENTION", raising=False)
    model = get_builtin_model("openai", model_id)
    assert model is not None
    params = build_responses_params(
        model, make_context(), OpenAIResponsesOptions(cache_retention="long", session_id="session-2")
    )

    assert params["prompt_cache_key"] == "session-2"
    assert params.get("prompt_cache_retention") == retention
    assert params.get("prompt_cache_options") == cache_options


def test_responses_prompt_cache_retention_for_proxy_base_url(monkeypatch):
    monkeypatch.delenv("PIDREI_CACHE_RETENTION", raising=False)
    model = make_responses_model(base_url="https://my-proxy.example.com/v1")
    params = build_responses_params(model, make_context(), OpenAIResponsesOptions(cache_retention="long"))

    assert params["prompt_cache_retention"] == "24h"


def test_responses_omits_retention_when_supports_long_cache_retention_is_false(monkeypatch):
    monkeypatch.delenv("PIDREI_CACHE_RETENTION", raising=False)
    model = make_responses_model(compat=OpenAIResponsesCompat(supports_long_cache_retention=False))
    params = build_responses_params(model, make_context(), OpenAIResponsesOptions(cache_retention="long"))

    assert "prompt_cache_retention" not in params


# --- OpenAI Completions -------------------------------------------------------


def test_completions_prompt_cache_retention_for_proxy_base_url(monkeypatch):
    monkeypatch.delenv("PIDREI_CACHE_RETENTION", raising=False)
    model = make_completions_model(provider="custom", base_url="https://my-proxy.example.com/v1")
    params = build_completions_params(
        model, make_context(), OpenAICompletionsOptions(cache_retention="long", session_id="session-completions")
    )

    assert params["prompt_cache_key"] == "session-completions"
    assert params["prompt_cache_retention"] == "24h"


def test_completions_omits_retention_when_supports_long_cache_retention_is_false(monkeypatch):
    monkeypatch.delenv("PIDREI_CACHE_RETENTION", raising=False)
    model = make_completions_model(
        provider="custom",
        base_url="https://my-proxy.example.com/v1",
        compat=OpenAICompletionsCompat(supports_long_cache_retention=False),
    )
    params = build_completions_params(
        model, make_context(), OpenAICompletionsOptions(cache_retention="long", session_id="session-completions")
    )

    assert "prompt_cache_retention" not in params
    assert "prompt_cache_key" not in params


def test_completions_env_long_applies_retention(monkeypatch):
    monkeypatch.setenv("PIDREI_CACHE_RETENTION", "long")
    params = build_completions_params(
        make_completions_model(), make_context(), OpenAICompletionsOptions(session_id="sess")
    )

    assert params["prompt_cache_retention"] == "24h"
    assert params["prompt_cache_key"] == "sess"


@pytest.mark.parametrize("model_id", ["gpt-oss-120b", "qwen-3.8-27b"])
def test_completions_omits_strict_field_on_tools_for_cerebras(model_id):
    # pi passes `{type: "json_schema"}` through an `as any`; "prefer" is the valid
    # shape of the same request.
    model = get_builtin_model("cerebras", model_id)
    assert model is not None
    context = TranscriptContext(
        messages=[
            SystemMessage(
                content="test",
                tools_added=[
                    Tool(
                        name="t1",
                        description="strict tool",
                        parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
                        constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"),
                    ),
                    Tool(
                        name="t2",
                        description="non-strict tool",
                        parameters={"type": "object", "properties": {"y": {"type": "string"}}, "required": ["y"]},
                    ),
                ],
                timestamp=0,
            ),
            UserMessage(content="hello", timestamp=1),
        ]
    )

    params = build_completions_params(model, context, OpenAICompletionsOptions(api_key="fake-key", session_id="test"))

    assert model.compat is not None
    assert model.compat.supports_strict_mode is None
    assert params["tools"]
    for tool in params["tools"]:
        assert "strict" not in tool["function"]
