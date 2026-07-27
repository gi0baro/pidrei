"""Mirror of pi's fireworks-models.test.ts.

The integration half asserts on the request the anthropic-messages adapter would
send; pi captures it with a local HTTP server, `capture_request` records the same
headers and payload from the adapter's own transport (see anthropic_helpers).
"""

import pytest

from pidrei_ai.env_api_keys import find_env_keys, get_env_api_key
from pidrei_ai.providers.all import get_builtin_model, get_builtin_models
from pidrei_ai.types import (
    AnthropicMessagesCompat,
    Context,
    Model,
    ModelCost,
    SimpleStreamOptions,
    Tool,
    UserMessage,
)
from tests.anthropic_helpers import capture_request, now_ms


KIMI_K2P6 = "accounts/fireworks/models/kimi-k2p6"

FIREWORKS_ANTHROPIC_COMPAT = AnthropicMessagesCompat(
    send_session_affinity_headers=True,
    supports_eager_tool_input_streaming=False,
    supports_cache_control_on_tools=False,
    supports_long_cache_retention=False,
)


# --- catalog ------------------------------------------------------------------


def test_registers_default_kimi_k26_via_anthropic_messages():
    model = get_builtin_model("fireworks", KIMI_K2P6)

    assert model is not None
    assert model.api == "anthropic-messages"
    assert model.provider == "fireworks"
    assert model.base_url == "https://api.fireworks.ai/inference"
    assert model.reasoning is True
    assert model.input == ["text", "image"]
    assert model.context_window == 262000
    assert model.max_tokens == 262000
    assert model.cost == ModelCost(input=0.95, output=4, cache_read=0.16, cache_write=0)


def test_registers_the_fire_pass_turbo_router_model():
    model = next(
        (
            candidate
            for candidate in get_builtin_models("fireworks")
            if candidate.id.startswith("accounts/fireworks/routers/") and candidate.id.endswith("-turbo")
        ),
        None,
    )

    assert model is not None
    assert model.api == "anthropic-messages"
    assert model.base_url == "https://api.fireworks.ai/inference"
    assert model.input == ["text", "image"]


def test_aligns_glm_52_fast_with_glm_52_openai_compatible_config():
    base = get_builtin_model("fireworks", "accounts/fireworks/models/glm-5p2")
    fast = get_builtin_model("fireworks", "accounts/fireworks/routers/glm-5p2-fast")

    assert base is not None and fast is not None
    assert fast.api == base.api
    assert fast.base_url == base.base_url
    assert fast.compat == base.compat
    assert fast.thinking_level_map == base.thinking_level_map


@pytest.mark.tonio
async def test_resolves_fireworks_api_key_from_the_environment():
    env = {"FIREWORKS_API_KEY": "test-fireworks-key"}

    assert find_env_keys("fireworks", env) == ["FIREWORKS_API_KEY"]
    assert await get_env_api_key("fireworks", env) == "test-fireworks-key"


def test_sets_fireworks_compat_for_session_affinity_and_unsupported_tool_fields():
    model = get_builtin_model("fireworks", KIMI_K2P6)

    assert model is not None
    assert model.compat is not None
    assert model.compat.send_session_affinity_headers is True
    assert model.compat.supports_eager_tool_input_streaming is False
    assert model.compat.supports_cache_control_on_tools is False
    assert model.compat.supports_long_cache_retention is False


# --- session affinity and tool compat on the wire -----------------------------

TOOL = Tool(
    name="lookup",
    description="Look up a value",
    parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
)


def make_fireworks_model(compat: AnthropicMessagesCompat | None = FIREWORKS_ANTHROPIC_COMPAT) -> Model:
    return Model(
        id=KIMI_K2P6,
        name="Kimi K2.6",
        api="anthropic-messages",
        provider="fireworks",
        base_url="https://api.fireworks.ai/inference",
        reasoning=True,
        input=["text", "image"],
        cost=ModelCost(input=0.95, output=4, cache_read=0.16, cache_write=0),
        context_window=262000,
        max_tokens=262000,
        compat=compat,
    )


def make_anthropic_model() -> Model:
    return Model(
        id="claude-opus-4-8",
        name="Claude Opus 4.8",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=200000,
        max_tokens=32000,
    )


def make_tool_context() -> Context:
    return Context(messages=[UserMessage(content="Use the tool", timestamp=now_ms())], tools=[TOOL])


@pytest.mark.tonio
async def test_sends_session_affinity_header_for_fireworks_models():
    headers, _ = await capture_request(
        make_fireworks_model(),
        SimpleStreamOptions(session_id="fireworks-session-1"),
        make_tool_context(),
    )

    assert headers["x-session-affinity"] == "fireworks-session-1"


@pytest.mark.tonio
async def test_omits_session_affinity_header_for_native_anthropic_models():
    headers, _ = await capture_request(
        make_anthropic_model(),
        SimpleStreamOptions(session_id="anthropic-session-1"),
        make_tool_context(),
    )

    assert "x-session-affinity" not in headers


@pytest.mark.tonio
async def test_omits_session_affinity_header_when_cache_retention_is_none():
    headers, _ = await capture_request(
        make_fireworks_model(),
        SimpleStreamOptions(session_id="fireworks-session-2", cache_retention="none"),
        make_tool_context(),
    )

    assert "x-session-affinity" not in headers


@pytest.mark.tonio
async def test_omits_cache_control_on_tools_for_fireworks_models():
    _, payload = await capture_request(make_fireworks_model(), SimpleStreamOptions(), make_tool_context())

    assert payload["tools"]
    assert "cache_control" not in payload["tools"][-1]


@pytest.mark.tonio
async def test_omits_eager_input_streaming_on_tools_for_fireworks_models():
    _, payload = await capture_request(make_fireworks_model(), SimpleStreamOptions(), make_tool_context())

    for tool in payload["tools"]:
        assert "eager_input_streaming" not in tool


@pytest.mark.tonio
async def test_sends_cache_control_on_tools_for_native_anthropic_models():
    _, payload = await capture_request(make_anthropic_model(), SimpleStreamOptions(), make_tool_context())

    assert payload["tools"][-1]["cache_control"]["type"] == "ephemeral"


@pytest.mark.tonio
async def test_sends_eager_input_streaming_on_tools_for_native_anthropic_models():
    _, payload = await capture_request(make_anthropic_model(), SimpleStreamOptions(), make_tool_context())

    assert payload["tools"][0]["eager_input_streaming"] is True
