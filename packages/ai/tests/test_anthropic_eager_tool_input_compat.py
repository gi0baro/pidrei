"""Mirror of pi's anthropic-eager-tool-input-compat.test.ts.

pi captures the request with a loopback HTTP server and reads the
`anthropic-beta` header; here everything goes through `on_payload` capture —
the beta list is the payload's `betas`, which the transport lifts into that
header (as pi's SDK does). Same observable request, no server.
"""

import pytest

from pidrei_ai.api.anthropic_messages import (
    FINE_GRAINED_TOOL_STREAMING_BETA,
)
from pidrei_ai.types import (
    AnthropicMessagesCompat,
    Context,
    JsonSchemaConstrainedSampling,
    Model,
    ModelCost,
    SimpleStreamOptions,
    Tool,
    UserMessage,
)
from tests.anthropic_helpers import capture_payload, now_ms


def create_model(compat: AnthropicMessagesCompat | None = None) -> Model:
    merged = AnthropicMessagesCompat(force_adaptive_thinking=True)
    if compat is not None:
        for field_name in (
            "supports_eager_tool_input_streaming",
            "supports_strict_tools",
            "supports_cache_control_on_tools",
        ):
            value = getattr(compat, field_name)
            if value is not None:
                setattr(merged, field_name, value)
    return Model(
        id="claude-opus-4-8",
        name="Claude Opus 4.8",
        api="anthropic-messages",
        provider="test-anthropic",
        base_url="http://127.0.0.1:9",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=200000,
        max_tokens=32000,
        compat=merged,
    )


TOOL = Tool(
    name="lookup",
    description="Look up a value",
    parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
)

SCHEMA_COMPATIBILITY_TOOL = Tool(
    name="lookup",
    description="Look up a value",
    parameters={
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
        "title": "LookupInput",
    },
)

STRICT_TOOL = Tool(
    name="lookup",
    description="Look up a value",
    parameters={
        "type": "object",
        "properties": {"value": {"type": "string"}, "optional": {"type": "number"}},
        "required": ["value"],
        "title": "StrictLookupInput",
    },
    constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"),
)


def create_context(tools: list[Tool] | None = None) -> Context:
    tool_list = [TOOL] if tools is None else tools
    return Context(
        messages=[UserMessage(content="Use the tool", timestamp=now_ms())],
        tools=tool_list if tool_list else None,
    )


async def capture_body(model: Model, context: Context) -> dict:
    return await capture_payload(model, SimpleStreamOptions(cache_retention="none"), context)


@pytest.mark.tonio
async def test_sends_per_tool_eager_input_streaming_by_default():
    body = await capture_body(create_model(), create_context())

    assert body["tools"][0]["eager_input_streaming"] is True
    assert "betas" not in body


@pytest.mark.tonio
async def test_uses_legacy_fine_grained_beta_when_eager_tool_input_streaming_is_disabled():
    model = create_model(AnthropicMessagesCompat(supports_eager_tool_input_streaming=False))
    body = await capture_body(model, create_context())

    assert "eager_input_streaming" not in body["tools"][0]
    assert body["betas"] == [FINE_GRAINED_TOOL_STREAMING_BETA]


@pytest.mark.tonio
async def test_does_not_send_legacy_beta_when_there_are_no_tools():
    model = create_model(AnthropicMessagesCompat(supports_eager_tool_input_streaming=False))
    body = await capture_body(model, create_context([]))

    assert "tools" not in body
    assert "betas" not in body


@pytest.mark.tonio
async def test_only_sends_the_full_input_schema_for_strict_json_schema_tools():
    legacy_body = await capture_body(
        create_model(AnthropicMessagesCompat(supports_strict_tools=True)),
        create_context([SCHEMA_COMPATIBILITY_TOOL]),
    )
    assert legacy_body["tools"][0]["input_schema"] == {
        "type": "object",
        "properties": SCHEMA_COMPATIBILITY_TOOL.parameters["properties"],
        "required": SCHEMA_COMPATIBILITY_TOOL.parameters["required"],
    }

    strict_body = await capture_body(
        create_model(AnthropicMessagesCompat(supports_strict_tools=True)),
        create_context([STRICT_TOOL]),
    )
    assert strict_body["tools"][0]["strict"] is True
    input_schema = strict_body["tools"][0]["input_schema"]
    assert input_schema["additionalProperties"] is False
    assert input_schema["required"] == ["value", "optional"]
    assert input_schema["properties"]["optional"] == {"anyOf": [{"type": "number"}, {"type": "null"}]}
    assert input_schema["title"] == "StrictLookupInput"
