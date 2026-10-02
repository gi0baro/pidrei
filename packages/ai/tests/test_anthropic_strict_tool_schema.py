"""Mirror of pi's anthropic-strict-tool-schema.test.ts."""

import pytest

from pidrei_ai.api.anthropic_messages import AnthropicOptions
from pidrei_ai.types import (
    AnthropicMessagesCompat,
    Context,
    JsonSchemaConstrainedSampling,
    Model,
    ModelCost,
    Tool,
    UserMessage,
)
from tests.anthropic_helpers import capture_payload, now_ms


def create_model() -> Model:
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
        compat=AnthropicMessagesCompat(force_adaptive_thinking=True, supports_strict_tools=True),
    )


def create_tool(parameters: dict, constrained_sampling: JsonSchemaConstrainedSampling | None = None) -> Tool:
    return Tool(
        name="lookup",
        description="Look up a value",
        parameters=parameters,
        constrained_sampling=constrained_sampling,
    )


def create_strict_tool(parameters: dict) -> Tool:
    return create_tool(parameters, JsonSchemaConstrainedSampling(strict="prefer"))


async def capture_first_tool(tool: Tool) -> dict:
    payload = await capture_payload(
        create_model(),
        AnthropicOptions(api_key="test-key", cache_retention="none"),
        Context(messages=[UserMessage(content="Use the tool", timestamp=now_ms())], tools=[tool]),
    )
    tools = payload.get("tools")
    if not tools:
        raise AssertionError("Expected a tool in the captured Anthropic payload")
    return tools[0]


@pytest.mark.tonio
async def test_only_sends_the_full_input_schema_for_strict_json_schema_tools():
    legacy_parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
        "title": "LookupInput",
    }
    legacy_tool = await capture_first_tool(create_tool(legacy_parameters))
    assert "strict" not in legacy_tool
    assert legacy_tool["input_schema"] == {
        "type": "object",
        "properties": legacy_parameters["properties"],
        "required": legacy_parameters["required"],
    }

    strict_tool = await capture_first_tool(
        create_strict_tool(
            {
                "type": "object",
                "properties": {"value": {"type": "string"}, "optional": {"type": "number"}},
                "required": ["value"],
                "title": "StrictLookupInput",
            }
        )
    )
    assert strict_tool["strict"] is True
    input_schema = strict_tool["input_schema"]
    assert input_schema["additionalProperties"] is False
    assert input_schema["required"] == ["value", "optional"]
    assert input_schema["properties"]["optional"] == {"anyOf": [{"type": "number"}, {"type": "null"}]}
    assert input_schema["title"] == "StrictLookupInput"


# https://github.com/earendil-works/pi/issues/9953
@pytest.mark.tonio
async def test_sends_prefer_tools_non_strict_when_they_use_keywords_anthropic_strict_mode_rejects():
    unsupported_parameters = [
        {
            "type": "object",
            "properties": {"timeoutMs": {"type": "integer", "minimum": 1, "maximum": 300000}},
        },
        {
            "type": "object",
            "properties": {
                "options": {
                    "type": "object",
                    "properties": {"tags": {"type": "array", "items": {"type": "string"}, "minItems": 2}},
                    "required": ["tags"],
                }
            },
            "required": ["options"],
        },
        {
            "type": "object",
            "properties": {"expression": {"type": "string", "format": "regex"}},
            "required": ["expression"],
        },
    ]
    for parameters in unsupported_parameters:
        tool = await capture_first_tool(create_strict_tool(parameters))
        assert "strict" not in tool

    supported_tool = await capture_first_tool(
        create_strict_tool(
            {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "minLength": 1, "maxLength": 1000, "pattern": "^[a-z]+$"},
                    "url": {"type": "string", "format": "uri"},
                    "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
                "required": ["code", "url", "tags"],
            }
        )
    )
    assert supported_tool["strict"] is True
