"""Partial mirror of pi ai/test/constrained-sampling.test.ts.

Holds the 0.84.2 strict-schema-conversion cases (7915cdac) and the 1.0.0
foreign-item-id replay case (bc2d8dc1); the pre-existing grammar/replay cases
remain covered by the adapter suites and are a recorded parity gap in
scripts/upstream_diff.py. pi builds schemas with TypeBox; plain
JSON-schema dicts carry the same shapes here.
"""

import re
import time

import pytest

from pidrei_ai.api.constrained_sampling import (
    UnsupportedStrictJsonSchemaError,
    make_strict_json_schema,
    resolve_json_schema_strict_sampling,
)
from pidrei_ai.api.openai_responses_shared import convert_responses_messages, convert_responses_tools
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    JsonSchemaConstrainedSampling,
    Model,
    ModelCost,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
)
from pidrei_ai.utils.transcript import normalize_context


def make_tool(**overrides) -> Tool:
    defaults = {
        "name": "lookup",
        "description": "Look up a value",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    }
    defaults.update(overrides)
    return Tool(**defaults)


def test_derives_strict_provider_schemas_without_changing_tool_definitions():
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "number"},
            "metadata": {"type": "object", "properties": {"enabled": {"type": "boolean"}}},
            "nullable": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["path", "metadata"],
    }

    strict = make_strict_json_schema(parameters)

    assert parameters["required"] == ["path", "metadata"]
    assert "additionalProperties" not in parameters
    assert strict["additionalProperties"] is False
    assert strict["required"] == ["path", "offset", "metadata", "nullable"]
    assert strict["properties"]["offset"] == {"anyOf": [{"type": "number"}, {"type": "null"}]}
    assert strict["properties"]["metadata"] == {
        "type": "object",
        "properties": {"enabled": {"anyOf": [{"type": "boolean"}, {"type": "null"}]}},
        "required": ["enabled"],
        "additionalProperties": False,
    }
    assert strict["properties"]["nullable"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}


@pytest.mark.parametrize(
    ("parameters", "error"),
    [
        (
            {
                "type": "object",
                "properties": {
                    "metadata": {"type": "object", "properties": {}, "additionalProperties": {"type": "string"}}
                },
                "required": ["metadata"],
            },
            "additionalProperties is unsupported",
        ),
        (
            {
                "allOf": [
                    {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
                    {"type": "object", "properties": {"b": {"type": "number"}}, "required": ["b"]},
                ]
            },
            "allOf schemas are unsupported",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "value": {
                        "anyOf": [
                            {"type": "object", "properties": {"nested": {"type": "string"}}, "required": ["nested"]},
                            {"type": "null"},
                        ]
                    }
                },
                "required": ["value"],
            },
            "object and array unions are unsupported",
        ),
        (
            {
                "type": "object",
                "properties": {"child": {"$ref": "https://example.com/child.json"}},
                "required": ["child"],
            },
            "$ref schemas are unsupported",
        ),
    ],
    ids=["schema-additionalProperties", "allOf", "object-union", "ref"],
)
def test_falls_back_or_rejects_schemas_that_cannot_be_safely_converted(parameters, error):
    tool = make_tool(parameters=parameters, constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"))

    with pytest.raises(UnsupportedStrictJsonSchemaError, match=re.escape(error)):
        make_strict_json_schema(parameters)
    assert resolve_json_schema_strict_sampling(tool, True) is None
    converted = convert_responses_tools([tool], supports_strict_mode=True)[0]
    assert converted["strict"] is False
    assert converted["parameters"] == parameters

    tool.constrained_sampling = JsonSchemaConstrainedSampling(strict="require")
    with pytest.raises(ValueError, match=re.escape(error)):
        resolve_json_schema_strict_sampling(tool, True)


def make_model() -> Model:
    return Model(
        id="gpt-test",
        name="GPT Test",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=False,
        input=["text", "image"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=4096,
    )


# earendil-works/radius#115: a gateway forwards another model's history as a foreign provider.
def test_drops_foreign_item_ids_when_replaying_grammar_calls_as_custom_responses_items():
    context = normalize_context(
        Context(
            messages=[
                AssistantMessage(
                    api="pi-messages",
                    provider="radius",
                    model="gpt-other",
                    content=[ToolCall(id="call_1|ctc_1", name="sample_tool", arguments={"payload": "abc"})],
                    usage=Usage(),
                    stop_reason="toolUse",
                    timestamp=int(time.time() * 1000),
                ),
                ToolResultMessage(
                    tool_call_id="call_1|ctc_1",
                    tool_name="sample_tool",
                    content=[TextContent(text="done")],
                    is_error=False,
                    timestamp=int(time.time() * 1000),
                ),
            ]
        )
    )

    messages = convert_responses_messages(
        make_model(), context, {"openai"}, grammar_tool_input_properties={"sample_tool": "payload"}
    )

    call = next((item for item in messages if item.get("type") == "custom_tool_call"), None)
    assert call is not None
    assert {key: call.get(key) for key in ("type", "call_id", "input")} == {
        "type": "custom_tool_call",
        "call_id": "call_1",
        "input": "abc",
    }
    assert call.get("id") is None
