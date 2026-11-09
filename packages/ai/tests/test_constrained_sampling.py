"""Partial mirror of pi ai/test/constrained-sampling.test.ts.

Holds the 0.84.2 strict-schema-conversion cases (7915cdac), the 1.0.0
foreign-item-id replay case (bc2d8dc1) and, from 1.5.0 (f715a15c), the
grammar input JSON buffer cases and the streamed custom tool input cases; the
other pre-existing grammar/replay cases remain covered by the adapter suites
and are a recorded parity gap in scripts/upstream_diff.py. pi builds schemas
with TypeBox; plain JSON-schema dicts carry the same shapes here.
"""

import json
import math
import re
import time

import pytest

from pidrei_ai.api.constrained_sampling import (
    GrammarToolInputJsonBuffer,
    UnsupportedStrictJsonSchemaError,
    append_grammar_tool_input_json_delta,
    close_grammar_tool_input_json,
    make_strict_json_schema,
    resolve_json_schema_strict_sampling,
)
from pidrei_ai.api.openai_responses_shared import (
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from pidrei_ai.builders import AssistantMessageBuilder, UsageBuilder
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
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
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


def test_keeps_grammar_input_json_deltas_append_only():
    buffer = GrammarToolInputJsonBuffer()
    first = append_grammar_tool_input_json_delta(buffer, "payload", 'a"')
    second = close_grammar_tool_input_json(buffer, "payload", 'a"\nb')

    assert json.loads(f"{first}{second}") == {"payload": 'a"\nb'}
    assert close_grammar_tool_input_json(buffer, "payload", 'a"\nb') is None
    with pytest.raises(ValueError, match='grammar tool input for property "payload" changed after it was closed'):
        close_grammar_tool_input_json(buffer, "payload", "changed")
    with pytest.raises(ValueError, match='grammar tool input for property "payload" changed after it was closed'):
        append_grammar_tool_input_json_delta(buffer, "payload", "x")


def test_rejects_a_final_grammar_input_that_does_not_continue_the_streamed_input():
    buffer = GrammarToolInputJsonBuffer()
    append_grammar_tool_input_json_delta(buffer, "payload", "abc")
    with pytest.raises(ValueError, match='grammar tool input for property "payload" changed non-monotonically'):
        close_grammar_tool_input_json(buffer, "payload", "abd")


async def run_custom_tool_call(events: list[dict]) -> tuple[AssistantMessageBuilder, list[dict], list[str]]:
    """Process Responses events for a `sample_tool` grammar call; returns the output,
    the arguments at each `toolcall_start` and the `toolcall_delta` texts."""
    output = AssistantMessageBuilder(
        content=[],
        api="openai-responses",
        provider="openai",
        model="gpt-test",
        usage=UsageBuilder(),
        stop_reason="pending",
        timestamp=0,
    )
    stream = AssistantMessageEventStream()

    async def provider_events():
        for event in events:
            yield event

    await process_responses_stream(
        provider_events(), output, stream, make_model(), grammar_tool_input_properties={"sample_tool": "payload"}
    )
    stream.end()
    starts: list[dict] = []
    deltas: list[str] = []
    async for event in stream:
        if event.type == "toolcall_start":
            starts.append(event.partial.content[event.content_index].arguments)
        elif event.type == "toolcall_delta":
            deltas.append(event.delta)
    return output, starts, deltas


def custom_tool_call_events(initial_input: str, deltas: list[str]) -> list[dict]:
    item = {"type": "custom_tool_call", "call_id": "call_1", "id": "ctc_1", "name": "sample_tool"}
    complete = initial_input + "".join(deltas)
    return [
        {"type": "response.output_item.added", "output_index": 0, "item": {**item, "input": initial_input}},
        *(
            {"type": "response.custom_tool_call_input.delta", "output_index": 0, "item_id": "ctc_1", "delta": delta}
            for delta in deltas
        ),
        {"type": "response.custom_tool_call_input.done", "output_index": 0, "item_id": "ctc_1", "input": complete},
        {"type": "response.output_item.done", "output_index": 0, "item": {**item, "input": complete}},
        {
            "type": "response.completed",
            "response": {"status": "completed", "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}},
        },
    ]


# #9265
@pytest.mark.tonio
async def test_streams_custom_tool_input_in_time_linear_in_its_length():
    # pi measures V8 heap allocation; this measures wall time, best of two runs.
    async def run(size: int) -> float:
        best = math.inf
        deltas = [f'chunk {index} "quoted" \\ \n'.ljust(32, "x") for index in range(size // 32)]
        for _ in range(2):
            started = time.perf_counter()
            _, _, json_deltas = await run_custom_tool_call(custom_tool_call_events("", deltas))
            best = min(best, time.perf_counter() - started)
        assert json.loads("".join(json_deltas)) == {"payload": "".join(deltas)}
        return best

    small = await run(25_600)
    large = await run(256_000)
    assert large / small < 25


@pytest.mark.tonio
async def test_starts_custom_responses_tool_calls_with_their_initial_input():
    output, starts, deltas = await run_custom_tool_call(custom_tool_call_events("a", ["b", "c"]))

    assert output.stop_reason == "toolUse"
    assert starts == [{"payload": "a"}]
    assert output.freeze().content == [ToolCall(id="call_1|ctc_1", name="sample_tool", arguments={"payload": "abc"})]
    assert json.loads("".join(deltas)) == {"payload": "abc"}
