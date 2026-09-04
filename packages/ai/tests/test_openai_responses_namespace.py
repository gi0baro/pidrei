"""Mirror of pi ai/test/openai-responses-namespace.test.ts."""

import time

import pytest

from pidrei_ai.api.openai_responses_shared import (
    convert_responses_messages,
    process_responses_stream,
)
from pidrei_ai.builders import AssistantMessageBuilder, UsageBuilder
from pidrei_ai.types import AssistantMessage, Context, Model, ModelCost, ToolCall
from pidrei_ai.utils.event_stream import AssistantMessageEventStream


MODEL = Model(
    id="gpt-5.4",
    name="GPT-5.4",
    api="openai-responses",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=True,
    input=["text"],
    cost=ModelCost(),
    context_window=400_000,
    max_tokens=128_000,
)


# Step 2 translation (PROPER_MT_DESIGN.md): the streamed output message is a
# producer-private builder; pi passes its mutable message here.
def create_output() -> AssistantMessageBuilder:
    return AssistantMessageBuilder(
        content=[],
        api=MODEL.api,
        provider=MODEL.provider,
        model=MODEL.id,
        usage=UsageBuilder(),
        stop_reason="pending",
        timestamp=int(time.time() * 1000),
    )


async def create_function_call_events():
    yield {
        "type": "response.output_item.added",
        "sequence_number": 0,
        "output_index": 0,
        "item": {"type": "function_call", "id": "fc_test", "call_id": "call_test", "name": "lookup", "arguments": ""},
    }
    yield {
        "type": "response.output_item.done",
        "sequence_number": 1,
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_test",
            "call_id": "call_test",
            "name": "lookup",
            "arguments": '{"value":"hello"}',
            "namespace": "dynamic_tools",
        },
    }
    yield {
        "type": "response.completed",
        "sequence_number": 2,
        "response": {"id": "resp_test", "status": "completed"},
    }


async def create_custom_tool_call_events():
    yield {
        "type": "response.output_item.added",
        "sequence_number": 0,
        "output_index": 0,
        "item": {"type": "custom_tool_call", "id": "ctc_test", "call_id": "call_test", "name": "query", "input": ""},
    }
    yield {
        "type": "response.output_item.done",
        "sequence_number": 1,
        "output_index": 0,
        "item": {
            "type": "custom_tool_call",
            "id": "ctc_test",
            "call_id": "call_test",
            "name": "query",
            "input": "hello",
            "namespace": "dynamic_tools",
        },
    }
    yield {
        "type": "response.completed",
        "sequence_number": 2,
        "response": {"id": "resp_test", "status": "completed"},
    }


def get_tool_call(output: AssistantMessage) -> ToolCall:
    block = output.content[0] if output.content else None
    if block is None or block.type != "toolCall":
        raise AssertionError("Expected toolCall block")
    return block


@pytest.mark.tonio
async def test_omits_an_absent_error_message():
    output = create_output()
    await process_responses_stream(create_function_call_events(), output, AssistantMessageEventStream(), MODEL)

    assert output.error_message is None


@pytest.mark.tonio
async def test_round_trips_a_function_namespace_received_only_on_output_item_done():
    output = create_output()
    await process_responses_stream(create_function_call_events(), output, AssistantMessageEventStream(), MODEL)

    tool_call = get_tool_call(output)
    assert tool_call.id == "call_test|fc_test"
    assert tool_call.name == "lookup"
    assert tool_call.arguments == {"value": "hello"}
    assert tool_call.namespace == "dynamic_tools"

    replayed = next(
        item
        for item in convert_responses_messages(MODEL, Context(messages=[output]), {"openai"})
        if item["type"] == "function_call"
    )
    assert replayed == {
        "type": "function_call",
        "id": "fc_test",
        "call_id": "call_test",
        "name": "lookup",
        "arguments": '{"value": "hello"}',
        "namespace": "dynamic_tools",
    }


@pytest.mark.tonio
async def test_round_trips_a_custom_tool_namespace_received_only_on_output_item_done():
    output = create_output()
    grammar_tool_input_properties = {"query": "input"}
    await process_responses_stream(
        create_custom_tool_call_events(),
        output,
        AssistantMessageEventStream(),
        MODEL,
        grammar_tool_input_properties=grammar_tool_input_properties,
    )

    tool_call = get_tool_call(output)
    assert tool_call.id == "call_test|ctc_test"
    assert tool_call.name == "query"
    assert tool_call.arguments == {"input": "hello"}
    assert tool_call.namespace == "dynamic_tools"

    replayed = next(
        item
        for item in convert_responses_messages(
            MODEL,
            Context(messages=[output]),
            {"openai"},
            grammar_tool_input_properties=grammar_tool_input_properties,
        )
        if item["type"] == "custom_tool_call"
    )
    assert replayed == {
        "type": "custom_tool_call",
        "id": "ctc_test",
        "call_id": "call_test",
        "name": "query",
        "input": "hello",
        "namespace": "dynamic_tools",
    }


def test_drops_namespaces_when_the_target_cannot_replay_their_load_items():
    from dataclasses import replace

    output = create_output()
    output.content.extend(
        [
            ToolCall(
                id="call_function|fc_test",
                name="lookup",
                arguments={"value": "hello"},
                namespace="dynamic_tools",
            ),
            ToolCall(
                id="call_custom|ctc_test",
                name="query",
                arguments={"input": "hello"},
                namespace="dynamic_tools",
            ),
        ]
    )
    target_models = [
        replace(MODEL, id="gpt-5.2", name="GPT-5.2"),
        replace(MODEL, provider="azure-openai-responses"),
        replace(
            MODEL,
            api="openai-codex-responses",
            provider="openai-codex",
            id="gpt-5.3-codex-spark",
            name="GPT-5.3 Codex Spark",
        ),
    ]

    for target_model in target_models:
        replayed = convert_responses_messages(
            target_model,
            Context(messages=[output]),
            {"openai"},
            grammar_tool_input_properties={"query": "input"},
        )
        function_call = next((item for item in replayed if item["type"] == "function_call"), None)
        custom_tool_call = next((item for item in replayed if item["type"] == "custom_tool_call"), None)
        assert function_call is not None
        assert "namespace" not in function_call
        assert custom_tool_call is not None
        assert "namespace" not in custom_tool_call


def test_does_not_add_a_namespace_to_ordinary_function_calls():
    output = create_output()
    output.content.append(ToolCall(id="call_test|fc_test", name="lookup", arguments={"value": "hello"}))

    replayed = next(
        (
            item
            for item in convert_responses_messages(MODEL, Context(messages=[output]), {"openai"})
            if item["type"] == "function_call"
        ),
        None,
    )
    assert replayed is not None
    assert "namespace" not in replayed
