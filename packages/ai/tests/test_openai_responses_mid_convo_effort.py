"""Mirror of pi's openai-responses-mid-convo-effort.test.ts.

pi captures the request body with a mocked `fetch`; the mirror captures the
params the adapter hands its client through the sibling suites' `FakeClient`.
"""

from dataclasses import replace

import pytest

from pidrei_ai.api.openai_responses import OpenAIResponsesOptions, stream as stream_responses
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    Message,
    Model,
    ModelCost,
    OpenAIResponsesCompat,
    TextContent,
    ToolResultMessage,
    UserMessage,
)
from pidrei_ai.utils.transcript import normalize_context
from tests.test_openai_responses import FakeClient


THINKING_LEVEL_MAP = {
    "off": "none",
    "minimal": None,
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}

_DEFAULT_COMPAT = OpenAIResponsesCompat(supports_mid_convo_effort=True)


def openai_model(compat: OpenAIResponsesCompat = _DEFAULT_COMPAT) -> Model:
    return Model(
        id="gpt-6-luna",
        name="GPT-6 Luna",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=True,
        thinking_level_map=THINKING_LEVEL_MAP,
        input=["text"],
        cost=ModelCost(),
        context_window=272000,
        max_tokens=128000,
        compat=compat,
    )


def user(text: str) -> Message:
    return UserMessage(content=text, timestamp=0)


def tool_result(call_id: str) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=call_id,
        tool_name="lookup",
        content=[TextContent(text="result")],
        is_error=False,
        timestamp=0,
    )


def update(effort: str) -> dict:
    return {"type": "configuration_update", "reasoning": {"effort": effort}}


def response_events(response_id: str, with_tool_call: bool) -> list[dict]:
    item = (
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{}",
            "status": "completed",
        }
        if with_tool_call
        else {
            "type": "message",
            "id": f"msg_{response_id}",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "answer", "annotations": []}],
        }
    )
    return [
        {"type": "response.created", "response": {"id": response_id}},
        {"type": "response.output_item.added", "output_index": 0, "item": item},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": {"id": response_id, "status": "completed"}},
    ]


async def run_openai(
    messages: list[Message],
    reasoning_effort: str | None,
    *,
    api_key: str = "sk-test",
    model: Model | None = None,
    with_tool_call: bool = False,
) -> tuple[dict, AssistantMessage]:
    client = FakeClient(response_events("resp_1", with_tool_call))
    message = await stream_responses(
        model if model is not None else openai_model(),
        normalize_context(Context(messages=list(messages))),
        OpenAIResponsesOptions(api_key=api_key, reasoning_effort=reasoning_effort, client=client),
    ).result()
    assert client.requests, "Request payload was not captured"
    assert message.stop_reason != "error"
    return client.requests[0], message


@pytest.mark.tonio
async def test_keeps_reasoning_effort_constant_and_only_appends_input_when_the_level_changes():
    history: list[Message] = [user("one")]
    first_payload, first_message = await run_openai(history, "high")
    history.extend([first_message, user("two")])
    second_payload, second_message = await run_openai(history, "high")
    history.extend([second_message, user("three")])
    third_payload, third_message = await run_openai(history, "low")

    assert first_payload["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "one"}]},
        update("high"),
    ]
    for payload in (first_payload, second_payload, third_payload):
        assert payload["reasoning"]["effort"] == "medium"
    assert second_payload["input"][: len(first_payload["input"])] == first_payload["input"]
    assert third_payload["input"][: len(second_payload["input"])] == second_payload["input"]
    assert [item for item in third_payload["input"] if item.get("type") == "configuration_update"] == [
        update("high"),
        update("high"),
        update("low"),
    ]
    assert third_payload["input"][-1] == update("low")
    assert [message.provider_thinking_level for message in (first_message, second_message, third_message)] == [
        "high",
        "high",
        "low",
    ]


@pytest.mark.tonio
async def test_applies_a_level_change_made_during_a_tool_loop_to_the_next_reply():
    history: list[Message] = [user("one")]
    tool_call_payload, tool_call_message = await run_openai(history, "high", with_tool_call=True)
    history.extend([tool_call_message, tool_result("call_1|fc_1")])
    after_tool_payload, after_tool_message = await run_openai(history, "low")

    assert after_tool_payload["input"][: len(tool_call_payload["input"])] == tool_call_payload["input"]
    function_call_output, last = after_tool_payload["input"][-2:]
    assert function_call_output["type"] == "function_call_output"
    assert function_call_output["call_id"] == "call_1"
    assert last == update("low")
    assert after_tool_message.provider_thinking_level == "low"


@pytest.mark.tonio
async def test_adds_items_only_before_replies_recorded_by_this_model():
    _, unstamped = await run_openai([user("one")], "high", model=openai_model(OpenAIResponsesCompat()))
    # Pi spreads the message into new literals; the frozen message is replaced.
    other_provider = replace(unstamped, provider="opencode", provider_thinking_level="high")
    other_model = replace(unstamped, model="gpt-6-sol", provider_thinking_level="high")
    recorded = replace(unstamped, provider_thinking_level="high")

    payload, _ = await run_openai(
        [
            user("one"),
            unstamped,
            user("two"),
            other_provider,
            user("three"),
            other_model,
            user("four"),
            recorded,
            user("five"),
        ],
        "low",
    )

    assert [item.get("type", item.get("role")) for item in payload["input"]] == [
        "user",
        "message",
        "user",
        "message",
        "user",
        "message",
        "user",
        "configuration_update",
        "message",
        "user",
        "configuration_update",
    ]
    assert [item for item in payload["input"] if item.get("type") == "configuration_update"] == [
        update("high"),
        update("low"),
    ]


@pytest.mark.tonio
@pytest.mark.parametrize(
    ("api_key", "model", "level", "effort"),
    [
        pytest.param("chatgpt-access-token", openai_model(), "low", "low", id="Sign in with ChatGPT"),
        pytest.param(
            "sk-test",
            openai_model(OpenAIResponsesCompat(supports_mid_convo_effort=False)),
            "low",
            "low",
            id="the compat flag turned off",
        ),
        pytest.param("sk-test", openai_model(), None, "none", id="thinking off"),
    ],
)
async def test_sends_the_level_as_reasoning_effort(api_key, model, level, effort):
    _, first = await run_openai([user("one")], "high", api_key=api_key, model=model)
    payload, message = await run_openai([user("one"), first, user("two")], level, api_key=api_key, model=model)

    assert payload["reasoning"]["effort"] == effort
    assert not any(item.get("type") == "configuration_update" for item in payload["input"])
    assert message.provider_thinking_level is None
