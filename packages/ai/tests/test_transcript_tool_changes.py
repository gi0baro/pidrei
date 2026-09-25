"""Mirror of pi ai/test/transcript-tool-changes.test.ts.

pi drives the public compat `streamSimple`, which normalizes the context and
dispatches on `model.api`; `capture_payload` does the same over the adapter
modules directly.
"""

from dataclasses import replace
from typing import Any

import pytest

from pidrei_ai.api import anthropic_messages, openai_completions, openai_responses
from pidrei_ai.types import (
    AnthropicMessagesCompat,
    Context,
    Model,
    ModelCost,
    OpenAICompletionsCompat,
    OpenAIResponsesCompat,
    SimpleStreamOptions,
    SystemMessage,
    Tool,
    ToolReference,
    UserMessage,
)
from pidrei_ai.utils.transcript import normalize_context


class PayloadCaptured(Exception):
    pass


_STREAM_SIMPLE = {
    "anthropic-messages": anthropic_messages.stream_simple,
    "openai-responses": openai_responses.stream_simple,
    "openai-completions": openai_completions.stream_simple,
}


def tool(name: str) -> Tool:
    return Tool(name=name, description=f"{name} tool", parameters={"type": "object", "properties": {}})


async def capture_payload(model: Model, context: Context) -> dict[str, Any]:
    captured: list[dict[str, Any]] = []

    async def on_payload(payload, _model):
        captured.append(payload)
        raise PayloadCaptured()

    stream = _STREAM_SIMPLE[model.api](
        model, normalize_context(context), SimpleStreamOptions(api_key="test-key", on_payload=on_payload)
    )
    await stream.result()
    if not captured:
        raise AssertionError("Expected payload capture")
    return captured[0]


def make_model(model_id: str, name: str, api: str, provider: str, compat: Any = None, reasoning: bool = True) -> Model:
    return Model(
        id=model_id,
        name=name,
        api=api,
        provider=provider,
        base_url="http://127.0.0.1:9",
        reasoning=reasoning,
        input=["text"],
        cost=ModelCost(),
        context_window=100000,
        max_tokens=1000,
        compat=compat,
    )


BASE_TOOL = tool("base_tool")
LATE_TOOL = tool("late_tool")
CONTEXT = Context(
    messages=[
        SystemMessage(
            content="base prompt",
            sections={"rules": "<rules>\nold rules\n</rules>", "docs": "<docs>\nread docs\n</docs>"},
            tools_added=[BASE_TOOL],
            timestamp=0,
        ),
        UserMessage(content="before", timestamp=1),
        SystemMessage(
            content="updated guidance",
            sections={"rules": "<rules>\nnew rules\n</rules>", "docs": None},
            tools_removed=[ToolReference(name="base_tool")],
            tools_added=[LATE_TOOL],
            timestamp=2,
        ),
    ]
)
ADDITION_CONTEXT = Context(
    messages=[
        SystemMessage(content="base prompt", tools_added=[BASE_TOOL], timestamp=0),
        UserMessage(content="before", timestamp=1),
        SystemMessage(content="updated guidance", tools_added=[LATE_TOOL], timestamp=2),
    ]
)

ANTHROPIC_NATIVE_MODEL = make_model(
    "claude-opus-5",
    "Claude Opus 5",
    "anthropic-messages",
    "anthropic",
    AnthropicMessagesCompat(supports_mid_convo_system_messages=True, supports_mid_convo_tool_changes=True),
)

MID_CONVERSATION_TOOL_CHANGES_BETA = "mid-conversation-tool-changes-2026-07-01"


@pytest.mark.tonio
async def test_sends_anthropic_updates_and_tool_changes_in_native_system_messages():
    payload = await capture_payload(ANTHROPIC_NATIVE_MODEL, CONTEXT)

    assert MID_CONVERSATION_TOOL_CHANGES_BETA in payload["betas"]
    assert [block["text"] for block in payload["system"]] == [
        "base prompt\n\n<rules>\nold rules\n</rules>\n\n<docs>\nread docs\n</docs>"
    ]
    # Initial tools stay active and carry the cache breakpoint; the placeholder and every
    # later declaration are deferred; the removed tool stays declared.
    tools = payload["tools"]
    assert [tool["name"] for tool in tools] == ["base_tool", "__pi_deferred_placeholder__", "late_tool"]
    assert tools[0]["cache_control"] == {"type": "ephemeral"}
    assert tools[1]["defer_loading"] is True
    assert tools[2]["defer_loading"] is True
    assert "defer_loading" not in tools[0]
    assert "cache_control" not in tools[1]
    assert "cache_control" not in tools[2]
    update = payload["messages"][-1]
    assert update["role"] == "system"
    assert [block["type"] for block in update["content"]] == ["text", "tool_removal", "tool_addition"]
    assert update["content"][1]["tool"]["name"] == "base_tool"
    assert update["content"][2]["tool"]["name"] == "late_tool"
    assert "updated guidance" in update["content"][0]["text"]
    assert "<rules>\nnew rules\n</rules>" in update["content"][0]["text"]
    assert 'Removed system prompt section "docs"' in update["content"][0]["text"]

    # The placeholder is declared before any change so its scaffolding is cached from request one.
    initial = await capture_payload(ANTHROPIC_NATIVE_MODEL, Context(messages=CONTEXT.messages[:2]))
    assert [tool["name"] for tool in initial["tools"]] == ["base_tool", "__pi_deferred_placeholder__"]


@pytest.mark.tonio
async def test_sends_the_current_anthropic_tool_list_when_native_tool_changes_cannot_express_the_history():
    redefined_tool = replace(BASE_TOOL, description="changed")
    fallback_contexts = [
        # Same-name redefinition: blocks reference tools by name only.
        Context(
            messages=[
                SystemMessage(content="base prompt", tools_added=[BASE_TOOL], timestamp=0),
                SystemMessage(
                    content="updated guidance",
                    tools_removed=[ToolReference(name="base_tool")],
                    tools_added=[redefined_tool],
                    timestamp=2,
                ),
            ]
        ),
        # No initial tool: Anthropic rejects an all-deferred tool list.
        Context(
            messages=[
                SystemMessage(content="base prompt", timestamp=0),
                SystemMessage(content="updated guidance", tools_added=[redefined_tool], timestamp=2),
            ]
        ),
    ]
    for fallback_context in fallback_contexts:
        payload = await capture_payload(ANTHROPIC_NATIVE_MODEL, fallback_context)
        assert MID_CONVERSATION_TOOL_CHANGES_BETA not in payload.get("betas", [])
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["name"] == "base_tool"
        assert payload["tools"][0]["description"] == "changed"
        assert payload["tools"][0]["cache_control"] == {"type": "ephemeral"}
        assert "defer_loading" not in payload["tools"][0]
        assert [block["type"] for block in payload["messages"][-1]["content"]] == ["text"]


@pytest.mark.tonio
async def test_folds_anthropic_updates_into_the_system_prompt_without_native_support():
    model = make_model("claude-sonnet-4-5", "Claude Sonnet 4.5", "anthropic-messages", "anthropic")
    payload = await capture_payload(model, CONTEXT)

    assert MID_CONVERSATION_TOOL_CHANGES_BETA not in payload.get("betas", [])
    assert [block["text"] for block in payload["system"]] == [
        "base prompt\n\nupdated guidance\n\n<rules>\nnew rules\n</rules>"
    ]
    assert [value["name"] for value in payload["tools"]] == ["late_tool"]
    assert [message["role"] for message in payload["messages"]] == ["user"]


@pytest.mark.tonio
async def test_requires_both_anthropic_capabilities_for_native_tool_changes():
    model = make_model(
        "claude-opus-5",
        "Claude Opus 5",
        "anthropic-messages",
        "anthropic",
        AnthropicMessagesCompat(supports_mid_convo_tool_changes=True),
    )
    payload = await capture_payload(model, CONTEXT)

    assert MID_CONVERSATION_TOOL_CHANGES_BETA not in payload.get("betas", [])
    assert [value["name"] for value in payload["tools"]] == ["late_tool"]
    assert [message["role"] for message in payload["messages"]] == ["user"]


@pytest.mark.tonio
async def test_anchors_openai_additions_at_their_developer_message():
    model = make_model(
        "gpt-5.4",
        "GPT-5.4",
        "openai-responses",
        "openai",
        OpenAIResponsesCompat(supports_mid_convo_system_messages=True, supports_additional_tools=True),
    )
    payload = await capture_payload(model, ADDITION_CONTEXT)

    assert [value["name"] for value in payload["tools"]] == ["base_tool"]
    additional = next(item for item in payload["input"] if item.get("type") == "additional_tools")
    assert [value["name"] for value in additional["tools"]] == ["late_tool"]
    assert [
        item.get("content") for item in payload["input"] if item.get("role") == "developer" and item.get("type") is None
    ] == ["base prompt", "updated guidance"]


@pytest.mark.tonio
async def test_maps_system_message_additions_into_synthetic_tool_search():
    model = make_model(
        "gpt-5.4",
        "GPT-5.4",
        "openai-responses",
        "openai",
        OpenAIResponsesCompat(supports_mid_convo_system_messages=True, supports_tool_search=True),
    )
    payload = await capture_payload(model, ADDITION_CONTEXT)

    assert [value["name"] for value in payload["tools"]] == ["base_tool"]
    assert "tool_search_call" in [item.get("type") for item in payload["input"]]
    output = next(item for item in payload["input"] if item.get("type") == "tool_search_output")
    assert [value["name"] for value in output["tools"]] == ["late_tool"]


@pytest.mark.tonio
async def test_folds_openai_updates_into_the_leading_developer_message_without_native_support():
    model = make_model(
        "gpt-4.1", "GPT-4.1", "openai-responses", "openai", OpenAIResponsesCompat(supports_additional_tools=True)
    )
    payload = await capture_payload(model, CONTEXT)

    assert [value["name"] for value in payload["tools"]] == ["late_tool"]
    assert [item.get("type") or item.get("role") for item in payload["input"]] == ["developer", "user"]
    assert payload["input"][0]["content"] == "base prompt\n\nupdated guidance\n\n<rules>\nnew rules\n</rules>"


@pytest.mark.tonio
async def test_falls_back_to_the_complete_current_tool_state_when_removals_are_unsupported():
    model = make_model(
        "gpt-5.4",
        "GPT-5.4",
        "openai-responses",
        "openai",
        OpenAIResponsesCompat(supports_mid_convo_system_messages=True, supports_additional_tools=True),
    )
    payload = await capture_payload(model, CONTEXT)

    assert [value["name"] for value in payload["tools"]] == ["late_tool"]
    assert not any(item.get("type") == "additional_tools" for item in payload["input"])
    assert len([item for item in payload["input"] if item.get("role") == "developer"]) == 2


@pytest.mark.tonio
async def test_anchors_kimi_additions_in_tool_bearing_system_messages():
    model = make_model(
        "kimi-k3",
        "Kimi K3",
        "openai-completions",
        "moonshotai",
        OpenAICompletionsCompat(supports_mid_convo_system_messages=True, supports_mid_convo_tool_additions=True),
    )
    payload = await capture_payload(model, ADDITION_CONTEXT)

    assert [value["function"]["name"] for value in payload["tools"]] == ["base_tool"]
    tool_message = next(message for message in payload["messages"] if "tools" in message)
    assert [value["function"]["name"] for value in tool_message["tools"]] == ["late_tool"]
    assert [message.get("content") for message in payload["messages"] if message["role"] == "system"] == [
        "base prompt",
        None,
        "updated guidance",
    ]


@pytest.mark.tonio
async def test_keeps_kimi_k2_system_text_inline_without_dynamic_tool_messages():
    model = make_model(
        "kimi-k2.7-code",
        "Kimi K2.7 Code",
        "openai-completions",
        "moonshotai",
        OpenAICompletionsCompat(supports_mid_convo_system_messages=True),
    )
    payload = await capture_payload(model, ADDITION_CONTEXT)

    assert [value["function"]["name"] for value in payload["tools"]] == ["base_tool", "late_tool"]
    assert not any("tools" in message for message in payload["messages"])
    assert [message.get("content") for message in payload["messages"] if message["role"] == "system"] == [
        "base prompt",
        "updated guidance",
    ]


@pytest.mark.tonio
async def test_folds_openai_compatible_updates_into_the_system_prompt_without_native_support():
    model = make_model("custom-model", "Custom model", "openai-completions", "custom-provider", reasoning=False)
    payload = await capture_payload(model, CONTEXT)

    assert [value["function"]["name"] for value in payload["tools"]] == ["late_tool"]
    assert [message["role"] for message in payload["messages"]] == ["system", "user"]
    assert payload["messages"][0]["content"] == "base prompt\n\nupdated guidance\n\n<rules>\nnew rules\n</rules>"
