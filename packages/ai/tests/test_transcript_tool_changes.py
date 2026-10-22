"""Mirror of pi ai/test/transcript-tool-changes.test.ts.

pi drives the public compat `streamSimple`, which normalizes the context and
dispatches on `model.api`; `capture_payload` does the same over the adapter
modules directly.
"""

from dataclasses import replace
from itertools import pairwise
from typing import Any

import pytest

from pidrei_ai.api import anthropic_messages, openai_completions, openai_responses
from pidrei_ai.providers.all import get_builtin_model
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
from pidrei_ai.utils.transcript import TranscriptTools, normalize_context, resolve_transcript_tools


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
DROP_TOOL = tool("drop_tool")


async def capture_tool_history(model: Model) -> list[dict[str, Any]]:
    """Capture one request after each step: initial tools, removal of drop_tool, then two loads.
    Returns the payloads after the removal and after each load, plus the one before the removal."""
    steps = [
        [UserMessage(content="first", timestamp=1)],
        [
            SystemMessage(content="", tools_removed=[ToolReference(name="drop_tool")], timestamp=2),
            UserMessage(content="after removal", timestamp=3),
        ],
        [
            SystemMessage(content="", tools_added=[LATE_TOOL], timestamp=4),
            UserMessage(content="after first load", timestamp=5),
        ],
        [
            SystemMessage(content="", tools_added=[tool("later_tool")], timestamp=6),
            UserMessage(content="after second load", timestamp=7),
        ],
    ]
    messages: list = [SystemMessage(content="base prompt", tools_added=[BASE_TOOL, DROP_TOOL], timestamp=0)]
    payloads = []
    for step in steps:
        messages.extend(step)
        payloads.append(await capture_payload(model, Context(messages=list(messages))))
    return payloads


def assert_growing_prefix(inputs: list[list]) -> None:
    """Every request's message list starts with the previous request's message list."""
    for previous, current in pairwise(inputs):
        assert current[: len(previous)] == previous


ADDITIONAL_TOOLS_MODEL = make_model(
    "gpt-5.4",
    "GPT-5.4",
    "openai-responses",
    "openai",
    OpenAIResponsesCompat(supports_mid_convo_system_messages=True, supports_additional_tools=True),
)
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

INLINE_TOOLS_BETA = "inline-tools-2026-09-15"
MID_CONVERSATION_TOOL_CHANGES_BETA = "mid-conversation-tool-changes-2026-07-01"


@pytest.mark.tonio
async def test_sends_anthropic_updates_and_tool_changes_in_native_system_messages():
    payload = await capture_payload(ANTHROPIC_NATIVE_MODEL, CONTEXT)

    assert INLINE_TOOLS_BETA in payload["betas"]
    assert MID_CONVERSATION_TOOL_CHANGES_BETA not in payload["betas"]
    assert [block["text"] for block in payload["system"]] == [
        "base prompt\n\n<rules>\nold rules\n</rules>\n\n<docs>\nread docs\n</docs>"
    ]
    # The top-level list holds only the initial tools (with the cache breakpoint) and the
    # deferred placeholder; later tools never appear there.
    tools = payload["tools"]
    assert [tool["name"] for tool in tools] == ["base_tool", "__pi_deferred_placeholder__"]
    assert tools[0]["cache_control"] == {"type": "ephemeral"}
    assert tools[1]["defer_loading"] is True
    assert "defer_loading" not in tools[0]
    assert "cache_control" not in tools[1]
    update = payload["messages"][-1]
    assert update["role"] == "system"
    assert [block["type"] for block in update["content"]] == ["text", "tool_removal", "tool_addition"]
    assert update["content"][1]["tool"] == {"type": "tool_reference", "name": "base_tool"}
    addition = update["content"][2]
    assert addition["tool"]["type"] == "tool_definition"
    assert addition["tool"]["definition"]["name"] == "late_tool"
    assert addition["tool"]["definition"]["description"] == "late_tool tool"
    assert addition["cache_control"] == {"type": "ephemeral"}
    # Cache control goes on the block, never on the inline definition; nothing is deferred.
    assert "cache_control" not in addition["tool"]["definition"]
    assert "defer_loading" not in addition["tool"]["definition"]
    assert "updated guidance" in update["content"][0]["text"]
    assert "<rules>\nnew rules\n</rules>" in update["content"][0]["text"]
    assert 'Removed system prompt section "docs"' in update["content"][0]["text"]

    # The placeholder is declared before any change so its scaffolding is cached from request one.
    initial = await capture_payload(ANTHROPIC_NATIVE_MODEL, Context(messages=CONTEXT.messages[:2]))
    assert [tool["name"] for tool in initial["tools"]] == ["base_tool", "__pi_deferred_placeholder__"]


@pytest.mark.tonio
async def test_redefines_an_anthropic_tool_by_value_under_the_same_name():
    redefined_tool = replace(BASE_TOOL, description="changed")
    payload = await capture_payload(
        ANTHROPIC_NATIVE_MODEL,
        Context(
            messages=[
                SystemMessage(content="base prompt", tools_added=[BASE_TOOL], timestamp=0),
                UserMessage(content="before", timestamp=1),
                SystemMessage(
                    content="",
                    tools_removed=[ToolReference(name="base_tool")],
                    tools_added=[redefined_tool],
                    timestamp=2,
                ),
            ]
        ),
    )

    assert INLINE_TOOLS_BETA in payload["betas"]
    # The top-level declaration keeps the original definition, so the cached prefix survives.
    assert [tool["name"] for tool in payload["tools"]] == ["base_tool", "__pi_deferred_placeholder__"]
    assert payload["tools"][0]["description"] == "base_tool tool"
    # The new definition replaces the old one in place; no removal block is sent for it.
    update = payload["messages"][-1]
    assert update["role"] == "system"
    assert len(update["content"]) == 1
    assert update["content"][0]["type"] == "tool_addition"
    assert update["content"][0]["tool"]["type"] == "tool_definition"
    assert update["content"][0]["tool"]["definition"]["name"] == "base_tool"
    assert update["content"][0]["tool"]["definition"]["description"] == "changed"


@pytest.mark.tonio
async def test_sends_the_current_anthropic_tool_list_without_an_initial_tool():
    # Anthropic rejects a tool list where every tool (here: the placeholder) is deferred.
    payload = await capture_payload(
        ANTHROPIC_NATIVE_MODEL,
        Context(
            messages=[
                SystemMessage(content="base prompt", timestamp=0),
                UserMessage(content="before", timestamp=1),
                SystemMessage(content="updated guidance", tools_added=[LATE_TOOL], timestamp=2),
            ]
        ),
    )
    assert INLINE_TOOLS_BETA not in payload.get("betas", [])
    assert len(payload["tools"]) == 1
    assert payload["tools"][0]["name"] == "late_tool"
    assert payload["tools"][0]["cache_control"] == {"type": "ephemeral"}
    assert "defer_loading" not in payload["tools"][0]
    assert [block["type"] for block in payload["messages"][-1]["content"]] == ["text"]


@pytest.mark.tonio
async def test_folds_anthropic_updates_into_the_system_prompt_without_native_support():
    model = make_model("claude-sonnet-4-5", "Claude Sonnet 4.5", "anthropic-messages", "anthropic")
    payload = await capture_payload(model, CONTEXT)

    assert INLINE_TOOLS_BETA not in payload.get("betas", [])
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

    assert INLINE_TOOLS_BETA not in payload.get("betas", [])
    assert [value["name"] for value in payload["tools"]] == ["late_tool"]
    assert [message["role"] for message in payload["messages"]] == ["user"]


def _additional_tools_names(payload: dict[str, Any]) -> list[list[str]]:
    return [
        [value["name"] for value in item["tools"]]
        for item in payload["input"]
        if item.get("type") == "additional_tools"
    ]


def _contents(payload: dict[str, Any], needle: str) -> list[str]:
    return [item["content"] for item in payload["input"] if needle in (item.get("content") or "")]


@pytest.mark.tonio
async def test_declares_every_openai_tool_in_additional_tools_at_its_developer_message():
    payload = await capture_payload(ADDITIONAL_TOOLS_MODEL, ADDITION_CONTEXT)

    # A later additional_tools item cannot replace a top-level tool, so no tool is declared there.
    assert "tools" not in payload
    assert [names[0] for names in _additional_tools_names(payload)] == ["base_tool", "late_tool"]
    # The incremental notice keeps models from reading a later item as the complete tool set.
    shape = [item.get("type") or item.get("content") for item in payload["input"]]
    assert len(shape) == 6
    assert shape[:2] == ["additional_tools", "base prompt"]
    assert isinstance(shape[2], list)
    assert "incremental tools update" in shape[3]
    assert shape[4:] == ["additional_tools", "updated guidance"]


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
async def test_announces_openai_removals_as_text():
    payload = await capture_payload(ADDITIONAL_TOOLS_MODEL, CONTEXT)

    assert "tools" not in payload
    assert [names[0] for names in _additional_tools_names(payload)] == ["base_tool", "late_tool"]
    update = payload["input"][-1]["content"]
    assert "updated guidance" in update
    assert "The following tools are no longer available. Do not call them:\n- base_tool" in update


@pytest.mark.tonio
async def test_redefines_an_openai_tool_with_additional_tools_instead_of_announcing_a_removal():
    payload = await capture_payload(
        ADDITIONAL_TOOLS_MODEL,
        Context(
            messages=[
                SystemMessage(content="base prompt", tools_added=[BASE_TOOL], timestamp=0),
                UserMessage(content="before", timestamp=1),
                SystemMessage(
                    content="",
                    tools_removed=[ToolReference(name="base_tool")],
                    tools_added=[replace(BASE_TOOL, description="changed")],
                    timestamp=2,
                ),
            ]
        ),
    )

    assert "tools" not in payload
    items = payload["input"]
    assert len(items) == 5
    assert items[0]["type"] == "additional_tools"
    assert [(value["name"], value["description"]) for value in items[0]["tools"]] == [("base_tool", "base_tool tool")]
    assert (items[1]["role"], items[1]["content"]) == ("developer", "base prompt")
    assert items[2]["role"] == "user"
    assert items[3]["role"] == "developer"
    assert "latest definition replaces the earlier one" in items[3]["content"]
    assert items[4]["type"] == "additional_tools"
    assert [(value["name"], value["description"]) for value in items[4]["tools"]] == [("base_tool", "changed")]


# pi #10940: deferred loads after a removal must not rewrite the cached prefix.
@pytest.mark.tonio
async def test_keeps_the_openai_request_prefix_across_a_removal_and_later_loads():
    payloads = await capture_tool_history(ADDITIONAL_TOOLS_MODEL)

    for payload in payloads:
        assert "tools" not in payload
    assert_growing_prefix([payload["input"] for payload in payloads])
    last = payloads[-1]
    assert _additional_tools_names(last) == [["base_tool", "drop_tool"], ["late_tool"], ["later_tool"]]
    assert _contents(last, "no longer available. Do not call them:\n- drop_tool")
    # Each later load restates the tools that are still unavailable.
    notices = _contents(last, "incremental tools update")
    assert len(notices) == 2
    assert all("\nStill unavailable: drop_tool." in notice for notice in notices)


@pytest.mark.tonio
async def test_stops_listing_a_tool_as_unavailable_once_it_is_declared_again():
    payload = await capture_payload(
        ADDITIONAL_TOOLS_MODEL,
        Context(
            messages=[
                SystemMessage(content="base prompt", tools_added=[BASE_TOOL, DROP_TOOL], timestamp=0),
                SystemMessage(content="", tools_removed=[ToolReference(name="drop_tool")], timestamp=1),
                SystemMessage(content="", tools_added=[DROP_TOOL], timestamp=2),
                UserMessage(content="go", timestamp=3),
            ]
        ),
    )

    notices = _contents(payload, "incremental tools update")
    assert notices
    assert "Still unavailable" not in notices[0]
    assert payload["input"][-2]["type"] == "additional_tools"
    assert [value["name"] for value in payload["input"][-2]["tools"]] == ["drop_tool"]


@pytest.mark.tonio
async def test_does_not_list_a_tool_as_still_unavailable_before_its_removal_is_announced():
    payload = await capture_payload(
        ADDITIONAL_TOOLS_MODEL,
        Context(
            messages=[
                SystemMessage(content="base prompt", tools_added=[BASE_TOOL, DROP_TOOL], timestamp=0),
                SystemMessage(
                    content="",
                    tools_added=[LATE_TOOL],
                    tools_removed=[ToolReference(name="drop_tool")],
                    timestamp=1,
                ),
                UserMessage(content="go", timestamp=2),
            ]
        ),
    )

    # Like Codex: notice, then the tools, then the removal notice.
    notice, tools_item, removal, user = payload["input"][-4:]
    assert "incremental tools update" in notice["content"]
    assert "Still unavailable" not in notice["content"]
    assert tools_item["type"] == "additional_tools"
    assert [value["name"] for value in tools_item["tools"]] == ["late_tool"]
    assert "no longer available. Do not call them:\n- drop_tool" in removal["content"]
    assert user["role"] == "user"


# pi #10940: synthetic tool search cannot remove tools, so a removal rewrites the prefix once.
@pytest.mark.tonio
async def test_keeps_the_openai_tool_search_prefix_across_loads_after_a_removal():
    model = replace(
        ADDITIONAL_TOOLS_MODEL,
        compat=OpenAIResponsesCompat(supports_mid_convo_system_messages=True, supports_tool_search=True),
    )
    before_removal, *after_removal = await capture_tool_history(model)

    assert [value["name"] for value in before_removal["tools"]] == ["base_tool", "drop_tool"]
    for payload in after_removal:
        assert [value["name"] for value in payload["tools"]] == ["base_tool"]
    assert_growing_prefix([payload["input"] for payload in after_removal])
    assert [
        item["tools"][0]["name"] for item in after_removal[-1]["input"] if item.get("type") == "tool_search_output"
    ] == ["late_tool", "later_tool"]


# pi #10940: removals and redeclarations move the top-level boundary once; plain additions stay anchored.
def test_resolves_the_top_level_tool_boundary_per_transport():
    messages = [
        SystemMessage(content="base prompt", tools_added=[BASE_TOOL, DROP_TOOL], timestamp=0),
        SystemMessage(content="", tools_removed=[ToolReference(name="drop_tool")], timestamp=1),
        SystemMessage(content="", tools_added=[DROP_TOOL], timestamp=2),
        SystemMessage(content="", tools_added=[replace(LATE_TOOL, description="x")], timestamp=3),
        SystemMessage(content="", tools_added=[replace(LATE_TOOL, description="y")], timestamp=4),
    ]

    assert resolve_transcript_tools(messages[:2], "additions") == TranscriptTools(
        request_tools=[BASE_TOOL], folded_system_messages=2
    )
    # Kimi K3 ignores a removed tool that a later tool message declares again, so the re-added
    # drop_tool moves the boundary; the new late_tool after it stays anchored.
    assert resolve_transcript_tools(messages[:4], "additions") == TranscriptTools(
        request_tools=[BASE_TOOL, DROP_TOOL], folded_system_messages=3
    )
    # The redefinition of late_tool moves the boundary to its message.
    assert resolve_transcript_tools(messages, "additions") == TranscriptTools(
        request_tools=[BASE_TOOL, DROP_TOOL, replace(LATE_TOOL, description="y")], folded_system_messages=5
    )
    assert resolve_transcript_tools(messages, "changes") == TranscriptTools(request_tools=[], folded_system_messages=0)
    assert resolve_transcript_tools(messages, "none") == TranscriptTools(
        request_tools=[BASE_TOOL, DROP_TOOL, replace(LATE_TOOL, description="y")], folded_system_messages=5
    )


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


# pi #10940: Kimi tool messages cannot remove tools, so a removal rewrites the prefix once.
@pytest.mark.tonio
async def test_keeps_the_kimi_prefix_across_loads_after_a_removal():
    before_removal, *after_removal = await capture_tool_history(get_builtin_model("kimi-coding", "k3"))

    assert [value["function"]["name"] for value in before_removal["tools"]] == ["base_tool", "drop_tool"]
    for payload in after_removal:
        assert [value["function"]["name"] for value in payload["tools"]] == ["base_tool"]
    assert_growing_prefix([payload["messages"] for payload in after_removal])
    assert [
        message["tools"][0]["function"]["name"] for message in after_removal[-1]["messages"] if "tools" in message
    ] == ["late_tool", "later_tool"]


# pi #10720: Kimi Coding uses Chat Completions so K3-style models keep added tools out of the top-level list.
@pytest.mark.tonio
@pytest.mark.parametrize("model_id", ["k3", "k3-256k", "kimi-for-coding"])
async def test_anchors_additions_for_kimi_coding(model_id):
    payload = await capture_payload(get_builtin_model("kimi-coding", model_id), ADDITION_CONTEXT)

    assert [value["function"]["name"] for value in payload["tools"]] == ["base_tool"]
    tool_message = next(message for message in payload["messages"] if "tools" in message)
    assert [value["function"]["name"] for value in tool_message["tools"]] == ["late_tool"]


# pi #10720: Kimi Code Highspeed rejects tool-bearing system messages.
@pytest.mark.tonio
async def test_keeps_kimi_coding_highspeed_additions_in_top_level_tools():
    payload = await capture_payload(get_builtin_model("kimi-coding", "kimi-for-coding-highspeed"), ADDITION_CONTEXT)

    assert [value["function"]["name"] for value in payload["tools"]] == ["base_tool", "late_tool"]
    assert not any("tools" in message for message in payload["messages"])


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
