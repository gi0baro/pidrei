"""Mirror of pi ai/test/system-message-replay.test.ts."""

from dataclasses import dataclass
from typing import Any

from pidrei_ai.types import (
    AssistantMessage,
    Context,
    SystemMessage,
    TextContent,
    Tool,
    ToolReference,
    TranscriptContext,
    Usage,
    UserMessage,
)
from pidrei_ai.utils.text import get_system_message_text, render_system_message_update
from pidrei_ai.utils.transcript import (
    ToolStateChanges,
    collapse_system_messages,
    declarations_equal,
    get_current_system_message,
    get_current_system_prompt,
    get_tool_state_changes,
    has_non_additive_tool_changes,
    has_tool_redefinitions,
    normalize_context,
)


def tool(name: str, description: str | None = None) -> Tool:
    return Tool(
        name=name,
        description=description if description is not None else f"{name} tool",
        parameters={"type": "object", "properties": {}},
    )


TRANSCRIPT = normalize_context(
    Context(
        messages=[
            SystemMessage(
                content="base",
                sections={"a": "<a>1</a>", "b": "<b>1</b>"},
                tools_added=[tool("first")],
                timestamp=10,
            ),
            UserMessage(content="hello", timestamp=11),
            SystemMessage(content="also do this", timestamp=12),
            AssistantMessage(
                content=[TextContent(text="ok")],
                api="openai-responses",
                provider="openai",
                model="mock",
                usage=Usage(),
                stop_reason="stop",
                timestamp=13,
            ),
            SystemMessage(
                content="",
                sections={"a": "<a>2</a>", "b": None, "c": "<c>1</c>"},
                tools_removed=[ToolReference(name="first")],
                tools_added=[tool("second")],
                timestamp=14,
            ),
        ]
    )
)


def test_replays_content_sections_and_tools_into_one_leading_message():
    current = get_current_system_message(TRANSCRIPT.messages)
    assert current == SystemMessage(
        content="base\n\nalso do this",
        sections={"a": "<a>2</a>", "c": "<c>1</c>"},
        tools_added=[tool("second")],
        timestamp=10,
    )
    assert get_current_system_prompt(TRANSCRIPT.messages) == "base\n\nalso do this\n\n<a>2</a>\n\n<c>1</c>"


def test_collapse_keeps_only_non_system_messages_after_the_replayed_head():
    collapsed = collapse_system_messages(TRANSCRIPT)
    assert [message.role for message in collapsed.messages] == ["system", "user", "assistant"]
    assert collapse_system_messages(collapsed) == collapsed


def test_replay_of_a_transcript_without_system_messages_is_empty():
    context = normalize_context(Context(messages=[UserMessage(content="hi", timestamp=1)]))
    assert get_current_system_message(context.messages) is None
    assert get_current_system_prompt(context.messages) == ""
    assert collapse_system_messages(context).messages == context.messages


def test_a_late_full_patch_on_a_transcript_without_a_leading_message_replays_as_the_prompt():
    context: TranscriptContext = normalize_context(
        Context(
            messages=[
                UserMessage(content="old session", timestamp=1),
                SystemMessage(
                    content="",
                    sections={"preamble": "You are pi."},
                    tools_added=[tool("x")],
                    timestamp=2,
                ),
            ]
        )
    )
    assert get_current_system_prompt(context.messages) == "You are pi."
    head = collapse_system_messages(context).messages[0]
    assert head.role == "system"
    assert head.tools_added == [tool("x")]


def test_renders_complete_prompts_and_framed_updates():
    leading = TRANSCRIPT.messages[0]
    update = TRANSCRIPT.messages[4]
    assert leading.role == "system" and update.role == "system"
    assert get_system_message_text(leading) == "base\n\n<a>1</a>\n\n<b>1</b>"
    assert render_system_message_update(update) == (
        'Updated system prompt section "a":\n\n<a>2</a>'
        "\n\n"
        'Removed system prompt section "b".'
        "\n\n"
        'Updated system prompt section "c":\n\n<c>1</c>'
    )


def test_normalizes_the_legacy_prompt_and_tool_fields_into_a_leading_system_message():
    messages = [UserMessage(content="hi", timestamp=1)]
    assert normalize_context(Context(messages=messages)) == TranscriptContext(messages=messages)
    assert normalize_context(Context(system_prompt="", tools=[], messages=messages)) == TranscriptContext(
        messages=messages
    )
    assert normalize_context(Context(system_prompt="be brief", tools=[tool("a")], messages=messages)).messages == [
        SystemMessage(content="be brief", tools_added=[tool("a")], timestamp=0),
        *messages,
    ]


@dataclass
class _ExecutableTool:
    name: str
    description: str
    parameters: dict[str, Any]
    constrained_sampling: Any = None

    async def execute(self) -> None:
        return None


def test_compares_tool_declarations_without_executable_or_undefined_fields():
    base = tool("a")
    executable = _ExecutableTool(name=base.name, description=base.description, parameters=base.parameters)
    assert declarations_equal(executable, tool("a")) is True
    assert declarations_equal(tool("a"), tool("a", "changed")) is False
    constrained_off = Tool(
        name=base.name, description=base.description, parameters=base.parameters, constrained_sampling=False
    )
    assert declarations_equal(tool("a"), constrained_off) is False


def test_tool_state_changes_treat_changed_definitions_as_removal_plus_addition():
    changes = get_tool_state_changes([tool("a"), tool("b")], [tool("b", "changed"), tool("c")])
    assert changes == ToolStateChanges(
        tools_added=[tool("b", "changed"), tool("c")],
        tools_removed=[ToolReference(name="a"), ToolReference(name="b")],
    )
    assert get_tool_state_changes([tool("a")], [tool("a")]) == ToolStateChanges(tools_added=[], tools_removed=[])


def test_detects_non_additive_tool_history_and_redefinitions():
    assert has_non_additive_tool_changes(TRANSCRIPT.messages) is True
    assert has_tool_redefinitions(TRANSCRIPT.messages) is False
    additive = normalize_context(
        Context(
            messages=[
                SystemMessage(content="", tools_added=[tool("a")], timestamp=1),
                SystemMessage(content="", tools_added=[tool("b")], timestamp=2),
            ]
        )
    )
    assert has_non_additive_tool_changes(additive.messages) is False
    redeclared = normalize_context(
        Context(
            messages=[
                SystemMessage(content="", tools_added=[tool("a")], timestamp=1),
                SystemMessage(content="", tools_added=[tool("a", "changed")], timestamp=2),
            ]
        )
    )
    assert has_non_additive_tool_changes(redeclared.messages) is True
    assert has_tool_redefinitions(redeclared.messages) is True
