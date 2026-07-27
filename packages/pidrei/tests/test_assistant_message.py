"""Mirror of pi coding-agent test/assistant-message.test.ts."""

import re
import time

import pytest

from pidrei.modes.interactive.components import AssistantMessageComponent, UserMessageComponent
from pidrei.modes.interactive.theme import init_theme
from pidrei.utils.ansi import strip_ansi
from pidrei_ai.types import AssistantMessage, TextContent, ThinkingContent, ToolCall, Usage


OSC133_ZONE_START = "\x1b]133;A\x07"
OSC133_ZONE_END = "\x1b]133;B\x07"
OSC133_ZONE_FINAL = "\x1b]133;C\x07"


def create_assistant_message(content, stop_reason: str = "stop") -> AssistantMessage:
    return AssistantMessage(
        content=content,
        api="openai-responses",
        provider="openai",
        model="gpt-4o-mini",
        usage=Usage(),
        stop_reason=stop_reason,
        timestamp=int(time.time() * 1000),
    )


class TestAssistantMessageComponent:
    @pytest.mark.tonio
    async def test_adds_osc_133_zone_markers_to_assistant_messages_without_tool_calls(self):
        await init_theme("dark")

        component = AssistantMessageComponent(create_assistant_message([TextContent(text="hello")]))
        lines = component.render(40)

        assert len(lines) != 0
        assert OSC133_ZONE_START in lines[0]
        assert lines[-1].startswith(OSC133_ZONE_END + OSC133_ZONE_FINAL)

    @pytest.mark.tonio
    async def test_does_not_add_osc_133_zone_markers_when_assistant_message_contains_tool_calls(self):
        await init_theme("dark")

        component = AssistantMessageComponent(
            create_assistant_message(
                [
                    TextContent(text="calling tool"),
                    ToolCall(id="tool-1", name="read", arguments={"path": "file.txt"}),
                ]
            )
        )
        rendered = "\n".join(component.render(60))

        assert OSC133_ZONE_START not in rendered
        assert OSC133_ZONE_END not in rendered
        assert OSC133_ZONE_FINAL not in rendered

    @pytest.mark.tonio
    async def test_renders_length_stops_as_visible_errors(self):
        await init_theme("dark")

        component = AssistantMessageComponent(
            create_assistant_message([ThinkingContent(thinking="private reasoning")], stop_reason="length"),
            True,
        )
        rendered = "\n".join(component.render(80))

        assert "Thinking..." in rendered
        assert "maximum output token limit" in rendered
        assert "response may be incomplete" in rendered

    @pytest.mark.tonio
    async def test_coalesces_adjacent_thinking_blocks_into_one_hidden_thinking_label(self):
        await init_theme("dark")

        component = AssistantMessageComponent(
            create_assistant_message(
                [
                    ThinkingContent(thinking="first thought"),
                    ThinkingContent(thinking=""),
                    ThinkingContent(thinking="second thought"),
                    TextContent(text="answer"),
                ]
            ),
            True,
        )
        rendered = strip_ansi("\n".join(component.render(80)))

        assert len(re.findall(r"Thinking\.\.\.", rendered)) == 1
        assert "answer" in rendered

    @pytest.mark.tonio
    async def test_uses_configured_output_padding_for_text_and_thinking(self):
        await init_theme("dark")

        component = AssistantMessageComponent(
            create_assistant_message(
                [
                    TextContent(text="hello"),
                    ThinkingContent(thinking="reasoning"),
                ]
            ),
            False,
            None,
            "Thinking...",
            1,
        )
        lines = [strip_ansi(line) for line in component.render(80)]

        assert any(" hello" in line for line in lines)
        assert any(" reasoning" in line for line in lines)

        component.set_output_pad(0)
        updated_lines = [strip_ansi(line) for line in component.render(80)]
        assert any(line.startswith("hello") for line in updated_lines)
        assert any(line.startswith("reasoning") for line in updated_lines)

    @pytest.mark.tonio
    async def test_uses_configured_output_padding_for_user_messages(self):
        await init_theme("dark")

        padded_component = UserMessageComponent("hello", None, 1)
        padded_lines = [strip_ansi(line) for line in padded_component.render(40)]
        assert any(line.startswith(" hello") for line in padded_lines)

        unpadded_component = UserMessageComponent("hello", None, 0)
        unpadded_lines = [strip_ansi(line) for line in unpadded_component.render(40)]
        assert any(line.startswith("hello") for line in unpadded_lines)
