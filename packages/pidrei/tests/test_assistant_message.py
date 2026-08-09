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
    async def test_renders_length_stops_with_neutral_truncation_wording(self):
        await init_theme("dark")

        component = AssistantMessageComponent(
            create_assistant_message([ThinkingContent(thinking="private reasoning")], stop_reason="length"),
            True,
        )
        rendered = "\n".join(component.render(80))

        assert "Thinking..." in rendered
        assert "Response was truncated before completion." in rendered

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
    async def test_chains_markdown_transformers_in_registration_order(self):
        await init_theme("dark")
        calls: list[str] = []
        contexts: list[dict] = []

        def formula(markdown, context):
            calls.append("formula")
            contexts.append(context)
            return markdown.replace("$x^2$", "x²")

        def suffix(markdown, _context):
            calls.append("suffix")
            return f"{markdown} Done."

        message = create_assistant_message([TextContent(text="The result is $x^2$.")])
        component = AssistantMessageComponent(message, False, None, "Thinking...", 1, [formula, suffix])

        assert "The result is x². Done." in strip_ansi("\n".join(component.render(80)))
        assert calls == ["formula", "suffix"]
        assert contexts == [{"messageType": "assistant", "isStreaming": False, "availableWidth": 78}]

    @pytest.mark.tonio
    async def test_identifies_partial_assistant_markdown_as_streaming(self):
        await init_theme("dark")
        streaming_states: list[bool] = []

        def transformer(markdown, context):
            streaming_states.append(context["isStreaming"])
            return markdown if context["isStreaming"] else f"{markdown} transformed"

        message = create_assistant_message([TextContent(text="partial")])
        component = AssistantMessageComponent(None, False, None, "Thinking...", 1, [transformer])

        component.update_content(message, True)
        assert "transformed" not in strip_ansi("\n".join(component.render(80)))

        component.update_content(message, False)
        assert "partial transformed" in strip_ansi("\n".join(component.render(80)))
        assert streaming_states == [True, False]

    @pytest.mark.tonio
    async def test_reapplies_markdown_transformers_when_available_width_changes(self):
        await init_theme("dark")
        available_widths: list[int] = []

        def transformer(markdown, context):
            available_widths.append(context["availableWidth"])
            return f"{markdown} ({context['availableWidth']})"

        component = AssistantMessageComponent(
            create_assistant_message([TextContent(text="answer")]),
            False,
            None,
            "Thinking...",
            1,
            [transformer],
        )

        assert "answer (78)" in strip_ansi("\n".join(component.render(80)))
        component.render(80)
        assert "answer (58)" in strip_ansi("\n".join(component.render(60)))
        assert available_widths == [78, 58]

    @pytest.mark.tonio
    async def test_continues_the_markdown_transformer_chain_when_a_transformer_raises(self):
        await init_theme("dark")
        calls: list[str] = []

        def first(markdown, _context):
            calls.append("first")
            return markdown.replace("still", "remains")

        def broken(_markdown, _context):
            calls.append("throw")
            raise RuntimeError("broken transformer")

        def last(markdown, _context):
            calls.append("last")
            return f"{markdown} after error"

        component = AssistantMessageComponent(
            create_assistant_message([TextContent(text="still visible")]),
            False,
            None,
            "Thinking...",
            1,
            [first, broken, last],
        )

        assert "remains visible after error" in strip_ansi("\n".join(component.render(80)))
        assert calls == ["first", "throw", "last"]

    @pytest.mark.tonio
    async def test_transforms_text_and_thinking_markdown_without_mutating_the_original_message(self):
        await init_theme("dark")
        message = create_assistant_message([TextContent(text="answer"), ThinkingContent(thinking="reasoning")])
        component = AssistantMessageComponent(
            message,
            False,
            None,
            "Thinking...",
            1,
            [lambda markdown, context: f"{context['messageType']}:{markdown}"],
        )

        rendered = strip_ansi("\n".join(component.render(80)))
        assert "assistant:answer" in rendered
        assert "assistant-thinking:reasoning" in rendered
        assert message.content == [TextContent(text="answer"), ThinkingContent(thinking="reasoning")]

    @pytest.mark.tonio
    async def test_uses_configured_output_padding_for_user_messages(self):
        await init_theme("dark")

        padded_component = UserMessageComponent("hello", None, 1)
        padded_lines = [strip_ansi(line) for line in padded_component.render(40)]
        assert any(line.startswith(" hello") for line in padded_lines)

        unpadded_component = UserMessageComponent("hello", None, 0)
        unpadded_lines = [strip_ansi(line) for line in unpadded_component.render(40)]
        assert any(line.startswith("hello") for line in unpadded_lines)
