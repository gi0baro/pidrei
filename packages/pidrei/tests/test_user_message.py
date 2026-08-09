"""Mirror of pi coding-agent test/user-message.test.ts."""

import pytest

from pidrei.modes.interactive.components import UserMessageComponent
from pidrei.modes.interactive.theme import init_theme
from pidrei.utils.ansi import strip_ansi


OSC133_ZONE_START = "\x1b]133;A\x07"
OSC133_ZONE_END = "\x1b]133;B\x07"
OSC133_ZONE_FINAL = "\x1b]133;C\x07"
BG_RESET = "\x1b[49m"


class TestUserMessageComponent:
    @pytest.mark.tonio
    async def test_keeps_user_message_height_stable_while_moving_closing_osc_markers_off_line_end(self):
        await init_theme("dark")

        component = UserMessageComponent("hello")
        lines = component.render(20)

        assert len(lines) == 3
        assert OSC133_ZONE_START in lines[0]
        assert lines[0].endswith(BG_RESET)
        assert OSC133_ZONE_END not in lines[0]
        assert "hello" in lines[1]
        assert lines[2].startswith(OSC133_ZONE_END + OSC133_ZONE_FINAL)
        assert lines[2].endswith(BG_RESET)

    @pytest.mark.tonio
    async def test_chains_markdown_transformers_with_user_message_context(self):
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

        component = UserMessageComponent("The input is $x^2$.", None, 1, [formula, suffix])

        assert "The input is x². Done." in strip_ansi("\n".join(component.render(80)))
        assert calls == ["formula", "suffix"]
        assert contexts == [{"messageType": "user", "isStreaming": False, "availableWidth": 78}]

    @pytest.mark.tonio
    async def test_reapplies_markdown_transformers_when_invalidated(self):
        await init_theme("dark")
        state = {"suffix": "before"}
        component = UserMessageComponent(
            "Message", None, 1, [lambda markdown, _context: f"{markdown} {state['suffix']}"]
        )

        assert "Message before" in strip_ansi("\n".join(component.render(80)))

        state["suffix"] = "after"
        component.invalidate()

        assert "Message after" in strip_ansi("\n".join(component.render(80)))
