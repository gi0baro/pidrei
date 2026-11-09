"""Mirror of pi coding-agent test/extension-keybinding-hints.test.ts.

Issue #4748: in pi an extension that imports keyText/keyHint can resolve its
own package copy, which does not see the host's keybindings. Python loads one
copy, but the API is pi's: the host passes its own key_text/key_hint to
renderers, as `"keyText"`/`"keyHint"` in the context and options.
"""

import os
import time
from types import SimpleNamespace

import pytest

from pidrei.core.extensions.types import ToolDefinition
from pidrei.core.keybindings import KeybindingsManager
from pidrei.core.messages import CustomMessage
from pidrei.modes.interactive.components import ToolExecutionComponent
from pidrei.modes.interactive.components.custom_entry import CustomEntryComponent
from pidrei.modes.interactive.components.custom_message import CustomMessageComponent
from pidrei.modes.interactive.theme import init_theme
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import Text, set_keybindings


EXPECTED = "ctrl+e | ctrl+e to expand"


def hint_text(hints: dict) -> str:
    return f"{hints['keyText']('app.tools.expand')} | {strip_ansi(hints['keyHint']('app.tools.expand', 'to expand'))}"


@pytest.fixture(autouse=True)
async def _setup():
    await init_theme("dark")
    # A user override proves the hints read the host's configured keybindings, not the defaults.
    set_keybindings(KeybindingsManager.in_memory({"app.tools.expand": "ctrl+e"}))


def test_tool_renderers_receive_key_text_and_key_hint_in_the_render_context():
    seen: list[str] = []

    async def execute(*_args):
        return None

    def render_call(_args, _theme, context):
        seen.append(hint_text(context))
        return Text("call", 0, 0)

    tool_definition = ToolDefinition(
        name="custom_tool",
        label="custom_tool",
        description="custom tool",
        parameters={},
        execute=execute,
        render_call=render_call,
    )
    tui = SimpleNamespace(request_render=lambda: None)
    ToolExecutionComponent("custom_tool", "tool-1", {}, {}, tool_definition, tui, os.getcwd()).render(80)

    assert seen[-1] == EXPECTED


def test_message_renderers_receive_key_text_and_key_hint_in_the_render_options():
    seen: list[str] = []
    message = CustomMessage(custom_type="test", content="custom", display=True, timestamp=int(time.time() * 1000))

    def renderer(_message, options, _theme):
        seen.append(hint_text(options))
        return Text("custom", 0, 0)

    CustomMessageComponent(message, renderer)

    assert seen == [EXPECTED]


def test_entry_renderers_receive_key_text_and_key_hint_in_the_render_options():
    seen: list[str] = []
    entry = {
        "type": "custom",
        "id": "entry-1",
        "parentId": None,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "customType": "test",
        "data": None,
    }

    def renderer(_entry, options, _theme):
        seen.append(hint_text(options))
        return Text("custom", 0, 0)

    CustomEntryComponent(entry, renderer)

    assert seen == [EXPECTED]
