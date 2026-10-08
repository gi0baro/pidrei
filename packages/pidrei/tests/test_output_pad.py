"""Mirror of pi coding-agent test/output-pad.test.ts."""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pidrei.core.extensions.types import ToolDefinition
from pidrei.core.messages import CompactionSummaryMessage
from pidrei.core.tools.edit import create_edit_tool_definition
from pidrei.core.tools.renderers import with_built_in_renderers
from pidrei.modes.interactive.components import (
    BashExecutionComponent,
    CompactionSummaryMessageComponent,
    ToolExecutionComponent,
)
from pidrei.modes.interactive.theme import init_theme
from pidrei.utils.ansi import strip_ansi
from pidrei_tui.tui_main_screen import TuiMainScreen


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tui" / "tests"))
from virtual_terminal import VirtualTerminal


async def _execute(_tool_call_id, _params, cancel=None, on_update=None, ctx=None):
    return SimpleNamespace(content=[{"type": "text", "text": "ok"}], details={})


TOOL = ToolDefinition(
    name="custom_tool", label="custom_tool", description="custom tool", parameters={}, execute=_execute
)

# pi's `/[\w$(]/`: JavaScript's `\w` is ASCII-only.
_HAS_TEXT = re.compile(r"[\w$(]", re.ASCII)


def render_lines(component) -> list[str]:
    """Text lines without ANSI codes or trailing fill. Blank lines and full-width borders are skipped."""
    return [line for line in (strip_ansi(line).rstrip() for line in component.render(60)) if _HAS_TEXT.search(line)]


def create_tool(ui, definition, output_pad: int) -> ToolExecutionComponent:
    component = ToolExecutionComponent("custom_tool", "id", {}, {"outputPad": output_pad}, definition, ui, "/")
    component.update_result({"content": [{"type": "text", "text": "ok"}], "isError": False})
    return component


def create_bash_execution(ui, output_pad: int):
    component = BashExecutionComponent("pwd", ui, False, output_pad)
    component.append_output("/tmp")
    component.set_complete(1, False)
    return component


def create_self_rendered_edit_result(ui, output_pad: int):
    # pi's createEditToolDefinition carries its renderers; here callers merge them in.
    component = ToolExecutionComponent(
        "edit",
        "id",
        {"path": "file.txt", "edits": [{"oldText": "old", "newText": "new"}]},
        {"outputPad": output_pad},
        with_built_in_renderers("edit", create_edit_tool_definition("/")),
        ui,
        "/",
    )
    component.update_result({"content": [{"type": "text", "text": "Could not find old text"}], "isError": True})
    return component


def create_compaction_summary(_ui, output_pad: int):
    return CompactionSummaryMessageComponent(
        CompactionSummaryMessage(summary="summary", tokens_before=10, timestamp=0), None, output_pad
    )


COMPONENTS = {
    "bash execution": create_bash_execution,
    "tool execution": lambda ui, output_pad: create_tool(ui, TOOL, output_pad),
    "tool execution without a definition": lambda ui, output_pad: create_tool(ui, None, output_pad),
    "self-rendered edit result": create_self_rendered_edit_result,
    "compaction summary": create_compaction_summary,
}


@pytest.mark.tonio
@pytest.mark.parametrize("name", COMPONENTS)
async def test_renders_at_output_pad_0_and_1(name):
    await init_theme("dark")
    ui = TuiMainScreen(VirtualTerminal(80, 24))

    component = COMPONENTS[name](ui, 0)
    lines = render_lines(component)
    assert lines
    assert [line for line in lines if line.startswith(" ")] == []
    component.set_output_pad(1)
    assert render_lines(component) == [f" {line}" for line in lines]
