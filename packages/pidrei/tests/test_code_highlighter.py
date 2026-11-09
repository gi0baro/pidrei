"""Mirror of pi coding-agent test/code-highlighter.test.ts.

pi spies on syntax-highlight's `highlight`; theme.py imports it by name, so the
spy wraps the theme module's binding. pi also checks that loading the deferred
highlight.js grammars changes the result; pygments has no such registry (see
`utils/syntax_highlight.py`), so that step has no counterpart, and the theme
change is checked against the previous result under the old theme instead.
"""

import importlib
import threading
from types import SimpleNamespace

import pytest

from pidrei.core.tools.renderers.write import write_renderers
from pidrei.extensions.codemode import CodemodeToolDetails
from pidrei.extensions.codemode.renderer import CODEMODE_RENDERERS
from pidrei.modes.interactive.components import ToolExecutionComponent
from pidrei.modes.interactive.theme import CodeHighlighter, highlight_code, init_theme
from pidrei_ai.types import TextContent

from .ui_timer_helpers import manual_ui_timers


# The package's `theme` attribute is the theme object, which shadows the submodule's name.
theme_module = importlib.import_module("pidrei.modes.interactive.theme.theme")

# The elapsed-time tick rebuilds the row under the UI state lock.
TUI = SimpleNamespace(request_render=lambda: None, state_lock=threading.RLock())


@pytest.fixture(autouse=True)
async def _theme():
    await init_theme("dark")
    yield
    await init_theme("dark")


@pytest.fixture
def highlight_calls(monkeypatch) -> list[str]:
    calls: list[str] = []
    real = theme_module.highlight

    def counting(code, *args, **kwargs):
        calls.append(code)
        return real(code, *args, **kwargs)

    monkeypatch.setattr(theme_module, "highlight", counting)
    return calls


@pytest.mark.tonio
async def test_reuses_its_last_result_until_the_code_or_the_theme_change():
    code = 'const page = html`<div class="x">ok</div>`;'
    highlighter = CodeHighlighter()
    first = highlighter.highlight(code, "javascript")
    assert highlighter.highlight(code, "javascript") is first
    assert highlighter.highlight("const a = 1;", "javascript") is not first

    dark = highlighter.highlight(code, "javascript")
    await init_theme("light")
    light = highlighter.highlight(code, "javascript")
    assert light == tuple(highlight_code(code, "javascript"))
    assert light != dark


@pytest.mark.tonio
async def test_codemode_highlights_the_script_once_across_progress_timer_ticks_expanding_and_completion(
    highlight_calls,
):
    with manual_ui_timers() as timers:
        row = ToolExecutionComponent("codemode", "c", {"code": "await read({})"}, {}, CODEMODE_RENDERERS, TUI, "/")
        row.set_args_complete()
        row.mark_execution_started()
        row.update_result({"content": [], "details": CodemodeToolDetails(calls=[]), "isError": False}, True)
        # pi advances fake timers by 2 s: two ticks of the elapsed-time interval.
        [(_delay, _cancelled, tick)] = timers.scheduled
        tick()
        tick()
        row.set_expanded(True)
        row.update_result(
            {"content": [TextContent(text="done")], "details": CodemodeToolDetails(calls=[]), "isError": False}, False
        )
        row.render(80)
    assert len(highlight_calls) == 1


@pytest.mark.tonio
async def test_write_highlights_unchanged_first_lines_and_a_complete_file_only_once(highlight_calls):
    content = "\n".join(f"const v{i} = {i};" for i in range(60))
    row = ToolExecutionComponent("write", "w", {"path": "a.ts", "content": content}, {}, write_renderers, TUI, "/")
    row.update_args({"path": "a.ts", "content": f"{content} //"})
    row.render(80)
    highlight_calls.clear()
    # Only the changed last line is highlighted again.
    row.update_args({"path": "a.ts", "content": f"{content} // more"})
    row.render(80)
    assert len(highlight_calls) == 1

    row.set_args_complete()
    row.set_expanded(True)
    row.set_expanded(False)
    row.render(80)
    assert len(highlight_calls) == 2
