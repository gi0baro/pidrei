"""Mirror of pi coding-agent test/bash-execution-width.test.ts.

BashExecutionComponent's collapsed output must respect the render-time
width, not a stale captured width. Regression test for pi#2569.
(tonio-marked: the loader animation spawns timer tasks.)
"""

from types import SimpleNamespace

import pytest

from pidrei.modes.interactive.components import BashExecutionComponent
from pidrei.modes.interactive.theme import init_theme
from pidrei_tui import visible_width


def create_tui_stub():
    return SimpleNamespace(request_render=lambda: None)


@pytest.mark.tonio
async def test_collapsed_preview_lines_respect_render_time_width_not_construction_time_width():
    init_theme(None, False)
    narrow_width = 80

    component = BashExecutionComponent("pwd", create_tui_stub())

    # Add output with long lines that will wrap differently at different widths
    long_line = "x" * 150
    component.append_output(f"{long_line}\n{long_line}\n")

    # Complete the command so it enters collapsed mode
    component.set_complete(0, False)

    # Render at the narrow width (simulating a resize or split pane)
    lines = component.render(narrow_width)

    # Every rendered line must fit within the narrow width
    for i, line in enumerate(lines):
        w = visible_width(line)
        assert w <= narrow_width, f"Line {i} visible_width={w} > {narrow_width}"


@pytest.mark.tonio
async def test_re_computes_lines_when_width_changes_between_renders():
    init_theme(None, False)
    component = BashExecutionComponent("echo hello", create_tui_stub())

    long_line = "abcdefghij" * 20  # 200 chars
    component.append_output(f"{long_line}\n")
    component.set_complete(0, False)

    # First render at width 200
    lines200 = component.render(200)
    for line in lines200:
        assert visible_width(line) <= 200

    # Second render at width 60 (split pane scenario)
    lines60 = component.render(60)
    for i, line in enumerate(lines60):
        w = visible_width(line)
        assert w <= 60, f"Line {i} visible_width={w} > 60"
