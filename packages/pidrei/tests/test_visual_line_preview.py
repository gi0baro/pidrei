"""pidrei-only: VisualLinePreview (pi's visual-truncate.ts).

pi covers it through the codemode and MCP result renderers; here the bash and
codemode renderers use it, and the MCP renderer ports later.
"""

import pytest

from pidrei.modes.interactive.components.visual_truncate import VisualLinePreview
from pidrei_tui import Text, visible_width


LONG_LINE = " ".join(f"word{i}" for i in range(40))


def hint(hidden: int) -> str:
    return f"[{hidden} hidden]"


def wrapped(text: str, width: int) -> list[str]:
    return Text(text, 0, 0).render(width)


@pytest.mark.parametrize(("keep", "expected"), [("end", "hint first"), ("start", "hint last")])
def test_limits_one_long_line_to_wrapped_lines_with_the_hint_on_the_hidden_side(keep, expected):
    all_lines = wrapped(LONG_LINE, 20)
    assert len(all_lines) > 3
    hidden = len(all_lines) - 3

    lines = VisualLinePreview(text=LONG_LINE, max_visual_lines=3, keep=keep, format_hint=hint).render(20)

    if expected == "hint first":
        assert lines == [hint(hidden), *all_lines[-3:]]
    else:
        assert lines == [*all_lines[:3], hint(hidden)]


def test_shows_short_output_without_a_hint():
    text = "one\ntwo"
    preview = VisualLinePreview(text=text, max_visual_lines=3, keep="end", format_hint=hint)
    assert preview.render(20) == wrapped(text, 20)


def test_truncates_the_hint_to_the_width():
    preview = VisualLinePreview(
        text=LONG_LINE, max_visual_lines=1, keep="end", format_hint=lambda hidden: "x" * 50 + str(hidden)
    )
    first = preview.render(20)[0]
    assert visible_width(first) == 20
    assert "..." in first


def test_rewraps_when_the_width_changes():
    preview = VisualLinePreview(text=LONG_LINE, max_visual_lines=3, keep="start", format_hint=hint)
    narrow = preview.render(20)
    wide = preview.render(60)
    assert wide[:3] == wrapped(LONG_LINE, 60)[:3]
    assert wide != narrow
