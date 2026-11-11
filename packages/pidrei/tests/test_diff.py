"""Mirror of pi coding-agent test/diff.test.ts."""

import pytest

from pidrei.modes.interactive.components.diff import render_diff
from pidrei.modes.interactive.theme import init_theme, theme
from pidrei.utils.ansi import strip_ansi


@pytest.fixture(autouse=True)
async def _setup():
    await init_theme("dark")


@pytest.mark.parametrize(
    ("diff", "expected"),
    [
        pytest.param(
            "-83   await commitRecordingChunk({\n+83   const result = await commitRecordingChunk({",
            ["-83   await commitRecordingChunk({", "+83   const result = await commitRecordingChunk({"],
            id="text is inserted at the start of a line",
        ),
        pytest.param(
            "-12     const value = oldValue\n+12     const value = newValue",
            ["-12     const value = oldValue", "+12     const value = newValue"],
            id="indented text is replaced",
        ),
    ],
)
def test_preserves_indentation(diff, expected):
    rendered = render_diff(diff)

    assert strip_ansi(rendered).split("\n") == expected


def test_does_not_highlight_shared_indentation_as_an_insertion():
    rendered = render_diff("-83   await commitRecordingChunk({\n+83   const result = await commitRecordingChunk({")
    inverse_insertion = theme.inverse("const result = ")

    assert inverse_insertion in rendered
    assert rendered.index("+83   ") < rendered.index(inverse_insertion)
