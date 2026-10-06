"""Partial mirror of pi coding-agent test/syntax-highlight.test.ts: the
"theme syntax highlighting" cases. The renderer block specs highlight.js's
HTML output, which pygments has no counterpart for."""

import pytest

from pidrei.modes.interactive.theme import get_markdown_theme, highlight_code, init_theme, theme
from pidrei_tui import reset_capabilities_cache, set_capabilities


@pytest.fixture(autouse=True)
def _reset_capabilities(request):
    request.addfinalizer(reset_capabilities_cache)


class TestThemeSyntaxHighlighting:
    # #10143
    @pytest.mark.tonio
    async def test_colors_each_line_of_python_docstrings_independently(self):
        set_capabilities({"images": None, "trueColor": True, "hyperlinks": False})
        await init_theme("dark")
        code = '"""\nline one\n\nline two\n"""\nafter'
        ansi = theme.get_fg_ansi("syntaxString")
        expected = [
            f'{ansi}"""\x1b[39m',
            f"{ansi}line one\x1b[39m",
            "",
            f"{ansi}line two\x1b[39m",
            f'{ansi}"""\x1b[39m',
            "after",
        ]

        assert highlight_code(code, "python") == expected
        assert get_markdown_theme()["highlightCode"](code, "python") == expected
