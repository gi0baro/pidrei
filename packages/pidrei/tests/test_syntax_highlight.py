"""Partial mirror of pi coding-agent test/syntax-highlight.test.ts: the
"theme syntax highlighting" cases. The renderer block specs highlight.js's
HTML output, which pygments has no counterpart for; its one case that only
asserts `getLanguageFromPath` is mirrored here, in the theme-level block."""

import pytest

from pidrei.modes.interactive.theme import (
    get_language_from_path,
    get_markdown_theme,
    highlight_code,
    init_theme,
    theme,
)
from pidrei_tui import reset_capabilities_cache, set_capabilities


@pytest.fixture(autouse=True)
def _reset_capabilities(request):
    request.addfinalizer(reset_capabilities_cache)


class TestThemeSyntaxHighlighting:
    def test_maps_file_extensions_of_eagerly_loaded_languages_to_their_language(self):
        assert get_language_from_path("lib/main.dart") == "dart"
        assert get_language_from_path("flake.nix") == "nix"
        assert get_language_from_path("script.groovy") == "groovy"
        assert get_language_from_path("build.gradle") == "groovy"
        assert get_language_from_path("src/Main.kt") == "kotlin"

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
            f"{theme.get_fg_ansi('text')}after\x1b[39m",
        ]

        assert highlight_code(code, "python") == expected
        assert get_markdown_theme()["highlightCode"](code, "python") == expected

    # Regression for #11092.
    @pytest.mark.tonio
    async def test_colors_unscoped_highlighted_code_with_the_theme_foreground(self):
        await init_theme("light")

        lines = highlight_code("plainIdentifier\nconst answer = 42;", "typescript")
        text = theme.get_fg_ansi("text")

        assert lines[0] == f"{text}plainIdentifier\x1b[39m"
        # pi asserts `<text> answer = ` as one run: highlight.js leaves the
        # identifier and the spaces around it as one unscoped stretch. pygments
        # gives the space and `answer` a token each, and scopes `=` as an
        # operator, so the identifier is asserted on its own.
        assert f"{text}answer\x1b[39m" in lines[1]
