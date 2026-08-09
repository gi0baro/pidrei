"""Mirror of pi tui test/markdown.test.ts."""

import re

import pytest

from pidrei_tui.components.markdown import Markdown
from pidrei_tui.terminal_image import reset_capabilities_cache, set_capabilities
from pidrei_tui.tui_main_screen import TuiMainScreen

from .chalk_like import chalk
from .themes import default_markdown_theme
from .virtual_terminal import VirtualTerminal


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(line: str) -> str:
    return _ANSI_RE.sub("", line)


# Lists


def test_should_render_simple_nested_list():
    markdown = Markdown(
        "- Item 1\n  - Nested 1.1\n  - Nested 1.2\n- Item 2",
        0,
        0,
        default_markdown_theme,
    )

    lines = markdown.render(80)

    # Check that we have content
    assert len(lines) > 0

    # Strip ANSI codes for checking
    plain_lines = [strip_ansi(line) for line in lines]

    # Check structure
    assert any("- Item 1" in line for line in plain_lines)
    assert any("    - Nested 1.1" in line for line in plain_lines)
    assert any("    - Nested 1.2" in line for line in plain_lines)
    assert any("- Item 2" in line for line in plain_lines)


def test_should_render_deeply_nested_list():
    markdown = Markdown(
        "- Level 1\n  - Level 2\n    - Level 3\n      - Level 4",
        0,
        0,
        default_markdown_theme,
    )

    lines = markdown.render(80)
    plain_lines = [strip_ansi(line) for line in lines]

    # Check proper indentation
    assert any("- Level 1" in line for line in plain_lines)
    assert any("    - Level 2" in line for line in plain_lines)
    assert any("        - Level 3" in line for line in plain_lines)
    assert any("            - Level 4" in line for line in plain_lines)


def test_should_render_ordered_nested_list():
    markdown = Markdown(
        "1. First\n   1. Nested first\n   2. Nested second\n2. Second",
        0,
        0,
        default_markdown_theme,
    )

    lines = markdown.render(80)
    plain_lines = [strip_ansi(line) for line in lines]

    assert any("1. First" in line for line in plain_lines)
    assert any("    1. Nested first" in line for line in plain_lines)
    assert any("    2. Nested second" in line for line in plain_lines)
    assert any("2. Second" in line for line in plain_lines)


def test_should_normalize_ordered_list_markers_by_default():
    markdown = Markdown("1. alpha\n1. beta\n1. gamma", 0, 0, default_markdown_theme)

    lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

    assert lines == ["1. alpha", "2. beta", "3. gamma"]


def test_should_preserve_source_list_markers_when_configured():
    markdown = Markdown(
        "  4. forth\n  3. third\n\n10) ten\n7) seven\n\n+ plus\n* star\n- minus\n+",
        0,
        0,
        default_markdown_theme,
        None,
        {"preserveOrderedListMarkers": True},
    )

    lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

    assert lines == [
        "4. forth",
        "3. third",
        "",
        "10) ten",
        "7) seven",
        "",
        "+ plus",
        "* star",
        "- minus",
        "+",
    ]


def test_should_render_mixed_ordered_and_unordered_nested_lists():
    markdown = Markdown(
        "1. Ordered item\n   - Unordered nested\n   - Another nested\n2. Second ordered\n   - More nested",
        0,
        0,
        default_markdown_theme,
    )

    lines = markdown.render(80)
    plain_lines = [strip_ansi(line) for line in lines]

    assert any("1. Ordered item" in line for line in plain_lines)
    assert any("    - Unordered nested" in line for line in plain_lines)
    assert any("2. Second ordered" in line for line in plain_lines)


def test_should_render_blank_lines_between_loose_list_items():
    markdown = Markdown(
        "1. Lorem ipsum dolor sit amet.\n"
        "\n"
        "   Ut enim ad minim veniam.\n"
        "\n"
        "2. Duis aute irure dolor.\n"
        "\n"
        "   Excepteur sint occaecat cupidatat.\n"
        "\n"
        "3. Beep boop",
        0,
        0,
        default_markdown_theme,
    )

    lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

    assert lines == [
        "1. Lorem ipsum dolor sit amet.",
        "",
        "   Ut enim ad minim veniam.",
        "",
        "2. Duis aute irure dolor.",
        "",
        "   Excepteur sint occaecat cupidatat.",
        "",
        "3. Beep boop",
    ]


def test_should_render_task_list_markers():
    markdown = Markdown("- [ ] beep\n- [x] boop", 0, 0, default_markdown_theme)

    lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

    assert lines == ["- [ ] beep", "- [x] boop"]


def test_should_maintain_numbering_when_code_blocks_are_not_indented_llm_output():
    # When code blocks aren't indented, marked parses each item as a separate list.
    # We use token.start to preserve the original numbering.
    markdown = Markdown(
        "1. First item\n"
        "\n"
        "```typescript\n"
        "// code block\n"
        "```\n"
        "\n"
        "2. Second item\n"
        "\n"
        "```typescript\n"
        "// another code block\n"
        "```\n"
        "\n"
        "3. Third item",
        0,
        0,
        default_markdown_theme,
    )

    lines = markdown.render(80)
    plain_lines = [strip_ansi(line).strip() for line in lines]

    # Find all lines that start with a number and period
    numbered_lines = [line for line in plain_lines if re.match(r"^\d+\.", line)]

    # Should have 3 numbered items
    assert len(numbered_lines) == 3, f"Expected 3 numbered items, got: {', '.join(numbered_lines)}"

    # Check the actual numbers
    assert numbered_lines[0].startswith("1."), f'First item should be "1.", got: {numbered_lines[0]}'
    assert numbered_lines[1].startswith("2."), f'Second item should be "2.", got: {numbered_lines[1]}'
    assert numbered_lines[2].startswith("3."), f'Third item should be "3.", got: {numbered_lines[2]}'


def test_should_indent_wrapped_unordered_list_lines():
    markdown = Markdown("- alpha beta gamma delta epsilon", 0, 0, default_markdown_theme)

    lines = [strip_ansi(line).rstrip() for line in markdown.render(20)]

    assert lines == ["- alpha beta gamma", "  delta epsilon"]


def test_should_indent_wrapped_ordered_list_lines():
    markdown = Markdown("1. alpha beta gamma delta epsilon", 0, 0, default_markdown_theme)

    lines = [strip_ansi(line).rstrip() for line in markdown.render(20)]

    assert lines == ["1. alpha beta gamma", "   delta epsilon"]


def test_should_indent_wrapped_ordered_list_lines_with_multi_digit_markers():
    markdown = Markdown("10. alpha beta gamma delta epsilon", 0, 0, default_markdown_theme)

    lines = [strip_ansi(line).rstrip() for line in markdown.render(21)]

    assert lines == ["10. alpha beta gamma", "    delta epsilon"]


def test_should_indent_wrapped_nested_list_lines():
    markdown = Markdown("- parent\n  - alpha beta gamma delta epsilon", 0, 0, default_markdown_theme)

    lines = [strip_ansi(line).rstrip() for line in markdown.render(24)]

    assert lines == ["- parent", "    - alpha beta gamma", "      delta epsilon"]


def test_should_indent_wrapped_nested_list_lines_under_ordered_parents():
    markdown = Markdown("1. parent\n   - alpha beta gamma delta epsilon", 0, 0, default_markdown_theme)

    lines = [strip_ansi(line).rstrip() for line in markdown.render(24)]

    assert lines == ["1. parent", "    - alpha beta gamma", "      delta epsilon"]


def test_should_render_and_wrap_blockquotes_inside_list_items():
    markdown = Markdown("- > alpha beta gamma delta epsilon zeta", 0, 0, default_markdown_theme)

    lines = [strip_ansi(line).rstrip() for line in markdown.render(24)]

    assert lines == ["- │ alpha beta gamma", "  │ delta epsilon zeta"]


def test_should_render_and_wrap_code_blocks_inside_list_items():
    markdown = Markdown(
        "- ```ts\n  alpha beta gamma delta epsilon zeta\n  ```",
        0,
        0,
        default_markdown_theme,
    )

    lines = [strip_ansi(line).rstrip() for line in markdown.render(24)]

    assert lines == ["- ```ts", "    alpha beta gamma", "  delta epsilon zeta", "  ```"]


# Tables


def test_should_render_simple_table():
    markdown = Markdown(
        "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |",
        0,
        0,
        default_markdown_theme,
    )

    lines = markdown.render(80)
    plain_lines = [strip_ansi(line) for line in lines]

    # Check table structure
    assert any("Name" in line for line in plain_lines)
    assert any("Age" in line for line in plain_lines)
    assert any("Alice" in line for line in plain_lines)
    assert any("Bob" in line for line in plain_lines)
    # Check for table borders
    assert any("│" in line for line in plain_lines)
    assert any("─" in line for line in plain_lines)


def test_should_render_row_dividers_between_data_rows():
    markdown = Markdown(
        "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |",
        0,
        0,
        default_markdown_theme,
    )

    lines = markdown.render(80)
    plain_lines = [strip_ansi(line) for line in lines]
    divider_lines = [line for line in plain_lines if "┼" in line]

    assert len(divider_lines) == 2, "Expected header + row divider"


def test_should_keep_column_width_at_least_the_longest_word():
    longest_word = "superlongword"
    markdown = Markdown(
        f"| Column One | Column Two |\n| --- | --- |\n| {longest_word} short | otherword |\n| small | tiny |",
        0,
        0,
        default_markdown_theme,
    )

    lines = markdown.render(32)
    plain_lines = [strip_ansi(line) for line in lines]
    data_line = next((line for line in plain_lines if longest_word in line), None)
    assert data_line, "Expected data row containing longest word"

    segments = data_line.split("│")[1:-1]
    first_segment = segments[0]
    assert first_segment, "Expected first column segment"
    first_column_width = len(first_segment) - 2

    assert first_column_width >= len(longest_word), (
        f"Expected first column width >= {len(longest_word)}, got {first_column_width}"
    )


def test_should_render_table_with_alignment():
    markdown = Markdown(
        "| Left | Center | Right |\n| :--- | :---: | ---: |\n| A | B | C |\n| Long text | Middle | End |",
        0,
        0,
        default_markdown_theme,
    )

    lines = markdown.render(80)
    plain_lines = [strip_ansi(line) for line in lines]

    # Check headers
    assert any("Left" in line for line in plain_lines)
    assert any("Center" in line for line in plain_lines)
    assert any("Right" in line for line in plain_lines)
    # Check content
    assert any("Long text" in line for line in plain_lines)


def test_should_handle_tables_with_varying_column_widths():
    markdown = Markdown(
        "| Short | Very long column header |\n| --- | --- |\n| A | This is a much longer cell content |\n| B | Short |",
        0,
        0,
        default_markdown_theme,
    )

    lines = markdown.render(80)

    # Should render without errors
    assert len(lines) > 0

    plain_lines = [strip_ansi(line) for line in lines]
    assert any("Very long column header" in line for line in plain_lines)
    assert any("This is a much longer cell content" in line for line in plain_lines)


def test_should_wrap_table_cells_when_table_exceeds_available_width():
    markdown = Markdown(
        "| Command | Description | Example |\n"
        "| --- | --- | --- |\n"
        "| npm install | Install all dependencies | npm install |\n"
        "| npm run build | Build the project | npm run build |",
        0,
        0,
        default_markdown_theme,
    )

    # Render at narrow width that forces wrapping
    lines = markdown.render(50)
    plain_lines = [strip_ansi(line).rstrip() for line in lines]

    # All lines should fit within width
    for line in plain_lines:
        assert len(line) <= 50, f'Line exceeds width 50: "{line}" (length: {len(line)})'

    # Content should still be present (possibly wrapped across lines)
    all_text = " ".join(plain_lines)
    assert "Command" in all_text, "Should contain 'Command'"
    assert "Description" in all_text, "Should contain 'Description'"
    assert "npm install" in all_text, "Should contain 'npm install'"
    assert "Install" in all_text, "Should contain 'Install'"


def test_should_wrap_long_cell_content_to_multiple_lines():
    markdown = Markdown(
        "| Header |\n| --- |\n| This is a very long cell content that should wrap |",
        0,
        0,
        default_markdown_theme,
    )

    # Render at width that forces the cell to wrap
    lines = markdown.render(25)
    plain_lines = [strip_ansi(line).rstrip() for line in lines]

    # Should have multiple data rows due to wrapping
    data_rows = [line for line in plain_lines if line.startswith("│") and "─" not in line]
    assert len(data_rows) > 2, f"Expected wrapped rows, got {len(data_rows)} rows"

    # All content should be preserved (may be split across lines)
    all_text = " ".join(plain_lines)
    assert "very long" in all_text, "Should preserve 'very long'"
    assert "cell content" in all_text, "Should preserve 'cell content'"
    assert "should wrap" in all_text, "Should preserve 'should wrap'"


def test_should_wrap_long_unbroken_tokens_inside_table_cells_not_only_at_line_start():
    # Pin to no-hyperlinks so width checks work on plain text without OSC 8 sequences.
    set_capabilities({"images": None, "trueColor": False, "hyperlinks": False})
    url = "https://example.com/this/is/a/very/long/url/that/should/wrap"
    markdown = Markdown(
        f"| Value |\n| --- |\n| prefix {url} |",
        0,
        0,
        default_markdown_theme,
    )

    width = 30
    lines = markdown.render(width)
    reset_capabilities_cache()
    plain_lines = [strip_ansi(line).rstrip() for line in lines]

    for line in plain_lines:
        assert len(line) <= width, f'Line exceeds width {width}: "{line}" (length: {len(line)})'

    # Borders should stay intact (exactly 2 vertical borders for a 1-col table)
    table_lines = [line for line in plain_lines if line.startswith("│")]
    assert len(table_lines) > 0, "Expected table rows to render"
    for line in table_lines:
        border_count = line.count("│")
        assert border_count == 2, f'Expected 2 borders, got {border_count}: "{line}"'

    # Strip box drawing characters + whitespace so we can assert the URL is preserved
    # even if it was split across multiple wrapped lines.
    extracted = re.sub(r"[│├┤─\s]", "", "".join(plain_lines))
    assert "prefix" in extracted, "Should preserve 'prefix'"
    assert url in extracted, "Should preserve URL"


def test_should_wrap_styled_inline_code_inside_table_cells_without_breaking_borders():
    markdown = Markdown(
        "| Code |\n| --- |\n| `averyveryveryverylongidentifier` |",
        0,
        0,
        default_markdown_theme,
    )

    width = 20
    lines = markdown.render(width)
    joined_output = "\n".join(lines)
    assert "\x1b[33m" in joined_output, "Inline code should be styled (yellow)"

    plain_lines = [strip_ansi(line).rstrip() for line in lines]
    for line in plain_lines:
        assert len(line) <= width, f'Line exceeds width {width}: "{line}" (length: {len(line)})'

    table_lines = [line for line in plain_lines if line.startswith("│")]
    for line in table_lines:
        border_count = line.count("│")
        assert border_count == 2, f'Expected 2 borders, got {border_count}: "{line}"'


def test_should_handle_extremely_narrow_width_gracefully():
    markdown = Markdown(
        "| A | B | C |\n| --- | --- | --- |\n| 1 | 2 | 3 |",
        0,
        0,
        default_markdown_theme,
    )

    # Very narrow width
    lines = markdown.render(15)
    plain_lines = [strip_ansi(line).rstrip() for line in lines]

    # Should not crash and should produce output
    assert len(lines) > 0, "Should produce output"

    # Lines should not exceed width
    for line in plain_lines:
        assert len(line) <= 15, f'Line exceeds width 15: "{line}" (length: {len(line)})'


def test_should_render_table_correctly_when_it_fits_naturally():
    markdown = Markdown(
        "| A | B |\n| --- | --- |\n| 1 | 2 |",
        0,
        0,
        default_markdown_theme,
    )

    # Wide width where table fits naturally
    lines = markdown.render(80)
    plain_lines = [strip_ansi(line).rstrip() for line in lines]

    # Should have proper table structure
    header_line = next((line for line in plain_lines if "A" in line and "B" in line), None)
    assert header_line, "Should have header row"
    assert "│" in header_line, "Header should have borders"

    separator_line = next((line for line in plain_lines if "├" in line and "┼" in line), None)
    assert separator_line, "Should have separator row"

    data_line = next((line for line in plain_lines if "1" in line and "2" in line), None)
    assert data_line, "Should have data row"


def test_should_respect_padding_x_when_calculating_table_width():
    markdown = Markdown(
        "| Column One | Column Two |\n| --- | --- |\n| Data 1 | Data 2 |",
        2,  # paddingX = 2
        0,
        default_markdown_theme,
    )

    # Width 40 with paddingX=2 means contentWidth=36
    lines = markdown.render(40)
    plain_lines = [strip_ansi(line).rstrip() for line in lines]

    # All lines should respect width
    for line in plain_lines:
        assert len(line) <= 40, f'Line exceeds width 40: "{line}" (length: {len(line)})'

    # Table rows should have left padding
    table_row = next((line for line in plain_lines if "│" in line), None)
    assert table_row is not None and table_row.startswith("  "), "Table should have left padding"


def test_should_not_add_a_trailing_blank_line_when_table_is_the_last_rendered_block():
    markdown = Markdown(
        "| Name |\n| --- |\n| Alice |",
        0,
        0,
        default_markdown_theme,
    )

    lines = markdown.render(80)
    plain_lines = [strip_ansi(line).rstrip() for line in lines]

    assert plain_lines[-1] != "", f"Expected table to end without a blank line: {plain_lines!r}"


# Combined features


def test_should_render_lists_and_tables_together():
    markdown = Markdown(
        "# Test Document\n\n- Item 1\n  - Nested item\n- Item 2\n\n| Col1 | Col2 |\n| --- | --- |\n| A | B |",
        0,
        0,
        default_markdown_theme,
    )

    lines = markdown.render(80)
    plain_lines = [strip_ansi(line) for line in lines]

    # Check heading
    assert any("Test Document" in line for line in plain_lines)
    # Check list
    assert any("- Item 1" in line for line in plain_lines)
    assert any("    - Nested item" in line for line in plain_lines)
    # Check table
    assert any("Col1" in line for line in plain_lines)
    assert any("│" in line for line in plain_lines)


# Backslash escapes


def test_should_normalize_escaped_punctuation_by_default():
    markdown = Markdown('"\\"', 0, 0, default_markdown_theme)

    lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

    assert lines == ['""']


def test_should_preserve_source_backslash_escapes_when_configured():
    markdown = Markdown('"\\"', 0, 0, default_markdown_theme, None, {"preserveBackslashEscapes": True})

    lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

    assert lines == ['"\\"']


# Pre-styled text (thinking traces)


def test_should_preserve_gray_italic_styling_after_inline_code():
    # This replicates how thinking content is rendered in assistant-message.ts
    markdown = Markdown(
        "This is thinking with `inline code` and more text after",
        1,
        0,
        default_markdown_theme,
        {
            "color": chalk.gray,
            "italic": True,
        },
    )

    lines = markdown.render(80)
    joined_output = "\n".join(lines)

    # Should contain the inline code block
    assert "inline code" in joined_output

    # The output should have ANSI codes for gray (90) and italic (3)
    assert "\x1b[90m" in joined_output, "Should have gray color code"
    assert "\x1b[3m" in joined_output, "Should have italic code"

    # Verify that inline code is styled (theme uses yellow)
    assert "\x1b[33m" in joined_output, "Should style inline code"


def test_should_preserve_gray_italic_styling_after_bold_text():
    markdown = Markdown(
        "This is thinking with **bold text** and more after",
        1,
        0,
        default_markdown_theme,
        {
            "color": chalk.gray,
            "italic": True,
        },
    )

    lines = markdown.render(80)
    joined_output = "\n".join(lines)

    # Should contain bold text
    assert "bold text" in joined_output

    # The output should have ANSI codes for gray (90) and italic (3)
    assert "\x1b[90m" in joined_output, "Should have gray color code"
    assert "\x1b[3m" in joined_output, "Should have italic code"

    # Should have bold codes (1 or 22 for bold on/off)
    assert "\x1b[1m" in joined_output, "Should have bold code"


@pytest.mark.tonio
async def test_should_not_leak_styles_into_following_lines_when_rendered_in_tui():
    class MarkdownWithInput:
        def __init__(self, markdown: Markdown) -> None:
            self.markdown_line_count = 0
            self._markdown = markdown

        def render(self, width: int) -> list[str]:
            lines = self._markdown.render(width)
            self.markdown_line_count = len(lines)
            return [*lines, "INPUT"]

        def invalidate(self) -> None:
            self._markdown.invalidate()

    markdown = Markdown(
        "This is thinking with `inline code`",
        1,
        0,
        default_markdown_theme,
        {
            "color": chalk.gray,
            "italic": True,
        },
    )

    terminal = VirtualTerminal(80, 6)
    tui = TuiMainScreen(terminal)
    component = MarkdownWithInput(markdown)
    tui.add_child(component)
    await tui.start()
    await terminal.wait_for_render()

    assert component.markdown_line_count > 0
    input_row = component.markdown_line_count
    assert terminal.get_cell_italic(input_row, 0) == 0
    await tui.stop()


# Spacing after code blocks


def test_should_have_only_one_blank_line_between_code_block_and_following_paragraph():
    markdown = Markdown(
        'hello world\n\n```js\nconst hello = "world";\n```\n\nagain, hello world',
        0,
        0,
        default_markdown_theme,
    )

    lines = markdown.render(80)
    plain_lines = [strip_ansi(line).rstrip() for line in lines]

    closing_backticks_index = plain_lines.index("```") if "```" in plain_lines else -1
    assert closing_backticks_index != -1, "Should have closing backticks"

    after_backticks = plain_lines[closing_backticks_index + 1 :]
    empty_line_count = next((i for i, line in enumerate(after_backticks) if line != ""), -1)

    assert empty_line_count == 1, (
        f"Expected 1 empty line after code block, but found {empty_line_count}. "
        f"Lines after backticks: {after_backticks[:5]!r}"
    )


def test_should_normalize_paragraph_and_code_block_spacing_to_one_blank_line():
    cases = [
        "hello this is text\n```\ncode block\n```\nmore text",
        "hello this is text\n\n```\ncode block\n```\n\nmore text",
    ]
    expected_lines = ["hello this is text", "", "```", "  code block", "```", "", "more text"]

    for text in cases:
        markdown = Markdown(text, 0, 0, default_markdown_theme)
        lines = markdown.render(80)
        plain_lines = [strip_ansi(line).rstrip() for line in lines]

        assert plain_lines == expected_lines, f"Unexpected spacing for markdown: {text!r}"


def test_should_not_add_a_trailing_blank_line_when_code_block_is_the_last_rendered_block():
    cases = ["```js\nconst hello = 'world';\n```", "hello world\n\n```js\nconst hello = 'world';\n```"]

    for text in cases:
        markdown = Markdown(text, 0, 0, default_markdown_theme)
        lines = markdown.render(80)
        plain_lines = [strip_ansi(line).rstrip() for line in lines]

        assert plain_lines[-1] != "", f"Expected code block to end without a blank line: {plain_lines!r}"


# Spacing after dividers


def test_should_have_only_one_blank_line_between_divider_and_following_paragraph():
    markdown = Markdown(
        "hello world\n\n---\n\nagain, hello world",
        0,
        0,
        default_markdown_theme,
    )

    lines = markdown.render(80)
    plain_lines = [strip_ansi(line).rstrip() for line in lines]

    divider_index = next((i for i, line in enumerate(plain_lines) if "─" in line), -1)
    assert divider_index != -1, "Should have divider"

    after_divider = plain_lines[divider_index + 1 :]
    empty_line_count = next((i for i, line in enumerate(after_divider) if line != ""), -1)

    assert empty_line_count == 1, (
        f"Expected 1 empty line after divider, but found {empty_line_count}. Lines after divider: {after_divider[:5]!r}"
    )


def test_should_not_add_a_trailing_blank_line_when_divider_is_the_last_rendered_block():
    markdown = Markdown("---", 0, 0, default_markdown_theme)
    lines = markdown.render(80)
    plain_lines = [strip_ansi(line).rstrip() for line in lines]

    assert plain_lines[-1] != "", f"Expected divider to end without a blank line: {plain_lines!r}"


# Spacing after headings


def test_should_have_only_one_blank_line_between_heading_and_following_paragraph():
    markdown = Markdown(
        "# Hello\n\nThis is a paragraph",
        0,
        0,
        default_markdown_theme,
    )

    lines = markdown.render(80)
    plain_lines = [strip_ansi(line).rstrip() for line in lines]

    heading_index = next((i for i, line in enumerate(plain_lines) if "Hello" in line), -1)
    assert heading_index != -1, "Should have heading"

    after_heading = plain_lines[heading_index + 1 :]
    empty_line_count = next((i for i, line in enumerate(after_heading) if line != ""), -1)

    assert empty_line_count == 1, (
        f"Expected 1 empty line after heading, but found {empty_line_count}. Lines after heading: {after_heading[:5]!r}"
    )


def test_should_not_add_a_trailing_blank_line_when_heading_is_the_last_rendered_block():
    markdown = Markdown("# Hello", 0, 0, default_markdown_theme)
    lines = markdown.render(80)
    plain_lines = [strip_ansi(line).rstrip() for line in lines]

    assert plain_lines[-1] != "", f"Expected heading to end without a blank line: {plain_lines!r}"


# Spacing after blockquotes


def test_should_have_only_one_blank_line_between_blockquote_and_following_paragraph():
    markdown = Markdown(
        "hello world\n\n> This is a quote\n\nagain, hello world",
        0,
        0,
        default_markdown_theme,
    )

    lines = markdown.render(80)
    plain_lines = [strip_ansi(line).rstrip() for line in lines]

    quote_index = next((i for i, line in enumerate(plain_lines) if "This is a quote" in line), -1)
    assert quote_index != -1, "Should have blockquote"

    after_quote = plain_lines[quote_index + 1 :]
    empty_line_count = next((i for i, line in enumerate(after_quote) if line != ""), -1)

    assert empty_line_count == 1, (
        f"Expected 1 empty line after blockquote, but found {empty_line_count}. Lines after quote: {after_quote[:5]!r}"
    )


def test_should_not_add_a_trailing_blank_line_when_blockquote_is_the_last_rendered_block():
    markdown = Markdown("> This is a quote", 0, 0, default_markdown_theme)
    lines = markdown.render(80)
    plain_lines = [strip_ansi(line).rstrip() for line in lines]

    assert plain_lines[-1] != "", f"Expected blockquote to end without a blank line: {plain_lines!r}"


# Blockquotes with multiline content


def test_should_apply_consistent_styling_to_all_lines_in_lazy_continuation_blockquote():
    # Markdown "lazy continuation" - second line without > is still part of the quote
    markdown = Markdown(
        ">Foo\nbar",
        0,
        0,
        default_markdown_theme,
        {
            "color": chalk.magenta,  # This should NOT be applied to blockquotes
        },
    )

    lines = markdown.render(80)

    # Both lines should have the quote border
    plain_lines = [strip_ansi(line) for line in lines]
    quoted_lines = [line for line in plain_lines if line.startswith("│ ")]
    assert len(quoted_lines) == 2, f"Expected 2 quoted lines, got: {plain_lines!r}"

    # Both lines should have italic (from theme.quote styling)
    foo_line = next((line for line in lines if "Foo" in line), None)
    bar_line = next((line for line in lines if "bar" in line), None)
    assert foo_line, "Should have Foo line"
    assert bar_line, "Should have bar line"

    # Check that both have italic (\x1b[3m) - blockquotes use theme styling, not default message color
    assert "\x1b[3m" in foo_line, f"Foo line should have italic: {foo_line}"
    assert "\x1b[3m" in bar_line, f"bar line should have italic: {bar_line}"

    # Blockquotes should NOT have the default message color (magenta)
    assert "\x1b[35m" not in foo_line, f"Foo line should NOT have magenta color: {foo_line}"
    assert "\x1b[35m" not in bar_line, f"bar line should NOT have magenta color: {bar_line}"


def test_should_apply_consistent_styling_to_explicit_multiline_blockquote():
    markdown = Markdown(
        ">Foo\n>bar",
        0,
        0,
        default_markdown_theme,
        {
            "color": chalk.cyan,  # This should NOT be applied to blockquotes
        },
    )

    lines = markdown.render(80)

    # Both lines should have the quote border
    plain_lines = [strip_ansi(line) for line in lines]
    quoted_lines = [line for line in plain_lines if line.startswith("│ ")]
    assert len(quoted_lines) == 2, f"Expected 2 quoted lines, got: {plain_lines!r}"

    # Both lines should have italic (from theme.quote styling)
    foo_line = next((line for line in lines if "Foo" in line), None)
    bar_line = next((line for line in lines if "bar" in line), None)
    assert foo_line is not None and "\x1b[3m" in foo_line, f"Foo line should have italic: {foo_line}"
    assert bar_line is not None and "\x1b[3m" in bar_line, f"bar line should have italic: {bar_line}"

    # Blockquotes should NOT have the default message color (cyan)
    assert "\x1b[36m" not in foo_line, f"Foo line should NOT have cyan color: {foo_line}"
    assert "\x1b[36m" not in bar_line, f"bar line should NOT have cyan color: {bar_line}"


def test_should_render_list_content_inside_blockquotes():
    markdown = Markdown(
        "> 1. bla bla\n> - nested bullet",
        0,
        0,
        default_markdown_theme,
    )

    lines = markdown.render(80)
    plain_lines = [strip_ansi(line) for line in lines]
    quoted_lines = [line for line in plain_lines if line.startswith("│ ")]

    assert any("1. bla bla" in line for line in quoted_lines), f"Missing ordered list item: {quoted_lines!r}"
    assert any("- nested bullet" in line for line in quoted_lines), f"Missing unordered list item: {quoted_lines!r}"


def test_should_wrap_long_blockquote_lines_and_add_border_to_each_wrapped_line():
    long_text = "This is a very long blockquote line that should wrap to multiple lines when rendered"
    markdown = Markdown(f"> {long_text}", 0, 0, default_markdown_theme)

    # Render at narrow width to force wrapping
    lines = markdown.render(30)
    plain_lines = [strip_ansi(line).rstrip() for line in lines]

    # Filter to non-empty lines (exclude trailing blank line after blockquote)
    content_lines = [line for line in plain_lines if line]

    # Should have multiple lines due to wrapping
    assert len(content_lines) > 1, f"Expected multiple wrapped lines, got: {content_lines!r}"

    # Every content line should start with the quote border
    for line in content_lines:
        assert line.startswith("│ "), f'Wrapped line should have quote border: "{line}"'

    # All content should be preserved
    all_text = " ".join(content_lines)
    assert "very long" in all_text, "Should preserve 'very long'"
    assert "blockquote" in all_text, "Should preserve 'blockquote'"
    assert "multiple" in all_text, "Should preserve 'multiple'"


def test_should_properly_indent_wrapped_blockquote_lines_with_styling():
    markdown = Markdown(
        "> This is styled text that is long enough to wrap",
        0,
        0,
        default_markdown_theme,
        {
            "color": chalk.yellow,  # This should NOT be applied to blockquotes
            "italic": True,
        },
    )

    lines = markdown.render(25)
    plain_lines = [strip_ansi(line).rstrip() for line in lines]

    # Filter to non-empty lines
    content_lines = [line for line in plain_lines if line]

    # All lines should have the quote border
    for line in content_lines:
        assert line.startswith("│ "), f'Line should have quote border: "{line}"'

    # Check that italic is applied (from theme.quote)
    all_output = "\n".join(lines)
    assert "\x1b[3m" in all_output, "Should have italic"

    # Blockquotes should NOT have the default message color (yellow)
    assert "\x1b[33m" not in all_output, "Should NOT have yellow color from default style"


def test_should_render_inline_formatting_inside_blockquotes_and_reapply_quote_styling_after():
    markdown = Markdown("> Quote with **bold** and `code`", 0, 0, default_markdown_theme)

    lines = markdown.render(80)
    plain_lines = [strip_ansi(line) for line in lines]

    # Should have the quote border
    assert any(line.startswith("│ ") for line in plain_lines), "Should have quote border"

    # Content should be preserved
    all_plain = " ".join(plain_lines)
    assert "Quote with" in all_plain, "Should preserve 'Quote with'"
    assert "bold" in all_plain, "Should preserve 'bold'"
    assert "code" in all_plain, "Should preserve 'code'"

    all_output = "\n".join(lines)

    # Should have bold styling (\x1b[1m)
    assert "\x1b[1m" in all_output, "Should have bold styling"

    # Should have code styling (yellow = \x1b[33m from default_markdown_theme)
    assert "\x1b[33m" in all_output, "Should have code styling (yellow)"

    # Should have italic from quote styling (\x1b[3m)
    assert "\x1b[3m" in all_output, "Should have italic from quote styling"


# Heading with inline code


def test_should_preserve_heading_styling_after_inline_code():
    markdown = Markdown("### Why `sourceInfo` should not be optional", 0, 0, default_markdown_theme)

    lines = markdown.render(80)
    joined_output = "\n".join(lines)

    # The heading theme is bold+cyan. After the yellow inline code, the heading
    # styling (bold+cyan) must be restored so subsequent text is styled correctly.
    # bold = \x1b[1m, cyan = \x1b[36m, yellow = \x1b[33m
    assert "\x1b[33m" in joined_output, "Should have yellow for inline code"

    # Find the position of "should not be optional" in the raw output.
    # It must be preceded by heading style codes (bold+cyan), not appear unstyled.
    after_code_index = joined_output.find("should not be optional")
    assert after_code_index > 0, "Should contain text after inline code"

    # Look at the ANSI codes between the code span end and "should not be optional".
    # There should be bold (\x1b[1m) and cyan (\x1b[36m) re-applied.
    preceding_chunk = joined_output[max(0, after_code_index - 40) : after_code_index]
    assert "\x1b[1m" in preceding_chunk, f"Should re-apply bold before text after code: {preceding_chunk}"
    assert "\x1b[36m" in preceding_chunk, f"Should re-apply cyan before text after code: {preceding_chunk}"


def test_should_preserve_heading_styling_after_inline_code_for_h1():
    markdown = Markdown("# Title with `code` inside", 0, 0, default_markdown_theme)

    lines = markdown.render(80)
    joined_output = "\n".join(lines)

    after_code_index = joined_output.find("inside")
    assert after_code_index > 0, "Should contain text after inline code"

    preceding_chunk = joined_output[max(0, after_code_index - 40) : after_code_index]
    # H1 uses heading + bold + underline
    assert "\x1b[1m" in preceding_chunk, f"Should re-apply bold for h1: {preceding_chunk}"
    assert "\x1b[36m" in preceding_chunk, f"Should re-apply cyan for h1: {preceding_chunk}"
    assert "\x1b[4m" in preceding_chunk, f"Should re-apply underline for h1: {preceding_chunk}"


@pytest.mark.tonio
async def test_should_not_leak_h1_underline_into_padding_when_inline_code_is_the_last_token():
    markdown = Markdown("# Important distinction from `open()`", 0, 0, default_markdown_theme)
    terminal = VirtualTerminal(80, 4)
    tui = TuiMainScreen(terminal)
    tui.add_child(markdown)
    await tui.start()
    await terminal.wait_for_render()

    rendered_line = markdown.render(80)[0]
    assert rendered_line, "Should render heading line"
    content_width = len(strip_ansi(rendered_line).rstrip())
    assert content_width > 0, "Should have visible heading content"

    for col in range(content_width, 80):
        assert terminal.get_cell_underline(0, col) == 0, f"Expected no underline in padding at col {col}"

    await tui.stop()


def test_should_preserve_heading_styling_after_bold_text():
    markdown = Markdown("## Heading with **bold** and more", 0, 0, default_markdown_theme)

    lines = markdown.render(80)
    joined_output = "\n".join(lines)

    after_bold_index = joined_output.find("and more")
    assert after_bold_index > 0, "Should contain text after bold"

    preceding_chunk = joined_output[max(0, after_bold_index - 40) : after_bold_index]
    assert "\x1b[1m" in preceding_chunk, f"Should re-apply bold for h2: {preceding_chunk}"
    assert "\x1b[36m" in preceding_chunk, f"Should re-apply cyan for h2: {preceding_chunk}"


# Strikethrough syntax


def test_should_render_double_tilde_as_strikethrough():
    markdown = Markdown("Use ~~strikethrough~~ here", 0, 0, default_markdown_theme)

    lines = markdown.render(80)
    joined_output = "\n".join(lines)
    joined_plain = " ".join(strip_ansi(line) for line in lines)

    assert "\x1b[9m" in joined_output, "Should apply strikethrough styling"
    assert "strikethrough" in joined_plain, "Should include struck text content"
    assert "~~strikethrough~~" not in joined_plain, "Should not render delimiters as text"


def test_should_keep_single_tilde_as_plain_text():
    markdown = Markdown("Use ~strikethrough~ literally", 0, 0, default_markdown_theme)

    lines = markdown.render(80)
    joined_output = "\n".join(lines)
    joined_plain = " ".join(strip_ansi(line) for line in lines)

    assert "~strikethrough~" in joined_plain, "Single-tilde delimiters should remain visible"
    assert "\x1b[9m" not in joined_output, "Single-tilde text should not use strikethrough styling"


# Links


@pytest.fixture
def capabilities_reset():
    yield
    reset_capabilities_cache()


def test_should_not_duplicate_url_for_autolinked_emails(capabilities_reset):
    # Hyperlinks capability does not affect the mailto: display check.
    set_capabilities({"images": None, "trueColor": False, "hyperlinks": False})
    markdown = Markdown("Contact user@example.com for help", 0, 0, default_markdown_theme)

    lines = markdown.render(80)
    joined_plain = " ".join(strip_ansi(line) for line in lines)

    # Should contain the email once, not duplicated with mailto:
    assert "user@example.com" in joined_plain, "Should contain email"
    assert "mailto:" not in joined_plain, "Should not show mailto: prefix for autolinked emails"


def test_should_not_duplicate_url_for_bare_urls(capabilities_reset):
    set_capabilities({"images": None, "trueColor": False, "hyperlinks": False})
    markdown = Markdown("Visit https://example.com for more", 0, 0, default_markdown_theme)

    lines = markdown.render(80)
    joined_plain = " ".join(strip_ansi(line) for line in lines)

    # URL should appear only once
    url_count = len(re.findall(r"https://example\.com", joined_plain))
    assert url_count == 1, "URL should appear exactly once"


def test_should_show_url_in_parentheses_when_hyperlinks_are_not_supported(capabilities_reset):
    set_capabilities({"images": None, "trueColor": False, "hyperlinks": False})
    markdown = Markdown("[click here](https://example.com)", 0, 0, default_markdown_theme)

    lines = markdown.render(80)
    joined_plain = " ".join(strip_ansi(line) for line in lines)

    assert "click here" in joined_plain, "Should contain link text"
    assert "(https://example.com)" in joined_plain, "Should show URL in parentheses"


def test_should_show_mailto_url_in_parentheses_when_hyperlinks_are_not_supported(capabilities_reset):
    set_capabilities({"images": None, "trueColor": False, "hyperlinks": False})
    markdown = Markdown("[Email me](mailto:test@example.com)", 0, 0, default_markdown_theme)

    lines = markdown.render(80)
    joined_plain = " ".join(strip_ansi(line) for line in lines)

    assert "Email me" in joined_plain, "Should contain link text"
    assert "(mailto:test@example.com)" in joined_plain, "Should show mailto URL in parentheses"


def test_should_emit_osc8_hyperlink_sequence_when_terminal_supports_hyperlinks(capabilities_reset):
    set_capabilities({"images": None, "trueColor": False, "hyperlinks": True})
    markdown = Markdown("[click here](https://example.com)", 0, 0, default_markdown_theme)

    lines = markdown.render(80)
    joined = "".join(lines)

    # OSC 8 open: ESC ] 8 ; ; <url> ESC \
    assert "\x1b]8;;https://example.com\x1b\\" in joined, "Should contain OSC 8 open sequence"
    # OSC 8 close: ESC ] 8 ; ; ESC \
    assert "\x1b]8;;\x1b\\" in joined, "Should contain OSC 8 close sequence"
    # Visible text is present
    plain_lines = [re.sub(r"\x1b[^a-zA-Z]*[a-zA-Z]|\x1b\].*?\x1b\\", "", line) for line in lines]
    assert "click here" in "".join(plain_lines), "Should contain link text"
    # URL is NOT printed inline as plain text
    raw_plain = [strip_ansi(re.sub(r"\x1b\]8;;[^\x1b]*\x1b\\", "", line)) for line in lines]
    assert "(https://example.com)" not in "".join(raw_plain), "URL should not appear inline in parentheses"


def test_should_use_osc8_for_mailto_links_when_terminal_supports_hyperlinks(capabilities_reset):
    set_capabilities({"images": None, "trueColor": False, "hyperlinks": True})
    markdown = Markdown("[Email me](mailto:test@example.com)", 0, 0, default_markdown_theme)

    lines = markdown.render(80)
    joined = "".join(lines)

    assert "\x1b]8;;mailto:test@example.com\x1b\\" in joined, "Should contain OSC 8 open with mailto URL"
    assert "\x1b]8;;\x1b\\" in joined, "Should contain OSC 8 close sequence"


def test_should_use_osc8_for_bare_urls_when_terminal_supports_hyperlinks(capabilities_reset):
    set_capabilities({"images": None, "trueColor": False, "hyperlinks": True})
    markdown = Markdown("Visit https://example.com for more", 0, 0, default_markdown_theme)

    lines = markdown.render(80)
    joined = "".join(lines)

    assert "\x1b]8;;https://example.com\x1b\\" in joined, "Should contain OSC 8 hyperlink"
    # URL should not also appear as raw parenthetical text
    raw_plain = [strip_ansi(re.sub(r"\x1b\]8;;[^\x1b]*\x1b\\", "", line)) for line in lines]
    assert "(https://example.com)" not in "".join(raw_plain), "URL should not appear twice"


# HTML-like tags in text


def test_should_render_content_with_html_like_tags_as_text():
    # When the model emits something like <thinking>content</thinking> in regular text,
    # marked might treat it as HTML and hide the content
    markdown = Markdown(
        "This is text with <thinking>hidden content</thinking> that should be visible",
        0,
        0,
        default_markdown_theme,
    )

    lines = markdown.render(80)
    joined_plain = " ".join(strip_ansi(line) for line in lines)

    # The content inside the tags should be visible
    assert "hidden content" in joined_plain or "<thinking>" in joined_plain, (
        "Should render HTML-like tags or their content as text, not hide them"
    )


def test_should_render_html_tags_in_code_blocks_correctly():
    markdown = Markdown("```html\n<div>Some HTML</div>\n```", 0, 0, default_markdown_theme)

    lines = markdown.render(80)
    joined_plain = "\n".join(strip_ansi(line) for line in lines)

    # HTML in code blocks should be visible
    assert "<div>" in joined_plain and "</div>" in joined_plain, "Should render HTML in code blocks"


# Streaming code fences


def test_stabilizes_partial_closing_fence_rendering():
    cases = [
        {
            "input": "```ts\nconst x = 1;\n``",
            "expected": ["```ts", "  const x = 1;", "```"],
        },
        {
            "input": "```md\nnot a closing fence:\n``\n```",
            "expected": ["```md", "  not a closing fence:", "  ``", "```"],
        },
        {
            "input": "```ts\n``",
            "expected": ["```ts", "", "```"],
        },
        {
            "input": "````\n```",
            "expected": ["```", "", "```"],
        },
        {
            "input": "~~~~~\n~~~~",
            "expected": ["```", "", "```"],
        },
        {
            "input": "```md\nnot a closing fence:\n``\n```\n\nafter",
            "expected": ["```md", "  not a closing fence:", "  ``", "```", "", "after"],
        },
    ]

    for case in cases:
        markdown = Markdown(case["input"], 0, 0, default_markdown_theme)
        lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

        assert lines == case["expected"]

    partial = Markdown("```ts\nconst x = 1;\n``", 0, 0, default_markdown_theme)
    complete = Markdown("```ts\nconst x = 1;\n```", 0, 0, default_markdown_theme)

    assert len(partial.render(80)) == len(complete.render(80))


# LaTeX math


def test_renders_inline_dollar_and_parenthesis_delimiters():
    markdown = Markdown(
        r"A map $\mathbb{C}^3 \to \mathbb{C}^3$, $xy$, $x-y$, $-x$, $\frac{1}{2}$, and \(s \to \infty\).",
        0,
        0,
        default_markdown_theme,
    )

    lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

    assert lines == ["A map ℂ³ → ℂ³, xy, x-y, -x, 1/2, and s → ∞."]


def test_renders_display_dollar_delimiters_without_markdown_escape_corruption():
    markdown = Markdown(
        "Before\n\n$$\\{3x+2y,\\; x \\in \\{0, \\pm 1\\}\\}$$\n\nafter",
        0,
        0,
        default_markdown_theme,
    )

    lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

    assert lines == ["Before", "", "{3x+2y, x ∈ {0, ± 1}}", "", "after"]


def test_renders_display_bracket_delimiters():
    markdown = Markdown(
        "Before\n\n\\[\nE \\approx \\frac{0.1\\ \\text{lux}}{100\\ \\text{lm/W}}\n\\]\n\nafter",
        0,
        0,
        default_markdown_theme,
    )

    lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

    assert lines == ["Before", "", "    0.1 lux", "E ≈ ────────", "    100 lm/W", "", "after"]


def test_aligns_matrix_rows_with_the_opening_delimiter():
    markdown = Markdown(
        "Consider the matrix\n\n\\[\nA=\n\\begin{pmatrix}\n\\pi & 0\\\\\n0 & \\frac{1}{\\pi}\n\\end{pmatrix}.\n\\]",
        0,
        0,
        default_markdown_theme,
    )

    lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

    assert lines == ["Consider the matrix", "", "A = ⎛ π │ 0   ⎞", "    ⎝ 0 │ 1/π ⎠."]


def test_renders_lower_limits_beneath_display_operators():
    markdown = Markdown(
        "\\[\n\\lim_{x\\to 0}\\frac{\\frac{\\sin x}{x}-1}{\\frac{e^x-1}{x}-1}=0\n\\]",
        0,
        0,
        default_markdown_theme,
    )

    lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

    assert lines == ["     (sin x)/x-1", "lim  ─────────── = 0", "x→0  (eˣ-1)/x-1"]


def test_renders_math_inside_lists_and_tables():
    markdown = Markdown(
        "- Formula: $F_1 = u^2$\n\n| Value |\n| --- |\n| $\\mathbb{C}^3$ |",
        0,
        0,
        default_markdown_theme,
    )

    output = "\n".join(strip_ansi(line).rstrip() for line in markdown.render(80))

    assert "- Formula: F₁ = u²" in output
    assert "│ ℂ³" in output


def test_does_not_treat_currency_shell_variables_or_code_spans_as_math():
    source = "Costs $5 and $10 or $8k–$12k; use `$x$`, $HOME, and $" + "{PATH}."
    markdown = Markdown(source, 0, 0, default_markdown_theme)

    lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

    assert lines == ["Costs $5 and $10 or $8k–$12k; use $x$, $HOME, and $" + "{PATH}."]

    shell_variables = "Paths: $HOME/$USER and $XDG_CONFIG_HOME/$APP_CONFIG"
    shell_lines = [
        strip_ansi(line).rstrip() for line in Markdown(shell_variables, 0, 0, default_markdown_theme).render(80)
    ]
    assert shell_lines == [shell_variables]


@pytest.mark.parametrize("source", [r"Unknown $x + \unknown{y}$ after", r"Streaming $\mathbb{C}^3"])
def test_preserves_unsupported_and_incomplete_latex_exactly(source):
    markdown = Markdown(source, 0, 0, default_markdown_theme)
    lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]
    assert lines == [source]


def test_preserves_incomplete_backslash_delimiters_while_streaming():
    inline = Markdown(r"Map \(\mathbb{C}^3", 0, 0, default_markdown_theme)
    assert [strip_ansi(line).rstrip() for line in inline.render(80)] == [r"Map \(\mathbb{C}^3"]

    display = Markdown("\\[\nx^2", 0, 0, default_markdown_theme)
    assert [strip_ansi(line).rstrip() for line in display.render(80)] == ["\\[", "x^2"]


def test_does_not_render_latex_inside_escaped_delimiters_or_code_fences():
    source = "Escaped \\$x-y\\$.\n\n```text\n$\\mathbb{C}^3$\n```"
    markdown = Markdown(source, 0, 0, default_markdown_theme)
    lines = [strip_ansi(line).rstrip() for line in markdown.render(80)]

    assert lines == ["Escaped $x-y$.", "", "```text", r"  $\mathbb{C}^3$", "```"]


def test_allows_latex_rendering_to_be_disabled():
    markdown = Markdown(
        r"Map $\mathbb{C}^3 \to \mathbb{C}^3$",
        0,
        0,
        default_markdown_theme,
        None,
        {"renderLatex": False},
    )

    assert [strip_ansi(line).rstrip() for line in markdown.render(80)] == [r"Map $\mathbb{C}^3 \to \mathbb{C}^3$"]


def test_switches_from_raw_to_rendered_math_when_a_streamed_delimiter_closes():
    markdown = Markdown(r"Map $\mathbb{C}^3", 0, 0, default_markdown_theme)
    assert [strip_ansi(line).rstrip() for line in markdown.render(80)] == [r"Map $\mathbb{C}^3"]

    markdown.set_text(r"Map $\mathbb{C}^3$")

    assert [strip_ansi(line).rstrip() for line in markdown.render(80)] == ["Map ℂ³"]
