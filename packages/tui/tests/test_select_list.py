"""Mirror of pi tui test/select-list.test.ts."""

from pidrei_tui.components.select_list import SelectList
from pidrei_tui.utils import visible_width


test_theme = {
    "selectedPrefix": lambda text: text,
    "selectedText": lambda text: text,
    "description": lambda text: text,
    "scrollInfo": lambda text: text,
    "noMatch": lambda text: text,
}


def visible_index_of(line: str, text: str) -> int:
    index = line.find(text)
    assert index != -1
    return visible_width(line[:index])


def test_normalizes_multiline_descriptions_to_single_line():
    items = [
        {
            "value": "test",
            "label": "test",
            "description": "Line one\nLine two\nLine three",
        }
    ]

    select_list = SelectList(items, 5, test_theme)
    rendered = select_list.render(100)

    assert len(rendered) > 0
    assert "\n" not in rendered[0]
    assert "Line one Line two Line three" in rendered[0]


def test_keeps_descriptions_aligned_when_the_primary_text_is_truncated():
    items = [
        {"value": "short", "label": "short", "description": "short description"},
        {
            "value": "very-long-command-name-that-needs-truncation",
            "label": "very-long-command-name-that-needs-truncation",
            "description": "long description",
        },
    ]

    select_list = SelectList(items, 5, test_theme)
    rendered = select_list.render(80)

    assert visible_index_of(rendered[0], "short description") == visible_index_of(rendered[1], "long description")


def test_uses_the_configured_minimum_primary_column_width():
    items = [
        {"value": "a", "label": "a", "description": "first"},
        {"value": "bb", "label": "bb", "description": "second"},
    ]

    select_list = SelectList(
        items,
        5,
        test_theme,
        {
            "minPrimaryColumnWidth": 12,
            "maxPrimaryColumnWidth": 20,
        },
    )
    rendered = select_list.render(80)

    assert rendered[0].find("first") == 14
    assert rendered[1].find("second") == 14


def test_uses_the_configured_maximum_primary_column_width():
    items = [
        {
            "value": "very-long-command-name-that-needs-truncation",
            "label": "very-long-command-name-that-needs-truncation",
            "description": "first",
        },
        {"value": "short", "label": "short", "description": "second"},
    ]

    select_list = SelectList(
        items,
        5,
        test_theme,
        {
            "minPrimaryColumnWidth": 12,
            "maxPrimaryColumnWidth": 20,
        },
    )
    rendered = select_list.render(80)

    assert visible_index_of(rendered[0], "first") == 22
    assert visible_index_of(rendered[1], "second") == 22


def test_allows_overriding_primary_truncation_while_preserving_description_alignment():
    items = [
        {
            "value": "very-long-command-name-that-needs-truncation",
            "label": "very-long-command-name-that-needs-truncation",
            "description": "first",
        },
        {"value": "short", "label": "short", "description": "second"},
    ]

    def truncate_primary(context: dict) -> str:
        text = context["text"]
        max_width = context["maxWidth"]
        if len(text) <= max_width:
            return text
        return f"{text[: max(0, max_width - 1)]}…"

    select_list = SelectList(
        items,
        5,
        test_theme,
        {
            "minPrimaryColumnWidth": 12,
            "maxPrimaryColumnWidth": 12,
            "truncatePrimary": truncate_primary,
        },
    )
    rendered = select_list.render(80)

    assert "…" in rendered[0]
    assert visible_index_of(rendered[0], "first") == visible_index_of(rendered[1], "second")
