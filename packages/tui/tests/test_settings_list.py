"""Mirror of pi tui test/settings-list.test.ts."""

import re

import pytest

from pidrei_tui.components.settings_list import SettingsList


TEST_THEME = {
    # label/value also take the selected flag here; pi's SettingsListTheme
    # types them as `(text, selected?) => string` and its test omits the arg.
    "label": lambda text, selected=False: text,
    "value": lambda text, selected=False: text,
    "description": lambda text: text,
    "cursor": "> ",
    "hint": lambda text: text,
}

ITEMS = [{"id": "tui-mode", "label": "TUI mode", "currentValue": "regular", "values": ["regular", "fullscreen"]}]


def _make_list(changes: list[dict]) -> SettingsList:
    def on_change(item_id, value) -> None:
        changes.append({"id": item_id, "value": value})

    def on_cancel() -> None:
        pass

    return SettingsList([dict(item) for item in ITEMS], 10, TEST_THEME, on_change, on_cancel, {"enableSearch": True})


@pytest.mark.tonio
async def test_includes_spaces_in_an_active_search_instead_of_changing_the_selected_setting():
    changes: list[dict] = []
    settings_list = _make_list(changes)

    for character in "TUI mode":
        settings_list.handle_input(character)

    assert changes == []
    assert re.search("TUI mode", settings_list.render(80)[0])

    settings_list.handle_input("\r")
    assert changes == [{"id": "tui-mode", "value": "fullscreen"}]


@pytest.mark.tonio
async def test_keeps_space_as_a_change_shortcut_before_a_search_query_is_entered():
    changes: list[dict] = []
    settings_list = _make_list(changes)

    settings_list.handle_input(" ")

    assert changes == [{"id": "tui-mode", "value": "fullscreen"}]


def test_moves_by_one_visible_page_with_selection_page_keys():
    # #7629
    settings_list = SettingsList(
        [{"id": f"setting-{index}", "label": f"Setting {index}", "currentValue": str(index)} for index in range(12)],
        5,
        TEST_THEME,
        lambda *args: None,
        lambda: None,
    )

    settings_list.handle_input("\x1b[6~")
    assert re.search("> Setting 5", "\n".join(settings_list.render(80)))
    settings_list.handle_input("\x1b[6~")
    settings_list.handle_input("\x1b[6~")
    assert re.search("> Setting 11", "\n".join(settings_list.render(80)))
    settings_list.handle_input("\x1b[5~")
    assert re.search("> Setting 6", "\n".join(settings_list.render(80)))
