"""Mirror of pi tui test/mouse-components.test.ts."""

from dataclasses import replace

import pytest
import tonio.colored as tonio

from pidrei_tui.components.editor import Editor
from pidrei_tui.components.input import Input
from pidrei_tui.components.select_list import SelectList
from pidrei_tui.components.settings_list import SettingsList
from pidrei_tui.tui import Container, TuiMouseEvent
from pidrei_tui.tui_alt_screen import TuiAltScreen

from .virtual_terminal import VirtualTerminal


def mouse(event_type: str, x: int, y: int, width: int = 80, height: int = 10) -> TuiMouseEvent:
    return TuiMouseEvent(
        type=event_type,
        button="left",
        x=x,
        y=y,
        screen_x=x,
        screen_y=y,
        width=width,
        height=height,
        click_count=1 if event_type == "click" else None,
    )


SELECT_THEME = {
    "selectedPrefix": lambda text: text,
    "selectedText": lambda text: text,
    "description": lambda text: text,
    "scrollInfo": lambda text: text,
    "noMatch": lambda text: text,
}

SETTINGS_THEME = {
    "label": lambda text, selected=False: text,
    "value": lambda text, selected=False: text,
    "description": lambda text: text,
    "cursor": "> ",
    "hint": lambda text: text,
}

EDITOR_THEME = {"borderColor": lambda text: text, "selectList": SELECT_THEME}


class InputOverlay(Container):
    def __init__(self) -> None:
        super().__init__()
        self.input = Input()
        self.add_child(self.input)

    async def handle_input(self, data: str) -> None:
        await self.input.handle_input(data)


@pytest.mark.tonio
async def test_positions_a_single_line_input_cursor_on_press():
    input_component = Input()
    input_component.set_value("hello")
    input_component.render(20)

    result = await input_component.handle_mouse(mouse("press", 4, 0, 20, 1))
    assert result is not None and result.handled is True
    await input_component.handle_input("X")
    assert input_component.get_value() == "heXllo"


@pytest.mark.tonio
async def test_selects_and_activates_list_rows():
    select_list = SelectList(
        [
            {"value": "a", "label": "A"},
            {"value": "b", "label": "B"},
            {"value": "c", "label": "C"},
            {"value": "d", "label": "D"},
            {"value": "e", "label": "E"},
        ],
        3,
        SELECT_THEME,
    )
    selected: list[str] = []

    async def on_select(item: dict) -> None:
        selected.append(item["value"])

    select_list.on_select = on_select

    result = await select_list.handle_mouse(mouse("press", 1, 2, 40, 3))
    assert result is not None and result.handled is True
    assert select_list.get_selected_item()["value"] == "c"
    result = await select_list.handle_mouse(mouse("click", 1, 2, 40, 3))
    assert result is not None and result.handled is True
    assert selected == ["c"]


@pytest.mark.tonio
async def test_activates_settings_rows():
    changes: list[dict] = []

    async def on_change(item_id: str, value: str) -> None:
        changes.append({"id": item_id, "value": value})

    async def on_cancel() -> None:
        pass

    settings_list = SettingsList(
        [
            {"id": "mode", "label": "Mode", "currentValue": "one", "values": ["one", "two"]},
            {"id": "other", "label": "Other", "currentValue": "off", "values": ["off", "on"]},
            {"id": "third", "label": "Third", "currentValue": "low", "values": ["low", "high"]},
            {"id": "fourth", "label": "Fourth", "currentValue": "x", "values": ["x", "y"]},
        ],
        3,
        SETTINGS_THEME,
        on_change,
        on_cancel,
    )

    await settings_list.handle_mouse(mouse("press", 1, 2, 40, 5))
    await settings_list.handle_mouse(mouse("click", 1, 2, 40, 5))
    assert changes == [{"id": "third", "value": "high"}]


@pytest.mark.tonio
@pytest.mark.parametrize("row", [0, 4])
async def test_ignores_hover_and_clicks_visible_select_list_row_after_scrolling(row):
    select_list = SelectList(
        [{"value": f"item-{i}", "label": f"Item {i}"} for i in range(12)],
        5,
        SELECT_THEME,
    )
    changes: list[str] = []
    selected: list[str] = []

    async def on_selection_change(item: dict) -> None:
        changes.append(item["value"])

    async def on_select(item: dict) -> None:
        selected.append(item["value"])

    select_list.on_selection_change = on_selection_change
    select_list.on_select = on_select
    select_list.set_selected_index(5)
    await select_list.handle_mouse(replace(mouse("wheel", 1, row), wheel_delta=1))
    assert select_list.get_selected_item()["value"] == "item-6"
    assert changes == ["item-6"]
    before = select_list.render(80)
    assert before[row].endswith(f"Item {4 + row}")

    for y in [0, 1, 2, 3, 4, row]:
        assert await select_list.handle_mouse(replace(mouse("move", 1, y), button="none")) is None
        assert select_list.render(80) == before
    assert select_list.get_selected_item()["value"] == "item-6"
    assert changes == ["item-6"]
    assert selected == []

    await select_list.handle_mouse(mouse("press", 1, row))
    select_list.render(80)
    await select_list.handle_mouse(mouse("click", 1, row))
    assert selected == [f"item-{4 + row}"]
    assert changes == ["item-6", f"item-{4 + row}"]


@pytest.mark.tonio
@pytest.mark.parametrize("row", [0, 4])
async def test_ignores_hover_and_clicks_visible_settings_row_after_scrolling(row):
    changes: list[dict] = []

    async def on_change(item_id: str, value: str) -> None:
        changes.append({"id": item_id, "value": value})

    async def on_cancel() -> None:
        pass

    settings_list = SettingsList(
        [
            {
                "id": f"item-{i}",
                "label": f"Item {i}",
                "description": f"Description {i}",
                "currentValue": "off",
                "values": ["off", "on"],
            }
            for i in range(12)
        ],
        5,
        SETTINGS_THEME,
        on_change,
        on_cancel,
        {"enableSearch": True},
    )
    settings_list.select_item("item-5")
    await settings_list.handle_mouse(replace(mouse("wheel", 1, row + 2), wheel_delta=1))
    before = settings_list.render(80)
    assert before[4].startswith("> Item 6")
    assert f"Item {4 + row} " in before[row + 2]

    for y in [0, 1, 2, 3, 4, row]:
        assert await settings_list.handle_mouse(replace(mouse("move", 1, y + 2), button="none")) is None
        assert settings_list.render(80) == before
    assert changes == []

    await settings_list.handle_mouse(mouse("press", 1, row + 2))
    settings_list.render(80)
    await settings_list.handle_mouse(mouse("click", 1, row + 2))
    assert changes == [{"id": f"item-{4 + row}", "value": "on"}]


@pytest.mark.tonio
async def test_keeps_a_delegating_overlay_focused_when_its_nested_input_is_clicked():
    terminal = VirtualTerminal(20, 4)
    tui = TuiAltScreen(terminal)
    overlay = InputOverlay()
    overlay.input.set_value("hi")
    await tui.start()
    tui.show_overlay(overlay, {"anchor": "top-left", "width": 20})
    await terminal.wait_for_render()

    since = terminal.frames
    await terminal.send_input("\x1b[<0;5;1M")
    await terminal.send_input("\x1b[<0;5;1m")
    await terminal.send_input("!")
    await terminal.wait_for_render(since)

    assert overlay.input.get_value() == "hi!"
    assert tui.get_focused_component() is overlay
    await tui.stop()


@pytest.mark.tonio
async def test_positions_and_focuses_the_multiline_editor_through_alternate_screen_dispatch():
    terminal = VirtualTerminal(20, 6)
    tui = TuiAltScreen(terminal)
    editor = Editor(tui, EDITOR_THEME)
    editor.set_text("hello")
    tui.add_child(editor)
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    await terminal.send_input("\x1b[<0;3;2M")
    await terminal.send_input("\x1b[<0;3;2m")
    await terminal.send_input("X")
    await terminal.wait_for_render(since)

    assert editor.get_text() == "heXllo"
    assert tui.get_focused_component() is editor
    await tui.stop()


@pytest.mark.tonio
async def test_selects_and_copies_editor_text_on_drag_instead_of_moving_the_cursor():
    terminal = VirtualTerminal(20, 6)
    copied: list[str] = []
    # The copy runs as a detached task after the release; wait on its signal
    # rather than on a frame that may precede it.
    copied_event = tonio.Event()

    async def copy_selection(text: str) -> bool:
        copied.append(text)
        copied_event.set()
        return True

    tui = TuiAltScreen(terminal, None, None, copy_selection=copy_selection)
    editor = Editor(tui, EDITOR_THEME)
    editor.set_text("hello world")
    tui.add_child(editor)
    await tui.start()
    await terminal.wait_for_render()
    cursor_before = editor.get_cursor()

    since = terminal.frames
    await terminal.send_input("\x1b[<0;1;2M")
    await terminal.send_input("\x1b[<32;5;2M")
    await terminal.send_input("\x1b[<0;5;2m")
    await terminal.wait_for_render(since)
    await copied_event.wait(2)

    assert copied_event.is_set()
    assert copied == ["hello"]
    assert editor.get_cursor() == cursor_before
    await tui.stop()
