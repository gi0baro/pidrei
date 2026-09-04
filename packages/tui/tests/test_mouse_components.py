"""Mirror of pi tui test/mouse-components.test.ts."""

import pytest

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
