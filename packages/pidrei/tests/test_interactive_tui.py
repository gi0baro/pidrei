"""Mirror of pi coding-agent test/interactive-tui.test.ts.

The renderer choice and the one behaviour that differs between the two
renderers: the main screen reserves a status row so clear-on-shrink has
something to overwrite, the alternate screen repaints the whole viewport and
does not need it.

pi's fake `this` for `clearStatusIndicator` becomes a `_BareInteractiveMode`
(the same pattern as test_interactive_mode_status.py) so the real method runs
against plain attributes.
"""

import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pidrei.modes.interactive import interactive_mode
from pidrei.modes.interactive.interactive_mode import InteractiveMode, create_interactive_tui
from pidrei_tui import Container, Text, is_viewport_tui


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tui" / "tests"))
from virtual_terminal import VirtualTerminal


@contextlib.contextmanager
def _recording_clipboard(copied: list[str]):
    """pi mocks the clipboard module; here the imported name is swapped back."""

    async def copy(text: str) -> None:
        copied.append(text)

    original = interactive_mode.copy_to_clipboard
    interactive_mode.copy_to_clipboard = copy
    try:
        yield
    finally:
        interactive_mode.copy_to_clipboard = original


class RecordingTerminal(VirtualTerminal):
    """VirtualTerminal that records every write and counts start/stop."""

    def __init__(self, columns: int = 80, rows: int = 24) -> None:
        super().__init__(columns, rows)
        self.writes: list[str] = []
        self.start_count = 0
        self.stop_count = 0

    async def start(self, on_input, on_resize) -> None:
        self.start_count += 1
        await super().start(on_input, on_resize)

    async def write(self, data: str) -> None:
        self.writes.append(data)
        await super().write(data)

    async def stop(self) -> None:
        self.stop_count += 1
        await super().stop()


class _BareInteractiveMode(InteractiveMode):
    """InteractiveMode without the constructor, for the fake-this case."""

    def __init__(self) -> None:
        pass


class _DisposeRecorder:
    kind = "working"

    def __init__(self) -> None:
        self.calls = 0

    def dispose(self) -> None:
        self.calls += 1


@pytest.mark.tonio
async def test_selects_the_alternate_screen_renderer_only_when_requested():
    main_terminal = RecordingTerminal()
    main_tui = create_interactive_tui(
        tui_mode="regular", show_hardware_cursor=False, log_directory="/tmp", terminal=main_terminal
    )
    assert main_tui.mode == "regular"
    assert is_viewport_tui(main_tui) is False
    await main_tui.start()
    await main_terminal.wait_for_render()
    assert not any("\x1b[?1049h" in write for write in main_terminal.writes)
    await main_tui.stop()

    alt_terminal = RecordingTerminal()
    alt_tui = create_interactive_tui(
        tui_mode="fullscreen", show_hardware_cursor=False, log_directory="/tmp", terminal=alt_terminal
    )
    assert alt_tui.mode == "fullscreen"
    assert is_viewport_tui(alt_tui) is True
    await alt_tui.start()
    await alt_terminal.wait_for_render()
    assert any("\x1b[?1049h" in write for write in alt_terminal.writes)
    await alt_tui.stop()


class _SwitchContext(_BareInteractiveMode):
    """Fake `this` for `_switch_tui_mode`/`_stop_interactive_tui`.

    pi builds the same shape with `Object.create(InteractiveMode.prototype)`.
    """

    def __init__(self, renderer, layout_root) -> None:
        self._renderer = renderer
        self._main_screen_render_state = None
        self._fullscreen_layout_root = layout_root
        self._options = {"tuiMode": "regular"}
        self._theme_controller = SimpleNamespace(rebind_tui=_noop_async)
        self._extension_terminal_input_subscriptions = set()
        self.ui = interactive_mode.create_interactive_tui_reference(lambda: self._renderer)


async def _noop_async() -> None:
    return None


class _InvalidationProbe:
    """Component that records the renderer mode active at each invalidate()."""

    def __init__(self, get_mode) -> None:
        self.focused = False
        self._get_mode = get_mode
        self.invalidated_modes: list[str] = []

    def render(self, _width: int) -> list[str]:
        return ["content"]

    def invalidate(self) -> None:
        self.invalidated_modes.append(self._get_mode())


@pytest.mark.tonio
async def test_replaces_the_renderer_while_preserving_components_and_focus():
    terminal = RecordingTerminal(40, 8)
    renderer = create_interactive_tui(
        tui_mode="regular", show_hardware_cursor=False, log_directory="/tmp", terminal=terminal
    )
    context = _SwitchContext(renderer, None)
    component = _InvalidationProbe(lambda: context.ui.mode)
    context._fullscreen_layout_root = component
    renderer.add_child(component)
    renderer.set_focus(component)

    await renderer.start()
    await terminal.wait_for_render()
    assert await context._switch_tui_mode("fullscreen", restore_progress=False) is True
    await terminal.wait_for_render()

    assert context.ui.mode == "fullscreen"
    assert context._renderer.children == [component]
    assert context._renderer.get_focused_component() is component
    assert component.focused is True
    assert component.invalidated_modes == ["fullscreen"]
    assert [terminal.start_count, terminal.stop_count] == [2, 1]

    await context._stop_interactive_tui()

    assert context.ui.mode == "regular"
    assert [terminal.start_count, terminal.stop_count] == [2, 3]


class _CopyRecorder:
    """Fake `this` for `_handle_copy_command` (pi calls the prototype method)."""

    def __init__(self, ui) -> None:
        self.ui = ui
        self._renderer = ui
        self.session = SimpleNamespace(get_last_assistant_text=lambda: "assistant response")
        self.statuses: list[str] = []
        self.errors: list[str] = []

    def show_status(self, message: str) -> None:
        self.statuses.append(message)

    def show_error(self, message: str) -> None:
        self.errors.append(message)


@pytest.mark.tonio
async def test_flashes_copied_for_the_copy_shortcut_in_fullscreen_mode():
    copied: list[str] = []
    terminal = RecordingTerminal(40, 4)
    ui = create_interactive_tui(
        tui_mode="fullscreen", show_hardware_cursor=False, log_directory="/tmp", terminal=terminal
    )
    context = _CopyRecorder(ui)

    await ui.start()
    try:
        await terminal.wait_for_render()
        with _recording_clipboard(copied):
            since = terminal.frames
            await InteractiveMode._handle_copy_command(context, {"flashConfirmation": True})
            await terminal.wait_for_render(since)

        assert copied == ["assistant response"]
        assert context.statuses == []
        assert context.errors == []
        assert any("Copied!" in line for line in terminal.get_viewport())
    finally:
        await ui.stop()


@pytest.mark.tonio
async def test_keeps_the_status_line_confirmation_for_the_copy_shortcut_in_regular_mode():
    copied: list[str] = []
    ui = create_interactive_tui(
        tui_mode="regular", show_hardware_cursor=False, log_directory="/tmp", terminal=RecordingTerminal()
    )
    context = _CopyRecorder(ui)

    with _recording_clipboard(copied):
        await InteractiveMode._handle_copy_command(context, {"flashConfirmation": True})

    assert context.statuses == ["Copied last agent message to clipboard"]
    assert context.errors == []


@pytest.mark.parametrize(("tui_mode", "expected_children"), [("regular", 1), ("fullscreen", 0)])
def test_reserves_status_height_only_on_the_main_screen_renderer(tui_mode, expected_children):
    indicator = _DisposeRecorder()
    mode = _BareInteractiveMode()
    mode._active_status_indicator = indicator
    mode._status_container = Container()
    mode._options = {"tuiMode": tui_mode}
    mode.ui = type("_Ui", (), {"get_clear_on_shrink": staticmethod(lambda: True)})()
    mode._idle_status = Text("", 0, 0)

    mode._clear_status_indicator()

    assert indicator.calls == 1
    assert len(mode._status_container.children) == expected_children
