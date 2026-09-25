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
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive import interactive_mode, tui_renderer
from pidrei.modes.interactive.components.status_indicator import (
    BranchSummaryStatusIndicator,
    CompactionStatusIndicator,
    RetryStatusIndicator,
    WorkingStatusIndicator,
)
from pidrei.modes.interactive.interactive_mode import InteractiveMode, create_interactive_tui
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei_tui import Container, ScrollView, Text, get_keybindings, is_viewport_tui, set_keybindings

from .ui_timer_helpers import manual_ui_timers


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tui" / "tests"))
from virtual_terminal import VirtualTerminal


@contextlib.contextmanager
def _recording_clipboard(copied: list[str]):
    """pi mocks the clipboard module; here the imported names are swapped back.

    Two importers: the renderer's `copy_selection` (tui_renderer) and the
    interactive mode's own copy paths.
    """

    async def copy(text: str) -> None:
        copied.append(text)

    originals = (interactive_mode.copy_to_clipboard, tui_renderer.copy_to_clipboard)
    interactive_mode.copy_to_clipboard = copy
    tui_renderer.copy_to_clipboard = copy
    try:
        yield
    finally:
        interactive_mode.copy_to_clipboard, tui_renderer.copy_to_clipboard = originals


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
    def __init__(self, kind: str = "working") -> None:
        self.kind = kind
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
        self._extension_registry_guard = threading.Lock()
        self.runtime_host = SimpleNamespace(
            session=SimpleNamespace(settings_manager=SimpleNamespace(get_fullscreen_copy_on_select=lambda: True))
        )
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
async def test_replaces_the_renderer_and_restores_the_previous_screen_for_resume_hint_exits():
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

    await context._stop_interactive_tui("resume-hint")

    assert context.ui.mode == "fullscreen"
    assert [terminal.start_count, terminal.stop_count] == [2, 2]


@pytest.mark.tonio
async def test_a_settings_tui_mode_change_made_on_the_owner_switches_the_renderer():
    # The settings list runs its change callback on the owner (it is input
    # handling), and the switch stops the renderer, whose barrier waits on
    # that owner: awaited there, the switch never returned.
    terminal = RecordingTerminal(40, 8)
    renderer = create_interactive_tui(
        tui_mode="regular", show_hardware_cursor=False, log_directory="/tmp", terminal=terminal
    )
    context = _SwitchContext(renderer, None)
    component = _InvalidationProbe(lambda: context.ui.mode)
    context._fullscreen_layout_root = component
    renderer.add_child(component)
    saved_modes: list[str] = []
    context.runtime_host.session.settings_manager.set_tui_mode = saved_modes.append
    context.runtime_host.session.settings_manager.get_show_terminal_progress = lambda: False
    context._active_status_indicator = None
    context._status_container = Container()
    statuses: list[str] = []
    announced = tonio.Event()

    def show_status(message: str) -> None:
        statuses.append(message)
        announced.set()

    context.show_status = show_status

    await renderer.start()
    await terminal.wait_for_render()

    async def settings_input() -> None:
        context._on_settings_tui_mode_change("fullscreen", selector=None)

    renderer.input_owner.post(settings_input)
    await announced.wait(5)

    assert statuses == ["TUI mode: fullscreen"]
    assert saved_modes == ["fullscreen"]
    assert context.ui.mode == "fullscreen"
    assert context._renderer.children == [component]
    await context._stop_interactive_tui("resume-hint")


@pytest.mark.tonio
async def test_an_overlay_opened_while_the_switch_stops_the_renderer_keeps_the_previous_one():
    # Overlays cannot be carried to the next renderer. One whose opening was
    # queued on the owner when the switch began is shown by the time the
    # renderer has stopped: the switch must see it, and resume the previous
    # renderer with it instead of dropping it.
    terminal = RecordingTerminal(40, 8)
    renderer = create_interactive_tui(
        tui_mode="regular", show_hardware_cursor=False, log_directory="/tmp", terminal=terminal
    )
    context = _SwitchContext(renderer, None)
    base = _InvalidationProbe(lambda: context.ui.mode)
    renderer.add_child(base)
    await renderer.start()
    await terminal.wait_for_render()

    owner = renderer.input_owner
    release = tonio.Event()

    async def busy() -> None:
        await release.wait(5)

    overlay = _InvalidationProbe(lambda: context.ui.mode)

    async def open_overlay() -> None:
        renderer.show_overlay(overlay)

    owner.post(busy)
    owner.post(open_overlay)

    # The stop's barrier queues behind the busy job: release it only once
    # the switch has committed to stopping.
    stopping = tonio.Event()
    owner_run = owner.run

    async def run(fn) -> None:
        stopping.set()
        await owner_run(fn)

    owner.run = run
    switched = tonio.Result()

    async def switch() -> None:
        switched.store(await context._switch_tui_mode("fullscreen", restore_progress=False))

    async with tonio.scope() as scope:
        scope.spawn(switch())
        await stopping.wait(5)
        del owner.run
        release.set()

    assert switched.fetch() is False
    assert context._renderer is renderer
    assert renderer.has_overlay_entries
    assert renderer.children == [base]
    assert [terminal.start_count, terminal.stop_count] == [2, 1]
    await renderer.stop()


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
async def test_copies_an_active_fullscreen_selection_when_copy_on_select_is_disabled():
    copied: list[str] = []
    terminal = RecordingTerminal(40, 4)
    ui = create_interactive_tui(
        tui_mode="fullscreen",
        show_hardware_cursor=False,
        log_directory="/tmp",
        terminal=terminal,
        fullscreen_copy_on_select=False,
    )
    context = _CopyRecorder(ui)
    last_text_calls: list[bool] = []

    def get_last_assistant_text() -> str:
        last_text_calls.append(True)
        return "assistant response"

    context.session = SimpleNamespace(get_last_assistant_text=get_last_assistant_text)
    ui.add_child(Text("alpha\nbeta\ngamma\ndelta", 0, 0))

    await ui.start()
    try:
        await terminal.wait_for_render()
        with _recording_clipboard(copied):
            since = terminal.frames
            await terminal.send_input("\x1b[<0;1;1M")
            await terminal.send_input("\x1b[<32;4;2M")
            await terminal.send_input("\x1b[<0;4;2m")
            await terminal.wait_for_render(since)
            copied.clear()

            since = terminal.frames
            await InteractiveMode._handle_copy_command(context, {"flashConfirmation": True, "preferSelection": True})
            await terminal.wait_for_render(since)

        assert copied == ["alpha\nbeta"]
        assert last_text_calls == []
        assert context.statuses == []
        assert context.errors == []
        assert any("Copied!" in line for line in terminal.get_viewport())
    finally:
        await ui.stop()


@pytest.mark.tonio
async def test_copies_the_last_assistant_message_with_an_active_fullscreen_selection_when_copy_on_select_is_enabled():
    copied: list[str] = []
    terminal = RecordingTerminal(40, 4)
    ui = create_interactive_tui(
        tui_mode="fullscreen", show_hardware_cursor=False, log_directory="/tmp", terminal=terminal
    )
    context = _CopyRecorder(ui)
    last_text_calls: list[bool] = []

    def get_last_assistant_text() -> str:
        last_text_calls.append(True)
        return "assistant response"

    context.session = SimpleNamespace(get_last_assistant_text=get_last_assistant_text)
    ui.add_child(Text("alpha\nbeta\ngamma\ndelta", 0, 0))

    await ui.start()
    try:
        await terminal.wait_for_render()
        with _recording_clipboard(copied):
            since = terminal.frames
            await terminal.send_input("\x1b[<0;1;1M")
            await terminal.send_input("\x1b[<32;4;2M")
            await terminal.send_input("\x1b[<0;4;2m")
            await terminal.wait_for_render(since)
            copied.clear()

            since = terminal.frames
            await InteractiveMode._handle_copy_command(context, {"flashConfirmation": True, "preferSelection": True})
            await terminal.wait_for_render(since)

        assert copied == ["assistant response"]
        assert last_text_calls == [True]
        assert context.statuses == []
        assert context.errors == []
        assert any("Copied!" in line for line in terminal.get_viewport())
    finally:
        await ui.stop()


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
            await InteractiveMode._handle_copy_command(context, {"flashConfirmation": True, "preferSelection": True})
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
        await InteractiveMode._handle_copy_command(context, {"flashConfirmation": True, "preferSelection": True})

    assert context.statuses == ["Copied last agent message to clipboard"]
    assert context.errors == []


class _StatusEditor:
    """pi's StatusEditor fake: the two members `is_working_status_editor` checks."""

    def __init__(self, embed_working_status: bool) -> None:
        self.embed_working_status = embed_working_status
        self.indicators: list = []

    def set_working_status_indicator(self, indicator) -> None:
        self.indicators.append(indicator)


def _clear_status_context(*, tui_mode: str, indicator, embedded: bool, default_editor, editor):
    mode = _BareInteractiveMode()
    mode._active_status_indicator = indicator
    mode._active_working_indicator_embedded = embedded
    mode._status_container = Container()
    mode._default_editor = default_editor
    mode.editor = editor
    mode._options = {"tuiMode": tui_mode}
    # post_ui applies inline like an un-started TUI (island relaxation,
    # PROPER_MT_DESIGN step 1).
    mode.ui = type(
        "_Ui",
        (),
        {"get_clear_on_shrink": staticmethod(lambda: True), "post_ui": staticmethod(lambda fn: fn())},
    )()
    mode._idle_status = Text("", 0, 0)
    return mode


@pytest.mark.tonio
@pytest.mark.parametrize("embed_working_status", [True, False])
async def test_routes_every_status_through_the_editor_opt_in(embed_working_status):
    init_theme_sync("dark")
    tui = SimpleNamespace(request_render=lambda force=False: None)
    editor = _StatusEditor(embed_working_status=embed_working_status)
    mode = _clear_status_context(
        tui_mode="regular",
        indicator=None,
        embedded=False,
        default_editor=_StatusEditor(embed_working_status=True),
        editor=editor,
    )
    # The retry countdown would expire on another task while the test disposes it.
    with manual_ui_timers():
        indicators = [
            WorkingStatusIndicator(tui, "Working"),
            CompactionStatusIndicator(tui, "manual"),
            CompactionStatusIndicator(tui, "threshold"),
            CompactionStatusIndicator(tui, "overflow"),
            BranchSummaryStatusIndicator(tui),
            RetryStatusIndicator(tui, 1, 3, 1000),
        ]
    try:
        for indicator in indicators:
            mode._show_status_indicator(indicator)
            assert mode._active_status_indicator is indicator
            assert mode._active_working_indicator_embedded is embed_working_status
            if embed_working_status:
                assert editor.indicators[-1] is indicator
                assert len(mode._status_container.children) == 0
            else:
                assert mode._status_container.children == [indicator]
    finally:
        for indicator in indicators:
            indicator.dispose()


@pytest.mark.parametrize("kind", ["working", "compaction", "branchSummary", "retry"])
def test_does_not_reserve_separate_status_height_for_an_embedded_indicator(kind):
    indicator = _DisposeRecorder(kind)
    editor = _StatusEditor(embed_working_status=True)
    mode = _clear_status_context(
        tui_mode="regular", indicator=indicator, embedded=True, default_editor=editor, editor=editor
    )

    mode._clear_status_indicator()

    assert indicator.calls == 1
    # pi asserts `toHaveBeenCalledWith(undefined)`: the editor is both the
    # default and the active one, so the border is cleared through each role.
    assert editor.indicators and all(cleared is None for cleared in editor.indicators)
    assert len(mode._status_container.children) == 0


@pytest.mark.parametrize(("tui_mode", "expected_children"), [("regular", 1), ("fullscreen", 0)])
def test_uses_the_standalone_row_for_a_custom_editor_that_has_not_opted_in(tui_mode, expected_children):
    default_editor = _StatusEditor(embed_working_status=True)
    custom_editor = _StatusEditor(embed_working_status=False)
    mode = _clear_status_context(
        tui_mode=tui_mode,
        indicator=_DisposeRecorder(),
        embedded=False,
        default_editor=default_editor,
        editor=custom_editor,
    )

    mode._clear_status_indicator()

    assert default_editor.indicators and all(cleared is None for cleared in default_editor.indicators)
    assert custom_editor.indicators == []
    assert len(mode._status_container.children) == expected_children


@pytest.mark.tonio
async def test_shows_the_configured_jump_to_bottom_shortcut_while_scrolled_up():
    init_theme_sync("dark")
    previous_keybindings = get_keybindings()
    set_keybindings(KeybindingsManager({"tui.altScreen.bottom": "ctrl+j"}))
    terminal = RecordingTerminal(50, 4)
    ui = create_interactive_tui(
        tui_mode="fullscreen", show_hardware_cursor=False, log_directory="/tmp", terminal=terminal
    )
    ui.set_layout_root(
        ScrollView(Text("\n".join(f"line {index + 1}" for index in range(8)), 0, 0), {"follow": "end", "primary": True})
    )
    await ui.start()
    try:
        await terminal.wait_for_render()
        await terminal.send_input("\x1b[<64;1;1M")
        assert await terminal.until(lambda: "↓ Jump to latest message · Ctrl+J" in terminal.get_viewport()[3]), (
            terminal.get_viewport()
        )
    finally:
        await ui.stop()
        set_keybindings(previous_keybindings)
