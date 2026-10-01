"""pidrei-only: the extension UI protocol (spec/ui-island.md, "Extensions").

pi's extensions reach its one-threaded TUI directly; here `ctx.ui` calls go
through the UI state lock, the dialogs mount at call time, component
factories run under the lock, and component code gets a guarded `tui`.
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.core.extensions.runner import ExtensionRunner
from pidrei.core.extensions.types import ExtensionRuntime
from pidrei.core.session_manager import SessionManager
from pidrei.modes.interactive.extension_tui import ExtensionTui, guard_overlay_handle
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.interactive.theme import get_editor_theme, init_theme
from pidrei_tui import Container, Editor, OverlayHandle, TuiMainScreen


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tui" / "tests"))
from virtual_terminal import VirtualTerminal


def _held(lock) -> bool:
    """Whether `lock` is held: a probe thread cannot take it without waiting."""
    taken: list[bool] = []

    def probe() -> None:
        acquired = lock.acquire(blocking=False)
        if acquired:
            lock.release()
        taken.append(acquired)

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return not taken[0]


class _Component:
    def __init__(self, label: str = "component") -> None:
        self.focused = False
        self._label = label

    def render(self, _width) -> list[str]:
        return [self._label]

    def invalidate(self) -> None:
        pass


def _custom_mode(ui) -> SimpleNamespace:
    """Fake `this` for `_show_extension_custom`."""
    editor = _Component("editor")
    editor.get_text = lambda: "draft"
    editor.set_text = lambda _text: None
    container = Container()
    container.add_child(editor)
    return SimpleNamespace(
        editor=editor,
        _editor_container=container,
        _keybindings={},
        ui=ui,
        _extension_tui=ExtensionTui(ui),
        _dispose_active_selector=lambda: None,
    )


@pytest.mark.tonio
async def test_a_dialog_called_without_await_is_open_before_the_call_returns():
    # From a synchronous `handle_input`, the dialog must be mounted before the
    # next key is routed: the call itself opens it, and the handle it returns
    # yields the answer.
    answers = tonio.Event()
    opened: list[str] = []

    async def answer() -> str:
        await answers.wait(5)
        return "b"

    def select(title, _options, _opts=None):
        opened.append(title)
        return tonio.spawn(answer())

    runner = ExtensionRunner([], ExtensionRuntime(), "/", SessionManager.in_memory(), None)
    runner.set_ui_context(SimpleNamespace(select=select), "tui")

    handle = runner.get_ui_context().select("pick", ["a", "b"])

    assert opened == ["pick"]
    answers.set()
    assert await handle == "b"


@pytest.mark.tonio
async def test_custom_runs_its_factory_and_mount_in_one_hold_of_the_lock():
    await init_theme("dark")
    ui = TuiMainScreen(VirtualTerminal(40, 10))
    mode = _custom_mode(ui)
    component = _Component()
    held_in_factory: list[bool] = []
    closers: list = []

    def factory(tui, _theme, _keybindings, done):
        held_in_factory.append(_held(ui.state_lock))
        closers.append(done)
        assert not hasattr(tui, "state_lock")  # the guarded wrapper
        return component

    handle = InteractiveMode._show_extension_custom(mode, factory)

    assert held_in_factory == [True]
    assert mode._editor_container.children == [component]
    assert ui.get_focused_component() is component

    async def close_elsewhere() -> None:
        closers[0]("closed")

    tonio.spawn.without_tracking(close_elsewhere())
    assert await handle == "closed"
    assert mode._editor_container.children == [mode.editor]


@pytest.mark.tonio
async def test_custom_refuses_an_async_factory_and_keeps_the_editor():
    await init_theme("dark")
    ui = TuiMainScreen(VirtualTerminal(40, 10))
    mode = _custom_mode(ui)

    async def factory(_tui, _theme, _keybindings, _done):
        return _Component()

    with pytest.raises(TypeError):
        InteractiveMode._show_extension_custom(mode, factory)
    assert mode._editor_container.children == [mode.editor]


@pytest.mark.tonio
async def test_paste_to_editor_is_applied_before_the_call_returns():
    await init_theme("dark")
    ui = TuiMainScreen(VirtualTerminal(40, 10))
    editor = Editor(ui, get_editor_theme())
    mode = SimpleNamespace(ui=ui, editor=editor)
    ctx_ui = InteractiveMode._create_extension_ui_context(mode)

    ctx_ui.paste_to_editor("pasted")

    assert ctx_ui.get_editor_text() == "pasted"


@pytest.mark.tonio
async def test_extension_timers_run_under_the_lock_and_report_their_errors():
    ui = TuiMainScreen(VirtualTerminal(40, 10))
    reported: list[BaseException] = []
    got_error = tonio.Event()

    async def on_error(error: BaseException) -> None:
        reported.append(error)
        got_error.set()

    ui.set_render_error_handler(on_error)
    tui = ExtensionTui(ui)
    held: list[bool] = []

    def fire() -> None:
        held.append(_held(ui.state_lock))
        raise RuntimeError("tick failed")

    tui.timeout(0, fire)
    await got_error.wait(5)

    assert held == [True]
    assert [str(error) for error in reported] == ["tick failed"]


@pytest.mark.tonio
async def test_extension_timers_refuse_async_callbacks():
    tui = ExtensionTui(TuiMainScreen(VirtualTerminal(40, 10)))

    async def tick() -> None:
        pass

    with pytest.raises(TypeError):
        tui.interval(10, tick)


@pytest.mark.tonio
async def test_extension_spawn_reports_what_escapes_the_work():
    ui = TuiMainScreen(VirtualTerminal(40, 10))
    reported: list[BaseException] = []
    got_error = tonio.Event()

    async def on_error(error: BaseException) -> None:
        reported.append(error)
        got_error.set()

    ui.set_render_error_handler(on_error)

    async def work() -> None:
        raise RuntimeError("work failed")

    ExtensionTui(ui).spawn(work())
    await got_error.wait(5)

    assert [str(error) for error in reported] == ["work failed"]


@pytest.mark.tonio
async def test_a_guarded_overlay_handle_calls_through_the_lock():
    ui = TuiMainScreen(VirtualTerminal(40, 10))
    held: list[bool] = []

    def record(*_args) -> None:
        held.append(_held(ui.state_lock))

    raw = OverlayHandle(
        hide=record,
        set_hidden=record,
        is_hidden=record,
        focus=record,
        unfocus=record,
        is_focused=record,
        get_bounds=record,
    )
    handle = guard_overlay_handle(ui, raw)

    handle.set_hidden(True)
    handle.focus()
    handle.hide()

    assert held == [True, True, True]
