"""Mirror of pi coding-agent test/interactive-mode-extension-dialogs.test.ts.

pi drives InteractiveMode.prototype methods on a Proxy that falls back to the prototype; here
``_DialogContext`` does the same with the unbound methods.
"""

import threading
from functools import partial
from types import SimpleNamespace

import pytest

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.interactive.program_status_reporter import BlockedStatus
from pidrei.modes.interactive.theme import init_theme
from pidrei_tui import Container, OverlayHandle, Text, set_keybindings
from pidrei_utils.cancel import CancelToken


class _Editor(Text):
    def __init__(self) -> None:
        super().__init__("editor")
        self.editor_text = ""

    def get_text(self) -> str:
        return self.editor_text

    def set_text(self, text: str) -> None:
        self.editor_text = text


class _DialogContext(SimpleNamespace):
    """Attributes set here shadow InteractiveMode; every other name is its method, bound to this context."""

    def __getattr__(self, name):
        return partial(getattr(InteractiveMode, name), self)


def _create_context() -> _DialogContext:
    noop = lambda *_args: None
    blocked_calls: list = []
    context = _DialogContext(
        _extension_dialogs=[],
        _editor_container=Container(),
        editor=_Editor(),
        focused=None,
        blocked_calls=blocked_calls,
        overlays=[],
        _next_blocking_overlay_id=0,
        _extension_overlays={},
        _program_status=SimpleNamespace(set_blocked=lambda key, status: blocked_calls.append((key, status))),
        _dispose_active_selector=noop,
        _toggle_tool_output_expansion=noop,
        _extension_tui=None,
        _keybindings=None,
    )

    def show_overlay(component, _options=None) -> OverlayHandle:
        context.overlays.append(component)

        def hide() -> None:
            context.overlays = [overlay for overlay in context.overlays if overlay is not component]

        return OverlayHandle(
            hide=hide,
            set_hidden=noop,
            is_hidden=lambda: False,
            focus=noop,
            unfocus=noop,
            is_focused=lambda: False,
            get_bounds=lambda: None,
        )

    context.ui = SimpleNamespace(
        state_lock=threading.RLock(),
        set_focus=lambda component: setattr(context, "focused", component),
        request_render=noop,
        show_overlay=show_overlay,
        apply=lambda fn: fn(),
    )
    return context


def _visible(context: _DialogContext) -> str:
    assert context._editor_container.children == [context.focused]
    return "\n".join(context.focused.render(80))


@pytest.fixture
async def dialog_setup():
    await init_theme("dark")
    set_keybindings(KeybindingsManager.in_memory())


# Regression for #11166: a second dialog replaced the first one and the first promise never settled.
@pytest.mark.tonio
async def test_shows_the_earlier_confirm_again_after_the_later_one_is_answered(dialog_setup):
    context = _create_context()
    first = context._show_extension_confirm("First", "first dialog")
    second = context._show_extension_confirm("Second", "second dialog")

    assert "Second" in _visible(context)
    assert context.blocked_calls[-1] == ("extension-dialog", BlockedStatus("permission", "Second"))
    context.focused.handle_input("\x1b")
    assert await second is False

    assert "First" in _visible(context)
    assert context.blocked_calls[-1] == ("extension-dialog", BlockedStatus("permission", "First"))
    context.focused.handle_input("\r")
    assert await first is True
    assert context.focused is context.editor
    assert context.blocked_calls[-1] == ("extension-dialog", None)


@pytest.mark.tonio
async def test_removes_a_hidden_dialog_when_its_signal_aborts(dialog_setup):
    context = _create_context()
    abort = CancelToken()
    first = context._show_extension_confirm("First", "first dialog", {"signal": abort})
    second = context._show_extension_confirm("Second", "second dialog")

    abort.cancel()
    assert await first is False
    assert "Second" in _visible(context)

    context.focused.handle_input("\r")
    assert await second is True
    assert context.focused is context.editor


# Regression for #9909: a second inline custom component replaced the first one and the first promise never settled.
@pytest.mark.tonio
async def test_stacks_concurrent_inline_custom_components(dialog_setup):
    context = _create_context()
    dones: list = []

    def open_custom(label: str):
        def factory(_tui, _theme, _keybindings, done):
            dones.append(done)
            return Text(label)

        return context._show_extension_custom(factory)

    first = open_custom("custom one")
    second = open_custom("custom two")
    assert len(dones) == 2
    assert "custom two" in _visible(context)

    dones[1]("two")
    assert await second == "two"
    assert "custom one" in _visible(context)

    dones[0]("one")
    assert await first == "one"
    assert context.focused is context.editor


@pytest.mark.tonio
async def test_shows_a_confirm_opened_from_an_inline_custom_component_and_then_restores_the_component(
    dialog_setup,
):
    context = _create_context()
    opened: dict = {}

    class CustomForm(Text):
        def handle_input(self, _data: str) -> None:
            opened["confirmed"] = context._show_extension_confirm("Discard?", "discard changes")

    def factory(_tui, _theme, _keybindings, finish):
        opened["done"] = finish
        return CustomForm("custom form")

    custom = context._show_extension_custom(factory)
    assert "custom form" in _visible(context)

    context.focused.handle_input("x")
    assert "Discard?" in _visible(context)
    context.focused.handle_input("\x1b")
    assert await opened["confirmed"] is False

    assert "custom form" in _visible(context)
    opened["done"]("kept")
    assert await custom == "kept"
    assert context.focused is context.editor


# Regression for #11428: inline custom dialogs could not report blocked program status.
@pytest.mark.tonio
async def test_reports_blocked_status_for_an_inline_custom_component_that_opts_in(dialog_setup):
    context = _create_context()
    dones: list = []

    def factory(label: str):
        def build(_tui, _theme, _keybindings, done):
            dones.append(done)
            return Text(label)

        return build

    first = context._show_extension_custom(
        factory("approval"), {"blocked": {"kind": "permission", "message": "Approve?"}}
    )
    assert "approval" in _visible(context)
    assert context.blocked_calls[-1] == ("extension-dialog", BlockedStatus("permission", "Approve?"))

    loader = context._show_extension_custom(factory("loading"))
    assert "loading" in _visible(context)
    assert context.blocked_calls[-1] == ("extension-dialog", None)

    dones[1]("loaded")
    assert await loader == "loaded"
    assert context.blocked_calls[-1] == ("extension-dialog", BlockedStatus("permission", "Approve?"))

    dones[0]("yes")
    assert await first == "yes"
    assert context.focused is context.editor
    assert context.blocked_calls[-1] == ("extension-dialog", None)


# Regression for #11816: overlays ignored the blocked option.
@pytest.mark.tonio
async def test_reports_blocked_status_for_each_opted_in_overlay_until_it_is_done_also_while_hidden(dialog_setup):
    context = _create_context()
    dones: list = []
    handles: list = []

    def open_overlay(text: str, blocked: dict | None = None):
        def factory(_tui, _theme, _keybindings, done):
            dones.append(done)
            return Text(text)

        return context._show_extension_custom(
            factory, {"overlay": True, "onHandle": handles.append, "blocked": blocked}
        )

    first = open_overlay("first", {"kind": "question", "message": "First?"})
    assert len(context.overlays) == 1
    assert context.blocked_calls[-1] == ("extension-overlay:0", BlockedStatus("question", "First?"))

    second = open_overlay("second", {"kind": "question", "message": "Second?"})
    assert len(context.overlays) == 2
    assert context.blocked_calls[-1] == ("extension-overlay:1", BlockedStatus("question", "Second?"))

    loader = open_overlay("loading")
    assert len(context.overlays) == 3
    dones[2]("loaded")
    assert await loader == "loaded"

    handles[0].set_hidden(True)
    dones[1]("two")
    assert await second == "two"
    assert context.blocked_calls[-1] == ("extension-overlay:1", None)

    dones[0]("one")
    assert await first == "one"
    assert context.blocked_calls[-1] == ("extension-overlay:0", None)
    assert len(context.blocked_calls) == 4


# Regression for #11816: /reload and session switches drop overlays without calling done().
@pytest.mark.tonio
async def test_clears_overlay_blocked_status_when_the_extension_ui_is_reset(dialog_setup):
    context = _create_context()
    _stub_the_rest_of_reset(context)
    context._show_extension_custom(
        lambda _tui, _theme, _keybindings, _done: Text("question"),
        {"overlay": True, "blocked": {"kind": "question", "message": "Question?"}},
    )
    assert len(context.overlays) == 1
    assert context.blocked_calls[-1] == ("extension-overlay:0", BlockedStatus("question", "Question?"))

    context._reset_extension_ui()
    assert context.blocked_calls[-1] == ("extension-overlay:0", None)
    assert len(context._extension_overlays) == 0
    assert len(context.overlays) == 0


# Regression for #11431: replacing the editor while a dialog was open hid the dialog and left it unanswerable.
@pytest.mark.tonio
async def test_keeps_an_open_dialog_visible_when_the_editor_component_is_replaced(dialog_setup):
    context = _create_context()
    context.editor.on_submit = None
    context.editor.on_change = None
    context._default_editor = context.editor
    context._active_status_indicator = None
    confirmed = context._show_extension_confirm("Title", "Continue?")

    class CustomEditor:
        on_submit = None
        on_change = None

        def render(self, _width: int) -> list[str]:
            return ["custom editor"]

        def invalidate(self) -> None:
            pass

        def get_text(self) -> str:
            return ""

        def set_text(self, _text: str) -> None:
            pass

    custom_editor = CustomEditor()
    context._set_custom_editor_component(lambda _tui, _theme, _keybindings: custom_editor)

    assert "Title" in _visible(context)
    context.focused.handle_input("\r")
    assert await confirmed is True
    assert context.focused is custom_editor
    assert context._editor_container.children == [custom_editor]


def _stub_the_rest_of_reset(context: _DialogContext) -> None:
    """The rest of `_reset_extension_ui` is not under test here."""
    noop = lambda *_args: None
    for name in (
        "_clear_extension_terminal_input_listeners",
        "_set_custom_editor_component",
        "_set_extension_footer",
        "_set_extension_header",
        "_clear_extension_widgets",
        "_setup_autocomplete_provider",
        "_update_terminal_title",
        "_set_working_indicator",
        "_set_hidden_thinking_label",
    ):
        setattr(context, name, noop)
    context._footer_data_provider = SimpleNamespace(clear_extension_statuses=noop)
    context._extension_registry_guard = threading.Lock()
    context._default_editor = context.editor
    context._footer = SimpleNamespace(invalidate=noop)
    context._active_status_indicator = None


@pytest.mark.tonio
async def test_reset_disposes_every_dialog_when_one_closes_itself_from_dispose(dialog_setup):
    context = _create_context()
    _stub_the_rest_of_reset(context)
    disposed: set = set()

    def open_custom(label: str):
        class ClosesOnDispose(Text):
            def dispose(self) -> None:
                disposed.add(label)
                self.done(None)

        def factory(_tui, _theme, _keybindings, done):
            component = ClosesOnDispose(label)
            component.done = done
            return component

        return context._show_extension_custom(factory)

    first = open_custom("custom one")
    second = open_custom("custom two")

    context._reset_extension_ui()

    assert disposed == {"custom one", "custom two"}
    assert await first is None
    assert await second is None
    assert context.focused is context.editor
