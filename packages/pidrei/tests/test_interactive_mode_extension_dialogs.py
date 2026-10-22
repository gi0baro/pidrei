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
from pidrei_tui import Container, Text, set_keybindings
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
        _program_status=SimpleNamespace(set_blocked=lambda key, status: blocked_calls.append((key, status))),
        _dispose_active_selector=noop,
        _toggle_tool_output_expansion=noop,
        _extension_tui=None,
        _keybindings=None,
    )
    context.ui = SimpleNamespace(
        state_lock=threading.RLock(),
        set_focus=lambda component: setattr(context, "focused", component),
        request_render=noop,
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
