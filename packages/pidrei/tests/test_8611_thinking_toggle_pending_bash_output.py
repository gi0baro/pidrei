"""Mirror of pi's suite/regressions/8611-thinking-toggle-pending-bash-output.test.ts.

pi grabs the private methods off InteractiveMode.prototype and calls them on a
fake `this`; the Python functions are called the same way on a stub object.
"""

import os
from types import SimpleNamespace

import pytest

from pidrei.modes.interactive.components import ToolExecutionComponent
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import Container


@pytest.fixture(autouse=True)
def _theme():
    init_theme_sync("dark")


def render_chat(container: Container) -> str:
    return strip_ansi("\n".join(container.render(120)))


@pytest.mark.tonio
async def test_preserves_partial_bash_output():
    # tonio-marked: the bash renderer starts its running-spinner Interval,
    # which needs a runtime to land in (detached timers outside one only
    # produce an unawaited-coroutine warning).
    render_requests: list = []
    ui = SimpleNamespace(request_render=lambda: render_requests.append(True))
    chat_container = Container()
    component = ToolExecutionComponent(
        "bash",
        "tool-8611",
        {"command": "echo first; sleep 10"},
        {"showImages": False},
        None,
        ui,
        os.getcwd(),
    )
    component.mark_execution_started()
    component.update_result({"content": [{"type": "text", "text": "first"}], "isError": False}, True)
    chat_container.add_child(component)

    set_hidden_calls: list = []
    status_calls: list = []
    fake_self = SimpleNamespace(
        _hide_thinking_block=False,
        settings_manager=SimpleNamespace(set_hide_thinking_block=set_hidden_calls.append),
        _chat_container=chat_container,
        ui=ui,
        show_status=status_calls.append,
    )
    fake_self._update_thinking_block_visibility = lambda: InteractiveMode._update_thinking_block_visibility(fake_self)

    assert "first" in render_chat(chat_container)
    InteractiveMode._toggle_thinking_block_visibility(fake_self)

    assert set_hidden_calls == [True]
    assert component in chat_container.children
    assert "first" in render_chat(chat_container)
