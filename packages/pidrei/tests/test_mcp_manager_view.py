"""Mirror of pi's mcp-manager-view.test.ts."""

import pytest
import tonio.colored as tonio

from pidrei.core.keybindings import KeybindingsManager
from pidrei.extensions.mcp.ui import McpManagerView, McpMenu
from pidrei.modes.interactive.theme import init_theme, theme
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import set_keybindings


ESCAPE = "\x1b"


class _FakeTui:
    """What the manager view uses of its TUI. `applied` is set after each
    `apply`, so a test can wait for a screen a spawned `menu` shows."""

    def __init__(self) -> None:
        self.applied = tonio.Event()

    def request_render(self, force: bool = False) -> None:
        pass

    def apply(self, fn):
        result = fn()
        self.applied.set()
        return result


def rendered(view: McpManagerView) -> str:
    return strip_ansi("\n".join(view.render(80)))


@pytest.fixture(autouse=True)
async def _setup():
    await init_theme("dark")
    # Keybindings are a global singleton; reset per test.
    set_keybindings(KeybindingsManager.in_memory())


# pi #10565
@pytest.mark.tonio
async def test_cancels_the_running_operation_with_the_cancel_key():
    view = McpManagerView(_FakeTui(), theme, KeybindingsManager.in_memory())
    cancels: list[bool] = []
    view.status("Sign in to issues", "Contacting the authorization server…", lambda: cancels.append(True))
    assert "cancel" in rendered(view)
    view.handle_input(ESCAPE)
    assert cancels == [True]


@pytest.mark.tonio
async def test_ignores_the_cancel_key_for_operations_that_cannot_be_cancelled():
    view = McpManagerView(_FakeTui(), theme, KeybindingsManager.in_memory())
    view.status("Sign in to issues", "Connecting…")
    assert "cancel" not in rendered(view)
    view.handle_input(ESCAPE)


# MCP manager menu


# #7629
@pytest.mark.tonio
async def test_captures_fullscreen_viewport_keys_only_while_a_menu_list_is_shown():
    tui = _FakeTui()
    view = McpManagerView(tui, theme, KeybindingsManager.in_memory())
    items = [{"value": f"server-{index}", "label": f"server-{index}"} for index in range(20)]
    # pi's `menu` renders before it returns its promise; here it is a coroutine, so it is spawned
    # and the test waits for the screen it shows.
    result = tonio.spawn(
        view.menu(lambda: McpMenu(title="MCP servers", items=items, confirm_label="open", cancel_label="close"))
    )
    await tui.applied.wait(5)
    assert tui.applied.is_set()
    assert view.captures_viewport_input is True

    view.handle_input("\x1b[6~")  # PageDown
    view.handle_input("\r")
    assert await result == "server-12"

    view.status("Sign in to issues", "Connecting…")
    assert view.captures_viewport_input is False
