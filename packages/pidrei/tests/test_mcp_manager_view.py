"""Mirror of pi's mcp-manager-view.test.ts."""

import pytest

from pidrei.core.keybindings import KeybindingsManager
from pidrei.extensions.mcp.ui import McpManagerView
from pidrei.modes.interactive.theme import init_theme, theme
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import set_keybindings


ESCAPE = "\x1b"


class _FakeTui:
    """What the manager view uses of its TUI."""

    def request_render(self, force: bool = False) -> None:
        pass

    def apply(self, fn):
        return fn()


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
