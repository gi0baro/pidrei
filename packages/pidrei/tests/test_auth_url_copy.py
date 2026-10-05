"""Mirror of pi's auth-url-copy.test.ts.

pi mocks `copyToClipboard` and `openBrowser` with `vi.mock`; here the names the
components import are monkeypatched. pi's `vi.waitFor` on the rendered hint
becomes an Event the fake TUI sets when the copy's hint update renders.
"""

import threading
from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.core.keybindings import KeybindingsManager
from pidrei.extensions.mcp.ui import McpManagerView
from pidrei.modes.interactive.components import auth_url, login_dialog
from pidrei.modes.interactive.components.login_dialog import LoginDialogComponent
from pidrei.modes.interactive.theme import init_theme, theme
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import set_keybindings
from pidrei_utils.cancel import CancelToken


URL = f"https://auth.example.invalid/authorize?{'x' * 300}"
CTRL_X = "\x18"


class _FakeTui:
    """What the login dialog and the MCP manager view use of their TUI."""

    def __init__(self) -> None:
        self.state_lock = threading.RLock()
        self.terminal = SimpleNamespace(write_sync=lambda _data: None)
        self.rendered = tonio.Event()
        self.applied = tonio.Event()

    def request_render(self, force: bool = False) -> None:
        self.rendered.set()

    def apply(self, fn):
        with self.state_lock:
            result = fn()
        self.applied.set()
        return result

    def spawn(self, coro) -> None:
        tonio.spawn.without_tracking(coro)


def rendered(component) -> str:
    return strip_ansi("\n".join(component.render(80)))


@pytest.fixture(autouse=True)
async def _setup(monkeypatch):
    await init_theme("dark")
    # Keybindings are a global singleton; reset per test.
    set_keybindings(KeybindingsManager.in_memory())
    monkeypatch.setattr(login_dialog, "open_browser", lambda _url: None)


@pytest.fixture
def copied(monkeypatch) -> list[str]:
    calls: list[str] = []

    async def copy_to_clipboard(text: str, _write_terminal) -> None:
        calls.append(text)

    monkeypatch.setattr(auth_url, "copy_to_clipboard", copy_to_clipboard)
    return calls


@pytest.mark.tonio
async def test_login_dialog_copies_the_auth_url_instead_of_typing_into_the_code_input(copied):
    tui = _FakeTui()
    dialog = LoginDialogComponent(tui, "test", lambda *_args: None)
    dialog.show_auth(URL)
    # pi's `void dialog.showManualInput(...)`: the input is never submitted, so its wait is closed unstarted.
    pending_input = dialog.show_manual_input("Paste the code:")
    assert "ctrl+x to copy" in rendered(dialog)

    tui.rendered = tonio.Event()
    dialog.handle_input(CTRL_X)
    await tui.rendered.wait(5)
    assert tui.rendered.is_set()
    assert "Copied URL to clipboard" in rendered(dialog)
    assert copied == [URL]
    pending_input.close()


@pytest.mark.tonio
async def test_login_dialog_ignores_the_copy_key_without_an_auth_url(copied):
    dialog = LoginDialogComponent(_FakeTui(), "test", lambda *_args: None)
    dialog.show_device_code({"userCode": "ABCD", "verificationUri": "https://example.invalid/device"})
    dialog.handle_input(CTRL_X)
    assert copied == []


@pytest.mark.tonio
async def test_mcp_sign_in_screen_copies_the_authorization_url(copied):
    tui = _FakeTui()
    view = McpManagerView(tui, theme, KeybindingsManager.in_memory())
    cancel = CancelToken()
    answer = tonio.spawn(view.redirect_url("Sign in to issues", URL, cancel))
    await tui.applied.wait(5)
    assert tui.applied.is_set()
    assert "ctrl+x to copy" in rendered(view)

    # `handle_input` itself requests a frame; the copy's hint update is the next apply.
    tui.applied = tonio.Event()
    view.handle_input(CTRL_X)
    await tui.applied.wait(5)
    assert tui.applied.is_set()
    assert "Copied URL to clipboard" in rendered(view)
    assert copied == [URL]

    cancel.cancel()
    assert await answer is None
