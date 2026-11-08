"""Mirror of pi coding-agent test/login-dialog-device-code.test.ts.

pi mocks `copyToClipboard` and `openBrowser` with `vi.mock`; here the names the
components import are monkeypatched. pi's `await Promise.resolve()` after
Enter becomes waiting for the copy's result status, which lands from the
copy's own coroutine.
"""

import threading
from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive.components import auth_url, login_dialog
from pidrei.modes.interactive.components.login_dialog import LoginDialogComponent
from pidrei.modes.interactive.theme import init_theme
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import set_keybindings


DEVICE = {"userCode": "ABCD-EFGH", "verificationUri": "https://example.com/device"}
ENTER = "\r"
CTRL_X = "\x18"


class _FakeTui:
    def __init__(self) -> None:
        self.state_lock = threading.RLock()
        self.terminal = SimpleNamespace(write_sync=lambda _data: None)

    def request_render(self, force: bool = False) -> None:
        pass

    def apply(self, fn):
        with self.state_lock:
            return fn()


class _Calls:
    def __init__(self) -> None:
        self.browser: list[str] = []
        self.clipboard: list[str] = []
        self.browser_error: Exception | None = None
        self.clipboard_error: Exception | None = None


@pytest.fixture
async def calls(monkeypatch) -> _Calls:
    await init_theme("dark")
    # Keybindings are a global singleton; reset per test.
    set_keybindings(KeybindingsManager.in_memory())
    recorded = _Calls()

    def open_browser(url: str) -> None:
        recorded.browser.append(url)
        if recorded.browser_error is not None:
            raise recorded.browser_error

    async def copy_to_clipboard(text: str, _write_terminal) -> None:
        recorded.clipboard.append(text)
        if recorded.clipboard_error is not None:
            raise recorded.clipboard_error

    monkeypatch.setattr(login_dialog, "open_browser", open_browser)
    monkeypatch.setattr(login_dialog, "copy_to_clipboard", copy_to_clipboard)
    monkeypatch.setattr(auth_url, "copy_to_clipboard", copy_to_clipboard)
    return recorded


def create_dialog() -> tuple[LoginDialogComponent, list]:
    completions: list = []
    dialog = LoginDialogComponent(_FakeTui(), "custom-provider", lambda *args: completions.append(args))
    dialog.focused = True
    return dialog, completions


def copy_settled(dialog: LoginDialogComponent) -> tonio.Event:
    """Set once the copy's result status has been applied."""
    settled = tonio.Event()
    set_status = dialog._set_device_code_status

    def status(hint, message: str) -> None:
        set_status(hint, message)
        if message in ("Code copied to clipboard", "Copy code manually"):
            settled.set()

    dialog._set_device_code_status = status
    return settled


def render(dialog: LoginDialogComponent) -> str:
    return strip_ansi("\n".join(dialog.render(120)))


# Regression #9282: device-code convenience actions require explicit confirmation.


@pytest.mark.tonio
async def test_opens_the_verification_page_and_copies_the_code_once_after_confirmation(calls):
    dialog, _ = create_dialog()
    domain = dialog.show_prompt("GitHub Enterprise URL/domain (blank for github.com)")
    dialog.handle_input(ENTER)
    assert await domain == ""
    dialog.show_device_code(DEVICE)
    dialog.show_waiting("Waiting for authentication...")
    assert "to open browser and copy code" in render(dialog)
    assert calls.browser == []
    assert calls.clipboard == []

    settled = copy_settled(dialog)
    dialog.handle_input(ENTER)
    await settled.wait(5)
    dialog.handle_input(ENTER)
    assert calls.browser == [DEVICE["verificationUri"]]
    assert calls.clipboard == [DEVICE["userCode"]]
    output = render(dialog)
    assert "Code copied to clipboard" in output
    assert f"Enter code: {DEVICE['userCode']}" in output
    assert "Waiting for authentication..." in output
    assert "to open browser and copy code" not in output


@pytest.mark.parametrize(
    ("open_browser_on_confirm", "copy_code_on_confirm"), [(False, False), (True, False), (False, True)]
)
@pytest.mark.tonio
async def test_honors_browser_and_clipboard_provider_overrides(calls, open_browser_on_confirm, copy_code_on_confirm):
    dialog, _ = create_dialog()
    settled = copy_settled(dialog)
    dialog.show_device_code(
        {**DEVICE, "openBrowserOnConfirm": open_browser_on_confirm, "copyCodeOnConfirm": copy_code_on_confirm}
    )
    dialog.handle_input(ENTER)
    if copy_code_on_confirm:
        await settled.wait(5)
    assert len(calls.browser) == (1 if open_browser_on_confirm else 0)
    assert len(calls.clipboard) == (1 if copy_code_on_confirm else 0)
    assert dialog.signal.cancelled is False


@pytest.mark.tonio
async def test_clears_the_previous_auth_url_copy_action_when_showing_a_device_code(calls):
    dialog, _ = create_dialog()
    dialog.show_auth("https://example.com/authorize")
    calls.browser.clear()
    dialog.show_device_code(DEVICE)
    dialog.handle_input(CTRL_X)
    assert calls.clipboard == []
    assert calls.browser == []
    settled = copy_settled(dialog)
    dialog.handle_input(ENTER)
    await settled.wait(5)
    assert calls.clipboard == [DEVICE["userCode"]]


@pytest.mark.tonio
async def test_clears_device_actions_when_switching_to_browser_authentication(calls):
    dialog, _ = create_dialog()
    dialog.show_device_code(DEVICE)
    dialog.show_auth("https://example.com/authorize")
    calls.browser.clear()
    dialog.handle_input(ENTER)
    assert calls.clipboard == []
    assert calls.browser == []
    copied = tonio.Event()
    copy_url = dialog._auth_url.copy

    async def copy() -> None:
        await copy_url()
        copied.set()

    dialog._auth_url.copy = copy
    dialog.handle_input(CTRL_X)
    await copied.wait(5)
    assert calls.clipboard == ["https://example.com/authorize"]


@pytest.mark.tonio
async def test_clears_both_action_types_when_showing_details(calls):
    dialog, _ = create_dialog()
    dialog.show_auth("https://example.com/authorize")
    dialog.show_details(["Choose a login method"])
    dialog.handle_input(CTRL_X)
    assert calls.clipboard == []
    dialog.show_device_code(DEVICE)
    dialog.show_details(["Choose a login method"])
    calls.browser.clear()
    dialog.handle_input(ENTER)
    assert calls.clipboard == []
    assert calls.browser == []


@pytest.mark.tonio
async def test_cancels_without_opening_a_browser_or_changing_the_clipboard(calls):
    dialog, completions = create_dialog()
    dialog.show_device_code(DEVICE)
    dialog.handle_input("\x1b")
    dialog.handle_input(ENTER)
    assert dialog.signal.cancelled is True
    assert completions == [(False, "Login cancelled")]
    assert calls.browser == []
    assert calls.clipboard == []


@pytest.mark.tonio
async def test_keeps_manual_login_available_when_browser_and_clipboard_access_fail(calls):
    calls.browser_error = Exception("Browser unavailable")
    calls.clipboard_error = Exception("Clipboard unavailable")
    dialog, completions = create_dialog()
    dialog.show_device_code(DEVICE)
    dialog.show_waiting("Waiting for authentication...")
    settled = copy_settled(dialog)
    dialog.handle_input(ENTER)
    await settled.wait(5)
    assert calls.browser == [DEVICE["verificationUri"]]
    assert calls.clipboard == [DEVICE["userCode"]]
    output = render(dialog)
    assert DEVICE["verificationUri"] in output
    assert f"Enter code: {DEVICE['userCode']}" in output
    assert "Copy code manually" in output
    assert "Waiting for authentication..." in output
    assert dialog.signal.cancelled is False
    assert completions == []


# login URLs


@pytest.mark.tonio
async def test_opens_only_http_s_sign_in_urls(calls):
    dialog, _ = create_dialog()
    dialog.show_auth("file:///Users/victim/Downloads/payload.app")
    assert calls.browser == []

    dialog.show_auth("https://example.com/authorize")
    assert calls.browser == ["https://example.com/authorize"]


@pytest.mark.tonio
async def test_keeps_control_characters_in_urls_from_ending_terminal_hyperlinks(calls):
    dialog, _ = create_dialog()
    dialog.show_device_code({**DEVICE, "verificationUri": "https://example.com/device\x07\x1b]52;c;Zm9v\x07"})
    dialog.show_info("Docs", [{"label": "Help", "url": "https://example.com/help\x07\x1b]52;c;Zm9v\x07"}])
    output = "\n".join(dialog.render(120))

    assert "\x1b]52" not in output
    assert "https://example.com/device%07%1B]52;c;Zm9v" in render(dialog)
    assert "Help: https://example.com/help%07%1B]52;c;Zm9v" in render(dialog)


@pytest.mark.tonio
async def test_replaces_control_characters_in_urls_that_are_not_web_urls(calls):
    injected = "x\x07\x1b]52;c;Zm9v\x07"
    dialog, _ = create_dialog()
    dialog.show_auth(f"file:///tmp/{injected}")
    assert "\x1b]52" not in "\n".join(dialog.render(120))

    dialog.show_device_code({**DEVICE, "verificationUri": injected})
    assert "\x1b]52" not in "\n".join(dialog.render(120))
    assert "x ]52;c;Zm9v" in render(dialog)

    dialog.show_info("Docs", [{"label": "Help", "url": injected}])
    assert "\x1b]52" not in "\n".join(dialog.render(120))
    assert "Help: x ]52;c;Zm9v" in render(dialog)
    assert calls.browser == []
