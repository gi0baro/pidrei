"""Mirror of pi coding-agent src/modes/interactive/components/login-dialog.ts.

Login dialog component - replaces the editor during OAuth login flows.
pi's manual-input/prompt Promises become tonio-Event-backed awaitables; the
AbortController is the ai-layer CancelToken because the dialog's signal is
handed to `model_runtime.login`, which relies on the full ai token contract
(`raise_if_cancelled`, `reason`, `never`). Escape cancels with
`LoginCancelledError` so the abort surfaces as "Login cancelled" (which the
interactive mode suppresses) rather than as a login failure.
"""

import sys

import tonio.colored as tonio

from pidrei_ai.auth.oauth.urls import trusted_http_url
from pidrei_tui import (
    Container,
    Input,
    Spacer,
    Text,
    TruncatedText,
    get_keybindings,
    hyperlink,
    replace_control_characters,
)
from pidrei_utils.cancel import CancelToken

from ....utils.clipboard import copy_to_clipboard
from ....utils.open_browser import open_browser
from ....utils.web_url import display_url
from ..theme import theme
from .auth_url import AuthUrlComponent
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint


# pi's `webUrl` is `trusted_http_url` (see `utils/web_url.py`).


def _web_link(text: str, url: str) -> str:
    """`text` as a terminal hyperlink to `url` when that is a web URL, else plain `text`."""
    target = trusted_http_url(url)
    return text if target is None else hyperlink(text, target)


class LoginCancelledError(Exception):
    def __init__(self) -> None:
        super().__init__("Login cancelled")


class LoginDialogComponent(Container):
    def __init__(self, tui, provider_id: str, on_complete, provider_name_override=None, title_override=None) -> None:
        super().__init__()
        self._tui = tui
        self._on_complete = on_complete
        self._abort_controller = CancelToken()
        # The shown sign-in URL, which `app.message.copy` copies.
        self._auth_url: AuthUrlComponent | None = None
        self._input_event: tonio.Event | None = None
        self._input_value: str | None = None
        self._input_error: Exception | None = None
        # The device code's pending confirm action, and the hint row it reuses for progress/result text.
        self._device_code_action: tuple[dict, Container] | None = None
        self._device_code_hint: Container | None = None
        self._focused = False

        provider_name = provider_name_override or provider_id
        title = title_override if title_override is not None else f"Login to {provider_name}"

        # Top border
        self.add_child(DynamicBorder())

        # Title
        self.add_child(Text(theme.fg("accent", theme.bold(title)), 1, 0))

        # Dynamic content area
        self._content_container = Container()
        self.add_child(self._content_container)

        # Input (always present, used when needed)
        self._input = Input()

        def on_submit(_value=None) -> None:
            if self._input_event is not None and not self._input_event.is_set():
                value = self._input.get_value()
                self._replace_input_with_submitted_text(value)
                self._input_value = value
                self._input_error = None
                event = self._input_event
                self._input_event = None
                event.set()

        self._input.on_submit = on_submit
        self._input.on_escape = lambda: self._cancel()

        # Bottom border
        self.add_child(DynamicBorder())

    # Focusable implementation - propagate to input for IME cursor positioning
    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._input.focused = value

    @property
    def signal(self) -> CancelToken:
        return self._abort_controller

    def _replace_input_with_submitted_text(self, value: str) -> None:
        self._content_container.children = [
            Text(f"> {value}", 0, 0) if child is self._input else child for child in self._content_container.children
        ]

    def _clear_device_code_action(self) -> None:
        if self._device_code_hint is not None:
            self._content_container.remove_child(self._device_code_hint)
        self._device_code_hint = None
        self._device_code_action = None

    def _cancel(self) -> None:
        # On the input path, which holds the UI state lock.
        self._clear_device_code_action()
        self._abort_controller.cancel(LoginCancelledError())
        if self._input_event is not None and not self._input_event.is_set():
            self._input_error = LoginCancelledError()
            event = self._input_event
            self._input_event = None
            event.set()
        self._on_complete(False, "Login cancelled")

    # The show_* steps are driven by the login task while the dialog may be
    # mounted and taking input: each content change applies under the UI
    # state lock, in call order.

    def _post_content(self, apply) -> None:
        with self._tui.state_lock:
            apply()
            self._tui.request_render()

    def show_auth(self, url: str, instructions: str | None = None) -> None:
        """Called by on_auth callback - show URL and optional instructions."""

        target = trusted_http_url(url)

        def apply() -> None:
            self._clear_device_code_action()
            self._content_container.clear()
            self._content_container.add_child(Spacer(1))
            self._auth_url = AuthUrlComponent(self._tui, target or replace_control_characters(url))
            self._content_container.add_child(self._auth_url)

            if instructions:
                self._content_container.add_child(Spacer(1))
                self._content_container.add_child(Text(theme.fg("warning", instructions), 1, 0))

        self._post_content(apply)
        if target is not None:
            open_browser(target)

    def show_device_code(self, info: dict) -> None:
        """Called by on_device_code callback - show URL and user code.

        ``info`` is a ``{"verificationUri", "userCode", "openBrowserOnConfirm"?,
        "copyCodeOnConfirm"?}`` record; the two flags default to true.
        """
        info = {
            **info,
            "openBrowserOnConfirm": info.get("openBrowserOnConfirm") is not False,
            "copyCodeOnConfirm": info.get("copyCodeOnConfirm") is not False,
        }

        def apply() -> None:
            self._auth_url = None
            self._clear_device_code_action()
            self._content_container.clear()
            self._content_container.add_child(Spacer(1))
            shown_url = display_url(info["verificationUri"])
            self._content_container.add_child(Text(theme.fg("accent", _web_link(shown_url, shown_url)), 1, 0))

            click_hint = "Cmd+click to open" if sys.platform == "darwin" else "Ctrl+click to open"
            self._content_container.add_child(
                Text(theme.fg("dim", _web_link(click_hint, info["verificationUri"])), 1, 0)
            )
            self._content_container.add_child(Spacer(1))
            self._content_container.add_child(Text(theme.fg("warning", f"Enter code: {info['userCode']}"), 1, 0))

            if not self.signal.cancelled and (info["openBrowserOnConfirm"] or info["copyCodeOnConfirm"]):
                if info["openBrowserOnConfirm"]:
                    action_label = "to open browser and copy code" if info["copyCodeOnConfirm"] else "to open browser"
                else:
                    action_label = "to copy code"
                # Reuse this single row for progress/result text, including in narrow terminals.
                hint = Container()
                hint.add_child(TruncatedText(f"({key_hint('tui.select.confirm', action_label)})", 1, 0))
                self._content_container.add_child(hint)
                self._device_code_hint = hint
                self._device_code_action = (info, hint)

        self._post_content(apply)

    def _set_device_code_status(self, hint: Container, message: str) -> None:
        def apply() -> None:
            if self.signal.cancelled or not self._focused or hint not in self._content_container.children:
                return
            hint.children = [TruncatedText(theme.fg("dim", message), 1, 0)]

        self._post_content(apply)

    def _activate_device_code(self, info: dict, hint: Container) -> None:
        self._set_device_code_status(hint, "Copying code..." if info["copyCodeOnConfirm"] else "Browser open requested")
        if info["copyCodeOnConfirm"]:

            async def copy() -> None:
                try:
                    await copy_to_clipboard(info["userCode"], self._tui.terminal.write_sync)
                except Exception:
                    self._set_device_code_status(hint, "Copy code manually")
                else:
                    self._set_device_code_status(hint, "Code copied to clipboard")

            # Do not delay polling or browser opening while clipboard tools are running.
            tonio.spawn.without_tracking(copy())
        if info["openBrowserOnConfirm"]:
            try:
                target = trusted_http_url(info["verificationUri"])
                if target is not None:
                    open_browser(target)
            except Exception:
                pass  # Launcher failures must not interrupt login.

    def _await_input(self):
        self._input_event = tonio.Event()

        async def wait(event: tonio.Event):
            await event.wait(None)
            if self._input_error is not None:
                error = self._input_error
                self._input_error = None
                raise error
            return self._input_value

        return wait(self._input_event)

    def show_manual_input(self, prompt: str):
        """Show input for manual code/URL entry (for callback server providers).

        Returns an awaitable resolving to the submitted value.
        """
        # The wait is published before the input can be shown (and submitted).
        response = self._await_input()

        def apply() -> None:
            self._clear_device_code_action()
            self._input.set_value("")
            self._content_container.add_child(Spacer(1))
            self._content_container.add_child(Text(theme.fg("dim", prompt), 1, 0))
            self._content_container.add_child(self._input)
            self._content_container.add_child(Text(f"({key_hint('tui.select.cancel', 'to cancel')})", 1, 0))

        self._post_content(apply)
        return response

    def show_prompt(self, message: str, placeholder: str | None = None):
        """Called by on_prompt callback - show prompt and wait for input.

        Does NOT clear content, appends to existing (preserves URL from
        show_auth). Returns an awaitable resolving to the submitted value.
        """
        # The wait is published before the input can be shown (and submitted).
        response = self._await_input()

        def apply() -> None:
            self._clear_device_code_action()
            self._content_container.add_child(Spacer(1))
            self._content_container.add_child(Text(theme.fg("text", message), 1, 0))
            if placeholder:
                self._content_container.add_child(Text(theme.fg("dim", f"e.g., {placeholder}"), 1, 0))
            self._content_container.add_child(self._input)
            self._content_container.add_child(
                Text(
                    f"({key_hint('tui.select.cancel', 'to cancel,')} {key_hint('tui.select.confirm', 'to submit')})",
                    1,
                    0,
                )
            )
            self._input.set_value("")

        self._post_content(apply)
        return response

    def show_details(self, lines: list) -> None:
        """Show informational text before another login step."""

        def apply() -> None:
            self._auth_url = None
            self._clear_device_code_action()
            self._content_container.clear()
            self._content_container.add_child(Spacer(1))
            for line in lines:
                self._content_container.add_child(Text(line, 1, 0))

        self._post_content(apply)

    def show_info(self, message: str, links: list | None = None, show_close_hint: bool = False) -> None:
        """Show provider-owned information and links without an auth flow.

        Links are ``{"url", "label"?}`` records.
        """
        links = links or []

        def apply() -> None:
            self._content_container.add_child(Spacer(1))
            self._content_container.add_child(Text(theme.fg("text", message), 1, 0))
            for link in links:
                url = display_url(link["url"])
                text = f"{link['label']}: {url}" if link.get("label") else url
                self._content_container.add_child(Text(theme.fg("accent", _web_link(text, url)), 1, 0))
            if show_close_hint:
                self._content_container.add_child(Spacer(1))
                self._content_container.add_child(Text(f"({key_hint('tui.select.cancel', 'to close')})", 1, 0))

        self._post_content(apply)

    def show_waiting(self, message: str) -> None:
        """Show waiting message (for polling flows like GitHub Copilot)."""

        def apply() -> None:
            self._content_container.add_child(Spacer(1))
            self._content_container.add_child(Text(theme.fg("dim", message), 1, 0))
            self._content_container.add_child(Text(f"({key_hint('tui.select.cancel', 'to cancel')})", 1, 0))

        self._post_content(apply)

    def show_progress(self, message: str) -> None:
        """Called by on_progress callback."""
        self._post_content(lambda: self._content_container.add_child(Text(theme.fg("dim", message), 1, 0)))

    def handle_input(self, data: str) -> None:
        kb = get_keybindings()

        if kb.matches(data, "tui.select.cancel"):
            self._cancel()
            return
        if self._auth_url is not None and kb.matches(data, "app.message.copy"):
            # pi's `void this.authUrl.copy()`; `copy` reports its own failure in the hint.
            tonio.spawn.without_tracking(self._auth_url.copy())
            return

        if (
            self._device_code_action is not None
            and self._input_event is None
            and not self.signal.cancelled
            and self._focused
            and kb.matches(data, "tui.select.confirm")
        ):
            info, hint = self._device_code_action
            # Consume once so a repeated Enter cannot launch multiple browsers or clipboard writes.
            self._device_code_action = None
            self._activate_device_code(info, hint)
            self._tui.request_render()
            return

        # Pass to input
        self._input.handle_input(data)
