"""Mirror of pi coding-agent src/modes/interactive/components/login-dialog.ts.

Login dialog component - replaces the editor during OAuth login flows.
pi's manual-input/prompt Promises become tonio-Event-backed awaitables; the
AbortController is the tui-local CancelToken.
"""

import sys

import tonio.colored as tonio

from pidrei_tui import Container, Input, Spacer, Text, get_keybindings
from pidrei_tui.components.cancellable_loader import CancelToken

from ....utils.open_browser import open_browser
from ..theme import theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint


class LoginCancelledError(Exception):
    def __init__(self) -> None:
        super().__init__("Login cancelled")


class LoginDialogComponent(Container):
    def __init__(self, tui, provider_id: str, on_complete, provider_name_override=None, title_override=None) -> None:
        super().__init__()
        self._tui = tui
        self._on_complete = on_complete
        self._abort_controller = CancelToken()
        self._input_event: tonio.Event | None = None
        self._input_value: str | None = None
        self._input_error: Exception | None = None
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

    def _cancel(self) -> None:
        self._abort_controller.cancel()
        if self._input_event is not None and not self._input_event.is_set():
            self._input_error = LoginCancelledError()
            event = self._input_event
            self._input_event = None
            event.set()
        self._on_complete(False, "Login cancelled")

    def show_auth(self, url: str, instructions: str | None = None) -> None:
        """Called by on_auth callback - show URL and optional instructions."""
        self._content_container.clear()
        self._content_container.add_child(Spacer(1))
        linked_url = f"\x1b]8;;{url}\x07{url}\x1b]8;;\x07"
        self._content_container.add_child(Text(theme.fg("accent", linked_url), 1, 0))

        click_hint = "Cmd+click to open" if sys.platform == "darwin" else "Ctrl+click to open"
        hyperlink = f"\x1b]8;;{url}\x07{click_hint}\x1b]8;;\x07"
        self._content_container.add_child(Text(theme.fg("dim", hyperlink), 1, 0))

        if instructions:
            self._content_container.add_child(Spacer(1))
            self._content_container.add_child(Text(theme.fg("warning", instructions), 1, 0))

        open_browser(url)
        self._tui.request_render()

    def show_device_code(self, info: dict) -> None:
        """Called by on_device_code callback - show URL and user code.

        ``info`` is a ``{"verificationUri", "userCode"}`` record.
        """
        self._content_container.clear()
        self._content_container.add_child(Spacer(1))
        verification_uri = info["verificationUri"]
        linked_url = f"\x1b]8;;{verification_uri}\x07{verification_uri}\x1b]8;;\x07"
        self._content_container.add_child(Text(theme.fg("accent", linked_url), 1, 0))

        click_hint = "Cmd+click to open" if sys.platform == "darwin" else "Ctrl+click to open"
        hyperlink = f"\x1b]8;;{verification_uri}\x07{click_hint}\x1b]8;;\x07"
        self._content_container.add_child(Text(theme.fg("dim", hyperlink), 1, 0))
        self._content_container.add_child(Spacer(1))
        self._content_container.add_child(Text(theme.fg("warning", f"Enter code: {info['userCode']}"), 1, 0))

        self._tui.request_render()

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
        self._input.set_value("")
        self._content_container.add_child(Spacer(1))
        self._content_container.add_child(Text(theme.fg("dim", prompt), 1, 0))
        self._content_container.add_child(self._input)
        self._content_container.add_child(Text(f"({key_hint('tui.select.cancel', 'to cancel')})", 1, 0))
        self._tui.request_render()

        return self._await_input()

    def show_prompt(self, message: str, placeholder: str | None = None):
        """Called by on_prompt callback - show prompt and wait for input.

        Does NOT clear content, appends to existing (preserves URL from
        show_auth). Returns an awaitable resolving to the submitted value.
        """
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
        self._tui.request_render()

        return self._await_input()

    def show_details(self, lines: list) -> None:
        """Show informational text before another login step."""
        self._content_container.clear()
        self._content_container.add_child(Spacer(1))
        for line in lines:
            self._content_container.add_child(Text(line, 1, 0))
        self._tui.request_render()

    def show_info(self, message: str, links: list | None = None, show_close_hint: bool = False) -> None:
        """Show provider-owned information and links without an auth flow.

        Links are ``{"url", "label"?}`` records.
        """
        links = links or []
        self._content_container.add_child(Spacer(1))
        self._content_container.add_child(Text(theme.fg("text", message), 1, 0))
        for link in links:
            text = f"{link['label']}: {link['url']}" if link.get("label") else link["url"]
            hyperlink = f"\x1b]8;;{link['url']}\x07{text}\x1b]8;;\x07"
            self._content_container.add_child(Text(theme.fg("accent", hyperlink), 1, 0))
        if show_close_hint:
            self._content_container.add_child(Spacer(1))
            self._content_container.add_child(Text(f"({key_hint('tui.select.cancel', 'to close')})", 1, 0))
        self._tui.request_render()

    def show_waiting(self, message: str) -> None:
        """Show waiting message (for polling flows like GitHub Copilot)."""
        self._content_container.add_child(Spacer(1))
        self._content_container.add_child(Text(theme.fg("dim", message), 1, 0))
        self._content_container.add_child(Text(f"({key_hint('tui.select.cancel', 'to cancel')})", 1, 0))
        self._tui.request_render()

    def show_progress(self, message: str) -> None:
        """Called by on_progress callback."""
        self._content_container.add_child(Text(theme.fg("dim", message), 1, 0))
        self._tui.request_render()

    def handle_input(self, data: str) -> None:
        kb = get_keybindings()

        if kb.matches(data, "tui.select.cancel"):
            self._cancel()
            return

        # Pass to input
        self._input.handle_input(data)
