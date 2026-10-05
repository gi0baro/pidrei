"""Mirror of pi coding-agent src/modes/interactive/components/auth-url.ts.

`tui` is the TUI or the `ExtensionTui` the MCP manager holds: the component
uses only what both provide (`apply`, `request_render`,
`terminal.write_sync`). It is built under the UI state lock by its host;
`copy()` runs on its own coroutine (pi's `void link.copy()`), so its hint
update goes through `tui.apply`.
"""

import sys

from pidrei_tui import Container, Text, hyperlink

from ....utils.clipboard import copy_to_clipboard
from ..theme import theme
from .keybinding_hints import key_hint


class AuthUrlComponent(Container):
    """A sign-in URL with a click hint and a copy hint. Hosts spawn `copy()` when
    `app.message.copy` is pressed, since a long URL wraps and often cannot be
    selected or clicked as a whole (SSH, tmux)."""

    def __init__(self, tui, url: str) -> None:
        super().__init__()
        self._tui = tui
        self.url = url
        self.add_child(Text(theme.fg("accent", hyperlink(url, url)), 1, 0))
        self._hint = Text("", 1, 0)
        self.add_child(self._hint)
        self._set_hint(key_hint("app.message.copy", "to copy"))

    def _set_hint(self, suffix: str) -> None:
        click_hint = "Cmd+click to open" if sys.platform == "darwin" else "Ctrl+click to open"
        self._hint.set_text(f"{theme.fg('dim', hyperlink(click_hint, self.url))} {theme.fg('dim', '•')} {suffix}")
        self._tui.request_render()

    async def copy(self) -> None:
        try:
            await copy_to_clipboard(self.url, self._tui.terminal.write_sync)
            suffix = theme.fg("success", "Copied URL to clipboard")
        except Exception as error:
            suffix = theme.fg("error", str(error))
        self._tui.apply(lambda: self._set_hint(suffix))
