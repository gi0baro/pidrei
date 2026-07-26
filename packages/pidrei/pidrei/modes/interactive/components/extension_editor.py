"""Mirror of pi coding-agent src/modes/interactive/components/extension-editor.ts.

Multi-line editor component for extensions. Supports Ctrl+G for external
editor.
"""

import os

import tonio.colored as tonio

from pidrei_tui import Container, Editor, Spacer, Text, get_keybindings

from ..external_editor import edit_in_external_editor
from ..theme import get_editor_theme, theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint


class ExtensionEditorComponent(Container):
    def __init__(
        self,
        tui,
        keybindings,
        title: str,
        prefill: str | None,
        on_submit,
        on_cancel,
        options: dict | None = None,
        external_editor_command: str | None = None,
    ) -> None:
        super().__init__()

        self._tui = tui
        self._keybindings = keybindings
        self._external_editor_command = (
            external_editor_command or os.environ.get("VISUAL") or os.environ.get("EDITOR") or "nano"
        )
        self._on_submit_callback = on_submit
        self._on_cancel_callback = on_cancel
        self._focused = False

        # Add top border
        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))

        # Add title
        self.add_child(Text(theme.fg("accent", title), 1, 0))
        self.add_child(Spacer(1))

        # Create editor
        self._editor = Editor(tui, get_editor_theme(), options)
        if prefill:
            self._editor.set_text(prefill)
        # Wire up Enter to submit (Shift+Enter for newlines, like the main editor)
        self._editor.on_submit = lambda text: self._on_submit_callback(text)
        self.add_child(self._editor)

        self.add_child(Spacer(1))

        # Add hint
        hint = (
            key_hint("tui.select.confirm", "submit")
            + "  "
            + key_hint("tui.input.newLine", "newline")
            + "  "
            + key_hint("tui.select.cancel", "cancel")
            + f"  {key_hint('app.editor.external', 'external editor')}"
        )
        self.add_child(Text(hint, 1, 0))

        self.add_child(Spacer(1))

        # Add bottom border
        self.add_child(DynamicBorder())

    # Focusable implementation - propagate to editor for IME cursor positioning
    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._editor.focused = value

    def handle_input(self, key_data: str) -> None:
        kb = get_keybindings()
        # Escape or Ctrl+C to cancel
        if kb.matches(key_data, "tui.select.cancel"):
            self._on_cancel_callback()
            return

        # External editor (app keybinding)
        if self._keybindings.matches(key_data, "app.editor.external"):
            tonio.spawn.without_tracking(self._handle_open_external_editor())
            return

        # Forward to editor
        self._editor.handle_input(key_data)

    async def _handle_open_external_editor(self) -> None:
        content = self._editor.get_text()
        await self._tui.stop()
        try:
            result = await edit_in_external_editor({"command": self._external_editor_command, "content": content})
            if result["status"] == "complete":
                self._editor.set_text(result["content"])
        finally:
            await self._tui.start()
            self._tui.request_render(True)
