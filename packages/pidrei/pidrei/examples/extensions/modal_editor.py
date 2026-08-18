"""Modal Editor

Vim-like modal editing on top of the default editor:

- Escape: insert → normal mode (in normal mode, aborts the agent)
- i: normal → insert mode
- hjkl: navigation in normal mode
- ctrl+c, ctrl+d, etc. work in both modes

Start pidrei with this extension:
    pidrei -e ./examples/extensions/modal_editor.py
"""

from pidrei.modes.interactive.components import CustomEditor
from pidrei_tui import matches_key, truncate_to_width, visible_width


# Normal mode key mappings: key -> escape sequence (or None for mode switch)
NORMAL_KEYS: dict[str, str | None] = {
    "h": "\x1b[D",  # left
    "j": "\x1b[B",  # down
    "k": "\x1b[A",  # up
    "l": "\x1b[C",  # right
    "0": "\x01",  # line start
    "$": "\x05",  # line end
    "x": "\x1b[3~",  # delete char
    "i": None,  # insert mode
    "a": None,  # append (insert + right)
}


class ModalEditor(CustomEditor):
    def __init__(self, tui, theme, keybindings) -> None:
        super().__init__(tui, theme, keybindings)
        self.mode = "insert"  # "normal" | "insert"

    async def handle_input(self, data: str) -> None:
        # Escape toggles to normal mode, or passes through for app handling
        if matches_key(data, "escape"):
            if self.mode == "insert":
                self.mode = "normal"
            else:
                await super().handle_input(data)  # abort agent, etc.
            return

        # Insert mode: pass everything through
        if self.mode == "insert":
            await super().handle_input(data)
            return

        # Normal mode: check mapped keys
        if data in NORMAL_KEYS:
            seq = NORMAL_KEYS[data]
            if data == "i":
                self.mode = "insert"
            elif data == "a":
                self.mode = "insert"
                await super().handle_input("\x1b[C")  # move right first
            elif seq:
                await super().handle_input(seq)
            return

        # Pass control sequences (ctrl+c, etc.) to super, ignore printable chars
        if len(data) == 1 and ord(data) >= 32:
            return
        await super().handle_input(data)

    def render(self, width: int) -> list[str]:
        lines = super().render(width)
        if not lines:
            return lines

        # Add mode indicator to bottom border
        label = " NORMAL " if self.mode == "normal" else " INSERT "
        if visible_width(lines[-1]) >= len(label):
            lines[-1] = truncate_to_width(lines[-1], width - len(label), "") + label
        return lines


def extension(pi):
    async def on_session_start(_event, ctx) -> None:
        ctx.ui.set_editor_component(lambda tui, theme, keybindings: ModalEditor(tui, theme, keybindings))

    pi.on("session_start", on_session_start)
