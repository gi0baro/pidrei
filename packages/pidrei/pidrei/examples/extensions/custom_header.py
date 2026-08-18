"""Custom Header

Demonstrates `ctx.ui.set_header()` for replacing the built-in header
(logo + keybinding hints) with a custom component showing the pi mascot.

`/builtin-header` restores the default.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/custom_header.py
"""

from pidrei.config import VERSION


# --- PI MASCOT ---
# Based on pi_mascot.ts - the pi agent character
def get_pi_mascot(theme) -> list[str]:
    # --- COLORS ---
    # 3b1b Blue: R=80, G=180, B=230
    def pi_blue(text: str) -> str:
        return theme.fg("accent", text)

    def white(text: str) -> str:
        return text  # Use plain white (or theme.fg("text", text))

    def black(text: str) -> str:
        return theme.fg("dim", text)  # Use dim for contrast

    # --- GLYPHS ---
    block = "█"
    pupil = "▌"  # Vertical half-block for the pupil

    # --- CONSTRUCTION ---

    # 1. The Eye Unit: [White Full Block][Black Vertical Sliver]
    # This creates the "looking sideways" effect
    eye = f"{white(block)}{black(pupil)}"

    # 2. Line 1: The Eyes
    # 5 spaces indent aligns them with the start of the legs
    line_eyes = f"     {eye}  {eye}"

    # 3. Line 2: The Wide Top Bar (The "Overhang")
    # 14 blocks wide for that serif-style roof
    line_bar = f"  {pi_blue(block * 14)}"

    # 4. Lines 3-6: The Legs
    # Indented 5 spaces relative to the very left edge
    # Leg width: 2 blocks | Gap: 4 blocks
    line_leg = f"     {pi_blue(block * 2)}    {pi_blue(block * 2)}"

    # --- ASSEMBLY ---
    return ["", line_eyes, line_bar, line_leg, line_leg, line_leg, line_leg, ""]


class MascotHeader:
    """Header component: `render(width) -> list[str]` plus `invalidate()`."""

    def __init__(self, theme) -> None:
        self._theme = theme

    def render(self, _width: int) -> list[str]:
        mascot_lines = get_pi_mascot(self._theme)
        # Add a subtitle with hint
        subtitle = f"{self._theme.fg('muted', '   shitty coding agent')}{self._theme.fg('dim', f' v{VERSION}')}"
        return [*mascot_lines, subtitle]

    def invalidate(self) -> None:
        pass


def extension(pi):
    # Set custom header immediately on load (if UI is available)
    async def on_session_start(_event, ctx) -> None:
        if ctx.mode == "tui":
            ctx.ui.set_header(lambda _ui, theme: MascotHeader(theme))

    pi.on("session_start", on_session_start)

    # Command to restore built-in header
    async def restore_header(_args, ctx) -> None:
        ctx.ui.set_header(None)
        ctx.ui.notify("Built-in header restored", "info")

    pi.register_command(
        "builtin-header",
        handler=restore_header,
        description="Restore built-in header with keybinding hints",
    )
