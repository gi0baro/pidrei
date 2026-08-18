"""Working Indicator

Demonstrates `ctx.ui.set_working_indicator()` for customizing the inline
working indicator shown while pidrei is streaming a response.

Commands:
    /working-indicator           Show current mode
    /working-indicator dot       Use a static dot indicator
    /working-indicator pulse     Use a custom animated indicator
    /working-indicator none      Hide the indicator entirely
    /working-indicator spinner   Restore an animated spinner
    /working-indicator reset     Restore pidrei's default spinner

Start pidrei with this extension:
    pidrei -e ./examples/extensions/working_indicator.py
"""

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
PASTEL_RAINBOW = [
    "\x1b[38;2;255;179;186m",
    "\x1b[38;2;255;223;186m",
    "\x1b[38;2;255;255;186m",
    "\x1b[38;2;186;255;201m",
    "\x1b[38;2;186;225;255m",
    "\x1b[38;2;218;186;255m",
]
RESET_FG = "\x1b[39m"
HIDDEN_INDICATOR = {"frames": []}

MODE_LABELS = {
    "dot": "static dot",
    "none": "hidden",
    "pulse": "custom pulse",
    "spinner": "custom spinner",
    "default": "pidrei default spinner",
}


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{RESET_FG}"


def get_indicator(mode: str) -> dict | None:
    if mode == "dot":
        return {"frames": [colorize("●", PASTEL_RAINBOW[0])]}
    if mode == "none":
        return HIDDEN_INDICATOR
    if mode == "pulse":
        return {
            "frames": [
                colorize("·", PASTEL_RAINBOW[0]),
                colorize("•", PASTEL_RAINBOW[2]),
                colorize("●", PASTEL_RAINBOW[4]),
                colorize("•", PASTEL_RAINBOW[5]),
            ],
            "intervalMs": 120,
        }
    if mode == "spinner":
        return {
            "frames": [
                colorize(frame, PASTEL_RAINBOW[index % len(PASTEL_RAINBOW)])
                for index, frame in enumerate(SPINNER_FRAMES)
            ],
            "intervalMs": 80,
        }
    # "default": None restores pidrei's built-in spinner
    return None


def extension(pi):
    state = {"mode": "spinner"}

    def apply_indicator(ctx) -> None:
        ctx.ui.set_working_indicator(get_indicator(state["mode"]))
        ctx.ui.set_status("working-indicator", ctx.ui.theme.fg("dim", f"Indicator: {MODE_LABELS[state['mode']]}"))

    async def on_session_start(_event, ctx) -> None:
        apply_indicator(ctx)

    async def run_command(args: str, ctx) -> None:
        next_mode = args.strip().lower()
        if not next_mode:
            ctx.ui.notify(f"Working indicator: {MODE_LABELS[state['mode']]}", "info")
            return

        if next_mode not in ("dot", "none", "pulse", "spinner", "reset"):
            ctx.ui.notify("Usage: /working-indicator [dot|pulse|none|spinner|reset]", "error")
            return

        state["mode"] = "default" if next_mode == "reset" else next_mode
        apply_indicator(ctx)
        ctx.ui.notify(f"Working indicator set to: {MODE_LABELS[state['mode']]}", "info")

    pi.on("session_start", on_session_start)
    pi.register_command(
        "working-indicator",
        handler=run_command,
        description="Set the streaming working indicator: dot, pulse, none, spinner, or reset.",
    )
