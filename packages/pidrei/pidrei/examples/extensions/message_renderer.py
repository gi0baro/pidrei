"""Custom message rendering.

Shows how to use `register_message_renderer` to control how custom messages
appear in the TUI, with colors, formatting, and expandable details.

Usage: /status [warn|error] message - sends a status message with custom
rendering.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/message_renderer.py
"""

import time

from pidrei_tui import Box, Text


def extension(pi):
    # Register custom renderer for "status-update" messages
    def render_status(message, options, theme):
        details = message.details or {}
        level = details.get("level", "info")

        # Color based on level
        color = "error" if level == "error" else "warning" if level == "warn" else "success"
        prefix = theme.fg(color, f"[{level.upper()}]")

        text = f"{prefix} {message.content}"

        # Show timestamp when expanded
        if options.get("expanded") and details.get("timestamp"):
            stamp = time.strftime("%H:%M:%S", time.localtime(details["timestamp"] / 1000))
            text += f"\n{theme.fg('dim', f'  at {stamp}')}"

        # Use a Box with customMessageBg for consistent styling; outputPad is
        # the horizontal padding the transcript uses, so the box lines up with
        # the rest of the output.
        box = Box(options.get("outputPad", 1), 1, lambda t: theme.bg("customMessageBg", t))
        box.add_child(Text(text, 0, 0))
        return box

    pi.register_message_renderer("status-update", render_status)

    # Command to send status messages
    async def status_command(args: str, _ctx) -> None:
        parts = args.strip().split()
        level = "info"
        content = args.strip()

        # Check for level prefix
        if parts and parts[0] in ("warn", "error"):
            level = parts[0]
            content = " ".join(parts[1:]) or "Status update"

        pi.send_message(
            {
                "customType": "status-update",
                "content": content,
                "display": True,
                "details": {"level": level, "timestamp": int(time.time() * 1000)},
            }
        )

    pi.register_command(
        "status", handler=status_command, description="Send a status message (usage: /status [warn|error] message)"
    )
