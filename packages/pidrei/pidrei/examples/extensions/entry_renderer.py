"""Custom entry rendering.

Shows how to render durable extension data inside the chat without sending it
to the LLM. Custom entries are stored in the session via `pi.append_entry()`
and rendered in interactive mode via `pi.register_entry_renderer()`.

Usage: /status-card [message]

Start pidrei with this extension:
    pidrei -e ./examples/extensions/entry_renderer.py
"""

import time

from pidrei_tui import Box, Text


def extension(pi):
    def render_status_card(entry, options, theme):
        data = entry.get("data") or {"message": "No data", "timestamp": int(time.time() * 1000)}
        box = Box(1, 1, lambda text: theme.bg("customMessageBg", text))
        box.add_child(Text(f"{theme.fg('accent', '[status]')} {data['message']}", 0, 0))

        if options.get("expanded"):
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data["timestamp"] / 1000))
            box.add_child(Text(theme.fg("dim", stamp), 0, 0))

        return box

    pi.register_entry_renderer("status-card", render_status_card)

    async def status_card_command(args: str, _ctx) -> None:
        await pi.append_entry(
            "status-card",
            {"message": args.strip() or "Status card", "timestamp": int(time.time() * 1000)},
        )

    pi.register_command(
        "status-card",
        handler=status_card_command,
        description="Render a durable status card that is not sent to the LLM",
    )
