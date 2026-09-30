"""Raw provider event viewer.

Usage: /debug-provider [on|off]
With no argument, the command toggles capture. Captured events are persisted
as expandable custom entries.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/debug_provider.py
"""

import copy
import json

from pidrei.modes.interactive.components import key_hint
from pidrei_tui import Box, Text


ENTRY_TYPE = "debug-provider-events"
STATUS_KEY = "debug-provider"


async def extension(pi):
    state = {"enabled": False, "active_events": None, "completed_entry": None}

    def render_provider_debug(entry, options, theme):
        data = entry.get("data")
        if not data:
            return Text(theme.fg("warning", "[provider debug] Missing event data"), 0, 0)

        box = Box(1, 1, lambda text: theme.bg("customMessageBg", text))
        events = data["events"]
        count = f"{len(events)} event{'' if len(events) == 1 else 's'}"
        expand_hint = "" if options.get("expanded") else f" ({key_hint('app.tools.expand', 'to view events')})"
        box.add_child(
            Text(
                f"{theme.fg('accent', '[provider debug]')} {data['provider']}/{data['model']} "
                f"({data['api']}) · {count}{expand_hint}",
                0,
                0,
            )
        )
        if options.get("expanded"):
            box.add_child(Text(json.dumps(events, indent=2, default=str), 0, 0))
        return box

    pi.register_entry_renderer(ENTRY_TYPE, render_provider_debug)

    async def debug_provider_command(args: str, ctx) -> None:
        requested_state = args.strip().lower()
        if requested_state not in ("", "on", "off"):
            ctx.ui.notify("Usage: /debug-provider [on|off]", "warning")
            return

        state["enabled"] = not state["enabled"] if requested_state == "" else requested_state == "on"
        if not state["enabled"]:
            state["active_events"] = None
            state["completed_entry"] = None
        ctx.ui.set_status(STATUS_KEY, "provider debug" if state["enabled"] else None)
        ctx.ui.notify(f"Provider event capture {'enabled' if state['enabled'] else 'disabled'}", "info")

    pi.register_command(
        "debug-provider",
        handler=debug_provider_command,
        description="Toggle capture of raw provider stream events",
    )

    async def on_turn_start(_event, _ctx) -> None:
        state["active_events"] = [] if state["enabled"] else None

    async def on_provider_stream_event(event, _ctx) -> None:
        if state["active_events"] is not None:
            state["active_events"].append(copy.deepcopy(event["data"]))

    async def on_message_end(event, _ctx) -> None:
        message = event["message"]
        if message.role != "assistant" or state["active_events"] is None:
            return
        state["completed_entry"] = {
            "provider": message.provider,
            "api": message.api,
            "model": message.model,
            "events": state["active_events"],
        }
        state["active_events"] = None

    async def on_turn_end(_event, _ctx) -> None:
        if state["completed_entry"] is None:
            return
        await pi.append_entry(ENTRY_TYPE, state["completed_entry"])
        state["completed_entry"] = None

    pi.on("turn_start", on_turn_start)
    pi.on("provider_stream_event", on_provider_stream_event)
    pi.on("message_end", on_message_end)
    pi.on("turn_end", on_turn_end)
