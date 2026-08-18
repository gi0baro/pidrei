"""Status Line

Demonstrates `ctx.ui.set_status()` for displaying persistent status text in
the footer. Shows turn progress with themed colors.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/status_line.py
"""


def extension(pi):
    state = {"turn_count": 0}

    async def on_session_start(_event, ctx) -> None:
        theme = ctx.ui.theme
        ctx.ui.set_status("status-demo", theme.fg("dim", "Ready"))

    async def on_turn_start(_event, ctx) -> None:
        state["turn_count"] += 1
        theme = ctx.ui.theme
        spinner = theme.fg("accent", "●")
        text = theme.fg("dim", f" Turn {state['turn_count']}...")
        ctx.ui.set_status("status-demo", spinner + text)

    async def on_turn_end(_event, ctx) -> None:
        theme = ctx.ui.theme
        check = theme.fg("success", "✓")
        text = theme.fg("dim", f" Turn {state['turn_count']} complete")
        ctx.ui.set_status("status-demo", check + text)

    pi.on("session_start", on_session_start)
    pi.on("turn_start", on_turn_start)
    pi.on("turn_end", on_turn_end)
