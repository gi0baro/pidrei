"""Hidden Thinking Label

Demonstrates `ctx.ui.set_hidden_thinking_label()` for customizing the label
shown when thinking blocks are hidden.

Test:
    1. Load this extension
    2. Hide thinking blocks with Ctrl+T
    3. Ask for something that produces reasoning output
    4. The collapsed thinking block label will show the custom text

Commands:
    /thinking-label <text>   Set a custom hidden thinking label
    /thinking-label          Reset to the default label

Start pidrei with this extension:
    pidrei -e ./examples/extensions/hidden_thinking_label.py
"""

DEFAULT_LABEL = "Pondering..."


def extension(pi):
    state = {"label": DEFAULT_LABEL}

    async def on_session_start(_event, ctx) -> None:
        ctx.ui.set_hidden_thinking_label(state["label"])

    async def set_label(args: str, ctx) -> None:
        next_label = args.strip()

        if not next_label:
            state["label"] = DEFAULT_LABEL
            ctx.ui.set_hidden_thinking_label()
            ctx.ui.notify(f"Hidden thinking label reset to: {DEFAULT_LABEL}")
            return

        state["label"] = next_label
        ctx.ui.set_hidden_thinking_label(next_label)
        ctx.ui.notify(f"Hidden thinking label set to: {next_label}")

    pi.on("session_start", on_session_start)
    pi.register_command(
        "thinking-label",
        handler=set_label,
        description="Set the hidden thinking label. Use without args to reset.",
    )
