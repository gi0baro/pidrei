"""RPC Extension UI Demo

Purpose-built extension that exercises all RPC-supported extension UI methods,
demonstrating the full extension UI protocol when pidrei runs with
`--mode rpc` (it works interactively too).

UI methods exercised:
- select() - on tool_call for dangerous bash commands
- confirm() - on session_before_switch
- input() - via /rpc-input command
- editor() - via /rpc-editor command
- notify() - after each dialog completes
- set_status() - on turn_start/turn_end
- set_widget() - on session_start
- set_title() - on session_start
- set_editor_text() - via /rpc-prefill command

Start pidrei with this extension:
    pidrei -e ./examples/extensions/rpc_demo.py
"""

import re


_DANGEROUS_RE = re.compile(r"\brm\s+(-rf?|--recursive)|\bsudo\b", re.IGNORECASE)


def extension(pi):
    state = {"turn_count": 0}

    # -- set_title, set_widget, set_status on session lifecycle --

    async def on_session_start(event, ctx) -> None:
        ctx.ui.set_title("pidrei RPC Demo (new session)" if event["reason"] == "new" else "pidrei RPC Demo")
        ctx.ui.set_widget("rpc-demo", ["--- RPC Extension UI Demo ---", "Loaded and ready."])
        ctx.ui.set_status("rpc-demo", f"Turns: {state['turn_count']}")

    # -- set_status on turn lifecycle --

    async def on_turn_start(_event, ctx) -> None:
        state["turn_count"] += 1
        ctx.ui.set_status("rpc-demo", f"Turn {state['turn_count']} running...")

    async def on_turn_end(_event, ctx) -> None:
        ctx.ui.set_status("rpc-demo", f"Turn {state['turn_count']} done")

    # -- select on dangerous tool calls --

    async def on_tool_call(event, ctx):
        if event["toolName"] != "bash":
            return None

        command = event["input"].get("command", "")
        if not _DANGEROUS_RE.search(command):
            return None

        if not ctx.has_ui:
            return {"block": True, "reason": "Dangerous command blocked (no UI)"}

        choice = await ctx.ui.select(f"Dangerous command: {command}", ["Allow", "Block"])
        if choice != "Allow":
            ctx.ui.notify("Command blocked by user", "warning")
            return {"block": True, "reason": "Blocked by user"}
        ctx.ui.notify("Command allowed", "info")
        return None

    # -- confirm on session clear --

    async def on_session_before_switch(event, ctx):
        if event["reason"] != "new":
            return None
        if not ctx.has_ui:
            return None

        confirmed = await ctx.ui.confirm("Clear session?", "All messages will be lost.")
        if not confirmed:
            ctx.ui.notify("Clear cancelled", "info")
            return {"cancel": True}
        return None

    # -- input via command --

    async def rpc_input_command(_args, ctx) -> None:
        value = await ctx.ui.input("Enter a value", "type something...")
        if value:
            ctx.ui.notify(f"You entered: {value}", "info")
        else:
            ctx.ui.notify("Input cancelled", "info")

    # -- editor via command --

    async def rpc_editor_command(_args, ctx) -> None:
        text = await ctx.ui.editor("Edit some text", "Line 1\nLine 2\nLine 3")
        if text:
            ctx.ui.notify(f"Editor submitted ({len(text.split('\n'))} lines)", "info")
        else:
            ctx.ui.notify("Editor cancelled", "info")

    # -- set_editor_text via command --

    async def rpc_prefill_command(_args, ctx) -> None:
        ctx.ui.set_editor_text("This text was set by the rpc-demo extension.")
        ctx.ui.notify("Editor prefilled", "info")

    pi.on("session_start", on_session_start)
    pi.on("turn_start", on_turn_start)
    pi.on("turn_end", on_turn_end)
    pi.on("tool_call", on_tool_call)
    pi.on("session_before_switch", on_session_before_switch)
    pi.register_command(
        "rpc-input",
        description="Prompt for text input (demonstrates ctx.ui.input in RPC)",
        handler=rpc_input_command,
    )
    pi.register_command(
        "rpc-editor",
        description="Open multi-line editor (demonstrates ctx.ui.editor in RPC)",
        handler=rpc_editor_command,
    )
    pi.register_command(
        "rpc-prefill",
        description="Prefill the input editor (demonstrates ctx.ui.set_editor_text in RPC)",
        handler=rpc_prefill_command,
    )
