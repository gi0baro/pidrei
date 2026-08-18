"""System Prompt Header

Displays a status widget showing the system prompt length.

Demonstrates `ctx.get_system_prompt()` for accessing the effective system
prompt.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/system_prompt_header.py
"""


def extension(pi):
    async def on_agent_start(_event, ctx):
        prompt = ctx.get_system_prompt()
        ctx.ui.set_status("system-prompt", f"System: {len(prompt)} chars")

    async def on_session_shutdown(_event, ctx):
        ctx.ui.set_status("system-prompt", None)

    pi.on("agent_start", on_agent_start)
    pi.on("session_shutdown", on_session_shutdown)
