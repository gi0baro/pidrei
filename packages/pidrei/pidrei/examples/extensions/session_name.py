"""Session naming.

Shows `set_session_name`/`get_session_name` to give sessions friendly names
that appear in the session selector instead of the first message.

Usage: /session-name [name] - set or show session name

Start pidrei with this extension:
    pidrei -e ./examples/extensions/session_name.py
"""


def extension(pi):
    async def run(args, ctx):
        name = args.strip()

        if name:
            await pi.set_session_name(name)
            ctx.ui.notify(f"Session named: {name}", "info")
        else:
            current = pi.get_session_name()
            ctx.ui.notify(f"Session: {current}" if current else "No session name set", "info")

    pi.register_command(
        "session-name",
        handler=run,
        description="Set or show session name (usage: /session-name [new name])",
    )
