"""Commands Extension

Demonstrates the `pi.get_commands()` API by providing a /commands command
that lists all available slash commands in the current session.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/commands.py

Then use /commands to see available commands, or /commands extension to
filter by source.
"""


def format_command(cmd) -> str:
    desc = f" - {cmd.description}" if cmd.description else ""
    return f"/{cmd.name}{desc}"


def extension(pi):
    async def get_argument_completions(prefix: str):
        sources = ["extension", "prompt", "skill"]
        filtered = [s for s in sources if s.startswith(prefix)]
        return [{"value": s, "label": s} for s in filtered] or None

    async def handler(args: str, ctx) -> None:
        commands = pi.get_commands()
        source_filter = args.strip()

        # Filter by source if specified
        filtered = [c for c in commands if c.source == source_filter] if source_filter else commands

        if not filtered:
            ctx.ui.notify(
                f"No {source_filter} commands found" if source_filter else "No commands found",
                "info",
            )
            return

        # Build selection items grouped by source
        items: list[str] = []
        for key, label in (("extension", "Extensions"), ("prompt", "Prompts"), ("skill", "Skills")):
            cmds = [c for c in filtered if c.source == key]
            if cmds:
                items.append(f"--- {label} ---")
                items.extend(format_command(c) for c in cmds)

        # Show in a selector (user can scroll and see all commands)
        selected = await ctx.ui.select("Available Commands", items)

        # If user selected a command (not a header), offer to show its path
        if selected and not selected.startswith("---"):
            cmd_name = selected.split(" - ")[0][1:]  # Remove leading /
            cmd = next((c for c in commands if c.name == cmd_name), None)
            if cmd is not None and cmd.source_info.path:
                show_path = await ctx.ui.confirm(cmd.name, f"View source path?\n{cmd.source_info.path}")
                if show_path:
                    ctx.ui.notify(cmd.source_info.path, "info")

    pi.register_command(
        "commands",
        description="List available slash commands",
        handler=handler,
        get_argument_completions=get_argument_completions,
    )
