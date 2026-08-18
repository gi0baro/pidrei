"""Project Trust

Demonstrates the project_trust event. Install globally or pass via -e:

    mkdir -p ~/.pidrei/agent/extensions
    cp examples/extensions/project_trust.py ~/.pidrei/agent/extensions/

Or:

    pidrei -e ./examples/extensions/project_trust.py

Try it in a project containing .pidrei, AGENTS.md/CLAUDE.md, or .agents/skills.
"""


def extension(pi):
    load_count = 0
    load_count += 1

    # Multiple handlers in one extension are allowed. The first handler that
    # returns {"trusted": "yes"} or {"trusted": "no"} wins and suppresses the
    # built-in trust prompt. Return {"trusted": "undecided"} to let another
    # handler or the built-in flow decide.
    async def on_project_trust(event, ctx):
        ctx.ui.notify(f"project_trust fired for {event['cwd']} (mode: {ctx.mode}, load: {load_count})", "info")

        if not ctx.has_ui:
            return {"trusted": "undecided"}

        choice = await ctx.ui.select(
            f"Project trust for:\n{event['cwd']}",
            [
                "Trust and remember",
                "Trust with note and remember",
                "Trust this session",
                "Do not trust this session",
                "Let built-in prompt decide",
            ],
        )

        if choice == "Trust with note and remember":
            note = await ctx.ui.input("Project trust note", "Optional note for this demo")
            ctx.ui.notify(f"Recorded demo note: {note}" if note else "No demo note entered", "info")
            return {"trusted": "yes", "remember": True}
        if choice == "Trust and remember":
            return {"trusted": "yes", "remember": True}
        if choice == "Trust this session":
            return {"trusted": "yes"}
        if choice == "Do not trust this session":
            return {"trusted": "no"}
        return {"trusted": "undecided"}

    async def on_session_start(_event, ctx):
        ctx.ui.notify(f"project-trust example loaded after trust resolution in {ctx.cwd}", "info")

    pi.on("project_trust", on_project_trust)
    pi.on("session_start", on_session_start)
