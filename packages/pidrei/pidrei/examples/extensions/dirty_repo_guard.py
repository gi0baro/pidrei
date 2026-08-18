"""Dirty Repo Guard

Prevents session changes when there are uncommitted git changes. Useful to
ensure work is committed before switching context.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/dirty_repo_guard.py
"""


async def check_dirty_repo(pi, ctx, action: str) -> dict[str, bool] | None:
    # Check for uncommitted changes.
    result = await pi.exec("git", ["status", "--porcelain"])

    if result.code != 0:
        # Not a git repo, allow the action.
        return None

    if not result.stdout.strip():
        return None

    if not ctx.has_ui:
        # In non-interactive mode, block by default.
        return {"cancel": True}

    # Count changed files.
    changed_files = len([line for line in result.stdout.strip().split("\n") if line])

    choice = await ctx.ui.select(
        f"You have {changed_files} uncommitted file(s). {action} anyway?",
        ["Yes, proceed anyway", "No, let me commit first"],
    )
    if choice != "Yes, proceed anyway":
        ctx.ui.notify("Commit your changes first", "warning")
        return {"cancel": True}

    return None


def extension(pi):
    async def on_before_switch(event, ctx):
        action = "new session" if event["reason"] == "new" else "switch session"
        return await check_dirty_repo(pi, ctx, action)

    async def on_before_fork(_event, ctx):
        return await check_dirty_repo(pi, ctx, "fork")

    pi.on("session_before_switch", on_before_switch)
    pi.on("session_before_fork", on_before_fork)
