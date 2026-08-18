"""Git Checkpoint

Creates git stash checkpoints at each turn so /fork can restore code state.
When forking, offers to restore code to that point in history.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/git_checkpoint.py
"""


def extension(pi):
    checkpoints: dict[str, str] = {}
    state: dict[str, str | None] = {"current_entry_id": None}

    # Track the current entry ID when tool results are saved.
    async def on_tool_result(_event, ctx) -> None:
        leaf = ctx.session_manager.get_leaf_entry()
        if leaf:
            state["current_entry_id"] = leaf["id"]

    async def on_turn_start(_event, _ctx) -> None:
        # Create a git stash entry before the LLM makes changes.
        result = await pi.exec("git", ["stash", "create"])
        ref = result.stdout.strip()
        entry_id = state["current_entry_id"]
        if ref and entry_id:
            checkpoints[entry_id] = ref

    async def on_before_fork(event, ctx) -> None:
        ref = checkpoints.get(event["entryId"])
        if not ref:
            return

        if not ctx.has_ui:
            # In non-interactive mode, don't restore automatically.
            return

        choice = await ctx.ui.select(
            "Restore code state?",
            ["Yes, restore code to that point", "No, keep current code"],
        )

        if choice and choice.startswith("Yes"):
            await pi.exec("git", ["stash", "apply", ref])
            ctx.ui.notify("Code restored to checkpoint", "info")

    async def on_agent_end(_event, _ctx) -> None:
        # Clear checkpoints after the agent completes.
        checkpoints.clear()

    pi.on("tool_result", on_tool_result)
    pi.on("turn_start", on_turn_start)
    pi.on("session_before_fork", on_before_fork)
    pi.on("agent_end", on_agent_end)
