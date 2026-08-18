"""Auto-Commit on Exit

Automatically commits changes when the session shuts down. Uses the last
assistant message to generate a commit message.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/auto_commit_on_exit.py
"""


def extension(pi):
    async def on_session_shutdown(_event, ctx) -> None:
        # Check for uncommitted changes.
        status = await pi.exec("git", ["status", "--porcelain"])
        if status.code != 0 or not status.stdout.strip():
            # Not a git repo or no changes.
            return

        # Find the last assistant message for commit context.
        last_assistant_text = ""
        for entry in reversed(ctx.session_manager.get_entries()):
            if entry.get("type") != "message":
                continue
            message = entry.get("message")
            if getattr(message, "role", None) != "assistant":
                continue
            last_assistant_text = "\n".join(
                block.text for block in message.content if getattr(block, "type", None) == "text"
            )
            break

        # Generate a simple commit message.
        first_line = last_assistant_text.split("\n")[0] or "Work in progress"
        suffix = "..." if len(first_line) > 50 else ""
        commit_message = f"[pidrei] {first_line[:50]}{suffix}"

        # Stage and commit.
        await pi.exec("git", ["add", "-A"])
        commit = await pi.exec("git", ["commit", "-m", commit_message])

        if commit.code == 0 and ctx.has_ui:
            ctx.ui.notify(f"Auto-committed: {commit_message}", "info")

    pi.on("session_shutdown", on_session_shutdown)
