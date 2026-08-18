"""Confirm Destructive Actions

Prompts for confirmation before destructive session actions (clear, switch,
fork). Demonstrates how to cancel session events using the before_* events.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/confirm_destructive.py
"""


def extension(pi):
    async def on_before_switch(event, ctx):
        if not ctx.has_ui:
            return None

        if event["reason"] == "new":
            confirmed = await ctx.ui.confirm(
                "Clear session?",
                "This will delete all messages in the current session.",
            )
            if not confirmed:
                ctx.ui.notify("Clear cancelled", "info")
                return {"cancel": True}
            return None

        # reason == "resume" - check if there are unsaved changes (messages
        # since last assistant response)
        entries = ctx.session_manager.get_entries()
        has_unsaved_work = any(
            entry.get("type") == "message" and getattr(entry.get("message"), "role", None) == "user"
            for entry in entries
        )

        if has_unsaved_work:
            confirmed = await ctx.ui.confirm(
                "Switch session?",
                "You have messages in the current session. Switch anyway?",
            )
            if not confirmed:
                ctx.ui.notify("Switch cancelled", "info")
                return {"cancel": True}

        return None

    async def on_before_fork(event, ctx):
        if not ctx.has_ui:
            return None

        choice = await ctx.ui.select(
            f"Fork from entry {event['entryId'][:8]}?",
            ["Yes, create fork", "No, stay in current session"],
        )
        if choice != "Yes, create fork":
            ctx.ui.notify("Fork cancelled", "info")
            return {"cancel": True}

        return None

    pi.on("session_before_switch", on_before_switch)
    pi.on("session_before_fork", on_before_fork)
