"""Entry bookmarking.

Shows `pi.set_label` to mark entries with labels for easy navigation in /tree.
Labels appear in the tree view and help you find important points.

Usage: /bookmark [label] - bookmark the last assistant message

Start pidrei with this extension:
    pidrei -e ./examples/extensions/bookmark.py
"""

import time


def extension(pi):
    async def bookmark(args, ctx):
        label = args.strip() or f"bookmark-{int(time.time() * 1000)}"

        # Find the last assistant message entry
        for entry in reversed(ctx.session_manager.get_entries()):
            if entry.get("type") == "message" and getattr(entry.get("message"), "role", None) == "assistant":
                await pi.set_label(entry["id"], label)
                ctx.ui.notify(f"Bookmarked as: {label}", "info")
                return

        ctx.ui.notify("No assistant message to bookmark", "warning")

    # Remove bookmark
    async def unbookmark(_args, ctx):
        for entry in reversed(ctx.session_manager.get_entries()):
            label = ctx.session_manager.get_label(entry["id"])
            if label:
                await pi.set_label(entry["id"], None)
                ctx.ui.notify(f"Removed bookmark: {label}", "info")
                return

        ctx.ui.notify("No bookmarked entry found", "warning")

    pi.register_command(
        "bookmark",
        handler=bookmark,
        description="Bookmark last message (usage: /bookmark [label])",
    )
    pi.register_command(
        "unbookmark",
        handler=unbookmark,
        description="Remove bookmark from last labeled entry",
    )
