"""Input Transform

Demonstrates the `input` event for intercepting user input.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/input_transform.py

Then type these inside pidrei:
    ?quick What is Python?  → "Respond briefly: What is Python?"
    ping                    → "pong" (instant, no LLM)
    time                    → current time (instant, no LLM)
"""

from datetime import datetime


def extension(pi):
    async def on_input(event, ctx):
        # Source-based logic: skip processing for extension-injected messages
        if event["source"] == "extension":
            return {"action": "continue"}

        # Transform: ?quick prefix for brief responses
        if event["text"].startswith("?quick "):
            query = event["text"][7:].strip()
            if not query:
                ctx.ui.notify("Usage: ?quick <question>", "warning")
                return {"action": "handled"}
            return {"action": "transform", "text": f"Respond briefly in 1-2 sentences: {query}"}

        # Handle: instant responses without LLM (extension shows its own feedback)
        if event["text"].lower() == "ping":
            ctx.ui.notify("pong", "info")
            return {"action": "handled"}
        if event["text"].lower() == "time":
            ctx.ui.notify(datetime.now().astimezone().strftime("%c"), "info")
            return {"action": "handled"}

        return {"action": "continue"}

    pi.on("input", on_input)
