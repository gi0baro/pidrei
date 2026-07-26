"""Streaming-Aware Input Gate

Demonstrates `event["streamingBehavior"]` to skip expensive pre-processing
during mid-stream steering, where low latency matters.

This extension prepends `git diff --stat` output when the user mentions file
changes, giving the model immediate context. During steering the exec call is
skipped so the correction reaches the model without delay.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/input_transform_streaming.py
"""

import re


TRIGGER = re.compile(r"\b(changes?|diff|modified)\b", re.IGNORECASE)


def extension(pi):
    async def on_input(event, _ctx):
        # During steering, skip the exec call — corrections should be fast.
        if event.get("streamingBehavior") == "steer":
            return {"action": "continue"}

        if not TRIGGER.search(event["text"]):
            return {"action": "continue"}

        result = await pi.exec("git", ["diff", "--stat"])
        if result.code != 0 or not result.stdout.strip():
            return {"action": "continue"}

        return {
            "action": "transform",
            "text": f"{event['text']}\n\nCurrent uncommitted changes:\n```\n{result.stdout.strip()}\n```",
        }

    pi.on("input", on_input)
