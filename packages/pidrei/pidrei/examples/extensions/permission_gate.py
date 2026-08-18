"""Permission Gate

Prompts for confirmation before running potentially dangerous bash commands.
Patterns checked: rm -rf, sudo, chmod/chown 777.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/permission_gate.py
"""

import re


DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+(-rf?|--recursive)", re.IGNORECASE),
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\b(chmod|chown)\b.*777", re.IGNORECASE),
]


def extension(pi):
    async def on_tool_call(event, ctx):
        if event["toolName"] != "bash":
            return None

        command = event["input"].get("command", "")
        if not any(pattern.search(command) for pattern in DANGEROUS_PATTERNS):
            return None

        if not ctx.has_ui:
            # In non-interactive mode, block by default.
            return {"block": True, "reason": "Dangerous command blocked (no UI for confirmation)"}

        choice = await ctx.ui.select(f"Dangerous command:\n\n  {command}\n\nAllow?", ["Yes", "No"])
        if choice != "Yes":
            return {"block": True, "reason": "Blocked by user"}

        return None

    pi.on("tool_call", on_tool_call)
