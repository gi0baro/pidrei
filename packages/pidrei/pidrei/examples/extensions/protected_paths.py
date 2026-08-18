"""Protected Paths

Blocks write and edit operations to protected paths. Useful for preventing
accidental modifications to sensitive files.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/protected_paths.py
"""

PROTECTED_PATHS = [".env", ".git/", "node_modules/"]


def extension(pi):
    async def on_tool_call(event, ctx):
        if event["toolName"] not in ("write", "edit"):
            return None

        # Both tools name the parameter `path` and accept `file_path` too.
        path = event["input"].get("path") or event["input"].get("file_path") or ""
        if not any(protected in path for protected in PROTECTED_PATHS):
            return None

        if ctx.has_ui:
            ctx.ui.notify(f"Blocked write to protected path: {path}", "warning")
        return {"block": True, "reason": f'Path "{path}" is protected'}

    pi.on("tool_call", on_tool_call)
