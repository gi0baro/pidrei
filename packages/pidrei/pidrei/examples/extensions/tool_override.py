"""Tool Override

Extensions can register tools with the same name as built-in tools to replace
them. This is useful for:
- Adding logging or auditing to tool calls
- Implementing access control or sandboxing
- Routing tool calls to remote systems
- Modifying tool behavior for specific workflows

This example overrides the `read` tool to:
1. Log all file access to a log file
2. Block access to sensitive paths (e.g., .env files)
3. Perform the read itself (a simplified implementation)

Since no render_call/render_result are provided, the built-in renderer is used
automatically (syntax highlighting, line numbers, truncation warnings).

Start pidrei with this extension:
    pidrei -e ./examples/extensions/tool_override.py
"""

import os
import re
from datetime import UTC, datetime

import tonio.colored as tonio
from tonio.colored import fs

from pidrei.config import get_agent_dir
from pidrei.core.extensions.types import ToolDefinition
from pidrei.core.tools import with_file_mutation_queue
from pidrei.core.tools.file_mutation_queue import resolve_mutation_queue_key
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent


LOG_FILE = os.path.join(get_agent_dir(), "read-access.log")

# Paths that are blocked from reading
BLOCKED_PATTERNS = [
    re.compile(r"\.env$"),
    re.compile(r"\.env\..+$"),
    re.compile(r"secrets?\.(json|yaml|yml|toml)$", re.IGNORECASE),
    re.compile(r"credentials?\.(json|yaml|yml|toml)$", re.IGNORECASE),
    re.compile(r"/\.ssh/"),
    re.compile(r"/\.aws/"),
    re.compile(r"/\.gnupg/"),
]


def _is_blocked_path(path: str) -> bool:
    return any(pattern.search(path) for pattern in BLOCKED_PATTERNS)


def _append_line(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)


async def _log_access(path: str, allowed: bool, reason: str | None = None) -> None:
    timestamp = datetime.now(UTC).isoformat()
    status = "ALLOWED" if allowed else "BLOCKED"
    suffix = f" ({reason})" if reason else ""
    line = f"[{timestamp}] {status}: {path}{suffix}\n"

    try:
        queue_key = await resolve_mutation_queue_key(LOG_FILE)

        async def append() -> None:
            await tonio.spawn_blocking(_append_line, LOG_FILE, line)

        await with_file_mutation_queue(LOG_FILE, append, queue_key=queue_key)
    except Exception:
        # Ignore logging errors
        pass


def extension(pi):
    async def execute(_tool_call_id, params, _cancel=None, _on_update=None, ctx=None):
        path = params["path"]
        offset = params.get("offset")
        limit = params.get("limit")
        absolute_path = os.path.abspath(os.path.join(ctx.cwd, path))

        # Check if path is blocked
        if _is_blocked_path(absolute_path):
            await _log_access(absolute_path, False, "matches blocked pattern")
            return AgentToolResult(
                content=[
                    TextContent(
                        text=(
                            f'Access denied: "{path}" matches a blocked pattern (sensitive file). This tool '
                            "blocks access to .env files, secrets, credentials, and SSH/AWS/GPG directories."
                        )
                    )
                ],
                details={"blocked": True},
            )

        # Log allowed access
        await _log_access(absolute_path, True)

        # Perform the actual read (simplified implementation)
        try:
            content = (await fs.Path(absolute_path).read_bytes()).decode("utf-8", "replace")
            lines = content.split("\n")

            # Apply offset and limit
            start_line = max(0, int(offset) - 1) if offset else 0
            end_line = start_line + int(limit) if limit else len(lines)
            text = "\n".join(lines[start_line:end_line])

            # Basic truncation (50KB limit)
            max_bytes = 50 * 1024
            if len(text.encode("utf-8")) > max_bytes:
                text = f"{text[:max_bytes]}\n\n[Output truncated at 50KB]"

            return AgentToolResult(content=[TextContent(text=text)], details={"lines": len(lines)})
        except OSError as error:
            return AgentToolResult(
                content=[TextContent(text=f"Error reading file: {error}")],
                details={"error": True},
            )

    pi.register_tool(
        ToolDefinition(
            name="read",  # Same name as built-in - this will override it
            label="read (audited)",
            description=(
                "Read the contents of a file with access logging. Some sensitive paths "
                "(.env, secrets, credentials) are blocked."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
                    "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
                    "limit": {"type": "number", "description": "Maximum number of lines to read"},
                },
                "required": ["path"],
            },
            execute=execute,
            # No render_call/render_result - uses the built-in renderer
            # automatically (syntax highlighting, truncation warnings, etc.)
        )
    )

    # Also register a command to view the access log
    async def read_log(_args: str, ctx) -> None:
        try:
            log = (await fs.Path(LOG_FILE).read_bytes()).decode("utf-8", "replace")
            lines = log.strip().split("\n")[-20:]  # Last 20 entries
            ctx.ui.notify("Recent file access:\n" + "\n".join(lines), "info")
        except OSError:
            ctx.ui.notify("No access log found", "info")

    pi.register_command("read-log", handler=read_log, description="View the file access log")
