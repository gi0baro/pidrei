"""Mirror of pi coding-agent src/core/tools/ls.ts."""

import os
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio
from tonio.colored import fs

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent
from pidrei_ai.utils.tasks import gather

from ..extensions.types import ToolDefinition
from .path_utils import resolve_to_cwd
from .renderers.ls import ls_renderers
from .tool_definition_wrapper import WrappedDefinitionTool, wrap_tool_definition
from .truncate import DEFAULT_MAX_BYTES, TruncationResult, format_size, truncate_head


LS_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Directory to list (default: current directory)"},
        "limit": {"type": "number", "description": "Maximum number of entries to return (default: 500)"},
    },
}

LS_TOOL_SYSTEM_PROMPT_CONTRIBUTION: dict[str, Any] = {
    "snippet": "List directory contents",
    "guidelines": (),
}

DEFAULT_LIMIT = 500


@dataclass(slots=True)
class LsToolDetails:
    truncation: TruncationResult | None = None
    entry_limit_reached: int | None = None


class LocalLsOperations:
    async def exists(self, absolute_path: str) -> bool:
        return await fs.Path(absolute_path).exists()

    async def is_directory(self, absolute_path: str) -> bool:
        return await fs.Path(absolute_path).is_dir()

    async def readdir(self, absolute_path: str) -> list[str]:
        return await tonio.spawn_blocking(os.listdir, absolute_path)


def create_ls_tool_definition(cwd: str, *, operations: Any = None) -> ToolDefinition:
    # `operations` members (`exists`/`is_directory`/`readdir`) are async-only.
    # pi types them `Promise<T> | T` because its sync defaults block the event
    # loop (`existsSync`); a sync impl doing real fs work would block this
    # runtime, so the union is deliberately not ported.
    ops = operations if operations is not None else LocalLsOperations()

    async def execute(_tool_call_id, params, cancel=None, _on_update=None, ctx=None):
        path = params.get("path")
        limit = params.get("limit")

        if cancel is not None and cancel.cancelled:
            raise Exception("Operation aborted")

        dir_path = resolve_to_cwd(path or ".", (ctx.cwd if ctx is not None else None) or cwd)
        effective_limit = int(limit) if limit is not None else DEFAULT_LIMIT

        # Check if path exists.
        if not await ops.exists(dir_path):
            raise Exception(f"Path not found: {dir_path}")

        # Check if path is a directory.
        if not await ops.is_directory(dir_path):
            raise Exception(f"Not a directory: {dir_path}")

        # Read directory entries.
        try:
            entries = list(await ops.readdir(dir_path))
        except Exception as error:
            raise Exception(f"Cannot read directory: {error}")

        # Sort alphabetically, case-insensitive.
        entries.sort(key=lambda entry: entry.lower())

        # Format entries with directory indicators.
        async def classify(entry: str) -> str | None:
            full_path = os.path.join(dir_path, entry)
            try:
                suffix = "/" if await ops.is_directory(full_path) else ""
                if not await fs.Path(full_path).exists():
                    return None  # Skip entries we cannot stat (broken symlinks).
            except Exception:
                return None  # Skip entries we cannot stat.
            return entry + suffix

        # Stat entries concurrently, one limit-sized window at a time: the
        # limit counts listed entries, so skipped ones pull in the next window.
        results: list[str] = []
        position = 0
        while position < len(entries) and len(results) < effective_limit:
            window = entries[position : position + effective_limit - len(results)]
            position += len(window)
            classified = await gather(*(classify(entry) for entry in window))
            results.extend(item for item in classified if item is not None)
        entry_limit_reached = len(results) >= effective_limit and position < len(entries)

        if not results:
            return AgentToolResult(content=[TextContent(text="(empty directory)")], details=None)

        raw_output = "\n".join(results)
        # Apply byte truncation. There is no separate line limit because entry count is already capped.
        truncation = truncate_head(raw_output, max_lines=2**53 - 1)
        output = truncation.content
        details = LsToolDetails()
        has_details = False
        # Build actionable notices for truncation and entry limits.
        notices: list[str] = []
        if entry_limit_reached:
            notices.append(f"{effective_limit} entries limit reached. Use limit={effective_limit * 2} for more")
            details.entry_limit_reached = effective_limit
            has_details = True
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details.truncation = truncation
            has_details = True
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        return AgentToolResult(content=[TextContent(text=output)], details=details if has_details else None)

    return ToolDefinition(
        name="ls",
        label="ls",
        description=(
            "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. "
            f"Includes dotfiles. Output is truncated to {DEFAULT_LIMIT} entries or "
            f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
        ),
        prompt_snippet=LS_TOOL_SYSTEM_PROMPT_CONTRIBUTION["snippet"],
        parameters=LS_SCHEMA,
        execute=execute,
        render_call=ls_renderers.render_call,
        render_result=ls_renderers.render_result,
    )


def create_ls_tool(cwd: str, **options) -> WrappedDefinitionTool:
    return wrap_tool_definition(create_ls_tool_definition(cwd, **options))
