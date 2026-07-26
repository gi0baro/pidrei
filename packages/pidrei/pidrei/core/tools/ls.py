"""Mirror of pi coding-agent src/core/tools/ls.ts."""

import os
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent
from pidrei_tui import Text

from ...modes.interactive.components.keybinding_hints import key_hint
from ..extensions.types import ToolDefinition
from .path_utils import path_exists, resolve_to_cwd
from .render_utils import get_text_output, render_tool_path, str_or_none
from .tool_definition_wrapper import WrappedDefinitionTool, wrap_tool_definition
from .truncate import DEFAULT_MAX_BYTES, TruncationResult, format_size, truncate_head


def _format_ls_call(args: dict | None, theme, cwd: str) -> str:
    args = args or {}
    limit = args.get("limit")
    path_display = render_tool_path(str_or_none(args.get("path")), theme, cwd, {"emptyFallback": "."})
    text = f"{theme.fg('toolTitle', theme.bold('ls'))} {path_display}"
    if limit is not None:
        text += theme.fg("toolOutput", f" (limit {limit})")
    return text


def _format_ls_result(result, options: dict, theme, show_images: bool) -> str:
    output = get_text_output(result, show_images).strip()
    text = ""
    if output:
        lines = output.split("\n")
        max_lines = len(lines) if options.get("expanded") else 20
        display_lines = lines[:max_lines]
        remaining = len(lines) - max_lines
        text += "\n" + "\n".join(theme.fg("toolOutput", line) for line in display_lines)
        if remaining > 0:
            text += (
                theme.fg("muted", f"\n... ({remaining} more lines,")
                + " "
                + key_hint("app.tools.expand", "to expand")
                + theme.fg("muted", ")")
            )

    details = result.get("details") if isinstance(result, dict) else getattr(result, "details", None)
    entry_limit = getattr(details, "entry_limit_reached", None) if details is not None else None
    truncation = getattr(details, "truncation", None) if details is not None else None
    if entry_limit or (truncation is not None and truncation.truncated):
        warnings = []
        if entry_limit:
            warnings.append(f"{entry_limit} entries limit")
        if truncation is not None and truncation.truncated:
            max_bytes = truncation.max_bytes if truncation.max_bytes is not None else DEFAULT_MAX_BYTES
            warnings.append(f"{format_size(max_bytes)} limit")
        text += "\n" + theme.fg("warning", f"[Truncated: {', '.join(warnings)}]")
    return text


LS_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Directory to list (default: current directory)"},
        "limit": {"type": "number", "description": "Maximum number of entries to return (default: 500)"},
    },
}

DEFAULT_LIMIT = 500


@dataclass(slots=True)
class LsToolDetails:
    truncation: TruncationResult | None = None
    entry_limit_reached: int | None = None


class LocalLsOperations:
    async def exists(self, absolute_path: str) -> bool:
        return path_exists(absolute_path)

    async def is_directory(self, absolute_path: str) -> bool:
        return os.path.isdir(absolute_path)

    async def readdir(self, absolute_path: str) -> list[str]:
        return await tonio.spawn_blocking(os.listdir, absolute_path)


async def _maybe_await(value: Any) -> Any:
    import inspect

    return await value if inspect.isawaitable(value) else value


def create_ls_tool_definition(cwd: str, *, operations: Any = None) -> ToolDefinition:
    ops = operations if operations is not None else LocalLsOperations()

    async def execute(_tool_call_id, params, cancel=None, _on_update=None, _ctx=None):
        path = params.get("path")
        limit = params.get("limit")

        if cancel is not None and cancel.cancelled:
            raise Exception("Operation aborted")

        dir_path = resolve_to_cwd(path or ".", cwd)
        effective_limit = int(limit) if limit is not None else DEFAULT_LIMIT

        # Check if path exists.
        if not await _maybe_await(ops.exists(dir_path)):
            raise Exception(f"Path not found: {dir_path}")

        # Check if path is a directory.
        if not await _maybe_await(ops.is_directory(dir_path)):
            raise Exception(f"Not a directory: {dir_path}")

        # Read directory entries.
        try:
            entries = list(await _maybe_await(ops.readdir(dir_path)))
        except Exception as error:
            raise Exception(f"Cannot read directory: {error}")

        # Sort alphabetically, case-insensitive.
        entries.sort(key=lambda entry: entry.lower())

        # Format entries with directory indicators.
        results: list[str] = []
        entry_limit_reached = False
        for entry in entries:
            if len(results) >= effective_limit:
                entry_limit_reached = True
                break

            full_path = os.path.join(dir_path, entry)
            try:
                suffix = "/" if await _maybe_await(ops.is_directory(full_path)) else ""
                if not os.path.exists(full_path):
                    continue  # Skip entries we cannot stat (broken symlinks).
            except Exception:  # noqa: S112
                continue  # Skip entries we cannot stat.
            results.append(entry + suffix)

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

    def render_call(args, theme, context):
        text = context["lastComponent"] if isinstance(context.get("lastComponent"), Text) else Text("", 0, 0)
        text.set_text(_format_ls_call(args, theme, context["cwd"]))
        return text

    def render_result(result, options, theme, context):
        text = context["lastComponent"] if isinstance(context.get("lastComponent"), Text) else Text("", 0, 0)
        text.set_text(_format_ls_result(result, options, theme, context["showImages"]))
        return text

    return ToolDefinition(
        name="ls",
        label="ls",
        description=(
            "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. "
            f"Includes dotfiles. Output is truncated to {DEFAULT_LIMIT} entries or "
            f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
        ),
        prompt_snippet="List directory contents",
        parameters=LS_SCHEMA,
        execute=execute,
        render_call=render_call,
        render_result=render_result,
    )


def create_ls_tool(cwd: str, **options) -> WrappedDefinitionTool:
    return wrap_tool_definition(create_ls_tool_definition(cwd, **options))
