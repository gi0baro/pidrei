"""Mirror of pi coding-agent src/core/tools/grep.ts."""

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio
from tonio.colored import fs

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent
from pidrei_tui import Text

from ...modes.interactive.components.keybinding_hints import key_hint
from ...utils.tools_manager import ensure_tool, missing_tool_message
from ..extensions.types import ToolDefinition
from .path_utils import resolve_to_cwd
from .render_utils import get_text_output, invalid_arg_text, shorten_path, str_or_none
from .tool_definition_wrapper import WrappedDefinitionTool, wrap_tool_definition
from .truncate import (
    DEFAULT_MAX_BYTES,
    GREP_MAX_LINE_LENGTH,
    TruncationResult,
    format_size,
    truncate_head,
    truncate_line,
)


def _format_grep_call(args: dict | None, theme) -> str:
    args = args or {}
    pattern = str_or_none(args.get("pattern"))
    raw_path = str_or_none(args.get("path"))
    path = shorten_path(raw_path or ".") if raw_path is not None else None
    glob = str_or_none(args.get("glob"))
    limit = args.get("limit")
    invalid_arg = invalid_arg_text(theme)
    text = (
        theme.fg("toolTitle", theme.bold("grep"))
        + " "
        + (invalid_arg if pattern is None else theme.fg("accent", f"/{pattern or ''}/"))
        + theme.fg("toolOutput", f" in {invalid_arg if path is None else path}")
    )
    if glob:
        text += theme.fg("toolOutput", f" ({glob})")
    if limit is not None:
        text += theme.fg("toolOutput", f" limit {limit}")
    return text


def _format_grep_result(result, options: dict, theme, show_images: bool) -> str:
    output = get_text_output(result, show_images).strip()
    text = ""
    if output:
        lines = output.split("\n")
        max_lines = len(lines) if options.get("expanded") else 15
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
    match_limit = getattr(details, "match_limit_reached", None) if details is not None else None
    truncation = getattr(details, "truncation", None) if details is not None else None
    lines_truncated = getattr(details, "lines_truncated", None) if details is not None else None
    if match_limit or (truncation is not None and truncation.truncated) or lines_truncated:
        warnings = []
        if match_limit:
            warnings.append(f"{match_limit} matches limit")
        if truncation is not None and truncation.truncated:
            max_bytes = truncation.max_bytes if truncation.max_bytes is not None else DEFAULT_MAX_BYTES
            warnings.append(f"{format_size(max_bytes)} limit")
        if lines_truncated:
            warnings.append("some lines truncated")
        text += "\n" + theme.fg("warning", f"[Truncated: {', '.join(warnings)}]")
    return text


GREP_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Search pattern (regex or literal string)"},
        "path": {"type": "string", "description": "Directory or file to search (default: current directory)"},
        "glob": {"type": "string", "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'"},
        "ignoreCase": {"type": "boolean", "description": "Case-insensitive search (default: false)"},
        "literal": {
            "type": "boolean",
            "description": "Treat pattern as literal string instead of regex (default: false)",
        },
        "context": {
            "type": "number",
            "description": "Number of lines to show before and after each match (default: 0)",
        },
        "limit": {"type": "number", "description": "Maximum number of matches to return (default: 100)"},
    },
    "required": ["pattern"],
}

GREP_TOOL_SYSTEM_PROMPT_CONTRIBUTION: dict[str, Any] = {
    "snippet": "Search file contents for patterns (respects .gitignore)",
    "guidelines": (),
}

DEFAULT_LIMIT = 100


@dataclass(slots=True)
class GrepToolDetails:
    truncation: TruncationResult | None = None
    match_limit_reached: int | None = None
    lines_truncated: bool | None = None


class LocalGrepOperations:
    async def is_directory(self, absolute_path: str) -> bool:
        if not await fs.Path(absolute_path).exists():
            raise Exception(f"Path not found: {absolute_path}")
        return await fs.Path(absolute_path).is_dir()

    async def read_file(self, absolute_path: str) -> str:
        return await fs.Path(absolute_path).read_text(encoding="utf-8", errors="replace", newline="")


async def _run_and_capture_lines(argv: list[str], cancel) -> tuple[int | None, list[str], str]:
    """Spawn a process and collect stdout lines and stderr text; kill on cancel."""
    process = await tonio.open_process(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_parts: list[bytes] = []
    stderr_parts: list[bytes] = []

    unsubscribe = None
    if cancel is not None:
        if cancel.cancelled:
            process.kill()
        else:
            unsubscribe = cancel.on_cancel(lambda _reason: process.kill())

    async def read_all(stream, parts: list[bytes]) -> None:
        if stream is None:
            return
        try:
            while True:
                chunk = await stream.receive_some()
                if not chunk:
                    return
                parts.append(chunk)
        except Exception:
            pass

    try:
        await tonio.spawn(read_all(process.stdout, stdout_parts), read_all(process.stderr, stderr_parts))
        exit_code = await process.wait()
    finally:
        if unsubscribe is not None:
            unsubscribe()

    stdout_text = b"".join(stdout_parts).decode("utf-8", "replace")
    stderr_text = b"".join(stderr_parts).decode("utf-8", "replace")
    lines = stdout_text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return exit_code, lines, stderr_text


def create_grep_tool_definition(cwd: str, *, operations: Any = None) -> ToolDefinition:
    custom_ops = operations

    async def execute(_tool_call_id, params, cancel=None, _on_update=None, _ctx=None):
        pattern = params["pattern"]
        search_dir = params.get("path")
        glob = params.get("glob")
        ignore_case = params.get("ignoreCase")
        literal = params.get("literal")
        context = params.get("context")
        limit = params.get("limit")

        if cancel is not None and cancel.cancelled:
            raise Exception("Operation aborted")

        rg_path = await ensure_tool("rg")
        if not rg_path:
            raise Exception(missing_tool_message("rg"))

        search_path = resolve_to_cwd(search_dir or ".", cwd)
        ops = custom_ops if custom_ops is not None else LocalGrepOperations()
        try:
            is_directory = await ops.is_directory(search_path)
        except Exception:
            raise Exception(f"Path not found: {search_path}")

        context_value = int(context) if context and context > 0 else 0
        effective_limit = max(1, int(limit) if limit is not None else DEFAULT_LIMIT)

        def format_path(file_path: str) -> str:
            if is_directory:
                relative = os.path.relpath(file_path, search_path)
                if relative and not relative.startswith(".."):
                    return relative
            return os.path.basename(file_path)

        file_cache: dict[str, list[str]] = {}

        async def get_file_lines(file_path: str) -> list[str]:
            lines = file_cache.get(file_path)
            if lines is None:
                try:
                    content = await ops.read_file(file_path)
                    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                except Exception:
                    lines = []
                file_cache[file_path] = lines
            return lines

        args: list[str] = [rg_path, "--json", "--line-number", "--color=never", "--hidden"]
        if ignore_case:
            args.append("--ignore-case")
        if literal:
            args.append("--fixed-strings")
        if glob:
            args.extend(["--glob", glob])
        args.extend(["--", pattern, search_path])

        exit_code, raw_lines, stderr = await _run_and_capture_lines(args, cancel)
        if cancel is not None and cancel.cancelled:
            raise Exception("Operation aborted")

        # Collect matches from the JSON event stream, capped at the match limit.
        matches: list[tuple[str, int, str | None]] = []
        match_limit_reached = False
        lines_truncated = False
        for line in raw_lines:
            if not line.strip() or len(matches) >= effective_limit:
                if len(matches) >= effective_limit:
                    break
                continue
            try:
                event = json.loads(line)
            except Exception:  # noqa: S112
                continue
            if event.get("type") == "match":
                data = event.get("data") or {}
                file_path = (data.get("path") or {}).get("text")
                line_number = data.get("line_number")
                line_text = (data.get("lines") or {}).get("text")
                if file_path and isinstance(line_number, int):
                    matches.append((file_path, line_number, line_text))
                if len(matches) >= effective_limit:
                    match_limit_reached = True
                    break

        if not matches:
            if exit_code not in (0, 1) and not match_limit_reached:
                raise Exception(stderr.strip() or f"ripgrep exited with code {exit_code}")
            return AgentToolResult(content=[TextContent(text="No matches found")], details=None)

        if exit_code not in (0, 1, None) and not match_limit_reached:
            raise Exception(stderr.strip() or f"ripgrep exited with code {exit_code}")

        output_lines: list[str] = []

        async def format_block(file_path: str, line_number: int) -> list[str]:
            nonlocal lines_truncated
            relative_path = format_path(file_path)
            lines = await get_file_lines(file_path)
            if not lines:
                return [f"{relative_path}:{line_number}: (unable to read file)"]
            block: list[str] = []
            start = max(1, line_number - context_value) if context_value > 0 else line_number
            end = min(len(lines), line_number + context_value) if context_value > 0 else line_number
            for current in range(start, end + 1):
                line_text = lines[current - 1] if current - 1 < len(lines) else ""
                sanitized = line_text.replace("\r", "")
                is_match_line = current == line_number
                # Truncate long lines so grep output stays compact.
                truncated = truncate_line(sanitized)
                if truncated.was_truncated:
                    lines_truncated = True
                if is_match_line:
                    block.append(f"{relative_path}:{current}: {truncated.text}")
                else:
                    block.append(f"{relative_path}-{current}- {truncated.text}")
            return block

        for file_path, line_number, line_text in matches:
            if context_value == 0 and line_text is not None:
                relative_path = format_path(file_path)
                sanitized = line_text.replace("\r\n", "\n").replace("\r", "")
                sanitized = sanitized.removesuffix("\n")
                truncated = truncate_line(sanitized)
                if truncated.was_truncated:
                    lines_truncated = True
                output_lines.append(f"{relative_path}:{line_number}: {truncated.text}")
            else:
                output_lines.extend(await format_block(file_path, line_number))

        raw_output = "\n".join(output_lines)
        # Apply byte truncation. There is no line limit here because the match limit already capped rows.
        truncation = truncate_head(raw_output, max_lines=2**53 - 1)
        output = truncation.content
        details = GrepToolDetails()
        has_details = False
        # Build actionable notices for truncation and match limits.
        notices: list[str] = []
        if match_limit_reached:
            notices.append(
                f"{effective_limit} matches limit reached. Use limit={effective_limit * 2} for more, or refine pattern"
            )
            details.match_limit_reached = effective_limit
            has_details = True
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details.truncation = truncation
            has_details = True
        if lines_truncated:
            notices.append(f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines")
            details.lines_truncated = True
            has_details = True
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"
        return AgentToolResult(content=[TextContent(text=output)], details=details if has_details else None)

    def render_call(args, theme, context):
        text = context["lastComponent"] if isinstance(context.get("lastComponent"), Text) else Text("", 0, 0)
        text.set_text(_format_grep_call(args, theme))
        return text

    def render_result(result, options, theme, context):
        text = context["lastComponent"] if isinstance(context.get("lastComponent"), Text) else Text("", 0, 0)
        text.set_text(_format_grep_result(result, options, theme, context["showImages"]))
        return text

    return ToolDefinition(
        name="grep",
        label="grep",
        description=(
            "Search file contents for a pattern. Returns matching lines with file paths and line numbers. "
            f"Respects .gitignore. Output is truncated to {DEFAULT_LIMIT} matches or "
            f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). Long lines are truncated to "
            f"{GREP_MAX_LINE_LENGTH} chars."
        ),
        prompt_snippet=GREP_TOOL_SYSTEM_PROMPT_CONTRIBUTION["snippet"],
        parameters=GREP_SCHEMA,
        execute=execute,
        render_call=render_call,
        render_result=render_result,
    )


def create_grep_tool(cwd: str, **options) -> WrappedDefinitionTool:
    return wrap_tool_definition(create_grep_tool_definition(cwd, **options))
