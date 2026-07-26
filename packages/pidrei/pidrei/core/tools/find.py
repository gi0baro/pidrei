"""Mirror of pi coding-agent src/core/tools/find.ts."""

import os
from dataclasses import dataclass
from typing import Any

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent
from pidrei_tui import Text

from ...modes.interactive.components.keybinding_hints import key_hint
from ...utils.tools_manager import ensure_tool
from ..extensions.types import ToolDefinition
from .grep import _run_and_capture_lines
from .path_utils import path_exists, resolve_to_cwd
from .render_utils import get_text_output, invalid_arg_text, shorten_path, str_or_none
from .tool_definition_wrapper import WrappedDefinitionTool, wrap_tool_definition
from .truncate import DEFAULT_MAX_BYTES, TruncationResult, format_size, truncate_head


def _format_find_call(args: dict | None, theme) -> str:
    args = args or {}
    pattern = str_or_none(args.get("pattern"))
    raw_path = str_or_none(args.get("path"))
    path = shorten_path(raw_path or ".") if raw_path is not None else None
    limit = args.get("limit")
    invalid_arg = invalid_arg_text(theme)
    text = (
        theme.fg("toolTitle", theme.bold("find"))
        + " "
        + (invalid_arg if pattern is None else theme.fg("accent", pattern or ""))
        + theme.fg("toolOutput", f" in {invalid_arg if path is None else path}")
    )
    if limit is not None:
        text += theme.fg("toolOutput", f" (limit {limit})")
    return text


def _format_find_result(result, options: dict, theme, show_images: bool) -> str:
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
    result_limit = getattr(details, "result_limit_reached", None) if details is not None else None
    truncation = getattr(details, "truncation", None) if details is not None else None
    if result_limit or (truncation is not None and truncation.truncated):
        warnings = []
        if result_limit:
            warnings.append(f"{result_limit} results limit")
        if truncation is not None and truncation.truncated:
            max_bytes = truncation.max_bytes if truncation.max_bytes is not None else DEFAULT_MAX_BYTES
            warnings.append(f"{format_size(max_bytes)} limit")
        text += "\n" + theme.fg("warning", f"[Truncated: {', '.join(warnings)}]")
    return text


FIND_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'",
        },
        "path": {"type": "string", "description": "Directory to search in (default: current directory)"},
        "limit": {"type": "number", "description": "Maximum number of results (default: 1000)"},
    },
    "required": ["pattern"],
}

DEFAULT_LIMIT = 1000


@dataclass(slots=True)
class FindToolDetails:
    truncation: TruncationResult | None = None
    result_limit_reached: int | None = None


def _throw_if_aborted(cancel: Any) -> None:
    if cancel is not None and cancel.cancelled:
        raise Exception("Operation aborted")


def _build_details_result(relativized: list[str], effective_limit: int) -> AgentToolResult:
    result_limit_reached = len(relativized) >= effective_limit
    raw_output = "\n".join(relativized)
    truncation = truncate_head(raw_output, max_lines=2**53 - 1)
    result_output = truncation.content
    details = FindToolDetails()
    has_details = False
    notices: list[str] = []
    if result_limit_reached:
        notices.append(
            f"{effective_limit} results limit reached. Use limit={effective_limit * 2} for more, or refine pattern"
        )
        details.result_limit_reached = effective_limit
        has_details = True
    if truncation.truncated:
        notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
        details.truncation = truncation
        has_details = True
    if notices:
        result_output += f"\n\n[{'. '.join(notices)}]"
    return AgentToolResult(content=[TextContent(text=result_output)], details=details if has_details else None)


def create_find_tool_definition(cwd: str, *, operations: Any = None) -> ToolDefinition:
    custom_ops = operations

    async def execute(_tool_call_id, params, cancel=None, _on_update=None, _ctx=None):
        pattern = params["pattern"]
        search_dir = params.get("path")
        limit = params.get("limit")

        _throw_if_aborted(cancel)

        search_path = resolve_to_cwd(search_dir or ".", cwd)
        effective_limit = int(limit) if limit is not None else DEFAULT_LIMIT

        # If custom operations provide glob(), use that instead of fd.
        if custom_ops is not None and getattr(custom_ops, "glob", None) is not None:
            if not await _maybe_await(custom_ops.exists(search_path)):
                raise Exception(f"Path not found: {search_path}")
            _throw_if_aborted(cancel)
            results = await _maybe_await(
                custom_ops.glob(
                    pattern, search_path, ignore=["**/node_modules/**", "**/.git/**"], limit=effective_limit
                )
            )
            _throw_if_aborted(cancel)
            if not results:
                return AgentToolResult(content=[TextContent(text="No files found matching pattern")], details=None)

            # Relativize paths against the search root for stable output.
            relativized = []
            for result_path in results:
                if result_path.startswith(search_path):
                    relativized.append(result_path[len(search_path) + 1 :])
                else:
                    relativized.append(os.path.relpath(result_path, search_path))
            return _build_details_result(relativized, effective_limit)

        # Default implementation uses fd.
        fd_path = await ensure_tool("fd", True)
        _throw_if_aborted(cancel)
        if not fd_path:
            raise Exception("fd is not available and could not be downloaded")

        args: list[str] = [fd_path, "--glob", "--color=never", "--hidden"]

        # fd normally ignores .gitignore outside git repos, so keep --no-require-git
        # there. Inside repos, use fd's default git-aware behavior so parent
        # .gitignore rules stop at nested repo boundaries (pi#5960).
        inside_git_repo = False
        current = search_path
        while True:
            if path_exists(os.path.join(current, ".git")):
                inside_git_repo = True
                break
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        if not inside_git_repo:
            args.append("--no-require-git")
        args.extend(["--max-results", str(effective_limit)])

        # fd --glob matches against the basename unless --full-path is set; in --full-path
        # mode it matches against the absolute candidate path, so a path-containing
        # pattern like 'src/**/*.spec.ts' needs a leading '**/' to match anything.
        effective_pattern = pattern
        if "/" in pattern:
            args.append("--full-path")
            if not pattern.startswith("/") and not pattern.startswith("**/") and pattern != "**":
                effective_pattern = f"**/{pattern}"
        args.extend(["--", effective_pattern, search_path])

        exit_code, lines, stderr = await _run_and_capture_lines(args, cancel)
        _throw_if_aborted(cancel)

        output = "\n".join(lines)
        if exit_code != 0:
            error_message = stderr.strip() or f"fd exited with code {exit_code}"
            if not output:
                raise Exception(error_message)
        if not output:
            return AgentToolResult(content=[TextContent(text="No files found matching pattern")], details=None)

        relativized: list[str] = []
        for raw_line in lines:
            line = raw_line.removesuffix("\r").strip()
            if not line:
                continue
            had_trailing_slash = line.endswith("/")
            if line.startswith(search_path):
                relative_path = line[len(search_path) + 1 :]
            else:
                relative_path = os.path.relpath(line, search_path)
            if had_trailing_slash and not relative_path.endswith("/"):
                relative_path += "/"
            relativized.append(relative_path)

        return _build_details_result(relativized, effective_limit)

    def render_call(args, theme, context):
        text = context["lastComponent"] if isinstance(context.get("lastComponent"), Text) else Text("", 0, 0)
        text.set_text(_format_find_call(args, theme))
        return text

    def render_result(result, options, theme, context):
        text = context["lastComponent"] if isinstance(context.get("lastComponent"), Text) else Text("", 0, 0)
        text.set_text(_format_find_result(result, options, theme, context["showImages"]))
        return text

    return ToolDefinition(
        name="find",
        label="find",
        description=(
            "Search for files by glob pattern. Returns matching file paths relative to the search directory. "
            f"Respects .gitignore. Output is truncated to {DEFAULT_LIMIT} results or "
            f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
        ),
        prompt_snippet="Find files by glob pattern (respects .gitignore)",
        parameters=FIND_SCHEMA,
        execute=execute,
        render_call=render_call,
        render_result=render_result,
    )


async def _maybe_await(value: Any) -> Any:
    import inspect

    return await value if inspect.isawaitable(value) else value


def create_find_tool(cwd: str, **options) -> WrappedDefinitionTool:
    return wrap_tool_definition(create_find_tool_definition(cwd, **options))
