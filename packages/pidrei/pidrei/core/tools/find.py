"""Mirror of pi coding-agent src/core/tools/find.ts."""

import os
from dataclasses import dataclass
from typing import Any

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent

from ...utils.tools_manager import ensure_tool, missing_tool_message
from ..extensions.types import ToolDefinition
from .grep import _run_streaming_lines
from .path_utils import path_exists, resolve_to_cwd
from .renderers.find import find_renderers
from .tool_definition_wrapper import WrappedDefinitionTool, wrap_tool_definition
from .truncate import DEFAULT_MAX_BYTES, TruncationResult, format_size, truncate_head


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

FIND_TOOL_SYSTEM_PROMPT_CONTRIBUTION: dict[str, Any] = {
    "snippet": "Find files by glob pattern (respects .gitignore)",
    "guidelines": (),
}

DEFAULT_LIMIT = 1000


@dataclass(slots=True)
class FindToolDetails:
    truncation: TruncationResult | None = None
    result_limit_reached: int | None = None


def _throw_if_aborted(cancel: Any) -> None:
    if cancel is not None and cancel.cancelled:
        raise Exception("Operation aborted")


def relativize_find_result_path(result_path: str, search_path: str) -> str:
    """Relativize a find result against the search root, keeping a trailing separator.

    POSIX-only, so pi's injectable `pathModule` and its `\\`-separator branches
    are dropped: `os.sep` is always `/` here.
    """
    had_trailing_separator = result_path.endswith(os.sep)
    relative_path = os.path.relpath(result_path, search_path) if os.path.isabs(result_path) else result_path
    posix_path = relative_path.replace(os.sep, "/")
    return f"{posix_path}/" if had_trailing_separator and not posix_path.endswith("/") else posix_path


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
    # `operations` members (`exists`/`glob`) are async-only. pi types them
    # `Promise<T> | T` because its sync default blocks the event loop
    # (`existsSync`); a sync impl doing real fs work would block this runtime,
    # so the union is deliberately not ported.
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
            if not await custom_ops.exists(search_path):
                raise Exception(f"Path not found: {search_path}")
            _throw_if_aborted(cancel)
            results = await custom_ops.glob(
                pattern, search_path, ignore=["**/node_modules/**", "**/.git/**"], limit=effective_limit
            )
            _throw_if_aborted(cancel)
            if not results:
                return AgentToolResult(content=[TextContent(text="No files found matching pattern")], details=None)

            # Relativize paths against the search root for stable output.
            relativized = [relativize_find_result_path(result_path, search_path) for result_path in results]
            return _build_details_result(relativized, effective_limit)

        # Default implementation uses fd.
        fd_path = await ensure_tool("fd")
        _throw_if_aborted(cancel)
        if not fd_path:
            raise Exception(missing_tool_message("fd"))

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

        lines: list[str] = []

        def collect(line: str) -> bool:
            lines.append(line)
            return False

        exit_code, stderr = await _run_streaming_lines(args, cancel, collect)
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
            relativized.append(relativize_find_result_path(line, search_path))

        return _build_details_result(relativized, effective_limit)

    return ToolDefinition(
        name="find",
        label="find",
        description=(
            "Search for files by glob pattern. Returns matching file paths relative to the search directory. "
            f"Respects .gitignore. Output is truncated to {DEFAULT_LIMIT} results or "
            f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
        ),
        prompt_snippet=FIND_TOOL_SYSTEM_PROMPT_CONTRIBUTION["snippet"],
        parameters=FIND_SCHEMA,
        execute=execute,
        render_call=find_renderers.render_call,
        render_result=find_renderers.render_result,
    )


def create_find_tool(cwd: str, **options) -> WrappedDefinitionTool:
    return wrap_tool_definition(create_find_tool_definition(cwd, **options))
