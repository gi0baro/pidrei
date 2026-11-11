"""Mirror of pi coding-agent src/core/tools/find.ts."""

import dataclasses
import math
import os
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent

from ...utils.tools_manager import ensure_tool, missing_tool_message
from ..extensions.types import ToolDefinition
from .grep import _run_streaming_lines
from .path_utils import path_exists_blocking, resolve_to_cwd
from .renderers.find import find_renderers
from .tool_definition_wrapper import WrappedDefinitionTool, wrap_tool_definition
from .truncate import DEFAULT_MAX_BYTES, TruncationResult, format_size, truncate_head, utf8_byte_length


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
# JS `Number.MAX_SAFE_INTEGER`.
_MAX_SAFE_INTEGER = 2**53 - 1


@dataclass(slots=True)
class FindToolDetails:
    truncation: TruncationResult | None = None
    result_limit_reached: int | None = None


def _is_inside_git_repo_blocking(path: str) -> bool:
    """Whether `path` or an ancestor has a `.git` entry (blocking: pool-side)."""
    current = path
    while True:
        if path_exists_blocking(os.path.join(current, ".git")):
            return True
        parent = os.path.dirname(current)
        if parent == current:
            return False
        current = parent


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


def _build_details_result(
    truncation: TruncationResult, effective_limit: int, limit_notice: str | None
) -> AgentToolResult:
    """The result both paths share; `limit_notice` is set when the result limit was reached."""
    result_output = truncation.content
    details = FindToolDetails()
    has_details = False
    notices: list[str] = []
    if limit_notice is not None:
        notices.append(limit_notice)
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

    async def execute(_tool_call_id, params, cancel=None, _on_update=None, ctx=None):
        pattern = params["pattern"]
        search_dir = params.get("path")
        limit = params.get("limit")

        _throw_if_aborted(cancel)

        search_path = resolve_to_cwd(search_dir or ".", (ctx.cwd if ctx is not None else None) or cwd)
        # fd treats --max-results 0 as unlimited, so a limit below 1 means 1.
        effective_limit = max(1, math.floor(limit if limit is not None else DEFAULT_LIMIT))

        # If custom operations provide glob(), use that instead of fd.
        if custom_ops is not None and getattr(custom_ops, "glob", None) is not None:
            if not await custom_ops.exists(search_path):
                raise Exception(f"Path not found: {search_path}")
            _throw_if_aborted(cancel)
            # One result past the limit tells "exactly N matches" apart from "more than N".
            results = await custom_ops.glob(
                pattern, search_path, ignore=["**/node_modules/**", "**/.git/**"], limit=effective_limit + 1
            )
            _throw_if_aborted(cancel)
            if not results:
                return AgentToolResult(content=[TextContent(text="No files found matching pattern")], details=None)

            result_limit_reached = len(results) > effective_limit
            # Relativize paths against the search root for stable output.
            relativized = [
                relativize_find_result_path(result_path, search_path) for result_path in results[:effective_limit]
            ]
            truncation = truncate_head("\n".join(relativized), max_lines=_MAX_SAFE_INTEGER)
            return _build_details_result(
                truncation,
                effective_limit,
                f"{effective_limit} results limit reached" if result_limit_reached else None,
            )

        # Default implementation uses fd.
        fd_path = await ensure_tool("fd")
        _throw_if_aborted(cancel)
        if not fd_path:
            raise Exception(missing_tool_message("fd"))

        args: list[str] = [fd_path, "--color=never", "--hidden"]

        # fd normally ignores .gitignore outside git repos, so keep --no-require-git
        # there. Inside repos, use fd's default git-aware behavior so parent
        # .gitignore rules stop at nested repo boundaries (pi#5960).
        if not await tonio.spawn_blocking(_is_inside_git_repo_blocking, search_path):
            args.append("--no-require-git")

        # fd --glob matches against the basename unless --full-path is set; in --full-path
        # mode it matches against the absolute candidate path, so a path-containing
        # pattern like 'src/**/*.spec.ts' needs a leading '**/' to match anything.
        effective_pattern = pattern
        is_path_pattern = "/" in effective_pattern
        if (
            is_path_pattern
            and not effective_pattern.startswith("/")
            and not effective_pattern.startswith("**/")
            and effective_pattern != "**"
        ):
            effective_pattern = f"**/{effective_pattern}"
        # One result past the limit tells "exactly N matches" apart from "more than N" (#7121).
        args.extend(["--glob", "--max-results", str(effective_limit + 1)])
        if is_path_pattern:
            args.append("--full-path")
        args.extend(["--", effective_pattern, search_path])

        # An abort during the .git lookup above must not start fd.
        _throw_if_aborted(cancel)

        # Results within the limit, stored only up to the first one past the byte cap:
        # truncate_head drops the rest anyway, so they are counted but not kept.
        stored_results: list[str] = []
        result_count = 0
        # Byte size of all results within the limit, each counted with a trailing newline.
        output_bytes = 0
        result_limit_reached = False

        def collect(raw_line: str) -> bool:
            nonlocal result_count, output_bytes, result_limit_reached
            line = raw_line.removesuffix("\r").strip()
            if not line:
                return False
            if result_count == effective_limit:
                result_limit_reached = True
                return False
            result = relativize_find_result_path(line, search_path)
            result_count += 1
            if output_bytes - 1 <= DEFAULT_MAX_BYTES:
                stored_results.append(result)
            output_bytes += utf8_byte_length(result) + 1
            return False

        exit_code, stderr = await _run_streaming_lines(args, cancel, collect)
        _throw_if_aborted(cancel)

        if exit_code != 0 and result_count == 0:
            raise Exception(stderr.strip() or f"fd exited with code {exit_code}")
        if result_count == 0:
            return AgentToolResult(content=[TextContent(text="No files found matching pattern")], details=None)

        # Stored results extend past the byte cap when more exist, so truncate_head decides
        # truncation as for the full output; only the totals need the unstored rest.
        truncation = dataclasses.replace(
            truncate_head("\n".join(stored_results), max_lines=_MAX_SAFE_INTEGER),
            total_lines=result_count,
            total_bytes=output_bytes - 1,
        )
        limit_notice = (
            f"{effective_limit} results limit reached. Use limit={effective_limit * 2} for more, or refine pattern"
            if result_limit_reached
            else None
        )
        return _build_details_result(truncation, effective_limit, limit_notice)

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
