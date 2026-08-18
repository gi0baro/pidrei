"""Truncated Tool

Demonstrates proper output truncation for custom tools.

Custom tools MUST truncate their output to avoid overwhelming the LLM context.
The built-in limit is 50KB (~10k tokens) and 2000 lines, whichever is hit
first.

This example shows how to:
1. Use the built-in truncation utilities
2. Write full output to a temp file when truncated
3. Inform the LLM where to find the complete output
4. Custom rendering of tool calls and results

The `rg` tool here wraps ripgrep with proper truncation. Compare this to the
built-in `grep` tool in pidrei/core/tools/grep.py for a more complete
implementation.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/truncated_tool.py
"""

import os
import tempfile
from dataclasses import dataclass

import tonio.colored as tonio
from tonio.colored import fs

from pidrei.core.extensions.types import ToolDefinition
from pidrei.core.tools import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_head,
    with_file_mutation_queue,
)
from pidrei.core.tools.file_mutation_queue import resolve_mutation_queue_key
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent
from pidrei_tui import Text


RG_PARAMS = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Search pattern (regex)"},
        "path": {"type": "string", "description": "Directory to search (default: current directory)"},
        "glob": {"type": "string", "description": "File glob pattern, e.g. '*.py'"},
    },
    "required": ["pattern"],
}


@dataclass(slots=True)
class RgDetails:
    pattern: str
    path: str | None = None
    glob: str | None = None
    match_count: int = 0
    truncation: TruncationResult | None = None
    full_output_path: str | None = None


def _details_get(details, key: str):
    if isinstance(details, dict):
        return details.get(key)
    return getattr(details, key, None)


def _first_text(result) -> str | None:
    content = result["content"] if isinstance(result, dict) else result.content
    first = content[0] if content else None
    if first is None:
        return None
    if (first.get("type") if isinstance(first, dict) else first.type) != "text":
        return None
    return first.get("text") if isinstance(first, dict) else first.text


def extension(pi):
    async def execute(_tool_call_id, params, _cancel=None, _on_update=None, ctx=None):
        pattern = params["pattern"]
        search_path = params.get("path")
        glob = params.get("glob")

        # Build the ripgrep invocation. pi shells out with execSync; pidrei
        # runs subprocesses through pi.exec, which never blocks the runtime.
        args = ["--line-number", "--color=never"]
        if glob:
            args.extend(["--glob", glob])
        args.append(pattern)
        args.append(search_path or ".")

        result = await pi.exec("rg", args, cwd=ctx.cwd)

        # ripgrep exits with 1 when no matches were found (and is silent on
        # stderr); anything else with stderr output is a real failure.
        if result.code not in (0, 1) or (result.code == 1 and result.stderr.strip()):
            raise Exception(f"ripgrep failed: {result.stderr.strip() or f'exit code {result.code}'}")

        if not result.stdout.strip():
            return AgentToolResult(
                content=[TextContent(text="No matches found")],
                details=RgDetails(pattern=pattern, path=search_path, glob=glob, match_count=0),
            )
        output = result.stdout

        # Apply truncation using built-in utilities.
        # truncate_head keeps the first N lines/bytes (good for search results)
        # truncate_tail keeps the last N lines/bytes (good for logs/command output)
        truncation = truncate_head(output, max_lines=DEFAULT_MAX_LINES, max_bytes=DEFAULT_MAX_BYTES)

        # Count matches (each non-empty line with a match)
        match_count = sum(1 for line in output.split("\n") if line.strip())

        details = RgDetails(pattern=pattern, path=search_path, glob=glob, match_count=match_count)
        result_text = truncation.content

        if truncation.truncated:
            # Save full output to a temp file so the LLM can access it if needed
            temp_dir = await tonio.spawn_blocking(tempfile.mkdtemp, prefix="pidrei-rg-")
            temp_file = os.path.join(temp_dir, "output.txt")
            queue_key = await resolve_mutation_queue_key(temp_file)

            async def write_full_output() -> None:
                await fs.Path(temp_file).write_text(output, encoding="utf-8")

            await with_file_mutation_queue(temp_file, write_full_output, queue_key=queue_key)

            details.truncation = truncation
            details.full_output_path = temp_file

            # Add a truncation notice - this helps the LLM understand the output is incomplete
            truncated_lines = truncation.total_lines - truncation.output_lines
            truncated_bytes = truncation.total_bytes - truncation.output_bytes

            result_text += (
                f"\n\n[Output truncated: showing {truncation.output_lines} of {truncation.total_lines} lines"
                f" ({format_size(truncation.output_bytes)} of {format_size(truncation.total_bytes)})."
                f" {truncated_lines} lines ({format_size(truncated_bytes)}) omitted."
                f" Full output saved to: {temp_file}]"
            )

        return AgentToolResult(content=[TextContent(text=result_text)], details=details)

    # Custom rendering of the tool call (shown before/during execution)
    def render_call(args, theme, _context):
        args = args or {}
        text = theme.fg("toolTitle", theme.bold("rg "))
        text += theme.fg("accent", f'"{args.get("pattern", "")}"')
        if args.get("path"):
            text += theme.fg("muted", f" in {args['path']}")
        if args.get("glob"):
            text += theme.fg("dim", f" --glob {args['glob']}")
        return Text(text, 0, 0)

    # Custom rendering of the tool result
    def render_result(result, options, theme, _context):
        details = result["details"] if isinstance(result, dict) else result.details

        # Handle streaming/partial results
        if options.get("isPartial"):
            return Text(theme.fg("warning", "Searching..."), 0, 0)

        # No matches
        if not details or not _details_get(details, "match_count"):
            return Text(theme.fg("dim", "No matches found"), 0, 0)

        # Build result display
        text = theme.fg("success", f"{_details_get(details, 'match_count')} matches")

        # Show truncation warning if applicable
        truncation = _details_get(details, "truncation")
        if truncation is not None and truncation.truncated:
            text += theme.fg("warning", " (truncated)")

        # In expanded view, show the actual matches
        if options.get("expanded"):
            output = _first_text(result)
            if output is not None:
                # Show first 20 lines in expanded view, or all if fewer
                lines = output.split("\n")
                for line in lines[:20]:
                    text += f"\n{theme.fg('dim', line)}"
                if len(lines) > 20:
                    text += f"\n{theme.fg('muted', '... (use read tool to see full output)')}"

            # Show temp file path if truncated
            full_output_path = _details_get(details, "full_output_path")
            if full_output_path:
                text += f"\n{theme.fg('dim', f'Full output: {full_output_path}')}"

        return Text(text, 0, 0)

    pi.register_tool(
        ToolDefinition(
            name="rg",
            label="ripgrep",
            # Document the truncation limits in the tool description so the LLM knows
            description=(
                f"Search file contents using ripgrep. Output is truncated to {DEFAULT_MAX_LINES} lines or "
                f"{format_size(DEFAULT_MAX_BYTES)} (whichever is hit first). If truncated, full output is "
                "saved to a temp file."
            ),
            parameters=RG_PARAMS,
            execute=execute,
            render_call=render_call,
            render_result=render_result,
        )
    )
