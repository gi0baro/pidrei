"""Built-in Tool Renderer

Custom rendering for built-in tools.

Demonstrates how to override the rendering of built-in tools (read, bash,
edit, write) without changing their behavior. Each tool is re-registered with
the same name, delegating execution to the original implementation while
providing compact custom render_call/render_result functions.

This is useful for users who prefer more concise tool output, or who want to
highlight specific information (e.g., showing only the diff stats for edit,
or just the exit code for bash).

How it works:
- register_tool() with the same name as a built-in replaces it entirely
- We create instances of the original tools via create_read_tool(), etc.
  and delegate execute() to them
- render_call() controls what's shown when the tool is invoked
- render_result() controls what's shown after execution completes
- render_shell="self" lets a tool render its own outer shell instead of
  using the default boxed shell from ToolExecutionComponent
- The `expanded` flag in render_result's options indicates whether the user
  has toggled the tool output open (ctrl+e)

Start pidrei with this extension:
    pidrei -e ./examples/extensions/built_in_tool_renderer.py
"""

import os
import re

from pidrei.core.extensions.types import ToolDefinition
from pidrei.core.tools import (
    create_bash_tool,
    create_edit_tool,
    create_read_tool,
    create_write_tool,
)
from pidrei_tui import Text


_EXIT_CODE_RE = re.compile(r"Command exited with code (\d+)")


def _blocks(result):
    return result["content"] if isinstance(result, dict) else result.content


def _block_type(block) -> str | None:
    if block is None:
        return None
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def _block_text(block) -> str:
    if block is None:
        return ""
    text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
    return text or ""


def _first_block(result):
    blocks = _blocks(result)
    return blocks[0] if blocks else None


def _details_get(result, key: str):
    details = result["details"] if isinstance(result, dict) else result.details
    if details is None:
        return None
    if isinstance(details, dict):
        return details.get(key)
    return getattr(details, key, None)


def extension(pi):
    cwd = os.getcwd()

    # --- Read tool: show path and line count ---
    original_read = create_read_tool(cwd)

    async def execute_read(tool_call_id, params, cancel=None, on_update=None, ctx=None):
        return await original_read.execute(tool_call_id, params, cancel, on_update, ctx)

    def render_read_call(args, theme, _context):
        args = args or {}
        text = theme.fg("toolTitle", theme.bold("read "))
        text += theme.fg("accent", args.get("path") or "")
        if args.get("offset") or args.get("limit"):
            parts = []
            if args.get("offset"):
                parts.append(f"offset={args['offset']}")
            if args.get("limit"):
                parts.append(f"limit={args['limit']}")
            text += theme.fg("dim", f" ({', '.join(parts)})")
        return Text(text, 0, 0)

    def render_read_result(result, options, theme, _context):
        if options.get("isPartial"):
            return Text(theme.fg("warning", "Reading..."), 0, 0)

        content = _first_block(result)
        if _block_type(content) == "image":
            return Text(theme.fg("success", "Image loaded"), 0, 0)
        if _block_type(content) != "text":
            return Text(theme.fg("error", "No content"), 0, 0)

        lines = _block_text(content).split("\n")
        text = theme.fg("success", f"{len(lines)} lines")

        truncation = _details_get(result, "truncation")
        if truncation is not None and truncation.truncated:
            text += theme.fg("warning", f" (truncated from {truncation.total_lines})")

        if options.get("expanded"):
            for line in lines[:15]:
                text += f"\n{theme.fg('dim', line)}"
            if len(lines) > 15:
                text += f"\n{theme.fg('muted', f'... {len(lines) - 15} more lines')}"

        return Text(text, 0, 0)

    pi.register_tool(
        ToolDefinition(
            name="read",
            label="read",
            description=original_read.description,
            parameters=original_read.parameters,
            execute=execute_read,
            render_call=render_read_call,
            render_result=render_read_result,
        )
    )

    # --- Bash tool: show command and exit code ---
    original_bash = create_bash_tool(cwd)

    async def execute_bash(tool_call_id, params, cancel=None, on_update=None, ctx=None):
        return await original_bash.execute(tool_call_id, params, cancel, on_update, ctx)

    def render_bash_call(args, theme, _context):
        args = args or {}
        text = theme.fg("toolTitle", theme.bold("$ "))
        command = args.get("command") or ""
        text += theme.fg("accent", f"{command[:77]}..." if len(command) > 80 else command)
        if args.get("timeout"):
            text += theme.fg("dim", f" (timeout: {args['timeout']}s)")
        return Text(text, 0, 0)

    def render_bash_result(result, options, theme, context):
        if options.get("isPartial"):
            return Text(theme.fg("warning", "Running..."), 0, 0)

        content = _first_block(result)
        output = _block_text(content) if _block_type(content) == "text" else ""

        # pi's bash output carries an "exit code: N" line; pidrei's raises with
        # "Command exited with code N" on failure, so parse that instead.
        exit_match = _EXIT_CODE_RE.search(output)
        exit_code = int(exit_match.group(1)) if exit_match else None
        line_count = sum(1 for line in output.split("\n") if line.strip())

        if exit_code is None and context.get("isError"):
            text = theme.fg("error", "failed")
        elif exit_code is None or exit_code == 0:
            text = theme.fg("success", "done")
        else:
            text = theme.fg("error", f"exit {exit_code}")
        text += theme.fg("dim", f" ({line_count} lines)")

        truncation = _details_get(result, "truncation")
        if truncation is not None and truncation.truncated:
            text += theme.fg("warning", " [truncated]")

        if options.get("expanded"):
            lines = output.split("\n")
            for line in lines[:20]:
                text += f"\n{theme.fg('dim', line)}"
            if len(lines) > 20:
                text += f"\n{theme.fg('muted', '... more output')}"

        return Text(text, 0, 0)

    pi.register_tool(
        ToolDefinition(
            name="bash",
            label="bash",
            description=original_bash.description,
            parameters=original_bash.parameters,
            execute=execute_bash,
            render_call=render_bash_call,
            render_result=render_bash_result,
        )
    )

    # --- Edit tool: show path and diff stats ---
    original_edit = create_edit_tool(cwd)

    async def execute_edit(tool_call_id, params, cancel=None, on_update=None, ctx=None):
        return await original_edit.execute(tool_call_id, params, cancel, on_update, ctx)

    def render_edit_call(args, theme, _context):
        args = args or {}
        text = theme.fg("toolTitle", theme.bold("edit "))
        text += theme.fg("accent", args.get("path") or "")
        return Text(text, 0, 0)

    def render_edit_result(result, options, theme, _context):
        if options.get("isPartial"):
            return Text(theme.fg("warning", "Editing..."), 0, 0)

        content = _first_block(result)
        if _block_type(content) == "text" and _block_text(content).startswith("Error"):
            return Text(theme.fg("error", _block_text(content).split("\n")[0]), 0, 0)

        diff = _details_get(result, "diff")
        if not diff:
            return Text(theme.fg("success", "Applied"), 0, 0)

        # Count additions and removals from the diff
        diff_lines = diff.split("\n")
        additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
        removals = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))

        text = theme.fg("success", f"+{additions}")
        text += theme.fg("dim", " / ")
        text += theme.fg("error", f"-{removals}")

        if options.get("expanded"):
            for line in diff_lines[:30]:
                if line.startswith("+") and not line.startswith("+++"):
                    text += f"\n{theme.fg('success', line)}"
                elif line.startswith("-") and not line.startswith("---"):
                    text += f"\n{theme.fg('error', line)}"
                else:
                    text += f"\n{theme.fg('dim', line)}"
            if len(diff_lines) > 30:
                text += f"\n{theme.fg('muted', f'... {len(diff_lines) - 30} more diff lines')}"

        return Text(text, 0, 0)

    pi.register_tool(
        ToolDefinition(
            name="edit",
            label="edit",
            description=original_edit.description,
            parameters=original_edit.parameters,
            render_shell="self",
            execute=execute_edit,
            render_call=render_edit_call,
            render_result=render_edit_result,
        )
    )

    # --- Write tool: show path and size ---
    original_write = create_write_tool(cwd)

    async def execute_write(tool_call_id, params, cancel=None, on_update=None, ctx=None):
        return await original_write.execute(tool_call_id, params, cancel, on_update, ctx)

    def render_write_call(args, theme, _context):
        args = args or {}
        text = theme.fg("toolTitle", theme.bold("write "))
        text += theme.fg("accent", args.get("path") or "")
        line_count = len((args.get("content") or "").split("\n"))
        text += theme.fg("dim", f" ({line_count} lines)")
        return Text(text, 0, 0)

    def render_write_result(result, options, theme, _context):
        if options.get("isPartial"):
            return Text(theme.fg("warning", "Writing..."), 0, 0)

        content = _first_block(result)
        if _block_type(content) == "text" and _block_text(content).startswith("Error"):
            return Text(theme.fg("error", _block_text(content).split("\n")[0]), 0, 0)

        return Text(theme.fg("success", "Written"), 0, 0)

    pi.register_tool(
        ToolDefinition(
            name="write",
            label="write",
            description=original_write.description,
            parameters=original_write.parameters,
            execute=execute_write,
            render_call=render_write_call,
            render_result=render_write_result,
        )
    )
