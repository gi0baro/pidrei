"""Minimal Mode

Demonstrates a "minimal" tool display mode.

This extension overrides built-in tools to provide custom rendering:
- Collapsed mode: Only shows the tool call (command/path), no output
- Expanded mode: Shows full output like the built-in renderers

This demonstrates how a "minimal mode" could work, where ctrl+e cycles
through:
- Standard: Shows truncated output (current default)
- Expanded: Shows full output (current expanded)
- Minimal: Shows only tool call, no output (this extension's collapsed mode)

Start pidrei with this extension:
    pidrei -e ./examples/extensions/minimal_mode.py

Then use ctrl+e to toggle between minimal (collapsed) and full (expanded)
views.
"""

import os

from pidrei.core.extensions.types import ToolDefinition
from pidrei.core.tools import (
    create_bash_tool,
    create_edit_tool,
    create_find_tool,
    create_grep_tool,
    create_ls_tool,
    create_read_tool,
    create_write_tool,
)
from pidrei.core.tools.render_utils import shorten_path
from pidrei_tui import Text


def _find_text_block(result):
    content = result["content"] if isinstance(result, dict) else result.content
    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type == "text":
            return block.get("text") if isinstance(block, dict) else block.text
    return None


def _create_built_in_tools(cwd: str) -> dict:
    return {
        "read": create_read_tool(cwd),
        "bash": create_bash_tool(cwd),
        "edit": create_edit_tool(cwd),
        "write": create_write_tool(cwd),
        "find": create_find_tool(cwd),
        "grep": create_grep_tool(cwd),
        "ls": create_ls_tool(cwd),
    }


def extension(pi):
    # Cache for built-in tools by cwd (in the closure: module globals reset
    # on /reload)
    tool_cache: dict[str, dict] = {}

    def get_built_in_tools(cwd: str) -> dict:
        tools = tool_cache.get(cwd)
        if tools is None:
            tools = _create_built_in_tools(cwd)
            tool_cache[cwd] = tools
        return tools

    # The parameter schemas and descriptions are taken verbatim from the
    # built-ins at registration time; execution re-resolves against the
    # session cwd.
    registration_tools = get_built_in_tools(os.getcwd())

    def make_delegate(name: str):
        async def execute(tool_call_id, params, cancel=None, on_update=None, ctx=None):
            tools = get_built_in_tools(ctx.cwd)
            return await tools[name].execute(tool_call_id, params, cancel, on_update, ctx)

        return execute

    def path_display(args, theme) -> str:
        path = shorten_path(args.get("path") or "")
        return theme.fg("accent", path) if path else theme.fg("toolOutput", "...")

    def render_minimal_result(result, options, theme, _context):
        """Shared result renderer: nothing collapsed, full output expanded."""
        # Minimal mode: show nothing in collapsed state
        if not options.get("expanded"):
            return Text("", 0, 0)

        # Expanded mode: show full output
        text = _find_text_block(result)
        if not text or not text.strip():
            return Text("", 0, 0)

        output = "\n".join(theme.fg("toolOutput", line) for line in text.strip().split("\n"))
        return Text(f"\n{output}", 0, 0)

    def make_count_result_renderer(unit: str):
        """Result renderer for search/list tools: a count collapsed, full
        output expanded."""

        def render_result(result, options, theme, _context):
            text = _find_text_block(result)
            if not options.get("expanded"):
                # Minimal: just show the count
                count = sum(1 for line in (text or "").strip().split("\n") if line) if text else 0
                if count > 0:
                    return Text(theme.fg("muted", f" → {count} {unit}"), 0, 0)
                return Text("", 0, 0)

            # Expanded: show full results
            if not text:
                return Text("", 0, 0)
            output = "\n".join(theme.fg("toolOutput", line) for line in text.strip().split("\n"))
            return Text(f"\n{output}", 0, 0)

        return render_result

    # =========================================================================
    # Read Tool
    # =========================================================================
    def render_read_call(args, theme, _context):
        args = args or {}
        display = path_display(args, theme)

        # Show line range if specified
        if args.get("offset") is not None or args.get("limit") is not None:
            start_line = args.get("offset") if args.get("offset") is not None else 1
            end_line = start_line + args["limit"] - 1 if args.get("limit") is not None else ""
            display += theme.fg("warning", f":{start_line}{f'-{end_line}' if end_line else ''}")

        return Text(f"{theme.fg('toolTitle', theme.bold('read'))} {display}", 0, 0)

    pi.register_tool(
        ToolDefinition(
            name="read",
            label="read",
            description=registration_tools["read"].description,
            parameters=registration_tools["read"].parameters,
            execute=make_delegate("read"),
            render_call=render_read_call,
            render_result=render_minimal_result,
        )
    )

    # =========================================================================
    # Bash Tool
    # =========================================================================
    def render_bash_call(args, theme, _context):
        args = args or {}
        command = args.get("command") or "..."
        timeout = args.get("timeout")
        timeout_suffix = theme.fg("muted", f" (timeout {timeout}s)") if timeout else ""
        return Text(theme.fg("toolTitle", theme.bold(f"$ {command}")) + timeout_suffix, 0, 0)

    pi.register_tool(
        ToolDefinition(
            name="bash",
            label="bash",
            description=registration_tools["bash"].description,
            parameters=registration_tools["bash"].parameters,
            execute=make_delegate("bash"),
            render_call=render_bash_call,
            render_result=render_minimal_result,
        )
    )

    # =========================================================================
    # Write Tool
    # =========================================================================
    def render_write_call(args, theme, _context):
        args = args or {}
        line_count = len(args["content"].split("\n")) if args.get("content") else 0
        line_info = theme.fg("muted", f" ({line_count} lines)") if line_count > 0 else ""
        return Text(f"{theme.fg('toolTitle', theme.bold('write'))} {path_display(args, theme)}{line_info}", 0, 0)

    def render_write_result(result, options, theme, _context):
        # Minimal mode: show nothing (the file was written)
        if not options.get("expanded"):
            return Text("", 0, 0)

        # Expanded mode: show the error if any
        text = _find_text_block(result)
        if text:
            return Text(f"\n{theme.fg('error', text)}", 0, 0)
        return Text("", 0, 0)

    pi.register_tool(
        ToolDefinition(
            name="write",
            label="write",
            description=registration_tools["write"].description,
            parameters=registration_tools["write"].parameters,
            execute=make_delegate("write"),
            render_call=render_write_call,
            render_result=render_write_result,
        )
    )

    # =========================================================================
    # Edit Tool
    # =========================================================================
    def render_edit_call(args, theme, _context):
        args = args or {}
        return Text(f"{theme.fg('toolTitle', theme.bold('edit'))} {path_display(args, theme)}", 0, 0)

    def render_edit_result(result, options, theme, _context):
        # Minimal mode: show nothing in collapsed state
        if not options.get("expanded"):
            return Text("", 0, 0)

        # Expanded mode: show diff or error
        text = _find_text_block(result)
        if not text:
            return Text("", 0, 0)

        # For errors, show the error message
        if "Error" in text or "error" in text:
            return Text(f"\n{theme.fg('error', text)}", 0, 0)

        # Otherwise show the text (would be nice to show the actual diff here)
        return Text(f"\n{theme.fg('toolOutput', text)}", 0, 0)

    pi.register_tool(
        ToolDefinition(
            name="edit",
            label="edit",
            description=registration_tools["edit"].description,
            parameters=registration_tools["edit"].parameters,
            execute=make_delegate("edit"),
            render_call=render_edit_call,
            render_result=render_edit_result,
        )
    )

    # =========================================================================
    # Find Tool
    # =========================================================================
    def render_find_call(args, theme, _context):
        args = args or {}
        pattern = args.get("pattern") or ""
        path = shorten_path(args.get("path") or ".")
        text = f"{theme.fg('toolTitle', theme.bold('find'))} {theme.fg('accent', pattern)}"
        text += theme.fg("toolOutput", f" in {path}")
        if args.get("limit") is not None:
            text += theme.fg("toolOutput", f" (limit {args['limit']})")
        return Text(text, 0, 0)

    pi.register_tool(
        ToolDefinition(
            name="find",
            label="find",
            description=registration_tools["find"].description,
            parameters=registration_tools["find"].parameters,
            execute=make_delegate("find"),
            render_call=render_find_call,
            render_result=make_count_result_renderer("files"),
        )
    )

    # =========================================================================
    # Grep Tool
    # =========================================================================
    def render_grep_call(args, theme, _context):
        args = args or {}
        pattern = args.get("pattern") or ""
        path = shorten_path(args.get("path") or ".")
        text = f"{theme.fg('toolTitle', theme.bold('grep'))} {theme.fg('accent', f'/{pattern}/')}"
        text += theme.fg("toolOutput", f" in {path}")
        if args.get("glob"):
            text += theme.fg("toolOutput", f" ({args['glob']})")
        if args.get("limit") is not None:
            text += theme.fg("toolOutput", f" limit {args['limit']}")
        return Text(text, 0, 0)

    pi.register_tool(
        ToolDefinition(
            name="grep",
            label="grep",
            description=registration_tools["grep"].description,
            parameters=registration_tools["grep"].parameters,
            execute=make_delegate("grep"),
            render_call=render_grep_call,
            render_result=make_count_result_renderer("matches"),
        )
    )

    # =========================================================================
    # Ls Tool
    # =========================================================================
    def render_ls_call(args, theme, _context):
        args = args or {}
        path = shorten_path(args.get("path") or ".")
        text = f"{theme.fg('toolTitle', theme.bold('ls'))} {theme.fg('accent', path)}"
        if args.get("limit") is not None:
            text += theme.fg("toolOutput", f" (limit {args['limit']})")
        return Text(text, 0, 0)

    pi.register_tool(
        ToolDefinition(
            name="ls",
            label="ls",
            description=registration_tools["ls"].description,
            parameters=registration_tools["ls"].parameters,
            execute=make_delegate("ls"),
            render_call=render_ls_call,
            render_result=make_count_result_renderer("entries"),
        )
    )
