"""Mirror of pi coding-agent src/core/tools/renderers/index.ts.

Built-in tool renderers, without the tools themselves.

A presentation displays tool calls and results; it does not execute them and
does not need their parameter schemas. Importing this instead of
`core.tools` keeps the execution path out of a process that only renders.

pi's `powershell` renderer is dropped with the tool (POSIX-only; see
`core/tools/__init__.py`).
"""

from dataclasses import replace

from .bash import BASH_UPDATE_THROTTLE_S, create_shell_renderers
from .edit import edit_renderers
from .find import find_renderers
from .grep import grep_renderers
from .ls import ls_renderers
from .read import read_renderers
from .types import ToolRenderers
from .write import write_renderers


def create_all_tool_renderers() -> dict[str, ToolRenderers]:
    """Renderers for every built-in tool, keyed by tool name."""
    return {
        "read": read_renderers,
        "bash": create_shell_renderers("$"),
        "edit": edit_renderers,
        "write": write_renderers,
        "grep": grep_renderers,
        "find": find_renderers,
        "ls": ls_renderers,
    }


def with_built_in_renderers(tool_name: str, definition):
    """Merge built-in renderers into a tool definition that does not supply its own.

    `ToolExecutionComponent` used to do this lookup itself, which forced every
    presentation to import the tool implementations. Callers do it now, so a
    process that renders can import renderers alone.
    """
    built_in = create_all_tool_renderers().get(tool_name)
    if definition is None:
        return built_in
    if built_in is None:
        return definition
    return replace(
        definition,
        render_call=definition.render_call or built_in.render_call,
        render_result=definition.render_result or built_in.render_result,
    )


__all__ = [
    "BASH_UPDATE_THROTTLE_S",
    "ToolRenderers",
    "create_all_tool_renderers",
    "create_shell_renderers",
    "edit_renderers",
    "find_renderers",
    "grep_renderers",
    "ls_renderers",
    "read_renderers",
    "with_built_in_renderers",
    "write_renderers",
]
