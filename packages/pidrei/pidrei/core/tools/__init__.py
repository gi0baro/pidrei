"""Mirror of pi coding-agent src/core/tools/index.ts.

pi's `powershell` tool (upstream 80e62761) is dropped surface: it is
Windows-only by construction — `getPowerShellConfig` throws off win32 — and
pidrei is POSIX-only, so `powershell` is absent from ALL_TOOL_NAMES, the
factory tables and the SDK exports. The refactor that commit made to share one
implementation between the two shell tools *is* ported: see `ShellToolConfig`
and `create_shell_tool_definition` in `bash.py`.
"""

from ..extensions.types import ToolDefinition
from .bash import (
    BASH_TOOL_CONFIG,
    BashSpawnContext,
    BashToolDetails,
    LocalShellOperations,
    ShellToolConfig,
    create_bash_tool,
    create_bash_tool_definition,
    create_local_bash_operations,
    create_shell_tool_definition,
)
from .edit import EditToolDetails, create_edit_tool, create_edit_tool_definition
from .file_mutation_queue import with_file_mutation_queue
from .find import FindToolDetails, create_find_tool, create_find_tool_definition
from .grep import GrepToolDetails, create_grep_tool, create_grep_tool_definition
from .ls import LsToolDetails, create_ls_tool, create_ls_tool_definition
from .read import ReadToolDetails, create_read_tool, create_read_tool_definition
from .tool_definition_wrapper import (
    WrappedDefinitionTool,
    create_tool_definition_from_agent_tool,
    wrap_tool_definition,
    wrap_tool_definitions,
)
from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_head,
    truncate_line,
    truncate_tail,
)
from .write import create_write_tool, create_write_tool_definition


ALL_TOOL_NAMES: set[str] = {"read", "bash", "edit", "write", "grep", "find", "ls"}

_DEFINITION_FACTORIES = {
    "read": create_read_tool_definition,
    "bash": create_bash_tool_definition,
    "edit": create_edit_tool_definition,
    "write": create_write_tool_definition,
    "grep": create_grep_tool_definition,
    "find": create_find_tool_definition,
    "ls": create_ls_tool_definition,
}

_TOOL_FACTORIES = {
    "read": create_read_tool,
    "bash": create_bash_tool,
    "edit": create_edit_tool,
    "write": create_write_tool,
    "grep": create_grep_tool,
    "find": create_find_tool,
    "ls": create_ls_tool,
}


def create_tool_definition(tool_name: str, cwd: str, options: dict | None = None) -> ToolDefinition:
    factory = _DEFINITION_FACTORIES.get(tool_name)
    if factory is None:
        raise Exception(f"Unknown tool name: {tool_name}")
    return factory(cwd, **((options or {}).get(tool_name) or {}))


def create_tool(tool_name: str, cwd: str, options: dict | None = None) -> WrappedDefinitionTool:
    factory = _TOOL_FACTORIES.get(tool_name)
    if factory is None:
        raise Exception(f"Unknown tool name: {tool_name}")
    return factory(cwd, **((options or {}).get(tool_name) or {}))


def create_coding_tool_definitions(cwd: str, options: dict | None = None) -> list[ToolDefinition]:
    return [create_tool_definition(name, cwd, options) for name in ("read", "bash", "edit", "write")]


def create_read_only_tool_definitions(cwd: str, options: dict | None = None) -> list[ToolDefinition]:
    return [create_tool_definition(name, cwd, options) for name in ("read", "grep", "find", "ls")]


def create_all_tool_definitions(cwd: str, options: dict | None = None) -> dict[str, ToolDefinition]:
    return {name: create_tool_definition(name, cwd, options) for name in _DEFINITION_FACTORIES}


def create_coding_tools(cwd: str, options: dict | None = None) -> list[WrappedDefinitionTool]:
    return [create_tool(name, cwd, options) for name in ("read", "bash", "edit", "write")]


def create_read_only_tools(cwd: str, options: dict | None = None) -> list[WrappedDefinitionTool]:
    return [create_tool(name, cwd, options) for name in ("read", "grep", "find", "ls")]


def create_all_tools(cwd: str, options: dict | None = None) -> dict[str, WrappedDefinitionTool]:
    return {name: create_tool(name, cwd, options) for name in _TOOL_FACTORIES}


__all__ = [
    "ALL_TOOL_NAMES",
    "BASH_TOOL_CONFIG",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "BashSpawnContext",
    "BashToolDetails",
    "EditToolDetails",
    "FindToolDetails",
    "GrepToolDetails",
    "LocalShellOperations",
    "LsToolDetails",
    "ReadToolDetails",
    "ShellToolConfig",
    "ToolDefinition",
    "TruncationResult",
    "WrappedDefinitionTool",
    "create_all_tool_definitions",
    "create_all_tools",
    "create_bash_tool",
    "create_bash_tool_definition",
    "create_coding_tool_definitions",
    "create_coding_tools",
    "create_edit_tool",
    "create_edit_tool_definition",
    "create_find_tool",
    "create_find_tool_definition",
    "create_grep_tool",
    "create_grep_tool_definition",
    "create_local_bash_operations",
    "create_ls_tool",
    "create_ls_tool_definition",
    "create_read_only_tool_definitions",
    "create_read_only_tools",
    "create_read_tool",
    "create_read_tool_definition",
    "create_shell_tool_definition",
    "create_tool",
    "create_tool_definition",
    "create_tool_definition_from_agent_tool",
    "create_write_tool",
    "create_write_tool_definition",
    "format_size",
    "truncate_head",
    "truncate_line",
    "truncate_tail",
    "with_file_mutation_queue",
    "wrap_tool_definition",
    "wrap_tool_definitions",
]
