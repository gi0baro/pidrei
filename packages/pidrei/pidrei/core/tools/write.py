"""Mirror of pi coding-agent src/core/tools/write.ts (execute path; renderers Phase 4)."""

import os
from typing import Any

from tonio.colored import fs

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent

from ..extensions.types import ToolDefinition
from .file_mutation_queue import with_file_mutation_queue
from .path_utils import resolve_to_cwd
from .tool_definition_wrapper import WrappedDefinitionTool, wrap_tool_definition


WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
        "content": {"type": "string", "description": "Content to write to the file"},
    },
    "required": ["path", "content"],
}


class LocalWriteOperations:
    async def write_file(self, absolute_path: str, content: str) -> None:
        await fs.Path(absolute_path).write_text(content, encoding="utf-8", newline="")

    async def mkdir(self, dir: str) -> None:
        await fs.Path(dir).mkdir(parents=True, exist_ok=True)


def _js_string_length(text: str) -> int:
    """JS String.length (UTF-16 code units) for the byte-count message parity."""
    return sum(2 if ord(char) > 0xFFFF else 1 for char in text)


def create_write_tool_definition(cwd: str, *, operations: Any = None) -> ToolDefinition:
    ops = operations if operations is not None else LocalWriteOperations()

    async def execute(_tool_call_id, params, cancel=None, _on_update=None, _ctx=None):
        path = params["path"]
        content = params["content"]
        absolute_path = resolve_to_cwd(path, cwd)
        dir = os.path.dirname(absolute_path)

        async def run():
            # Do not release the mutation queue while an in-flight filesystem
            # operation may still finish: check cancellation after each await.
            def throw_if_aborted() -> None:
                if cancel is not None and cancel.cancelled:
                    raise Exception("Operation aborted")

            throw_if_aborted()
            # Create parent directories if needed.
            await ops.mkdir(dir)
            throw_if_aborted()

            # Write the file contents.
            await ops.write_file(absolute_path, content)
            throw_if_aborted()

            return AgentToolResult(
                content=[TextContent(text=f"Successfully wrote {_js_string_length(content)} bytes to {path}")],
                details=None,
            )

        return await with_file_mutation_queue(absolute_path, run)

    return ToolDefinition(
        name="write",
        label="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        prompt_snippet="Create or overwrite files",
        prompt_guidelines=["Use write only for new files or complete rewrites."],
        parameters=WRITE_SCHEMA,
        execute=execute,
    )


def create_write_tool(cwd: str, **options) -> WrappedDefinitionTool:
    return wrap_tool_definition(create_write_tool_definition(cwd, **options))
