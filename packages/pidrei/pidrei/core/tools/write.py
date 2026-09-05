"""Mirror of pi coding-agent src/core/tools/write.ts."""

import os
from typing import Any

from tonio.colored import fs

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent

from ..experimental import get_experimental_tool_sampling
from ..extensions.types import ToolDefinition
from .file_mutation_queue import resolve_mutation_queue_key, with_file_mutation_queue
from .path_utils import resolve_to_cwd
from .renderers.write import write_renderers
from .tool_definition_wrapper import WrappedDefinitionTool, wrap_tool_definition


WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
        "content": {"type": "string", "description": "Content to write to the file"},
    },
    "required": ["path", "content"],
}

WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION: dict[str, Any] = {
    "snippet": "Create or overwrite files",
    "guidelines": ("Use write only for new files or complete rewrites.",),
}


class LocalWriteOperations:
    async def write_file(self, absolute_path: str, content: str) -> None:
        await fs.Path(absolute_path).write_text(content, encoding="utf-8", newline="")

    async def mkdir(self, dir: str) -> None:
        await fs.Path(dir).mkdir(parents=True, exist_ok=True)


def create_write_tool_definition(cwd: str, *, operations: Any = None) -> ToolDefinition:
    ops = operations if operations is not None else LocalWriteOperations()

    async def execute(_tool_call_id, params, cancel=None, _on_update=None, ctx=None):
        path = params["path"]
        content = params["content"]
        absolute_path = resolve_to_cwd(path, (ctx.cwd if ctx is not None else None) or cwd)
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
                content=[TextContent(text=f"Successfully wrote to {path}")],
                details=None,
            )

        queue_key = await resolve_mutation_queue_key(absolute_path)
        return await with_file_mutation_queue(absolute_path, run, queue_key=queue_key)

    return ToolDefinition(
        name="write",
        label="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        prompt_snippet=WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION["snippet"],
        prompt_guidelines=list(WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION["guidelines"]),
        parameters=WRITE_SCHEMA,
        constrained_sampling=get_experimental_tool_sampling(),
        execute=execute,
        render_call=write_renderers.render_call,
        render_result=write_renderers.render_result,
    )


def create_write_tool(cwd: str, **options) -> WrappedDefinitionTool:
    return wrap_tool_definition(create_write_tool_definition(cwd, **options))
