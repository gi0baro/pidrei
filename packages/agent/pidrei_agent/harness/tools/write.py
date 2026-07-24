"""Write tool (port of pi `harness/tools/write.ts`)."""

from typing import Any

from pidrei_ai.types import TextContent
from pidrei_ai.utils.cancel import CancelToken

from ...types import AgentToolResult, AgentToolUpdateCallback
from ..types import AgentHarnessTool, get_or_throw
from .file_mutation_queue import with_file_mutation_queue
from .path_utils import resolve_tool_path
from .tool_context import ExecutionToolContext


_WRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
        "content": {"type": "string", "description": "Content to write to the file"},
    },
    "required": ["path", "content"],
}


def _js_string_length(text: str) -> int:
    """JS `String.length` (UTF-16 code units) — pi reports this as the byte count."""
    return sum(2 if ord(char) > 0xFFFF else 1 for char in text)


class WriteTool(AgentHarnessTool[ExecutionToolContext, None]):
    name = "write"
    label = "write"
    description = (
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
        "Automatically creates parent directories."
    )
    parameters = _WRITE_SCHEMA

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        cancel: CancelToken | None,
        on_update: AgentToolUpdateCallback[None] | None,
        context: ExecutionToolContext,
    ) -> AgentToolResult[None]:
        path: str = params["path"]
        content: str = params["content"]
        env = context.env
        absolute_path = await resolve_tool_path(env, path, cancel)

        async def mutation() -> AgentToolResult[None]:
            if cancel is not None and cancel.cancelled:
                raise Exception("Operation aborted")
            get_or_throw(await env.write_file(absolute_path, content, cancel))
            if cancel is not None and cancel.cancelled:
                raise Exception("Operation aborted")
            return AgentToolResult(
                content=[TextContent(text=f"Successfully wrote {_js_string_length(content)} bytes to {path}")],
                details=None,
            )

        return await with_file_mutation_queue(env, absolute_path, mutation)


def create_write_tool() -> WriteTool:
    return WriteTool()
