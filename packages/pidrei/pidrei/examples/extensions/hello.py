"""Hello Tool

Minimal custom tool example.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/hello.py
"""

from pidrei.core.extensions.types import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent


def extension(pi):
    async def execute(_tool_call_id, params, _cancel, _on_update, _ctx):
        return AgentToolResult(
            content=[TextContent(text=f"Hello, {params['name']}!")],
            details={"greeted": params["name"]},
        )

    pi.register_tool(
        ToolDefinition(
            name="hello",
            label="Hello",
            description="A simple greeting tool",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name to greet"},
                },
                "required": ["name"],
            },
            execute=execute,
        )
    )
