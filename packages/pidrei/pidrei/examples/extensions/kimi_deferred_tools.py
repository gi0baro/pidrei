"""Minimal Kimi deferred-tool loading demo.

Only a `tool_search` tool is active at session start; the model must call it
to discover and activate the Calculator. Activation happens through
`pi.set_active_tools` inside the tool's execute — the runtime notices the
addition and exposes the new tool from the next turn.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/kimi_deferred_tools.py

Example prompt: Use the available tools to calculate 100 + 500. Do not
calculate it yourself.
"""

from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent


def calculate(_expr: str) -> str:
    return "42"


def extension(pi):
    async def execute_calculator(_tool_call_id, params, *_rest):
        return AgentToolResult(content=[TextContent(text=calculate(params["expr"]))], details={})

    pi.register_tool(
        ToolDefinition(
            name="Calculator",
            label="Calculator",
            description="Evaluate a simple arithmetic expression.",
            parameters={
                "type": "object",
                "properties": {"expr": {"type": "string", "description": "An expression such as 100 + 500"}},
                "required": ["expr"],
            },
            execute=execute_calculator,
        )
    )

    async def execute_tool_search(_tool_call_id, params, *_rest):
        if "calc" not in params["query"].lower():
            return AgentToolResult(
                content=[TextContent(text="The relevant tools do not exist.")],
                details={"matches": [], "added": []},
            )

        active = pi.get_active_tools()
        added = [] if "Calculator" in active else ["Calculator"]
        if added:
            pi.set_active_tools([*active, *added])

        return AgentToolResult(
            content=[TextContent(text="Success. Found 1 matching tool(s)")],
            details={"matches": ["Calculator"], "added": added},
        )

    pi.register_tool(
        ToolDefinition(
            name="tool_search",
            label="Tool Search",
            description="Find and activate tools for a capability.",
            prompt_snippet="Search for additional tools when the active tools cannot perform the task",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Capability to search for"}},
                "required": ["query"],
            },
            execute=execute_tool_search,
        )
    )

    async def on_session_start(_event, _ctx):
        pi.set_active_tools(["tool_search"])

    pi.on("session_start", on_session_start)
