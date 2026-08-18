"""Dynamic Tools Extension

Demonstrates registering tools after session initialization.

- Registers one tool during session_start
- Registers additional tools at runtime via /add-echo-tool <name>

Start pidrei with this extension:
    pidrei -e ./examples/extensions/dynamic_tools.py
"""

import re

from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent


ECHO_PARAMS = {
    "type": "object",
    "properties": {"message": {"type": "string", "description": "Message to echo"}},
    "required": ["message"],
}

TOOL_NAME_RE = re.compile(r"^[a-z0-9_]+$")


def normalize_tool_name(input_text: str) -> str | None:
    trimmed = input_text.strip().lower()
    if not trimmed or not TOOL_NAME_RE.match(trimmed):
        return None
    return trimmed


def extension(pi):
    registered_tool_names: set[str] = set()

    def register_echo_tool(name: str, label: str, prefix: str) -> bool:
        if name in registered_tool_names:
            return False

        registered_tool_names.add(name)

        async def execute(_tool_call_id, params, *_rest):
            return AgentToolResult(
                content=[TextContent(text=f"{prefix}{params['message']}")],
                details={"tool": name, "prefix": prefix},
            )

        pi.register_tool(
            ToolDefinition(
                name=name,
                label=label,
                description=f"Echo a message with prefix: {prefix}",
                prompt_snippet=f"Echo back user-provided text with {prefix.strip()} prefix",
                prompt_guidelines=["Use echo_session when the user asks for exact echo output."],
                parameters=ECHO_PARAMS,
                execute=execute,
            )
        )
        return True

    async def on_session_start(_event, ctx):
        register_echo_tool("echo_session", "Echo Session", "[session] ")
        ctx.ui.notify("Registered dynamic tool: echo_session", "info")

    pi.on("session_start", on_session_start)

    async def add_echo_tool(args: str, ctx) -> None:
        tool_name = normalize_tool_name(args or "")
        if tool_name is None:
            ctx.ui.notify("Usage: /add-echo-tool <tool_name> (lowercase, numbers, underscores)", "warning")
            return

        if not register_echo_tool(tool_name, f"Echo {tool_name}", f"[{tool_name}] "):
            ctx.ui.notify(f"Tool already registered: {tool_name}", "warning")
            return

        ctx.ui.notify(f"Registered dynamic tool: {tool_name}", "info")

    pi.register_command(
        "add-echo-tool",
        description="Register a new echo tool dynamically: /add-echo-tool <tool_name>",
        handler=add_echo_tool,
    )
