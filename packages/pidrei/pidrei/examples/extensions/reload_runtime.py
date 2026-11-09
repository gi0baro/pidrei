"""Reload Runtime

Demonstrates ctx.reload() from the extension command context and an
LLM-callable tool that requests a reload once the agent has settled.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/reload_runtime.py
"""

from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent


async def extension(pi):
    # Command entrypoint for reload.
    # Treat reload as terminal for this handler.
    async def handle_reload(_args, ctx) -> None:
        await ctx.reload()

    pi.register_command(
        "reload-runtime",
        handler=handle_reload,
        description="Reload extensions, skills, prompts, themes, and context files",
    )

    # LLM-callable tool. Tools get the plain extension context, which has no
    # reload(), and pidrei cannot reload while the agent is running. The tool
    # records the request and ends the run; the agent_settled handler then
    # runs the command above.
    reload_requested = False

    async def reload_runtime(_tool_call_id, _params, _cancel=None, _on_update=None, _ctx=None):
        nonlocal reload_requested
        reload_requested = True
        return AgentToolResult(
            content=[TextContent(text="Reload requested. pidrei reloads after this response.")],
            details={},
            terminate=True,
        )

    pi.register_tool(
        ToolDefinition(
            name="reload_runtime",
            label="Reload Runtime",
            description="Reload extensions, skills, prompts, themes, and context files",
            parameters={"type": "object", "properties": {}},
            execute=reload_runtime,
        )
    )

    async def on_agent_settled(_event, _ctx) -> None:
        nonlocal reload_requested
        if not reload_requested:
            return
        reload_requested = False
        # Sent from agent_settled, the command runs after every settled handler
        # finishes, while the agent is idle. expandPromptTemplates dispatches it
        # as a command, not as model input.
        pi.send_user_message("/reload-runtime", {"expandPromptTemplates": True})

    pi.on("agent_settled", on_agent_settled)
