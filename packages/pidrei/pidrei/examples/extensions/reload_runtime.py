"""Reload Runtime

Demonstrates ctx.reload() from the extension command context and an
LLM-callable tool that queues a follow-up command to trigger reload.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/reload_runtime.py
"""

from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent


def extension(pi):
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
    # reload(); only command handlers do. Instead, queue a follow-up user
    # command that executes the command above. expandPromptTemplates makes the
    # queued text dispatch as a command instead of going to the model as text.
    async def reload_runtime(_tool_call_id, _params, _cancel=None, _on_update=None, _ctx=None):
        pi.send_user_message("/reload-runtime", {"deliverAs": "followUp", "expandPromptTemplates": True})
        return AgentToolResult(
            content=[TextContent(text="Queued /reload-runtime as a follow-up command.")],
            details={},
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
