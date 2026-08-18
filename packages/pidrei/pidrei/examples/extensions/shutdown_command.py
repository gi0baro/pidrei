"""Shutdown Command

Adds a /quit command that allows extensions to trigger clean shutdown.
Demonstrates how extensions can use ctx.shutdown() to exit pidrei cleanly.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/shutdown_command.py
"""

from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent


def extension(pi):
    # Register a /quit command that cleanly exits pidrei.
    async def handle_quit(_args, ctx) -> None:
        ctx.shutdown()

    pi.register_command("quit", handler=handle_quit, description="Exit pidrei cleanly")

    # You can also create a tool that shuts down after completing work.
    async def finish_and_exit(_tool_call_id, _params, _cancel=None, _on_update=None, ctx=None):
        # Do any final work here...
        # Request graceful shutdown (deferred until the agent is idle).
        ctx.shutdown()

        # This return is sent to the LLM before shutdown occurs.
        return AgentToolResult(
            content=[TextContent(text="Shutdown requested. Exiting after this response.")],
            details={},
        )

    pi.register_tool(
        ToolDefinition(
            name="finish_and_exit",
            label="Finish and Exit",
            description="Complete a task and exit pidrei",
            parameters={"type": "object", "properties": {}},
            execute=finish_and_exit,
        )
    )

    # You could also create a more complex tool with parameters.
    async def deploy_and_exit(_tool_call_id, params, _cancel=None, on_update=None, ctx=None):
        if on_update is not None:
            on_update(
                AgentToolResult(content=[TextContent(text=f"Deploying to {params['environment']}...")], details={})
            )

        # Example deployment logic
        # result = await pi.exec("npm", ["run", "deploy", params["environment"]])

        # On success, request graceful shutdown.
        if on_update is not None:
            on_update(AgentToolResult(content=[TextContent(text="Deployment complete, exiting...")], details={}))
        ctx.shutdown()

        return AgentToolResult(
            content=[TextContent(text="Done! Shutdown requested.")],
            details={"environment": params["environment"]},
        )

    pi.register_tool(
        ToolDefinition(
            name="deploy_and_exit",
            label="Deploy and Exit",
            description="Deploy the application and exit pidrei",
            parameters={
                "type": "object",
                "properties": {
                    "environment": {
                        "type": "string",
                        "description": "Target environment (e.g., production, staging)",
                    },
                },
                "required": ["environment"],
            },
            execute=deploy_and_exit,
        )
    )
