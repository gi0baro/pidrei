"""Mirror of pi coding-agent src/core/extensions/wrapper.ts.

Tool wrappers for extension-registered tools. These only adapt tool execution
so extension tools receive the runner context; tool call/result interception
is handled by AgentSession via agent-core hooks.
"""

from typing import TYPE_CHECKING

from pidrei_agent.types import AgentTool

from ..tools.tool_definition_wrapper import wrap_tool_definition
from .types import RegisteredTool


if TYPE_CHECKING:
    from .runner import ExtensionRunner


def wrap_registered_tool(registered_tool: RegisteredTool, runner: ExtensionRunner) -> AgentTool:
    """Wrap a RegisteredTool into an AgentTool.

    Uses the runner's create_context() for consistent context across tools and event handlers.
    """
    return wrap_tool_definition(registered_tool.definition, runner.create_context)


def wrap_registered_tools(registered_tools: list[RegisteredTool], runner: ExtensionRunner) -> list[AgentTool]:
    """Wrap all registered tools into AgentTools."""
    return [wrap_registered_tool(tool, runner) for tool in registered_tools]
