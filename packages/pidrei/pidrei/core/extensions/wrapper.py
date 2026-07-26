"""Mirror of pi coding-agent src/core/extensions/wrapper.ts.

Tool wrappers for extension-registered tools. These only adapt tool execution
so extension tools receive the runner context; tool call/result interception
is handled by AgentSession via agent-core hooks.
"""

import dataclasses
from typing import TYPE_CHECKING

from pidrei_agent.types import AgentTool

from ..tools.tool_definition_wrapper import wrap_tool_definition
from .types import RegisteredTool


if TYPE_CHECKING:
    from .runner import ExtensionRunner


class _RegisteredToolAgentTool(AgentTool):
    """AgentTool for a RegisteredTool: executes with the runner context and
    reports tools activated by the execution via added_tool_names."""

    def __init__(self, registered_tool: RegisteredTool, runner: ExtensionRunner):
        tool = wrap_tool_definition(registered_tool.definition, runner.create_context)
        self.definition = tool.definition
        self.name = tool.name
        self.label = tool.label
        self.description = tool.description
        self.parameters = tool.parameters
        self.constrained_sampling = tool.constrained_sampling
        self.prepare_arguments = tool.prepare_arguments
        self.execution_mode = tool.execution_mode
        self.prompt_snippet = tool.prompt_snippet
        self.prompt_guidelines = tool.prompt_guidelines
        self._inner = tool
        self._runner = runner

    async def execute(self, tool_call_id, params, cancel=None, on_update=None, ctx=None):
        active_before = self._runner.get_active_tools()
        result = await self._inner.execute(tool_call_id, params, cancel, on_update, ctx)
        active_after = self._runner.get_active_tools()
        if not all(name in active_after for name in active_before):
            return result

        before_names = set(active_before)
        added_tool_names = [name for name in active_after if name not in before_names]
        if not added_tool_names:
            return result
        existing = list(getattr(result, "added_tool_names", None) or [])
        merged = list(dict.fromkeys([*existing, *added_tool_names]))
        return dataclasses.replace(result, added_tool_names=merged)


def wrap_registered_tool(registered_tool: RegisteredTool, runner: ExtensionRunner) -> AgentTool:
    """Wrap a RegisteredTool into an AgentTool using the runner's context."""
    return _RegisteredToolAgentTool(registered_tool, runner)


def wrap_registered_tools(registered_tools: list[RegisteredTool], runner: ExtensionRunner) -> list[AgentTool]:
    """Wrap all registered tools into AgentTools."""
    return [wrap_registered_tool(tool, runner) for tool in registered_tools]
