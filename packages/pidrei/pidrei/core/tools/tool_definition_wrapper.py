"""Mirror of pi coding-agent src/core/tools/tool-definition-wrapper.ts."""

from typing import Any

from pidrei_agent.types import AgentTool

from ..extensions.types import ToolDefinition


class WrappedDefinitionTool(AgentTool):
    """AgentTool bridging a ToolDefinition for the core runtime."""

    def __init__(self, definition: ToolDefinition, ctx_factory=None):
        self.definition = definition
        self.name = definition.name
        self.label = definition.label
        self.description = definition.description
        self.parameters = definition.parameters
        self.constrained_sampling = definition.constrained_sampling
        self.prepare_arguments = definition.prepare_arguments
        self.execution_mode = definition.execution_mode
        self.prompt_snippet = definition.prompt_snippet
        self.prompt_guidelines = definition.prompt_guidelines
        self._ctx_factory = ctx_factory

    async def execute(self, tool_call_id, params, cancel=None, on_update=None, ctx=None):
        if ctx is None and self._ctx_factory is not None:
            ctx = self._ctx_factory()
        return await self.definition.execute(tool_call_id, params, cancel, on_update, ctx)


def wrap_tool_definition(definition: ToolDefinition, ctx_factory=None) -> WrappedDefinitionTool:
    """Wrap a ToolDefinition into an AgentTool for the core runtime."""
    return WrappedDefinitionTool(definition, ctx_factory)


def wrap_tool_definitions(definitions: list[ToolDefinition], ctx_factory=None) -> list[WrappedDefinitionTool]:
    """Wrap multiple ToolDefinitions into AgentTools for the core runtime."""
    return [wrap_tool_definition(definition, ctx_factory) for definition in definitions]


def create_tool_definition_from_agent_tool(tool: AgentTool) -> ToolDefinition:
    """Synthesize a minimal ToolDefinition from an AgentTool.

    This keeps AgentSession's internal registry definition-first even when a
    caller provides plain AgentTool overrides without prompt metadata.
    """

    async def execute(tool_call_id: str, params: Any, cancel: Any, on_update: Any, _ctx: Any):
        return await tool.execute(tool_call_id, params, cancel, on_update)

    return ToolDefinition(
        name=tool.name,
        label=tool.label,
        description=tool.description,
        parameters=tool.parameters,
        constrained_sampling=getattr(tool, "constrained_sampling", None),
        prepare_arguments=getattr(tool, "prepare_arguments", None),
        execution_mode=getattr(tool, "execution_mode", None),
        execute=execute,
    )
