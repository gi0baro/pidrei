"""Mirror of pi coding-agent src/core/tools/tool-definition-wrapper.ts.

The context factory builds a tool context per call (pi's
`createToolContext(toolCallId, signal)`), carrying the call's nested-call
scope (`nested_tool_calls.py`). A model-issued call owns its scope: the
wrapper opens the feed, and when `execute` returns it folds the nested calls
onto the result (`nested_calls`/`nested_usage`, which the agent loop puts on
the tool result message). A nested call runs through a copy bound to the
scope its caller handed down (`with_nested_scope`), so the calls it makes land
in the top-level record, and it never drains.
"""

import copy
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from pidrei_agent.types import AgentTool, AgentToolResult
from pidrei_ai.types import TextContent
from pidrei_ai.utils.cancel import CancelToken

from ..extensions.types import ToolDefinition
from ..nested_tool_calls import NestedCallFeed, NestedCallScope


# Creates the context for one tool call: (tool_call_id, cancel, scope) -> ctx.
type ToolContextFactory = Callable[[str, CancelToken | None, NestedCallScope], Any]


class WrappedDefinitionTool(AgentTool):
    """AgentTool bridging a ToolDefinition for the core runtime."""

    def __init__(self, definition: ToolDefinition, ctx_factory: ToolContextFactory | None = None):
        self.definition = definition
        self.name = definition.name
        self.label = definition.label
        self.description = definition.description
        self.parameters = definition.parameters
        self.output_schema = definition.output_schema
        self.constrained_sampling = definition.constrained_sampling
        self.prepare_arguments = definition.prepare_arguments
        self.execution_mode = definition.execution_mode
        self.prompt_snippet = definition.prompt_snippet
        self.prompt_guidelines = definition.prompt_guidelines
        self._ctx_factory = ctx_factory
        self._nested_scope: NestedCallScope | None = None

    def with_nested_scope(self, scope: NestedCallScope) -> WrappedDefinitionTool:
        """A copy whose calls run as nested calls under `scope`."""
        bound = copy.copy(self)
        bound._nested_scope = scope
        return bound

    async def execute(self, tool_call_id, params, cancel=None, on_update=None, ctx=None):
        if ctx is not None or self._ctx_factory is None:
            return await self.definition.execute(tool_call_id, params, cancel, on_update, ctx)
        if self._nested_scope is not None:
            ctx = self._ctx_factory(tool_call_id, cancel, self._nested_scope)
            return await self.definition.execute(tool_call_id, params, cancel, on_update, ctx)

        feed = NestedCallFeed()
        ctx = self._ctx_factory(tool_call_id, cancel, NestedCallScope(parent_id=tool_call_id, feed=feed))
        try:
            result = await self.definition.execute(tool_call_id, params, cancel, on_update, ctx)
        except Exception as error:
            summary = feed.drain()
            if summary.calls is None and summary.usage is None:
                raise
            # pi records the calls on the error result the loop makes of the
            # raised error; that result is built here, so the record rides on it.
            return AgentToolResult(
                content=[TextContent(text=str(error))],
                details={},
                is_error=True,
                nested_calls=summary.calls,
                nested_usage=summary.usage,
            )
        summary = feed.drain()
        if summary.calls is None and summary.usage is None:
            return result
        return replace(result, nested_calls=summary.calls, nested_usage=summary.usage)


def wrap_tool_definition(
    definition: ToolDefinition, ctx_factory: ToolContextFactory | None = None
) -> WrappedDefinitionTool:
    """Wrap a ToolDefinition into an AgentTool for the core runtime."""
    return WrappedDefinitionTool(definition, ctx_factory)


def wrap_tool_definitions(
    definitions: list[ToolDefinition], ctx_factory: ToolContextFactory | None = None
) -> list[WrappedDefinitionTool]:
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
        output_schema=tool.output_schema,
        constrained_sampling=getattr(tool, "constrained_sampling", None),
        prepare_arguments=getattr(tool, "prepare_arguments", None),
        execution_mode=getattr(tool, "execution_mode", None),
        execute=execute,
    )
