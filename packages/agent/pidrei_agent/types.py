"""Agent-level types (port of pi `agent/src/types.ts`).

This module currently carries the tool-facing subset needed by the harness
tools; the agent-loop configuration types land together with the loop port.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pidrei_ai.types import ImageContent, TextContent, Usage


# How tool calls from a single assistant message are executed:
# - "sequential": each tool call is prepared, executed, and finalized before the next one starts.
# - "parallel": tool calls are prepared sequentially, then allowed tools execute concurrently.
type ToolExecutionMode = Literal["sequential", "parallel"]

# How many queued user messages are injected when the agent loop reaches a queue drain point.
type QueueMode = Literal["all", "one-at-a-time"]

# Thinking/reasoning level for models that support it. Unlike pi-ai's
# `ThinkingLevel`, the agent-level union includes "off" (pi redeclares it the
# same way in agent/src/types.ts).
type ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]


@dataclass(slots=True)
class AgentToolResult[TDetails]:
    """Final or partial result produced by a tool."""

    # Text or image content returned to the model.
    content: list[TextContent | ImageContent] = field(default_factory=list)
    # Arbitrary structured details for logs or UI rendering.
    details: TDetails | None = None
    # Usage from the final tool execution itself, if available. Not used for
    # main LLM context accounting.
    usage: Usage | None = None
    # Names of tools introduced by this result and available from this
    # transcript point onward.
    added_tool_names: list[str] | None = None
    # Hint that the agent should stop after the current tool batch. Early
    # termination only happens when every finalized tool result in the batch
    # sets this to True.
    terminate: bool | None = None


# Callback used by tools to stream partial execution updates. The callback is
# scoped to the current `execute()` invocation; calls made after the tool
# settles are ignored.
type AgentToolUpdateCallback[TDetails] = Callable[[AgentToolResult[TDetails]], None]

# Compatibility shim for raw tool-call arguments before schema validation.
type PrepareArguments = Callable[[Any], Any]
