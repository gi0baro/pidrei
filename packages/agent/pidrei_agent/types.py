"""Agent-level types (port of pi `agent/src/types.ts`)."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pidrei_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    ImageContent,
    Message,
    Model,
    SimpleStreamOptions,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
)
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.event_stream import AssistantMessageEventStream


# Stream function used by the agent loop. `Models.stream_simple` satisfies this
# shape. Contract: must not raise for request/model/runtime failures — failures
# must be encoded in the returned stream via protocol events and a final
# AssistantMessage with stop_reason "error"/"aborted" and error_message.
type StreamFn = Callable[
    [Model, Context, SimpleStreamOptions | None],
    AssistantMessageEventStream | Awaitable[AssistantMessageEventStream],
]

# How tool calls from a single assistant message are executed:
# - "sequential": each tool call is prepared, executed, and finalized before the next one starts.
# - "parallel": tool calls are prepared sequentially, then allowed tools execute concurrently.
#   `tool_execution_end` is emitted in tool completion order after each tool is finalized,
#   while tool-result message artifacts are emitted later in assistant source order.
type ToolExecutionMode = Literal["sequential", "parallel"]

# How many queued user messages are injected when the agent loop reaches a queue drain point.
type QueueMode = Literal["all", "one-at-a-time"]

# Thinking/reasoning level for models that support it. Unlike pi-ai's
# `ThinkingLevel`, the agent-level union includes "off" (pi redeclares it the
# same way in agent/src/types.ts).
type ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]

# Union of LLM messages + app-defined custom messages. pi models the custom arm
# with declaration merging; in Python any object with a `role` attribute (and
# `timestamp`) can participate — `convert_to_llm` owns turning custom messages
# into LLM messages or filtering them out.
type AgentMessage = Message | Any

# A single tool call content block emitted by an assistant message.
AgentToolCall = ToolCall


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


class AgentTool[TDetails]:
    """Tool definition used by the agent runtime.

    pi builds tools as object literals typed `AgentTool`; the Python port uses
    a base class — concrete tools subclass it (or applications construct
    lightweight instances) and implement `execute`, which must raise on
    failure instead of encoding errors in `content`.
    """

    name: str
    label: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (pi: TypeBox TSchema)
    # Per-tool execution mode override; None applies the loop default.
    execution_mode: ToolExecutionMode | None = None
    prepare_arguments: PrepareArguments | None = None

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancel: CancelToken | None,
        on_update: AgentToolUpdateCallback[TDetails] | None,
    ) -> AgentToolResult[TDetails]:
        raise NotImplementedError


@dataclass(slots=True)
class AgentContext:
    """Context snapshot passed into the low-level agent loop."""

    # System prompt included with the request.
    system_prompt: str
    # Transcript visible to the model.
    messages: list[AgentMessage]
    # Tools available for this run.
    tools: list[AgentTool] | None = None


@dataclass(slots=True)
class BeforeToolCallResult:
    """Result returned from `before_tool_call`.

    Returning `block=True` prevents the tool from executing; the loop emits an
    error tool result instead, with `reason` as its text (or a default blocked
    message).
    """

    block: bool | None = None
    reason: str | None = None


@dataclass(slots=True)
class AfterToolCallResult:
    """Partial override returned from `after_tool_call`.

    Merge semantics are field-by-field: a provided (non-None) field replaces
    the corresponding tool result value in full; None keeps the original.
    There is no deep merge.
    """

    content: list[TextContent | ImageContent] | None = None
    details: Any = None
    is_error: bool | None = None
    # Usage from the final tool execution itself, if available. Not used for
    # main LLM context accounting.
    usage: Usage | None = None
    # Hint that the agent should stop after the current tool batch. Early
    # termination only happens when every finalized tool result in the batch
    # sets this to True.
    terminate: bool | None = None


@dataclass(slots=True)
class BeforeToolCallContext:
    """Context passed to `before_tool_call`."""

    # The assistant message that requested the tool call.
    assistant_message: AssistantMessage
    # The raw tool call block from `assistant_message.content`.
    tool_call: AgentToolCall
    # Validated tool arguments for the target tool schema.
    args: Any
    # Current agent context at the time the tool call is prepared.
    context: AgentContext


@dataclass(slots=True)
class AfterToolCallContext:
    """Context passed to `after_tool_call`."""

    # The assistant message that requested the tool call.
    assistant_message: AssistantMessage
    # The raw tool call block from `assistant_message.content`.
    tool_call: AgentToolCall
    # Validated tool arguments for the target tool schema.
    args: Any
    # The executed tool result before any `after_tool_call` overrides are applied.
    result: AgentToolResult[Any]
    # Whether the executed tool result is currently treated as an error.
    is_error: bool
    # Current agent context at the time the tool call is finalized.
    context: AgentContext


@dataclass(slots=True)
class ShouldStopAfterTurnContext:
    """Context passed to `should_stop_after_turn` (and `prepare_next_turn`)."""

    # The assistant message that completed the turn.
    message: AssistantMessage
    # Tool result messages passed to the preceding `turn_end` event.
    tool_results: list[ToolResultMessage]
    # Current agent context after the turn's assistant message and tool results
    # have been appended.
    context: AgentContext
    # Messages that this loop invocation will return if it exits at this point.
    new_messages: list[AgentMessage]


PrepareNextTurnContext = ShouldStopAfterTurnContext


@dataclass(slots=True)
class AgentLoopTurnUpdate:
    """Replacement runtime state used by the agent loop before the next provider request."""

    # Context for the next provider request.
    context: AgentContext | None = None
    # Model for the next provider request.
    model: Model | None = None
    # Thinking level for the next provider request (None keeps the current one).
    thinking_level: ThinkingLevel | None = None


@dataclass(slots=True, kw_only=True)
class AgentLoopConfig(SimpleStreamOptions):
    """Agent loop configuration (extends the provider stream options like pi).

    All hook contracts mirror pi: hooks must not raise — raising interrupts the
    low-level loop without producing a normal event sequence. Callbacks may be
    sync or async; the loop awaits coroutine results.
    """

    model: Model

    # Converts AgentMessage[] to LLM-compatible Message[] before each LLM call;
    # messages that cannot be converted must be filtered out.
    convert_to_llm: Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]]

    # Optional transform applied to the context before `convert_to_llm`
    # (context-window management, external injection, ...).
    transform_context: Callable[[list[AgentMessage], CancelToken | None], Awaitable[list[AgentMessage]]] | None = None

    # Resolves an API key dynamically for each LLM call (short-lived tokens).
    get_api_key: Callable[[str], str | None | Awaitable[str | None]] | None = None

    # Called after each turn fully completes; returning True exits the loop
    # before polling steering/follow-up queues.
    should_stop_after_turn: Callable[[ShouldStopAfterTurnContext], bool | Awaitable[bool]] | None = None

    # Called after `turn_end`, before the loop decides whether another provider
    # request should start. Return replacement state or None to keep current.
    prepare_next_turn: (
        Callable[[PrepareNextTurnContext], AgentLoopTurnUpdate | None | Awaitable[AgentLoopTurnUpdate | None]] | None
    ) = None

    # Returns steering messages to inject into the conversation mid-run.
    get_steering_messages: Callable[[], list[AgentMessage] | Awaitable[list[AgentMessage]]] | None = None

    # Returns follow-up messages to process after the agent would otherwise stop.
    get_follow_up_messages: Callable[[], list[AgentMessage] | Awaitable[list[AgentMessage]]] | None = None

    # Tool execution mode. Default: "parallel".
    tool_execution: ToolExecutionMode | None = None

    # Called before a tool executes, after argument validation. Return
    # `BeforeToolCallResult(block=True)` to prevent execution.
    before_tool_call: (
        Callable[
            [BeforeToolCallContext, CancelToken | None],
            BeforeToolCallResult | None | Awaitable[BeforeToolCallResult | None],
        ]
        | None
    ) = None

    # Called after a tool finishes executing, before `tool_execution_end` and
    # tool-result message events are emitted.
    after_tool_call: (
        Callable[
            [AfterToolCallContext, CancelToken | None],
            AfterToolCallResult | None | Awaitable[AfterToolCallResult | None],
        ]
        | None
    ) = None


# --- events -------------------------------------------------------------------
# Events emitted by the agent for UI updates. `agent_end` is the last event
# emitted for a run.


@dataclass(slots=True)
class AgentStartEvent:
    type: Literal["agent_start"] = "agent_start"


@dataclass(slots=True)
class AgentEndEvent:
    messages: list[AgentMessage]
    type: Literal["agent_end"] = "agent_end"


@dataclass(slots=True)
class TurnStartEvent:
    type: Literal["turn_start"] = "turn_start"


@dataclass(slots=True)
class TurnEndEvent:
    # A turn is one assistant response + any tool calls/results.
    message: AgentMessage
    tool_results: list[ToolResultMessage]
    type: Literal["turn_end"] = "turn_end"


@dataclass(slots=True)
class MessageStartEvent:
    # Emitted for user, assistant, and toolResult messages.
    message: AgentMessage
    type: Literal["message_start"] = "message_start"


@dataclass(slots=True)
class MessageUpdateEvent:
    # Only emitted for assistant messages during streaming.
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent
    type: Literal["message_update"] = "message_update"


@dataclass(slots=True)
class MessageEndEvent:
    message: AgentMessage
    type: Literal["message_end"] = "message_end"


@dataclass(slots=True)
class ToolExecutionStartEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    type: Literal["tool_execution_start"] = "tool_execution_start"


@dataclass(slots=True)
class ToolExecutionUpdateEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any
    type: Literal["tool_execution_update"] = "tool_execution_update"


@dataclass(slots=True)
class ToolExecutionEndEvent:
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool
    type: Literal["tool_execution_end"] = "tool_execution_end"


type AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
)
