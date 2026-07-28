"""Agent (port of pi `agent/src/agent.ts`).

Stateful wrapper around the low-level agent loop: owns the current transcript,
emits lifecycle events, executes tools, and exposes queueing APIs for steering
and follow-up messages.

Runtime mapping notes:
- `AbortController`/`AbortSignal` become a per-run `CancelToken`; `abort()`
  cancels it and `Agent.signal` exposes it.
- The run promise becomes a tonio `Event` awaited by `wait_for_idle()`.
- Listener registration order is preserved with a list (JS `Set` iterates in
  insertion order; Python's doesn't).
- pi's `continue()` is `continue_()` (`continue` is a Python keyword).
"""

import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio

from pidrei_ai.types import (
    AssistantMessage,
    ImageContent,
    Message,
    Model,
    ModelCost,
    TextContent,
    ThinkingBudgets,
    Transport,
    Usage,
    UserMessage,
)
from pidrei_ai.utils.cancel import CancelToken

from .agent_loop import run_agent_loop, run_agent_loop_continue
from .stream_fn import get_default_stream_fn
from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentTool,
    BeforeToolCallContext,
    BeforeToolCallResult,
    MessageEndEvent,
    MessageStartEvent,
    PrepareNextTurnContext,
    QueueMode,
    StreamFn,
    ThinkingLevel,
    ToolExecutionMode,
    TurnEndEvent,
)


async def default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    return [m for m in messages if getattr(m, "role", None) in ("user", "assistant", "toolResult")]


def _default_model() -> Model:
    return Model(
        id="unknown",
        name="unknown",
        api="unknown",  # type: ignore[arg-type]
        provider="unknown",
        base_url="",
        reasoning=False,
        input=[],
        cost=ModelCost(),
        context_window=0,
        max_tokens=0,
    )


class AgentState:
    """Public agent state (pi: `AgentState` + `createMutableAgentState`).

    Assigning `tools` or `messages` copies the provided top-level list; reading
    them returns the internal list (appending to it is visible, as in pi).
    """

    def __init__(
        self,
        system_prompt: str = "",
        model: Model | None = None,
        thinking_level: ThinkingLevel = "off",
        tools: list[AgentTool] | None = None,
        messages: list[AgentMessage] | None = None,
    ):
        self.system_prompt = system_prompt
        self.model = model if model is not None else _default_model()
        self.thinking_level: ThinkingLevel = thinking_level
        self._tools: list[AgentTool] = list(tools) if tools is not None else []
        self._messages: list[AgentMessage] = list(messages) if messages is not None else []
        # True while the agent is processing a prompt or continuation. Remains
        # True until awaited `agent_end` listeners settle.
        self.is_streaming = False
        # Partial assistant message for the current streamed response, if any.
        self.streaming_message: AgentMessage | None = None
        # Tool call ids currently executing.
        self.pending_tool_calls: set[str] = set()
        # Error message from the most recent failed or aborted assistant turn, if any.
        self.error_message: str | None = None

    @property
    def tools(self) -> list[AgentTool]:
        return self._tools

    @tools.setter
    def tools(self, next_tools: list[AgentTool]) -> None:
        self._tools = list(next_tools)

    @property
    def messages(self) -> list[AgentMessage]:
        return self._messages

    @messages.setter
    def messages(self, next_messages: list[AgentMessage]) -> None:
        self._messages = list(next_messages)


@dataclass(slots=True)
class AgentInitialState:
    """Initial state subset accepted by `Agent` (pi: `AgentOptions.initialState`)."""

    system_prompt: str | None = None
    model: Model | None = None
    thinking_level: ThinkingLevel | None = None
    tools: list[AgentTool] | None = None
    messages: list[AgentMessage] | None = None


class PendingMessageQueue:
    def __init__(self, mode: QueueMode):
        self._messages: list[AgentMessage] = []
        self._lock = threading.Lock()
        self.mode: QueueMode = mode

    def enqueue(self, message: AgentMessage) -> None:
        with self._lock:
            self._messages.append(message)

    def has_items(self) -> bool:
        return len(self._messages) > 0

    def drain(self) -> list[AgentMessage]:
        with self._lock:
            if self.mode == "all":
                drained = list(self._messages)
                self._messages = []
                return drained

            if not self._messages:
                return []
            first = self._messages[0]
            self._messages = self._messages[1:]
            return [first]

    def clear(self) -> None:
        with self._lock:
            self._messages = []


@dataclass(slots=True)
class _ActiveRun:
    done: tonio.Event
    cancel: CancelToken


@dataclass(slots=True)
class _RunOptions:
    skip_initial_steering_poll: bool = False


class Agent:
    """Stateful wrapper around the low-level agent loop."""

    def __init__(
        self,
        *,
        stream_fn: StreamFn | None = None,
        initial_state: AgentInitialState | None = None,
        convert_to_llm: Callable[[list[AgentMessage]], Awaitable[list[Message]]] | None = None,
        transform_context: Callable[[list[AgentMessage], CancelToken | None], Awaitable[list[AgentMessage]]]
        | None = None,
        get_api_key: Callable[[str], Awaitable[str | None]] | None = None,
        on_payload: Any = None,
        on_response: Any = None,
        before_tool_call: Callable[
            [BeforeToolCallContext, CancelToken | None],
            Awaitable[BeforeToolCallResult | None],
        ]
        | None = None,
        after_tool_call: Callable[
            [AfterToolCallContext, CancelToken | None],
            Awaitable[AfterToolCallResult | None],
        ]
        | None = None,
        prepare_next_turn: Callable[[CancelToken | None], Awaitable[AgentLoopTurnUpdate | None]] | None = None,
        prepare_next_turn_with_context: Callable[
            [PrepareNextTurnContext, CancelToken | None],
            Awaitable[AgentLoopTurnUpdate | None],
        ]
        | None = None,
        steering_mode: QueueMode | None = None,
        follow_up_mode: QueueMode | None = None,
        session_id: str | None = None,
        thinking_budgets: ThinkingBudgets | None = None,
        transport: Transport | None = None,
        max_retry_delay_ms: float | None = None,
        tool_execution: ToolExecutionMode | None = None,
    ):
        initial = initial_state if initial_state is not None else AgentInitialState()
        self._state = AgentState(
            system_prompt=initial.system_prompt if initial.system_prompt is not None else "",
            model=initial.model,
            thinking_level=initial.thinking_level if initial.thinking_level is not None else "off",
            tools=initial.tools,
            messages=initial.messages,
        )
        # Listener order matters: awaited in registration order.
        self._listeners: list[Callable[[AgentEvent, CancelToken], Awaitable[None]]] = []
        self._steering_queue = PendingMessageQueue(steering_mode if steering_mode is not None else "one-at-a-time")
        self._follow_up_queue = PendingMessageQueue(follow_up_mode if follow_up_mode is not None else "one-at-a-time")
        self._active_run: _ActiveRun | None = None

        self.convert_to_llm = convert_to_llm if convert_to_llm is not None else default_convert_to_llm
        self.transform_context = transform_context
        self.stream_function: StreamFn = stream_fn if stream_fn is not None else get_default_stream_fn()
        self.get_api_key = get_api_key
        self.on_payload = on_payload
        self.on_response = on_response
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.prepare_next_turn = prepare_next_turn
        self.prepare_next_turn_with_context = prepare_next_turn_with_context
        # Session identifier forwarded to providers for cache-aware backends.
        self.session_id = session_id
        # Optional per-level thinking token budgets forwarded to the stream function.
        self.thinking_budgets = thinking_budgets
        # Preferred transport forwarded to the stream function.
        self.transport: Transport = transport if transport is not None else "auto"
        # Optional cap for provider-requested retry delays.
        self.max_retry_delay_ms = max_retry_delay_ms
        # Tool execution strategy for assistant messages containing multiple tool calls.
        self.tool_execution: ToolExecutionMode = tool_execution if tool_execution is not None else "parallel"

    def subscribe(self, listener: Callable[[AgentEvent, CancelToken], Awaitable[None]]) -> Callable[[], None]:
        """Subscribe to agent lifecycle events.

        Listener results are awaited in subscription order and are included in
        the current run's settlement. Listeners also receive the active cancel
        token for the current run.

        `agent_end` is the final emitted event for a run, but the agent does
        not become idle until all awaited listeners for that event have settled.
        """
        self._listeners.append(listener)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return unsubscribe

    @property
    def state(self) -> AgentState:
        """Current agent state.

        Assigning `state.tools` or `state.messages` copies the provided
        top-level list.
        """
        return self._state

    @property
    def steering_mode(self) -> QueueMode:
        """Controls how queued steering messages are drained."""
        return self._steering_queue.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self._steering_queue.mode = mode

    @property
    def follow_up_mode(self) -> QueueMode:
        """Controls how queued follow-up messages are drained."""
        return self._follow_up_queue.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_queue.mode = mode

    def steer(self, message: AgentMessage) -> None:
        """Queue a message to be injected after the current assistant turn finishes."""
        self._steering_queue.enqueue(message)

    def follow_up(self, message: AgentMessage) -> None:
        """Queue a message to run only after the agent would otherwise stop."""
        self._follow_up_queue.enqueue(message)

    def clear_steering_queue(self) -> None:
        """Remove all queued steering messages."""
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        """Remove all queued follow-up messages."""
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        """Remove all queued steering and follow-up messages."""
        self.clear_steering_queue()
        self.clear_follow_up_queue()

    def has_queued_messages(self) -> bool:
        """Returns True when either queue still contains pending messages."""
        return self._steering_queue.has_items() or self._follow_up_queue.has_items()

    @property
    def signal(self) -> CancelToken | None:
        """Active cancel token for the current run, if any (pi: `signal`)."""
        run = self._active_run
        return run.cancel if run is not None else None

    def abort(self) -> None:
        """Abort the current run, if one is active."""
        run = self._active_run
        if run is not None:
            run.cancel.cancel()

    async def wait_for_idle(self) -> None:
        """Resolve when the current run and all awaited event listeners have finished.

        This resolves after `agent_end` listeners settle.
        """
        run = self._active_run
        if run is None:
            return
        await run.done.wait(None)

    def reset(self) -> None:
        """Clear transcript state, runtime state, and queued messages."""
        self._state.messages = []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        self._state.error_message = None
        self.clear_follow_up_queue()
        self.clear_steering_queue()

    async def prompt(
        self, input: str | AgentMessage | list[AgentMessage], images: list[ImageContent] | None = None
    ) -> None:
        """Start a new prompt from text, a single message, or a batch of messages."""
        if self._active_run is not None:
            raise Exception(
                "Agent is already processing a prompt. Use steer() or followUp() to queue messages, "
                "or wait for completion."
            )
        messages = self._normalize_prompt_input(input, images)
        await self._run_prompt_messages(messages)

    async def continue_(self) -> None:
        """Continue from the current transcript (pi: `continue()`).

        The last message must be a user or tool-result message.
        """
        if self._active_run is not None:
            raise Exception("Agent is already processing. Wait for completion before continuing.")

        last_message = self._state.messages[-1] if self._state.messages else None
        if last_message is None:
            raise Exception("No messages to continue from")

        if getattr(last_message, "role", None) == "assistant":
            queued_steering = self._steering_queue.drain()
            if queued_steering:
                await self._run_prompt_messages(queued_steering, _RunOptions(skip_initial_steering_poll=True))
                return

            queued_follow_ups = self._follow_up_queue.drain()
            if queued_follow_ups:
                await self._run_prompt_messages(queued_follow_ups)
                return

            raise Exception("Cannot continue from message role: assistant")

        await self._run_continuation()

    def _normalize_prompt_input(
        self, input: str | AgentMessage | list[AgentMessage], images: list[ImageContent] | None
    ) -> list[AgentMessage]:
        if isinstance(input, list):
            return input

        if not isinstance(input, str):
            return [input]

        content: list[TextContent | ImageContent] = [TextContent(text=input)]
        if images:
            content.extend(images)
        return [UserMessage(content=content, timestamp=int(time.time() * 1000))]

    async def _run_prompt_messages(self, messages: list[AgentMessage], options: _RunOptions | None = None) -> None:
        async def executor(cancel: CancelToken) -> None:
            await run_agent_loop(
                messages,
                self._create_context_snapshot(),
                self._create_loop_config(options if options is not None else _RunOptions()),
                self._process_events,
                cancel,
                self.stream_function,
            )

        await self._run_with_lifecycle(executor)

    async def _run_continuation(self) -> None:
        async def executor(cancel: CancelToken) -> None:
            await run_agent_loop_continue(
                self._create_context_snapshot(),
                self._create_loop_config(),
                self._process_events,
                cancel,
                self.stream_function,
            )

        await self._run_with_lifecycle(executor)

    def _create_context_snapshot(self) -> AgentContext:
        return AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=list(self._state.tools),
        )

    def _create_loop_config(self, options: _RunOptions | None = None) -> AgentLoopConfig:
        skip_initial_steering_poll = options is not None and options.skip_initial_steering_poll

        prepare_next_turn = None
        if self.prepare_next_turn_with_context is not None or self.prepare_next_turn is not None:

            async def prepare_next_turn(context: PrepareNextTurnContext) -> AgentLoopTurnUpdate | None:
                if self.prepare_next_turn_with_context is not None:
                    return await self.prepare_next_turn_with_context(context, self.signal)
                if self.prepare_next_turn is not None:
                    return await self.prepare_next_turn(self.signal)
                return None

        async def get_steering_messages() -> list[AgentMessage]:
            nonlocal skip_initial_steering_poll
            if skip_initial_steering_poll:
                skip_initial_steering_poll = False
                return []
            return self._steering_queue.drain()

        async def get_follow_up_messages() -> list[AgentMessage]:
            return self._follow_up_queue.drain()

        return AgentLoopConfig(
            model=self._state.model,
            reasoning=None if self._state.thinking_level == "off" else self._state.thinking_level,
            session_id=self.session_id,
            on_payload=self.on_payload,
            on_response=self.on_response,
            transport=self.transport,
            thinking_budgets=self.thinking_budgets,
            max_retry_delay_ms=self.max_retry_delay_ms,
            tool_execution=self.tool_execution,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            prepare_next_turn=prepare_next_turn,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
        )

    async def _run_with_lifecycle(self, executor: Callable[[CancelToken], Awaitable[None]]) -> None:
        if self._active_run is not None:
            raise Exception("Agent is already processing.")

        run = _ActiveRun(done=tonio.Event(), cancel=CancelToken())
        self._active_run = run

        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None

        try:
            await executor(run.cancel)
        except Exception as error:
            await self._handle_run_failure(error, run.cancel.cancelled)
        finally:
            self._finish_run()

    async def _handle_run_failure(self, error: Exception, aborted: bool) -> None:
        failure_message = AssistantMessage(
            content=[TextContent(text="")],
            api=self._state.model.api,
            provider=self._state.model.provider,
            model=self._state.model.id,
            usage=Usage(),
            stop_reason="aborted" if aborted else "error",
            error_message=str(error),
            timestamp=int(time.time() * 1000),
        )
        await self._process_events(MessageStartEvent(message=failure_message))
        await self._process_events(MessageEndEvent(message=failure_message))
        await self._process_events(TurnEndEvent(message=failure_message, tool_results=[]))
        await self._process_events(AgentEndEvent(messages=[failure_message]))

    def _finish_run(self) -> None:
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        run = self._active_run
        self._active_run = None
        if run is not None:
            run.done.set()

    async def _process_events(self, event: AgentEvent) -> None:
        """Reduce internal state for a loop event, then await listeners.

        `agent_end` only means no further loop events will be emitted. The run
        is considered idle later, after all awaited listeners for `agent_end`
        finish and `_finish_run()` clears runtime-owned state.
        """
        if event.type == "message_start" or event.type == "message_update":
            self._state.streaming_message = event.message
        elif event.type == "message_end":
            self._state.streaming_message = None
            self._state.messages.append(event.message)
        elif event.type == "tool_execution_start":
            pending_tool_calls = set(self._state.pending_tool_calls)
            pending_tool_calls.add(event.tool_call_id)
            self._state.pending_tool_calls = pending_tool_calls
        elif event.type == "tool_execution_end":
            pending_tool_calls = set(self._state.pending_tool_calls)
            pending_tool_calls.discard(event.tool_call_id)
            self._state.pending_tool_calls = pending_tool_calls
        elif event.type == "turn_end":
            if getattr(event.message, "role", None) == "assistant" and event.message.error_message:
                self._state.error_message = event.message.error_message
        elif event.type == "agent_end":
            self._state.streaming_message = None

        run = self._active_run
        if run is None:
            raise Exception("Agent listener invoked outside active run")
        for listener in list(self._listeners):
            await listener(event, run.cancel)
