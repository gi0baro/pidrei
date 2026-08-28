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
- pi's single thread made "listeners never overlap" free; here tool bodies run
  in parallel and emit concurrently, so each run owns one dispatcher task
  (`_dispatch_events`) that reduces state and awaits listeners in emit order.
- pi's loose queue/lifecycle fields (the pending-message queues, the active
  run, the "already processing" guards) are the internal `_AgentMailbox`
  (PROPER_MT_DESIGN.md §6): a standing actor task that owns the queues and
  run admission. Enqueues and clears stay sync (fire-and-forget mailbox
  sends — FIFO makes them visible to every later drain or query); only
  `has_queued_messages` is awaited in pidrei (it needs the mailbox's
  answer), and `abort`/`signal`/`wait_for_idle` work through the published
  run record — the cancel token and the `done` event are the cross-task
  signals.
- pi's `continue()` is `continue_()` (`continue` is a Python keyword).
"""

import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio
from tonio.colored.sync import channel

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
    ShouldStopAfterTurnContext,
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


class _PendingMessages:
    """One steering/follow-up queue inside the agent mailbox (pi: `PendingMessageQueue`).

    Task-confined: the message list is touched only inside jobs executed by
    the mailbox consumer, so pi's queue logic ports verbatim with no lock.
    `mode` is the one field written from outside (the sync mode setters
    rebind it; the drain job reads it once) — a published value, not queue
    state.
    """

    __slots__ = ("_messages", "mode")

    def __init__(self, mode: QueueMode):
        self._messages: list[AgentMessage] = []
        self.mode: QueueMode = mode

    def enqueue(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def has_items(self) -> bool:
        return len(self._messages) > 0

    def drain(self) -> list[AgentMessage]:
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
        self._messages = []


#: Set to a file path to enable the dispatcher stall meter (appended to).
_STALL_LOG_ENV = "PIDREI_DISPATCH_STALL_LOG"
_STALL_THRESHOLD_S = 0.050


def _append_stall_log(path: str, text: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)


class _DispatchStallMeter:
    """Per-run observation-latency meter for the dispatcher — the §1 gate data.

    PROPER_MT_DESIGN.md keeps the fused observer pipeline (§1) until a trace
    shows the dispatcher stalled behind a slow listener while producers queue;
    this meter is that trace. Enabled by pointing ``PIDREI_DISPATCH_STALL_LOG``
    at a file — when unset the dispatcher takes no timestamps, so the default
    path pays nothing. Per event it records the observation latency (event
    received on the dispatcher → ``observed.set()``, i.e. reduce + all
    listeners) attributed to the event type; events over the ~50 ms threshold
    get one line each, and a per-type summary line is appended when the run's
    dispatcher closes. Only used on the dispatcher task — no synchronization.
    """

    __slots__ = ("_dropped", "_path", "_stalls", "_totals")

    _MAX_STALL_LINES = 500

    def __init__(self, path: str):
        self._path = path
        # event type -> [count, total seconds, max seconds]
        self._totals: dict[str, list[float]] = {}
        self._stalls: list[str] = []
        self._dropped = 0

    def record(self, event_type: str, elapsed: float) -> None:
        entry = self._totals.setdefault(event_type, [0, 0.0, 0.0])
        entry[0] += 1
        entry[1] += elapsed
        entry[2] = max(entry[2], elapsed)
        if elapsed >= _STALL_THRESHOLD_S:
            if len(self._stalls) < self._MAX_STALL_LINES:
                self._stalls.append(f"stall {event_type}: {elapsed * 1000:.1f} ms\n")
            else:
                self._dropped += 1

    async def flush(self) -> None:
        summary = ", ".join(
            f"{event_type}: n={int(count)} total={total * 1000:.1f}ms max={peak * 1000:.1f}ms"
            for event_type, (count, total, peak) in self._totals.items()
        )
        lines = "".join(self._stalls)
        if self._dropped:
            lines += f"({self._dropped} further stalls not listed)\n"
        await tonio.spawn_blocking(_append_stall_log, self._path, f"{lines}run summary — {summary}\n")


@dataclass(slots=True)
class _PendingEvent:
    """One emitted event waiting for the run's dispatcher to observe it."""

    event: AgentEvent
    observed: tonio.Event
    error: BaseException | None = None


@dataclass(slots=True)
class _ActiveRun:
    done: tonio.Event
    cancel: CancelToken
    # Ordered feed to the run's dispatcher task (`_PendingEvent | None`;
    # `None` closes it).
    events: Any
    # Set by the run lifecycle when something escapes the run (a listener
    # failure surfacing through `_handle_run_failure`, a cancellation);
    # re-raised at the `prompt()`/`continue_()` awaiter after `done`.
    error: BaseException | None = None


@dataclass(slots=True)
class _MailboxJob:
    """One unit of mailbox work: a sync closure over mailbox-owned state."""

    fn: Callable[[], Any]
    done: tonio.Event | None = None
    result: Any = None
    error: BaseException | None = None


class _AgentMailbox:
    """Standing actor task owning one `Agent`'s queues and run admission
    (PROPER_MT_DESIGN.md §6).

    pi keeps this state as loose fields on the Agent; here one consumer task
    owns it and every operation arrives as a `_MailboxJob` on the channel,
    executed in send (FIFO) order. Contract:

    - **Queue state is task-local.** `steering`/`follow_up` are touched only
      inside jobs; there is no shared queue structure and no lock. Mutations
      (enqueues, clears) are `post(fn)` — fire-and-forget, which suffices
      because every observer also goes through the channel and FIFO puts it
      behind the mutation; only operations needing an answer back (drains,
      the has-items query, admission) are awaited `run(fn)` calls.
    - **Jobs are sync closures.** They never await, so a job can never wait
      on the mailbox and self-deadlock; posted jobs must not raise (an
      awaited job's error re-raises at its caller).
    - **Run admission is a job.** `claim(run)` raises pi's "Agent is already
      processing." while the slot is held, so concurrent `prompt()` calls
      admit exactly one run in channel order. The admit job spawns the run
      lifecycle as a detached task ("the mailbox's run handling" — the
      dispatcher runs as its child); the lifecycle ends by posting
      `Agent._finish_run`, which releases the slot and only then sets the
      run's `done` event — an admission sent after `done` is therefore
      FIFO-behind the release and succeeds.
    - **`current` is the published run record**: rebound only by mailbox
      jobs, read (pinned) without a lock by the sync surfaces — `signal`,
      `abort` (firing the record's cancel token *is* the abort signal),
      `wait_for_idle`, and the entry points' pi-message guard checks.
    - **Lifetime is signaled, not scoped.** The consumer is spawned at
      construction (constructing an `Agent` therefore requires the running
      tonio runtime — as does everything in pidrei) and holds only the
      channel receiver: when the Agent (and with it this mailbox, the last
      sender) is garbage-collected the channel closes, buffered jobs still
      run, and the consumer exits on the closed-channel error.
    """

    __slots__ = ("_sender", "current", "follow_up", "steering")

    def __init__(self, steering_mode: QueueMode, follow_up_mode: QueueMode):
        self.steering = _PendingMessages(steering_mode)
        self.follow_up = _PendingMessages(follow_up_mode)
        self.current: _ActiveRun | None = None
        sender, receiver = channel.unbounded()
        self._sender = sender
        tonio.spawn.without_tracking(_consume_mailbox(receiver))

    def post(self, fn: Callable[[], Any]) -> None:
        """Run `fn` on the mailbox, fire-and-forget, in send order."""
        self._sender.send(_MailboxJob(fn))

    async def run(self, fn: Callable[[], Any]) -> Any:
        """Run `fn` on the mailbox and wait for it; returns its result."""
        job = _MailboxJob(fn, done=tonio.Event())
        self._sender.send(job)
        await job.done.wait(None)
        if job.error is not None:
            raise job.error
        return job.result

    def claim(self, run: _ActiveRun) -> None:
        if self.current is not None:
            raise Exception("Agent is already processing.")
        self.current = run

    def release(self, run: _ActiveRun) -> None:
        if self.current is run:
            self.current = None


async def _consume_mailbox(receiver: Any) -> None:
    # Module-level on purpose: holding a mailbox reference here would keep
    # the last channel sender alive and the closed-channel exit unreachable.
    while True:
        try:
            job = await receiver.receive()
        except BrokenPipeError:
            # The Agent (and its mailbox) was dropped; nothing can send again.
            return
        try:
            job.result = job.fn()
        except BaseException as error:
            if job.done is None:
                raise
            job.error = error
        finally:
            if job.done is not None:
                job.done.set()


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
        should_stop_after_turn: Callable[[ShouldStopAfterTurnContext, CancelToken | None], Awaitable[bool]]
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
        self._mailbox = _AgentMailbox(
            steering_mode if steering_mode is not None else "one-at-a-time",
            follow_up_mode if follow_up_mode is not None else "one-at-a-time",
        )

        self.convert_to_llm = convert_to_llm if convert_to_llm is not None else default_convert_to_llm
        self.transform_context = transform_context
        self.stream_function: StreamFn = stream_fn if stream_fn is not None else get_default_stream_fn()
        self.get_api_key = get_api_key
        self.on_payload = on_payload
        self.on_response = on_response
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.should_stop_after_turn = should_stop_after_turn
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
        return self._mailbox.steering.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self._mailbox.steering.mode = mode

    @property
    def follow_up_mode(self) -> QueueMode:
        """Controls how queued follow-up messages are drained."""
        return self._mailbox.follow_up.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self._mailbox.follow_up.mode = mode

    def steer(self, message: AgentMessage) -> None:
        """Queue a message to be injected after the current assistant turn finishes.

        Fire-and-forget mailbox send: every queue observer goes through the
        mailbox too, so any later drain or `has_queued_messages()` sees it.
        """
        mailbox = self._mailbox
        mailbox.post(lambda: mailbox.steering.enqueue(message))

    def follow_up(self, message: AgentMessage) -> None:
        """Queue a message to run only after the agent would otherwise stop.

        Fire-and-forget mailbox send: every queue observer goes through the
        mailbox too, so any later drain or `has_queued_messages()` sees it.
        """
        mailbox = self._mailbox
        mailbox.post(lambda: mailbox.follow_up.enqueue(message))

    def clear_steering_queue(self) -> None:
        """Remove all queued steering messages (applied in mailbox order)."""
        self._mailbox.post(self._mailbox.steering.clear)

    def clear_follow_up_queue(self) -> None:
        """Remove all queued follow-up messages (applied in mailbox order)."""
        self._mailbox.post(self._mailbox.follow_up.clear)

    def clear_all_queues(self) -> None:
        """Remove all queued steering and follow-up messages."""
        self.clear_steering_queue()
        self.clear_follow_up_queue()

    async def has_queued_messages(self) -> bool:
        """Resolves True when either queue still contains pending messages.

        Awaited (pi's call is sync) — the answer lives on the mailbox task;
        FIFO orders it after every enqueue and clear already sent.
        """
        mailbox = self._mailbox
        return await mailbox.run(lambda: mailbox.steering.has_items() or mailbox.follow_up.has_items())

    @property
    def signal(self) -> CancelToken | None:
        """Active cancel token for the current run, if any (pi: `signal`)."""
        run = self._mailbox.current
        return run.cancel if run is not None else None

    def abort(self) -> None:
        """Abort the current run, if one is active."""
        run = self._mailbox.current
        if run is not None:
            run.cancel.cancel()

    async def wait_for_idle(self) -> None:
        """Resolve when the current run and all awaited event listeners have finished.

        This resolves after `agent_end` listeners settle.
        """
        run = self._mailbox.current
        if run is None:
            return
        await run.done.wait(None)

    def reset(self) -> None:
        """Clear transcript state, runtime state, and queued messages."""
        if self._mailbox.current is not None:
            raise Exception("Agent is already processing. Wait for completion before resetting.")

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
        if self._mailbox.current is not None:
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
        if self._mailbox.current is not None:
            raise Exception("Agent is already processing. Wait for completion before continuing.")

        last_message = self._state.messages[-1] if self._state.messages else None
        if last_message is None:
            raise Exception("No messages to continue from")

        if getattr(last_message, "role", None) == "assistant":
            queued_steering = await self._mailbox.run(self._mailbox.steering.drain)
            if queued_steering:
                await self._run_prompt_messages(queued_steering, _RunOptions(skip_initial_steering_poll=True))
                return

            queued_follow_ups = await self._mailbox.run(self._mailbox.follow_up.drain)
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

        should_stop_after_turn = None
        if self.should_stop_after_turn is not None:
            configured_should_stop = self.should_stop_after_turn

            async def should_stop_after_turn(context: ShouldStopAfterTurnContext) -> bool:
                return await configured_should_stop(context, self.signal)

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
            return await self._mailbox.run(self._mailbox.steering.drain)

        async def get_follow_up_messages() -> list[AgentMessage]:
            return await self._mailbox.run(self._mailbox.follow_up.drain)

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
            should_stop_after_turn=should_stop_after_turn,
            prepare_next_turn=prepare_next_turn,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
        )

    async def _run_with_lifecycle(self, executor: Callable[[CancelToken], Awaitable[None]]) -> None:
        sender, receiver = channel.unbounded()
        run = _ActiveRun(done=tonio.Event(), cancel=CancelToken(), events=sender)

        def admit() -> None:
            # Admission is serialized on the mailbox: exactly one concurrent
            # caller wins the slot, the rest raise here (the entry points'
            # earlier checks give the pi-specific messages in every
            # sequential case). Streaming state flips before the awaiting
            # caller resumes, preserving pi's observable order.
            self._mailbox.claim(run)
            self._state.is_streaming = True
            self._state.streaming_message = None
            self._state.error_message = None
            tonio.spawn.without_tracking(self._run_lifecycle(run, executor, receiver))

        await self._mailbox.run(admit)
        try:
            await run.done.wait(None)
        except BaseException:
            # The awaiter is being unwound: fire the run's token — the same
            # signal `abort()` sends — and propagate; the run winds itself
            # down and anyone needing idleness awaits `wait_for_idle()`.
            run.cancel.cancel()
            raise
        if run.error is not None:
            raise run.error

    async def _run_lifecycle(
        self, run: _ActiveRun, executor: Callable[[CancelToken], Awaitable[None]], receiver: Any
    ) -> None:
        """The mailbox's run handling, spawned by the admit job."""
        try:
            # The dispatcher is a child of the run: it outlives every emitter
            # (tool tasks are joined by the loop before the executor returns,
            # and the failure events below go through it too) and is closed
            # and joined before the run is considered idle.
            async with tonio.scope() as scope:
                scope.spawn(self._dispatch_events(receiver))
                try:
                    await executor(run.cancel)
                except Exception as error:
                    await self._handle_run_failure(error, run.cancel.cancelled)
                finally:
                    run.events.send(None)
        except GeneratorExit:
            raise
        except BaseException as error:
            # Re-raised at the awaiting entry point after `done` — this task
            # is detached, so nothing else would observe the failure.
            run.error = error
        finally:
            self._mailbox.post(lambda: self._finish_run(run))

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

    def _finish_run(self, run: _ActiveRun) -> None:
        # Runs as a mailbox job: the slot is free before `done` is set, so an
        # admission sent after `done` lands FIFO-behind the release and wins.
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        self._mailbox.release(run)
        run.done.set()

    async def _process_events(self, event: AgentEvent) -> None:
        """Hand a loop event to the run's dispatcher and wait until it is observed.

        Parallel work, serialized observation: tool bodies run as parallel
        tasks and emit concurrently, but the state reducer and the listeners
        only ever run on the dispatcher task, one event at a time, in emit
        order — pi's single-thread contract ("listeners never overlap"),
        which the loop relies on when it reads state a listener set. Awaiting
        the ticket keeps `await emit(...)` meaning "listeners have settled",
        and a listener failure still surfaces at the emitter.

        `agent_end` only means no further loop events will be emitted. The run
        is considered idle later, after all awaited listeners for `agent_end`
        finish and `_finish_run()` clears runtime-owned state.
        """
        run = self._mailbox.current
        if run is None:
            raise Exception("Agent listener invoked outside active run")
        pending = _PendingEvent(event=event, observed=tonio.Event())
        run.events.send(pending)
        await pending.observed.wait()
        if pending.error is not None:
            raise pending.error

    async def _dispatch_events(self, receiver: Any) -> None:
        """Single consumer of a run's events: reduce state, then await listeners in order.

        A `message_update` carries a frozen per-delta snapshot of the
        provider's message (PROPER_MT_DESIGN.md step 2 — the old "live view,
        deltas are authoritative" doctrine is retired), so any listener may
        hold any event's message indefinitely.
        """
        stall_log = os.environ.get(_STALL_LOG_ENV)
        meter = _DispatchStallMeter(stall_log) if stall_log else None
        while True:
            pending = await receiver.receive()
            if pending is None:
                if meter is not None:
                    await meter.flush()
                return
            started = time.monotonic() if meter is not None else 0.0
            try:
                self._reduce(pending.event)
                run = self._mailbox.current
                cancel = run.cancel if run is not None else CancelToken()
                for listener in list(self._listeners):
                    await listener(pending.event, cancel)
            except BaseException as error:
                pending.error = error
            finally:
                pending.observed.set()
                if meter is not None:
                    meter.record(pending.event.type, time.monotonic() - started)

    def _reduce(self, event: AgentEvent) -> None:
        if event.type == "message_start" or event.type == "message_update":
            self._state.streaming_message = event.message
        elif event.type == "message_end":
            self._state.streaming_message = None
            self._state.messages.append(event.message)
        elif event.type == "tool_execution_start":
            # Copy-on-write (pi's `new Set`): readers on other tasks may be
            # iterating the previous set.
            self._state.pending_tool_calls = self._state.pending_tool_calls | {event.tool_call_id}
        elif event.type == "tool_execution_end":
            self._state.pending_tool_calls = self._state.pending_tool_calls - {event.tool_call_id}
        elif event.type == "turn_end":
            if getattr(event.message, "role", None) == "assistant" and event.message.error_message:
                self._state.error_message = event.message.error_message
        elif event.type == "agent_end":
            self._state.streaming_message = None
