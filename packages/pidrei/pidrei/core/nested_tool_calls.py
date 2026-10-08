"""Mirror of pi coding-agent src/core/nested-tool-calls.ts.

Tool calls that a tool makes while it runs (`ctx.execute_tool()`), for example
from codemode scripts. The agent loop does not know about them: the session
runs each one through the agent's tool pipeline (`run_tool_call`) with its own
hooks, emits `tool_execution_*` events with `parent_tool_call_id`, and records
the calls and their usage on the model-issued call's tool result message.

Nothing here runs until a tool calls `ctx.execute_tool()`.

The shape differs from pi (the `nested-calls-channel` recipe). pi keeps a
per-session `scopes` map and a mutable recorder per parent that nested calls
write into, and the session copies it onto the tool result at
`message_start`. Here nested calls run in parallel on workers and the tool
result is frozen at publication, so nothing shares a record:

- Every model-issued call gets a `NestedCallFeed` (a channel), created by the
  tool wrapper with the call's context. Nested calls send `NestedCallStarted`
  before they run and `NestedCallFinished` after; a nested tool's context
  inherits the feed, so calls at every depth land in the top-level record.
- When the parent's `execute` returns, the wrapper drains the feed and folds
  the messages in order (`fold_nested_calls`), applying pi's limits in
  `started` order. A `started` without its `finished` is "unfinished". The
  feed is then closed; a late send is dropped, never an error.
- The id counter (`<parent>/<n>`) lives on the calling tool's scope; it is the
  only thing siblings share.

Deviation: the record's instant is "when `execute` returned", one step before
pi's `message_start`. A call still unwinding during the `tool_result` hooks is
"unfinished" here where pi may report it "ok"/"error".
"""

import json
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from tonio.colored import sync
from tonio.colored.sync import channel

from pidrei_agent.types import (
    AgentEvent,
    AgentTool,
    AgentToolCall,
    AgentToolCallOutcome,
    AgentToolResult,
    AgentToolUpdateCallback,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from pidrei_ai.types import NestedToolCallRecord, NestedToolCalls, ToolCall, Usage
from pidrei_utils import clock
from pidrei_utils.cancel import CancelToken

from .usage_totals import combine_usage


@dataclass(slots=True, frozen=True)
class NestedCallLimits:
    max_calls: int
    max_argument_bytes_per_call: int
    max_argument_bytes_total: int
    max_error_chars: int


# Limits of the nested-call record on a tool result: arguments over the
# per-call or total size are omitted, calls beyond the count are dropped, and
# the record is marked incomplete when any of that happens.
NESTED_CALL_LIMITS = NestedCallLimits(
    max_calls=256,
    max_argument_bytes_per_call=8 * 1024,
    max_argument_bytes_total=32 * 1024,
    max_error_chars=500,
)


@dataclass(slots=True, frozen=True)
class NestedCallStarted:
    id: str
    name: str
    arguments: Any


@dataclass(slots=True, frozen=True)
class NestedCallFinished:
    id: str
    is_error: bool
    error_text: str
    duration_ms: int
    usage: Usage | None = None


type NestedCallMessage = NestedCallStarted | NestedCallFinished


@dataclass(slots=True, frozen=True)
class NestedCallSummary:
    """What the nested calls of one model-issued tool call leave on its tool result message."""

    # Becomes `nested_calls`. None when no nested call was made.
    calls: NestedToolCalls | None
    # Summed `usage` of the nested results, added to the message's `usage`.
    usage: Usage | None


def fold_nested_calls(messages: list[NestedCallMessage]) -> NestedCallSummary:
    """Fold a feed's messages, in the order they were sent, into the bounded
    record pi's `NestedCallRecorder` keeps."""
    limits = NESTED_CALL_LIMITS
    records: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    complete = True
    argument_bytes = 0
    usage: Usage | None = None
    for message in messages:
        if isinstance(message, NestedCallStarted):
            if len(records) >= limits.max_calls:
                complete = False
                continue
            record: dict[str, Any] = {"id": message.id, "name": message.name, "status": "unfinished"}
            text = json.dumps(
                message.arguments if message.arguments is not None else {}, ensure_ascii=False, separators=(",", ":")
            )
            size = len(text.encode("utf-8"))
            if size > limits.max_argument_bytes_per_call or argument_bytes + size > limits.max_argument_bytes_total:
                record["arguments_bytes"] = size
                complete = False
            else:
                record["arguments"] = json.loads(text)
                argument_bytes += size
            records.append(record)
            by_id[message.id] = record
            continue
        # Summed usage of every nested result, including calls dropped from the record.
        if message.usage is not None:
            usage = combine_usage(usage, message.usage) if usage is not None else message.usage
        record = by_id.get(message.id)
        if record is None:
            continue
        record["status"] = "error" if message.is_error else "ok"
        record["duration_ms"] = message.duration_ms
        if message.is_error and message.error_text:
            record["error"] = message.error_text[: limits.max_error_chars]
    if not records and complete:
        return NestedCallSummary(calls=None, usage=usage)
    calls = [NestedToolCallRecord(**record) for record in records]
    return NestedCallSummary(
        calls=NestedToolCalls(calls=calls, complete=complete and all(call.status != "unfinished" for call in calls)),
        usage=usage,
    )


class NestedCallFeed:
    """The channel the nested calls of one model-issued call report into.

    Closing it and a producer's "still open? then send" are one step under the
    guard, so a drain collects every message sent before it and a late sender
    (a call the tool left running) is dropped instead of queueing forever.
    """

    __slots__ = ("_closed", "_guard", "_receiver", "_sender")

    def __init__(self) -> None:
        self._sender, self._receiver = channel.unbounded()
        self._guard = threading.Lock()
        self._closed = False

    def send(self, message: NestedCallMessage) -> None:
        with self._guard:
            if not self._closed:
                self._sender.send(message)

    def drain(self) -> NestedCallSummary:
        """Close the feed and fold what was sent. Called once, by the tool
        wrapper, when the parent's `execute` returned."""
        with self._guard:
            self._closed = True
        messages: list[NestedCallMessage] = []
        receiver = self._receiver
        while (message := receiver.receive_nowait()) is not receiver.Empty:
            messages.append(message)
        return fold_nested_calls(messages)


@dataclass(slots=True)
class NestedCallScope:
    """The nested-call state of one tool call's context: calls it makes get
    the ids `<parent_id>/<n>` and report into `feed`, shared by the whole tree
    below the model-issued call."""

    parent_id: str
    feed: NestedCallFeed
    # Set inside a call that holds the exclusive queue, so its own nested calls do not wait on it.
    holds_queue: bool = False
    _next_id: int = 1
    _id_guard: threading.Lock = field(default_factory=threading.Lock)

    def next_call_id(self) -> str:
        with self._id_guard:
            number = self._next_id
            self._next_id += 1
        return f"{self.parent_id}/{number}"


@dataclass(slots=True, kw_only=True)
class ExecuteToolOptions:
    """Options for `ctx.execute_tool()`."""

    # Defaults to the calling tool's cancel token.
    cancel: CancelToken | None = None
    # Receives partial results of the nested tool, in addition to `tool_execution_update` events.
    on_update: AgentToolUpdateCallback[Any] | None = None


class NestedToolCallHost(Protocol):
    def get_tools(self) -> list[AgentTool]:
        """Tools nested calls resolve against."""
        ...

    def is_sequential(self) -> bool:
        """Whether every nested call runs exclusively, as when the agent executes tool calls sequentially."""
        ...

    def run_tool_call(
        self,
        tool_call: AgentToolCall,
        parent_tool_call_id: str,
        scope: NestedCallScope,
        cancel: CancelToken | None,
        on_update: Callable[[AgentToolResult[Any]], Awaitable[None]],
    ) -> Awaitable[AgentToolCallOutcome]:
        """Run the call through the tool pipeline, with hooks that report
        `parent_tool_call_id`, the target tool bound to `scope`."""
        ...

    def emit(self, event: AgentEvent) -> Awaitable[None]: ...


def _text_of(result: AgentToolResult[Any]) -> str:
    return "\n".join(block.text for block in result.content or [] if block.type == "text")


class NestedToolCallRunner:
    def __init__(self, host: NestedToolCallHost):
        self._host = host
        # Serializes nested calls that must not run concurrently, first come
        # first served (pi: the `queueTail` promise chain). A call cancelled
        # while it waits leaves the queue.
        self._queue_lock = sync.Lock()

    async def execute(
        self,
        scope: NestedCallScope,
        name: str,
        args: Any,
        options: ExecuteToolOptions | None = None,
    ) -> AgentToolCallOutcome:
        """Run `name` on behalf of the call that owns `scope`. The nested call
        gets the id `<parent id>/<n>`. Never raises for tool failures: they
        come back as `is_error=True`."""
        options = options if options is not None else ExecuteToolOptions()
        host = self._host
        parent_id = scope.parent_id
        tool_call = ToolCall(id=scope.next_call_id(), name=name, arguments=args if args is not None else {})
        started_at = clock.monotonic()
        scope.feed.send(NestedCallStarted(id=tool_call.id, name=name, arguments=tool_call.arguments))
        await host.emit(
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id, tool_name=name, args=tool_call.arguments, parent_tool_call_id=parent_id
            )
        )

        tool = next((entry for entry in host.get_tools() if entry.name == name), None)
        exclusive = not scope.holds_queue and (
            host.is_sequential() or (tool is not None and tool.execution_mode == "sequential")
        )
        child_scope = NestedCallScope(
            parent_id=tool_call.id, feed=scope.feed, holds_queue=scope.holds_queue or exclusive
        )

        async def on_update(partial_result: AgentToolResult[Any]) -> None:
            if options.on_update is not None:
                options.on_update(partial_result)
            await host.emit(
                ToolExecutionUpdateEvent(
                    tool_call_id=tool_call.id,
                    tool_name=name,
                    args=tool_call.arguments,
                    partial_result=partial_result,
                    parent_tool_call_id=parent_id,
                )
            )

        if exclusive:
            async with self._queue_lock:
                outcome = await host.run_tool_call(tool_call, parent_id, child_scope, options.cancel, on_update)
        else:
            outcome = await host.run_tool_call(tool_call, parent_id, child_scope, options.cancel, on_update)

        # pi: Math.round(performance.now() - startedAt).
        duration_ms = int((clock.monotonic() - started_at) * 1000 + 0.5)
        # Nested results are not persisted, so their usage is only counted through the record.
        scope.feed.send(
            NestedCallFinished(
                id=tool_call.id,
                is_error=outcome.is_error,
                error_text=_text_of(outcome.result),
                duration_ms=duration_ms,
                usage=outcome.result.usage,
            )
        )
        await host.emit(
            ToolExecutionEndEvent(
                tool_call_id=tool_call.id,
                tool_name=name,
                result=outcome.result,
                is_error=outcome.is_error,
                duration_ms=outcome.duration_ms,
                parent_tool_call_id=parent_id,
            )
        )
        return outcome
