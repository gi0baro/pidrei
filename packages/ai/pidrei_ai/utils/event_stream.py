"""Port of pi's `EventStream` (packages/ai/src/utils/event-stream.ts).

The central producer/consumer primitive of the whole system: producers `push()`
events synchronously, consumers `async for` over the stream, and `result()`
resolves once a completing event arrives (or `end()` is called with a result).

Differences from the TypeScript original are strictly about the runtime:
`push()`/`end()` are thread-safe (producers may run on any tonio worker
thread), and delivery goes through a tonio unbounded channel instead of a
queue-plus-waiters pair. Observable behavior is unchanged, including the
quirk that `end()` without a result leaves `result()` pending; `fail()` /
`spawn_producer()` exist so a producer that dies never leaves it pending.
"""

import threading
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

import tonio.colored as tonio
from tonio.colored import Event
from tonio.colored.sync import channel
from tonio.exceptions import CancelledError

from pidrei_ai.types import AssistantMessage, AssistantMessageEvent, ErrorEvent
from pidrei_ai.utils.cancel import AbortError, CancelToken


_SENTINEL = object()
_UNSET: Any = object()


class EventStream[T, R]:
    def __init__(self, is_complete: Callable[[T], bool], extract_result: Callable[[T], R]) -> None:
        self._is_complete = is_complete
        self._extract_result = extract_result
        self._sender, self._receiver = channel.unbounded()
        self._lock = threading.Lock()
        self._done = False
        self._result: R | None = None
        self._error: BaseException | None = None
        self._result_event = Event()

    def push(self, event: T) -> None:
        """Deliver an event to the consumer; ignored after the stream is done."""
        with self._lock:
            if self._done:
                return
            if self._is_complete(event):
                self._done = True
                self._result = self._extract_result(event)
                self._sender.send(event)
                self._sender.send(_SENTINEL)
                self._result_event.set()
            else:
                self._sender.send(event)

    def end(self, result: R = _UNSET) -> None:
        """Terminate the stream; optionally resolve `result()` (first write wins)."""
        with self._lock:
            if result is not _UNSET and not self._result_event.is_set():
                self._result = result
                self._result_event.set()
            if self._done:
                return
            self._done = True
            self._sender.send(_SENTINEL)

    def fail(self, error: BaseException) -> None:
        """Terminate the stream because its producer died; `result()` raises.

        In JS an unresolved promise is garbage; here a consumer parked in
        `result()` is a worker-side waiter with no timeout, so a producer
        failure must settle it.
        """
        with self._lock:
            if not self._result_event.is_set():
                self._error = error
                self._result_event.set()
            if self._done:
                return
            self._done = True
            self._sender.send(_SENTINEL)

    @property
    def done(self) -> bool:
        return self._done

    def spawn_producer(self, producer: Coroutine[Any, Any, None], cancel: CancelToken | None = None) -> None:
        """Run `producer` as the child of a scope this stream owns.

        The tonio shape of pi's "fetch with an AbortSignal": the producer's
        awaits — request head, retry backoff, every body read — are plain
        awaits; cancelling `cancel` cancels the scope, and the owner task,
        which waits inside the scope for either the producer to finish or
        the token to fire, then leaves it, which is when tonio evaluates the
        cancellation and unwinds the child at its current suspension point.
        Nothing is paid per chunk.

        After a cancel the producer may not get to run its own error path
        (a child parked on I/O is not resumed), so the owner terminates the
        stream itself via `_abort` if it is still open. If the producer
        escapes with anything else, the stream fails (`result()` raises) —
        adapters convert `Exception` to an error event themselves; this
        covers what they don't.
        """
        finished = tonio.Event()
        started = False

        async def _child() -> None:
            nonlocal started
            started = True
            try:
                await producer
            except CancelledError:
                raise  # the owner terminates the stream via `_abort`
            except BaseException as error:
                self.fail(error)
                raise
            finally:
                finished.set()

        # A token that is already cancelled does not arm the scope: pi's
        # adapters still run up to the fetch (`on_payload` included) and
        # fail on their own cancellation checks; pre-cancelling would skip
        # the producer entirely.
        armed = cancel is not None and not cancel.cancelled

        async def _owner() -> None:
            unsubscribe = None
            child = _child()
            async with tonio.scope() as scope:
                scope.spawn(child)
                if armed:

                    def _on_cancel(_reason: BaseException) -> None:
                        scope.cancel()
                        finished.set()

                    unsubscribe = cancel.on_cancel(_on_cancel)
                await finished.wait()
            if unsubscribe is not None:
                unsubscribe()
            if not started:
                # Cancelled before its first step: the scope drops the
                # coroutine without awaiting it; close both so nothing warns.
                child.close()
                producer.close()
            if armed and cancel.cancelled and not self._done:
                self._abort(cancel)

        tonio.spawn.without_tracking(_owner())

    def _abort(self, cancel: CancelToken) -> None:
        """Terminate a stream whose producer was cancelled before it could."""
        reason = cancel.reason
        self.fail(reason if reason is not None else AbortError("Operation was aborted"))

    async def __aiter__(self) -> AsyncIterator[T]:
        while True:
            item = await self._receiver.receive()
            if item is _SENTINEL:
                return
            yield item

    async def result(self) -> R:
        """Resolve to the extracted result of the completing event."""
        await self._result_event.wait()
        if self._error is not None:
            raise self._error
        return self._result  # type: ignore[return-value]


class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    """Event stream of a single assistant response (pi: event-stream.ts:69-83).

    Completes on a `done` event (result: the final message) or an `error`
    event (result: the error `AssistantMessage` — returned, not raised).
    """

    def __init__(self) -> None:
        super().__init__(self._event_is_complete, self._event_extract_result)
        # The producer's in-progress message. Set by adapters as soon as it
        # exists so a cancel that lands while the producer is parked on I/O
        # still terminates with pi's "aborted message with partial content".
        self.partial: AssistantMessage | None = None

    def _abort(self, cancel: CancelToken) -> None:
        message = self.partial
        if message is None:
            super()._abort(cancel)
            return
        message.stop_reason = "aborted"
        message.error_message = "Request was aborted"
        self.push(ErrorEvent(reason="aborted", error=message))
        self.end()

    @staticmethod
    def _event_is_complete(event: AssistantMessageEvent) -> bool:
        return event.type in ("done", "error")

    @staticmethod
    def _event_extract_result(event: AssistantMessageEvent) -> AssistantMessage:
        if event.type == "done":
            return event.message
        if event.type == "error":
            return event.error
        raise ValueError("Unexpected event type for final result")


def create_assistant_message_event_stream() -> AssistantMessageEventStream:
    """Factory for AssistantMessageEventStream (for use in extensions)."""
    return AssistantMessageEventStream()
