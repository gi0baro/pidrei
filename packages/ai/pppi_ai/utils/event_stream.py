"""Port of pi's `EventStream` (packages/ai/src/utils/event-stream.ts).

The central producer/consumer primitive of the whole system: producers `push()`
events synchronously, consumers `async for` over the stream, and `result()`
resolves once a completing event arrives (or `end()` is called with a result).

Differences from the TypeScript original are strictly about the runtime:
`push()`/`end()` are thread-safe (producers may run on any tonio worker
thread), and delivery goes through a tonio unbounded channel instead of a
queue-plus-waiters pair. Observable behavior is unchanged, including the
quirk that `end()` without a result leaves `result()` pending forever.
"""

import threading
from collections.abc import AsyncIterator, Callable
from typing import Any

from tonio.colored import Event
from tonio.colored.sync import channel


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

    async def __aiter__(self) -> AsyncIterator[T]:
        while True:
            item = await self._receiver.receive()
            if item is _SENTINEL:
                return
            yield item

    async def result(self) -> R:
        """Resolve to the extracted result of the completing event."""
        await self._result_event.wait()
        return self._result  # type: ignore[return-value]
