"""Harness events (port of pi `harness/events.ts`).

Listeners are async-only (project callback policy; pi's `void | Promise<void>`
union is not ported). pi's `emit` fires listeners synchronously and discards
returned promises; here direct `on()` listeners detach onto the runtime like
`core/event_bus.py`, so no part of a listener body runs inline during `emit`.
Watch delivery, however, is ORDER-PRESERVING by contract (a watch consumer
reconstructs state from its snapshot plus a gap-free ordered event stream), so
each started watch drains through an unbounded channel into a single detached
task that awaits the listener per event — pi gets the same ordering for free
from single-threaded synchronous dispatch.
"""

import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import tonio.colored as tonio
from tonio.colored.sync import channel


@dataclass(slots=True, kw_only=True)
class RunStartEvent:
    lane: str
    run_id: str
    type: Literal["run_start"] = "run_start"


@dataclass(slots=True, kw_only=True)
class RunEndEvent:
    lane: str
    run_id: str
    outcome: Literal["completed", "aborted", "failed"]
    leaf_id: str
    type: Literal["run_end"] = "run_end"


type HarnessEvent = RunStartEvent | RunEndEvent
type HarnessEventListener = Callable[[Any], Awaitable[None]]


class Events(Protocol):
    def on(self, type: str, listener: HarnessEventListener) -> Callable[[], None]:
        """Register a passive listener for future events and return its unsubscribe function.

        Earlier events are not replayed and no current-state snapshot is provided; use a lane
        or session watch for both.
        """
        ...


_STOP = object()


class WatchHandle[TSnapshot]:
    def __init__(
        self,
        snapshot: TSnapshot,
        start: Callable[[HarnessEventListener], None],
        unsubscribe: Callable[[], None],
    ):
        self.snapshot = snapshot
        self.start = start
        self.unsubscribe = unsubscribe


class HarnessEventBus:
    def __init__(self) -> None:
        self._listeners: dict[str, list[HarnessEventListener]] = {}
        self._watch_listeners: list[Callable[[HarnessEvent], None]] = []
        # pi's listener bookkeeping is event-loop-atomic; tonio tasks run on
        # real threads, so registration/emission share a sync critical section.
        self._guard = threading.Lock()

    def on(self, type: str, listener: HarnessEventListener) -> Callable[[], None]:
        """Register a listener for future events of one type and return its unsubscribe function.

        Earlier events are not replayed, and no snapshot or event buffer is provided.
        """

        # Wrap this event-specific callback so it can be stored as a general
        # HarnessEvent listener. Keep the wrapper reference so unsubscribe can
        # remove that exact function from the list.
        async def receive(event: HarnessEvent) -> None:
            if event.type == type:
                await listener(event)

        with self._guard:
            self._listeners.setdefault(type, []).append(receive)

        def unsubscribe() -> None:
            with self._guard:
                listeners = self._listeners.get(type)
                if listeners is not None and receive in listeners:
                    listeners.remove(receive)
                    if not listeners:
                        del self._listeners[type]

        return unsubscribe

    def emit(self, event: HarnessEvent) -> None:
        """Publish an event to current event subscriptions and watch subscriptions."""
        with self._guard:
            listeners = list(self._listeners.get(event.type, ()))
            watch_listeners = list(self._watch_listeners)
        # Deliver only to direct listeners registered for this event type.
        # Awaitables are not awaited because emit() is synchronous.
        for listener in listeners:
            tonio.spawn.without_tracking(listener(event))
        # Deliver every event to each watcher; watch() handles buffering until start().
        for watch_listener in watch_listeners:
            watch_listener(event)

    def watch[TSnapshot](self, capture_snapshot: Callable[[], TSnapshot]) -> WatchHandle[TSnapshot]:
        state_guard = threading.Lock()
        sender: Any = None
        buffered: list[HarnessEvent] = []
        unsubscribed = False

        def receive(event: HarnessEvent) -> None:
            with state_guard:
                if sender is not None:
                    sender.send(event)
                else:
                    buffered.append(event)

        with self._guard:
            self._watch_listeners.append(receive)
        snapshot = capture_snapshot()

        def start(next_listener: HarnessEventListener) -> None:
            nonlocal sender
            with state_guard:
                if sender is not None or unsubscribed:
                    return
                new_sender, receiver = channel.unbounded()
                # The channel stays the single ordered path: buffered events
                # enter it before the sender is published, so reentrant
                # emissions during the flush preserve order (pi's re-buffering
                # flush loop).
                for event in buffered:
                    new_sender.send(event)
                buffered.clear()
                sender = new_sender

            async def drain() -> None:
                while True:
                    try:
                        item = await receiver.receive()
                    except BrokenPipeError:
                        # The sender closure was dropped without unsubscribe()
                        # (a collected WatchHandle): no event can ever arrive
                        # again, so a closed channel is the same end-of-watch
                        # as _STOP — not an error to leak to tonio's printer.
                        return
                    if item is _STOP:
                        return
                    await next_listener(item)

            tonio.spawn.without_tracking(drain())

        def unsubscribe() -> None:
            nonlocal unsubscribed
            with self._guard:
                if receive in self._watch_listeners:
                    self._watch_listeners.remove(receive)
            with state_guard:
                unsubscribed = True
                buffered.clear()
                if sender is not None:
                    sender.send(_STOP)

        return WatchHandle(snapshot, start, unsubscribe)
