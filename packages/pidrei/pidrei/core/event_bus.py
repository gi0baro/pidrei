"""Mirror of pi coding-agent src/core/event-bus.ts.

pi wraps a Node EventEmitter whose handlers are fire-and-forget async
functions with error logging. Handlers are async-only here (async-only
callback policy); each emitted event detaches the handler's awaitable onto
the runtime, preserving the fire-and-forget contract. Unlike pi, no part of
a handler runs inline during `emit` — a JS async function executes its body
up to the first `await` synchronously, a spawned coroutine does not.
"""

import sys
import threading
from collections.abc import Awaitable, Callable
from typing import Any

import tonio.colored as tonio


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[Any], Awaitable[Any]]]] = {}
        self._guard = threading.Lock()

    def emit(self, channel: str, data: Any) -> None:
        with self._guard:
            handlers = list(self._handlers.get(channel, ()))
        for handler in handlers:
            try:
                awaitable = handler(data)
            except Exception as error:
                print(f"Event handler error ({channel}):", error, file=sys.stderr)
                continue
            tonio.spawn.without_tracking(self._await_handler(channel, awaitable))

    async def _await_handler(self, channel: str, awaitable: Any) -> None:
        try:
            await awaitable
        except Exception as error:
            print(f"Event handler error ({channel}):", error, file=sys.stderr)

    def on(self, channel: str, handler: Callable[[Any], Awaitable[Any]]) -> Callable[[], None]:
        with self._guard:
            self._handlers.setdefault(channel, []).append(handler)

        def unsubscribe() -> None:
            with self._guard:
                handlers = self._handlers.get(channel)
                if handlers and handler in handlers:
                    handlers.remove(handler)

        return unsubscribe

    def clear(self) -> None:
        with self._guard:
            self._handlers.clear()


def create_event_bus() -> EventBus:
    return EventBus()
