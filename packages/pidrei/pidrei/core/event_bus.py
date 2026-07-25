"""Mirror of pi coding-agent src/core/event-bus.ts.

pi wraps a Node EventEmitter whose handlers are fire-and-forget async
functions with error logging. Here sync handler results are delivered
inline and awaitable results are detached onto the runtime, preserving
the fire-and-forget contract.
"""

import inspect
import sys
import threading
from collections.abc import Callable
from typing import Any

import tonio.colored as tonio


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[Any], Any]]] = {}
        self._guard = threading.Lock()

    def emit(self, channel: str, data: Any) -> None:
        with self._guard:
            handlers = list(self._handlers.get(channel, ()))
        for handler in handlers:
            try:
                result = handler(data)
            except Exception as error:
                print(f"Event handler error ({channel}):", error, file=sys.stderr)
                continue
            if inspect.isawaitable(result):
                tonio.spawn.without_tracking(self._await_handler(channel, result))

    async def _await_handler(self, channel: str, awaitable: Any) -> None:
        try:
            await awaitable
        except Exception as error:
            print(f"Event handler error ({channel}):", error, file=sys.stderr)

    def on(self, channel: str, handler: Callable[[Any], Any]) -> Callable[[], None]:
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
