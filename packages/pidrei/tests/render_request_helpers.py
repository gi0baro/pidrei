"""A stand-in TUI that lets tests wait for a component's background updates."""

import threading
from collections.abc import Callable

import tonio.colored as tonio


class RenderRequests:
    """Stands in for the TUI handed to a component, which applies its updates
    under `state_lock` and calls `request_render()`. Components that refresh
    in the background (the model selector's catalog refresh) request a render
    as the last step of each update, so a test waits for the state it needs
    with `until` instead of polling the render output."""

    def __init__(self) -> None:
        self.state_lock = threading.RLock()
        self._lock = threading.Lock()
        self._waiters: list[tuple[Callable[[], bool], tonio.Event]] = []

    def request_render(self, force: bool = False) -> None:
        with self._lock:
            waiters = list(self._waiters)
        for predicate, reached in waiters:
            if not reached.is_set() and predicate():
                reached.set()

    async def until(self, predicate: Callable[[], bool]) -> None:
        reached = tonio.Event()
        waiter = (predicate, reached)
        with self._lock:
            self._waiters.append(waiter)
        # Registered first, then checked: a render request in between is not missed.
        if predicate():
            reached.set()
        await reached.wait(5)
        with self._lock:
            self._waiters.remove(waiter)
        assert reached.is_set(), "the component never rendered the expected state"
