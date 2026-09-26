"""Shared test seams for the session selector's pidrei-only `state_lock` and
`finish_before_next_input` arguments."""

import threading
from collections.abc import Callable
from typing import Self

import tonio.colored as tonio


class StateUpdates:
    """Stands in for `TUI.state_lock`: a reentrant lock that re-checks the
    waiters (under the lock) each time an outermost hold ends. Every selector
    state change happens in a hold (loads, progress, deletes and renames finish
    on detached tasks), so a test waits for the state it needs with `until`
    instead of sleeping."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._depth = 0
        self._waiters: list[tuple[Callable[[], bool], tonio.Event]] = []

    def __enter__(self) -> Self:
        self._lock.acquire()
        self._depth += 1
        return self

    def __exit__(self, *_exc) -> None:
        self._depth -= 1
        if self._depth == 0:
            for predicate, reached in list(self._waiters):
                if not reached.is_set() and predicate():
                    reached.set()
        self._lock.release()

    async def until(self, predicate: Callable[[], bool]) -> None:
        reached = tonio.Event()
        waiter = (predicate, reached)
        # Registered and checked in one hold: a change landing in between is
        # not missed.
        with self._lock:
            self._waiters.append(waiter)
            if predicate():
                reached.set()
        await reached.wait(5)
        with self._lock:
            self._waiters.remove(waiter)
        assert reached.is_set(), "the selector never reached the expected state"


class InputCompletions:
    """Stands in for `TUI.finish_before_next_input`: the input consumer awaits
    what a key registered before it handles the next key, and `press` does the
    same for a key sent straight to a component."""

    def __init__(self) -> None:
        self._pending: list = []

    def __call__(self, task) -> None:
        self._pending.append(task)

    async def press(self, component, data: str) -> None:
        component.handle_input(data)
        pending, self._pending = self._pending, []
        for task in pending:
            await task


def lists_sessions(selector, sessions) -> Callable[[], bool]:
    """The session list has applied `sessions` (the load, then the canonical-path map)."""
    return lambda: selector.get_session_list()._all_sessions == sessions
