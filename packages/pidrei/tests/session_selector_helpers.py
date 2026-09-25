"""Shared test seam for the session selector's pidrei-only `post_ui` argument."""

import threading
from collections.abc import Callable

import tonio.colored as tonio


class PostedUpdates:
    """Stands in for `TUI.post_ui`: no UI owner runs in these tests, so each posted
    selector update runs inline. Every selector state change arrives through a post
    (loads, progress, deletes and renames finish on detached tasks), so a test waits
    for the state it needs with `until` instead of sleeping."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waiters: list[tuple[Callable[[], bool], tonio.Event]] = []

    def __call__(self, fn: Callable[[], None]) -> None:
        fn()
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
        # Registered first, then checked: a post landing in between is not missed.
        if predicate():
            reached.set()
        await reached.wait(5)
        with self._lock:
            self._waiters.remove(waiter)
        assert reached.is_set(), "the selector never reached the expected state"


def lists_sessions(selector, sessions) -> Callable[[], bool]:
    """The session list has applied `sessions` (the load, then the canonical-path map)."""
    return lambda: selector.get_session_list()._all_sessions == sessions
