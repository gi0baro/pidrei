"""Per-path file mutation serialization (port of pi `tools/file-mutation-queue.ts`).

pi serializes mutations with per-key promise chains plus a registration chain
that keeps key computation (async canonical-path lookup) FIFO. The tonio port
keeps the same shape: a tonio `Lock` serializes registration (the critical
section awaits `canonical_path`), and each queue tail is a tonio `Event` set
when its mutation settles. A mutation only runs after the tail it observed at
registration has fired, so mutations targeting the same environment and
canonical path never interleave — including when one of them fails or is
aborted.
"""

import threading
import weakref
from collections.abc import Awaitable, Callable

import tonio.colored as tonio
from tonio.colored import sync

from ..types import get_or_throw


class _MutationQueueState:
    def __init__(self) -> None:
        self.registration_lock = sync.Lock()
        self.queues: dict[str, tonio.Event] = {}
        self.queues_guard = threading.Lock()


_states: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_states_guard = threading.Lock()


def _get_state(env) -> _MutationQueueState:
    with _states_guard:
        state = _states.get(env)
        if state is None:
            state = _MutationQueueState()
            _states[env] = state
        return state


async def _get_mutation_queue_key(env, path: str) -> str:
    absolute_path = get_or_throw(await env.absolute_path(path))
    canonical_path = await env.canonical_path(absolute_path)
    if canonical_path.ok:
        return canonical_path.value
    if canonical_path.error.code in ("not_found", "not_supported"):
        return absolute_path
    raise canonical_path.error


async def with_file_mutation_queue[T](env, path: str, fn: Callable[[], Awaitable[T]]) -> T:
    """Serialize file mutations targeting the same environment and canonical path."""
    state = _get_state(env)
    async with state.registration_lock:
        key = await _get_mutation_queue_key(env, path)
        with state.queues_guard:
            current_tail = state.queues.get(key)
            my_done = tonio.Event()
            state.queues[key] = my_done

    if current_tail is not None:
        await current_tail.wait(None)
    try:
        return await fn()
    finally:
        my_done.set()
        with state.queues_guard:
            if state.queues.get(key) is my_done:
                del state.queues[key]
