"""Mirror of pi coding-agent src/core/tools/file-mutation-queue.ts.

pi chains promises per real path; here each key holds a tonio Event tail
(same pattern as the agent-package queue, but keyed globally since these
tools run directly on the local filesystem).
"""

import os
import threading

import tonio.colored as tonio


_queues: dict[str, tonio.Event] = {}
_guard = threading.Lock()


def _mutation_queue_key(file_path: str) -> str:
    resolved = os.path.abspath(file_path)
    try:
        return os.path.realpath(resolved, strict=True)
    except FileNotFoundError, NotADirectoryError:
        return resolved


def resolve_mutation_queue_key(file_path: str):
    """Resolve the queue key off the runtime.

    `_mutation_queue_key` calls `realpath`, which is filesystem I/O, and
    `with_file_mutation_queue` cannot do it itself: registration has to stay
    synchronous (see below), so there is nowhere in it to await. Async callers
    resolve the key here first and hand it in.
    """
    return tonio.spawn_blocking(_mutation_queue_key, file_path)


def with_file_mutation_queue(file_path: str, fn, *, queue_key: str | None = None):
    """Serialize file mutation operations targeting the same file.
    Operations for different files still run in parallel.

    Registration happens synchronously at call time (pi chains the promise in
    call order); the returned coroutine waits its turn when awaited. That is
    load-bearing — the ordering tests rely on both registrations completing
    during argument evaluation, before the tasks are scheduled — so this must
    not become a coroutine.

    `queue_key` is the pre-resolved key from `resolve_mutation_queue_key`.
    Without it the key is resolved inline, which touches the filesystem and is
    only acceptable off the runtime.
    """
    key = queue_key if queue_key is not None else _mutation_queue_key(file_path)
    done = tonio.Event()
    with _guard:
        previous = _queues.get(key)
        _queues[key] = done

    async def run():
        if previous is not None:
            await previous.wait()
        try:
            return await fn()
        finally:
            with _guard:
                if _queues.get(key) is done:
                    del _queues[key]
            done.set()

    return run()
