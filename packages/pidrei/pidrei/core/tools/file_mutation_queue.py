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


def with_file_mutation_queue(file_path: str, fn):
    """Serialize file mutation operations targeting the same file.
    Operations for different files still run in parallel.

    Registration happens synchronously at call time (pi chains the promise in
    call order); the returned coroutine waits its turn when awaited.
    """
    key = _mutation_queue_key(file_path)
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
