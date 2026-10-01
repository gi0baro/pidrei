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


def _mutation_queue_key_blocking(file_path: str) -> str:
    resolved = os.path.abspath(file_path)
    try:
        return os.path.realpath(resolved, strict=True)
    except FileNotFoundError, NotADirectoryError:
        return resolved


def resolve_mutation_queue_key(file_path: str):
    """Resolve the queue key off the runtime.

    `_mutation_queue_key_blocking` calls `realpath`, which is filesystem I/O, and
    `with_file_mutation_queue` cannot do it itself: registration has to stay
    synchronous (see below), so there is nowhere in it to await. Async callers
    resolve the key here first and hand it in.
    """
    return tonio.spawn_blocking(_mutation_queue_key_blocking, file_path)


def with_file_mutation_queue(file_path: str, fn, *, queue_key: str):
    """Serialize file mutation operations targeting the same file.
    Operations for different files still run in parallel.

    Registration happens synchronously at call time (pi chains the promise in
    call order). That is load-bearing — the ordering tests rely on both
    registrations completing during argument evaluation, before the tasks are
    scheduled — so this must not become a coroutine.

    The mutation starts at call time too, detached: as in pi, once it is in
    line it waits its turn and runs to the end whatever happens to the caller,
    so a place in line is never left behind and a cancelled caller cannot tear
    a write. The returned coroutine only waits for its outcome.

    `queue_key` is `await resolve_mutation_queue_key(file_path)`: resolving it
    is filesystem I/O, which registration cannot await. pi's signature has no
    such argument.
    """
    key = queue_key
    done = tonio.Event()
    outcome = tonio.Result()  # (failed, value or error), stored by `run`
    with _guard:
        previous = _queues.get(key)
        _queues[key] = done

    async def run() -> None:
        if previous is not None:
            await previous.wait()
        try:
            outcome.store((False, await fn()))
        except Exception as error:
            outcome.store((True, error))
        finally:
            with _guard:
                if _queues.get(key) is done:
                    del _queues[key]
            done.set()

    async def join():
        await done.wait()
        failed, value = outcome.fetch()
        if failed:
            raise value
        return value

    tonio.spawn.without_tracking(run())
    return join()
