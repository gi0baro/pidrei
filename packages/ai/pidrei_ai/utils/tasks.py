"""Fan-out helpers over `tonio.spawn` with pi's `Promise.all` error shape.

`tonio.spawn(*coros)` reports child failures as
`ExceptionGroup("SpawnExceptionGroup", [...])`. pi's code awaits
`Promise.all`, whose rejection is the first failing promise's error itself,
and every caller (compaction errors, reload diagnostics, listing failures)
reports `str(error)`. `gather` keeps that contract: the first child failure
propagates bare, with the group attached as `__cause__` so nothing is lost.
"""

from collections.abc import Awaitable

import tonio.colored as tonio


def unwrap_spawn_error(error: BaseException) -> BaseException:
    """Return the first leaf failure inside nested `SpawnExceptionGroup`s."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return error


async def gather[T](*coros: Awaitable[T]) -> list[T]:
    """Run `coros` concurrently, returning their results in order.

    Raises the first child failure bare (pi's `Promise.all`), not the group.
    """
    if not coros:
        return []
    try:
        if len(coros) == 1:
            return [await tonio.spawn(coros[0])]
        return list(await tonio.spawn(*coros))
    except BaseExceptionGroup as group:
        leaf = unwrap_spawn_error(group)
        if leaf is group:
            raise
        raise leaf from group
