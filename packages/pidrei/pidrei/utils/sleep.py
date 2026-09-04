"""Mirror of pi coding-agent src/utils/sleep.ts (AbortSignal -> CancelToken)."""

import tonio.colored as tonio


async def sleep(ms: float, cancel=None) -> None:
    """Sleep for `ms` milliseconds; raise Exception("Aborted") when cancelled."""
    if cancel is not None and cancel.cancelled:
        raise Exception("Aborted")

    if cancel is None:
        await tonio.time.sleep(ms / 1000)
        return

    await cancel.wait(ms / 1000)
    if cancel.cancelled:
        raise Exception("Aborted")
