"""Port of pi's utils/sleep.ts."""

from pidrei_ai.utils import clock
from pidrei_ai.utils.cancel import AbortError, CancelToken


async def sleep(ms: float, cancel: CancelToken) -> None:
    """pi's abortable sleep rejects with the signal's reason."""
    try:
        await clock.sleep_ms(ms, cancel)
    except AbortError:
        cancel.raise_if_cancelled()
        raise
