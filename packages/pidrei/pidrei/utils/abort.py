"""Mirror of pi coding-agent src/utils/abort.ts.

pi-ai's twin (`pidrei_ai.utils.abort`) requires a token; this one accepts an
optional token — a None token awaits the operation directly.
"""

from collections.abc import Coroutine
from typing import Any

from pidrei_ai.utils.abort import operation_cancel, race_with_cancel as _race_with_required_cancel
from pidrei_ai.utils.cancel import CancelToken


__all__ = ["operation_cancel", "race_with_cancel"]


async def race_with_cancel[T](operation: Coroutine[Any, Any, T], cancel: CancelToken | None) -> T:
    """Stop waiting on cancellation while observing the abandoned operation
    through settlement."""
    if cancel is None:
        return await operation
    return await _race_with_required_cancel(operation, cancel)
