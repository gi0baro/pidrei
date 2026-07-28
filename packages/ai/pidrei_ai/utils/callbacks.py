"""Shared invoker for optional async-only callbacks.

Callback contracts are async-only (policy decided 2026-07-28): a slot holds
either None or an awaitable-returning callable. This is the one place the
None-guard lives, so optional-callback call sites stay a single expression.
"""

from collections.abc import Awaitable, Callable
from typing import Any


async def maybe_call(callback: Callable[..., Awaitable[Any]] | None, *args: Any) -> Any:
    """Await ``callback(*args)`` if the optional callback is set; else None."""
    if callback is None:
        return None
    return await callback(*args)
