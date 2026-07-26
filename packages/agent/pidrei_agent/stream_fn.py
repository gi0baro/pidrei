"""Default stream-function registry (port of pi `agent/src/stream-fn.ts`)."""

from .types import StreamFn


_default_stream_fn: StreamFn | None = None


def set_default_stream_fn(stream_fn: StreamFn | None) -> None:
    """Configure the fallback used by Agent and low-level loops when callers omit `stream_fn`.

    Hosts that provide a default model runtime can install its stream function
    here without making pidrei-agent depend on a provider catalog.
    """
    global _default_stream_fn
    _default_stream_fn = stream_fn


def get_default_stream_fn() -> StreamFn:
    if _default_stream_fn is None:
        raise Exception(
            "No default stream function configured. Pass stream_fn explicitly or call set_default_stream_fn()."
        )
    return _default_stream_fn
