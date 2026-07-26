"""The punkreq seam: the only pidrei module that imports punkreq.

Adapters obtain clients and HTTP types exclusively from here, so alpha-stage
punkreq API churn stays contained in one file (see PLAN.md). The defaults
encode the LLM-streaming idiom: bound connect and per-chunk reads, never the
whole request — a legitimately long SSE stream must not hit a total deadline.
"""

import inspect
import threading
from collections.abc import AsyncGenerator, AsyncIterable, Mapping

import tonio.colored as tonio
from punkreq import Limits, Timeout
from punkreq.tonio import Client

from pidrei_ai.utils.cancel import CancelToken


STREAMING_TIMEOUT = Timeout(connect=30.0, read=600.0, pool=30.0, total=None)
DEFAULT_LIMITS = Limits(max_connections=64)

_shared_client: Client | None = None
_shared_client_guard = threading.Lock()


def shared_client() -> Client:
    """Process-wide pooled client used by the API adapters."""
    global _shared_client
    with _shared_client_guard:
        if _shared_client is None:
            _shared_client = create_client()
        return _shared_client


_STREAM_DONE = object()
_STREAM_CANCELLED = object()


async def cancellable_bytes(source: AsyncIterable[bytes], cancel: CancelToken | None) -> AsyncGenerator[bytes]:
    """Yield chunks, aborting a pending read when the token cancels.

    Mirrors pi's fetch-abort semantics: each chunk read races the cancel token
    via `tonio.select`, so a hung read is genuinely interruptible and the
    transport's cancel-safe teardown releases the connection.
    """
    if cancel is None:
        async for chunk in source:
            yield chunk
        return

    iterator = aiter(source)
    while True:
        if cancel.cancelled:
            raise RuntimeError("Request was aborted")

        async def _next() -> object:
            try:
                return await anext(iterator)
            except StopAsyncIteration:
                return _STREAM_DONE

        async def _aborted() -> object:
            await cancel.wait()
            return _STREAM_CANCELLED

        winner = await tonio.select(_next(), _aborted())
        if winner is _STREAM_CANCELLED:
            raise RuntimeError("Request was aborted")
        if winner is _STREAM_DONE:
            return
        yield winner  # type: ignore[misc]


_DRAIN_TIMEOUT_S = 1.0


async def finish_body(body: AsyncIterable[bytes], response: object, *, drain: bool) -> None:
    """Settle the transport body an adapter stopped reading.

    Adapters stop at the terminal SSE event (`[DONE]`, `message_stop`), which
    leaves the transport's body generator suspended. Async generators are
    only finalized by `aclose()` or the garbage collector, and GC
    finalization cannot await — an abandoned body printed "async generator
    ignored GeneratorExit" on the user's terminal mid-turn (found by the boot
    smoke test) and held the pooled connection until GC ran.

    An SSE body ends immediately after its terminal event, so draining it
    lets the transport's own tail run (punkreq's `iter_raw` releases the
    response there) and nothing is left for the GC. The drain is bounded: a
    provider that keeps the connection open past the terminal event falls
    back to closing the response, which aborts the exchange. Error and cancel
    paths never drain — they abort. Injected test clients without a `close`
    are a no-op.
    """
    if drain:
        try:
            _result, completed = await tonio.time.timeout(_drain_body(body), _DRAIN_TIMEOUT_S)
            if completed:
                return
        except Exception:
            return
    await close_response(response)


async def _drain_body(body: AsyncIterable[bytes]) -> None:
    try:
        async for _chunk in body:
            pass
    except Exception:
        pass


async def close_response(response: object) -> None:
    """Abort a streaming response (punkreq's idempotent release)."""
    close = getattr(response, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def request_timeout(timeout_ms: float | None) -> Timeout:
    """Per-request timeout for streaming LLM calls.

    pi forwards `timeoutMs` to the SDK's whole-request timeout; for streaming
    the practical bound is per-chunk idleness, so it maps to `read` here while
    `total` stays disabled (a legitimately long stream must not be cut off).
    """
    if timeout_ms is None:
        return STREAMING_TIMEOUT
    return Timeout(connect=30.0, read=timeout_ms / 1000, pool=30.0, total=None)


def create_client(
    *,
    base_url: str = "",
    headers: Mapping[str, str] | None = None,
    proxy: str | None = None,
    verify: bool = True,
    timeout: Timeout = STREAMING_TIMEOUT,
    limits: Limits = DEFAULT_LIMITS,
) -> Client:
    return Client(
        base_url=base_url,
        headers=headers,
        proxy=proxy,
        verify=verify,
        timeout=timeout,
        limits=limits,
    )
