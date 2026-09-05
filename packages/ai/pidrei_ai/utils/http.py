"""The HTTP-stack seam: the only pidrei module that imports punkreq or httpunk.

Adapters obtain clients and HTTP types exclusively from here, so alpha-stage
punkreq API churn stays contained in one file (see PLAN.md). The defaults
encode the LLM-streaming idiom: bound connect and per-chunk reads, never the
whole request — a legitimately long SSE stream must not hit a total deadline.

`h1_server` reaches one layer below punkreq for the serving side, which punkreq
(a client) does not expose.
"""

import sys
import threading
from collections.abc import AsyncIterable, Mapping
from typing import Any

import tonio.colored as tonio
from httpunk import Backend, H1Connection, H1Server
from punkreq import Limits, Timeout, TimeoutException
from punkreq.tonio import Client
from tonio.exceptions import CancelledError

from pidrei_ai.utils.http_proxy import resolve_http_proxy_url_for_target


STREAMING_TIMEOUT = Timeout(connect=30.0, read=600.0, pool=30.0, total=None)
DEFAULT_LIMITS = Limits(max_connections=64)

# Re-exported so callers can recognize a timeout without importing punkreq
# themselves (this module is the only one that may).
RequestTimeout = TimeoutException

_shared_client: Client | None = None
_shared_client_guard = threading.Lock()
_proxied_clients: dict[str, Client] = {}


def shared_client() -> Client:
    """Process-wide pooled client used by the API adapters."""
    global _shared_client
    with _shared_client_guard:
        if _shared_client is None:
            _shared_client = create_client()
        return _shared_client


def client_for(target_url: str, env: Mapping[str, str] | None = None) -> Client:
    """The pooled client for `target_url`, honouring provider-scoped proxy env.

    The shared client already picks up `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` from
    `os.environ` (punkreq `trust_env`). Only a *scoped* override — a proxy var in
    `options.env`, which punkreq cannot see — needs its own client, so one is
    pooled per distinct proxy URL rather than created per request.
    """
    if not env:
        return shared_client()
    proxy = resolve_http_proxy_url_for_target(target_url, env)
    if proxy is None:
        # An `env` that resolves to no proxy may still be a NO_PROXY override of
        # an ambient proxy, so the shared client is only correct when the process
        # env agrees.
        if resolve_http_proxy_url_for_target(target_url) is None:
            return shared_client()
        return _no_proxy_client()
    with _shared_client_guard:
        client = _proxied_clients.get(proxy)
        if client is None:
            client = create_client(proxy=proxy)
            _proxied_clients[proxy] = client
        return client


_NO_PROXY = "<none>"


def _no_proxy_client() -> Client:
    with _shared_client_guard:
        client = _proxied_clients.get(_NO_PROXY)
        if client is None:
            client = create_client(trust_env=False)
            _proxied_clients[_NO_PROXY] = client
        return client


_DRAIN_TIMEOUT_S = 1.0


def abandon_response(response: object) -> None:
    """Release a response from inside a cancelled chain.

    A scope cancel is delivered at the child's next suspension and no
    suspension after it is served, so a close awaited from the unwinding
    `finally` never completes. The close runs on its own task instead, which
    is not part of the cancelled chain. Injected test clients without a
    `close` are a no-op.
    """
    close = getattr(response, "close", None)
    if close is None:
        return

    async def _close() -> None:
        try:
            await close()
        except Exception:
            pass

    tonio.spawn.without_tracking(_close())


def _unwinding_from_cancel(cancel: Any) -> bool:
    if cancel is not None and cancel.cancelled:
        return True
    return isinstance(sys.exception(), CancelledError)


async def finish_body(body: AsyncIterable[bytes], response: object, *, drain: bool, cancel: Any = None) -> None:
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
    back to closing the response, which aborts the exchange. Error paths
    never drain — they abort. Cancel paths (the token fired, or the caller's
    `finally` is unwinding a `CancelledError`) abort without suspending: the
    close is handed to `abandon_response`, because nothing awaited from a
    cancelled chain runs. Injected test clients without a `close` are a
    no-op.
    """
    if _unwinding_from_cancel(cancel):
        abandon_response(response)
        return
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
    """Abort a streaming response (punkreq's idempotent release).

    ``close`` is async-only (punkreq's and every adapter's is); the guard is
    for its *absence* on injected test clients, not for a sync variant.
    """
    close = getattr(response, "close", None)
    if close is None:
        return
    await close()


def request_timeout(timeout_ms: float | None) -> Timeout:
    """Per-request timeout for streaming LLM calls.

    pi forwards `timeoutMs` to the SDK's whole-request timeout; for streaming
    the practical bound is per-chunk idleness, so it maps to `read` here while
    `total` stays disabled (a legitimately long stream must not be cut off).
    """
    if timeout_ms is None:
        return STREAMING_TIMEOUT
    return Timeout(connect=30.0, read=timeout_ms / 1000, pool=30.0, total=None)


def h1_server(transport: Any) -> H1Server:
    """A server-side HTTP/1 connection over an already-accepted tonio socket.

    The OAuth callback servers' side of the seam. httpunk owns the protocol —
    head parsing, keep-alive, the automatic 400/414/431 for a malformed or
    oversized head, `Expect: 100-continue`, the slowloris header-read timeout —
    and the caller owns accepting the socket, the same split
    `hyper::server::conn::http1` makes. A tonio `SocketStream` is already the
    transport httpunk wants: `receive_some`/`send_all`/`close`.
    """
    return H1Server(transport, backend=Backend.tonio)


async def h1_client_upgrade(
    transport: Any,
    target: str,
    headers: Mapping[str, str],
) -> tuple[int, dict[str, str], Any | None]:
    """Client HTTP/1 request expecting a `101 Switching Protocols` upgrade.

    Returns `(status, response headers, raw upgraded stream or None)`. The
    WebSocket side of the seam (see `utils/websocket.py`): httpunk owns the
    handshake's HTTP — writing the head, parsing the response, and handing back
    an `H1Upgraded` whose reads "first drain any bytes already received past the
    response head", the classic handshake footgun. The caller owns the transport
    from there and never touches the `H1Connection` again (the driver detaches on
    upgrade, so closing the connection would close the caller's stream).

    punkreq does not expose upgrades, so this reaches one layer below it, the
    same split `h1_server` makes for the serving side.
    """
    connection = H1Connection(transport, backend=Backend.tonio)
    response = await connection.request("GET", target, headers=headers)
    response_headers = {key: bytes(value).decode("latin-1") for key, value in response.headers.items()}
    upgraded = response.upgraded
    if upgraded is None:
        await response.aclose()
    return response.status, response_headers, upgraded


def oneshot_timeout(timeout_ms: float | None) -> Timeout:
    """Whole-request bound for the non-streaming calls (OAuth token exchanges).

    The streaming default deliberately leaves `total` unset because a long SSE
    stream is legitimate; a token exchange is one small round trip, so pi's
    `AbortSignal.timeout(...)` maps to `total` here.
    """
    if timeout_ms is None:
        return STREAMING_TIMEOUT
    seconds = timeout_ms / 1000
    return Timeout(connect=min(30.0, seconds), read=seconds, pool=min(30.0, seconds), total=seconds)


def create_client(
    *,
    base_url: str = "",
    headers: Mapping[str, str] | None = None,
    proxy: str | None = None,
    verify: bool = True,
    timeout: Timeout = STREAMING_TIMEOUT,
    limits: Limits = DEFAULT_LIMITS,
    trust_env: bool = True,
) -> Client:
    return Client(
        base_url=base_url,
        headers=headers,
        proxy=proxy,
        verify=verify,
        timeout=timeout,
        limits=limits,
        trust_env=trust_env,
    )
