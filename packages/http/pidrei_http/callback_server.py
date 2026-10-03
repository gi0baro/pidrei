"""The loopback HTTP server OAuth flows redirect back to: the lower layer of
pi's packages/ai/src/auth/oauth/callback-server.ts.

This is what pi gets from `node:http`'s `createServer`: `httpunk.H1Server`
(through the seam) for the protocol and the accept loop below, serving a
handler that returns its response. The provider flows' shared handler
(`start_oauth_callback_server`) lives over it in
`pidrei_ai.auth.oauth.callback_server`; the flows that keep their own handler,
as pi keeps its own `createServer` there, use this layer directly.
"""

import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlsplit

import tonio.colored as tonio
from tonio.colored import net

from pidrei_http import http


_RESPONSE_HEADERS = {
    "content-type": "text/html; charset=utf-8",
    "cache-control": "no-store",
}


@dataclass(slots=True)
class CallbackRequest:
    method: str
    path: str
    query: dict[str, str] = field(default_factory=dict)

    def get(self, name: str) -> str | None:
        """`URL.searchParams.get`: the first value, or None."""
        return self.query.get(name)


@dataclass(slots=True)
class CallbackResponse:
    status: int
    html: str
    # Runs once the response is written, or the write failed (pi's code after
    # `res.end()`). A flow that settles its result from the handler does it
    # here: the waiter wakes on another thread and may drop this connection.
    after_sent: Callable[[], None] | None = None
    # The response headers; None is the flows' HTML page. `html` is then the
    # body in whatever type these name (MCP's default page is plain text).
    headers: Mapping[str, str] | None = None


CallbackHandler = Callable[[CallbackRequest], Awaitable[CallbackResponse]]


class OneShotValue:
    """A promise settled at most once, from any task.

    pi builds this inline in each flow out of a captured `resolve` and a
    `settled` flag; those flows run on one thread, this one does not, so the
    flag is behind a lock.
    """

    __slots__ = ("_event", "_lock", "_value")

    def __init__(self) -> None:
        self._event = tonio.Event()
        self._lock = threading.Lock()
        self._value: Any = None

    @property
    def settled(self) -> bool:
        return self._event.is_set()

    def settle(self, value: Any = None) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._value = value
            self._event.set()

    async def wait(self) -> Any:
        await self._event.wait()
        return self._value

    async def wait_for(self, timeout: float) -> bool:
        """Wait up to `timeout` seconds; True when settled (pi's `setTimeout` guard)."""
        await self._event.wait(timeout)
        return self._event.is_set()


class CallbackServer:
    """Handle over the listening socket and its accepted connections (pi holds
    the node `Server`)."""

    __slots__ = ("_connections", "_connections_closed", "_listener", "_lock", "port")

    def __init__(self, listener: Any, port: int):
        self._listener = listener
        self.port = port
        self._lock = threading.Lock()
        self._connections: set[Any] = set()
        self._connections_closed = False

    def close(self) -> None:
        """Stop accepting connections (node's `server.close()`); accepted ones
        keep being served."""
        self._listener.close()

    def close_all_connections(self) -> None:
        """node's `server.closeAllConnections()`, called after `close()`: drop
        every accepted connection, idle or not. A connection accepted
        concurrently is dropped as soon as it registers."""
        with self._lock:
            self._connections_closed = True
            connections, self._connections = self._connections, set()
        for stream in connections:
            stream.close()

    def _track(self, stream: Any) -> bool:
        with self._lock:
            if self._connections_closed:
                return False
            self._connections.add(stream)
            return True

    def _untrack(self, stream: Any) -> None:
        with self._lock:
            self._connections.discard(stream)


def _to_callback_request(request: Any) -> CallbackRequest:
    """An `httpunk.h1.ServerRequest` as the flows' request record.

    `target` is the request-target, so the query still needs splitting off — the
    one thing `new URL(req.url, "http://localhost")` did for pi.
    """
    split = urlsplit(request.target)
    query = {name: values[0] for name, values in parse_qs(split.query, keep_blank_values=True).items()}
    return CallbackRequest(method=request.method, path=split.path, query=query)


async def _serve_connection(server: CallbackServer, stream: Any, handle: CallbackHandler) -> None:
    if not server._track(stream):
        stream.close()
        return
    try:
        async with http.h1_server(stream) as connection:
            async for request in connection:
                await request.read()  # an OAuth redirect carries no body; drain for keep-alive
                response = await handle(_to_callback_request(request))
                try:
                    await request.respond(
                        response.status,
                        headers=_RESPONSE_HEADERS if response.headers is None else response.headers,
                        body=response.html.encode("utf-8"),
                    )
                finally:
                    if response.after_sent is not None:
                        response.after_sent()
    except Exception:
        # pi answers a handler crash with a 500 and keeps the server alive; a
        # dead socket is the other half of that, and neither can be reported to
        # the flow, which is waiting on its own future.
        pass
    finally:
        server._untrack(stream)


async def start_callback_server(*, host: str, port: int, handle: CallbackHandler) -> CallbackServer:
    """Listen on `host:port` (0 for an ephemeral port) and serve `handle`.

    Raises whatever the bind raises — a port already in use is a condition the
    flows decide about (anthropic surfaces it, openai-codex falls back).
    """
    listeners = await net.open_tcp_listeners(port, host=host)
    listener = listeners[0]
    for extra in listeners[1:]:  # pragma: no cover - a single host binds once
        extra.close()
    server = CallbackServer(listener, listener.socket.getsockname()[1])

    async def _accept_loop() -> None:
        while True:
            try:
                stream = await listener.accept()
            except Exception:
                return
            tonio.spawn.without_tracking(_serve_connection(server, stream, handle))

    tonio.spawn.without_tracking(_accept_loop())
    return server
