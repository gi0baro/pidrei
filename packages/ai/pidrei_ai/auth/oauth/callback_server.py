"""The loopback HTTP server the browser OAuth flows redirect back to.

pi calls `node:http`'s `createServer` inline in each flow. The equivalent split
here is `httpunk.H1Server` (through the seam) for the protocol and this module
for the accept loop — which is the division httpunk documents, and which pi's
`node:http` merely hides. What is left is the part where all three of pi's
servers actually differ, and that is what each flow supplies as a handler: fixed
vs ephemeral port, state check, one-shot claim, when the response is written
relative to the token exchange.

*Deviation:* a handler returns its response instead of writing it, so the page
is written *after* the handler settles the flow, where pi writes it first. That
is safe because closing the listener does not drop accepted connections — the
browser still gets its page — but a flow must not assume the browser has been
answered by the time its own future resolves.
"""

import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlsplit

import tonio.colored as tonio
from tonio.colored import net

from pidrei_ai.utils import http


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
    """Handle over the listening socket (pi holds the node `Server`)."""

    __slots__ = ("_listener", "port")

    def __init__(self, listener: Any, port: int):
        self._listener = listener
        self.port = port

    def close(self) -> None:
        """Stop accepting connections (node's `server.close()`)."""
        self._listener.close()


def _to_callback_request(request: Any) -> CallbackRequest:
    """An `httpunk.h1.ServerRequest` as the flows' request record.

    `target` is the request-target, so the query still needs splitting off — the
    one thing `new URL(req.url, "http://localhost")` did for pi.
    """
    split = urlsplit(request.target)
    query = {name: values[0] for name, values in parse_qs(split.query, keep_blank_values=True).items()}
    return CallbackRequest(method=request.method, path=split.path, query=query)


async def _serve_connection(stream: Any, handle: CallbackHandler) -> None:
    try:
        async with http.h1_server(stream) as server:
            async for request in server:
                await request.read()  # an OAuth redirect carries no body; drain for keep-alive
                response = await handle(_to_callback_request(request))
                await request.respond(
                    response.status,
                    headers=_RESPONSE_HEADERS,
                    body=response.html.encode("utf-8"),
                )
    except Exception:
        # pi answers a handler crash with a 500 and keeps the server alive; a
        # dead socket is the other half of that, and neither can be reported to
        # the flow, which is waiting on its own future.
        pass


async def start_callback_server(*, host: str, port: int, handle: CallbackHandler) -> CallbackServer:
    """Listen on `host:port` (0 for an ephemeral port) and serve `handle`.

    Raises whatever the bind raises — a port already in use is a condition the
    flows decide about (anthropic surfaces it, openai-codex falls back).
    """
    listeners = await net.open_tcp_listeners(port, host=host)
    listener = listeners[0]
    for extra in listeners[1:]:  # pragma: no cover - a single host binds once
        extra.close()
    bound_port = listener.socket.getsockname()[1]

    async def _accept_loop() -> None:
        while True:
            try:
                stream = await listener.accept()
            except Exception:
                return
            tonio.spawn.without_tracking(_serve_connection(stream, handle))

    tonio.spawn.without_tracking(_accept_loop())
    return CallbackServer(listener, bound_port)
