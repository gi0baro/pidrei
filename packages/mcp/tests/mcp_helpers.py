"""Mirror of pi mcp test/helpers.ts: loopback HTTP servers for the transport
and OAuth tests.

pi's are node `http` servers; these are httpunk's (through the pidrei-http
seam), served one connection per coroutine. `loopback_servers` gives a test
its servers and a shared client without keep-alive, so a connection ends with
its exchange. When the test ends the servers drop whatever connections are
left, as pi's helper does, and wait for every coroutine they started. A
handler that holds a response open waits on `request.peer_closed()`, which
resolves once the client hangs up.

It is entered inside the test body, never from a fixture: the runtime shuts
down every I/O registration when a `run_until_complete` returns, and a
fixture's setup, the test body and the fixture's teardown are separate runs.
A coroutine parked on a socket across that boundary (an accept loop, a
connection reader, a client's idle watcher) resumes to spin on a dead
registration, and the whole runtime stalls with it.
"""

import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import tonio.colored as tonio
from tonio.colored import net

from pidrei_http import http


type HttpHandler = Callable[[Any, str], Awaitable[None]]

_CLOSE_WAIT_S = 5.0


def header(request: Any, name: str) -> str | None:
    """A request header as text, None when absent."""
    value = request.headers.get(name)
    return None if value is None else bytes(value).decode("latin-1")


def headers_of(request: Any) -> dict[str, str]:
    return {bytes(key).decode("latin-1"): bytes(value).decode("latin-1") for key, value in request.headers.raw_items()}


async def read_body(request: Any) -> str:
    return (await request.read()).decode("utf-8")


async def read_json(request: Any) -> dict[str, Any]:
    return json.loads(await read_body(request))


class _Server:
    def __init__(self, listener: Any, handler: HttpHandler) -> None:
        self.listener = listener
        self.handler = handler
        self.origin = f"http://127.0.0.1:{listener.socket.getsockname()[1]}"
        # Every accepted connection, with the Event its coroutine sets when it ends.
        self.connections: list[tuple[Any, tonio.Event]] = []
        self.stopped = tonio.Event()

    async def serve(self, stream: Any, done: tonio.Event) -> None:
        try:
            async with http.h1_server(stream) as server:
                async for request in server:
                    try:
                        await self.handler(request, self.origin)
                    except Exception:
                        break
        except Exception:
            # The connection was dropped under a read: by its client, or by `close()`.
            pass
        finally:
            done.set()

    async def accept(self) -> None:
        try:
            while True:
                try:
                    stream = await self.listener.accept()
                except Exception:
                    return
                done = tonio.Event()
                self.connections.append((stream, done))
                tonio.spawn.without_tracking(self.serve(stream, done))
        finally:
            self.stopped.set()

    async def close(self) -> None:
        """Stop accepting, drop every connection (pi's `closeAllConnections`)
        and wait for the accept loop and each connection's coroutine to end:
        none of them may still be parked on its socket when the test body
        returns. A connection that never carried a request (a client that
        gave up while connecting) is dropped like any other."""
        self.listener.close()
        await self._ended(self.stopped, "its accept loop")
        for stream, done in self.connections:
            if not done.is_set():
                stream.close()
        for _stream, done in self.connections:
            await self._ended(done, "a connection")

    async def _ended(self, event: tonio.Event, what: str) -> None:
        await event.wait(_CLOSE_WAIT_S)
        if not event.is_set():
            raise AssertionError(f"the test server at {self.origin} did not shut down: {what} is still running")


class HttpServers:
    """`await servers.listen(handler)` starts a server and returns its origin;
    every server is shut down when the test ends."""

    def __init__(self) -> None:
        self._servers: list[_Server] = []

    async def listen(self, handler: HttpHandler) -> str:
        listeners = await net.open_tcp_listeners(0, host="127.0.0.1")
        for extra in listeners[1:]:
            extra.close()
        server = _Server(listeners[0], handler)
        self._servers.append(server)
        tonio.spawn.without_tracking(server.accept())
        return server.origin

    async def close(self) -> None:
        servers, self._servers = self._servers, []
        for server in servers:
            await server.close()


@contextlib.asynccontextmanager
async def loopback_servers(monkeypatch) -> AsyncIterator[HttpServers]:
    """Loopback test servers for one test body, and a shared HTTP client that
    keeps no idle connection; the servers are shut down (every connection
    dropped, every server coroutine ended), then the client is closed, before
    the body returns."""
    client = http.create_client(limits=http.Limits(max_keepalive_connections=0))
    monkeypatch.setattr(http, "shared_client", lambda: client)
    servers = HttpServers()
    try:
        yield servers
    finally:
        await servers.close()
        await client.close()
