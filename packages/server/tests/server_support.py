"""Shared harness for the server test suite.

Upstream's vitest files keep module-level `servers`/`clients`/`tempDirectories`
sets torn down in `afterEach`; here each
test owns a `Harness` and closes it in a `finally` block instead. Socket paths
live under the test's `sock_dir` (a deliberately short directory — see the
conftest fixture: deep `tmp_path` paths overflow `sun_path` on macOS); the
listener itself creates the per-server subdirectories (mirroring `mkdtemp`
per server upstream).
"""

import tonio.colored as tonio

from pidrei_server.server import PiServer
from pidrei_server.testing import ProtocolTestClient, TestServerService, connect_unix_test_client
from pidrei_server.transports.unix import UnixServerOptions, create_unix_server


async def flush(turns: int = 4) -> None:
    """Let spawned tasks progress (stand-in for JS microtask turns).

    A small positive sleep, not `sleep(0)` — see the note on the client
    package's copy of this helper: a zero sleep is not a guaranteed
    reschedule in tonio.
    """
    for _ in range(turns):
        await tonio.sleep(0.005)


async def settled(awaitable, what: str = "the awaited step", timeout: float = 5.0):
    """Await with a bound: a wedged choreography step fails here, with a name
    and this call site's line in the traceback, instead of parking forever
    and hanging the whole CI job (seen on linux and macOS 3.14t).

    Takes a `Deferred` (awaited via `wait()`) or any plain awaitable — the
    wire client's `next()`/`next_from()` return Deferreds while `request()`/
    `hello()` return coroutines/awaitables.
    """
    wait = awaitable.wait() if hasattr(awaitable, "wait") else awaitable
    value, completed = await tonio.time.timeout(wait, timeout)
    assert completed, f"timed out waiting for {what}"
    return value


class Harness:
    """Tracks servers and wire clients for one test; close in `finally`."""

    def __init__(self, sock_dir) -> None:
        self._sock_dir = sock_dir
        self._sequence = 0
        self.servers: list[PiServer] = []
        self.clients: list[ProtocolTestClient] = []

    def socket_path(self, nested: bool = False) -> str:
        self._sequence += 1
        base = self._sock_dir / f"srv{self._sequence}"
        return str(base / "p" / "n" / "server.sock" if nested else base / "server.sock")

    def make_server(self, path: str, service: TestServerService | None = None, **overrides) -> PiServer:
        server = create_unix_server(
            service if service is not None else TestServerService(), UnixServerOptions(path=path, **overrides)
        )
        self.servers.append(server)
        return server

    async def start_server(
        self, service: TestServerService | None = None, **overrides
    ) -> tuple[PiServer, TestServerService]:
        resolved_service = service if service is not None else TestServerService()
        server = self.make_server(self.socket_path(), resolved_service, **overrides)
        await server.start()
        return server, resolved_service

    async def connect(self, server: PiServer) -> ProtocolTestClient:
        client = await connect_unix_test_client(server.addresses[0])
        self.clients.append(client)
        return client

    async def close(self) -> None:
        for client in self.clients:
            await client.close()
        self.clients.clear()
        for server in self.servers:
            await server.close()
        self.servers.clear()
