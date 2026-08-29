"""Shared harness for the server test suite.

Upstream's vitest files keep module-level `servers`/`clients`/`tempDirectories`
sets torn down in `afterEach`; here each
test owns a `Harness` and closes it in a `finally` block instead. Socket paths
live under the test's `tmp_path`; the listener itself creates the per-server
subdirectories (mirroring `mkdtemp` per server upstream).
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


class Harness:
    """Tracks servers and wire clients for one test; close in `finally`."""

    def __init__(self, tmp_path) -> None:
        self._tmp_dir = tmp_path
        self._sequence = 0
        self.servers: list[PiServer] = []
        self.clients: list[ProtocolTestClient] = []

    def socket_path(self, nested: bool = False) -> str:
        self._sequence += 1
        base = self._tmp_dir / f"srv{self._sequence}"
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
