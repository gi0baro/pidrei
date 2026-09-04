"""Shared harness for the server transport tests.

Upstream's vitest files keep module-level `servers`/`clients`/`tempDirectories`
sets torn down in `afterEach`; here each test owns a `Harness` and closes it
in a `finally` block instead. Socket paths live under the test's `sock_dir`
(a deliberately short directory — see the conftest fixture: deep `tmp_path`
paths overflow `sun_path` on macOS); the listener itself creates the
per-server subdirectories (mirroring `mkdtemp` per server upstream).

The protocol server is not ported (UPSTREAM_EXPERIMENTAL_RULING.md), so the
listener is driven directly: the accept handler echoes bytes and a raw socket
round-trip stands in for the hello handshake as the liveness probe.
"""

import tonio.colored as tonio
from tonio.colored import net

from pidrei_server.connection import ByteConnection, ByteConnectionHandler
from pidrei_server.listener import PiServerListener
from pidrei_server.transports.unix import UnixListenerOptions, create_unix_listener


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

    Takes a `Deferred` (awaited via `wait()`) or any plain awaitable.
    """
    wait = awaitable.wait() if hasattr(awaitable, "wait") else awaitable
    value, completed = await tonio.time.timeout(wait, timeout)
    assert completed, f"timed out waiting for {what}"
    return value


def echo_acceptor(connection: ByteConnection) -> ByteConnectionHandler:
    """Accept handler that writes every inbound chunk straight back."""
    return ByteConnectionHandler(
        on_data=lambda chunk: connection.send(chunk),
        on_close=lambda: None,
        on_error=lambda error: None,
    )


class Harness:
    """Tracks listeners for one test; close in `finally`."""

    def __init__(self, sock_dir) -> None:
        self._sock_dir = sock_dir
        self._sequence = 0
        self.listeners: list[PiServerListener] = []

    def socket_path(self, nested: bool = False) -> str:
        self._sequence += 1
        base = self._sock_dir / f"srv{self._sequence}"
        return str(base / "p" / "n" / "server.sock" if nested else base / "server.sock")

    def make_listener(self, path: str, **overrides) -> PiServerListener:
        listener = create_unix_listener(UnixListenerOptions(path=path, **overrides))
        self.listeners.append(listener)
        return listener

    async def roundtrip(self, path: str, payload: bytes = b"ping") -> bytes:
        """Connect to a listener started with `echo_acceptor` and read the echo back."""
        stream = await net.open_unix_socket(path)
        try:
            await stream.send_all(payload)
            received = b""
            while len(received) < len(payload):
                chunk = await stream.receive_some()
                if not chunk:
                    break
                received += chunk
            return received
        finally:
            stream.close()

    async def close(self) -> None:
        for listener in self.listeners:
            await listener.close()
        self.listeners.clear()
