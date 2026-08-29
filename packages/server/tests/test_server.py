"""Port of pi server `test/server.test.ts`."""

import sys

import pytest
import tonio.colored as tonio
from tonio.colored import fs

from pidrei_protocol import ServerMessageDecoder
from pidrei_server import PiServer, PiServerOptions
from pidrei_server.promise import resolved
from pidrei_server.testing import TestServerService
from pidrei_server.transports.unix import UnixServerOptions, create_unix_server


SERVICE = TestServerService()


def test_requires_explicit_listeners():
    with pytest.raises(TypeError, match="listeners"):
        PiServer(SERVICE, PiServerOptions(listeners=None))


def test_rejects_unix_socket_paths_that_cannot_fit_in_sockaddr_un():
    with pytest.raises(TypeError, match="too long"):
        create_unix_server(SERVICE, UnixServerOptions(path=f"/tmp/{'x' * 512}"))


@pytest.mark.tonio
async def test_rejects_an_overlong_derived_private_unix_bind_path(tmp_path):
    max_length = 107 if sys.platform == "linux" else 103
    suffix_length = len(b"/tmp//s")
    path = f"/tmp/{'x' * (max_length - suffix_length)}/s"
    server = create_unix_server(TestServerService(), UnixServerOptions(path=path))

    with pytest.raises(TypeError, match="private Unix bind path.*too long"):
        await server.start()


@pytest.mark.tonio
async def test_rejects_concurrent_start_calls_without_leaking_the_unix_listener(tmp_path):
    path = str(tmp_path / "server.sock")
    server = create_unix_server(TestServerService(), UnixServerOptions(path=path))
    starting = server.start()
    with pytest.raises(Exception, match="starting"):
        await server.start()
    await starting
    await server.close()
    assert server.addresses == []
    with pytest.raises(FileNotFoundError):
        await fs.Path(path).lstat()


@pytest.mark.tonio
async def test_handshake_timeout_cleanup_does_not_wait_for_a_blocked_output_queue():
    class BlockedConnection:
        def __init__(self):
            self.closed = False
            self.final_chunk = None

        def send(self, chunk):
            return tonio.Event().wait()

        def close(self, final_chunk=None):
            self.final_chunk = final_chunk
            self.closed = True
            return resolved(None)

    core = PiServer(TestServerService(), PiServerOptions(listeners=[], max_frame_length=1024, handshake_timeout_ms=10))
    connection = BlockedConnection()
    core.accept(connection)

    for _ in range(200):
        if connection.closed:
            break
        await tonio.sleep(0.005)
    assert connection.closed is True
    assert isinstance(connection.final_chunk, bytes | bytearray)
    messages = ServerMessageDecoder().push(connection.final_chunk)
    assert len(messages) == 1
    assert messages[0]["type"] == "hello_error"
    assert messages[0]["error"]["code"] == "invalid_request"
    await core.close()


def test_rejects_timeout_values_above_nodes_maximum_timer_delay():
    path = "/tmp/pi-server-timeout-test.sock"
    with pytest.raises(TypeError, match="handshakeTimeoutMs"):
        create_unix_server(SERVICE, UnixServerOptions(path=path, handshake_timeout_ms=2_147_483_648))
    with pytest.raises(TypeError, match="gracefulCloseTimeoutMs"):
        create_unix_server(SERVICE, UnixServerOptions(path=path, graceful_close_timeout_ms=2_147_483_648))


def test_rejects_pending_byte_limits_smaller_than_one_maximum_frame(tmp_path):
    path = str(tmp_path / "server.sock")
    with pytest.raises(TypeError, match="maxPendingBytes"):
        create_unix_server(SERVICE, UnixServerOptions(path=path, max_frame_length=128, max_pending_bytes=131))
