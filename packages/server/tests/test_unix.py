"""Port of pi server `test/unix.test.ts` (Unix listener filesystem lifecycle).

The `fixtures/stale-socket-server.mjs` child is inlined as a `python -c`
script forked via tonio's process API: it binds the socket, prints readiness,
and sleeps until SIGKILLed, leaving a genuinely stale socket behind.
"""

import stat as stat_module
import subprocess
import sys

import pytest
import tonio.colored as tonio
from tonio.colored import fs

from pidrei_server.testing import TestServerService
from pidrei_server.transports.unix import UnixServerOptions, create_unix_server
from tests.server_support import Harness, settled


_STALE_SOCKET_SERVER = """
import socket, sys, time
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(sys.argv[1])
server.listen(1)
print("listening", flush=True)
time.sleep(60)
"""


@pytest.mark.tonio
async def test_rejects_a_live_listener_without_unlinking_it(sock_dir):
    harness = Harness(sock_dir)
    try:
        path = harness.socket_path()
        first = harness.make_server(path)
        await first.start()
        first_identity = await fs.Path(path).lstat()

        second = harness.make_server(path)
        with pytest.raises(Exception, match="already running"):
            await second.start()
        current_identity = await fs.Path(path).lstat()
        assert (current_identity.st_dev, current_identity.st_ino) == (
            first_identity.st_dev,
            first_identity.st_ino,
        )

        client = await harness.connect(first)
        assert (await settled(client.hello(), "the hello response"))["type"] == "hello"
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_never_unlinks_a_regular_file_at_the_configured_path(sock_dir):
    harness = Harness(sock_dir)
    try:
        path = sock_dir / "server.sock"
        await fs.Path(path).write_text("do not remove")
        await fs.Path(path).chmod(0o640)
        server = harness.make_server(str(path))
        with pytest.raises(Exception, match="non-socket"):
            await server.start()
        assert await fs.Path(path).read_text() == "do not remove"
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_creates_nested_temp_parents_restricts_permissions_and_removes_its_own_socket(sock_dir):
    harness = Harness(sock_dir)
    try:
        path = harness.socket_path(nested=True)
        server = harness.make_server(path)
        await server.start()
        stats = await fs.Path(path).lstat()
        assert stat_module.S_ISSOCK(stats.st_mode)
        assert stats.st_mode & 0o777 == 0o600

        await server.close()
        with pytest.raises(FileNotFoundError):
            await fs.Path(path).lstat()
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_does_not_remove_a_replacement_inode_during_shutdown(sock_dir):
    harness = Harness(sock_dir)
    try:
        path = harness.socket_path()
        server = harness.make_server(path)
        await server.start()
        await fs.Path(path).unlink()
        await fs.Path(path).write_text("replacement")

        closing = server.close()
        assert await fs.Path(path).read_text() == "replacement"
        await closing
        assert await fs.Path(path).read_text() == "replacement"
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_removes_a_genuinely_stale_socket_before_binding(sock_dir):
    harness = Harness(sock_dir)
    try:
        path = str(sock_dir / "server.sock")
        child = await tonio.open_process(
            [sys.executable, "-c", _STALE_SOCKET_SERVER, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            ready = await child.stdout.receive_some()
            assert b"listening" in ready
            stale_identity = await fs.Path(path).lstat()
            assert stat_module.S_ISSOCK(stale_identity.st_mode)
        finally:
            child.kill()
            await child.wait()

        server = harness.make_server(path)
        await server.start()
        live_identity = await fs.Path(path).lstat()
        assert stat_module.S_ISSOCK(live_identity.st_mode)
        client = await harness.connect(server)
        assert (await settled(client.hello(), "the hello response"))["type"] == "hello"
    finally:
        await harness.close()


def test_unix_server_preset_forwards_listener_and_server_options(sock_dir):
    server = create_unix_server(
        TestServerService(), UnixServerOptions(path=str(sock_dir / "server.sock"), server_id="preset-1")
    )
    assert server.id == "preset-1"
    assert server.addresses == []
