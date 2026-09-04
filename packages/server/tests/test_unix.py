"""Port of pi server `test/unix.test.ts` (Unix listener filesystem lifecycle).

The `fixtures/stale-socket-server.mjs` child is inlined as a `python -c`
script forked via tonio's process API: it binds the socket, prints readiness,
and sleeps until SIGKILLed, leaving a genuinely stale socket behind.

Driven against the listener directly (the protocol server is not ported —
UPSTREAM_EXPERIMENTAL_RULING.md): the echo round-trip replaces upstream's
hello handshake as the "is it serving" probe, and the unix-server preset case
is gone with the preset.
"""

import os
import stat as stat_module
import subprocess
import sys

import pytest
import tonio.colored as tonio
from tonio.colored import fs

from pidrei_server.transports.unix import get_unix_socket_path
from tests.server_support import Harness, echo_acceptor, settled


_STALE_SOCKET_SERVER = """
import socket, sys, time
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(sys.argv[1])
server.listen(1)
print("listening", flush=True)
time.sleep(60)
"""


@pytest.mark.tonio
async def test_derives_the_explicit_unix_socket_path_for_a_server_id(sock_dir):
    """pi: "creates an in-memory server ID and derives its explicit Unix socket
    path" — the hello round-trips are echo round-trips here."""
    harness = Harness(sock_dir)
    try:
        directory = str(sock_dir / "srv-id")
        server_id = "00000000-0000-4000-8000-000000000001"
        path = get_unix_socket_path(server_id, directory)

        assert path == os.path.join(directory, f"{server_id}.sock")

        first = harness.make_listener(path)
        await first.start(echo_acceptor)
        assert await settled(harness.roundtrip(path), "the first echo") == b"ping"
        await first.close()

        replacement = harness.make_listener(path)
        await replacement.start(echo_acceptor)
        assert await settled(harness.roundtrip(path), "the replacement echo") == b"ping"
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_rejects_a_live_listener_without_unlinking_it(sock_dir):
    harness = Harness(sock_dir)
    try:
        path = harness.socket_path()
        first = harness.make_listener(path)
        await first.start(echo_acceptor)
        first_identity = await fs.Path(path).lstat()

        second = harness.make_listener(path)
        with pytest.raises(Exception, match="already running"):
            await second.start(echo_acceptor)
        current_identity = await fs.Path(path).lstat()
        assert (current_identity.st_dev, current_identity.st_ino) == (
            first_identity.st_dev,
            first_identity.st_ino,
        )

        assert await settled(harness.roundtrip(path), "the echo") == b"ping"
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_never_unlinks_a_regular_file_at_the_configured_path(sock_dir):
    harness = Harness(sock_dir)
    try:
        path = sock_dir / "server.sock"
        await fs.Path(path).write_text("do not remove")
        await fs.Path(path).chmod(0o640)
        listener = harness.make_listener(str(path))
        with pytest.raises(Exception, match="non-socket"):
            await listener.start(echo_acceptor)
        assert await fs.Path(path).read_text() == "do not remove"
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_creates_nested_temp_parents_restricts_permissions_and_removes_its_own_socket(sock_dir):
    harness = Harness(sock_dir)
    try:
        path = harness.socket_path(nested=True)
        listener = harness.make_listener(path)
        await listener.start(echo_acceptor)
        stats = await fs.Path(path).lstat()
        assert stat_module.S_ISSOCK(stats.st_mode)
        assert stats.st_mode & 0o777 == 0o600
        assert os.listdir(os.path.dirname(path)) == ["server.sock"]

        await listener.close()
        with pytest.raises(FileNotFoundError):
            await fs.Path(path).lstat()
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_does_not_remove_a_replacement_inode_during_shutdown(sock_dir):
    harness = Harness(sock_dir)
    try:
        path = harness.socket_path()
        listener = harness.make_listener(path)
        await listener.start(echo_acceptor)
        await fs.Path(path).unlink()
        await fs.Path(path).write_text("replacement")

        closing = listener.close()
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

        listener = harness.make_listener(path)
        await listener.start(echo_acceptor)
        live_identity = await fs.Path(path).lstat()
        assert stat_module.S_ISSOCK(live_identity.st_mode)
        assert await settled(harness.roundtrip(path), "the echo") == b"ping"
    finally:
        await harness.close()
