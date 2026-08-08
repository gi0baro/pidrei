"""Unix-domain socket listener (port of pi server `transports/unix/listener.ts`).

Node's `net.Server` maps to a tonio unix listener plus one accept-loop task;
each accepted socket gets a reader task that fans into the server's byte
handlers and reports exactly one terminal close (with an error first when the
socket failed rather than being closed locally). All filesystem work — the
0o700 directory chain, stale-socket probe, hardlink-and-publish, dev/ino
identity checks, chmod — goes through `tonio.colored.fs`. The win32 chmod
skip is dropped (POSIX-only port); the ENOSYS/ENOTSUP tolerance stays.
"""

import errno
import hashlib
import stat as stat_module
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from posixpath import dirname, join

import tonio.colored as tonio
from tonio.colored import fs, net
from tonio.colored.sync import channel

from pidrei_protocol import DEFAULT_MAX_FRAME_LENGTH

from ...connection import ByteConnectionAcceptor, ByteConnectionHandler
from ...listener import PiServerListener
from ...promise import Deferred, driven, gather, rejected, resolved
from ...timers import Timer
from .types import UnixListenerOptions


DEFAULT_SOCKET_MODE = 0o600
DEFAULT_GRACEFUL_CLOSE_TIMEOUT_MS = 5_000
MAX_UINT32 = 0xFFFF_FFFF
MAX_TIMER_DELAY_MS = 2_147_483_647
SOCKET_PROBE_TIMEOUT_MS = 1_000
MAX_UNIX_SOCKET_PATH_BYTES = 107 if sys.platform == "linux" else 103

_MAX_SAFE_INTEGER = 2**53 - 1


def validate_unix_socket_path(path: str, description: str = "Unix socket path") -> None:
    if not path:
        raise TypeError(f"{description} must not be empty")
    if len(path.encode("utf-8")) > MAX_UNIX_SOCKET_PATH_BYTES:
        raise TypeError(f"{description} is too long; maximum is {MAX_UNIX_SOCKET_PATH_BYTES} UTF-8 bytes")


@dataclass(slots=True, frozen=True)
class _ResolvedUnixListenerOptions:
    path: str
    mode: int
    graceful_close_timeout_ms: int
    max_pending_bytes: int
    on_error: Callable[[Exception], None] | None


class UnixListener:
    def __init__(self, options: UnixListenerOptions) -> None:
        self._options = _resolve_unix_listener_options(options)
        self._path = self._options.path
        self._mode = self._options.mode
        self._connections: set[UnixByteConnection] = set()
        self._server: net.SocketListener | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._owned_bind_path: str | None = None
        self._bound_path: str | None = None
        self._closing = False
        self._close_deferred: Deferred | None = None
        self._accept: ByteConnectionAcceptor | None = None

    @property
    def address(self) -> str | None:
        return self._bound_path

    async def start(self, accept: ByteConnectionAcceptor) -> None:
        if self._server is not None:
            raise Exception("Unix listener is already started")
        if self._closing:
            raise Exception("Unix listener is closing or closed")
        self._accept = accept

        owned_bind_path = _get_owned_bind_path(self._path)
        validate_unix_socket_path(owned_bind_path, "PiServer private Unix bind path")
        await _make_private_directories(dirname(self._path))
        await _remove_stale_socket(self._path)
        await _remove_stale_socket(owned_bind_path)
        self._owned_bind_path = owned_bind_path
        try:
            server = await net.open_unix_listener(owned_bind_path)
        except Exception:
            await _remove_path(owned_bind_path)
            self._owned_bind_path = None
            raise
        self._server = server
        try:
            stats = await fs.Path(owned_bind_path).lstat()
            if not stat_module.S_ISSOCK(stats.st_mode):
                raise Exception(f"Unix listener path is not a socket after binding: {owned_bind_path}")
            self._socket_identity = (stats.st_dev, stats.st_ino)
            await fs.Path(self._path).hardlink_to(owned_bind_path)
            await _set_socket_mode(self._path, self._mode)
            self._bound_path = self._path
            tonio.spawn.without_tracking(self._run_accept_loop(server))
        except Exception:
            await self._close_server_and_cleanup(server)
            self._server = None
            raise

    def close(self) -> Awaitable[None]:
        if self._close_deferred is not None:
            return self._close_deferred
        self._closing = True
        self._close_deferred = driven(self._close_internal())
        return self._close_deferred

    async def _run_accept_loop(self, server: net.SocketListener) -> None:
        while True:
            try:
                stream = await server.accept()
            except Exception as error:
                # A closed listener wakes accept with an error; anything else
                # is a real transport failure (Node's server "error" event).
                if not self._closing and self._server is server:
                    self._report_error(error)
                return
            if self._closing:
                stream.close()
                continue
            connection = UnixByteConnection(
                stream, self._options.graceful_close_timeout_ms, self._options.max_pending_bytes
            )
            self._connections.add(connection)
            accept = self._accept
            if accept is None:
                stream.close()
                self._connections.discard(connection)
                continue
            handler = accept(connection)
            tonio.spawn.without_tracking(self._run_connection_reader(connection, stream, handler))

    async def _run_connection_reader(
        self, connection: UnixByteConnection, stream: net.SocketStream, handler: ByteConnectionHandler
    ) -> None:
        failure: Exception | None = None
        try:
            while True:
                chunk = await stream.receive_some()
                if not chunk:
                    break
                if connection.closed:
                    break
                handler.on_data(chunk)
        except Exception as error:
            if not connection.closed and not connection.closing:
                failure = error
        if failure is not None:
            handler.on_error(failure)
            stream.close()
        connection.mark_closed()
        self._connections.discard(connection)
        handler.on_close()

    async def _close_internal(self) -> None:
        self._bound_path = None
        server = self._server
        server_closed = driven(self._close_server_and_cleanup(server) if server is not None else self._cleanup_only())
        await gather([connection.close() for connection in list(self._connections)])
        await server_closed
        if self._owned_bind_path is not None:
            await _remove_path(self._owned_bind_path)
        self._owned_bind_path = None
        self._connections.clear()
        self._server = None

    async def _cleanup_only(self) -> None:
        await self._cleanup_owned_socket()

    async def _close_server_and_cleanup(self, server: net.SocketListener) -> None:
        try:
            server.close()
        except Exception as error:
            self._report_error(error)
        finally:
            await self._cleanup_owned_socket()
            if self._owned_bind_path is not None:
                await _remove_path(self._owned_bind_path)
            self._owned_bind_path = None

    async def _cleanup_owned_socket(self) -> None:
        identity = self._socket_identity
        self._socket_identity = None
        if identity is None:
            return
        try:
            current = await fs.Path(self._path).lstat()
        except FileNotFoundError:
            return
        if not stat_module.S_ISSOCK(current.st_mode) or (current.st_dev, current.st_ino) != identity:
            return

        preserved = join(dirname(self._path), f".c-{str(uuid.uuid4())[:6]}")
        try:
            await fs.Path(self._path).rename(preserved)
        except FileNotFoundError:
            return
        moved = await fs.Path(preserved).lstat()
        if stat_module.S_ISSOCK(moved.st_mode) and (moved.st_dev, moved.st_ino) == identity:
            await _remove_path(preserved)
            return
        try:
            await fs.Path(self._path).lstat()
        except FileNotFoundError:
            await fs.Path(preserved).rename(self._path)
        raise Exception(f"Unix listener path changed during cleanup; preserved replacement at {preserved}")

    def _report_error(self, error: object) -> None:
        on_error = self._options.on_error
        if on_error is None:
            return
        try:
            on_error(error if isinstance(error, Exception) else Exception(str(error)))
        except Exception:
            # Error observers cannot affect listener state.
            pass


_CLOSE_SENTINEL = object()


@dataclass(slots=True, frozen=True)
class _FinalWrite:
    data: bytes | None


class UnixByteConnection:
    """One accepted socket: ordered writer queue with a pending-byte cap.

    Exported only for transport-level verification (pi's `@internal`).
    """

    def __init__(self, stream: net.SocketStream, graceful_close_timeout_ms: int, max_pending_bytes: int) -> None:
        self._stream = stream
        self._graceful_close_timeout_ms = graceful_close_timeout_ms
        self._max_pending_bytes = max_pending_bytes
        self._pending_bytes = 0
        self._closed_value = False
        self._closing = False
        self._close_deferred: Deferred | None = None
        self._close_timer: Timer | None = None
        self._write_sender, self._write_receiver = channel.unbounded()
        tonio.spawn.without_tracking(self._run_writer())

    @property
    def closed(self) -> bool:
        return self._closed_value

    @property
    def closing(self) -> bool:
        return self._closing

    def send(self, chunk: bytes) -> Awaitable[None]:
        if not isinstance(chunk, bytes | bytearray):
            return rejected(TypeError("Unix connection chunks must be bytes"))
        if self._closed_value or self._closing:
            return rejected(Exception("Unix connection is closed"))
        if self._pending_bytes + len(chunk) > self._max_pending_bytes:
            return rejected(Exception("Unix connection exceeded its pending byte limit"))
        data = bytes(chunk)
        self._pending_bytes += len(data)
        deferred = Deferred()
        self._write_sender.send((data, deferred))
        return deferred

    def close(self, final_chunk: bytes | None = None) -> Awaitable[None]:
        if self._closed_value:
            self.mark_closed()
            return resolved(None)
        if self._close_deferred is not None:
            return self._close_deferred
        self._closing = True
        final_bytes = bytes(final_chunk) if final_chunk is not None else None
        deferred = Deferred()
        self._close_deferred = deferred
        self._close_timer = Timer(self._graceful_close_timeout_ms, self._force_close)
        self._write_sender.send(_FinalWrite(final_bytes))
        return deferred

    def mark_closed(self) -> None:
        if self._closed_value:
            return
        self._closed_value = True
        self._closing = True
        if self._close_timer is not None:
            self._close_timer.cancel()
        if self._close_deferred is not None:
            self._close_deferred.resolve(None)
        self._write_sender.send(_CLOSE_SENTINEL)

    def _force_close(self) -> None:
        self._stream.close()
        self.mark_closed()

    async def _run_writer(self) -> None:
        while True:
            item = await self._write_receiver.receive()
            if item is _CLOSE_SENTINEL:
                return
            if isinstance(item, _FinalWrite):
                if self._closed_value:
                    self.mark_closed()
                    return
                try:
                    if item.data is not None:
                        await self._stream.send_all(item.data)
                    self._stream.send_eof()
                except Exception:
                    self._stream.close()
                return
            data, deferred = item
            try:
                if self._closed_value or self._closing:
                    raise Exception("Unix connection is closed")
                await self._stream.send_all(data)
            except Exception as error:
                self._pending_bytes -= len(data)
                deferred.reject(error)
                continue
            self._pending_bytes -= len(data)
            deferred.resolve(None)


def _get_owned_bind_path(path: str) -> str:
    suffix = hashlib.sha256(path.encode("utf-8")).hexdigest()[:8]
    return join(dirname(path), f".p-{suffix}")


async def _make_private_directories(path: str) -> None:
    """`mkdir(recursive, mode 0o700)`: apply the mode to every missing level."""
    missing: list[fs.Path] = []
    current = fs.Path(path)
    while str(current) not in ("/", ""):
        if await current.exists():
            break
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        await directory.mkdir(mode=0o700, exist_ok=True)


async def _remove_stale_socket(path: str) -> None:
    try:
        original = await fs.Path(path).lstat()
    except FileNotFoundError:
        return
    if not stat_module.S_ISSOCK(original.st_mode):
        raise Exception(f"Refusing to remove non-socket Unix listener path: {path}")
    if await _is_socket_live(path):
        raise Exception(f"Unix listener is already running: {path}")

    preserved = join(dirname(path), f".s-{str(uuid.uuid4())[:6]}")
    try:
        await fs.Path(path).rename(preserved)
    except FileNotFoundError:
        return
    current = await fs.Path(preserved).lstat()
    if not stat_module.S_ISSOCK(current.st_mode) or (current.st_dev, current.st_ino) != (
        original.st_dev,
        original.st_ino,
    ):
        try:
            await fs.Path(path).lstat()
        except FileNotFoundError:
            await fs.Path(preserved).rename(path)
        raise Exception(f"Unix listener path changed while checking for a stale socket: {path}")
    await _remove_path(preserved)


async def _remove_path(path: str) -> None:
    try:
        await fs.Path(path).unlink()
    except FileNotFoundError:
        pass


_DEAD_SOCKET_ERRNOS = (errno.ECONNREFUSED, errno.ENOENT, errno.EPIPE, errno.ECONNRESET)


async def _is_socket_live(path: str) -> bool:
    try:
        stream, completed = await tonio.time.timeout(net.open_unix_socket(path), SOCKET_PROBE_TIMEOUT_MS / 1000)
    except OSError as error:
        if error.errno in _DEAD_SOCKET_ERRNOS:
            return False
        raise
    if not completed:
        # Mirror Node: an unresponsive endpoint is assumed live rather than removed.
        return True
    stream.close()
    return True


async def _set_socket_mode(path: str, mode: int) -> None:
    try:
        await fs.Path(path).chmod(mode)
    except OSError as error:
        if error.errno not in (errno.ENOSYS, errno.ENOTSUP):
            raise


def create_unix_listener(options: UnixListenerOptions) -> PiServerListener:
    return UnixListener(options)


def _resolve_unix_listener_options(options: UnixListenerOptions) -> _ResolvedUnixListenerOptions:
    validate_unix_socket_path(options.path, "PiServer Unix socket path")
    mode = options.mode if options.mode is not None else DEFAULT_SOCKET_MODE
    if isinstance(mode, bool) or not isinstance(mode, int) or mode < 0 or mode > 0o777:
        raise TypeError("PiServer Unix socket mode must be an integer between 0 and 0o777")
    max_frame_length = options.max_frame_length if options.max_frame_length is not None else DEFAULT_MAX_FRAME_LENGTH
    if (
        isinstance(max_frame_length, bool)
        or not isinstance(max_frame_length, int)
        or max_frame_length <= 0
        or max_frame_length > MAX_UINT32
    ):
        raise TypeError(f"PiServer maxFrameLength must be an integer between 1 and {MAX_UINT32}")
    max_pending_bytes = options.max_pending_bytes if options.max_pending_bytes is not None else max_frame_length * 4
    if (
        isinstance(max_pending_bytes, bool)
        or not isinstance(max_pending_bytes, int)
        or max_pending_bytes > _MAX_SAFE_INTEGER
        or max_pending_bytes < max_frame_length + 4
    ):
        raise TypeError("PiServer maxPendingBytes must be a safe integer at least maxFrameLength + 4")
    graceful_close_timeout_ms = (
        options.graceful_close_timeout_ms
        if options.graceful_close_timeout_ms is not None
        else DEFAULT_GRACEFUL_CLOSE_TIMEOUT_MS
    )
    if (
        isinstance(graceful_close_timeout_ms, bool)
        or not isinstance(graceful_close_timeout_ms, int)
        or graceful_close_timeout_ms <= 0
        or graceful_close_timeout_ms > MAX_TIMER_DELAY_MS
    ):
        raise TypeError(f"PiServer gracefulCloseTimeoutMs must be an integer between 1 and {MAX_TIMER_DELAY_MS}")
    return _ResolvedUnixListenerOptions(
        path=options.path,
        mode=mode,
        max_pending_bytes=max_pending_bytes,
        graceful_close_timeout_ms=graceful_close_timeout_ms,
        on_error=options.on_error,
    )
