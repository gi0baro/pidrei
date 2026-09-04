"""Unix-domain socket byte transport on tonio (port of pi client `unix.ts`).

Node's socket events map onto two tonio tasks: a reader delivering inbound
chunks and exactly one terminal close/error signal, and a writer draining an
ordered queue so `send()` accepts chunks synchronously in invocation order
while backpressure is carried by the returned awaitables and the pending-byte
cap. pi's win32 rejection is dropped (POSIX-only port).
"""

import socket as _stdlib_socket
import sys
from collections.abc import Awaitable
from dataclasses import dataclass

import tonio.colored as tonio
from tonio.colored import net
from tonio.colored.sync import channel

from pidrei_protocol import DEFAULT_MAX_FRAME_LENGTH

from .promise import Deferred, rejected
from .transport import ByteTransport, ByteTransportFactory, ByteTransportHandlers


MAX_UNIX_SOCKET_PATH_BYTES = 107 if sys.platform == "linux" else 103

_CLOSE_SENTINEL = object()


@dataclass(slots=True, frozen=True)
class UnixTransportOptions:
    path: str
    max_pending_bytes: int | None = None


def create_unix_transport_factory(options: UnixTransportOptions) -> ByteTransportFactory:
    """Creates fresh Unix-domain socket transports, one per connection attempt."""
    if len(options.path) == 0:
        raise TypeError("Unix transport path must not be empty")
    if len(options.path.encode("utf-8")) > MAX_UNIX_SOCKET_PATH_BYTES:
        raise TypeError(f"Unix transport path is too long; maximum is {MAX_UNIX_SOCKET_PATH_BYTES} UTF-8 bytes")
    max_pending_bytes = (
        options.max_pending_bytes if options.max_pending_bytes is not None else DEFAULT_MAX_FRAME_LENGTH * 4
    )
    if isinstance(max_pending_bytes, bool) or not isinstance(max_pending_bytes, int) or max_pending_bytes <= 0:
        raise TypeError("Unix transport maxPendingBytes must be a positive safe integer")

    async def factory(handlers: ByteTransportHandlers) -> ByteTransport:
        stream = await net.open_unix_socket(options.path)
        transport = UnixByteTransport(stream, max_pending_bytes)
        tonio.spawn.without_tracking(transport._run_writer(), transport._run_reader(handlers))
        return transport

    return factory


class UnixByteTransport:
    def __init__(self, stream: net.SocketStream, max_pending_bytes: int) -> None:
        self._stream = stream
        self._max_pending_bytes = max_pending_bytes
        # Terminal flag: set on local close and on the remote close/error the
        # handlers were told about (Node's `terminal`); gates both handler
        # delivery and further sends.
        self._closed = False
        self._pending_bytes = 0
        self._write_sender, self._write_receiver = channel.unbounded()

    def send(self, chunk: bytes) -> Awaitable[None]:
        if not isinstance(chunk, bytes | bytearray):
            return rejected(TypeError("Unix transport chunks must be bytes"))
        if self._closed:
            return rejected(Exception("Unix transport is closed"))
        if self._pending_bytes + len(chunk) > self._max_pending_bytes:
            return rejected(Exception("Unix transport exceeded its pending byte limit"))
        data = bytes(chunk)
        self._pending_bytes += len(data)
        deferred = Deferred()
        self._write_sender.send((data, deferred))
        return deferred.wait()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._write_sender.send(_CLOSE_SENTINEL)
        # Only the reader task closes the stream: closing the fd here races
        # the reader re-arming its read waiter on another worker (tonio
        # deregisters then closes; a registration made in between dies with
        # the fd and never wakes). A full shutdown keeps the fd alive and
        # wakes the reader with EOF; it closes on its way out.
        try:
            self._stream.socket.shutdown(_stdlib_socket.SHUT_RDWR)
        except OSError:
            pass  # already closed by the reader, or the peer is gone

    def _mark_remote_terminal(self) -> None:
        self._closed = True
        self._write_sender.send(_CLOSE_SENTINEL)

    async def _run_writer(self) -> None:
        while True:
            item = await self._write_receiver.receive()
            if item is _CLOSE_SENTINEL:
                return
            data, deferred = item
            try:
                if self._closed:
                    raise Exception("Unix transport is closed")
                await self._stream.send_all(data)
            except Exception as error:
                self._pending_bytes -= len(data)
                deferred.reject(error)
                continue
            self._pending_bytes -= len(data)
            deferred.resolve(None)

    async def _run_reader(self, handlers: ByteTransportHandlers) -> None:
        try:
            try:
                while True:
                    chunk = await self._stream.receive_some()
                    if self._closed:
                        return
                    if not chunk:
                        break
                    handlers.on_data(chunk)
            except Exception as error:
                if not self._closed:
                    self._mark_remote_terminal()
                    handlers.on_error(error)
                return
            if not self._closed:
                self._mark_remote_terminal()
                handlers.on_close()
        finally:
            # The reader owns the fd (see `close`); release it on every exit.
            self._stream.close()
