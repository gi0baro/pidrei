"""Mirror of pi server src/ipc/server.ts.

pi passes a request-handler callable with an `openRpcStream` property
attached via Object.assign; the Python equivalent is the two-field
IpcRequestHandler record.

node's socket.write is synchronously queued; the supervisor's event
subscribers rely on that (they are plain sync callbacks). The equivalent
here is a per-connection outbound channel drained by a writer task, so
callbacks stay synchronous and write ordering is preserved.
"""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio
from tonio.colored import net
from tonio.colored.sync.channel import unbounded

from ..config import get_socket_path
from .protocol import encode_message, parse_request_line


@dataclass(slots=True)
class IpcRequestHandler:
    # async (request: dict) -> response dict
    handle_request: Callable[[dict[str, Any]], Any]
    # (instance_id, on_response, on_session_event, on_ui_request) -> RpcStreamHandle | None,
    # where the handle has `async handle_request(request: dict)` and `close()`
    open_rpc_stream: Callable[..., Any]


class IpcServer:
    """Handle over the listening socket (pi returns the node Server)."""

    def __init__(self, listener: Any):
        self._listener = listener

    def close(self) -> None:
        self._listener.close()


_CLOSE = object()


class _ConnectionWriter:
    """Sync-callable write queue over one connection stream."""

    def __init__(self, stream: Any):
        self._stream = stream
        self._sender, self._receiver = unbounded()
        self._closed = False
        tonio.spawn.without_tracking(self._drain())

    async def _drain(self) -> None:
        while True:
            message = await self._receiver.receive()
            if message is _CLOSE:
                self._stream.close()
                return
            try:
                await self._stream.send_all(message.encode("utf-8"))
            except Exception:
                self._closed = True
                return

    def write(self, message: dict[str, Any]) -> None:
        if self._closed:
            return
        self._sender.send(encode_message(message))

    def end(self, message: dict[str, Any] | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        if message is not None:
            self._sender.send(encode_message(message))
        self._sender.send(_CLOSE)


async def _read_request_line(stream: Any, buffer: list[str]) -> str | None:
    """Read up to the first non-empty line; the leftover stays in buffer[0].
    Returns None when the connection closes before a full line arrives."""
    while True:
        while True:
            newline_index = buffer[0].find("\n")
            if newline_index == -1:
                break
            line = buffer[0][:newline_index].strip()
            buffer[0] = buffer[0][newline_index + 1 :]
            if line:
                return line
        chunk = await stream.receive_some()
        if not chunk:
            return None
        buffer[0] += chunk.decode("utf-8")


async def _handle_connection(stream: Any, handler: IpcRequestHandler) -> None:
    writer = _ConnectionWriter(stream)
    buffer = [""]

    line = await _read_request_line(stream, buffer)
    if line is None:
        writer.end()
        return

    try:
        request = parse_request_line(line)
        if request.get("type") == "rpc_stream":
            response = await handler.handle_request(request)
            if not response.get("ok") or response.get("type") != "rpc_ready" or not response.get("instance"):
                writer.end(response)
                return

            rpc_stream = handler.open_rpc_stream(
                request["instanceId"],
                writer.write,
                writer.write,
                writer.write,
            )
            if rpc_stream is None:
                writer.end({"type": "error", "ok": False, "error": f"Unknown instance: {request['instanceId']}"})
                return

            writer.write(response)
            try:
                # Sequential handling mirrors pi's per-connection promise chain.
                while True:
                    rpc_line = await _read_request_line(stream, buffer)
                    if rpc_line is None:
                        return
                    try:
                        await rpc_stream.handle_request(json.loads(rpc_line))
                    except Exception as rpc_error:
                        writer.write({"type": "error", "ok": False, "error": str(rpc_error)})
            finally:
                rpc_stream.close()
                writer.end()
            return

        response = await handler.handle_request(request)
        writer.end(response)
    except Exception as error:
        writer.end({"type": "error", "ok": False, "error": str(error)})


async def _accept_loop(listener: Any, handler: IpcRequestHandler) -> None:
    while True:
        try:
            stream = await listener.accept()
        except Exception:
            # Listener closed (server shutdown).
            return
        tonio.spawn.without_tracking(_handle_connection(stream, handler))


async def start_ipc_server(handler: IpcRequestHandler) -> IpcServer:
    socket_path = get_socket_path()
    await _remove_stale_socket_if_needed(socket_path)

    listener = await net.open_unix_listener(socket_path)
    tonio.spawn.without_tracking(_accept_loop(listener, handler))
    return IpcServer(listener)


async def _remove_stale_socket_if_needed(socket_path: str) -> None:
    if not os.path.exists(socket_path):
        return

    if await _is_socket_live(socket_path):
        raise Exception(f"server is already running: {socket_path}")

    os.unlink(socket_path)


async def _is_socket_live(socket_path: str) -> bool:
    try:
        stream = await net.open_unix_socket(socket_path)
    except ConnectionRefusedError, FileNotFoundError, BrokenPipeError, ConnectionResetError:
        return False
    stream.close()
    return True
