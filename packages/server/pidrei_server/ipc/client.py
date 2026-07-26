"""Mirror of pi server src/ipc/client.ts."""

from typing import Any

from tonio.colored import net

from ..config import get_socket_path
from .protocol import encode_message, parse_response_line


async def send_ipc_request(request: dict[str, Any]) -> dict[str, Any]:
    socket_path = get_socket_path()

    stream = await net.open_unix_socket(socket_path)
    try:
        await stream.send_all(encode_message(request).encode("utf-8"))

        buffer = ""
        while True:
            chunk = await stream.receive_some()
            if not chunk:
                raise Exception(f"Server socket closed before a response was received: {socket_path}")
            buffer += chunk.decode("utf-8")
            while True:
                newline_index = buffer.find("\n")
                if newline_index == -1:
                    break

                line = buffer[:newline_index].strip()
                buffer = buffer[newline_index + 1 :]
                if not line:
                    continue
                return parse_response_line(line)
    finally:
        stream.close()
