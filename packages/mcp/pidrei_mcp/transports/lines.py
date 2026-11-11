"""Mirror of pi mcp src/transports/lines.ts.

Each half is driven by one reader coroutine: the stdout reader owns the
pending message, the stderr reader the tail, and `end()` runs once the
stdout reader is done. The `stderr` tail is one `bytes` value, rebound
whole, so any thread may read it.
"""

from collections.abc import Awaitable, Callable

from ..protocol.jsonrpc import JsonRpcMessage, parse_json, parse_json_rpc_message
from .transport import DEFAULT_MAX_MESSAGE_BYTES


DEFAULT_MAX_STDERR_BYTES = 64 * 1024


class JsonRpcLineReader:
    """Reads a program's output as MCP's stdio framing: newline-delimited
    JSON-RPC messages on stdout, and a bounded tail of stderr for
    diagnostics."""

    def __init__(
        self,
        *,
        on_message: Callable[[JsonRpcMessage], None],
        on_error: Callable[[Exception], None],
        max_message_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        # Every stderr chunk, decoded as UTF-8.
        on_stderr: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._on_message = on_message
        self._on_error = on_error
        self._max_message_bytes = max_message_bytes
        self._max_stderr_bytes = max_stderr_bytes
        self._on_stderr = on_stderr
        # The bytes of the incomplete message received so far.
        self._stdout_chunks: list[bytes] = []
        self._stdout_pending_bytes = 0
        self._stderr_buffer = b""

    @property
    def stderr(self) -> str:
        """The last `max_stderr_bytes` of stderr."""
        return self._stderr_buffer.decode("utf-8", "replace")

    def read_stdout(self, chunk: bytes) -> None:
        """Takes the next stdout bytes and reports every complete message, or
        the error parsing it. Bytes of an incomplete message are kept as
        chunks and joined once when its newline arrives, so a large message
        that arrives in many chunks is copied and scanned only once."""
        max_message_bytes = DEFAULT_MAX_MESSAGE_BYTES if self._max_message_bytes is None else self._max_message_bytes
        start = 0
        while True:
            newline = chunk.find(b"\n", start)
            if newline < 0:
                if start < len(chunk):
                    self._stdout_chunks.append(chunk[start:])
                    self._stdout_pending_bytes += len(chunk) - start
                if self._stdout_pending_bytes > max_message_bytes:
                    self._stdout_chunks = []
                    self._stdout_pending_bytes = 0
                    self._on_error(RuntimeError(f"MCP stdio message exceeds {max_message_bytes} bytes"))
                return
            line = chunk[start:newline]
            if self._stdout_chunks:
                self._stdout_chunks.append(line)
                line = b"".join(self._stdout_chunks)
                self._stdout_chunks = []
                self._stdout_pending_bytes = 0
            start = newline + 1
            if len(line) > max_message_bytes:
                self._on_error(RuntimeError(f"MCP stdio message exceeds {max_message_bytes} bytes"))
                continue
            text = line.decode("utf-8", "replace").removesuffix("\r")
            if not text.strip():
                continue
            try:
                self._on_message(parse_json_rpc_message(parse_json(text)))
            except Exception as error:
                self._on_error(error)

    async def read_stderr(self, chunk: bytes) -> None:
        """Takes the next stderr bytes into the tail and passes them to
        `on_stderr`; a failing `on_stderr` is reported as an error."""
        max_stderr_bytes = DEFAULT_MAX_STDERR_BYTES if self._max_stderr_bytes is None else self._max_stderr_bytes
        buffer = self._stderr_buffer + chunk
        if len(buffer) > max_stderr_bytes:
            buffer = buffer[-max_stderr_bytes:]
        self._stderr_buffer = buffer
        if self._on_stderr is not None:
            try:
                await self._on_stderr(chunk.decode("utf-8", "replace"))
            except Exception as error:
                self._on_error(error)

    def end(self) -> None:
        """Ends stdout: a message without its newline is reported as an error
        and dropped."""
        if b"".join(self._stdout_chunks).decode("utf-8", "replace").strip():
            self._on_error(RuntimeError("MCP stdio server closed with an incomplete JSON-RPC message"))
        self._stdout_chunks = []
        self._stdout_pending_bytes = 0
