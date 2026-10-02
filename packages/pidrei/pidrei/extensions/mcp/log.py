"""Mirror of pi coding-agent src/extensions/mcp/log.ts: log messages MCP
servers send with `notifications/message`, appended to `mcp.log` in the agent
directory. Several pidrei processes may write to the same file, so every
message is one append. The file is rotated to `mcp.log.1` once it grows past
`MAX_LOG_BYTES`.

pi appends synchronously. Here each message is one job on the blocking pool;
a lock keeps one log's size bookkeeping and rotation to one job at a time, and
a server's messages arrive one at a time (the client delivers notifications in
order and awaits each listener), so they keep their order in the file.
"""

import json
import os
import re
from typing import Any

import tonio.colored as tonio
from tonio.colored.sync import Lock

from pidrei_utils import clock


MAX_LOG_BYTES = 5 * 1024 * 1024

_UNDEFINED = object()


def _format_data(data: Any) -> str:
    if isinstance(data, str):
        return data
    if data is _UNDEFINED:
        # `String(undefined)`: JSON.stringify gives undefined for it.
        return "undefined"
    try:
        return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    except TypeError, ValueError:
        return str(data)


def format_mcp_log_message(server: str, params: Any, now: str | None = None) -> str:
    """Format one `notifications/message` from `server` as a log line;
    continuation lines are indented. `now` is an ISO timestamp (default: now)."""
    message = params if isinstance(params, dict) else {"data": params}
    level = message["level"] if isinstance(message.get("level"), str) else "info"
    logger = f" {message['logger']}:" if isinstance(message.get("logger"), str) and message["logger"] else ""
    text = re.sub(r"\r?\n", "\n    ", _format_data(message.get("data", _UNDEFINED)))
    return f"{now or clock.now_iso()} [{server}] {level}{logger} {text}\n"


class McpServerLog:
    """Appends server log messages to one file. Write errors are ignored:
    logging must not break tools."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._size: int | None = None
        self._lock = Lock()

    async def write(self, server: str, params: Any) -> None:
        line = format_mcp_log_message(server, params)
        try:
            async with self._lock:
                self._size = await tonio.spawn_blocking(self._append_blocking, line, self._size)
        except Exception:
            # Ignore: the log is best effort.
            pass

    def _append_blocking(self, line: str, size: int | None) -> int:
        """Append `line`; returns the file's size as bookkept after it."""
        if size is None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            size = self._current_size_blocking()
        if size > MAX_LOG_BYTES:
            # Another process may have rotated it already; check before renaming.
            if self._current_size_blocking() > MAX_LOG_BYTES:
                os.rename(self.path, f"{self.path}.1")
            size = self._current_size_blocking()
        data = line.encode("utf-8")
        with open(self.path, "ab") as file:
            file.write(data)
        return size + len(data)

    def _current_size_blocking(self) -> int:
        try:
            return os.stat(self.path).st_size
        except OSError:
            return 0
