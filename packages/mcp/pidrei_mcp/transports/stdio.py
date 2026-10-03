"""Mirror of pi mcp src/transports/stdio.ts.

The server runs in its own process group (`start_new_session`), so closing
the transport can terminate the server's children too: wrappers like `npx`
or `uvx` would otherwise leave the server behind.

"Closed" means what Node's child `close` event means: the process exited
and both of its output pipes reached EOF. A child of the server still
holding a pipe keeps the transport open until it exits too.

Messages to the server are written by one writer coroutine over a channel,
in the order they were sent. Closing ends stdin once the queued messages
are written (Node's `stdin.end()`), then escalates per the spec: SIGTERM to
the group after a grace period, SIGKILL after the close timeout, and once
the process has closed, SIGTERM again for children that ignored stdin
closing. The escalation keeps running if the coroutine awaiting `close()`
is cancelled.
"""

import atexit
import os
import signal
import subprocess
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import tonio.colored as tonio
from tonio.colored.sync import channel

from pidrei_utils import timers

from ..protocol.jsonrpc import JsonRpcMessage, McpConnectionClosedError, parse_json, parse_json_rpc_message, stringify
from .transport import DEFAULT_MAX_MESSAGE_BYTES, SendResult, TransportEvents


DEFAULT_MAX_STDERR_BYTES = 64 * 1024
DEFAULT_CLOSE_TIMEOUT_MS = 2_000
# How long a server gets to exit on its own after stdin closes, before it is sent SIGTERM.
STDIN_CLOSE_GRACE_MS = 500

# Process groups of running servers, killed if the host exits without closing them.
_live_process_groups: set[int] = set()
_live_guard = threading.Lock()
_exit_hook_installed = False


def _kill_process_tree(pid: int, sig: int, *, child_reaped: bool) -> None:
    try:
        # The whole group, so wrappers like `npx` or `uvx` do not leave the server behind.
        os.killpg(pid, sig)
        return
    except OSError:
        # The group is gone or was never created; fall back to the direct child.
        pass
    if child_reaped:
        # Node's `child.kill()` does nothing once the child is gone; a reaped
        # pid may already belong to another process.
        return
    try:
        os.kill(pid, sig)
    except OSError:
        pass


def _kill_live_process_groups() -> None:
    with _live_guard:
        pids = list(_live_process_groups)
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            pass


def _track_process_group(pid: int) -> None:
    global _exit_hook_installed
    with _live_guard:
        _live_process_groups.add(pid)
        if not _exit_hook_installed:
            _exit_hook_installed = True
            atexit.register(_kill_live_process_groups)


def _untrack_process_group(pid: int) -> None:
    with _live_guard:
        _live_process_groups.discard(pid)


@dataclass(frozen=True, slots=True)
class StdioTransportOptions:
    command: str
    args: tuple[str, ...] | None = None
    cwd: str | None = None
    env: dict[str, str] | None = None
    inherit_env: bool = True
    stderr: Literal["pipe", "inherit"] = "pipe"
    on_stderr: Callable[[str], Awaitable[None]] | None = None
    max_message_bytes: int | None = None
    max_stderr_bytes: int | None = None
    # Time to wait for the server to exit after SIGTERM before sending SIGKILL. Default: 2000.
    close_timeout_ms: float | None = None


class StdioTransport(TransportEvents):
    def __init__(
        self,
        command: str,
        *,
        args: Sequence[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        inherit_env: bool = True,
        stderr: Literal["pipe", "inherit"] = "pipe",
        on_stderr: Callable[[str], Awaitable[None]] | None = None,
        max_message_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        close_timeout_ms: float | None = None,
    ) -> None:
        super().__init__()
        self.options = StdioTransportOptions(
            command=command,
            args=tuple(args) if args is not None else None,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            inherit_env=inherit_env,
            stderr=stderr,
            on_stderr=on_stderr,
            max_message_bytes=max_message_bytes,
            max_stderr_bytes=max_stderr_bytes,
            close_timeout_ms=close_timeout_ms,
        )
        self._lock = threading.Lock()
        self._process: Any = None
        self._pid: int | None = None
        self._started = False
        self._closed = False
        self._reaped = False
        self._escalation: tuple[timers.Timeout, timers.Timeout] | None = None
        self._process_closed = tonio.Event()
        self._stdout_buffer = bytearray()
        self._stderr_buffer = b""
        self._writes_sender, self._writes_receiver = channel.unbounded()

    @property
    def pid(self) -> int | None:
        return self._pid

    @property
    def stderr(self) -> str:
        return self._stderr_buffer.decode("utf-8", "replace")

    async def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("MCP stdio transport already started")
            if self._closed:
                raise McpConnectionClosedError()
            self._started = True
        options = self.options
        if options.inherit_env:
            env = {**os.environ, **(options.env or {})}
        else:
            env = dict(options.env or {})
        process = await tonio.open_process(
            [options.command, *(options.args or ())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None if options.stderr == "inherit" else subprocess.PIPE,
            cwd=options.cwd,
            env=env,
            start_new_session=True,
        )
        pid = process.pid
        _track_process_group(pid)
        with self._lock:
            closed_meanwhile = self._closed
            self._process = process
            self._pid = pid
        # Readers and the reaper run whatever happens next, so the process is
        # always reaped and its group untracked.
        stdout_done = tonio.Event()
        stderr_done = tonio.Event()
        tonio.spawn.without_tracking(self._read_stdout(process.stdout, stdout_done))
        if process.stderr is not None:
            tonio.spawn.without_tracking(self._read_stderr(process.stderr, stderr_done))
        else:
            stderr_done.set()
        tonio.spawn.without_tracking(self._reap(process, stdout_done, stderr_done))
        if closed_meanwhile:
            # `close()` ran while the process was being spawned and found no
            # child to stop: nothing will ever be sent to this one.
            process.stdin.close()
            _kill_process_tree(pid, signal.SIGKILL, child_reaped=False)
            raise McpConnectionClosedError()
        tonio.spawn.without_tracking(self._write_loop(process.stdin))

    def send(self, message: JsonRpcMessage) -> SendResult:
        payload = f"{stringify(message)}\n".encode()
        with self._lock:
            if not self._started or self._closed or self._process is None or self._process_closed.is_set():
                return SendResult.failed(McpConnectionClosedError())
            result = SendResult()
            self._writes_sender.send((payload, result))
            return result

    async def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            if process is None:
                no_child = True
            else:
                no_child = False
                # The writer writes what is queued, then ends stdin.
                self._writes_sender.close()
                exited = self._reaped
                if not exited:
                    # Shutdown per the spec: close stdin and let the server
                    # exit, then SIGTERM, then SIGKILL.
                    close_timeout_ms = self.options.close_timeout_ms
                    if close_timeout_ms is None:
                        close_timeout_ms = DEFAULT_CLOSE_TIMEOUT_MS
                    grace = min(STDIN_CLOSE_GRACE_MS, close_timeout_ms)
                    self._escalation = (
                        timers.Timeout(grace, lambda: self._signal_server(signal.SIGTERM)),
                        timers.Timeout(grace + close_timeout_ms, lambda: self._signal_server(signal.SIGKILL)),
                    )
        if no_child:
            self._emit_close()
            return
        if exited:
            # Exited already: the close follows once its pipes reach EOF.
            return
        await self._process_closed.wait()

    def _signal_server(self, sig: int) -> None:
        """An escalation step. A timer that fired just before the reaper
        cancelled it still runs: the reaped flag keeps it off the bare pid."""
        with self._lock:
            pid, reaped = self._pid, self._reaped
        if pid is not None:
            _kill_process_tree(pid, sig, child_reaped=reaped)

    async def _write_loop(self, stdin: Any) -> None:
        while True:
            try:
                payload, result = await self._writes_receiver.receive()
            except BrokenPipeError:
                break
            try:
                await stdin.send_all(payload)
            except Exception as error:
                result.settle(error)
                with self._lock:
                    closed = self._closed
                if not closed:
                    self._emit_error(error)
                continue
            result.settle()
        try:
            stdin.close()
        except Exception:
            pass

    async def _read_stdout(self, stream: Any, done: tonio.Event) -> None:
        try:
            while chunk := await stream.receive_some():
                self._handle_stdout(chunk)
        except Exception as error:
            self._emit_error(error)
        finally:
            done.set()

    async def _read_stderr(self, stream: Any, done: tonio.Event) -> None:
        try:
            while chunk := await stream.receive_some():
                await self._handle_stderr(chunk)
        except Exception as error:
            self._emit_error(error)
        finally:
            done.set()

    async def _reap(self, process: Any, stdout_done: tonio.Event, stderr_done: tonio.Event) -> None:
        try:
            await process.wait()
        finally:
            _untrack_process_group(process.pid)
            with self._lock:
                self._reaped = True
        await tonio.Waiter(stdout_done, stderr_done)
        if self._stdout_buffer.decode("utf-8", "replace").strip():
            self._emit_error(RuntimeError("MCP stdio server closed with an incomplete JSON-RPC message"))
        self._stdout_buffer = bytearray()
        with self._lock:
            escalation, self._escalation = self._escalation, None
            closing = self._closed
            # Nothing more can be written to a process that is gone.
            self._writes_sender.close()
            self._process_closed.set()
        if escalation is not None:
            for timer in escalation:
                timer.cancel()
        if closing:
            # Children of the server that ignored stdin closing would otherwise outlive it.
            _kill_process_tree(process.pid, signal.SIGTERM, child_reaped=True)
        self._emit_close()

    def _handle_stdout(self, chunk: bytes) -> None:
        self._stdout_buffer += chunk
        max_message_bytes = self.options.max_message_bytes or DEFAULT_MAX_MESSAGE_BYTES
        while True:
            newline = self._stdout_buffer.find(b"\n")
            if newline < 0:
                if len(self._stdout_buffer) > max_message_bytes:
                    self._stdout_buffer = bytearray()
                    self._emit_error(RuntimeError(f"MCP stdio message exceeds {max_message_bytes} bytes"))
                return
            line = bytes(self._stdout_buffer[:newline])
            del self._stdout_buffer[: newline + 1]
            if len(line) > max_message_bytes:
                self._emit_error(RuntimeError(f"MCP stdio message exceeds {max_message_bytes} bytes"))
                continue
            text = line.decode("utf-8", "replace").removesuffix("\r")
            if not text.strip():
                continue
            try:
                self._emit_message(parse_json_rpc_message(parse_json(text)))
            except Exception as error:
                self._emit_error(error)

    async def _handle_stderr(self, chunk: bytes) -> None:
        max_stderr_bytes = self.options.max_stderr_bytes or DEFAULT_MAX_STDERR_BYTES
        self._stderr_buffer = (self._stderr_buffer + chunk)[-max_stderr_bytes:]
        on_stderr = self.options.on_stderr
        if on_stderr is not None:
            try:
                await on_stderr(chunk.decode("utf-8", "replace"))
            except Exception as error:
                self._emit_error(error)
