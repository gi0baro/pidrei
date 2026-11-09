"""Mirror of pi coding-agent src/core/exec.ts.

Shared command execution utilities for extensions and custom tools.
"""

import codecs
import signal
import subprocess
import threading
from dataclasses import dataclass

import tonio.colored as tonio

from ..utils.process import EXIT_STDIO_GRACE_S, pipe_is_drained


#: The default combined stdout/stderr byte limit. pi also caps it by V8's
#: string limit, which has no counterpart here.
DEFAULT_EXEC_MAX_BUFFER = 256 * 1024 * 1024
#: How long a command gets after SIGTERM before SIGKILL.
_FORCE_KILL_DELAY_S = 5.0


@dataclass(slots=True)
class ExecResult:
    stdout: str
    stderr: str
    code: int
    #: The command was terminated by pidrei: timeout, cancel, or output over `max_buffer`.
    killed: bool
    #: Output exceeded `max_buffer`; stdout and stderr are incomplete and `code` is nonzero.
    truncated: bool


async def exec_command(
    command: str,
    args: list[str],
    cwd: str,
    *,
    cancel=None,
    timeout: float | None = None,
    max_buffer: int = DEFAULT_EXEC_MAX_BUFFER,
) -> ExecResult:
    """Execute a command (no shell) and return stdout/stderr/code.

    Supports a millisecond timeout, a CancelToken and an output cap
    (`max_buffer`, combined stdout/stderr bytes). Mirrors pi's contract: spawn
    and read errors resolve with code 1 (a spawn error's message in stderr);
    only an invalid `max_buffer` raises, before anything is spawned.
    """
    if isinstance(max_buffer, bool) or not isinstance(max_buffer, int) or max_buffer < 0:
        raise ValueError("max_buffer must be a nonnegative integer number of bytes")

    try:
        process = await tonio.open_process(
            [command, *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
        )
    except Exception as error:
        # Child-process errors (e.g. a missing command) are failed executions.
        return ExecResult(stdout="", stderr=str(error), code=1, killed=False, truncated=False)

    # Decode across chunks: a multi-byte UTF-8 character can be split between two chunks,
    # and decoding each chunk separately would corrupt it (earendil-works/pi#11312).
    stdout_decoder = codecs.getincrementaldecoder("utf-8")("replace")
    stderr_decoder = codecs.getincrementaldecoder("utf-8")("replace")
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    # The two pumps, the cancel callback and the watchdog run on their own threads.
    state_lock = threading.Lock()
    output_bytes = 0
    truncated = False
    killed = False
    exited = tonio.Event()

    async def force_kill_after_grace() -> None:
        await exited.wait(_FORCE_KILL_DELAY_S)
        if not exited.is_set():
            try:
                # Popen's kill checks whether the child exited first.
                process.kill()
            except Exception:
                pass

    def kill_process() -> None:
        nonlocal killed
        with state_lock:
            if killed:
                return
            killed = True
        try:
            process.send_signal(signal.SIGTERM)
        except Exception:
            pass
        tonio.spawn.without_tracking(force_kill_after_grace())

    unsubscribe = None
    if cancel is not None:
        if cancel.cancelled:
            kill_process()
        else:
            unsubscribe = cancel.on_cancel(lambda _reason: kill_process())

    async def watchdog() -> None:
        if timeout is None or timeout <= 0:
            return
        await exited.wait(timeout / 1000)
        if not exited.is_set():
            kill_process()

    def take_within_cap(data: bytes) -> bytes | None:
        """Keep only bytes within the combined cap (#11360). Close both pipes
        on overflow so writing descendants cannot keep the call pending:
        closing a pipe wakes the other pump parked on it."""
        nonlocal output_bytes, truncated
        with state_lock:
            if truncated:
                return None
            remaining = max_buffer - output_bytes
            if len(data) <= remaining:
                output_bytes += len(data)
                return data
            truncated = True
        for stream in (process.stdout, process.stderr):
            stream.close()
        kill_process()
        return data[:remaining] if remaining > 0 else None

    chunks_read = 0  # written under state_lock; the post-exit grace reads it
    readers_done = tonio.Event()

    async def pump(stream, decoder, parts: list[str]) -> None:
        nonlocal chunks_read
        try:
            while chunk := await stream.receive_some():
                with state_lock:
                    chunks_read += 1
                taken = take_within_cap(chunk)
                if taken:
                    parts.append(decoder.decode(taken))
                if taken is not chunk:
                    return
        except Exception:
            pass  # stream closed under us; keep whatever arrived

    async def read_streams() -> None:
        try:
            await tonio.spawn(
                pump(process.stdout, stdout_decoder, stdout_parts),
                pump(process.stderr, stderr_decoder, stderr_parts),
            )
        finally:
            readers_done.set()

    tonio.spawn.without_tracking(read_streams())
    watchdog_task = tonio.spawn.without_tracking(watchdog())

    code: int | None = None
    try:
        code = await process.wait()
    except Exception:
        code = 1
    finally:
        exited.set()
        if unsubscribe is not None:
            unsubscribe()
        del watchdog_task

    # Do not hang on inherited stdio handles held open by detached descendants: after the exit,
    # the pipes are closed once they stay idle for the grace window (pi's waitForChildProcess;
    # the bash tool's same loop explains the idle test).
    while not readers_done.is_set():
        with state_lock:
            before = chunks_read
        await readers_done.wait(EXIT_STDIO_GRACE_S)
        if readers_done.is_set():
            break
        with state_lock:
            idle = chunks_read == before
        if idle and all(map(pipe_is_drained, (process.stdout, process.stderr))):
            break
    for stream in (process.stdout, process.stderr):
        stream.close()
    # A reader that already pulled a chunk off the pipe still delivers it.
    await readers_done.wait()

    # On truncation, discard any incomplete UTF-8 suffix buffered by either decoder.
    # Flushing it would append a replacement character that was not in the accepted output.
    if not truncated:
        stdout_parts.append(stdout_decoder.decode(b"", final=True))
        stderr_parts.append(stderr_decoder.decode(b"", final=True))

    # pi reads Node's exit code, which is null for a child ended by a signal, as 0: a terminated
    # command reports `killed`, not a code. Here a signal is a negative return code.
    exit_code = code if code is not None and code >= 0 else 0
    return ExecResult(
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
        # A truncated result must never read as success.
        code=1 if truncated and exit_code == 0 else exit_code,
        killed=killed,
        truncated=truncated,
    )
