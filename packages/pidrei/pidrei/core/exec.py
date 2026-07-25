"""Mirror of pi coding-agent src/core/exec.ts.

Shared command execution utilities for extensions and custom tools.
"""

import signal
import subprocess
from dataclasses import dataclass

import tonio.colored as tonio


@dataclass(slots=True)
class ExecResult:
    stdout: str
    stderr: str
    code: int
    killed: bool


async def exec_command(
    command: str,
    args: list[str],
    cwd: str,
    *,
    cancel=None,
    timeout: float | None = None,
) -> ExecResult:
    """Execute a command (no shell) and return stdout/stderr/code.

    Supports a millisecond timeout and a CancelToken. Mirrors pi's contract:
    never raises; spawn/read errors resolve with code 1.
    """
    try:
        process = await tonio.open_process(
            [command, *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
        )
    except Exception:
        return ExecResult(stdout="", stderr="", code=1, killed=False)

    stdout_parts: list[bytes] = []
    stderr_parts: list[bytes] = []
    killed = False
    exited = tonio.Event()

    def kill_process() -> None:
        nonlocal killed
        if not killed:
            killed = True
            try:
                process.send_signal(signal.SIGTERM)
            except Exception:
                pass

    async def force_kill_after_grace() -> None:
        # Force kill after 5 seconds if SIGTERM doesn't work.
        waiter = await exited.wait(5.0)
        if not waiter.is_set():
            try:
                process.kill()
            except Exception:
                pass

    unsubscribe = None
    if cancel is not None:
        if cancel.cancelled:
            kill_process()
            tonio.spawn.without_tracking(force_kill_after_grace())
        else:

            def on_cancel(_reason) -> None:
                kill_process()
                tonio.spawn.without_tracking(force_kill_after_grace())

            unsubscribe = cancel.on_cancel(on_cancel)

    async def watchdog() -> None:
        if timeout is None or timeout <= 0:
            return
        waiter = await exited.wait(timeout / 1000)
        if not waiter.is_set():
            kill_process()
            await force_kill_after_grace()

    async def pump(stream, parts: list[bytes]) -> None:
        if stream is None:
            return
        try:
            while True:
                chunk = await stream.receive_some()
                if not chunk:
                    return
                parts.append(chunk)
        except Exception:
            pass

    watchdog_task = tonio.spawn.without_tracking(watchdog())

    code: int | None = None
    try:
        await tonio.spawn(pump(process.stdout, stdout_parts), pump(process.stderr, stderr_parts))
        code = await process.wait()
    except Exception:
        return ExecResult(
            stdout=b"".join(stdout_parts).decode("utf-8", "replace"),
            stderr=b"".join(stderr_parts).decode("utf-8", "replace"),
            code=1,
            killed=killed,
        )
    finally:
        exited.set()
        if unsubscribe is not None:
            unsubscribe()
        del watchdog_task

    return ExecResult(
        stdout=b"".join(stdout_parts).decode("utf-8", "replace"),
        stderr=b"".join(stderr_parts).decode("utf-8", "replace"),
        code=code if code is not None else 0,
        killed=killed,
    )
