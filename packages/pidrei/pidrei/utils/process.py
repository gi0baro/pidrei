"""Async `subprocess.run` whose deadline actually kills the child.

`tonio.run_process` takes no timeout, and wrapping it in `tonio.time.timeout`
does not supply one: when the wrapper is abandoned the child keeps running
(measured — a `sleep 1.5` outlived a caller dropped at 0.2s), un-reaped, with
its pipes still open. `subprocess.run(timeout=…)` kills and reaps. Porting a
timed subprocess to tonio without this helper would trade "bounded and killed"
for "unbounded and orphaned".

The shape: `reap()` owns `process.wait()` for the child's whole life — including
past our return — while the caller waits on an `Event` with the deadline. So
"give up" is simply *not awaiting the reaper*, which is what makes the deadline
a ceiling rather than a floor. `core/tools/bash.py` uses the same pieces for the
bash tool, but kills the whole process group and grants a post-exit grace window
for detached descendants; this one serves short probes and mirrors
`subprocess.run` instead.

Three choices match pi:

* the deadline sends **SIGTERM**, not SIGKILL. `subprocess.run` uses SIGKILL,
  but pi reaches these through Node's `spawnSync`/`child.kill()`, whose default
  `killSignal` is SIGTERM — and a `!cmd` credential helper deserves the chance
  to clean up. A child that ignores it is still SIGKILLed
  `_KILL_ESCALATION_S` later, inside the reaper, where the caller never waits
  for it.
* only the **direct child** is signalled. A `shell=True` command that spawns its
  own children can leave them behind; pi has the same hole, and closing it would
  mean a process group, which changes signal delivery for every call site.
* on timeout the caller returns **immediately** rather than waiting for the
  child to die, which is what makes the deadline a ceiling. pi does the same:
  `runTmuxShow` resolves the moment it calls `proc.kill()`.

Whatever output arrived before the deadline is attached to `TimeoutExpired`,
matching `subprocess.run`. That is safe because closing a pipe wakes a reader
parked in `receive_some` at once (measured: 0.000s, `EBADF`), so the drain tasks
can be joined without costing latency — and, just as importantly, without
leaving half-finished tasks behind for the runtime to trip over.

On the success path output is read to EOF, like `subprocess.run`, so a
grandchild holding a pipe open can outlast the child.

Bytes only, because `tonio.open_process` refuses `encoding`/`text`; callers
decode.
"""

import signal as signal_module
import subprocess
from collections.abc import Sequence
from typing import Any

import tonio.colored as tonio
from tonio.exceptions import ResourceBroken


#: How long a signalled child gets before SIGKILL. Never observed by the caller:
#: escalation runs in the reaper, after `run_command` has already returned.
_KILL_ESCALATION_S = 2.0


async def run_command(
    command: Sequence[str] | str,
    *,
    input: bytes | None = None,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
    capture_output: bool = False,
    timeout: float | None = None,
    check: bool = False,
    kill_signal: int = signal_module.SIGTERM,
    **options: Any,
) -> subprocess.CompletedProcess[bytes]:
    """Run `command` to completion without occupying a blocking-pool thread.

    Mirrors `subprocess.run`, including the exceptions — `TimeoutExpired` when
    the deadline passes, `CalledProcessError` under `check=True`, `OSError` when
    the binary is missing — so call sites keep their existing `except` clauses.
    """
    if capture_output:
        if stdout is not None or stderr is not None:
            raise ValueError("capture_output may not be used with stdout or stderr")
        stdout = stderr = subprocess.PIPE
    if input is not None:
        if stdin is not None:
            raise ValueError("input may not be used with stdin")
        stdin = subprocess.PIPE

    process = await tonio.open_process(command, stdin=stdin, stdout=stdout, stderr=stderr, **options)

    captured: dict[str, bytes] = {}
    result: dict[str, Any] = {}
    exited = tonio.Event()

    async def feed(stream) -> None:
        try:
            await stream.send_all(input)
        except BrokenPipeError, ResourceBroken:
            pass  # child exited before reading its input
        finally:
            stream.close()

    async def drain(name: str, stream) -> None:
        buffer = bytearray()
        try:
            while chunk := await stream.receive_some():
                buffer += chunk
        except Exception:
            pass  # stream closed under us; keep whatever arrived
        captured[name] = bytes(buffer)

    async def reap() -> None:
        """Owns `wait()` for the child's whole life, including past our return."""
        try:
            result["code"] = await process.wait()
        except BaseException as error:  # e.g. TONIO_BUGS #7; surface it, don't KeyError
            result["error"] = error
        finally:
            exited.set()

    def _signal(sig: int) -> None:
        """Signal the child, tolerating the race against `reap`.

        `Popen.send_signal` checks `poll()` and then calls `os.kill`, so `reap`
        can reap the child in between and turn the call into a
        `ProcessLookupError` — or, worse, into a signal aimed at a recycled pid.
        `escalate` runs detached, and an exception there is reported on stderr
        rather than to anyone who can act on it, so it is caught at the source.

        `send_signal` also goes straight to `Popen`, deliberately: it never
        touches tonio's `.returncode`/`.poll()`, which is what runs
        `_close_pidfd` — the unguarded read-modify-write under TONIO_BUGS #7.
        """
        try:
            process.send_signal(sig)
        except ProcessLookupError, PermissionError:
            pass  # already gone

    async def escalate() -> None:
        await exited.wait(_KILL_ESCALATION_S)
        if not exited.is_set():
            _signal(signal_module.SIGKILL)

    def abandon() -> None:
        """Signal, drop the pipes, leave `reap` to collect the corpse."""
        _signal(kill_signal)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        tonio.spawn.without_tracking(escalate())

    io: list = []
    if input is not None:
        io.append(feed(process.stdin))
    if process.stdout is not None:
        io.append(drain("stdout", process.stdout))
    if process.stderr is not None:
        io.append(drain("stderr", process.stderr))
    io_join = tonio.spawn.without_results(*io) if io else None
    tonio.spawn.without_tracking(reap())

    try:
        await exited.wait(timeout)
    except BaseException:
        abandon()  # our caller was cancelled: never leave the child behind
        raise

    if not exited.is_set():
        abandon()
        # Closing a pipe wakes a reader parked in `receive_some` immediately
        # (measured: 0.000s, EBADF), so joining here costs nothing and keeps the
        # deadline a ceiling — while leaving no half-finished tasks behind.
        if io_join is not None:
            await io_join
        raise subprocess.TimeoutExpired(command, timeout, output=captured.get("stdout"), stderr=captured.get("stderr"))

    if io_join is not None:
        await io_join
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()

    if "error" in result:
        raise result["error"]
    returncode = result["code"]
    out, err = captured.get("stdout"), captured.get("stderr")
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, command, output=out, stderr=err)
    return subprocess.CompletedProcess(command, returncode, out, err)
