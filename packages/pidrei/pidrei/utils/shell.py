"""Mirror of pi coding-agent src/utils/shell.ts (POSIX-only; the win32 Git-Bash
discovery, legacy-WSL stdin transport, and taskkill paths are not ported)."""

import os
import re
import shutil
import signal
import sys
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import tonio.colored as tonio
from tonio.colored import signals as tonio_signals

from ..config import get_bin_dir
from .fd_io import hard_exit
from .runtime_options import HANDLED_SIGNALS


@dataclass(slots=True)
class ShellConfig:
    shell: str
    args: list[str]
    command_transport: str | None = None  # "argv" | "stdin"


def _get_bash_shell_config(shell: str) -> ShellConfig:
    return ShellConfig(shell=shell, args=["-c"])


# Sync by design: `BashTool.exec` calls `get_shell_config_blocking` through
# `spawn_blocking`, so the PATH walk never runs on a runtime worker. Any new
# caller must do the same.
def _find_executable_on_path_blocking(executable: str) -> str | None:
    return shutil.which(executable)


def get_shell_config_blocking(custom_shell_path: str | None = None) -> ShellConfig:
    """Resolve shell configuration: user shellPath, then /bin/bash, then bash
    on PATH, then sh."""
    if custom_shell_path:
        if os.path.exists(custom_shell_path):
            return _get_bash_shell_config(custom_shell_path)
        raise Exception(f"Custom shell path not found: {custom_shell_path}")

    if os.path.exists("/bin/bash"):
        return _get_bash_shell_config("/bin/bash")

    bash_on_path = _find_executable_on_path_blocking("bash")
    if bash_on_path:
        return _get_bash_shell_config(bash_on_path)

    return ShellConfig(shell="sh", args=["-c"])  # noqa: S604


def get_shell_env() -> dict[str, str]:
    bin_dir = get_bin_dir()
    current_path = os.environ.get("PATH", "")
    path_entries = [entry for entry in current_path.split(os.pathsep) if entry]
    has_bin_dir = bin_dir in path_entries
    updated_path = current_path if has_bin_dir else os.pathsep.join(part for part in (bin_dir, current_path) if part)

    env = dict(os.environ)
    env["PATH"] = updated_path
    return env


# Control characters except tab/newline/CR, and the Unicode interlinear annotation
# characters U+FFF9..U+FFFB (they crash width measurement). pidrei also drops lone
# surrogates, which lossy decoding can leave in a Python str (a JS string cannot hold
# them after pi's decode, and they would fail on output encoding here).
_BINARY_OUTPUT_UNSAFE_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￹-￻\ud800-\udfff]")


def sanitize_binary_output(value: str) -> str:
    """Sanitize binary output for display/storage: drop control characters
    (except tab/newline/CR), Unicode interlinear annotation characters that
    crash width measurement, and lone surrogates."""
    return _BINARY_OUTPUT_UNSAFE_RE.sub("", value)


# Detached child processes must be tracked so they can be killed on parent
# shutdown signals (SIGHUP/SIGTERM).
_tracked_detached_child_pids: set[int] = set()
_tracked_guard = threading.Lock()


def track_detached_child_pid(pid: int) -> None:
    with _tracked_guard:
        _tracked_detached_child_pids.add(pid)


def untrack_detached_child_pid(pid: int) -> None:
    with _tracked_guard:
        _tracked_detached_child_pids.discard(pid)


def kill_tracked_detached_children() -> None:
    with _tracked_guard:
        pids = list(_tracked_detached_child_pids)
        _tracked_detached_child_pids.clear()
    for pid in pids:
        kill_process_tree(pid)


def signal_exit_status(signum: int) -> int:
    """The exit status of a process ended by SIGHUP (129) or SIGTERM (143)."""
    return 129 if signum == signal.SIGHUP else 143


async def exit_on_signal(signum: int) -> None:
    """The SIGTERM/SIGHUP handler while no mode runs (startup, the package and
    config commands): what the signal's default action did, after killing
    the tracked detached children."""
    kill_tracked_detached_children()
    hard_exit(signal_exit_status(signum))


async def stop_on_signal(_signum: int) -> None:
    """The SIGTSTP handler: stop the process, as SIGTSTP's default action did
    before the runtime handled it. SIGSTOP cannot be handled, so the kernel
    stops the process; SIGCONT resumes it."""
    os.kill(os.getpid(), signal.SIGSTOP)


async def _ignore_signal(_signum: int) -> None:
    return None


class SignalDispatcher:
    """The process's one receiver for every signal the runtime is created
    with (`runtime_options.HANDLED_SIGNALS`): `watch()` runs for the whole
    process and starts the handler installed for each signal it receives.
    Handlers start as no-ops; whoever installs one decides in it what the
    signal does.

    Each handler runs detached, as pi's `process.on` listeners do: a handler
    still awaiting (a shutdown disposing the runtime) never holds back the
    next signal, so a second SIGTERM reaches its "already shutting down,
    exit now" branch."""

    __slots__ = ("_handlers",)

    def __init__(self) -> None:
        self._handlers: dict[int, Callable[[int], Awaitable[None]]] = dict.fromkeys(HANDLED_SIGNALS, _ignore_signal)

    def set_handler(self, signum: int, handler: Callable[[int], Awaitable[None]]) -> None:
        self._handlers[signum] = handler

    async def watch(self) -> None:
        with tonio_signals.signal_receiver(*HANDLED_SIGNALS) as receiver:
            async for signum in receiver:
                tonio.spawn.without_tracking(self._handlers[signum](signum))


def kill_process_tree(pid: int) -> None:
    """Kill a process and all its children (POSIX process-group SIGKILL).

    `pid` leads its own process group (the shell is started with
    `start_new_session`), and only the group is signalled. pi falls back to
    the bare pid when the group kill fails; for a group leader that kill
    fails only once the leader is gone, and then the pid may already belong
    to another process (this is also called after the shell exited, for the
    processes it left behind).
    """
    if sys.platform == "darwin":
        # macOS signals group members individually after releasing the group lock. A killed
        # child can wake its waiting shell before that shell is signalled, letting `a; b`
        # start b during cancellation (#11252). Stop the group first to reduce this race;
        # this is best-effort, not an atomic freeze. SIGKILL also kills stopped processes.
        # Linux holds tasklist_lock across group signalling; child exit notification needs
        # its write lock, so the shell is signalled before observing the killed child's exit.
        try:
            os.killpg(pid, signal.SIGSTOP)
        except OSError:
            pass  # A failed stop must not prevent the kill attempt.
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass  # No member left: the leader and everything it started are gone.
