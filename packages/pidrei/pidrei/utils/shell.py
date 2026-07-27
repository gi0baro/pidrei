"""Mirror of pi coding-agent src/utils/shell.ts (POSIX-only; the win32 Git-Bash
discovery, legacy-WSL stdin transport, and taskkill paths are not ported)."""

import os
import shutil
import signal
import threading
from dataclasses import dataclass

from ..config import get_bin_dir


@dataclass(slots=True)
class ShellConfig:
    shell: str
    args: list[str]
    command_transport: str | None = None  # "argv" | "stdin"


def _get_bash_shell_config(shell: str) -> ShellConfig:
    return ShellConfig(shell=shell, args=["-c"])


# Sync by design: `BashTool.exec` calls `get_shell_config` through
# `spawn_blocking`, so the PATH walk never runs on a runtime worker. Any new
# caller must do the same.
def _find_bash_on_path() -> str | None:
    return shutil.which("bash")


def get_shell_config(custom_shell_path: str | None = None) -> ShellConfig:
    """Resolve shell configuration: user shellPath, then /bin/bash, then bash
    on PATH, then sh."""
    if custom_shell_path:
        if os.path.exists(custom_shell_path):
            return _get_bash_shell_config(custom_shell_path)
        raise Exception(f"Custom shell path not found: {custom_shell_path}")

    if os.path.exists("/bin/bash"):
        return _get_bash_shell_config("/bin/bash")

    bash_on_path = _find_bash_on_path()
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


def sanitize_binary_output(value: str) -> str:
    """Sanitize binary output for display/storage: drop control characters
    (except tab/newline/CR), lone surrogates, and Unicode format characters
    that crash width measurement."""
    output: list[str] = []
    for char in value:
        code = ord(char)
        if code in (0x09, 0x0A, 0x0D):
            output.append(char)
            continue
        if code <= 0x1F:
            continue
        if 0xFFF9 <= code <= 0xFFFB:
            continue
        if 0xD800 <= code <= 0xDFFF:  # Lone surrogates from lossy decoding
            continue
        output.append(char)
    return "".join(output)


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


def kill_process_tree(pid: int) -> None:
    """Kill a process and all its children (POSIX process-group SIGKILL)."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        # Fallback to killing just the child if process group kill fails
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass  # Process already dead
