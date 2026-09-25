"""Mirror of pi coding-agent src/utils/clipboard-command.ts.

Runs one clipboard tool through `utils.process.run_command`, which already
gives pi's contract its teeth: the deadline kills the child without holding a
blocking-pool thread, and `max_output_bytes` is Node's buffer cap. pi kills
with SIGKILL on both, so this does too.
"""

import signal
import subprocess
from collections.abc import Sequence

from .process import run_command


_DEFAULT_TIMEOUT_MS = 3000
_DEFAULT_MAX_BUFFER_BYTES = 50 * 1024 * 1024


async def run_clipboard_command(
    command: str,
    args: Sequence[str],
    *,
    input: str | None = None,
    timeout_ms: int | None = None,
    max_buffer_bytes: int | None = None,
) -> bytes | None:
    """None means the command failed; an empty bytes value is a successful result."""
    try:
        result = await run_command(
            [command, *args],
            # stdin is always a pipe, closed after the input (if any) is written.
            input=(input or "").encode("utf-8"),
            # Clipboard writers can daemonize. Do not give them output pipes to retain.
            stdout=subprocess.PIPE if input is None else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=(timeout_ms if timeout_ms is not None else _DEFAULT_TIMEOUT_MS) / 1000,
            kill_signal=signal.SIGKILL,
            max_output_bytes=max_buffer_bytes if max_buffer_bytes is not None else _DEFAULT_MAX_BUFFER_BYTES,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout if result.stdout is not None else b""
