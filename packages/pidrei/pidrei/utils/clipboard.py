"""Mirror of pi coding-agent src/utils/clipboard.ts (POSIX only).

Deviation: pi's optional native clipboard addon has no Python counterpart —
reads and writes go through the platform tools (pbcopy/pbpaste, wl-copy/
wl-paste, xclip/xsel, termux) with the same OSC 52 fallback for remote
sessions.

The detection chain is async rather than one offloaded unit: every branch of it
is a subprocess, which `utils.process.run_command` runs without holding a
blocking-pool thread. The probes that are *not* subprocesses stay where they
were — `is_wayland_session` only reads env vars, and the OSC 52 write is still
offloaded because it is a blocking write to the terminal.
"""

import base64
import os
import subprocess
import sys

import tonio.colored as tonio

from .clipboard_image import is_wayland_session
from .process import run_command


_MAX_OSC52_ENCODED_LENGTH = 100_000
_EXEC_TIMEOUT_S = 5.0


async def _run_with_input(command: list, text: str) -> bool:
    try:
        await run_command(
            command,
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_EXEC_TIMEOUT_S,
            check=True,
        )
        return True
    except OSError, subprocess.SubprocessError:
        return False


async def _read_output(command: list) -> str | None:
    try:
        result = await run_command(command, capture_output=True, timeout=_EXEC_TIMEOUT_S)
    except OSError, subprocess.SubprocessError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", "replace")


async def _copy_to_x11_clipboard(text: str) -> bool:
    if await _run_with_input(["xclip", "-selection", "clipboard"], text):
        return True
    return await _run_with_input(["xsel", "--clipboard", "--input"], text)


def _is_remote_session(env=None) -> bool:
    env = env if env is not None else os.environ
    return bool(env.get("SSH_CONNECTION") or env.get("SSH_CLIENT") or env.get("MOSH_CONNECTION"))


def _emit_osc52(text: str) -> bool:
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    if len(encoded) > _MAX_OSC52_ENCODED_LENGTH:
        return False
    sys.stdout.write(f"\x1b]52;c;{encoded}\x07")
    sys.stdout.flush()
    return True


async def read_clipboard_text() -> str | None:
    """Read plain text from the system clipboard, if a reader is available."""
    if sys.platform == "darwin":
        text = await _read_output(["pbpaste"])
    elif os.environ.get("TERMUX_VERSION"):
        text = await _read_output(["termux-clipboard-get"])
    elif is_wayland_session() and os.environ.get("WAYLAND_DISPLAY"):
        text = await _read_output(["wl-paste", "--no-newline", "--type", "text"])
    elif os.environ.get("DISPLAY"):
        text = await _read_output(["xclip", "-selection", "clipboard", "-o"])
        if text is None:
            text = await _read_output(["xsel", "--clipboard", "--output"])
    else:
        text = None

    return text or None


async def copy_to_clipboard(text: str) -> None:
    copied = False

    remote = _is_remote_session()

    if sys.platform == "darwin":
        copied = await _run_with_input(["pbcopy"], text)
    else:
        # Linux. Try Termux, Wayland, or X11 clipboard tools.
        if os.environ.get("TERMUX_VERSION"):
            copied = await _run_with_input(["termux-clipboard-set"], text)

        if not copied:
            has_wayland_display = bool(os.environ.get("WAYLAND_DISPLAY"))
            has_x11_display = bool(os.environ.get("DISPLAY"))
            wayland = is_wayland_session()
            if wayland and has_wayland_display:
                copied = await _run_with_input(["wl-copy"], text)
                if not copied and has_x11_display:
                    copied = await _copy_to_x11_clipboard(text)
            elif has_x11_display:
                copied = await _copy_to_x11_clipboard(text)

    if remote or not copied:
        # Still offloaded: this is a blocking write to the terminal, and it is
        # the one part of the chain a subprocess primitive does not cover.
        osc52_copied = await tonio.spawn_blocking(_emit_osc52, text)
        copied = copied or osc52_copied

    if not copied:
        raise Exception("Failed to copy to clipboard")
