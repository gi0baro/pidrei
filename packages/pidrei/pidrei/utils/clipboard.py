"""Mirror of pi coding-agent src/utils/clipboard.ts (POSIX only).

Deviation: pi's optional native clipboard addon has no Python counterpart —
reads and writes go through the platform tools (pbcopy/pbpaste, wl-copy/
wl-paste, xclip/xsel, termux) with the same OSC 52 fallback for remote
sessions.
"""

import base64
import os
import subprocess
import sys
from collections.abc import Awaitable

import tonio.colored as tonio

from .clipboard_image import is_wayland_session


_MAX_OSC52_ENCODED_LENGTH = 100_000
_EXEC_TIMEOUT_S = 5.0


# Sync by design: only ever runs inside `_copy_to_clipboard_sync`, which the
# async entry point hands to `spawn_blocking`.
def _run_with_input(command: list, text: str) -> bool:
    try:
        subprocess.run(  # noqa: S603
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


# Sync by design: only ever runs inside `_read_clipboard_text_sync`, which the
# async entry point hands to `spawn_blocking`.
def _read_output(command: list) -> str | None:
    try:
        result = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            timeout=_EXEC_TIMEOUT_S,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", "replace")


def _copy_to_x11_clipboard(text: str) -> bool:
    if _run_with_input(["xclip", "-selection", "clipboard"], text):
        return True
    return _run_with_input(["xsel", "--clipboard", "--input"], text)


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


def read_clipboard_text() -> Awaitable[str | None]:
    """Read plain text from the system clipboard, if a reader is available.

    Picking a platform tool and running it is one blocking unit, so it goes to
    the pool whole rather than a hop per candidate command.
    """
    return tonio.spawn_blocking(_read_clipboard_text_sync)


def _read_clipboard_text_sync() -> str | None:
    if sys.platform == "darwin":
        text = _read_output(["pbpaste"])
    elif os.environ.get("TERMUX_VERSION"):
        text = _read_output(["termux-clipboard-get"])
    elif is_wayland_session() and os.environ.get("WAYLAND_DISPLAY"):
        text = _read_output(["wl-paste", "--no-newline"])
    elif os.environ.get("DISPLAY"):
        text = _read_output(["xclip", "-selection", "clipboard", "-o"])
        if text is None:
            text = _read_output(["xsel", "--clipboard", "--output"])
    else:
        text = None

    return text or None


def copy_to_clipboard(text: str) -> Awaitable[None]:
    """Offloaded whole, like `read_clipboard_text`."""
    return tonio.spawn_blocking(_copy_to_clipboard_sync, text)


def _copy_to_clipboard_sync(text: str) -> None:
    copied = False

    remote = _is_remote_session()

    if sys.platform == "darwin":
        copied = _run_with_input(["pbcopy"], text)
    else:
        # Linux. Try Termux, Wayland, or X11 clipboard tools.
        if os.environ.get("TERMUX_VERSION"):
            copied = _run_with_input(["termux-clipboard-set"], text)

        if not copied:
            has_wayland_display = bool(os.environ.get("WAYLAND_DISPLAY"))
            has_x11_display = bool(os.environ.get("DISPLAY"))
            wayland = is_wayland_session()
            if wayland and has_wayland_display:
                copied = _run_with_input(["wl-copy"], text)
                if not copied and has_x11_display:
                    copied = _copy_to_x11_clipboard(text)
            elif has_x11_display:
                copied = _copy_to_x11_clipboard(text)

    if remote or not copied:
        osc52_copied = _emit_osc52(text)
        copied = copied or osc52_copied

    if not copied:
        raise Exception("Failed to copy to clipboard")
