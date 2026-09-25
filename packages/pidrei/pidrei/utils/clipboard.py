"""Mirror of pi coding-agent src/utils/clipboard.ts (POSIX only).

Deviation: pi's native platform clipboard helper (pi-tui `getNativeClipboard`)
has no Python counterpart. On Linux pi only reaches it after every platform
tool failed, so dropping it changes nothing but that last resort; on macOS it
is pi's primary path, and `pbpaste`/`pbcopy` stand in for it (pi already
falls back to `pbcopy` for writes).

Every candidate is a subprocess run through `run_clipboard_command`, which
does not hold a blocking-pool thread; the OSC 52 write stays offloaded because
it is a blocking write to the terminal.
"""

import base64
import os
import sys

import tonio.colored as tonio

from .clipboard_command import run_clipboard_command


_MAX_OSC52_ENCODED_LENGTH = 100_000


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
    """Read plain text from the system clipboard."""
    commands: list[tuple[str, list[str]]] = []
    if sys.platform == "linux":
        if os.environ.get("TERMUX_VERSION"):
            commands.append(("termux-clipboard-get", []))
        if os.environ.get("WAYLAND_DISPLAY"):
            commands.append(("wl-paste", ["--no-newline", "--type", "text"]))
        if os.environ.get("DISPLAY"):
            commands += [("xclip", ["-selection", "clipboard", "-out"]), ("xsel", ["--clipboard", "--output"])]
    elif sys.platform == "darwin":
        commands.append(("pbpaste", []))  # stand-in for pi's native reader
    for command, args in commands:
        data = await run_clipboard_command(command, args, timeout_ms=5000)
        if data is not None:
            return data.decode("utf-8", "replace") or None
    return None


async def copy_to_clipboard(text: str) -> None:
    p = sys.platform
    copied = False
    # pi tries its native writer first on non-Linux platforms; pbcopy is
    # already its command fallback there.
    commands: list[tuple[str, list[str]]] = []
    if p == "darwin":
        commands.append(("pbcopy", []))
    else:
        if os.environ.get("TERMUX_VERSION"):
            commands.append(("termux-clipboard-set", []))
        if os.environ.get("WAYLAND_DISPLAY"):
            commands.append(("wl-copy", []))
        if os.environ.get("DISPLAY"):
            commands += [("xclip", ["-selection", "clipboard"]), ("xsel", ["--clipboard", "--input"])]
    for command, args in commands:
        if await run_clipboard_command(command, args, input=text, timeout_ms=5000) is not None:
            copied = True
            break
    if _is_remote_session() or not copied:
        # Still offloaded: this is a blocking write to the terminal.
        copied = await tonio.spawn_blocking(_emit_osc52, text) or copied
    if not copied:
        raise Exception("Failed to copy to clipboard")
