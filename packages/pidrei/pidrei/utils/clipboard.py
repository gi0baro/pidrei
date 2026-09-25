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
import uuid

import tonio.colored as tonio
from tonio.colored import fs

from ..config import TEMP_DIR
from .clipboard_command import run_clipboard_command
from .temp_file_writer import discard_temp_file
from .wsl import is_wsl


_MAX_OSC52_ENCODED_LENGTH = 100_000


def _is_remote_session(env) -> bool:
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


def _write_private_text(path: fs.Path, text: str) -> None:
    # Created 0600 (it holds the copied text), which `fs.Path.write_text` cannot do.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


async def _copy_via_windows_clipboard(text: str) -> bool:
    """WSL without WSLg has no Linux display, so the Windows clipboard is written
    through interop. PowerShell reads the text from a file because `clip.exe` and
    PowerShell stdin decode piped bytes with the console code page, which mangles
    non-ASCII UTF-8."""
    tmp_file = TEMP_DIR / f"pidrei-wsl-clip-{uuid.uuid4()}.txt"
    try:
        copied = await _copy_through_file(tmp_file, text)
    except BaseException:
        # Cancelled: an await is not served on this path, so the cleanup of the
        # file (it holds the copied text) is detached.
        tonio.spawn.without_tracking(discard_temp_file(tmp_file))
        raise
    await discard_temp_file(tmp_file)
    return copied


async def _copy_through_file(tmp_file: fs.Path, text: str) -> bool:
    try:
        await tonio.spawn_blocking(_write_private_text, tmp_file, text)
        result = await run_clipboard_command("wslpath", ["-w", str(tmp_file)], timeout_ms=1000)
        win_path = result.decode("utf-8", "replace").strip() if result is not None else ""
        if not win_path:
            return False
        quoted = win_path.replace("'", "''")
        script = f"Set-Clipboard -Value ([System.IO.File]::ReadAllText('{quoted}', [System.Text.Encoding]::UTF8))"
        return (
            await run_clipboard_command("powershell.exe", ["-NoProfile", "-Command", script], timeout_ms=5000)
            is not None
        )
    except Exception:
        return False


async def copy_to_clipboard(text: str) -> None:
    p = sys.platform
    env = os.environ
    copied = False
    # pi tries its native writer first on non-Linux platforms; pbcopy is
    # already its command fallback there.
    commands: list[tuple[str, list[str]]] = []
    if p == "darwin":
        commands.append(("pbcopy", []))
    else:
        if env.get("TERMUX_VERSION"):
            commands.append(("termux-clipboard-set", []))
        if env.get("WAYLAND_DISPLAY"):
            commands.append(("wl-copy", []))
        if env.get("DISPLAY"):
            commands += [("xclip", ["-selection", "clipboard"]), ("xsel", ["--clipboard", "--input"])]
    for command, args in commands:
        if await run_clipboard_command(command, args, input=text, timeout_ms=5000) is not None:
            copied = True
            break
    # The OSC 52 writes stay offloaded: each is a blocking write to the terminal.
    osc52_emitted = False
    if not copied and p == "linux" and await tonio.spawn_blocking(is_wsl, env):
        # Windows Terminal supports OSC 52; prefer it over the slower PowerShell round trip.
        if env.get("WT_SESSION"):
            osc52_emitted = await tonio.spawn_blocking(_emit_osc52, text)
        copied = osc52_emitted or await _copy_via_windows_clipboard(text)
    # OSC 52 cannot be verified, so a desktop session with a display reports the failure
    # instead (#9618). Without a display the terminal is the only clipboard route (containers,
    # WSL without WSLg), and remote sessions always emit it to reach the client clipboard.
    headless = (
        p == "linux" and not env.get("DISPLAY") and not env.get("WAYLAND_DISPLAY") and not env.get("TERMUX_VERSION")
    )
    oversized = False
    if not osc52_emitted and (_is_remote_session(env) or (not copied and headless)):
        if await tonio.spawn_blocking(_emit_osc52, text):
            copied = True
        else:
            oversized = True
    if copied:
        return
    if oversized:
        raise Exception("Clipboard unavailable: text exceeds the OSC 52 size limit")
    if p == "linux":
        if env.get("TERMUX_VERSION"):
            raise Exception("Clipboard unavailable: install the Termux:API app and `termux-api` package")
        if env.get("WAYLAND_DISPLAY"):
            raise Exception("Clipboard unavailable: install `wl-clipboard` (`wl-copy`) or check Wayland access")
        if env.get("DISPLAY"):
            raise Exception("Clipboard unavailable: install `xclip` or `xsel`, or check X11 access")
    raise Exception("Clipboard unavailable")
