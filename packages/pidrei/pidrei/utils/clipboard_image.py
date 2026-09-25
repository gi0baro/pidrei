"""Mirror of pi coding-agent src/utils/clipboard-image.ts (POSIX only).

Deviation: pi's native platform clipboard helper (pi-tui `getNativeClipboard`)
has no Python counterpart. On Linux pi only reaches it after every platform
tool reported a failure, so here that last resort reads as "no image"; on
macOS `pngpaste` stands in for it. Pillow replaces Photon for PNG conversion.
Clipboard images are ``{"bytes", "mimeType"}`` records.

The readers keep pi's tri-state: ``_FAILED`` means the backend failed (try the
next one), ``None`` means it answered with no image (stop).
"""

import os
import re
import sys
import tempfile
import uuid

import tonio.colored as tonio
from tonio.colored import fs

from .clipboard_command import run_clipboard_command
from .image_process import convert_image_bytes_to_png
from .wsl import is_wsl


SUPPORTED_IMAGE_MIME_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")

_DEFAULT_LIST_TIMEOUT_MS = 1000
_DEFAULT_POWERSHELL_TIMEOUT_MS = 5000

#: pi's `undefined` reader result: the backend failed, so the chain continues.
_FAILED = object()


def is_wayland_session(env=None) -> bool:
    env = env if env is not None else os.environ
    return bool(env.get("WAYLAND_DISPLAY")) or env.get("XDG_SESSION_TYPE") == "wayland"


def _base_mime_type(mime_type: str) -> str:
    return mime_type.split(";")[0].strip().lower()


def extension_for_image_mime_type(mime_type: str) -> str | None:
    base = _base_mime_type(mime_type)
    if base == "image/png":
        return "png"
    if base == "image/jpeg":
        return "jpg"
    if base == "image/webp":
        return "webp"
    if base == "image/gif":
        return "gif"
    return None


def _select_preferred_image_mime_type(mime_types: list) -> str | None:
    normalized = [{"raw": t.strip(), "base": _base_mime_type(t.strip())} for t in mime_types if t.strip()]

    for preferred in SUPPORTED_IMAGE_MIME_TYPES:
        match = next((t for t in normalized if t["base"] == preferred), None)
        if match is not None:
            return match["raw"]

    any_image = next((t for t in normalized if t["base"].startswith("image/")), None)
    return any_image["raw"] if any_image is not None else None


def _is_supported_image_mime_type(mime_type: str) -> bool:
    return _base_mime_type(mime_type) in SUPPORTED_IMAGE_MIME_TYPES


# _FAILED means the backend failed; None means it has no image. An empty
# Wayland clipboard must not fall through to stale X11 clipboard contents.
async def _read_clipboard_image_via_wl_paste() -> dict | object | None:
    listed = await run_clipboard_command("wl-paste", ["--list-types"], timeout_ms=_DEFAULT_LIST_TIMEOUT_MS)
    if listed is None:
        return _FAILED

    types = [t.strip() for t in re.split(r"\r?\n", listed.decode("utf-8", "replace")) if t.strip()]

    selected_type = _select_preferred_image_mime_type(types)
    if not selected_type:
        return None

    data = await run_clipboard_command("wl-paste", ["--type", selected_type, "--no-newline"])
    if data is None:
        return _FAILED
    if len(data) == 0:
        return None

    return {"bytes": data, "mimeType": _base_mime_type(selected_type)}


async def _read_clipboard_image_via_powershell() -> dict | None:
    """WSL fallback: PowerShell can access the Windows clipboard directly.

    On WSL, the Linux clipboard (Wayland/X11) does not receive image data
    from Windows screenshots (Win+Shift+S).
    """
    tmp_file = os.path.join(tempfile.gettempdir(), f"pidrei-wsl-clip-{uuid.uuid4()}.png")

    try:
        win_path_result = await run_clipboard_command("wslpath", ["-w", tmp_file], timeout_ms=_DEFAULT_LIST_TIMEOUT_MS)
        if win_path_result is None:
            return None

        win_path = win_path_result.decode("utf-8", "replace").strip()
        if not win_path:
            return None

        ps_quoted_win_path = win_path.replace("'", "''")
        ps_script = "; ".join(
            [
                "Add-Type -AssemblyName System.Windows.Forms",
                "Add-Type -AssemblyName System.Drawing",
                f"$path = '{ps_quoted_win_path}'",
                "$img = [System.Windows.Forms.Clipboard]::GetImage()",
                (
                    "if ($img) { $img.Save($path, [System.Drawing.Imaging.ImageFormat]::Png); "
                    "Write-Output 'ok' } else { Write-Output 'empty' }"
                ),
            ]
        )

        result = await run_clipboard_command(
            "powershell.exe", ["-NoProfile", "-Command", ps_script], timeout_ms=_DEFAULT_POWERSHELL_TIMEOUT_MS
        )
        if result is None:
            return None

        output = result.decode("utf-8", "replace").strip()
        if output != "ok":
            return None

        data = await fs.Path(tmp_file).read_bytes()
        if len(data) == 0:
            return None

        return {"bytes": data, "mimeType": "image/png"}
    except OSError:
        return None
    finally:
        try:
            await fs.Path(tmp_file).unlink()
        except OSError:
            pass


async def _read_clipboard_image_via_xclip() -> dict | object | None:
    targets = await run_clipboard_command(
        "xclip", ["-selection", "clipboard", "-t", "TARGETS", "-o"], timeout_ms=_DEFAULT_LIST_TIMEOUT_MS
    )

    candidate_types: list = []
    if targets is not None:
        candidate_types = [t.strip() for t in re.split(r"\r?\n", targets.decode("utf-8", "replace")) if t.strip()]

    preferred = _select_preferred_image_mime_type(candidate_types)
    if targets is not None and not preferred:
        return None
    # dict.fromkeys: ordered de-duplication (pi's `new Set`).
    try_types = dict.fromkeys([preferred, *SUPPORTED_IMAGE_MIME_TYPES] if preferred else SUPPORTED_IMAGE_MIME_TYPES)

    for mime_type in try_types:
        data = await run_clipboard_command("xclip", ["-selection", "clipboard", "-t", mime_type, "-o"])
        if data is not None and len(data) > 0:
            return {"bytes": data, "mimeType": _base_mime_type(mime_type)}

    return _FAILED


async def _read_clipboard_image_via_pngpaste() -> dict | None:
    """macOS: pngpaste if installed (stand-in for pi's native reader)."""
    data = await run_clipboard_command("pngpaste", ["-"])
    if data:
        return {"bytes": data, "mimeType": "image/png"}
    return None


async def read_clipboard_image(options: dict | None = None) -> dict | None:
    """Probe the clipboard for an image.

    Every branch here is a subprocess, so the chain runs async through
    `run_command` rather than going to the pool whole. The two things in it that
    are *not* subprocesses keep their offload, which is the trap an earlier
    partial fix fell into from the other side: `is_wsl` reads `/proc/version`,
    and `convert_image_bytes_to_png` is CPU-bound.
    """
    options = options or {}
    env = options.get("env") if options.get("env") is not None else os.environ
    platform = options.get("platform") if options.get("platform") is not None else sys.platform

    if env.get("TERMUX_VERSION"):
        return None

    image: dict | object | None = _FAILED

    if platform == "linux":
        wsl = await tonio.spawn_blocking(is_wsl, env)
        if is_wayland_session(env) or wsl:
            image = await _read_clipboard_image_via_wl_paste()
        if image is _FAILED:
            image = await _read_clipboard_image_via_xclip()
        # Preserve Linux's empty/unavailable distinction if Windows has no image.
        if not isinstance(image, dict) and wsl:
            image = await _read_clipboard_image_via_powershell() or image
        # pi's native X11 reader is the last resort here; with none, a
        # failed chain reads as "no image".
    else:
        image = await _read_clipboard_image_via_pngpaste()

    if not isinstance(image, dict):
        return None

    # Convert unsupported formats (e.g., Windows DIB data wrapped as BMP) to PNG
    if not _is_supported_image_mime_type(image["mimeType"]):
        png_bytes = await tonio.spawn_blocking(convert_image_bytes_to_png, image["bytes"])
        if png_bytes is None:
            return None
        return {"bytes": png_bytes, "mimeType": "image/png"}

    return image
