"""Mirror of pi coding-agent src/utils/clipboard-image.ts (POSIX only).

Deviation: pi's optional native clipboard addon (clipboard-rs) has no Python
counterpart; the platform tools (wl-paste, xclip, PowerShell on WSL) carry
the whole load here, and Pillow replaces Photon for PNG conversion.
Clipboard images are ``{"bytes", "mimeType"}`` records.
"""

import os
import re
import subprocess
import sys
import tempfile
import uuid

import tonio.colored as tonio

from .image_process import convert_image_bytes_to_png


SUPPORTED_IMAGE_MIME_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")

_DEFAULT_LIST_TIMEOUT_S = 1.0
_DEFAULT_READ_TIMEOUT_S = 3.0
_DEFAULT_POWERSHELL_TIMEOUT_S = 5.0


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


# Sync by design: reached only from `_read_clipboard_image_sync`, which the
# async entry point hands to `spawn_blocking`.
def _run_command(command: list, *, timeout_s: float = _DEFAULT_READ_TIMEOUT_S, env=None) -> dict:
    try:
        result = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            timeout=timeout_s,
            env=env,
            check=False,
        )
    except OSError, subprocess.TimeoutExpired:
        return {"ok": False, "stdout": b""}

    if result.returncode != 0:
        return {"ok": False, "stdout": b""}

    return {"ok": True, "stdout": result.stdout}


def _read_clipboard_image_via_wl_paste() -> dict | None:
    listed = _run_command(["wl-paste", "--list-types"], timeout_s=_DEFAULT_LIST_TIMEOUT_S)
    if not listed["ok"]:
        return None

    types = [t.strip() for t in re.split(r"\r?\n", listed["stdout"].decode("utf-8", "replace")) if t.strip()]

    selected_type = _select_preferred_image_mime_type(types)
    if not selected_type:
        return None

    data = _run_command(["wl-paste", "--type", selected_type, "--no-newline"])
    if not data["ok"] or len(data["stdout"]) == 0:
        return None

    return {"bytes": data["stdout"], "mimeType": _base_mime_type(selected_type)}


def _is_wsl(env=None) -> bool:
    env = env if env is not None else os.environ
    if env.get("WSL_DISTRO_NAME") or env.get("WSLENV"):
        return True

    try:
        with open("/proc/version", encoding="utf-8") as f:
            release = f.read()
        return re.search(r"microsoft|wsl", release, re.IGNORECASE) is not None
    except OSError:
        return False


def _read_clipboard_image_via_powershell() -> dict | None:
    """WSL fallback: PowerShell can access the Windows clipboard directly.

    On WSL, the Linux clipboard (Wayland/X11) does not receive image data
    from Windows screenshots (Win+Shift+S).
    """
    tmp_file = os.path.join(tempfile.gettempdir(), f"pidrei-wsl-clip-{uuid.uuid4()}.png")

    try:
        win_path_result = _run_command(["wslpath", "-w", tmp_file], timeout_s=_DEFAULT_LIST_TIMEOUT_S)
        if not win_path_result["ok"]:
            return None

        win_path = win_path_result["stdout"].decode("utf-8", "replace").strip()
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

        result = _run_command(
            ["powershell.exe", "-NoProfile", "-Command", ps_script],
            timeout_s=_DEFAULT_POWERSHELL_TIMEOUT_S,
        )
        if not result["ok"]:
            return None

        output = result["stdout"].decode("utf-8", "replace").strip()
        if output != "ok":
            return None

        with open(tmp_file, "rb") as f:
            data = f.read()
        if len(data) == 0:
            return None

        return {"bytes": data, "mimeType": "image/png"}
    except OSError:
        return None
    finally:
        try:
            os.unlink(tmp_file)
        except OSError:
            pass


def _read_clipboard_image_via_xclip() -> dict | None:
    targets = _run_command(
        ["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"], timeout_s=_DEFAULT_LIST_TIMEOUT_S
    )

    candidate_types: list = []
    if targets["ok"]:
        candidate_types = [
            t.strip() for t in re.split(r"\r?\n", targets["stdout"].decode("utf-8", "replace")) if t.strip()
        ]

    preferred = _select_preferred_image_mime_type(candidate_types) if candidate_types else None
    try_types = [preferred, *SUPPORTED_IMAGE_MIME_TYPES] if preferred else list(SUPPORTED_IMAGE_MIME_TYPES)

    for mime_type in try_types:
        data = _run_command(["xclip", "-selection", "clipboard", "-t", mime_type, "-o"])
        if data["ok"] and len(data["stdout"]) > 0:
            return {"bytes": data["stdout"], "mimeType": _base_mime_type(mime_type)}

    return None


def _read_clipboard_image_via_pngpaste() -> dict | None:
    """macOS: pngpaste if installed (stand-in for pi's native addon)."""
    data = _run_command(["pngpaste", "-"])
    if data["ok"] and len(data["stdout"]) > 0:
        return {"bytes": data["stdout"], "mimeType": "image/png"}
    return None


async def read_clipboard_image(options: dict | None = None) -> dict | None:
    """Probing the clipboard is one blocking unit — a `/proc` read plus a
    platform tool — so it goes to the pool whole. Replaces an earlier partial
    fix that offloaded only `_is_wsl` and left its subprocess siblings inline.
    """
    return await tonio.spawn_blocking(_read_clipboard_image_sync, options)


def _read_clipboard_image_sync(options: dict | None = None) -> dict | None:
    options = options or {}
    env = options.get("env") if options.get("env") is not None else os.environ
    platform = options.get("platform") if options.get("platform") is not None else sys.platform

    if env.get("TERMUX_VERSION"):
        return None

    image: dict | None = None

    if platform == "linux":
        wsl = _is_wsl(env)
        wayland = is_wayland_session(env)

        if wayland or wsl:
            image = _read_clipboard_image_via_wl_paste() or _read_clipboard_image_via_xclip()

        if image is None and wsl:
            image = _read_clipboard_image_via_powershell()

        if image is None and not wayland:
            image = _read_clipboard_image_via_xclip()
    else:
        image = _read_clipboard_image_via_pngpaste()

    if image is None:
        return None

    # Convert unsupported formats (e.g., BMP from WSLg) to PNG
    if not _is_supported_image_mime_type(image["mimeType"]):
        png_bytes = convert_image_bytes_to_png(image["bytes"])
        if png_bytes is None:
            return None
        return {"bytes": png_bytes, "mimeType": "image/png"}

    return image
