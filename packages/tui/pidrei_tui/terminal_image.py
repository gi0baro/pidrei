"""Mirror of pi tui src/terminal-image.ts.

Records (camelCase like pi): TerminalCapabilities = {"images": "kitty" |
"iterm2" | None, "trueColor": bool, "hyperlinks": bool}; CellDimensions /
ImageDimensions = {"widthPx": int, "heightPx": int}; image cell size =
{"columns": int, "rows": int}; render_image result = {"sequence": str,
"rows": int, "imageId": int | None}.

Port notes: JS ``Buffer.from(base64)`` never raises — the dimension sniffers
wrap ``base64.b64decode`` in try/except instead; ``Math.random``-based image
ids use ``random`` (collision avoidance, not security).
"""

import base64
import binascii
import math
import os
import random
import subprocess


# Default cell dimensions - updated by TUI when terminal responds to query
_cell_dimensions = {"widthPx": 9, "heightPx": 18}

_cached_capabilities: dict | None = None


def get_cell_dimensions() -> dict:
    return _cell_dimensions


def set_cell_dimensions(dims: dict) -> None:
    global _cell_dimensions
    _cell_dimensions = dims


def _probe_tmux_hyperlinks() -> bool:
    """Check whether the attached tmux client forwards OSC 8 hyperlinks.

    tmux only re-emits them when its `client_termfeatures` lists
    `hyperlinks`, and strips them otherwise. On any error falls back False.
    """
    try:
        result = subprocess.run(
            ["tmux", "display-message", "-p", "#{client_termfeatures}"],  # noqa: S607 - PATH lookup like pi's execSync
            capture_output=True,
            encoding="utf-8",
            timeout=0.25,
            stdin=subprocess.DEVNULL,
            check=True,
        )
        return "hyperlinks" in [feature.strip() for feature in result.stdout.split(",")]
    except Exception:
        return False


def detect_capabilities(tmux_forwards_hyperlink=_probe_tmux_hyperlinks) -> dict:
    term_program = (os.environ.get("TERM_PROGRAM") or "").lower()
    terminal_emulator = (os.environ.get("TERMINAL_EMULATOR") or "").lower()
    term = (os.environ.get("TERM") or "").lower()
    color_term = (os.environ.get("COLORTERM") or "").lower()
    has_true_color_hint = color_term in ("truecolor", "24bit")

    # Emit OSC 8 hyperlinks only when tmux confirms it forwards.
    # Image protocols are unreliable under tmux, so leave `images: None`.
    if os.environ.get("TMUX") or term.startswith("tmux"):
        return {"images": None, "trueColor": has_true_color_hint, "hyperlinks": tmux_forwards_hyperlink()}

    # screen does not forward OSC 8 hyperlinks, so keep them off there.
    if term.startswith("screen"):
        return {"images": None, "trueColor": has_true_color_hint, "hyperlinks": False}

    if os.environ.get("KITTY_WINDOW_ID") or term_program == "kitty":
        return {"images": "kitty", "trueColor": True, "hyperlinks": True}

    if term_program == "ghostty" or "ghostty" in term or os.environ.get("GHOSTTY_RESOURCES_DIR"):
        return {"images": "kitty", "trueColor": True, "hyperlinks": True}

    if os.environ.get("WEZTERM_PANE") or term_program == "wezterm":
        return {"images": "kitty", "trueColor": True, "hyperlinks": True}

    # Warp supports the Kitty graphics protocol and OSC 8 hyperlinks.
    if (
        term_program == "warpterminal"
        or os.environ.get("WARP_SESSION_ID")
        or os.environ.get("WARP_TERMINAL_SESSION_UUID")
    ):
        return {"images": "kitty", "trueColor": True, "hyperlinks": True}

    if os.environ.get("ITERM_SESSION_ID") or term_program == "iterm.app":
        return {"images": "iterm2", "trueColor": True, "hyperlinks": True}

    if os.environ.get("WT_SESSION"):
        return {"images": None, "trueColor": True, "hyperlinks": True}

    if term_program == "vscode":
        return {"images": None, "trueColor": True, "hyperlinks": True}

    if term_program == "alacritty":
        return {"images": None, "trueColor": True, "hyperlinks": True}

    if terminal_emulator == "jetbrains-jediterm":
        return {"images": None, "trueColor": True, "hyperlinks": False}

    # Unknown terminal: be conservative. OSC 8 is rendered invisibly as "just
    # text" on terminals that swallow it, which means the URL disappears from
    # the rendered output. Default to the legacy `text (url)` behavior unless we
    # have positively identified a hyperlink-capable terminal above.
    return {"images": None, "trueColor": has_true_color_hint, "hyperlinks": False}


def get_capabilities() -> dict:
    global _cached_capabilities
    if _cached_capabilities is None:
        _cached_capabilities = detect_capabilities()
    return _cached_capabilities


def reset_capabilities_cache() -> None:
    global _cached_capabilities
    _cached_capabilities = None


def set_capabilities(caps: dict) -> None:
    """Override the cached capabilities. Useful in tests to exercise both code paths."""
    global _cached_capabilities
    _cached_capabilities = caps


KITTY_PREFIX = "\x1b_G"
ITERM2_PREFIX = "\x1b]1337;File="


def is_image_line(line: str) -> bool:
    # Fast path: sequence at line start (single-row images)
    if line.startswith((KITTY_PREFIX, ITERM2_PREFIX)):
        return True
    # Slow path: sequence elsewhere (multi-row images have cursor-up prefix)
    return KITTY_PREFIX in line or ITERM2_PREFIX in line


def allocate_image_id() -> int:
    """Generate a random image ID for Kitty graphics protocol.

    Uses random IDs to avoid collisions between different module instances
    (e.g., main app vs extensions).
    """
    # Use random ID in range [1, 0xffffffff] to avoid collisions
    return random.randint(1, 0xFFFFFFFE)  # noqa: S311


def encode_kitty(
    base64_data: str,
    *,
    columns: int | None = None,
    rows: int | None = None,
    image_id: int | None = None,
    move_cursor: bool | None = None,
) -> str:
    """Encode a Kitty graphics transmit-and-display sequence.

    ``move_cursor`` controls whether Kitty should apply its default cursor
    movement after placement (default True).
    """
    chunk_size = 4096

    params = ["a=T", "f=100", "q=2"]

    if move_cursor is False:
        params.append("C=1")
    if columns:
        params.append(f"c={columns}")
    if rows:
        params.append(f"r={rows}")
    if image_id:
        params.append(f"i={image_id}")

    if len(base64_data) <= chunk_size:
        return f"\x1b_G{','.join(params)};{base64_data}\x1b\\"

    chunks: list[str] = []
    offset = 0
    is_first = True

    while offset < len(base64_data):
        chunk = base64_data[offset : offset + chunk_size]
        is_last = offset + chunk_size >= len(base64_data)

        if is_first:
            chunks.append(f"\x1b_G{','.join(params)},m=1;{chunk}\x1b\\")
            is_first = False
        elif is_last:
            chunks.append(f"\x1b_Gm=0;{chunk}\x1b\\")
        else:
            chunks.append(f"\x1b_Gm=1;{chunk}\x1b\\")

        offset += chunk_size

    return "".join(chunks)


def delete_kitty_image(image_id: int) -> str:
    """Delete a Kitty graphics image by ID (uppercase 'I' also frees the data)."""
    return f"\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\"


def delete_all_kitty_images() -> str:
    """Delete all visible Kitty graphics images (uppercase 'A' also frees the data)."""
    return "\x1b_Ga=d,d=A,q=2\x1b\\"


def encode_iterm2(
    base64_data: str,
    *,
    width: int | str | None = None,
    height: int | str | None = None,
    name: str | None = None,
    preserve_aspect_ratio: bool | None = None,
    inline: bool | None = None,
) -> str:
    params = [f"inline={1 if inline is not False else 0}"]

    if width is not None:
        params.append(f"width={width}")
    if height is not None:
        params.append(f"height={height}")
    if name:
        name_base64 = base64.b64encode(name.encode("utf-8")).decode("ascii")
        params.append(f"name={name_base64}")
    if preserve_aspect_ratio is False:
        params.append("preserveAspectRatio=0")

    return f"\x1b]1337;File={';'.join(params)}:{base64_data}\x07"


def calculate_image_cell_size(
    image_dimensions: dict,
    max_width_cells: int,
    max_height_cells: int | None = None,
    cell_dimensions: dict | None = None,
) -> dict:
    if cell_dimensions is None:
        cell_dimensions = {"widthPx": 9, "heightPx": 18}
    max_width = max(1, math.floor(max_width_cells))
    max_height = None if max_height_cells is None else max(1, math.floor(max_height_cells))
    image_width = max(1, image_dimensions["widthPx"])
    image_height = max(1, image_dimensions["heightPx"])

    width_scale = (max_width * cell_dimensions["widthPx"]) / image_width
    height_scale = width_scale if max_height is None else (max_height * cell_dimensions["heightPx"]) / image_height
    scale = min(width_scale, height_scale)

    scaled_width_px = image_width * scale
    scaled_height_px = image_height * scale
    columns = math.ceil(scaled_width_px / cell_dimensions["widthPx"])
    rows = math.ceil(scaled_height_px / cell_dimensions["heightPx"])

    return {
        "columns": max(1, min(max_width, columns)),
        "rows": max(1, rows if max_height is None else min(max_height, rows)),
    }


def calculate_image_rows(
    image_dimensions: dict,
    target_width_cells: int,
    cell_dimensions: dict | None = None,
) -> int:
    return calculate_image_cell_size(image_dimensions, target_width_cells, None, cell_dimensions)["rows"]


def _decode_base64(base64_data: str) -> bytes | None:
    try:
        return base64.b64decode(base64_data)
    except binascii.Error, ValueError:
        return None


def get_png_dimensions(base64_data: str) -> dict | None:
    buffer = _decode_base64(base64_data)
    if buffer is None or len(buffer) < 24:
        return None

    if buffer[0] != 0x89 or buffer[1] != 0x50 or buffer[2] != 0x4E or buffer[3] != 0x47:
        return None

    width = int.from_bytes(buffer[16:20], "big")
    height = int.from_bytes(buffer[20:24], "big")

    return {"widthPx": width, "heightPx": height}


def get_jpeg_dimensions(base64_data: str) -> dict | None:
    buffer = _decode_base64(base64_data)
    if buffer is None or len(buffer) < 2:
        return None

    if buffer[0] != 0xFF or buffer[1] != 0xD8:
        return None

    offset = 2
    while offset < len(buffer) - 9:
        if buffer[offset] != 0xFF:
            offset += 1
            continue

        marker = buffer[offset + 1]

        if 0xC0 <= marker <= 0xC2:
            height = int.from_bytes(buffer[offset + 5 : offset + 7], "big")
            width = int.from_bytes(buffer[offset + 7 : offset + 9], "big")
            return {"widthPx": width, "heightPx": height}

        if offset + 3 >= len(buffer):
            return None
        length = int.from_bytes(buffer[offset + 2 : offset + 4], "big")
        if length < 2:
            return None
        offset += 2 + length

    return None


def get_gif_dimensions(base64_data: str) -> dict | None:
    buffer = _decode_base64(base64_data)
    if buffer is None or len(buffer) < 10:
        return None

    if buffer[0:6] not in (b"GIF87a", b"GIF89a"):
        return None

    width = int.from_bytes(buffer[6:8], "little")
    height = int.from_bytes(buffer[8:10], "little")

    return {"widthPx": width, "heightPx": height}


def get_webp_dimensions(base64_data: str) -> dict | None:
    buffer = _decode_base64(base64_data)
    if buffer is None or len(buffer) < 30:
        return None

    if buffer[0:4] != b"RIFF" or buffer[8:12] != b"WEBP":
        return None

    chunk = buffer[12:16]
    if chunk == b"VP8 ":
        width = int.from_bytes(buffer[26:28], "little") & 0x3FFF
        height = int.from_bytes(buffer[28:30], "little") & 0x3FFF
        return {"widthPx": width, "heightPx": height}
    if chunk == b"VP8L":
        bits = int.from_bytes(buffer[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return {"widthPx": width, "heightPx": height}
    if chunk == b"VP8X":
        width = (buffer[24] | (buffer[25] << 8) | (buffer[26] << 16)) + 1
        height = (buffer[27] | (buffer[28] << 8) | (buffer[29] << 16)) + 1
        return {"widthPx": width, "heightPx": height}

    return None


def get_image_dimensions(base64_data: str, mime_type: str) -> dict | None:
    if mime_type == "image/png":
        return get_png_dimensions(base64_data)
    if mime_type == "image/jpeg":
        return get_jpeg_dimensions(base64_data)
    if mime_type == "image/gif":
        return get_gif_dimensions(base64_data)
    if mime_type == "image/webp":
        return get_webp_dimensions(base64_data)
    return None


def render_image(
    base64_data: str,
    image_dimensions: dict,
    *,
    max_width_cells: int | None = None,
    max_height_cells: int | None = None,
    preserve_aspect_ratio: bool | None = None,
    image_id: int | None = None,
    move_cursor: bool | None = None,
) -> dict | None:
    caps = get_capabilities()

    if not caps["images"]:
        return None

    max_width = max_width_cells if max_width_cells is not None else 80
    size = calculate_image_cell_size(image_dimensions, max_width, max_height_cells, get_cell_dimensions())

    if caps["images"] == "kitty":
        sequence = encode_kitty(
            base64_data,
            columns=size["columns"],
            rows=size["rows"],
            image_id=image_id,
            move_cursor=move_cursor,
        )
        return {"sequence": sequence, "rows": size["rows"], "imageId": image_id}

    if caps["images"] == "iterm2":
        sequence = encode_iterm2(
            base64_data,
            width=size["columns"],
            height="auto",
            preserve_aspect_ratio=preserve_aspect_ratio if preserve_aspect_ratio is not None else True,
        )
        return {"sequence": sequence, "rows": size["rows"], "imageId": None}

    return None


def hyperlink(text: str, url: str) -> str:
    """Wrap text in an OSC 8 hyperlink sequence.

    The text is rendered as a clickable hyperlink in terminals that support
    OSC 8 (Ghostty, Kitty, WezTerm, iTerm2, VSCode, and others). In terminals
    that do not support OSC 8, the escape sequences are ignored and only the
    plain text is displayed.
    """
    return f"\x1b]8;;{url}\x1b\\{text}\x1b]8;;\x1b\\"


def image_fallback(mime_type: str, dimensions: dict | None = None, filename: str | None = None) -> str:
    parts: list[str] = []
    if filename:
        parts.append(filename)
    parts.append(f"[{mime_type}]")
    if dimensions:
        parts.append(f"{dimensions['widthPx']}x{dimensions['heightPx']}")
    return f"[Image: {' '.join(parts)}]"
