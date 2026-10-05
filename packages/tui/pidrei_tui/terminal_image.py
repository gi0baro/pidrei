"""Mirror of pi tui src/terminal-image.ts.

Records (camelCase like pi): TerminalCapabilities = {"images": "kitty" |
"iterm2" | None, "trueColor": bool, "hyperlinks": bool}; CellDimensions /
ImageDimensions = {"widthPx": int, "heightPx": int}; image cell size =
{"columns": int, "rows": int}; render_image result = {"sequence": str,
"columns": int, "rows": int, "imageId": int | None}; KittyImageMetadata =
{"imageId": int, "columns": int, "rows": int, "widthPx": int,
"heightPx": int}.

Port notes: JS ``Buffer.from(base64)`` never raises — the dimension sniffers
wrap ``base64.b64decode`` in try/except instead; ``Math.random``-based image
ids use ``random`` (collision avoidance, not security).
"""

import base64
import binascii
import math
import os
import random
import re
import threading
from dataclasses import dataclass
from pathlib import Path


# Default cell dimensions - updated by TUI when terminal responds to query.
# Written by the input consumer (the cell-size reply) and read by image
# rendering, both under the UI state lock; the dict is replaced whole.
_cell_dimensions = {"widthPx": 9, "heightPx": 18}


@dataclass(frozen=True, slots=True)
class _CapabilityState:
    """The overrides and the capabilities computed from them, replaced as one
    (spec/ui-island.md, "Terminal-level state"): a cache computed from old overrides can never be
    stored next to new ones."""

    overrides: dict
    cached: dict | None


_capability_state = _CapabilityState({}, None)
# Serializes replacing `_capability_state`; never held while detecting.
_capability_lock = threading.Lock()

# Sentinel for "no override" (pi uses `undefined`, distinct from `images: null`).
_UNSET = object()


def get_cell_dimensions() -> dict:
    return _cell_dimensions


def set_cell_dimensions(dims: dict) -> None:
    global _cell_dimensions
    _cell_dimensions = dims


# The tmux probe's answer, once per process. pi runs the probe (`tmux
# display-message`) with `execSync` inside detection; here detection never
# runs a subprocess: `prime_capabilities` awaits the caller's probe before any
# render asks. The answer does not depend on the overrides, so changing them
# (which drops the cached capabilities) never needs it again. Unprimed,
# detection takes the probe's failure answer.
_tmux_hyperlinks: bool | None = None


def _tmux_forwards_hyperlinks() -> bool:
    return _tmux_hyperlinks is True


def _in_tmux() -> bool:
    return bool(os.environ.get("TMUX")) or (os.environ.get("TERM") or "").lower().startswith("tmux")


async def prime_capabilities(probe_tmux_hyperlinks) -> None:
    """Settle what detection cannot read from the environment, before a render
    asks: under tmux, `await probe_tmux_hyperlinks()` answers whether the
    attached client forwards OSC 8 hyperlinks (False on any failure). Run once
    per process, whatever the overrides say."""
    global _tmux_hyperlinks, _capability_state
    if _tmux_hyperlinks is not None or not _in_tmux():
        return
    _tmux_hyperlinks = await probe_tmux_hyperlinks()
    with _capability_lock:
        # Anything detected before the answer took the failure answer.
        _capability_state = _CapabilityState(_capability_state.overrides, None)


def _detect_capabilities_from_environment(tmux_forwards_hyperlink) -> dict:
    term_program = (os.environ.get("TERM_PROGRAM") or "").lower()
    terminal_emulator = (os.environ.get("TERMINAL_EMULATOR") or "").lower()
    term = (os.environ.get("TERM") or "").lower()
    color_term = (os.environ.get("COLORTERM") or "").lower()
    has_true_color_hint = color_term in ("truecolor", "24bit") or term.endswith("-direct")

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

    if term_program in ("alacritty", "vscode", "zed"):
        return {"images": None, "trueColor": True, "hyperlinks": True}

    if terminal_emulator == "jetbrains-jediterm":
        return {"images": None, "trueColor": True, "hyperlinks": False}

    # Unknown terminal: be conservative. OSC 8 is rendered invisibly as "just
    # text" on terminals that swallow it, which means the URL disappears from
    # the rendered output. Default to the legacy `text (url)` behavior unless we
    # have positively identified a hyperlink-capable terminal above.
    return {"images": None, "trueColor": has_true_color_hint, "hyperlinks": False}


def _parse_boolean_capability_override(value: str | None) -> bool | None:
    return True if value == "1" else False if value == "0" else None


def detect_capabilities(tmux_forwards_hyperlink=_tmux_forwards_hyperlinks) -> dict:
    hyperlinks = _parse_boolean_capability_override(os.environ.get("PIDREI_HYPERLINKS"))
    detected = _detect_capabilities_from_environment(
        tmux_forwards_hyperlink if hyperlinks is None else (lambda: hyperlinks)
    )
    image_protocol = (os.environ.get("PIDREI_IMAGE_PROTOCOL") or "").lower() or None
    if image_protocol in ("kitty", "iterm2"):
        images = image_protocol
    elif image_protocol in ("none", "0"):
        images = None
    else:
        images = _UNSET
    true_color = _parse_boolean_capability_override(os.environ.get("PIDREI_TRUE_COLOR"))
    return {
        **detected,
        **({"images": images} if images is not _UNSET else {}),
        **({"trueColor": true_color} if true_color is not None else {}),
        **({"hyperlinks": hyperlinks} if hyperlinks is not None else {}),
    }


def get_capabilities() -> dict:
    global _capability_state
    state = _capability_state
    if state.cached is not None:
        return state.cached
    hyperlinks = state.overrides.get("hyperlinks")
    capabilities = {
        **(detect_capabilities() if hyperlinks is None else detect_capabilities(lambda: hyperlinks)),
        **state.overrides,
    }
    with _capability_lock:
        # Stored only if the overrides did not change meanwhile; either way
        # this caller gets what it computed.
        if _capability_state is state:
            _capability_state = _CapabilityState(state.overrides, capabilities)
    return capabilities


def get_terminal_color_mode(capabilities: dict | None = None) -> str:
    """The ``TerminalColorMode`` ("256color" | "truecolor") for ``capabilities``
    (the detected ones by default)."""
    if capabilities is None:
        capabilities = get_capabilities()
    return "truecolor" if capabilities["trueColor"] else "256color"


def reset_capabilities_cache() -> None:
    """Forget everything detected, the tmux probe's answer included."""
    global _capability_state, _tmux_hyperlinks
    with _capability_lock:
        _capability_state = _CapabilityState(_capability_state.overrides, None)
        _tmux_hyperlinks = None


def set_capability_overrides(overrides: dict) -> None:
    """Override selected auto-detected capabilities."""
    global _capability_state
    with _capability_lock:
        current = _capability_state.overrides
        if (
            current.get("images", _UNSET) == overrides.get("images", _UNSET)
            and current.get("trueColor", _UNSET) == overrides.get("trueColor", _UNSET)
            and current.get("hyperlinks", _UNSET) == overrides.get("hyperlinks", _UNSET)
        ):
            return
        _capability_state = _CapabilityState({**overrides}, None)


def get_capability_overrides() -> dict:
    return _capability_state.overrides


def set_capabilities(caps: dict) -> None:
    """Override the cached capabilities. Useful in tests to exercise both code paths."""
    global _capability_state
    with _capability_lock:
        _capability_state = _CapabilityState(_capability_state.overrides, caps)


def replace_capabilities(expected: dict, caps: dict) -> bool:
    """Swap the cached capabilities for `caps` only while they are still
    `expected` (pidrei-only: the alternate screen's iTerm2 swap and restore;
    a change of overrides meanwhile wins)."""
    global _capability_state
    with _capability_lock:
        if _capability_state.cached is not expected:
            return False
        _capability_state = _CapabilityState(_capability_state.overrides, caps)
        return True


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


def delete_all_kitty_placements() -> str:
    """Delete all visible Kitty placements while retaining their uploaded image data."""
    return "\x1b_Ga=d,d=a,q=2\x1b\\"


_BASE64_NON_ALPHABET_RE = re.compile(r"[^A-Za-z0-9+/]")


def _base64_decoded_size(base64_data: str) -> int:
    """``Buffer.byteLength(data, "base64")``: never raises on malformed input."""
    return len(_BASE64_NON_ALPHABET_RE.sub("", base64_data.split("=", 1)[0])) * 3 // 4


def encode_iterm2(
    base64_data: str,
    *,
    width: int | str | None = None,
    height: int | str | None = None,
    name: str | None = None,
    preserve_aspect_ratio: bool | None = None,
    inline: bool | None = None,
) -> str:
    params = [
        f"inline={1 if inline is not False else 0}",
        f"size={_base64_decoded_size(base64_data)}",
    ]

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


# Kitty placement metadata, keyed by image id: {"imageId", "columns", "rows",
# "widthPx", "heightPx"}. The alternate-screen renderer needs the pixel size to
# crop a placement that is partially scrolled off the top of the viewport.
_kitty_image_metadata: dict[int, dict] = {}
_kitty_image_metadata_lock = threading.Lock()
_kitty_transmission_generation = 0

_KITTY_CONTROLS_RE = re.compile(r"\x1b_G([^;]*);")
_KITTY_IMAGE_ID_RE = re.compile(r"(?:^|,)i=(\d+)(?:,|$)")
_KITTY_ROWS_RE = re.compile(r"(?:^|,)r=(\d+)(?:,|$)")
_KITTY_CROP_CONTROL_RE = re.compile(r"^[yhr]=")


def register_kitty_image_metadata(metadata: dict) -> None:
    global _kitty_transmission_generation
    with _kitty_image_metadata_lock:
        _kitty_transmission_generation += 1
        _kitty_image_metadata.pop(metadata["imageId"], None)
        _kitty_image_metadata[metadata["imageId"]] = {
            **metadata,
            "transmissionGeneration": _kitty_transmission_generation,
        }
        if len(_kitty_image_metadata) > 1000:
            oldest_image_id = next(iter(_kitty_image_metadata), None)
            if oldest_image_id is not None:
                del _kitty_image_metadata[oldest_image_id]


def _get_registered_kitty_image_metadata_from_controls(controls: str) -> dict | None:
    image_id = _KITTY_IMAGE_ID_RE.search(controls)
    if image_id is None:
        return None
    with _kitty_image_metadata_lock:
        return _kitty_image_metadata.get(int(image_id.group(1)))


def _get_registered_kitty_image_metadata(line: str) -> dict | None:
    controls = _KITTY_CONTROLS_RE.search(line)
    return None if controls is None else _get_registered_kitty_image_metadata_from_controls(controls.group(1))


def _get_explicit_kitty_image_rows(controls: str) -> int | None:
    value = _KITTY_ROWS_RE.search(controls)
    if value is None:
        return None
    rows = int(value.group(1))
    return rows if rows > 0 else None


def _get_kitty_image_rows_from_controls(controls: str, fallback_rows: int) -> int:
    rows = _get_explicit_kitty_image_rows(controls)
    return rows if rows is not None else fallback_rows


def get_kitty_image_metadata(line: str) -> dict | None:
    metadata = _get_registered_kitty_image_metadata(line)
    if not metadata:
        return None
    return {
        "imageId": metadata["imageId"],
        "columns": metadata["columns"],
        "rows": metadata["rows"],
        "widthPx": metadata["widthPx"],
        "heightPx": metadata["heightPx"],
    }


# Controls that belong to a placement command rather than a transmission.
_KITTY_PLACEMENT_CONTROL_KEYS = frozenset(
    {"i", "p", "x", "y", "w", "h", "X", "Y", "c", "r", "C", "U", "z", "P", "Q", "H", "V"}
)

_KITTY_CHUNK_CONTINUES_RE = re.compile(r"(?:^|,)m=1(?:,|$)")


def get_kitty_image_placement_rows(line: str) -> int | None:
    """Read the number of rows covered by an image placement without scanning its payload."""
    controls = _KITTY_CONTROLS_RE.search(line)
    if controls is None:
        return None
    explicit_rows = _get_explicit_kitty_image_rows(controls.group(1))
    if explicit_rows is not None:
        return explicit_rows
    metadata = _get_registered_kitty_image_metadata_from_controls(controls.group(1))
    return metadata["rows"] if metadata is not None else None


def get_kitty_image_placement(line: str) -> dict | None:
    """Placement-only command for a `render_image` line, or None.

    Returns {"imageId", "transmissionGeneration", "transmissionBytes",
    "estimatedDecodedBytes", "rows", "sequence", "replacementLine"};
    `replacementLine` re-places an already-uploaded image without resending its
    payload.
    """
    match = _KITTY_CONTROLS_RE.search(line)
    if not match:
        return None
    metadata = _get_registered_kitty_image_metadata_from_controls(match.group(1))
    if not metadata:
        return None

    command_start = match.start()
    command_controls = match.group(1)
    while True:
        terminator = line.find("\x1b\\", command_start + len(KITTY_PREFIX))
        if terminator == -1:
            return None
        transmission_end = terminator + 2
        if not _KITTY_CHUNK_CONTINUES_RE.search(command_controls):
            break
        command_start = transmission_end
        if not line.startswith(KITTY_PREFIX, command_start):
            return None
        controls_end = line.find(";", command_start + len(KITTY_PREFIX))
        if controls_end == -1:
            return None
        command_controls = line[command_start + len(KITTY_PREFIX) : controls_end]

    controls = [
        control for control in match.group(1).split(",") if control.split("=", 1)[0] in _KITTY_PLACEMENT_CONTROL_KEYS
    ]
    sequence = f"\x1b_Ga=p,q=2,{','.join(controls)}\x1b\\"
    return {
        "imageId": metadata["imageId"],
        "transmissionGeneration": metadata["transmissionGeneration"],
        "transmissionBytes": transmission_end - match.start(),
        "estimatedDecodedBytes": metadata["widthPx"] * metadata["heightPx"] * 4,
        "rows": _get_kitty_image_rows_from_controls(match.group(1), metadata["rows"]),
        "sequence": sequence,
        "replacementLine": f"{line[: match.start()]}{sequence}{line[transmission_end:]}",
    }


def crop_kitty_image_line(line: str, hidden_rows: int, visible_rows: int) -> str:
    """Re-issue a Kitty placement cropped to its still-visible bottom rows."""
    metadata = get_kitty_image_metadata(line)
    match = _KITTY_CONTROLS_RE.search(line)
    if not metadata or not match or hidden_rows < 0 or hidden_rows >= metadata["rows"] or visible_rows <= 0:
        return line
    cropped_rows = min(visible_rows, metadata["rows"] - hidden_rows)
    if hidden_rows == 0 and cropped_rows == metadata["rows"]:
        return line
    source_y = math.floor(metadata["heightPx"] * hidden_rows / metadata["rows"])
    source_end = math.ceil(metadata["heightPx"] * (hidden_rows + cropped_rows) / metadata["rows"])
    source_height = max(1, min(metadata["heightPx"], source_end) - source_y)
    controls = [control for control in match.group(1).split(",") if not _KITTY_CROP_CONTROL_RE.match(control)]
    controls.extend([f"y={source_y}", f"h={source_height}", f"r={cropped_rows}"])
    return f"{line[: match.start()]}\x1b_G{','.join(controls)};{line[match.end() :]}"


def _choose_less_distorted_cell_count(upper_count: int, ideal_count: float) -> int:
    if upper_count <= 1:
        return upper_count

    lower_count = upper_count - 1
    upper_distortion = max(upper_count / ideal_count, ideal_count / upper_count)
    lower_distortion = max(lower_count / ideal_count, ideal_count / lower_count)
    return lower_count if lower_distortion < upper_distortion else upper_count


def calculate_image_cell_size(
    image_dimensions: dict,
    max_width_cells: int,
    max_height_cells: int | None = None,
    cell_dimensions: dict | None = None,
    optimize_aspect_ratio: bool = False,
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
    columns = max(1, min(max_width, math.ceil(scaled_width_px / cell_dimensions["widthPx"])))
    height_rows = scaled_height_px / cell_dimensions["heightPx"]
    rows = max(1, math.ceil(height_rows))
    if max_height is not None:
        rows = min(max_height, rows)

    if not optimize_aspect_ratio:
        return {"columns": columns, "rows": rows}

    if width_scale <= height_scale:
        ideal_rows = (columns * cell_dimensions["widthPx"] * image_height) / (image_width * cell_dimensions["heightPx"])
        rows = _choose_less_distorted_cell_count(rows, ideal_rows)
    else:
        ideal_columns = (rows * cell_dimensions["heightPx"] * image_width) / (image_height * cell_dimensions["widthPx"])
        columns = _choose_less_distorted_cell_count(columns, ideal_columns)

    return {"columns": columns, "rows": rows}


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
    # Reduce Kitty's cell-aligned distortion without shrinking iTerm2 reservations.
    size = calculate_image_cell_size(
        image_dimensions, max_width, max_height_cells, get_cell_dimensions(), caps["images"] == "kitty"
    )

    if caps["images"] == "kitty":
        if image_id is not None:
            register_kitty_image_metadata(
                {
                    "imageId": image_id,
                    "columns": size["columns"],
                    "rows": size["rows"],
                    "widthPx": image_dimensions["widthPx"],
                    "heightPx": image_dimensions["heightPx"],
                }
            )
        sequence = encode_kitty(
            base64_data,
            columns=size["columns"],
            rows=size["rows"],
            image_id=image_id,
            move_cursor=move_cursor,
        )
        return {"sequence": sequence, "columns": size["columns"], "rows": size["rows"], "imageId": image_id}

    if caps["images"] == "iterm2":
        sequence = encode_iterm2(
            base64_data,
            width=size["columns"],
            height="auto",
            preserve_aspect_ratio=preserve_aspect_ratio if preserve_aspect_ratio is not None else True,
        )
        return {"sequence": sequence, "columns": size["columns"], "rows": size["rows"], "imageId": None}

    return None


def hyperlink(text: str, url: str) -> str:
    """Wrap text in an OSC 8 hyperlink sequence.

    The text is rendered as a clickable hyperlink in terminals that support
    OSC 8 (Ghostty, Kitty, WezTerm, iTerm2, VSCode, and others). In terminals
    that do not support OSC 8, the escape sequences are ignored and only the
    plain text is displayed.
    """
    return f"\x1b]8;;{url}\x1b\\{text}\x1b]8;;\x1b\\"


def _shorten_image_path(filename: str) -> str:
    """Shorten home-prefixed absolute paths to ~/... for compact display."""
    home = os.path.expanduser("~")
    if home and home != "~" and (filename == home or filename.startswith((f"{home}/", f"{home}\\"))):
        return f"~{filename[len(home) :]}"
    return filename


def image_fallback(mime_type: str, dimensions: dict | None = None, filename: str | None = None) -> str:
    """Text fallback when the terminal cannot render inline images.

    Absolute paths are shown shortened (~/...) and, when OSC 8 hyperlinks are
    available, linked to file:// so the full path remains openable.
    """
    parts: list[str] = []
    if filename:
        display = _shorten_image_path(filename)
        if get_capabilities().get("hyperlinks") and os.path.isabs(filename):
            parts.append(hyperlink(display, Path(filename).as_uri()))
        else:
            parts.append(display)
    parts.append(f"[{mime_type}]")
    if dimensions:
        parts.append(f"{dimensions['widthPx']}x{dimensions['heightPx']}")
    return f"[Image: {' '.join(parts)}]"
