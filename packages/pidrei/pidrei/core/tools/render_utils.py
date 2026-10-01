"""Mirror of pi coding-agent src/core/tools/render-utils.ts."""

import json
import os
from pathlib import Path

from pidrei_tui import get_capabilities, get_image_dimensions, hyperlink, image_fallback

from ...utils.ansi import strip_ansi
from ...utils.paths import resolve_path
from ...utils.shell import sanitize_binary_output


def shorten_path(path) -> str:
    if not isinstance(path, str):
        return ""
    home = os.path.expanduser("~")
    if path.startswith(home):
        return f"~{path[len(home) :]}"
    return path


def link_path(styled_text: str, raw_path: str, cwd: str) -> str:
    if not get_capabilities()["hyperlinks"]:
        return styled_text
    absolute_path = resolve_path(raw_path, cwd)
    return hyperlink(styled_text, Path(absolute_path).as_uri())


def str_or_none(value):
    """pi's ``str()`` arg guard: strings pass, nullish becomes "", else None."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return None


def replace_tabs(text: str) -> str:
    return text.replace("\t", "   ")


def normalize_display_text(text: str) -> str:
    return text.replace("\r", "")


def get_text_output(result, show_images: bool) -> str:
    """Extract text output from a tool result record for display."""
    if not result:
        return ""

    content = result["content"] if isinstance(result, dict) else result.content
    text_blocks = [c for c in content if _block_type(c) == "text"]
    image_blocks = [c for c in content if _block_type(c) == "image"]

    output = "\n".join(
        sanitize_binary_output(strip_ansi(_block_get(c, "text") or "")).replace("\r", "") for c in text_blocks
    )

    caps = get_capabilities()
    if image_blocks and (not caps["images"] or not show_images):
        indicators = []
        for img in image_blocks:
            mime_type = _block_get(img, "mimeType") or "image/unknown"
            data = _block_get(img, "data")
            raw_mime = _block_get(img, "mimeType")
            dims = get_image_dimensions(data, raw_mime) if data and raw_mime else None
            indicators.append(image_fallback(mime_type, dims))
        image_indicators = "\n".join(indicators)
        output = f"{output}\n{image_indicators}" if output else image_indicators

    return output


def _block_type(block) -> str | None:
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def _block_get(block, key: str):
    if isinstance(block, dict):
        return block.get(key)
    # dataclass content blocks use snake_case field names
    snake = {"mimeType": "mime_type"}.get(key, key)
    return getattr(block, snake, None)


COLLAPSED_ARGS_CHARS = 100


def _json_stringify(value, indent: int | None = None) -> str:
    """pi's `JSON.stringify(value) ?? String(value)` for a display string."""
    try:
        if indent is None:
            return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        return json.dumps(value, indent=indent, ensure_ascii=False)
    except TypeError, ValueError:
        return str(value)


def format_tool_call_with_args(title: str, args, theme, expanded: bool) -> str:
    """Generic tool call header: the title followed by the arguments. Collapsed, they are
    `key=value` pairs on the title line, cut to `COLLAPSED_ARGS_CHARS`. Expanded, each is a
    `key: value` line below the title, with strings shown raw and continuation lines indented."""
    header = theme.fg("toolTitle", theme.bold(title))
    if args is None:
        return header
    entries = list(args.items()) if isinstance(args, dict) else [("args", args)]
    if not entries:
        return header
    if expanded:
        lines = []
        for key, value in entries:
            text = value if isinstance(value, str) else _json_stringify(value, 2)
            lines.append(f"  {key}: " + "\n    ".join(replace_tabs(text).replace("\r", "").split("\n")))
        return f"{header}\n{theme.fg('muted', chr(10).join(lines))}"
    pairs = " ".join(f"{key}={_json_stringify(value)}" for key, value in entries)
    preview = f"{pairs[: COLLAPSED_ARGS_CHARS - 3]}..." if len(pairs) > COLLAPSED_ARGS_CHARS else pairs
    return f"{header} {theme.fg('muted', preview)}"


def invalid_arg_text(theme) -> str:
    return theme.fg("error", "[invalid arg]")


def render_tool_path(raw_path, theme, cwd: str, options: dict | None = None) -> str:
    if raw_path is None:
        return invalid_arg_text(theme)
    value = raw_path or (options or {}).get("emptyFallback")
    if not value:
        return theme.fg("toolOutput", "...")
    return link_path(theme.fg("accent", shorten_path(value)), value, cwd)
