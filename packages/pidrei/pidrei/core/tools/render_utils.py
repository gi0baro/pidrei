"""Mirror of pi coding-agent src/core/tools/render-utils.ts."""

import json
import math
import os
from pathlib import Path

from pidrei_tui import get_capabilities, get_image_dimensions, hyperlink, image_fallback
from pidrei_utils import clock
from pidrei_utils.timers import Interval

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


def _parse_line_number(value) -> float | None:
    if isinstance(value, str) and value.strip() != "":
        try:
            value = float(value)
        except ValueError:
            return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value if isinstance(value, int) or math.isfinite(value) else None


def _js_number(value: float) -> str:
    # JS template interpolation of a number: integral floats print bare.
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def format_read_line_range(offset, limit) -> str:
    """Format finite numbers or numeric strings as a read line range, omitting malformed ranges."""
    # Strict schemas represent omitted optional arguments as null.
    if offset is None and limit is None:
        return ""
    start = 1 if offset is None else _parse_line_number(offset)
    if start is None:
        return ""
    if limit is None:
        return f":{_js_number(start)}"
    count = _parse_line_number(limit)
    if count is None:
        return ""
    end = start + count - 1
    return f":{_js_number(start)}{f'-{_js_number(end)}' if end else ''}"


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


def format_tool_duration(ms: float) -> str:
    """Duration as `4.2s` below a minute, then `1m 30s` and `2h 3m 4s`."""
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"

    total_seconds = int(seconds)
    minutes = total_seconds // 60
    remainder = total_seconds % 60
    if minutes < 60:
        return f"{minutes}m {remainder}s"

    return f"{minutes // 60}h {minutes % 60}m {remainder}s"


# Renderer state of `start_tool_timer` and `tool_timer_text`, kept in the render
# context's `state`: `startedAt`/`endedAt` (monotonic ms) and the redraw `interval`.


def start_tool_timer(context: dict) -> None:
    """Start the row's clock once execution starts. Call from `render_call`."""
    state = context["state"]
    if context["executionStarted"] and state.get("startedAt") is None:
        state["startedAt"] = clock.monotonic() * 1000
        state["endedAt"] = None


def tool_timer_text(context: dict, is_partial: bool) -> str | None:
    """The row's timing line, from `render_result`: `Elapsed 4.2s` while the
    result is partial, redrawn every second, and `Took 4.2s` once it is final.
    None when the row has no start time and no recorded duration."""
    state = context["state"]
    if state.get("startedAt") is not None and is_partial and not state.get("interval"):
        invalidate = context["invalidate"]

        def tick() -> None:
            invalidate()

        state["interval"] = Interval(1000, tick)
    if not is_partial or context["isError"]:
        if state.get("endedAt") is None:
            state["endedAt"] = clock.monotonic() * 1000
        if state.get("interval"):
            state["interval"].cancel()
            state["interval"] = None
    # A final result's recorded duration wins: it is monotonic and survives reloads. The renderer's own clock is the
    # fallback for live progress and for results stored without one.
    duration_ms = context["durationMs"]
    if not is_partial and duration_ms is not None:
        return f"Took {format_tool_duration(duration_ms)}"
    started_at = state.get("startedAt")
    if started_at is None:
        return None
    ended_at = state.get("endedAt")
    end_time = ended_at if ended_at is not None else clock.monotonic() * 1000
    return f"{'Elapsed' if is_partial else 'Took'} {format_tool_duration(end_time - started_at)}"


def invalid_arg_text(theme) -> str:
    return theme.fg("error", "[invalid arg]")


def render_tool_path(raw_path, theme, cwd: str, options: dict | None = None) -> str:
    if raw_path is None:
        return invalid_arg_text(theme)
    value = raw_path or (options or {}).get("emptyFallback")
    if not value:
        return theme.fg("toolOutput", "...")
    return link_path(theme.fg("accent", shorten_path(value)), value, cwd)
