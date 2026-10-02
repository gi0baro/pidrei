"""Mirror of pi coding-agent src/modes/interactive/components/visual-truncate.ts.

Shared utility for truncating text to visual lines (accounting for line
wrapping). Used by tool renderers and bash_execution for consistent behavior.
"""

from collections.abc import Callable
from typing import Literal

from pidrei_tui import Text, truncate_to_width


def truncate_to_visual_lines(
    text: str, max_visual_lines: int, width: int, padding_x: int = 0, keep: Literal["start", "end"] = "end"
) -> dict:
    """Truncate text to a maximum number of visual lines.

    ``padding_x`` is 0 when the result goes into a Box (which pads itself)
    and 1 when placed in a plain Container. ``keep`` selects which visual
    lines to keep: the last ones (default) or the first ones. Returns
    ``{"visualLines", "skippedCount"}``.
    """
    if not text:
        return {"visualLines": [], "skippedCount": 0}

    # Create a temporary Text component to render and get visual lines
    temp_text = Text(text, padding_x, 0)
    all_visual_lines = temp_text.render(width)

    if len(all_visual_lines) <= max_visual_lines:
        return {"visualLines": all_visual_lines, "skippedCount": 0}

    truncated_lines = all_visual_lines[:max_visual_lines] if keep == "start" else all_visual_lines[-max_visual_lines:]
    skipped_count = len(all_visual_lines) - max_visual_lines

    return {"visualLines": truncated_lines, "skippedCount": skipped_count}


class VisualLinePreview:
    """Collapsed tool output limited to a number of visual lines, like bash
    output. Limiting logical lines instead lets a single long line (such as
    minified JSON) wrap across the whole screen. Caches its lines per width,
    since it renders on every frame for every result in the transcript.

    ``keep`` selects which visual lines to keep; the hint goes before kept end
    lines and after kept start lines. ``format_hint`` builds the styled hint
    line for the number of hidden visual lines.
    """

    def __init__(
        self,
        *,
        text: str,
        max_visual_lines: int,
        keep: Literal["start", "end"],
        format_hint: Callable[[int], str],
    ) -> None:
        self._text = text
        self._max_visual_lines = max_visual_lines
        self._keep = keep
        self._format_hint = format_hint
        # One (width, lines) value, read once per render.
        self._cache: tuple[int, list[str]] | None = None

    def render(self, width: int) -> list[str]:
        cache = self._cache
        if cache is not None and cache[0] == width:
            return cache[1]
        preview = truncate_to_visual_lines(self._text, self._max_visual_lines, width, 0, self._keep)
        lines = preview["visualLines"]
        if preview["skippedCount"] > 0:
            hint = truncate_to_width(self._format_hint(preview["skippedCount"]), width, "...")
            lines = [*lines, hint] if self._keep == "start" else [hint, *lines]
        self._cache = (width, lines)
        return lines

    def invalidate(self) -> None:
        self._cache = None
