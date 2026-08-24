"""Text component - displays multi-line text with word wrapping.

Port of pi tui ``components/text.ts``.
"""

from ..utils import apply_background_to_line, visible_width, wrap_text_with_ansi


__all__ = ["Text"]


class Text:
    def __init__(self, text: str = "", padding_x: int = 1, padding_y: int = 1, custom_bg_fn=None) -> None:
        self._text = text
        self._padding_x = padding_x  # Left/right padding
        self._padding_y = padding_y  # Top/bottom padding
        self._custom_bg_fn = custom_bg_fn

        # Cache for rendered output: one immutable (text, width, lines) tuple,
        # read once per render. `invalidate()` runs from other threads (Loader
        # ticks, the agent, theme reloads); three separate fields could be
        # observed half-cleared and hand `None` to the render loop.
        self._cache: tuple[str, int, list[str]] | None = None

    def set_text(self, text: str) -> None:
        self._text = text
        self.invalidate()

    def set_custom_bg_fn(self, custom_bg_fn=None) -> None:
        self._custom_bg_fn = custom_bg_fn
        self.invalidate()

    def invalidate(self) -> None:
        self._cache = None

    def render(self, width: int) -> list[str]:
        text = self._text
        # Check cache
        cache = self._cache
        if cache is not None and cache[0] == text and cache[1] == width:
            return cache[2]

        # Don't render anything if there's no actual text
        if not text or text.strip() == "":
            result: list[str] = []
            self._cache = (text, width, result)
            return result

        # Replace tabs with 3 spaces
        normalized_text = text.replace("\t", "   ")

        # Reduce margins when necessary so content and padding fit within the available width.
        padding_x = min(self._padding_x, max(0, (width - 1) // 2))
        content_width = max(1, width - padding_x * 2)

        # Wrap text (this preserves ANSI codes but does NOT pad)
        wrapped_lines = wrap_text_with_ansi(normalized_text, content_width)

        # Add margins and background to each line
        left_margin = " " * padding_x
        right_margin = " " * padding_x
        content_lines: list[str] = []

        for line in wrapped_lines:
            line_with_margins = left_margin + line + right_margin

            # Apply background if specified (this also pads to full width)
            if self._custom_bg_fn is not None:
                content_lines.append(apply_background_to_line(line_with_margins, width, self._custom_bg_fn))
            else:
                # No background - just pad to width with spaces
                visible_len = visible_width(line_with_margins)
                padding_needed = max(0, width - visible_len)
                content_lines.append(line_with_margins + " " * padding_needed)

        # Add top/bottom padding (empty lines)
        empty_line = " " * width
        empty_lines: list[str] = []
        for _ in range(self._padding_y):
            line = (
                apply_background_to_line(empty_line, width, self._custom_bg_fn)
                if self._custom_bg_fn is not None
                else empty_line
            )
            empty_lines.append(line)

        result = [*empty_lines, *content_lines, *empty_lines]

        # Update cache
        self._cache = (text, width, result)

        return result if result else [""]
