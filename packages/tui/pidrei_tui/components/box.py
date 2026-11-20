"""Box component - a container that applies padding and background to all children.

Port of pi tui ``components/box.ts``.
"""

from collections.abc import Sequence
from dataclasses import replace

from ..tui import (
    ChildLines,
    TuiMouseDispatchResult,
    TuiMouseEvent,
    dispatch_mouse_to_child_lines,
    render_child_lines,
)
from ..utils import visible_width


__all__ = ["Box"]


class Box:
    def __init__(self, padding_x: int = 1, padding_y: int = 1, bg_fn=None) -> None:
        self.children: list = []
        self._padding_x = padding_x
        self._padding_y = padding_y
        self._bg_fn = bg_fn

        # Cache for rendered output: {"childLines", "width", "bgSample", "lines"}
        self._cache: dict | None = None
        self._child_lines: ChildLines | None = None

    def add_child(self, component) -> None:
        self.children.append(component)
        self._invalidate_cache()

    def remove_child(self, component) -> None:
        try:
            self.children.remove(component)
        except ValueError:
            return
        self._invalidate_cache()

    def clear(self) -> None:
        self.children = []
        self._invalidate_cache()

    def set_children(self, children: list) -> None:
        """Replace the children in one step (see `Container.set_children`)."""
        self.children = list(children)
        self._invalidate_cache()

    def set_bg_fn(self, bg_fn=None) -> None:
        self._bg_fn = bg_fn
        # Don't invalidate here - we'll detect bgFn changes by sampling output

    def set_padding_x(self, padding_x: int) -> None:
        self._padding_x = padding_x
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        self._cache = None
        self._child_lines = None

    def invalidate(self) -> None:
        self._invalidate_cache()
        for child in self.children:
            invalidate = getattr(child, "invalidate", None)
            if invalidate is not None:
                invalidate()

    def handle_mouse(self, event: TuiMouseEvent) -> TuiMouseDispatchResult | None:
        content_width = max(1, event.width - self._padding_x * 2)
        content_y = event.y - self._padding_y
        content_x = event.x - self._padding_x
        if content_y < 0 or content_x < 0 or content_x >= content_width:
            return None

        child_lines = self._child_lines
        layout = (
            child_lines
            if child_lines is not None and child_lines.width == content_width
            else render_child_lines(self.children, content_width, None)
        )
        return dispatch_mouse_to_child_lines(layout, replace(event, x=content_x, y=content_y, width=content_width))

    def render(self, width: int) -> Sequence[str]:
        if not self.children:
            return []

        content_width = max(1, width - self._padding_x * 2)
        left_pad = " " * self._padding_x

        # Keep the child lines unpadded: unchanged children keep their line list, so the cache check
        # below is one identity comparison. Padding here would create new strings every frame.
        self._child_lines = render_child_lines(self.children, content_width, self._child_lines)
        child_lines = self._child_lines.lines

        if not child_lines:
            return []

        # Check if bgFn output changed by sampling
        bg_sample = self._bg_fn("test") if self._bg_fn is not None else None

        # Check cache validity
        cache = self._cache
        if (
            cache is not None
            and cache["childLines"] is child_lines
            and cache["width"] == width
            and cache["bgSample"] == bg_sample
        ):
            return cache["lines"]

        # Apply background and padding
        result: list[str] = []

        # Top padding
        for _ in range(self._padding_y):
            result.append(self._apply_bg("", width))

        # Content
        for line in child_lines:
            result.append(self._apply_bg(left_pad + line, width))

        # Bottom padding
        for _ in range(self._padding_y):
            result.append(self._apply_bg("", width))

        # Update cache
        self._cache = {"childLines": child_lines, "width": width, "bgSample": bg_sample, "lines": result}

        return result

    def _apply_bg(self, line: str, width: int) -> str:
        vis_len = visible_width(line)
        pad_needed = max(0, width - vis_len)
        padded = line + " " * pad_needed
        # Already padded to width, so apply the background directly instead of measuring the line again.
        return self._bg_fn(padded) if self._bg_fn is not None else padded
