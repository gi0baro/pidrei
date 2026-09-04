"""Box component - a container that applies padding and background to all children.

Port of pi tui ``components/box.ts``.
"""

from dataclasses import replace

from ..tui import TuiMouseDispatchResult, TuiMouseEvent, dispatch_mouse_event
from ..utils import apply_background_to_line, visible_width


__all__ = ["Box"]


class Box:
    def __init__(self, padding_x: int = 1, padding_y: int = 1, bg_fn=None) -> None:
        self.children: list = []
        self._padding_x = padding_x
        self._padding_y = padding_y
        self._bg_fn = bg_fn

        # Cache for rendered output: {"childLines", "width", "bgSample", "lines"}
        self._cache: dict | None = None
        # Geometry of the last rendered frame, for hit-testing without
        # re-rendering: {"width", "children": [(component, height), ...]}.
        self._mouse_layout: dict | None = None

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

    def _invalidate_cache(self) -> None:
        self._cache = None

    def _match_cache(self, width: int, child_lines: list[str], bg_sample: str | None) -> bool:
        cache = self._cache
        return (
            cache is not None
            and cache["width"] == width
            and cache["bgSample"] == bg_sample
            and cache["childLines"] == child_lines
        )

    def invalidate(self) -> None:
        self._invalidate_cache()
        for child in self.children:
            invalidate = getattr(child, "invalidate", None)
            if invalidate is not None:
                invalidate()

    async def handle_mouse(self, event: TuiMouseEvent) -> TuiMouseDispatchResult | None:
        content_width = max(1, event.width - self._padding_x * 2)
        content_y = event.y - self._padding_y
        content_x = event.x - self._padding_x
        if content_y < 0 or content_x < 0 or content_x >= content_width:
            return None

        layout = self._mouse_layout
        mouse_children = (
            layout["children"]
            if layout is not None and layout["width"] == content_width
            else [(component, len(component.render(content_width))) for component in self.children]
        )
        child_y = 0
        for child, child_height in mouse_children:
            if child_y <= content_y < child_y + child_height:
                return await dispatch_mouse_event(
                    child,
                    replace(event, x=content_x, y=content_y - child_y, width=content_width, height=child_height),
                )
            child_y += child_height
        return None

    def render(self, width: int) -> list[str]:
        if not self.children:
            return []

        content_width = max(1, width - self._padding_x * 2)
        left_pad = " " * self._padding_x

        # Render all children
        child_lines: list[str] = []
        mouse_children: list[tuple] = []
        for child in self.children:
            lines = child.render(content_width)
            mouse_children.append((child, len(lines)))
            for line in lines:
                child_lines.append(left_pad + line)
        self._mouse_layout = {"width": content_width, "children": mouse_children}

        if not child_lines:
            return []

        # Check if bgFn output changed by sampling
        bg_sample = self._bg_fn("test") if self._bg_fn is not None else None

        # Check cache validity
        if self._match_cache(width, child_lines, bg_sample):
            return self._cache["lines"]

        # Apply background and padding
        result: list[str] = []

        # Top padding
        for _ in range(self._padding_y):
            result.append(self._apply_bg("", width))

        # Content
        for line in child_lines:
            result.append(self._apply_bg(line, width))

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

        if self._bg_fn is not None:
            return apply_background_to_line(padded, width, self._bg_fn)
        return padded
