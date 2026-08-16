"""Mirror of pi tui src/components/scroll-view.ts.

A single-child viewport the layout engine sizes and the alternate-screen
renderer scrolls. The scroll state lives here, not in the renderer, so nested
regions can each own their position.

Options (camelCase like pi's ``ScrollViewOptions``): ``axis`` ("vertical"),
``follow`` ("none" | "end"), ``primary``, ``overscroll`` ("chain" |
"contain"), ``scrollbar`` ("hidden" | "auto" | "always"), ``scrollbarStyle``
(a ``str -> str`` styler) and ``scrollbarHideDelayMs``.
"""

import math

from .._timers import Timeout
from ..layout_node import LAYOUT_NODE
from ..tui import Container


class ScrollView(Container):
    """Scrollable viewport around exactly one child component."""

    def __init__(self, component, options: dict | None = None) -> None:
        super().__init__()
        options = options or {}
        axis = options.get("axis")
        if axis is not None and axis != "vertical":
            raise Exception(f"Unsupported ScrollView axis: {axis}")
        self._child = component
        self.children.append(component)
        self._follow_end = (options.get("follow") or "none") == "end"
        self._following_end = self._follow_end
        self._follow_suppressed_at_end = False
        self.primary = options.get("primary") or False
        self.overscroll = options.get("overscroll") or "chain"
        self._current_scrollbar = options.get("scrollbar") or "hidden"
        self.scrollbar_style = options.get("scrollbarStyle") or (lambda text: f"\x1b[100m{text}\x1b[49m")
        self._scrollbar_hide_delay_ms = max(
            0, math.floor(options["scrollbarHideDelayMs"] if options.get("scrollbarHideDelayMs") is not None else 1000)
        )
        self._current_scroll_top = 0
        self._content_height = 0
        self._current_viewport_height = 0
        self._request_render_callback = None
        self._transient_scrollbar_visible = False
        self._scrollbar_active = False
        self._scrollbar_hide_timer: Timeout | None = None

    @property
    def scroll_top(self) -> int:
        return self._current_scroll_top

    @property
    def is_following_end(self) -> bool:
        return self._following_end

    @property
    def viewport_height(self) -> int:
        return self._current_viewport_height

    @property
    def scrollbar(self) -> str:
        return self._current_scrollbar

    @property
    def is_scrollbar_visible(self) -> bool:
        if self.scrollbar == "always":
            return self._current_viewport_height > 0
        return (
            self.scrollbar == "auto"
            and self._content_height > self._current_viewport_height
            and self._transient_scrollbar_visible
        )

    def set_scrollbar(self, scrollbar: str) -> None:
        if scrollbar == self._current_scrollbar:
            return
        self._current_scrollbar = scrollbar
        if scrollbar != "auto":
            self._hide_transient_scrollbar()
        elif self._scrollbar_active:
            self._mark_scrollbar_activity()
        if self._request_render_callback is not None:
            self._request_render_callback()

    def get_content_width(self, width: int) -> int:
        return width - 1 if self.scrollbar == "always" and width > 1 else width

    def _mark_scrollbar_activity(self) -> None:
        if self.scrollbar != "auto" or self._content_height <= self._current_viewport_height:
            return
        self._transient_scrollbar_visible = True
        if self._scrollbar_hide_timer is not None:
            self._scrollbar_hide_timer.cancel()
            self._scrollbar_hide_timer = None
        if self._scrollbar_active:
            return

        async def hide() -> None:
            self._scrollbar_hide_timer = None
            self._transient_scrollbar_visible = False
            if self._request_render_callback is not None:
                self._request_render_callback()

        self._scrollbar_hide_timer = Timeout(self._scrollbar_hide_delay_ms, hide)

    def _hide_transient_scrollbar(self) -> None:
        self._transient_scrollbar_visible = False
        if self._scrollbar_hide_timer is None:
            return
        self._scrollbar_hide_timer.cancel()
        self._scrollbar_hide_timer = None

    def set_scrollbar_active(self, active: bool) -> None:
        if active == self._scrollbar_active:
            return
        self._scrollbar_active = active
        self._mark_scrollbar_activity()

    def scroll_to(self, scroll_top: int, options: dict | None = None) -> None:
        """``options`` mirrors pi's ``ScrollViewScrollToOptions`` (``{"disableFollow"?}``).

        ``disableFollow`` keeps follow-end disabled even when the target is the
        current content end.
        """
        options = options or {}
        requested = math.trunc(scroll_top) if math.isfinite(scroll_top) else self._current_scroll_top
        max_scroll_top = max(0, self._content_height - self._current_viewport_height)
        nxt = max(0, min(max_scroll_top, requested))
        next_follow_suppressed_at_end = options.get("disableFollow") is True and nxt == max_scroll_top
        next_following_end = not next_follow_suppressed_at_end and self._follow_end and nxt == max_scroll_top
        if (
            nxt == self._current_scroll_top
            and next_following_end == self._following_end
            and next_follow_suppressed_at_end == self._follow_suppressed_at_end
        ):
            return
        moved = nxt != self._current_scroll_top
        self._current_scroll_top = nxt
        self._following_end = next_following_end
        self._follow_suppressed_at_end = next_follow_suppressed_at_end
        if moved:
            self._mark_scrollbar_activity()
        if self._request_render_callback is not None:
            self._request_render_callback()

    def scroll_by(self, lines: int) -> int:
        """Scroll by `lines` and return the leftover the caller may chain on."""
        requested = math.trunc(lines) if math.isfinite(lines) else 0
        if requested == 0:
            return 0
        max_scroll_top = max(0, self._content_height - self._current_viewport_height)
        start = max_scroll_top if self._following_end else self._current_scroll_top
        nxt = max(0, min(max_scroll_top, start + requested))
        moved = nxt - start
        was_following_end = self._following_end
        self._current_scroll_top = nxt
        self._following_end = self._follow_end and nxt == max_scroll_top
        self._follow_suppressed_at_end = False
        if moved != 0:
            self._mark_scrollbar_activity()
        if (moved != 0 or self._following_end != was_following_end) and self._request_render_callback is not None:
            self._request_render_callback()
        return requested - moved

    def scroll_to_start(self) -> None:
        changed = self._current_scroll_top != 0 or self._following_end != (
            self._follow_end and self._content_height <= self._current_viewport_height
        )
        self._current_scroll_top = 0
        self._following_end = self._follow_end and self._content_height <= self._current_viewport_height
        self._follow_suppressed_at_end = False
        if changed:
            self._mark_scrollbar_activity()
            if self._request_render_callback is not None:
                self._request_render_callback()

    def scroll_to_end(self) -> None:
        nxt = max(0, self._content_height - self._current_viewport_height)
        changed = self._current_scroll_top != nxt or self._following_end != self._follow_end
        self._current_scroll_top = nxt
        self._following_end = self._follow_end
        self._follow_suppressed_at_end = False
        if changed:
            self._mark_scrollbar_activity()
            if self._request_render_callback is not None:
                self._request_render_callback()

    def update_layout(self, content_height: int, viewport_height: int, request_render) -> None:
        self._content_height = max(0, math.floor(content_height))
        self._current_viewport_height = max(0, math.floor(viewport_height))
        self._request_render_callback = request_render
        max_scroll_top = max(0, self._content_height - self._current_viewport_height)
        if self._following_end:
            self._current_scroll_top = max_scroll_top
        else:
            self._current_scroll_top = max(0, min(self._current_scroll_top, max_scroll_top))
        if self._current_scroll_top < max_scroll_top:
            self._follow_suppressed_at_end = False
        if self._follow_end and self._current_scroll_top == max_scroll_top and not self._follow_suppressed_at_end:
            self._following_end = True
        if self._content_height <= self._current_viewport_height:
            self._hide_transient_scrollbar()

    def add_child(self, component) -> None:
        raise Exception("ScrollView has exactly one child")

    def remove_child(self, component) -> None:
        raise Exception("ScrollView child cannot be removed")

    def clear(self) -> None:
        raise Exception("ScrollView child cannot be cleared")

    def render(self, width: int) -> list[str]:
        content_width = self.get_content_width(width)
        lines = self._child.render(content_width)
        return lines if content_width == width else [f"{line} " for line in lines]

    def _layout_node(self) -> dict:
        return {"type": "scroll", "component": self._child, "state": self}


setattr(ScrollView, LAYOUT_NODE, ScrollView._layout_node)
