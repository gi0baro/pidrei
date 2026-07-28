"""SelectList component (port of pi tui ``components/select-list.ts``).

Items are ``{"value", "label", "description"?}`` records; ``theme`` is a
``{"selectedPrefix", "selectedText", "description", "scrollInfo", "noMatch"}``
record of style functions; ``layout`` mirrors pi's ``SelectListLayoutOptions``
(``{"minPrimaryColumnWidth", "maxPrimaryColumnWidth", "truncatePrimary"}``,
the latter receiving a ``{"text", "maxWidth", "columnWidth", "item",
"isSelected"}`` context record).
"""

import math
import re

from ..keybindings import get_keybindings
from ..utils import truncate_to_width, visible_width


__all__ = ["SelectList"]

DEFAULT_PRIMARY_COLUMN_WIDTH = 32
PRIMARY_COLUMN_GAP = 2
MIN_DESCRIPTION_WIDTH = 10

_SINGLE_LINE_RE = re.compile(r"[\r\n]+")


def _normalize_to_single_line(text: str) -> str:
    return _SINGLE_LINE_RE.sub(" ", text).strip()


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


class SelectList:
    def __init__(self, items: list[dict], max_visible: int, theme: dict, layout: dict | None = None) -> None:
        self._items = items
        self._filtered_items = items
        self._selected_index = 0
        self._max_visible = max_visible
        self._theme = theme
        self._layout = layout if layout is not None else {}

        self.on_select = None
        self.on_cancel = None
        self.on_selection_change = None

    def set_filter(self, filter_text: str) -> None:
        self._filtered_items = [item for item in self._items if item["value"].lower().startswith(filter_text.lower())]
        # Reset selection when filter changes
        self._selected_index = 0

    def set_selected_index(self, index: int) -> None:
        self._selected_index = max(0, min(index, len(self._filtered_items) - 1))

    def invalidate(self) -> None:
        # No cached state to invalidate currently
        pass

    def render(self, width: int) -> list[str]:
        lines: list[str] = []

        # If no items match filter, show message
        if not self._filtered_items:
            lines.append(self._theme["noMatch"]("  No matching commands"))
            return lines

        primary_column_width = self._get_primary_column_width()

        # Calculate visible range with scrolling
        start_index = max(
            0,
            min(
                self._selected_index - math.floor(self._max_visible / 2),
                len(self._filtered_items) - self._max_visible,
            ),
        )
        end_index = min(start_index + self._max_visible, len(self._filtered_items))

        # Render visible items
        for i in range(start_index, end_index):
            item = self._filtered_items[i]

            is_selected = i == self._selected_index
            description = item.get("description")
            description_single_line = _normalize_to_single_line(description) if description else None
            lines.append(self._render_item(item, is_selected, width, description_single_line, primary_column_width))

        # Add scroll indicators if needed
        if start_index > 0 or end_index < len(self._filtered_items):
            scroll_text = f"  ({self._selected_index + 1}/{len(self._filtered_items)})"
            # Truncate if too long for terminal
            lines.append(self._theme["scrollInfo"](truncate_to_width(scroll_text, width - 2, "")))

        return lines

    async def handle_input(self, key_data: str) -> None:
        kb = get_keybindings()
        # Up arrow - wrap to bottom when at top
        if kb.matches(key_data, "tui.select.up"):
            self._selected_index = (
                len(self._filtered_items) - 1 if self._selected_index == 0 else self._selected_index - 1
            )
            await self._notify_selection_change()
        # Down arrow - wrap to top when at bottom
        elif kb.matches(key_data, "tui.select.down"):
            self._selected_index = (
                0 if self._selected_index == len(self._filtered_items) - 1 else self._selected_index + 1
            )
            await self._notify_selection_change()
        # Enter
        elif kb.matches(key_data, "tui.select.confirm"):
            selected_item = (
                self._filtered_items[self._selected_index]
                if 0 <= self._selected_index < len(self._filtered_items)
                else None
            )
            if selected_item is not None and self.on_select is not None:
                # Callbacks are awaitable-returning (async-only policy): input
                # handling is async, and some selections persist.
                await self.on_select(selected_item)
        # Escape or Ctrl+C
        elif kb.matches(key_data, "tui.select.cancel"):
            if self.on_cancel is not None:
                await self.on_cancel()

    def _render_item(
        self,
        item: dict,
        is_selected: bool,
        width: int,
        description_single_line: str | None,
        primary_column_width: int,
    ) -> str:
        prefix = "→ " if is_selected else "  "
        prefix_width = visible_width(prefix)

        if description_single_line and width > 40:
            effective_primary_column_width = max(1, min(primary_column_width, width - prefix_width - 4))
            max_primary_width = max(1, effective_primary_column_width - PRIMARY_COLUMN_GAP)
            truncated_value = self._truncate_primary(
                item, is_selected, max_primary_width, effective_primary_column_width
            )
            truncated_value_width = visible_width(truncated_value)
            spacing = " " * max(1, effective_primary_column_width - truncated_value_width)
            description_start = prefix_width + truncated_value_width + len(spacing)
            remaining_width = width - description_start - 2  # -2 for safety

            if remaining_width > MIN_DESCRIPTION_WIDTH:
                truncated_desc = truncate_to_width(description_single_line, remaining_width, "")
                if is_selected:
                    return self._theme["selectedText"](f"{prefix}{truncated_value}{spacing}{truncated_desc}")

                desc_text = self._theme["description"](spacing + truncated_desc)
                return prefix + truncated_value + desc_text

        max_width = width - prefix_width - 2
        truncated_value = self._truncate_primary(item, is_selected, max_width, max_width)
        if is_selected:
            return self._theme["selectedText"](f"{prefix}{truncated_value}")

        return prefix + truncated_value

    def _get_primary_column_width(self) -> int:
        bounds = self._get_primary_column_bounds()
        widest_primary = 0
        for item in self._filtered_items:
            widest_primary = max(widest_primary, visible_width(self._get_display_value(item)) + PRIMARY_COLUMN_GAP)

        return _clamp(widest_primary, bounds["min"], bounds["max"])

    def _get_primary_column_bounds(self) -> dict:
        min_option = self._layout.get("minPrimaryColumnWidth")
        max_option = self._layout.get("maxPrimaryColumnWidth")
        raw_min = (
            min_option
            if min_option is not None
            else (max_option if max_option is not None else DEFAULT_PRIMARY_COLUMN_WIDTH)
        )
        raw_max = (
            max_option
            if max_option is not None
            else (min_option if min_option is not None else DEFAULT_PRIMARY_COLUMN_WIDTH)
        )

        return {
            "min": max(1, min(raw_min, raw_max)),
            "max": max(1, max(raw_min, raw_max)),
        }

    def _truncate_primary(self, item: dict, is_selected: bool, max_width: int, column_width: int) -> str:
        display_value = self._get_display_value(item)
        truncate_primary = self._layout.get("truncatePrimary")
        if truncate_primary is not None:
            truncated_value = truncate_primary(
                {
                    "text": display_value,
                    "maxWidth": max_width,
                    "columnWidth": column_width,
                    "item": item,
                    "isSelected": is_selected,
                }
            )
        else:
            truncated_value = truncate_to_width(display_value, max_width, "")

        return truncate_to_width(truncated_value, max_width, "")

    def _get_display_value(self, item: dict) -> str:
        return item.get("label") or item["value"]

    async def _notify_selection_change(self) -> None:
        selected_item = (
            self._filtered_items[self._selected_index]
            if 0 <= self._selected_index < len(self._filtered_items)
            else None
        )
        if selected_item is not None and self.on_selection_change is not None:
            # Awaitable-returning like `on_select`; pi runs the callback to
            # completion before the next render, so awaiting inline (not
            # detaching) is the matching order.
            await self.on_selection_change(selected_item)

    def get_selected_item(self) -> dict | None:
        if 0 <= self._selected_index < len(self._filtered_items):
            return self._filtered_items[self._selected_index]
        return None
