"""SettingsList component (port of pi tui ``components/settings-list.ts``).

Items are ``{"id", "label", "description"?, "currentValue", "values"?,
"submenu"?}`` records (``submenu`` is a callable receiving the current value
and an awaitable ``done(selected_value=None)`` callback and returning a
component); ``theme`` is a ``{"label", "value", "description", "cursor",
"hint"}`` record (``label``/``value`` take ``(text, selected)``); ``options``
mirrors ``SettingsListOptions`` (``{"enableSearch": bool}``).

``on_change``/``on_cancel`` must return an awaitable (async-only callback
policy): pi types them ``=> void``, but the port made several consumers async
(theme previews, settings persistence), and pi's bare call runs to completion
before the next statement — awaiting inline is the matching order.
"""

import math

from ..fuzzy import fuzzy_filter
from ..keybindings import get_keybindings
from ..utils import truncate_to_width, visible_width, wrap_text_with_ansi
from .input import Input


__all__ = ["SettingsList"]


class SettingsList:
    def __init__(
        self,
        items: list[dict],
        max_visible: int,
        theme: dict,
        on_change,
        on_cancel,
        options: dict | None = None,
    ) -> None:
        self._items = items
        self._filtered_items = items
        self._theme = theme
        self._selected_index = 0
        self._max_visible = max_visible
        self._on_change = on_change
        self._on_cancel = on_cancel
        options = options if options is not None else {}
        self._search_enabled = options.get("enableSearch", False)
        self._search_input = Input() if self._search_enabled else None

        # Submenu state
        self._submenu_component = None
        self._submenu_item_index: int | None = None

    def update_value(self, item_id: str, new_value: str) -> None:
        """Update an item's currentValue."""
        for item in self._items:
            if item["id"] == item_id:
                item["currentValue"] = new_value
                return

    def invalidate(self) -> None:
        if self._submenu_component is not None:
            invalidate = getattr(self._submenu_component, "invalidate", None)
            if invalidate is not None:
                invalidate()

    def render(self, width: int) -> list[str]:
        # If submenu is active, render it instead
        if self._submenu_component is not None:
            return self._submenu_component.render(width)

        return self._render_main_list(width)

    def _render_main_list(self, width: int) -> list[str]:
        lines: list[str] = []

        if self._search_enabled and self._search_input is not None:
            lines.extend(self._search_input.render(width))
            lines.append("")

        if not self._items:
            lines.append(self._theme["hint"]("  No settings available"))
            if self._search_enabled:
                self._add_hint_line(lines, width)
            return lines

        display_items = self._filtered_items if self._search_enabled else self._items
        if not display_items:
            lines.append(truncate_to_width(self._theme["hint"]("  No matching settings"), width))
            self._add_hint_line(lines, width)
            return lines

        # Calculate visible range with scrolling
        start_index = max(
            0,
            min(self._selected_index - math.floor(self._max_visible / 2), len(display_items) - self._max_visible),
        )
        end_index = min(start_index + self._max_visible, len(display_items))

        # Calculate max label width for alignment
        max_label_width = min(30, max(visible_width(item["label"]) for item in self._items))

        # Render visible items
        for i in range(start_index, end_index):
            item = display_items[i]

            is_selected = i == self._selected_index
            prefix = self._theme["cursor"] if is_selected else "  "
            prefix_width = visible_width(prefix)

            # Pad label to align values
            label_padded = item["label"] + " " * max(0, max_label_width - visible_width(item["label"]))
            label_text = self._theme["label"](label_padded, is_selected)

            # Calculate space for value
            separator = "  "
            used_width = prefix_width + max_label_width + visible_width(separator)
            value_max_width = width - used_width - 2

            value_text = self._theme["value"](truncate_to_width(item["currentValue"], value_max_width, ""), is_selected)

            lines.append(truncate_to_width(prefix + label_text + separator + value_text, width))

        # Add scroll indicator if needed
        if start_index > 0 or end_index < len(display_items):
            scroll_text = f"  ({self._selected_index + 1}/{len(display_items)})"
            lines.append(self._theme["hint"](truncate_to_width(scroll_text, width - 2, "")))

        # Add description for selected item
        selected_item = display_items[self._selected_index] if 0 <= self._selected_index < len(display_items) else None
        if selected_item is not None and selected_item.get("description"):
            lines.append("")
            wrapped_desc = wrap_text_with_ansi(selected_item["description"], width - 4)
            for line in wrapped_desc:
                lines.append(self._theme["description"](f"  {line}"))

        # Add hint
        self._add_hint_line(lines, width)

        return lines

    async def handle_input(self, data: str) -> None:
        # If submenu is active, delegate all input to it
        # The submenu's onCancel (triggered by escape) will call done() which closes it
        if self._submenu_component is not None:
            handle_input = getattr(self._submenu_component, "handle_input", None)
            if handle_input is not None:
                await handle_input(data)
            return

        # Main list input handling
        kb = get_keybindings()
        display_items = self._filtered_items if self._search_enabled else self._items
        if kb.matches(data, "tui.select.up"):
            if not display_items:
                return
            self._selected_index = len(display_items) - 1 if self._selected_index == 0 else self._selected_index - 1
        elif kb.matches(data, "tui.select.down"):
            if not display_items:
                return
            self._selected_index = 0 if self._selected_index == len(display_items) - 1 else self._selected_index + 1
        elif kb.matches(data, "tui.select.confirm") or data == " ":
            await self._activate_item()
        elif kb.matches(data, "tui.select.cancel"):
            await self._on_cancel()
        elif self._search_enabled and self._search_input is not None:
            sanitized = data.replace(" ", "")
            if not sanitized:
                return
            await self._search_input.handle_input(sanitized)
            self._apply_filter(self._search_input.get_value())

    async def _activate_item(self) -> None:
        items = self._filtered_items if self._search_enabled else self._items
        item = items[self._selected_index] if 0 <= self._selected_index < len(items) else None
        if item is None:
            return

        if item.get("submenu") is not None:
            # Open submenu, passing current value so it can pre-select correctly
            self._submenu_item_index = self._selected_index

            async def done(selected_value: str | None = None) -> None:
                if selected_value is not None:
                    item["currentValue"] = selected_value
                    await self._on_change(item["id"], selected_value)
                self._close_submenu()

            self._submenu_component = item["submenu"](item["currentValue"], done)
        elif item.get("values"):
            # Cycle through values
            values = item["values"]
            try:
                current_index = values.index(item["currentValue"])
            except ValueError:
                current_index = -1
            next_index = (current_index + 1) % len(values)
            new_value = values[next_index]
            item["currentValue"] = new_value
            await self._on_change(item["id"], new_value)

    def _close_submenu(self) -> None:
        self._submenu_component = None
        # Restore selection to the item that opened the submenu
        if self._submenu_item_index is not None:
            self._selected_index = self._submenu_item_index
            self._submenu_item_index = None

    def _apply_filter(self, query: str) -> None:
        self._filtered_items = fuzzy_filter(self._items, query, lambda item: item["label"])
        self._selected_index = 0

    def _add_hint_line(self, lines: list[str], width: int) -> None:
        lines.append("")
        lines.append(
            truncate_to_width(
                self._theme["hint"](
                    "  Type to search · Enter/Space to change · Esc to cancel"
                    if self._search_enabled
                    else "  Enter/Space to change · Esc to cancel"
                ),
                width,
            )
        )
