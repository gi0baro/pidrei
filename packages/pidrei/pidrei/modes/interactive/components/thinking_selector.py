"""Mirror of pi coding-agent src/modes/interactive/components/thinking-selector.ts."""

from pidrei_tui import Container, Input, SelectList, Spacer, Text, fuzzy_filter, get_keybindings, matches_key

from ..theme import get_select_list_theme, theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_display_text


THINKING_SELECT_LIST_LAYOUT = {"minPrimaryColumnWidth": 12, "maxPrimaryColumnWidth": 32}

LEVEL_DESCRIPTIONS = {
    "off": "No reasoning",
    "minimal": "Very brief reasoning (~1k tokens)",
    "low": "Light reasoning (~2k tokens)",
    "medium": "Moderate reasoning (~8k tokens)",
    "high": "Deep reasoning (~16k tokens)",
    "xhigh": "Extra-high reasoning (~32k tokens)",
    "max": "Maximum reasoning",
}


class ThinkingSelectorComponent(Container):
    """Component that renders a thinking level selector with borders."""

    def __init__(
        self,
        current_level: str,
        available_levels: list,
        on_select,
        on_cancel,
        on_select_as_default=None,
        default_thinking_level: str | None = None,
    ) -> None:
        super().__init__()
        self._on_select = on_select
        self._on_cancel = on_cancel
        self._on_select_as_default = on_select_as_default
        self._focused = False

        self._all_items = [
            {
                "value": level,
                "label": f"{'✓ ' if level == current_level else '  '}{level}",
                "description": f"{LEVEL_DESCRIPTIONS[level]} · default"
                if level == default_thinking_level
                else LEVEL_DESCRIPTIONS[level],
            }
            for level in available_levels
        ]

        # Add top border
        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))
        self.add_child(Text("Thinking Level", 0, 0))
        self.add_child(Spacer(1))
        self.add_child(Text(f"{key_display_text('app.thinking.cycle')} cycles thinking levels in-session", 0, 0))
        self.add_child(Spacer(1))

        # pi wires `searchInput.onSubmit` to forward Enter to the list; Enter is
        # a `tui.select.confirm` nav key, so `handle_input` routes it to the
        # list before the input ever sees it. See `settings_submenu`.
        self._search_input = Input()
        self.add_child(self._search_input)
        self.add_child(Spacer(1))

        # Create selector
        self._select_list = self._build_select_list(self._all_items, current_level)
        self._select_list_child_index = len(self.children)
        self.add_child(self._select_list)
        self.add_child(Spacer(1))
        self.add_child(Text(theme.fg("dim", "  Enter to select · Ctrl+S to set as default · Esc to cancel"), 0, 0))

        # Add bottom border
        self.add_child(DynamicBorder())

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._search_input.focused = value

    def _build_select_list(self, items: list, preselect: str | None = None) -> SelectList:
        select_list = SelectList(items, max(1, len(items)), get_select_list_theme(), THINKING_SELECT_LIST_LAYOUT)
        current_index = next((i for i, item in enumerate(items) if item["value"] == preselect), -1)
        if current_index != -1:
            select_list.set_selected_index(current_index)
        select_list.on_select = lambda item: self._on_select(item["value"])
        select_list.on_cancel = lambda: self._on_cancel()
        return select_list

    def _apply_filter(self, query: str) -> None:
        filtered = (
            fuzzy_filter(self._all_items, query, lambda item: f"{item['value']} {item.get('description') or ''}")
            if query
            else self._all_items
        )
        selected = self._select_list.get_selected_item()
        new_list = self._build_select_list(filtered, selected["value"] if selected is not None else None)
        children = list(self.children)
        children[self._select_list_child_index] = new_list
        self.set_children(children)
        self._select_list = new_list

    async def handle_input(self, key_data: str) -> None:
        if matches_key(key_data, "ctrl+s") and self._on_select_as_default is not None:
            item = self._select_list.get_selected_item()
            if item is not None:
                await self._on_select_as_default(item["value"])
            return

        kb = get_keybindings()
        is_nav = (
            kb.matches(key_data, "tui.select.up")
            or kb.matches(key_data, "tui.select.down")
            or kb.matches(key_data, "tui.select.confirm")
            or kb.matches(key_data, "tui.select.cancel")
        )
        if is_nav:
            await self._select_list.handle_input(key_data)
            return

        await self._search_input.handle_input(key_data)
        self._apply_filter(self._search_input.get_value())

    def get_select_list(self) -> SelectList:
        return self._select_list
