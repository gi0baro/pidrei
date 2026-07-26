"""Mirror of pi coding-agent src/modes/interactive/components/thinking-selector.ts."""

from pidrei_tui import Container, SelectList

from ..theme import get_select_list_theme
from .dynamic_border import DynamicBorder


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

    def __init__(self, current_level: str, available_levels: list, on_select, on_cancel) -> None:
        super().__init__()

        thinking_levels = [
            {"value": level, "label": level, "description": LEVEL_DESCRIPTIONS[level]} for level in available_levels
        ]

        # Add top border
        self.add_child(DynamicBorder())

        # Create selector
        self._select_list = SelectList(
            thinking_levels, len(thinking_levels), get_select_list_theme(), THINKING_SELECT_LIST_LAYOUT
        )

        # Preselect current level
        current_index = next((i for i, item in enumerate(thinking_levels) if item["value"] == current_level), -1)
        if current_index != -1:
            self._select_list.set_selected_index(current_index)

        self._select_list.on_select = lambda item: on_select(item["value"])
        self._select_list.on_cancel = lambda: on_cancel()

        self.add_child(self._select_list)

        # Add bottom border
        self.add_child(DynamicBorder())

    def get_select_list(self) -> SelectList:
        return self._select_list
