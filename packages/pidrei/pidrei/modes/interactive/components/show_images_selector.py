"""Mirror of pi coding-agent src/modes/interactive/components/show-images-selector.ts."""

from pidrei_tui import Container, SelectList

from ..theme import get_select_list_theme
from .dynamic_border import DynamicBorder


SHOW_IMAGES_SELECT_LIST_LAYOUT = {"minPrimaryColumnWidth": 12, "maxPrimaryColumnWidth": 32}


class ShowImagesSelectorComponent(Container):
    """Component that renders a show images selector with borders."""

    def __init__(self, current_value: bool, on_select, on_cancel) -> None:
        super().__init__()

        items = [
            {"value": "yes", "label": "Yes", "description": "Show images inline in terminal"},
            {"value": "no", "label": "No", "description": "Show text placeholder instead"},
        ]

        # Add top border
        self.add_child(DynamicBorder())

        # Create selector
        self._select_list = SelectList(items, 5, get_select_list_theme(), SHOW_IMAGES_SELECT_LIST_LAYOUT)

        # Preselect current value
        self._select_list.set_selected_index(0 if current_value else 1)

        self._select_list.on_select = lambda item: on_select(item["value"] == "yes")
        self._select_list.on_cancel = lambda: on_cancel()

        self.add_child(self._select_list)

        # Add bottom border
        self.add_child(DynamicBorder())

    def get_select_list(self) -> SelectList:
        return self._select_list
