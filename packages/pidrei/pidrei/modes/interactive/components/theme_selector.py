"""Mirror of pi coding-agent src/modes/interactive/components/theme-selector.ts."""

from pidrei_tui import Container, SelectList

from ..theme import get_select_list_theme
from .dynamic_border import DynamicBorder


THEME_SELECT_LIST_LAYOUT = {"minPrimaryColumnWidth": 12, "maxPrimaryColumnWidth": 32}


class ThemeSelectorComponent(Container):
    """Component that renders a theme selector."""

    def __init__(self, current_theme: str, themes: list, on_select, on_cancel, on_preview) -> None:
        super().__init__()
        self._on_preview = on_preview

        # `themes` is passed in: listing them reads the custom-theme
        # directory, which a constructor must not do on a runtime worker.
        theme_items = [
            {
                "value": name,
                "label": name,
                "description": "(current)" if name == current_theme else None,
            }
            for name in themes
        ]

        # Add top border
        self.add_child(DynamicBorder())

        # Create selector
        self._select_list = SelectList(theme_items, 10, get_select_list_theme(), THEME_SELECT_LIST_LAYOUT)

        # Preselect current theme
        if current_theme in themes:
            self._select_list.set_selected_index(themes.index(current_theme))

        self._select_list.on_select = lambda item: on_select(item["value"])
        self._select_list.on_cancel = lambda: on_cancel()
        self._select_list.on_selection_change = lambda item: self._on_preview(item["value"])

        self.add_child(self._select_list)

        # Add bottom border
        self.add_child(DynamicBorder())

    def get_select_list(self) -> SelectList:
        return self._select_list
