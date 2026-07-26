"""Mirror of pi coding-agent src/modes/interactive/components/custom-entry.ts."""

from pidrei_tui import Box, Container, Spacer, Text

from ..theme import theme


class CustomEntryComponent(Container):
    """Renders a custom session entry from extensions.

    The host owns transcript spacing; renderer output should provide only its
    content.
    """

    def __init__(self, entry: dict, renderer) -> None:
        super().__init__()
        self._entry = entry
        self._renderer = renderer
        self._custom_component = None
        self._expanded = False
        self._rebuild()

    def has_content(self) -> bool:
        return self._custom_component is not None

    def set_expanded(self, expanded: bool) -> None:
        if self._expanded != expanded:
            self._expanded = expanded
            self._rebuild()

    def invalidate(self) -> None:
        super().invalidate()
        self._rebuild()

    def _rebuild(self) -> None:
        self.clear()
        self._custom_component = None

        try:
            component = self._renderer(self._entry, {"expanded": self._expanded}, theme)
        except Exception as error:
            box = Box(1, 1, lambda text: theme.bg("customMessageBg", text))
            box.add_child(Text(theme.fg("error", f"[{self._entry['customType']}] renderer failed: {error}"), 0, 0))
            component = box

        if component is None:
            return

        self._custom_component = component
        self.add_child(Spacer(1))
        self.add_child(component)
