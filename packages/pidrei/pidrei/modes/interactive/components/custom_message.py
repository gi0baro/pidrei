"""Mirror of pi coding-agent src/modes/interactive/components/custom-message.ts."""

from pidrei_tui import Box, Container, Markdown, Spacer, Text

from ..theme import get_markdown_theme, theme


class CustomMessageComponent(Container):
    """Renders a custom message entry from extensions.

    Uses distinct styling to differentiate from user messages.
    """

    def __init__(self, message, custom_renderer=None, markdown_theme: dict | None = None) -> None:
        super().__init__()
        self._message = message
        self._custom_renderer = custom_renderer
        self._markdown_theme = markdown_theme if markdown_theme is not None else get_markdown_theme()
        self._custom_component = None
        self._expanded = False

        self.add_child(Spacer(1))

        # Create box with purple background (used for default rendering)
        self._box = Box(1, 1, lambda t: theme.bg("customMessageBg", t))

        self._rebuild()

    def set_expanded(self, expanded: bool) -> None:
        if self._expanded != expanded:
            self._expanded = expanded
            self._rebuild()

    def invalidate(self) -> None:
        super().invalidate()
        self._rebuild()

    def _rebuild(self) -> None:
        # Remove previous content component
        if self._custom_component is not None:
            self.remove_child(self._custom_component)
            self._custom_component = None
        self.remove_child(self._box)

        # Try custom renderer first - it handles its own styling
        if self._custom_renderer is not None:
            try:
                component = self._custom_renderer(self._message, {"expanded": self._expanded}, theme)
                if component is not None:
                    # Custom renderer provides its own styled component
                    self._custom_component = component
                    self.add_child(component)
                    return
            except Exception:
                # Fall through to default rendering
                pass

        # Default rendering uses our box
        self.add_child(self._box)
        self._box.clear()

        # Default rendering: label + content
        label = theme.fg("customMessageLabel", f"\x1b[1m[{self._message.custom_type}]\x1b[22m")
        self._box.add_child(Text(label, 0, 0))
        self._box.add_child(Spacer(1))

        # Extract text content
        content = self._message.content
        if isinstance(content, str):
            text = content
        else:
            text = "\n".join(c["text"] for c in content if c.get("type") == "text")

        self._box.add_child(
            Markdown(
                text,
                0,
                0,
                self._markdown_theme,
                {"color": lambda t: theme.fg("customMessageText", t)},
            )
        )
