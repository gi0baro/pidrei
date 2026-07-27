"""Interface for custom editor components (port of pi tui ``editor-component.ts``).

This allows extensions to provide their own editor implementation (e.g. vim
mode, emacs mode, custom keybindings) while maintaining compatibility with
the core application.
"""

from typing import Protocol


__all__ = ["EditorComponent"]


class EditorComponent(Protocol):
    """Component contract for editors (extends the ``Component`` protocol).

    Required: ``get_text``/``set_text``/``handle_input`` plus the ``Component``
    render surface, and the ``on_submit``/``on_change`` callback attributes.

    Optional (checked via ``hasattr`` by consumers, mirroring pi's optional
    interface members): ``add_to_history``, ``insert_text_at_cursor``,
    ``get_expanded_text`` (falls back to ``get_text``),
    ``set_autocomplete_provider``, ``border_color``, ``set_padding_x`` and
    ``set_autocomplete_max_visible``.
    """

    # Callbacks: on_submit(text) called when user submits (e.g. Enter key);
    # on_change(text) called when text changes. Either may be None.
    on_submit: object
    on_change: object

    def render(self, width: int) -> list[str]:
        """Render the component to lines for the given viewport width."""
        ...

    def invalidate(self) -> None:
        """Invalidate any cached rendering state."""
        ...

    def get_text(self) -> str:
        """Get the current text content."""
        ...

    def set_text(self, text: str) -> None:
        """Set the text content."""
        ...

    async def handle_input(self, data: str) -> None:
        """Handle raw terminal input (key presses, paste sequences, etc.)."""
        ...
