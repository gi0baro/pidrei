"""Mirror of pi coding-agent src/modes/interactive/components/user-message-selector.ts.

Message items are ``{"id", "text", "timestamp"?}`` records.
"""

from pidrei_tui import Container, Spacer, Text, get_keybindings, truncate_to_width
from pidrei_tui._timers import Timeout

from ..theme import theme
from .dynamic_border import DynamicBorder


class UserMessageList:
    """Custom user message list component with selection."""

    def __init__(self, messages: list, initial_selected_id: str | None = None) -> None:
        # Store messages in chronological order (oldest to newest)
        self._messages = messages
        self.on_select = None
        self.on_cancel = None
        self._max_visible = 10  # Max messages visible
        initial_index = (
            next((i for i, message in enumerate(messages) if message["id"] == initial_selected_id), -1)
            if initial_selected_id
            else -1
        )
        # Start with selected message if provided, else default to the most recent
        self._selected_index = initial_index if initial_index >= 0 else max(0, len(messages) - 1)

    def invalidate(self) -> None:
        # No cached state to invalidate currently
        pass

    def render(self, width: int) -> list:
        lines: list = []

        if not self._messages:
            lines.append(theme.fg("muted", "  No user messages found"))
            return lines

        # Calculate visible range with scrolling
        start_index = max(
            0,
            min(self._selected_index - self._max_visible // 2, len(self._messages) - self._max_visible),
        )
        end_index = min(start_index + self._max_visible, len(self._messages))

        # Render visible messages (2 lines per message + blank line)
        for i in range(start_index, end_index):
            message = self._messages[i]
            is_selected = i == self._selected_index

            # Normalize message to single line
            normalized_message = message["text"].replace("\n", " ").strip()

            # First line: cursor + message
            cursor = theme.fg("accent", "› ") if is_selected else "  "
            max_msg_width = width - 2  # Account for cursor (2 chars)
            truncated_msg = truncate_to_width(normalized_message, max_msg_width)
            message_line = cursor + (theme.bold(truncated_msg) if is_selected else truncated_msg)

            lines.append(message_line)

            # Second line: metadata (position in history)
            position = i + 1
            metadata = f"  Message {position} of {len(self._messages)}"
            lines.append(theme.fg("muted", metadata))
            lines.append("")  # Blank line between messages

        # Add scroll indicator if needed
        if start_index > 0 or end_index < len(self._messages):
            lines.append(theme.fg("muted", f"  ({self._selected_index + 1}/{len(self._messages)})"))

        return lines

    def handle_input(self, key_data: str) -> None:
        kb = get_keybindings()
        # Up arrow - go to previous (older) message, wrap to bottom when at top
        if kb.matches(key_data, "tui.select.up"):
            self._selected_index = len(self._messages) - 1 if self._selected_index == 0 else self._selected_index - 1
        # Down arrow - go to next (newer) message, wrap to top when at bottom
        elif kb.matches(key_data, "tui.select.down"):
            self._selected_index = 0 if self._selected_index == len(self._messages) - 1 else self._selected_index + 1
        # Enter - select message and branch
        elif kb.matches(key_data, "tui.select.confirm"):
            if 0 <= self._selected_index < len(self._messages) and self.on_select is not None:
                self.on_select(self._messages[self._selected_index]["id"])
        # Escape - cancel
        elif kb.matches(key_data, "tui.select.cancel") and self.on_cancel is not None:
            self.on_cancel()


class UserMessageSelectorComponent(Container):
    """Component that renders a user message selector for branching."""

    def __init__(self, messages: list, on_select, on_cancel, initial_selected_id: str | None = None) -> None:
        super().__init__()

        # Add header
        self.add_child(Spacer(1))
        self.add_child(Text(theme.bold("Fork from Message"), 1, 0))
        self.add_child(
            Text(
                theme.fg(
                    "muted",
                    "Select a user message to copy the active path up to that point into a new session",
                ),
                1,
                0,
            )
        )
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))

        # Create message list
        self._message_list = UserMessageList(messages, initial_selected_id)
        self._message_list.on_select = on_select
        self._message_list.on_cancel = on_cancel

        self.add_child(self._message_list)

        # Add bottom border
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())

        # Auto-cancel if no messages
        if not messages:
            Timeout(100, lambda: on_cancel())

    def get_message_list(self) -> UserMessageList:
        return self._message_list
