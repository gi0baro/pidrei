"""Input component - single-line text input with horizontal scrolling.

Port of pi tui ``components/input.ts``. Cursor positions are Python codepoint
indices (pi uses UTF-16 units); grapheme-aware stepping goes through the
``grapheme`` package instead of pi's shared ``Intl.Segmenter``.
"""

import grapheme as grapheme_lib

from ..keybindings import get_keybindings
from ..keys import decode_kitty_printable
from ..kill_ring import KillRing
from ..tui import CURSOR_MARKER, TuiMouseEvent, TuiMouseEventResult
from ..undo_stack import UndoStack
from ..utils import is_whitespace_char, slice_by_column, truncate_to_width, visible_width
from ..word_navigation import find_word_backward, find_word_forward


__all__ = ["Input"]


class Input:
    def __init__(self, options: dict | None = None) -> None:
        """``options`` mirrors pi's ``InputOptions``: ``prompt`` (default "> "),
        ``placeholder`` and ``placeholderStyle`` (a text -> styled text callable)."""
        options = options or {}
        self._value = ""
        self._cursor = 0  # Cursor position in the value
        self._prompt = options.get("prompt") if options.get("prompt") is not None else "> "
        self._placeholder = options.get("placeholder") or ""
        self._placeholder_style = options.get("placeholderStyle") or (lambda text: text)
        self._rendered_start_column = 0
        self.on_submit = None
        self.on_escape = None

        # Focusable interface - set by TUI when focus changes
        self.focused = False

        # Bracketed paste mode buffering
        self._paste_buffer = ""
        self._is_in_paste = False

        # Kill ring for Emacs-style kill/yank operations
        self._kill_ring = KillRing()
        self._last_action: str | None = None  # "kill" | "yank" | "type-word" | None

        # Undo support
        self._undo_stack = UndoStack()

    def get_value(self) -> str:
        return self._value

    def set_value(self, value: str) -> None:
        self._value = value
        self._cursor = min(self._cursor, len(value))

    async def handle_input(self, data: str) -> None:
        # Handle bracketed paste mode
        # Start of paste: \x1b[200~
        # End of paste: \x1b[201~

        # Check if we're starting a bracketed paste
        if "\x1b[200~" in data:
            self._is_in_paste = True
            self._paste_buffer = ""
            data = data.replace("\x1b[200~", "", 1)

        # If we're in a paste, buffer the data
        if self._is_in_paste:
            # Check if this chunk contains the end marker
            self._paste_buffer += data

            end_index = self._paste_buffer.find("\x1b[201~")
            if end_index != -1:
                # Extract the pasted content
                paste_content = self._paste_buffer[:end_index]

                # Process the complete paste
                self._handle_paste(paste_content)

                # Reset paste state
                self._is_in_paste = False

                # Handle any remaining input after the paste marker
                remaining = self._paste_buffer[end_index + 6 :]  # 6 = length of \x1b[201~
                self._paste_buffer = ""
                if remaining:
                    await self.handle_input(remaining)
            return

        kb = get_keybindings()

        # Escape/Cancel
        if kb.matches(data, "tui.select.cancel"):
            if self.on_escape is not None:
                self.on_escape()
            return

        # Undo
        if kb.matches(data, "tui.editor.undo"):
            self._undo()
            return

        # Submit
        if kb.matches(data, "tui.input.submit") or data == "\n":
            if self.on_submit is not None:
                self.on_submit(self._value)
            return

        # Deletion
        if kb.matches(data, "tui.editor.deleteCharBackward"):
            self._handle_backspace()
            return

        if kb.matches(data, "tui.editor.deleteCharForward"):
            self._handle_forward_delete()
            return

        if kb.matches(data, "tui.editor.deleteWordBackward"):
            self._delete_word_backwards()
            return

        if kb.matches(data, "tui.editor.deleteWordForward"):
            self._delete_word_forward()
            return

        if kb.matches(data, "tui.editor.deleteToLineStart"):
            self._delete_to_line_start()
            return

        if kb.matches(data, "tui.editor.deleteToLineEnd"):
            self._delete_to_line_end()
            return

        # Kill ring actions
        if kb.matches(data, "tui.editor.yank"):
            self._yank()
            return
        if kb.matches(data, "tui.editor.yankPop"):
            self._yank_pop()
            return

        # Cursor movement
        if kb.matches(data, "tui.editor.cursorLeft"):
            self._last_action = None
            if self._cursor > 0:
                before_cursor = self._value[: self._cursor]
                last_grapheme = None
                for last_grapheme in grapheme_lib.graphemes(before_cursor):  # noqa: B007
                    pass
                self._cursor -= len(last_grapheme) if last_grapheme else 1
            return

        if kb.matches(data, "tui.editor.cursorRight"):
            self._last_action = None
            if self._cursor < len(self._value):
                after_cursor = self._value[self._cursor :]
                first_grapheme = next(grapheme_lib.graphemes(after_cursor), None)
                self._cursor += len(first_grapheme) if first_grapheme else 1
            return

        if kb.matches(data, "tui.editor.cursorLineStart"):
            self._last_action = None
            self._cursor = 0
            return

        if kb.matches(data, "tui.editor.cursorLineEnd"):
            self._last_action = None
            self._cursor = len(self._value)
            return

        if kb.matches(data, "tui.editor.cursorWordLeft"):
            self._move_word_backwards()
            return

        if kb.matches(data, "tui.editor.cursorWordRight"):
            self._move_word_forwards()
            return

        # Kitty CSI-u printable character (e.g. \x1b[97u for 'a').
        # Terminals with Kitty protocol flag 1 (disambiguate) send CSI-u for all keys,
        # including plain printable characters. Decode before the control-char check
        # since CSI-u sequences contain \x1b which would be rejected.
        kitty_printable = decode_kitty_printable(data)
        if kitty_printable is not None:
            self._insert_character(kitty_printable)
            return

        # Regular character input - accept printable characters including Unicode,
        # but reject control characters (C0: 0x00-0x1F, DEL: 0x7F, C1: 0x80-0x9F)
        has_control_chars = any(ord(char) < 32 or ord(char) == 0x7F or 0x80 <= ord(char) <= 0x9F for char in data)
        if not has_control_chars:
            self._insert_character(data)

    async def handle_mouse(self, event: TuiMouseEvent) -> TuiMouseEventResult | None:
        if event.type != "press" or event.button != "left" or event.y != 0:
            return None
        visible_column = max(0, event.x - 2)
        target_column = self._rendered_start_column + visible_column
        current_column = 0
        self._cursor = len(self._value)
        index = 0
        for segment in grapheme_lib.graphemes(self._value):
            next_column = current_column + visible_width(segment)
            if target_column < next_column:
                self._cursor = index
                break
            current_column = next_column
            index += len(segment)
        self._last_action = None
        return TuiMouseEventResult(handled=True, focus=True)

    def _insert_character(self, char: str) -> None:
        # Undo coalescing: consecutive word chars coalesce into one undo unit
        if is_whitespace_char(char) or self._last_action != "type-word":
            self._push_undo()
        self._last_action = "type-word"

        self._value = self._value[: self._cursor] + char + self._value[self._cursor :]
        self._cursor += len(char)

    def _handle_backspace(self) -> None:
        self._last_action = None
        if self._cursor > 0:
            self._push_undo()
            before_cursor = self._value[: self._cursor]
            last_grapheme = None
            for last_grapheme in grapheme_lib.graphemes(before_cursor):  # noqa: B007
                pass
            grapheme_length = len(last_grapheme) if last_grapheme else 1
            self._value = self._value[: self._cursor - grapheme_length] + self._value[self._cursor :]
            self._cursor -= grapheme_length

    def _handle_forward_delete(self) -> None:
        self._last_action = None
        if self._cursor < len(self._value):
            self._push_undo()
            after_cursor = self._value[self._cursor :]
            first_grapheme = next(grapheme_lib.graphemes(after_cursor), None)
            grapheme_length = len(first_grapheme) if first_grapheme else 1
            self._value = self._value[: self._cursor] + self._value[self._cursor + grapheme_length :]

    def _delete_to_line_start(self) -> None:
        if self._cursor == 0:
            return
        self._push_undo()
        deleted_text = self._value[: self._cursor]
        self._kill_ring.push(deleted_text, prepend=True, accumulate=self._last_action == "kill")
        self._last_action = "kill"
        self._value = self._value[self._cursor :]
        self._cursor = 0

    def _delete_to_line_end(self) -> None:
        if self._cursor >= len(self._value):
            return
        self._push_undo()
        deleted_text = self._value[self._cursor :]
        self._kill_ring.push(deleted_text, prepend=False, accumulate=self._last_action == "kill")
        self._last_action = "kill"
        self._value = self._value[: self._cursor]

    def _delete_word_backwards(self) -> None:
        if self._cursor == 0:
            return

        # Save lastAction before cursor movement (moveWordBackwards resets it)
        was_kill = self._last_action == "kill"

        self._push_undo()

        old_cursor = self._cursor
        self._move_word_backwards()
        delete_from = self._cursor
        self._cursor = old_cursor

        deleted_text = self._value[delete_from : self._cursor]
        self._kill_ring.push(deleted_text, prepend=True, accumulate=was_kill)
        self._last_action = "kill"

        self._value = self._value[:delete_from] + self._value[self._cursor :]
        self._cursor = delete_from

    def _delete_word_forward(self) -> None:
        if self._cursor >= len(self._value):
            return

        # Save lastAction before cursor movement (moveWordForwards resets it)
        was_kill = self._last_action == "kill"

        self._push_undo()

        old_cursor = self._cursor
        self._move_word_forwards()
        delete_to = self._cursor
        self._cursor = old_cursor

        deleted_text = self._value[self._cursor : delete_to]
        self._kill_ring.push(deleted_text, prepend=False, accumulate=was_kill)
        self._last_action = "kill"

        self._value = self._value[: self._cursor] + self._value[delete_to:]

    def _yank(self) -> None:
        text = self._kill_ring.peek()
        if not text:
            return

        self._push_undo()

        self._value = self._value[: self._cursor] + text + self._value[self._cursor :]
        self._cursor += len(text)
        self._last_action = "yank"

    def _yank_pop(self) -> None:
        if self._last_action != "yank" or self._kill_ring.length <= 1:
            return

        self._push_undo()

        # Delete the previously yanked text (still at end of ring before rotation)
        prev_text = self._kill_ring.peek() or ""
        self._value = self._value[: self._cursor - len(prev_text)] + self._value[self._cursor :]
        self._cursor -= len(prev_text)

        # Rotate and insert new entry
        self._kill_ring.rotate()
        text = self._kill_ring.peek() or ""
        self._value = self._value[: self._cursor] + text + self._value[self._cursor :]
        self._cursor += len(text)
        self._last_action = "yank"

    def _push_undo(self) -> None:
        self._undo_stack.push({"value": self._value, "cursor": self._cursor})

    def _undo(self) -> None:
        snapshot = self._undo_stack.pop()
        if snapshot is None:
            return
        self._value = snapshot["value"]
        self._cursor = snapshot["cursor"]
        self._last_action = None

    def _move_word_backwards(self) -> None:
        if self._cursor == 0:
            return
        self._last_action = None
        self._cursor = find_word_backward(self._value, self._cursor)

    def _move_word_forwards(self) -> None:
        if self._cursor >= len(self._value):
            return
        self._last_action = None
        self._cursor = find_word_forward(self._value, self._cursor)

    def _handle_paste(self, pasted_text: str) -> None:
        self._last_action = None
        self._push_undo()

        # Clean the pasted text - remove newlines and carriage returns
        clean_text = pasted_text.replace("\r\n", "").replace("\r", "").replace("\n", "").replace("\t", "    ")

        # Insert at cursor position
        self._value = self._value[: self._cursor] + clean_text + self._value[self._cursor :]
        self._cursor += len(clean_text)

    def invalidate(self) -> None:
        # No cached state to invalidate currently
        pass

    def render(self, width: int) -> list[str]:
        # Calculate visible window
        prompt = self._prompt
        available_width = width - visible_width(prompt)

        if available_width <= 0:
            return [truncate_to_width(prompt, width, "")]

        if len(self._value) == 0 and self._placeholder:
            placeholder = truncate_to_width(self._placeholder, available_width, "")
            first_grapheme = next(grapheme_lib.graphemes(placeholder), None)
            at_cursor = first_grapheme if first_grapheme is not None else " "
            after_cursor = placeholder[len(at_cursor) :]
            marker = CURSOR_MARKER if self.focused else ""
            cursor_char = f"\x1b[7m{self._placeholder_style(at_cursor)}\x1b[27m"
            text_with_cursor = marker + cursor_char + self._placeholder_style(after_cursor)
            padding = " " * max(0, available_width - visible_width(text_with_cursor))
            return [prompt + text_with_cursor + padding]

        visible_text = ""
        cursor_display = self._cursor
        self._rendered_start_column = 0
        total_width = visible_width(self._value)

        if total_width < available_width:
            # Everything fits (leave room for cursor at end)
            visible_text = self._value
        else:
            # Need horizontal scrolling
            # Reserve one column for cursor if it's at the end
            scroll_width = available_width - 1 if self._cursor == len(self._value) else available_width
            cursor_col = visible_width(self._value[: self._cursor])

            if scroll_width > 0:
                half_width = scroll_width // 2

                if cursor_col < half_width:
                    # Cursor near start
                    start_col = 0
                elif cursor_col > total_width - half_width:
                    # Cursor near end
                    start_col = max(0, total_width - scroll_width)
                else:
                    # Cursor in middle
                    start_col = max(0, cursor_col - half_width)

                self._rendered_start_column = start_col
                visible_text = slice_by_column(self._value, start_col, scroll_width, True)
                before_cursor = slice_by_column(self._value, start_col, max(0, cursor_col - start_col), True)
                cursor_display = len(before_cursor)
            else:
                visible_text = ""
                cursor_display = 0

        # Build line with fake cursor
        # Insert cursor character at cursor position
        cursor_grapheme = next(grapheme_lib.graphemes(visible_text[cursor_display:]), None)

        before_cursor = visible_text[:cursor_display]
        at_cursor = cursor_grapheme if cursor_grapheme is not None else " "  # Char at cursor, or space if at end
        after_cursor = visible_text[cursor_display + len(at_cursor) :]

        # Hardware cursor marker (zero-width, emitted before fake cursor for IME positioning)
        marker = CURSOR_MARKER if self.focused else ""

        # Use inverse video to show cursor
        cursor_char = f"\x1b[7m{at_cursor}\x1b[27m"  # ESC[7m = reverse video, ESC[27m = normal
        text_with_cursor = before_cursor + marker + cursor_char + after_cursor

        # Calculate visual width
        visual_length = visible_width(text_with_cursor)
        padding = " " * max(0, available_width - visual_length)
        line = prompt + text_with_cursor + padding

        return [line]
