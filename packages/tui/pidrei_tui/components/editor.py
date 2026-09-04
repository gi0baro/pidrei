"""Multi-line editor component (port of pi tui ``components/editor.ts``).

Cursor positions are Python codepoint indices (pi uses UTF-16 units).
Grapheme segmentation goes through the pure-Python ``grapheme`` package and
word segmentation through ``utils.get_word_segmenter()``; both produce
``{"segment", "index", ...}`` records mirroring ``Intl.SegmentData``.

Async differences from pi (JS single-threaded event loop → tonio):

- the autocomplete debounce timer and the application of a provider's
  response run on the TUI's owner task (``tui.input_owner``), the same task
  that runs ``handle_input`` — editor state has exactly one writer, as on
  pi's thread; the provider call itself runs off it, one at a time (pi
  chains promises via ``autocompleteRequestTask``; here a ``sync.Lock``);
- the abort signal is a ``CancelToken`` (pi uses a DOM ``AbortController``).

Autocomplete therefore requires the editor to live on a tonio runtime; all
purely synchronous editing (no provider set) works without one.
"""

import copy
import math
import re
from dataclasses import replace

import grapheme as grapheme_lib
from tonio.colored import sync

from .._owner import TimerHandle
from ..keybindings import get_keybindings
from ..keys import decode_printable_key, matches_key
from ..kill_ring import KillRing
from ..tui import CURSOR_MARKER, TuiMouseEvent, TuiMouseEventResult
from ..undo_stack import UndoStack
from ..utils import (
    cjk_break_regex,
    get_word_segmenter,
    is_whitespace_char,
    slice_by_column,
    visible_width,
)
from ..word_navigation import find_word_backward, find_word_forward
from .cancellable_loader import CancelToken
from .select_list import SelectList


__all__ = ["Editor", "word_wrap_line"]

_word_segmenter = get_word_segmenter()


def _grapheme_segments(text: str) -> list[dict]:
    """Segment *text* into grapheme records (pi: ``getGraphemeSegmenter()``)."""
    result = []
    index = 0
    for segment in grapheme_lib.graphemes(text):
        result.append({"segment": segment, "index": index})
        index += len(segment)
    return result


# Regex matching paste markers like `[paste #1 +123 lines]` or `[paste #2 1234 chars]`.
PASTE_MARKER_REGEX = re.compile(r"\[paste #(\d+)( (\+\d+ lines|\d+ chars))?\]")

# Anchored version for single-segment testing.
PASTE_MARKER_SINGLE = re.compile(r"^\[paste #(\d+)( (\+\d+ lines|\d+ chars))?\]$")


def _is_paste_marker(segment: str) -> bool:
    """Check if a segment is a paste marker (i.e. was merged by segment_with_markers)."""
    return len(segment) >= 10 and PASTE_MARKER_SINGLE.match(segment) is not None


def _segment_with_markers(text: str, base_segment, valid_ids: set[int]) -> list[dict]:
    """Segment *text*, merging graphemes inside paste markers into single atomic segments.

    This makes cursor movement, deletion, word-wrap, etc. treat paste markers
    as single units. Only markers whose numeric ID exists in *valid_ids* are
    merged.
    """
    # Fast path: no paste markers in the text or no valid IDs.
    if not valid_ids or "[paste #" not in text:
        return list(base_segment(text))

    # Find all marker spans with valid IDs.
    markers = []
    for match in PASTE_MARKER_REGEX.finditer(text):
        if int(match.group(1)) not in valid_ids:
            continue
        markers.append({"start": match.start(), "end": match.end()})
    if not markers:
        return list(base_segment(text))

    # Build merged segment list.
    result: list[dict] = []
    marker_idx = 0

    for seg in base_segment(text):
        # Skip past markers that are entirely before this segment.
        while marker_idx < len(markers) and markers[marker_idx]["end"] <= seg["index"]:
            marker_idx += 1

        marker = markers[marker_idx] if marker_idx < len(markers) else None

        if marker is not None and marker["start"] <= seg["index"] < marker["end"]:
            # This segment falls inside a marker.
            # If this is the first segment of the marker, emit a merged segment.
            if seg["index"] == marker["start"]:
                marker_text = text[marker["start"] : marker["end"]]
                result.append({"segment": marker_text, "index": marker["start"], "isWordLike": False})
            # Otherwise skip (already merged into the first segment).
        else:
            result.append(seg)

    return result


def word_wrap_line(line: str, max_width: int, pre_segmented: list[dict] | None = None) -> list[dict]:
    """Split a line into word-wrapped chunks ``{"text", "startIndex", "endIndex"}``.

    Wraps at word boundaries when possible, falling back to character-level
    wrapping for words longer than the available width. *pre_segmented* is an
    optional pre-segmented grapheme list (e.g. with paste-marker awareness).
    """
    if not line or max_width <= 0:
        return [{"text": "", "startIndex": 0, "endIndex": 0}]

    line_width = visible_width(line)
    if line_width <= max_width:
        return [{"text": line, "startIndex": 0, "endIndex": len(line)}]

    chunks: list[dict] = []
    segments = pre_segmented if pre_segmented is not None else _grapheme_segments(line)

    current_width = 0
    chunk_start = 0

    # Wrap opportunity: the position after the last whitespace before a
    # non-whitespace grapheme, i.e. where a line break is allowed.
    wrap_opp_index = -1
    wrap_opp_width = 0

    for i, seg in enumerate(segments):
        grapheme = seg["segment"]
        g_width = visible_width(grapheme)
        char_index = seg["index"]
        is_ws = not _is_paste_marker(grapheme) and is_whitespace_char(grapheme)

        # Overflow check before advancing.
        if current_width + g_width > max_width:
            if wrap_opp_index >= 0 and current_width - wrap_opp_width + g_width <= max_width:
                # Backtrack to last wrap opportunity (the remaining content
                # plus the current grapheme still fits within max_width).
                chunks.append(
                    {"text": line[chunk_start:wrap_opp_index], "startIndex": chunk_start, "endIndex": wrap_opp_index}
                )
                chunk_start = wrap_opp_index
                current_width -= wrap_opp_width
            elif chunk_start < char_index:
                # No viable wrap opportunity: force-break at current position.
                # This also handles the case where backtracking to a word
                # boundary wouldn't help because the remaining content plus
                # the current grapheme (e.g. a wide character) still exceeds
                # max_width.
                chunks.append({"text": line[chunk_start:char_index], "startIndex": chunk_start, "endIndex": char_index})
                chunk_start = char_index
                current_width = 0
            wrap_opp_index = -1

        if g_width > max_width:
            # Single atomic segment wider than max_width (e.g. paste marker
            # in a narrow terminal). Re-wrap it at grapheme granularity.

            # The segment remains logically atomic for cursor
            # movement / editing — the split is purely visual for word-wrap layout.
            sub_chunks = word_wrap_line(grapheme, max_width)
            for sc in sub_chunks[:-1]:
                chunks.append(
                    {
                        "text": sc["text"],
                        "startIndex": char_index + sc["startIndex"],
                        "endIndex": char_index + sc["endIndex"],
                    }
                )
            last = sub_chunks[-1]
            chunk_start = char_index + last["startIndex"]
            current_width = visible_width(last["text"])
            wrap_opp_index = -1
            continue

        # Advance.
        current_width += g_width

        # Record wrap opportunity: whitespace followed by non-whitespace
        # (multiple spaces join; the break point is after the last space),
        # or at a boundary where either side is CJK (CJK allows breaking
        # between any adjacent characters).
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        if is_ws and nxt is not None and (_is_paste_marker(nxt["segment"]) or not is_whitespace_char(nxt["segment"])):
            wrap_opp_index = nxt["index"]
            wrap_opp_width = current_width
        elif not is_ws and nxt is not None and not is_whitespace_char(nxt["segment"]):
            is_cjk = not _is_paste_marker(grapheme) and cjk_break_regex.search(grapheme) is not None
            next_is_cjk = not _is_paste_marker(nxt["segment"]) and cjk_break_regex.search(nxt["segment"]) is not None
            if is_cjk or next_is_cjk:
                wrap_opp_index = nxt["index"]
                wrap_opp_width = current_width

    # Push final chunk.
    chunks.append({"text": line[chunk_start:], "startIndex": chunk_start, "endIndex": len(line)})

    return chunks


SLASH_COMMAND_SELECT_LIST_LAYOUT = {"minPrimaryColumnWidth": 12, "maxPrimaryColumnWidth": 32}

ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS = 20
DEFAULT_AUTOCOMPLETE_TRIGGER_CHARACTERS = ["@", "#"]

_CHARACTER_CLASS_ESCAPE_RE = re.compile(r"[\\^$.*+?()\[\]{}|-]")


def _escape_character_class(value: str) -> str:
    return _CHARACTER_CLASS_ESCAPE_RE.sub(lambda m: "\\" + m.group(0), value)


def _build_trigger_pattern(trigger_characters: list[str]):
    return re.compile(r"(?:^|[\s])[" + "".join(map(_escape_character_class, trigger_characters)) + r"][^\s]*$")


def _build_debounce_pattern(trigger_characters: list[str]):
    escaped_without_at = [_escape_character_class(c) for c in trigger_characters if c != "@"]
    return re.compile(r"(?:^|[ \t])(?:@(?:\"[^\"]*|[^\s]*)|[" + "".join(escaped_without_at) + r"][^\s]*)$")


def _create_scroll_border(direction: str, hidden_line_count: int, width: int) -> str:
    available_width = max(0, width)
    indicator = f"─── {direction} {hidden_line_count} more "
    remaining = available_width - visible_width(indicator)
    if remaining >= 0:
        return indicator + "─" * remaining

    ellipsis = "..."[:available_width]
    indicator_width = available_width - visible_width(ellipsis)
    return slice_by_column(indicator, 0, indicator_width, True) + ellipsis


def _is_finite_number(value) -> bool:
    """Mirror JS ``Number.isFinite`` for option validation."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class Editor:
    """Multi-line editor with history, kill ring, undo and autocomplete."""

    def __init__(self, tui, theme: dict, options: dict | None = None) -> None:
        options = options or {}
        self._state = {"lines": [""], "cursorLine": 0, "cursorCol": 0}
        self._layout_cache: tuple[tuple, list[dict]] | None = None

        # Focusable interface - set by TUI when focus changes
        self.focused = False

        self._tui = tui
        self._theme = theme

        # Store last render geometry for cursor navigation and mouse hit-testing.
        self._last_width = 80
        self._rendered_visible_line_count = 1
        self._rendered_autocomplete_height = 0

        # Vertical scrolling support
        self._scroll_offset = 0

        # Border color (can be changed dynamically)
        self.border_color = theme["borderColor"]

        # Autocomplete support
        self._autocomplete_provider = None
        self._autocomplete_trigger_characters = [*DEFAULT_AUTOCOMPLETE_TRIGGER_CHARACTERS]
        self._autocomplete_trigger_pattern = _build_trigger_pattern(self._autocomplete_trigger_characters)
        self._autocomplete_debounce_pattern = _build_debounce_pattern(self._autocomplete_trigger_characters)
        self._autocomplete_list: SelectList | None = None
        self._autocomplete_state: str | None = None  # "regular" | "force" | None
        self._autocomplete_prefix = ""
        self._autocomplete_abort: CancelToken | None = None
        self._autocomplete_debounce_timer: TimerHandle | None = None
        self._autocomplete_request_lock = sync.Lock()  # pi: `autocompleteRequestTask` chain
        self._autocomplete_start_token = 0
        self._autocomplete_request_id = 0

        # Paste tracking for large pastes
        self._pastes: dict[int, str] = {}
        self._paste_counter = 0

        # Bracketed paste mode buffering
        self._paste_buffer = ""
        self._is_in_paste = False

        # Prompt history for up/down navigation
        self._history: list[str] = []
        self._history_index = -1  # -1 = not browsing, 0 = most recent, 1 = older, etc.
        self._history_draft: dict | None = None

        # Kill ring for Emacs-style kill/yank operations
        self._kill_ring = KillRing()
        self._last_action: str | None = None  # "kill" | "yank" | "type-word" | None

        # Character jump mode
        self._jump_mode: str | None = None  # "forward" | "backward" | None

        # Preferred visual column for vertical cursor movement (sticky column)
        self._preferred_visual_col: int | None = None

        # When the cursor is snapped to the start of an atomic segment, e.g. a
        # paste marker, cursorCol no longer reflects where the cursor would have
        # landed. This field stores the pre-snap cursorCol so that the next
        # vertical move can resolve it to a visual column on whatever VL it
        # belongs to.
        self._snapped_from_cursor_col: int | None = None

        # Undo support
        self._undo_stack = UndoStack()

        self.on_submit = None
        self.on_change = None
        self.disable_submit = False

        padding_x = options.get("paddingX", 0)
        self._padding_x = max(0, math.floor(padding_x)) if _is_finite_number(padding_x) else 0
        max_visible = options.get("autocompleteMaxVisible", 5)
        self._autocomplete_max_visible = (
            max(3, min(20, math.floor(max_visible))) if _is_finite_number(max_visible) else 5
        )

    def _valid_paste_ids(self) -> set[int]:
        """Set of currently valid paste IDs, for marker-aware segmentation."""
        return set(self._pastes.keys())

    def _segment(self, text: str, mode: str) -> list[dict]:
        """Segment text with paste-marker awareness, only merging markers with valid IDs."""
        base = _word_segmenter.segment if mode == "word" else _grapheme_segments
        return _segment_with_markers(text, base, self._valid_paste_ids())

    def get_padding_x(self) -> int:
        return self._padding_x

    def set_padding_x(self, padding) -> None:
        new_padding = max(0, math.floor(padding)) if _is_finite_number(padding) else 0
        if self._padding_x != new_padding:
            self._padding_x = new_padding
            self._tui.request_render()

    def get_autocomplete_max_visible(self) -> int:
        return self._autocomplete_max_visible

    def set_autocomplete_max_visible(self, max_visible) -> None:
        new_max_visible = max(3, min(20, math.floor(max_visible))) if _is_finite_number(max_visible) else 5
        if self._autocomplete_max_visible != new_max_visible:
            self._autocomplete_max_visible = new_max_visible
            self._tui.request_render()

    def set_autocomplete_provider(self, provider) -> None:
        self._cancel_autocomplete()
        self._autocomplete_provider = provider
        self._set_autocomplete_trigger_characters(getattr(provider, "trigger_characters", None) or [])

    def add_to_history(self, text: str) -> None:
        """Add a prompt to history for up/down arrow navigation.

        Called after successful submission.
        """
        trimmed = text.strip()
        if not trimmed:
            return
        # Don't add consecutive duplicates
        if self._history and self._history[0] == trimmed:
            return
        self._history.insert(0, trimmed)
        # Limit history size
        if len(self._history) > 100:
            self._history.pop()

    def _is_editor_empty(self) -> bool:
        return len(self._state["lines"]) == 1 and self._state["lines"][0] == ""

    def _is_on_first_visual_line(self) -> bool:
        visual_lines = self._build_visual_line_map(self._last_width)
        return self._find_current_visual_line(visual_lines) == 0

    def _is_on_last_visual_line(self) -> bool:
        visual_lines = self._build_visual_line_map(self._last_width)
        return self._find_current_visual_line(visual_lines) == len(visual_lines) - 1

    def _navigate_history(self, direction: int) -> None:
        self._last_action = None
        if not self._history:
            return

        new_index = self._history_index - direction  # Up(-1) increases index, Down(1) decreases
        if new_index < -1 or new_index >= len(self._history):
            return

        # Capture state when first entering history browsing mode
        if self._history_index == -1 and new_index >= 0:
            self._push_undo_snapshot()
            self._history_draft = copy.deepcopy(self._state)

        self._history_index = new_index

        if self._history_index == -1:
            draft = self._history_draft
            self._history_draft = None
            if draft is not None:
                self._state = draft
                self._preferred_visual_col = None
                self._snapped_from_cursor_col = None
                self._scroll_offset = 0
                if self.on_change:
                    self.on_change(self.get_text())
            else:
                self._set_text_internal("")
        else:
            self._set_text_internal(self._history[self._history_index] or "", "start" if direction == -1 else "end")

    def _exit_history_browsing(self) -> None:
        self._history_index = -1
        self._history_draft = None

    def _set_text_internal(self, text: str, cursor_placement: str = "end") -> None:
        """Internal set_text that doesn't reset history state - used by _navigate_history."""
        lines = text.split("\n")
        self._state["lines"] = lines if lines else [""]
        self._state["cursorLine"] = 0 if cursor_placement == "start" else len(self._state["lines"]) - 1
        self._set_cursor_col(0 if cursor_placement == "start" else len(self._state["lines"][self._state["cursorLine"]]))
        # Reset scroll - render() will adjust to show cursor
        self._scroll_offset = 0

        if self.on_change:
            self.on_change(self.get_text())

    def invalidate(self) -> None:
        # No cached state to invalidate currently
        pass

    def render(self, width: int) -> list[str]:
        max_padding = max(0, (width - 1) // 2)
        padding_x = min(self._padding_x, max_padding)
        content_width = max(1, width - padding_x * 2)

        # Layout width: with padding the cursor can overflow into it,
        # without padding we reserve 1 column for the cursor.
        layout_width = max(1, content_width - (0 if padding_x else 1))

        # Store for cursor navigation (must match wrapping width)
        self._last_width = layout_width

        horizontal = self.border_color("─")

        # Layout the text
        layout_lines = self._layout_text(layout_width)

        # Calculate max visible lines: 30% of terminal height, minimum 5 lines
        terminal_rows = self._tui.terminal.rows
        max_visible_lines = max(5, math.floor(terminal_rows * 0.3))

        # Find the cursor line index in layout_lines
        cursor_line_index = next((i for i, line in enumerate(layout_lines) if line["hasCursor"]), -1)
        if cursor_line_index == -1:
            cursor_line_index = 0

        # Adjust scroll offset to keep cursor visible
        if cursor_line_index < self._scroll_offset:
            self._scroll_offset = cursor_line_index
        elif cursor_line_index >= self._scroll_offset + max_visible_lines:
            self._scroll_offset = cursor_line_index - max_visible_lines + 1

        # Clamp scroll offset to valid range
        max_scroll_offset = max(0, len(layout_lines) - max_visible_lines)
        self._scroll_offset = max(0, min(self._scroll_offset, max_scroll_offset))

        # Get visible lines slice
        visible_lines = layout_lines[self._scroll_offset : self._scroll_offset + max_visible_lines]
        self._rendered_visible_line_count = len(visible_lines)

        result: list[str] = []
        left_padding = " " * padding_x
        right_padding = left_padding

        # Render top border (with scroll indicator if scrolled down)
        if self._scroll_offset > 0:
            border = _create_scroll_border("↑", self._scroll_offset, width)
            result.append(self.border_color(border))
        else:
            result.append(horizontal * width)

        # Render each visible layout line
        # Emit hardware cursor marker when focused so TUI can position the
        # hardware cursor for IME candidate-window placement even while
        # autocomplete (e.g. slash-command menu) is visible.
        emit_cursor_marker = self.focused

        for layout_line in visible_lines:
            display_text = layout_line["text"]
            line_visible_width = visible_width(layout_line["text"])
            cursor_in_padding = False

            # Add cursor if this line has it
            if layout_line["hasCursor"] and layout_line.get("cursorPos") is not None:
                before = display_text[: layout_line["cursorPos"]]
                after = display_text[layout_line["cursorPos"] :]

                # Hardware cursor marker (zero-width, emitted before fake cursor for IME positioning)
                marker = CURSOR_MARKER if emit_cursor_marker else ""

                if after:
                    # Cursor is on a character (grapheme) - replace it with highlighted version
                    # Get the first grapheme from 'after'
                    after_graphemes = self._segment(after, "grapheme")
                    first_grapheme = after_graphemes[0]["segment"] if after_graphemes else ""
                    rest_after = after[len(first_grapheme) :]
                    cursor = f"\x1b[7m{first_grapheme}\x1b[0m"
                    display_text = before + marker + cursor + rest_after
                    # line_visible_width stays the same - we're replacing, not adding
                else:
                    # Cursor is at the end - add highlighted space
                    cursor = "\x1b[7m \x1b[0m"
                    display_text = before + marker + cursor
                    line_visible_width = line_visible_width + 1
                    # If cursor overflows content width into the padding, flag it
                    if line_visible_width > content_width and padding_x > 0:
                        cursor_in_padding = True

            # Calculate padding based on actual visible width
            padding = " " * max(0, content_width - line_visible_width)
            line_right_padding = right_padding[1:] if cursor_in_padding else right_padding

            # Render the line (no side borders, just horizontal lines above and below)
            result.append(f"{left_padding}{display_text}{padding}{line_right_padding}")

        # Render bottom border (with scroll indicator if more content below)
        lines_below = len(layout_lines) - (self._scroll_offset + len(visible_lines))
        if lines_below > 0:
            border = _create_scroll_border("↓", lines_below, width)
            result.append(self.border_color(border))
        else:
            result.append(horizontal * width)

        # Add autocomplete list if active
        self._rendered_autocomplete_height = 0
        if self._autocomplete_state and self._autocomplete_list is not None:
            autocomplete_result = self._autocomplete_list.render(content_width)
            self._rendered_autocomplete_height = len(autocomplete_result)
            for line in autocomplete_result:
                line_width = visible_width(line)
                line_padding = " " * max(0, content_width - line_width)
                result.append(f"{left_padding}{line}{line_padding}{right_padding}")

        return result

    async def handle_mouse(self, event: TuiMouseEvent) -> TuiMouseEventResult | None:
        autocomplete_start_row = self._rendered_visible_line_count + 2
        if (
            self._autocomplete_state
            and self._autocomplete_list is not None
            and autocomplete_start_row <= event.y < autocomplete_start_row + self._rendered_autocomplete_height
        ):
            max_padding = max(0, (event.width - 1) // 2)
            padding_x = min(self._padding_x, max_padding)
            content_width = max(1, event.width - padding_x * 2)
            result = await self._autocomplete_list.handle_mouse(
                replace(
                    event,
                    x=event.x - padding_x,
                    y=event.y - autocomplete_start_row,
                    width=content_width,
                    height=self._rendered_autocomplete_height,
                )
            )
            return replace(result, focus=True) if result is not None else None

        # Leave press/drag/release unhandled so the renderer's screen-level text
        # selection can run over the editor rows (drag to select, release to copy).
        # The renderer synthesizes a click when press and release land on the same
        # cell without movement, which is the gesture that positions the cursor.
        if event.type != "click" or event.button != "left":
            return None
        if event.y <= 0 or event.y > self._rendered_visible_line_count:
            return TuiMouseEventResult(handled=True, focus=True)

        visual_lines = self._build_visual_line_map(self._last_width)
        visual_line_index = self._scroll_offset + event.y - 1
        visual_line = visual_lines[visual_line_index] if visual_line_index < len(visual_lines) else None
        if visual_line is None:
            return TuiMouseEventResult(handled=True, focus=True)
        lines = self._state["lines"]
        logical_line = lines[visual_line["logicalLine"]] if visual_line["logicalLine"] < len(lines) else ""
        chunk_end = visual_line["startCol"] + visual_line["length"]
        chunk = logical_line[visual_line["startCol"] : chunk_end]
        max_padding = max(0, (event.width - 1) // 2)
        padding_x = min(self._padding_x, max_padding)
        target_column = max(0, event.x - padding_x)
        visible_column = 0
        target_index = len(chunk)
        last_grapheme_index = 0
        for grapheme in self._segment(chunk, "grapheme"):
            next_column = visible_column + visible_width(grapheme["segment"])
            last_grapheme_index = grapheme["index"]
            if target_column < next_column:
                target_index = grapheme["index"]
                break
            visible_column = next_column
        is_last_segment = (
            visual_line_index == len(visual_lines) - 1
            or visual_lines[visual_line_index + 1]["logicalLine"] != visual_line["logicalLine"]
        )
        if not is_last_segment and target_index == len(chunk) and len(chunk) > 0:
            target_index = last_grapheme_index

        self._state["cursorLine"] = visual_line["logicalLine"]
        self._set_cursor_col(visual_line["startCol"] + target_index)
        self._last_action = None
        self._exit_history_browsing()
        if self._autocomplete_state:
            self._update_autocomplete()
        return TuiMouseEventResult(handled=True, focus=True)

    async def handle_input(self, data: str) -> None:  # noqa: C901
        kb = get_keybindings()

        # Handle character jump mode (awaiting next character to jump to)
        if self._jump_mode is not None:
            # Cancel if the hotkey is pressed again
            if kb.matches(data, "tui.editor.jumpForward") or kb.matches(data, "tui.editor.jumpBackward"):
                self._jump_mode = None
                return

            printable = decode_printable_key(data)
            if printable is None and data and ord(data[0]) >= 32:
                printable = data
            if printable is not None:
                # Printable character - perform the jump
                direction = self._jump_mode
                self._jump_mode = None
                self._jump_to_char(printable, direction)
                return

            # Control character - cancel and fall through to normal handling
            self._jump_mode = None

        # Handle bracketed paste mode
        if "\x1b[200~" in data:
            self._is_in_paste = True
            self._paste_buffer = ""
            data = data.replace("\x1b[200~", "", 1)

        if self._is_in_paste:
            self._paste_buffer += data
            end_index = self._paste_buffer.find("\x1b[201~")
            if end_index != -1:
                paste_content = self._paste_buffer[:end_index]
                if paste_content:
                    self._handle_paste(paste_content)
                self._is_in_paste = False
                remaining = self._paste_buffer[end_index + 6 :]
                self._paste_buffer = ""
                if remaining:
                    await self.handle_input(remaining)
                return
            return

        # Ctrl+C - let parent handle (exit/clear)
        if kb.matches(data, "tui.input.copy"):
            return

        # Undo
        if kb.matches(data, "tui.editor.undo"):
            self._undo()
            return

        # Handle autocomplete mode
        if self._autocomplete_state and self._autocomplete_list is not None:
            if kb.matches(data, "tui.select.cancel"):
                self._cancel_autocomplete()
                return

            if kb.matches(data, "tui.select.up") or kb.matches(data, "tui.select.down"):
                await self._autocomplete_list.handle_input(data)
                return

            if kb.matches(data, "tui.input.tab"):
                selected = self._autocomplete_list.get_selected_item()
                if selected is not None and self._autocomplete_provider is not None:
                    self._push_undo_snapshot()
                    self._last_action = None
                    result = self._autocomplete_provider.apply_completion(
                        self._state["lines"],
                        self._state["cursorLine"],
                        self._state["cursorCol"],
                        selected,
                        self._autocomplete_prefix,
                    )
                    self._state["lines"] = result["lines"]
                    self._state["cursorLine"] = result["cursorLine"]
                    self._set_cursor_col(result["cursorCol"])
                    self._cancel_autocomplete()
                    if self.on_change:
                        self.on_change(self.get_text())
                return

            if kb.matches(data, "tui.select.confirm"):
                selected = self._autocomplete_list.get_selected_item()
                if selected is not None and self._autocomplete_provider is not None:
                    self._push_undo_snapshot()
                    self._last_action = None
                    result = self._autocomplete_provider.apply_completion(
                        self._state["lines"],
                        self._state["cursorLine"],
                        self._state["cursorCol"],
                        selected,
                        self._autocomplete_prefix,
                    )
                    self._state["lines"] = result["lines"]
                    self._state["cursorLine"] = result["cursorLine"]
                    self._set_cursor_col(result["cursorCol"])

                    if self._autocomplete_prefix.startswith("/"):
                        self._cancel_autocomplete()
                        # Fall through to submit
                    else:
                        self._cancel_autocomplete()
                        if self.on_change:
                            self.on_change(self.get_text())
                        return

        # Tab - trigger completion
        if kb.matches(data, "tui.input.tab") and not self._autocomplete_state:
            self._handle_tab_completion()
            return

        # Deletion actions
        if kb.matches(data, "tui.editor.deleteToLineEnd"):
            self._delete_to_end_of_line()
            return
        if kb.matches(data, "tui.editor.deleteToLineStart"):
            self._delete_to_start_of_line()
            return
        if kb.matches(data, "tui.editor.deleteWordBackward"):
            self._delete_word_backwards()
            return
        if kb.matches(data, "tui.editor.deleteWordForward"):
            self._delete_word_forward()
            return
        if kb.matches(data, "tui.editor.deleteCharBackward") or matches_key(data, "shift+backspace"):
            self._handle_backspace()
            return
        if kb.matches(data, "tui.editor.deleteCharForward") or matches_key(data, "shift+delete"):
            self._handle_forward_delete()
            return

        # Kill ring actions
        if kb.matches(data, "tui.editor.yank"):
            self._yank()
            return
        if kb.matches(data, "tui.editor.yankPop"):
            self._yank_pop()
            return

        # Dedicated history actions always browse entries instead of moving the cursor.
        if kb.matches(data, "tui.editor.historyPrevious"):
            self._cancel_autocomplete()
            self._navigate_history(-1)
            return
        if kb.matches(data, "tui.editor.historyNext"):
            self._cancel_autocomplete()
            self._navigate_history(1)
            return

        # Cursor movement actions
        if kb.matches(data, "tui.editor.cursorLineStart"):
            self._move_to_line_start()
            return
        if kb.matches(data, "tui.editor.cursorLineEnd"):
            self._move_to_line_end()
            return
        if kb.matches(data, "tui.editor.cursorWordLeft"):
            self._move_word_backwards()
            return
        if kb.matches(data, "tui.editor.cursorWordRight"):
            self._move_word_forwards()
            return

        # New line
        if (
            kb.matches(data, "tui.input.newLine")
            or (data and data[0] == "\n" and len(data) > 1)
            or data == "\x1b\r"
            or data == "\x1b[13;2~"
            or (len(data) > 1 and "\x1b" in data and "\r" in data)
            or data == "\n"
        ):
            if self._should_submit_on_backslash_enter(data, kb):
                self._handle_backspace()
                self._submit_value()
                return
            self._add_new_line()
            return

        # Submit (Enter)
        if kb.matches(data, "tui.input.submit"):
            if self.disable_submit:
                return

            # Workaround for terminals without Shift+Enter support:
            # If char before cursor is \, delete it and insert newline instead of submitting.
            current_line = self._current_line()
            if (
                self._state["cursorCol"] > 0
                and current_line[self._state["cursorCol"] - 1 : self._state["cursorCol"]] == "\\"
            ):
                self._handle_backspace()
                self._add_new_line()
                return

            self._submit_value()
            return

        # Arrow key navigation (with history support)
        if kb.matches(data, "tui.editor.cursorUp"):
            if self._is_on_first_visual_line() and (
                self._is_editor_empty() or self._history_index > -1 or self._state["cursorCol"] == 0
            ):
                self._navigate_history(-1)
            elif self._is_on_first_visual_line():
                # Already at top - jump to start of line
                self._move_to_line_start()
            else:
                self._move_cursor(-1, 0)
            return
        if kb.matches(data, "tui.editor.cursorDown"):
            if self._history_index > -1 and self._is_on_last_visual_line():
                self._navigate_history(1)
            elif self._is_on_last_visual_line():
                # Already at bottom - jump to end of line
                self._move_to_line_end()
            else:
                self._move_cursor(1, 0)
            return
        if kb.matches(data, "tui.editor.cursorRight"):
            self._move_cursor(0, 1)
            return
        if kb.matches(data, "tui.editor.cursorLeft"):
            self._move_cursor(0, -1)
            return

        # Page up/down - scroll by page and move cursor
        if kb.matches(data, "tui.editor.pageUp"):
            self._page_scroll(-1)
            return
        if kb.matches(data, "tui.editor.pageDown"):
            self._page_scroll(1)
            return

        # Character jump mode triggers
        if kb.matches(data, "tui.editor.jumpForward"):
            self._jump_mode = "forward"
            return
        if kb.matches(data, "tui.editor.jumpBackward"):
            self._jump_mode = "backward"
            return

        # Shift+Space - insert regular space
        if matches_key(data, "shift+space"):
            self._insert_character(" ")
            return

        printable = decode_printable_key(data)
        if printable is not None:
            self._insert_character(printable)
            return

        # Regular characters
        if data and ord(data[0]) >= 32:
            self._insert_character(data)

    def _current_line(self) -> str:
        lines = self._state["lines"]
        cursor_line = self._state["cursorLine"]
        return lines[cursor_line] if cursor_line < len(lines) else ""

    def _layout_text(self, content_width: int) -> list[dict]:
        # pi lays the text out on every frame. Here the frame rate is set by
        # the rest of the UI (a spinner, a streaming answer) while the editor
        # sits idle, so the layout is keyed on everything it reads: the lines
        # (a tuple of the same string objects compares by identity first),
        # the cursor and the width. The state is mutated in place, hence the
        # comparison rather than a version.
        key = (tuple(self._state["lines"]), self._state["cursorLine"], self._state["cursorCol"], content_width)
        cached = self._layout_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        layout_lines = self._compute_layout(content_width)
        self._layout_cache = (key, layout_lines)
        return layout_lines

    def _compute_layout(self, content_width: int) -> list[dict]:
        layout_lines: list[dict] = []

        if not self._state["lines"] or (len(self._state["lines"]) == 1 and self._state["lines"][0] == ""):
            # Empty editor
            layout_lines.append({"text": "", "hasCursor": True, "cursorPos": 0})
            return layout_lines

        # Process each logical line
        for i, line in enumerate(self._state["lines"]):
            is_current_line = i == self._state["cursorLine"]
            line_visible_width = visible_width(line)

            if line_visible_width <= content_width:
                # Line fits in one layout line
                if is_current_line:
                    layout_lines.append({"text": line, "hasCursor": True, "cursorPos": self._state["cursorCol"]})
                else:
                    layout_lines.append({"text": line, "hasCursor": False})
            else:
                # Line needs wrapping - use word-aware wrapping
                chunks = word_wrap_line(line, content_width, self._segment(line, "grapheme"))

                for chunk_index, chunk in enumerate(chunks):
                    cursor_pos = self._state["cursorCol"]
                    is_last_chunk = chunk_index == len(chunks) - 1

                    # Determine if cursor is in this chunk
                    # For word-wrapped chunks, we need to handle the case where
                    # cursor might be in trimmed whitespace at end of chunk
                    has_cursor_in_chunk = False
                    adjusted_cursor_pos = 0

                    if is_current_line:
                        if is_last_chunk:
                            # Last chunk: cursor belongs here if >= startIndex
                            has_cursor_in_chunk = cursor_pos >= chunk["startIndex"]
                            adjusted_cursor_pos = cursor_pos - chunk["startIndex"]
                        else:
                            # Non-last chunk: cursor belongs here if in range [startIndex, endIndex)
                            # But we need to handle the visual position in the trimmed text
                            has_cursor_in_chunk = chunk["startIndex"] <= cursor_pos < chunk["endIndex"]
                            if has_cursor_in_chunk:
                                adjusted_cursor_pos = cursor_pos - chunk["startIndex"]
                                # Clamp to text length (in case cursor was in trimmed whitespace)
                                adjusted_cursor_pos = min(adjusted_cursor_pos, len(chunk["text"]))

                    if has_cursor_in_chunk:
                        layout_lines.append(
                            {"text": chunk["text"], "hasCursor": True, "cursorPos": adjusted_cursor_pos}
                        )
                    else:
                        layout_lines.append({"text": chunk["text"], "hasCursor": False})

        return layout_lines

    def get_text(self) -> str:
        return "\n".join(self._state["lines"])

    def _expand_paste_markers(self, text: str) -> str:
        result = text
        for paste_id, paste_content in self._pastes.items():
            marker_regex = re.compile(r"\[paste #" + str(paste_id) + r"( (\+\d+ lines|\d+ chars))?\]")
            result = marker_regex.sub(lambda m, content=paste_content: content, result)
        return result

    def get_expanded_text(self) -> str:
        """Get text with paste markers expanded to their actual content.

        Use this when you need the full content (e.g., for external editor).
        """
        return self._expand_paste_markers("\n".join(self._state["lines"]))

    def get_lines(self) -> list[str]:
        return [*self._state["lines"]]

    def get_cursor(self) -> dict:
        return {"line": self._state["cursorLine"], "col": self._state["cursorCol"]}

    def set_text(self, text: str) -> None:
        self._cancel_autocomplete()
        self._last_action = None
        self._exit_history_browsing()
        normalized = self._normalize_text(text)
        # Push undo snapshot if content differs (makes programmatic changes undoable)
        if self.get_text() != normalized:
            self._push_undo_snapshot()
        self._pastes.clear()
        self._paste_counter = 0
        self._set_text_internal(normalized)

    def insert_text_at_cursor(self, text: str) -> None:
        """Insert text at the current cursor position.

        Used for programmatic insertion (e.g., clipboard image markers).
        This is atomic for undo - single undo restores entire pre-insert state.
        """
        if not text:
            return
        self._cancel_autocomplete()
        self._push_undo_snapshot()
        self._last_action = None
        self._exit_history_browsing()
        self._insert_text_at_cursor_internal(text)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for editor storage.

        Normalizes line endings (\\r\\n and \\r -> \\n) and expands tabs to 4
        spaces.
        """
        return text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")

    def _insert_text_at_cursor_internal(self, text: str) -> None:
        """Internal text insertion at cursor. Handles single and multi-line text.

        Does not push undo snapshots or trigger autocomplete - caller is
        responsible. Normalizes line endings and calls on_change once at the
        end.
        """
        if not text:
            return

        # Normalize line endings and tabs
        normalized = self._normalize_text(text)
        inserted_lines = normalized.split("\n")

        current_line = self._current_line()
        before_cursor = current_line[: self._state["cursorCol"]]
        after_cursor = current_line[self._state["cursorCol"] :]

        if len(inserted_lines) == 1:
            # Single line - insert at cursor position
            self._state["lines"][self._state["cursorLine"]] = before_cursor + normalized + after_cursor
            self._set_cursor_col(self._state["cursorCol"] + len(normalized))
        else:
            # Multi-line insertion
            self._state["lines"] = [
                # All lines before current line
                *self._state["lines"][: self._state["cursorLine"]],
                # The first inserted line merged with text before cursor
                before_cursor + inserted_lines[0],
                # All middle inserted lines
                *inserted_lines[1:-1],
                # The last inserted line with text after cursor
                inserted_lines[-1] + after_cursor,
                # All lines after current line
                *self._state["lines"][self._state["cursorLine"] + 1 :],
            ]

            self._state["cursorLine"] += len(inserted_lines) - 1
            self._set_cursor_col(len(inserted_lines[-1]))

        if self.on_change:
            self.on_change(self.get_text())

    def _insert_character(self, char: str, skip_undo_coalescing: bool = False) -> None:
        self._exit_history_browsing()

        # Undo coalescing (fish-style):
        # - Consecutive word chars coalesce into one undo unit
        # - Space captures state before itself (so undo removes space+following word together)
        # - Each space is separately undoable
        # Skip coalescing when called from atomic operations (e.g., _handle_paste)
        if not skip_undo_coalescing:
            if is_whitespace_char(char) or self._last_action != "type-word":
                self._push_undo_snapshot()
            self._last_action = "type-word"

        line = self._current_line()

        before = line[: self._state["cursorCol"]]
        after = line[self._state["cursorCol"] :]

        self._state["lines"][self._state["cursorLine"]] = before + char + after
        self._set_cursor_col(self._state["cursorCol"] + len(char))

        if self.on_change:
            self.on_change(self.get_text())

        # Check if we should trigger or update autocomplete
        if not self._autocomplete_state:
            # Auto-trigger for "/" at the start of a line (slash commands)
            if char == "/" and self._is_at_start_of_message():
                self._try_trigger_autocomplete()
            # Auto-trigger for symbol-based completion like @, #, or provider triggers at token boundaries
            elif char in self._autocomplete_trigger_characters:
                current_line = self._current_line()
                text_before_cursor = current_line[: self._state["cursorCol"]]
                char_before_symbol = (
                    text_before_cursor[len(text_before_cursor) - 2] if len(text_before_cursor) >= 2 else None
                )
                if len(text_before_cursor) == 1 or char_before_symbol == " " or char_before_symbol == "\t":
                    self._try_trigger_autocomplete()
            # Also auto-trigger when typing letters in a slash command or symbol completion context
            elif re.search(r"[a-zA-Z0-9.\-_]", char):
                current_line = self._current_line()
                text_before_cursor = current_line[: self._state["cursorCol"]]
                # Check if we're in a slash command (with or without space for
                # arguments) or in a symbol-based completion context like @, #,
                # or provider triggers
                if self._is_in_slash_command_context(text_before_cursor) or self._autocomplete_trigger_pattern.search(
                    text_before_cursor
                ):
                    self._try_trigger_autocomplete()
        else:
            self._update_autocomplete()

    def _handle_paste(self, pasted_text: str) -> None:
        self._cancel_autocomplete()
        self._exit_history_browsing()
        self._last_action = None

        self._push_undo_snapshot()

        # Some terminals (e.g. tmux popups with extended-keys-format=csi-u) re-encode
        # control bytes inside bracketed paste as CSI-u Ctrl+<letter> sequences
        # (ESC [ <codepoint> ; 5 u). Decode those back to their literal byte so the
        # per-char filter below preserves newlines instead of stripping ESC and
        # leaking the printable tail (e.g. "[106;5u") into the editor.
        def _decode_csi_u(match: re.Match) -> str:
            cp = int(match.group(1))
            if 97 <= cp <= 122:
                return chr(cp - 96)
            if 65 <= cp <= 90:
                return chr(cp - 64)
            return match.group(0)

        decoded_text = re.sub("\x1b\\[(\\d+);5u", _decode_csi_u, pasted_text)

        # Clean the pasted text: normalize line endings, expand tabs
        clean_text = self._normalize_text(decoded_text)

        # Filter out non-printable characters except newlines
        filtered_text = "".join(char for char in clean_text if char == "\n" or ord(char) >= 32)

        # If pasting a file path (starts with /, ~, or .) and the character before
        # the cursor is a word character, prepend a space for better readability
        if re.match(r"^[/~.]", filtered_text):
            current_line = self._current_line()
            char_before_cursor = current_line[self._state["cursorCol"] - 1] if self._state["cursorCol"] > 0 else ""
            if char_before_cursor and re.search(r"\w", char_before_cursor, re.ASCII):
                filtered_text = f" {filtered_text}"

        # Split into lines to check for large paste
        pasted_lines = filtered_text.split("\n")

        # Check if this is a large paste (> 10 lines or > 1000 characters)
        total_chars = len(filtered_text)
        if len(pasted_lines) > 10 or total_chars > 1000:
            # Store the paste and insert a marker
            self._paste_counter += 1
            paste_id = self._paste_counter
            self._pastes[paste_id] = filtered_text

            # Insert marker like "[paste #1 +123 lines]" or "[paste #1 1234 chars]"
            marker = (
                f"[paste #{paste_id} +{len(pasted_lines)} lines]"
                if len(pasted_lines) > 10
                else f"[paste #{paste_id} {total_chars} chars]"
            )
            self._insert_text_at_cursor_internal(marker)
            return

        # Single or multi-line paste - insert atomically (do not trigger
        # autocomplete during paste)
        self._insert_text_at_cursor_internal(filtered_text)

    def _add_new_line(self) -> None:
        self._cancel_autocomplete()
        self._exit_history_browsing()
        self._last_action = None

        self._push_undo_snapshot()

        current_line = self._current_line()

        before = current_line[: self._state["cursorCol"]]
        after = current_line[self._state["cursorCol"] :]

        # Split current line
        self._state["lines"][self._state["cursorLine"]] = before
        self._state["lines"].insert(self._state["cursorLine"] + 1, after)

        # Move cursor to start of new line
        self._state["cursorLine"] += 1
        self._set_cursor_col(0)

        if self.on_change:
            self.on_change(self.get_text())

    def _should_submit_on_backslash_enter(self, data: str, kb) -> bool:
        if self.disable_submit:
            return False
        if not matches_key(data, "enter"):
            return False
        submit_keys = kb.get_keys("tui.input.submit")
        has_shift_enter = "shift+enter" in submit_keys or "shift+return" in submit_keys
        if not has_shift_enter:
            return False

        current_line = self._current_line()
        return self._state["cursorCol"] > 0 and current_line[self._state["cursorCol"] - 1] == "\\"

    def _submit_value(self) -> None:
        self._cancel_autocomplete()
        result = self._expand_paste_markers("\n".join(self._state["lines"])).strip()

        self._state = {"lines": [""], "cursorLine": 0, "cursorCol": 0}
        self._pastes.clear()
        self._paste_counter = 0
        self._exit_history_browsing()
        self._scroll_offset = 0
        self._undo_stack.clear()
        self._last_action = None

        if self.on_change:
            self.on_change("")
        if self.on_submit:
            self.on_submit(result)

    def _handle_backspace(self) -> None:
        self._exit_history_browsing()
        self._last_action = None

        if self._state["cursorCol"] > 0:
            self._push_undo_snapshot()

            # Delete grapheme before cursor (handles emojis, combining characters, etc.)
            line = self._current_line()
            before_cursor = line[: self._state["cursorCol"]]

            # Find the last grapheme in the text before cursor
            graphemes = self._segment(before_cursor, "grapheme")
            last_grapheme = graphemes[-1] if graphemes else None
            grapheme_length = len(last_grapheme["segment"]) if last_grapheme else 1
            is_pasted_segmented = PASTE_MARKER_SINGLE.match(last_grapheme["segment"]) if last_grapheme else None

            if is_pasted_segmented:
                # This contains the id part e.g 4 from [paste #4 +123 lines]
                target_id = int(is_pasted_segmented.group(1))
                self._pastes.pop(target_id, None)
                self._paste_counter -= 1

                # Shift registry entries down in ascending id order, independent
                # of marker order in the text ([paste #3] becomes [paste #2] when
                # [paste #1] is removed).
                higher_ids = sorted(paste_id for paste_id in self._pastes if paste_id > target_id)
                for paste_id in higher_ids:
                    self._pastes[paste_id - 1] = self._pastes.pop(paste_id)

                # Renumber markers with ids greater than the removed one.
                def _renumber(match: re.Match) -> str:
                    marker_id = int(match.group(1))
                    if marker_id <= target_id:
                        return match.group(0)
                    return f"[paste #{marker_id - 1}{match.group(2) or ''}]"

                self._state["lines"] = [PASTE_MARKER_REGEX.sub(_renumber, line) for line in self._state["lines"]]

            line = self._current_line()

            before = line[: self._state["cursorCol"] - grapheme_length]
            after = line[self._state["cursorCol"] :]

            self._state["lines"][self._state["cursorLine"]] = before + after
            self._set_cursor_col(self._state["cursorCol"] - grapheme_length)
        elif self._state["cursorLine"] > 0:
            self._push_undo_snapshot()

            # Merge with previous line
            current_line = self._current_line()
            previous_line = self._state["lines"][self._state["cursorLine"] - 1]

            self._state["lines"][self._state["cursorLine"] - 1] = previous_line + current_line
            del self._state["lines"][self._state["cursorLine"]]

            self._state["cursorLine"] -= 1
            self._set_cursor_col(len(previous_line))

        if self.on_change:
            self.on_change(self.get_text())

        # Update or re-trigger autocomplete after backspace
        if self._autocomplete_state:
            self._update_autocomplete()
        else:
            # If autocomplete was cancelled (no matches), re-trigger if we're in a
            # completable context: a slash command, or a symbol-based completion
            # context like @, #, or provider triggers
            current_line = self._current_line()
            text_before_cursor = current_line[: self._state["cursorCol"]]
            if self._is_in_slash_command_context(text_before_cursor) or self._autocomplete_trigger_pattern.search(
                text_before_cursor
            ):
                self._try_trigger_autocomplete()

    def _set_cursor_col(self, col: int) -> None:
        """Set cursor column and clear the sticky column state.

        Use this for all non-vertical cursor movements to reset sticky column
        behavior.
        """
        self._state["cursorCol"] = col
        self._preferred_visual_col = None
        self._snapped_from_cursor_col = None

    def _move_to_visual_line(self, visual_lines: list[dict], current_visual_line: int, target_visual_line: int) -> None:
        """Move cursor to a target visual line, applying sticky column logic.

        Shared by _move_cursor() and _page_scroll().
        """
        current_vl = visual_lines[current_visual_line] if 0 <= current_visual_line < len(visual_lines) else None
        target_vl = visual_lines[target_visual_line] if 0 <= target_visual_line < len(visual_lines) else None
        if not (current_vl and target_vl):
            return

        # When the cursor was snapped to a segment start, resolve the pre-snap
        # position against the VL it belongs to. This gives the correct visual
        # column even after a resize reshuffles VLs.
        if self._snapped_from_cursor_col is not None:
            vl_index = self._find_visual_line_at(visual_lines, current_vl["logicalLine"], self._snapped_from_cursor_col)
            current_visual_col = self._snapped_from_cursor_col - visual_lines[vl_index]["startCol"]
        else:
            current_visual_col = self._state["cursorCol"] - current_vl["startCol"]

        # For non-last segments, clamp to length-1 to stay within the segment
        is_last_source_segment = (
            current_visual_line == len(visual_lines) - 1
            or visual_lines[current_visual_line + 1]["logicalLine"] != current_vl["logicalLine"]
        )
        source_max_visual_col = current_vl["length"] if is_last_source_segment else max(0, current_vl["length"] - 1)

        is_last_target_segment = (
            target_visual_line == len(visual_lines) - 1
            or visual_lines[target_visual_line + 1]["logicalLine"] != target_vl["logicalLine"]
        )
        target_max_visual_col = target_vl["length"] if is_last_target_segment else max(0, target_vl["length"] - 1)

        move_to_visual_col = self._compute_vertical_move_column(
            current_visual_col, source_max_visual_col, target_max_visual_col
        )

        # Set cursor position
        self._state["cursorLine"] = target_vl["logicalLine"]
        target_col = target_vl["startCol"] + move_to_visual_col
        logical_line = self._state["lines"][target_vl["logicalLine"]]
        self._state["cursorCol"] = min(target_col, len(logical_line))

        # Snap cursor to atomic segment boundary (e.g. paste markers)
        # so the cursor never lands in the middle of a multi-grapheme unit.
        # Single-grapheme segments don't need snapping.
        segments = self._segment(logical_line, "grapheme")
        for seg in segments:
            if seg["index"] > self._state["cursorCol"]:
                break
            if len(seg["segment"]) <= 1:
                continue
            if self._state["cursorCol"] < seg["index"] + len(seg["segment"]):
                is_continuation = seg["index"] < target_vl["startCol"]
                is_moving_down = target_visual_line > current_visual_line

                if is_continuation and is_moving_down:
                    # The segment started on a previous visual line, and we
                    # already visited it on the way down. Skip all remaining
                    # continuation VLs and land on the first VL past it.
                    seg_end = seg["index"] + len(seg["segment"])
                    nxt = target_visual_line + 1
                    while (
                        nxt < len(visual_lines)
                        and visual_lines[nxt]["logicalLine"] == target_vl["logicalLine"]
                        and visual_lines[nxt]["startCol"] < seg_end
                    ):
                        nxt += 1
                    if nxt < len(visual_lines):
                        self._move_to_visual_line(visual_lines, current_visual_line, nxt)
                        return

                # Snap to the start of the segment so it gets highlighted.
                # Store the pre-snap position so the next vertical move can
                # resolve it to the correct visual column.
                self._snapped_from_cursor_col = self._state["cursorCol"]
                self._state["cursorCol"] = seg["index"]
                return

        # No snap occurred – we moved out of the atomic segment.
        self._snapped_from_cursor_col = None

    def _compute_vertical_move_column(
        self, current_visual_col: int, source_max_visual_col: int, target_max_visual_col: int
    ) -> int:
        """Compute the target visual column for vertical cursor movement.

        Implements the sticky column decision table:

        | P | S | T | U | Scenario                                             | Set Preferred | Move To     |
        |---|---|---|---| ---------------------------------------------------- |---------------|-------------|
        | 0 | * | 0 | - | Start nav, target fits                               | None          | current     |
        | 0 | * | 1 | - | Start nav, target shorter                            | current       | target end  |
        | 1 | 0 | 0 | 0 | Clamped, target fits preferred                       | None          | preferred   |
        | 1 | 0 | 0 | 1 | Clamped, target longer but still can't fit preferred | keep          | target end  |
        | 1 | 0 | 1 | - | Clamped, target even shorter                         | keep          | target end  |
        | 1 | 1 | 0 | - | Rewrapped, target fits current                       | None          | current     |
        | 1 | 1 | 1 | - | Rewrapped, target shorter than current               | current       | target end  |

        Where:
        - P = preferred col is set
        - S = cursor in middle of source line (not clamped to end)
        - T = target line shorter than current visual col
        - U = target line shorter than preferred col
        """
        has_preferred = self._preferred_visual_col is not None  # P
        cursor_in_middle = current_visual_col < source_max_visual_col  # S
        target_too_short = target_max_visual_col < current_visual_col  # T

        if not has_preferred or cursor_in_middle:
            if target_too_short:
                # Cases 2 and 7
                self._preferred_visual_col = current_visual_col
                return target_max_visual_col

            # Cases 1 and 6
            self._preferred_visual_col = None
            return current_visual_col

        target_cant_fit_preferred = target_max_visual_col < self._preferred_visual_col  # U
        if target_too_short or target_cant_fit_preferred:
            # Cases 4 and 5
            return target_max_visual_col

        # Case 3
        result = self._preferred_visual_col
        self._preferred_visual_col = None
        return result

    def _move_to_line_start(self) -> None:
        self._last_action = None
        self._set_cursor_col(0)

    def _move_to_line_end(self) -> None:
        self._last_action = None
        current_line = self._current_line()
        self._set_cursor_col(len(current_line))

    def _delete_to_start_of_line(self) -> None:
        self._exit_history_browsing()

        current_line = self._current_line()

        if self._state["cursorCol"] > 0:
            self._push_undo_snapshot()

            # Calculate text to be deleted and save to kill ring (backward deletion = prepend)
            deleted_text = current_line[: self._state["cursorCol"]]
            self._kill_ring.push(deleted_text, prepend=True, accumulate=self._last_action == "kill")
            self._last_action = "kill"

            # Delete from start of line up to cursor
            self._state["lines"][self._state["cursorLine"]] = current_line[self._state["cursorCol"] :]
            self._set_cursor_col(0)
        elif self._state["cursorLine"] > 0:
            self._push_undo_snapshot()

            # At start of line - merge with previous line, treating newline as deleted text
            self._kill_ring.push("\n", prepend=True, accumulate=self._last_action == "kill")
            self._last_action = "kill"

            previous_line = self._state["lines"][self._state["cursorLine"] - 1]
            self._state["lines"][self._state["cursorLine"] - 1] = previous_line + current_line
            del self._state["lines"][self._state["cursorLine"]]
            self._state["cursorLine"] -= 1
            self._set_cursor_col(len(previous_line))

        if self.on_change:
            self.on_change(self.get_text())

    def _delete_to_end_of_line(self) -> None:
        self._exit_history_browsing()

        current_line = self._current_line()

        if self._state["cursorCol"] < len(current_line):
            self._push_undo_snapshot()

            # Calculate text to be deleted and save to kill ring (forward deletion = append)
            deleted_text = current_line[self._state["cursorCol"] :]
            self._kill_ring.push(deleted_text, prepend=False, accumulate=self._last_action == "kill")
            self._last_action = "kill"

            # Delete from cursor to end of line
            self._state["lines"][self._state["cursorLine"]] = current_line[: self._state["cursorCol"]]
        elif self._state["cursorLine"] < len(self._state["lines"]) - 1:
            self._push_undo_snapshot()

            # At end of line - merge with next line, treating newline as deleted text
            self._kill_ring.push("\n", prepend=False, accumulate=self._last_action == "kill")
            self._last_action = "kill"

            next_line = self._state["lines"][self._state["cursorLine"] + 1]
            self._state["lines"][self._state["cursorLine"]] = current_line + next_line
            del self._state["lines"][self._state["cursorLine"] + 1]

        if self.on_change:
            self.on_change(self.get_text())

    def _delete_word_backwards(self) -> None:
        self._exit_history_browsing()

        current_line = self._current_line()

        # If at start of line, behave like backspace at column 0 (merge with previous line)
        if self._state["cursorCol"] == 0:
            if self._state["cursorLine"] > 0:
                self._push_undo_snapshot()

                # Treat newline as deleted text (backward deletion = prepend)
                self._kill_ring.push("\n", prepend=True, accumulate=self._last_action == "kill")
                self._last_action = "kill"

                previous_line = self._state["lines"][self._state["cursorLine"] - 1]
                self._state["lines"][self._state["cursorLine"] - 1] = previous_line + current_line
                del self._state["lines"][self._state["cursorLine"]]
                self._state["cursorLine"] -= 1
                self._set_cursor_col(len(previous_line))
        else:
            self._push_undo_snapshot()

            # Save last_action before cursor movement (_move_word_backwards resets it)
            was_kill = self._last_action == "kill"

            old_cursor_col = self._state["cursorCol"]
            self._move_word_backwards()
            delete_from = self._state["cursorCol"]
            self._set_cursor_col(old_cursor_col)

            deleted_text = current_line[delete_from : self._state["cursorCol"]]
            self._kill_ring.push(deleted_text, prepend=True, accumulate=was_kill)
            self._last_action = "kill"

            self._state["lines"][self._state["cursorLine"]] = (
                current_line[:delete_from] + current_line[self._state["cursorCol"] :]
            )
            self._set_cursor_col(delete_from)

        if self.on_change:
            self.on_change(self.get_text())

    def _delete_word_forward(self) -> None:
        self._exit_history_browsing()

        current_line = self._current_line()

        # If at end of line, merge with next line (delete the newline)
        if self._state["cursorCol"] >= len(current_line):
            if self._state["cursorLine"] < len(self._state["lines"]) - 1:
                self._push_undo_snapshot()

                # Treat newline as deleted text (forward deletion = append)
                self._kill_ring.push("\n", prepend=False, accumulate=self._last_action == "kill")
                self._last_action = "kill"

                next_line = self._state["lines"][self._state["cursorLine"] + 1]
                self._state["lines"][self._state["cursorLine"]] = current_line + next_line
                del self._state["lines"][self._state["cursorLine"] + 1]
        else:
            self._push_undo_snapshot()

            # Save last_action before cursor movement (_move_word_forwards resets it)
            was_kill = self._last_action == "kill"

            old_cursor_col = self._state["cursorCol"]
            self._move_word_forwards()
            delete_to = self._state["cursorCol"]
            self._set_cursor_col(old_cursor_col)

            deleted_text = current_line[self._state["cursorCol"] : delete_to]
            self._kill_ring.push(deleted_text, prepend=False, accumulate=was_kill)
            self._last_action = "kill"

            self._state["lines"][self._state["cursorLine"]] = (
                current_line[: self._state["cursorCol"]] + current_line[delete_to:]
            )

        if self.on_change:
            self.on_change(self.get_text())

    def _handle_forward_delete(self) -> None:
        self._exit_history_browsing()
        self._last_action = None

        current_line = self._current_line()

        if self._state["cursorCol"] < len(current_line):
            self._push_undo_snapshot()

            # Delete grapheme at cursor position (handles emojis, combining characters, etc.)
            after_cursor = current_line[self._state["cursorCol"] :]

            # Find the first grapheme at cursor
            graphemes = self._segment(after_cursor, "grapheme")
            first_grapheme = graphemes[0] if graphemes else None
            grapheme_length = len(first_grapheme["segment"]) if first_grapheme else 1

            before = current_line[: self._state["cursorCol"]]
            after = current_line[self._state["cursorCol"] + grapheme_length :]
            self._state["lines"][self._state["cursorLine"]] = before + after
        elif self._state["cursorLine"] < len(self._state["lines"]) - 1:
            self._push_undo_snapshot()

            # At end of line - merge with next line
            next_line = self._state["lines"][self._state["cursorLine"] + 1]
            self._state["lines"][self._state["cursorLine"]] = current_line + next_line
            del self._state["lines"][self._state["cursorLine"] + 1]

        if self.on_change:
            self.on_change(self.get_text())

        # Update or re-trigger autocomplete after forward delete
        if self._autocomplete_state:
            self._update_autocomplete()
        else:
            # Re-trigger if we're in a completable context: a slash command, or a
            # symbol-based completion context like @, #, or provider triggers
            current_line = self._current_line()
            text_before_cursor = current_line[: self._state["cursorCol"]]
            if self._is_in_slash_command_context(text_before_cursor) or self._autocomplete_trigger_pattern.search(
                text_before_cursor
            ):
                self._try_trigger_autocomplete()

    def _build_visual_line_map(self, width: int) -> list[dict]:
        """Build a mapping from visual lines to logical positions.

        Returns a list where each element represents a visual line with:
        - logicalLine: index into state lines
        - startCol: starting column in the logical line
        - length: length of this visual line segment
        """
        visual_lines: list[dict] = []

        for i, line in enumerate(self._state["lines"]):
            line_vis_width = visible_width(line)
            if len(line) == 0:
                # Empty line still takes one visual line
                visual_lines.append({"logicalLine": i, "startCol": 0, "length": 0})
            elif line_vis_width <= width:
                visual_lines.append({"logicalLine": i, "startCol": 0, "length": len(line)})
            else:
                # Line needs wrapping - use word-aware wrapping
                chunks = word_wrap_line(line, width, self._segment(line, "grapheme"))
                for chunk in chunks:
                    visual_lines.append(
                        {
                            "logicalLine": i,
                            "startCol": chunk["startIndex"],
                            "length": chunk["endIndex"] - chunk["startIndex"],
                        }
                    )

        return visual_lines

    def _find_visual_line_at(self, visual_lines: list[dict], line: int, col: int) -> int:
        """Find the visual line index that contains the given logical position."""
        for i, vl in enumerate(visual_lines):
            if vl["logicalLine"] != line:
                continue
            offset = col - vl["startCol"]
            # Cursor is in this segment if it's within range. For the last
            # segment of a logical line, cursor can be at length (end position)
            is_last_segment_of_line = (
                i == len(visual_lines) - 1 or visual_lines[i + 1]["logicalLine"] != vl["logicalLine"]
            )
            if offset >= 0 and (offset < vl["length"] or (is_last_segment_of_line and offset == vl["length"])):
                return i
        return len(visual_lines) - 1

    def _find_current_visual_line(self, visual_lines: list[dict]) -> int:
        """Find the visual line index for the current cursor position."""
        return self._find_visual_line_at(visual_lines, self._state["cursorLine"], self._state["cursorCol"])

    def _move_cursor(self, delta_line: int, delta_col: int) -> None:
        self._last_action = None
        visual_lines = self._build_visual_line_map(self._last_width)
        current_visual_line = self._find_current_visual_line(visual_lines)

        if delta_line != 0:
            target_visual_line = current_visual_line + delta_line

            if 0 <= target_visual_line < len(visual_lines):
                self._move_to_visual_line(visual_lines, current_visual_line, target_visual_line)

        if delta_col != 0:
            current_line = self._current_line()

            if delta_col > 0:
                # Moving right - move by one grapheme (handles emojis, combining characters, etc.)
                if self._state["cursorCol"] < len(current_line):
                    after_cursor = current_line[self._state["cursorCol"] :]
                    graphemes = self._segment(after_cursor, "grapheme")
                    first_grapheme = graphemes[0] if graphemes else None
                    self._set_cursor_col(
                        self._state["cursorCol"] + (len(first_grapheme["segment"]) if first_grapheme else 1)
                    )
                elif self._state["cursorLine"] < len(self._state["lines"]) - 1:
                    # Wrap to start of next logical line
                    self._state["cursorLine"] += 1
                    self._set_cursor_col(0)
                else:
                    # At end of last line - can't move, but set preferred visual col for up/down navigation
                    current_vl = visual_lines[current_visual_line] if current_visual_line < len(visual_lines) else None
                    if current_vl:
                        self._preferred_visual_col = self._state["cursorCol"] - current_vl["startCol"]
            else:
                # Moving left - move by one grapheme (handles emojis, combining characters, etc.)
                if self._state["cursorCol"] > 0:
                    before_cursor = current_line[: self._state["cursorCol"]]
                    graphemes = self._segment(before_cursor, "grapheme")
                    last_grapheme = graphemes[-1] if graphemes else None
                    self._set_cursor_col(
                        self._state["cursorCol"] - (len(last_grapheme["segment"]) if last_grapheme else 1)
                    )
                elif self._state["cursorLine"] > 0:
                    # Wrap to end of previous logical line
                    self._state["cursorLine"] -= 1
                    prev_line = self._current_line()
                    self._set_cursor_col(len(prev_line))

        # Keep an open autocomplete picker in sync with the new cursor
        # position: cursor movement changes the text before the cursor, so a
        # picker computed for the old position is stale. Re-query so it
        # refreshes — or closes when the new position yields no suggestions —
        # mirroring _insert_character()/_handle_backspace(). Without this,
        # arrowing left from `/cmd ` back into the command name leaves the
        # argument picker showing against a `/cmd` prefix (and a Tab there
        # would concatenate the stale suggestion onto the partial command
        # name).
        if self._autocomplete_state:
            self._update_autocomplete()

    def _page_scroll(self, direction: int) -> None:
        """Scroll by a page (direction: -1 for up, 1 for down).

        Moves cursor by the page size while keeping it in bounds.
        """
        self._last_action = None
        terminal_rows = self._tui.terminal.rows
        page_size = max(5, math.floor(terminal_rows * 0.3))

        visual_lines = self._build_visual_line_map(self._last_width)
        current_visual_line = self._find_current_visual_line(visual_lines)
        target_visual_line = max(0, min(len(visual_lines) - 1, current_visual_line + direction * page_size))

        self._move_to_visual_line(visual_lines, current_visual_line, target_visual_line)

    def _move_word_backwards(self) -> None:
        self._last_action = None
        current_line = self._current_line()

        # If at start of line, move to end of previous line
        if self._state["cursorCol"] == 0:
            if self._state["cursorLine"] > 0:
                self._state["cursorLine"] -= 1
                prev_line = self._current_line()
                self._set_cursor_col(len(prev_line))
            return

        self._set_cursor_col(
            find_word_backward(
                current_line,
                self._state["cursorCol"],
                segment=lambda text: self._segment(text, "word"),
                is_atomic_segment=_is_paste_marker,
            )
        )

    def _yank(self) -> None:
        """Yank (paste) the most recent kill ring entry at cursor position."""
        if self._kill_ring.length == 0:
            return

        self._push_undo_snapshot()

        text = self._kill_ring.peek()
        self._insert_yanked_text(text)

        self._last_action = "yank"

    def _yank_pop(self) -> None:
        """Cycle through kill ring (only works immediately after yank or yank-pop).

        Replaces the last yanked text with the previous entry in the ring.
        """
        # Only works if we just yanked and have more than one entry
        if self._last_action != "yank" or self._kill_ring.length <= 1:
            return

        self._push_undo_snapshot()

        # Delete the previously yanked text (still at end of ring before rotation)
        self._delete_yanked_text()

        # Rotate the ring: move end to front
        self._kill_ring.rotate()

        # Insert the new most recent entry (now at end after rotation)
        text = self._kill_ring.peek()
        self._insert_yanked_text(text)

        self._last_action = "yank"

    def _insert_yanked_text(self, text: str) -> None:
        """Insert text at cursor position (used by yank operations)."""
        self._exit_history_browsing()
        lines = text.split("\n")

        if len(lines) == 1:
            # Single line - insert at cursor
            current_line = self._current_line()
            before = current_line[: self._state["cursorCol"]]
            after = current_line[self._state["cursorCol"] :]
            self._state["lines"][self._state["cursorLine"]] = before + text + after
            self._set_cursor_col(self._state["cursorCol"] + len(text))
        else:
            # Multi-line insert
            current_line = self._current_line()
            before = current_line[: self._state["cursorCol"]]
            after = current_line[self._state["cursorCol"] :]

            # First line merges with text before cursor
            self._state["lines"][self._state["cursorLine"]] = before + lines[0]

            # Insert middle lines
            for i in range(1, len(lines) - 1):
                self._state["lines"].insert(self._state["cursorLine"] + i, lines[i])

            # Last line merges with text after cursor
            last_line_index = self._state["cursorLine"] + len(lines) - 1
            self._state["lines"].insert(last_line_index, lines[-1] + after)

            # Update cursor position
            self._state["cursorLine"] = last_line_index
            self._set_cursor_col(len(lines[-1]))

        if self.on_change:
            self.on_change(self.get_text())

    def _delete_yanked_text(self) -> None:
        """Delete the previously yanked text (used by yank-pop).

        The yanked text is derived from the kill ring end since it hasn't been
        rotated yet.
        """
        yanked_text = self._kill_ring.peek()
        if not yanked_text:
            return

        yank_lines = yanked_text.split("\n")

        if len(yank_lines) == 1:
            # Single line - delete backward from cursor
            current_line = self._current_line()
            delete_len = len(yanked_text)
            before = current_line[: self._state["cursorCol"] - delete_len]
            after = current_line[self._state["cursorCol"] :]
            self._state["lines"][self._state["cursorLine"]] = before + after
            self._set_cursor_col(self._state["cursorCol"] - delete_len)
        else:
            # Multi-line delete - cursor is at end of last yanked line
            start_line = self._state["cursorLine"] - (len(yank_lines) - 1)
            start_col = len(self._state["lines"][start_line]) - len(yank_lines[0])

            # Get text after cursor on current line
            after_cursor = self._current_line()[self._state["cursorCol"] :]

            # Get text before yank start position
            before_yank = self._state["lines"][start_line][:start_col]

            # Remove all lines from start_line to cursorLine and replace with merged line
            self._state["lines"][start_line : start_line + len(yank_lines)] = [before_yank + after_cursor]

            # Update cursor
            self._state["cursorLine"] = start_line
            self._set_cursor_col(start_col)

        if self.on_change:
            self.on_change(self.get_text())

    def _push_undo_snapshot(self) -> None:
        self._undo_stack.push({"state": self._state, "pastes": self._pastes, "pasteCounter": self._paste_counter})

    def _undo(self) -> None:
        self._exit_history_browsing()
        snapshot = self._undo_stack.pop()
        if snapshot is None:
            return
        self._state.update(snapshot["state"])
        self._pastes = snapshot["pastes"]
        self._paste_counter = snapshot["pasteCounter"]
        self._last_action = None
        self._preferred_visual_col = None
        if self.on_change:
            self.on_change(self.get_text())

    def _jump_to_char(self, char: str, direction: str) -> None:
        """Jump to the first occurrence of a character in the specified direction.

        Multi-line search. Case-sensitive. Skips the current cursor position.
        """
        self._last_action = None
        is_forward = direction == "forward"
        lines = self._state["lines"]

        end = len(lines) if is_forward else -1
        step = 1 if is_forward else -1

        for line_idx in range(self._state["cursorLine"], end, step):
            line = lines[line_idx]
            is_current_line = line_idx == self._state["cursorLine"]

            # Current line: start after/before cursor; other lines: search full line
            if is_current_line:
                if is_forward:
                    idx = line.find(char, self._state["cursorCol"] + 1)
                else:
                    # JS lastIndexOf clamps a negative fromIndex to 0
                    search_from = max(0, self._state["cursorCol"] - 1)
                    idx = line.rfind(char, 0, search_from + 1)
            else:
                idx = line.find(char) if is_forward else line.rfind(char)

            if idx != -1:
                self._state["cursorLine"] = line_idx
                self._set_cursor_col(idx)
                return
        # No match found - cursor stays in place

    def _move_word_forwards(self) -> None:
        self._last_action = None
        current_line = self._current_line()

        # If at end of line, move to start of next line
        if self._state["cursorCol"] >= len(current_line):
            if self._state["cursorLine"] < len(self._state["lines"]) - 1:
                self._state["cursorLine"] += 1
                self._set_cursor_col(0)
            return

        self._set_cursor_col(
            find_word_forward(
                current_line,
                self._state["cursorCol"],
                segment=lambda text: self._segment(text, "word"),
                is_atomic_segment=_is_paste_marker,
            )
        )

    # Slash menu only allowed on the first line of the editor
    def _is_slash_menu_allowed(self) -> bool:
        return self._state["cursorLine"] == 0

    # Helper method to check if cursor is at start of message (for slash command detection)
    def _is_at_start_of_message(self) -> bool:
        if not self._is_slash_menu_allowed():
            return False
        current_line = self._current_line()
        before_cursor = current_line[: self._state["cursorCol"]]
        return before_cursor.strip() == "" or before_cursor.strip() == "/"

    def _is_in_slash_command_context(self, text_before_cursor: str) -> bool:
        return self._is_slash_menu_allowed() and text_before_cursor.lstrip().startswith("/")

    # Autocomplete methods
    def _get_best_autocomplete_match_index(self, items: list[dict], prefix: str) -> int:
        """Find the best autocomplete item index for the given prefix.

        Returns -1 if no match is found.

        Match priority:
        1. Exact match (prefix == item value) -> always selected
        2. Prefix match -> first item whose value starts with prefix
        3. No match -> -1 (keep default highlight)

        Matching is case-sensitive and checks item values only.
        """
        if not prefix:
            return -1

        first_prefix_index = -1

        for i, item in enumerate(items):
            value = item["value"]
            if value == prefix:
                return i  # Exact match always wins
            if first_prefix_index == -1 and value.startswith(prefix):
                first_prefix_index = i

        return first_prefix_index

    def _create_autocomplete_list(self, prefix: str, items: list[dict]) -> SelectList:
        layout = SLASH_COMMAND_SELECT_LIST_LAYOUT if prefix.startswith("/") else None
        select_list = SelectList(items, self._autocomplete_max_visible, self._theme["selectList"], layout)

        async def on_select(selected: dict) -> None:
            if self._autocomplete_provider is None:
                return
            self._push_undo_snapshot()
            self._last_action = None
            result = self._autocomplete_provider.apply_completion(
                self._state["lines"],
                self._state["cursorLine"],
                self._state["cursorCol"],
                selected,
                self._autocomplete_prefix,
            )
            self._state["lines"] = result["lines"]
            self._state["cursorLine"] = result["cursorLine"]
            self._set_cursor_col(result["cursorCol"])
            self._cancel_autocomplete()
            if self.on_change:
                self.on_change(self.get_text())

        select_list.on_select = on_select
        return select_list

    def _try_trigger_autocomplete(self, explicit_tab: bool = False) -> None:
        self._request_autocomplete(force=False, explicit_tab=explicit_tab)

    def _handle_tab_completion(self) -> None:
        if self._autocomplete_provider is None:
            return

        current_line = self._current_line()
        before_cursor = current_line[: self._state["cursorCol"]]

        if self._is_in_slash_command_context(before_cursor) and " " not in before_cursor.lstrip():
            self._handle_slash_command_completion()
        else:
            self._force_file_autocomplete(True)

    def _handle_slash_command_completion(self) -> None:
        self._request_autocomplete(force=False, explicit_tab=True)

    def _force_file_autocomplete(self, explicit_tab: bool = False) -> None:
        self._request_autocomplete(force=True, explicit_tab=explicit_tab)

    def _request_autocomplete(self, *, force: bool, explicit_tab: bool) -> None:
        if self._autocomplete_provider is None:
            return

        if force:
            should_trigger_fn = getattr(self._autocomplete_provider, "should_trigger_file_completion", None)
            should_trigger = should_trigger_fn is None or should_trigger_fn(
                self._state["lines"], self._state["cursorLine"], self._state["cursorCol"]
            )
            if not should_trigger:
                return

        self._cancel_autocomplete_request()
        self._autocomplete_start_token += 1
        start_token = self._autocomplete_start_token

        debounce_ms = self._get_autocomplete_debounce_ms(explicit_tab=explicit_tab, force=force)
        if debounce_ms > 0:

            async def _fire() -> None:
                self._autocomplete_debounce_timer = None
                self._start_autocomplete_request(start_token, force=force, explicit_tab=explicit_tab)

            self._autocomplete_debounce_timer = self._tui.input_owner.after(debounce_ms, _fire)
            return

        self._start_autocomplete_request(start_token, force=force, explicit_tab=explicit_tab)

    def _start_autocomplete_request(self, start_token: int, *, force: bool, explicit_tab: bool) -> None:
        # Runs on the UI owner (from `handle_input` or the debounce timer):
        # the snapshot is taken here, the provider is queried off the owner,
        # and the result is applied back on the owner. pi chains requests
        # through `autocompleteRequestTask` so one runs at a time; the lock
        # keeps that, and a request superseded while waiting is skipped.
        if start_token != self._autocomplete_start_token or self._autocomplete_provider is None:
            return
        provider = self._autocomplete_provider
        controller = CancelToken()
        self._autocomplete_abort = controller
        self._autocomplete_request_id += 1
        request_id = self._autocomplete_request_id
        snapshot_text = self.get_text()
        snapshot_line = self._state["cursorLine"]
        snapshot_col = self._state["cursorCol"]
        lines = list(self._state["lines"])

        async def _request_task() -> None:
            try:
                async with self._autocomplete_request_lock:
                    if controller.cancelled:
                        return
                    # Async-only, matching pi's `getSuggestions` (strictly
                    # Promise-returning, in deliberate contrast to the Awaitable
                    # union pi uses for `getArgumentCompletions`).
                    suggestions = await provider.get_suggestions(
                        lines, snapshot_line, snapshot_col, {"signal": controller, "force": force}
                    )

                async def apply() -> None:
                    self._apply_autocomplete_response(
                        request_id,
                        controller,
                        snapshot_text,
                        snapshot_line,
                        snapshot_col,
                        suggestions,
                        force=force,
                        explicit_tab=explicit_tab,
                    )

                await self._tui.input_owner.run(apply)
            except BaseException as error:
                # A scope child dying unretrieved is invisible (tonio can only
                # report it as UNHANDLED); a cancelled request may surface its
                # cancellation as an exception and is not an error.
                if isinstance(error, GeneratorExit):
                    raise
                if controller.cancelled:
                    return
                on_error = self._tui.input_owner.on_error
                if on_error is None:
                    raise
                on_error(error)

        self._tui.input_owner.spawn(_request_task())

    def _set_autocomplete_trigger_characters(self, trigger_characters: list[str]) -> None:
        nxt = [*DEFAULT_AUTOCOMPLETE_TRIGGER_CHARACTERS]
        for character in trigger_characters:
            if len(character) != 1 or character == "/" or is_whitespace_char(character) or character in nxt:
                continue
            nxt.append(character)
        self._autocomplete_trigger_characters = nxt
        self._autocomplete_trigger_pattern = _build_trigger_pattern(nxt)
        self._autocomplete_debounce_pattern = _build_debounce_pattern(nxt)

    def _get_autocomplete_debounce_ms(self, *, explicit_tab: bool, force: bool) -> int:
        if explicit_tab or force:
            return 0

        current_line = self._current_line()
        text_before_cursor = current_line[: self._state["cursorCol"]]
        if self._autocomplete_debounce_pattern.search(text_before_cursor):
            return ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS
        return 0

    def _apply_autocomplete_response(
        self,
        request_id: int,
        controller: CancelToken,
        snapshot_text: str,
        snapshot_line: int,
        snapshot_col: int,
        suggestions,
        *,
        force: bool,
        explicit_tab: bool,
    ) -> None:
        if self._autocomplete_provider is None:
            return

        if not self._is_autocomplete_request_current(
            request_id, controller, snapshot_text, snapshot_line, snapshot_col
        ):
            return

        self._autocomplete_abort = None

        if not suggestions or not isinstance(suggestions.get("items"), list) or not suggestions["items"]:
            self._cancel_autocomplete()
            self._tui.request_render()
            return

        if force and explicit_tab and len(suggestions["items"]) == 1:
            item = suggestions["items"][0]
            self._push_undo_snapshot()
            self._last_action = None
            result = self._autocomplete_provider.apply_completion(
                self._state["lines"],
                self._state["cursorLine"],
                self._state["cursorCol"],
                item,
                suggestions["prefix"],
            )
            self._state["lines"] = result["lines"]
            self._state["cursorLine"] = result["cursorLine"]
            self._set_cursor_col(result["cursorCol"])
            if self.on_change:
                self.on_change(self.get_text())
            self._tui.request_render()
            return

        self._apply_autocomplete_suggestions(suggestions, "force" if force else "regular")
        self._tui.request_render()

    def _is_autocomplete_request_current(
        self, request_id: int, controller: CancelToken, snapshot_text: str, snapshot_line: int, snapshot_col: int
    ) -> bool:
        return (
            not controller.cancelled
            and request_id == self._autocomplete_request_id
            and self.get_text() == snapshot_text
            and self._state["cursorLine"] == snapshot_line
            and self._state["cursorCol"] == snapshot_col
        )

    def _apply_autocomplete_suggestions(self, suggestions: dict, state: str) -> None:
        self._autocomplete_prefix = suggestions["prefix"]
        self._autocomplete_list = self._create_autocomplete_list(suggestions["prefix"], suggestions["items"])

        best_match_index = self._get_best_autocomplete_match_index(suggestions["items"], suggestions["prefix"])
        if best_match_index >= 0:
            self._autocomplete_list.set_selected_index(best_match_index)

        self._autocomplete_state = state

    def _cancel_autocomplete_request(self) -> None:
        self._autocomplete_start_token += 1
        if self._autocomplete_debounce_timer is not None:
            self._autocomplete_debounce_timer.cancel()
            self._autocomplete_debounce_timer = None
        if self._autocomplete_abort is not None:
            self._autocomplete_abort.cancel()
        self._autocomplete_abort = None

    def _clear_autocomplete_ui(self) -> None:
        self._autocomplete_state = None
        self._autocomplete_list = None
        self._autocomplete_prefix = ""

    def _cancel_autocomplete(self) -> None:
        self._cancel_autocomplete_request()
        self._clear_autocomplete_ui()

    def is_showing_autocomplete(self) -> bool:
        return self._autocomplete_state is not None

    def _update_autocomplete(self) -> None:
        if not self._autocomplete_state or self._autocomplete_provider is None:
            return
        self._request_autocomplete(force=self._autocomplete_state == "force", explicit_tab=False)
