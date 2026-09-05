"""Mirror of pi tui src/tui-main-screen.ts.

The differential renderer for the terminal's main screen and scrollback: it
diffs frame against frame, moves the cursor to the first changed row, and
repaints only what changed. ``tui_alt_screen`` is the other renderer; the
machinery both share lives in ``tui``.

Port deviations: ``_do_render`` and the exit sequence are async (the terminal
driver is); the crash/debug logs go through ``spawn_blocking`` rather than
Node's sync fs calls; env renames PI_TUI_DEBUG_REDRAW → PIDREI_TUI_DEBUG_REDRAW,
PI_TUI_DEBUG → PIDREI_TUI_DEBUG, log files pidrei-tui-debug.log /
pidrei-tui-crash.log.
"""

import os
import secrets
import tempfile
import time as _time
from datetime import UTC, datetime

import tonio.colored as tonio

from .terminal_image import delete_kitty_image, is_image_line
from .tui import TuiBase
from .utils import visible_width


KITTY_SEQUENCE_PREFIX = "\x1b_G"


def _parse_kitty_image_header(line: str) -> dict | None:
    """Parse ids/rows from a Kitty graphics sequence: {"ids": [int], "rows": int}."""
    sequence_start = line.find(KITTY_SEQUENCE_PREFIX)
    if sequence_start == -1:
        return None

    params_start = sequence_start + len(KITTY_SEQUENCE_PREFIX)
    params_end = line.find(";", params_start)
    if params_end == -1:
        return None

    ids: list[int] = []
    rows = 1
    params = line[params_start:params_end]
    for param in params.split(","):
        key, _, value = param.partition("=")
        if not _:
            continue
        try:
            number_value = int(value)
        except ValueError:
            continue
        if number_value <= 0 or number_value > 0xFFFFFFFF:
            continue
        if key == "i":
            ids.append(number_value)
        elif key == "r":
            rows = number_value
    return {"ids": ids, "rows": rows}


def _extract_kitty_image_ids(line: str) -> list[int]:
    header = _parse_kitty_image_header(line)
    return header["ids"] if header else []


def _extract_kitty_image_rows(line: str) -> int:
    header = _parse_kitty_image_header(line)
    return header["rows"] if header else 1


def _is_termux_session() -> bool:
    return bool(os.environ.get("TERMUX_VERSION"))


def _append_debug_log(path: str, message: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as log_file:
        log_file.write(message)


def _write_crash_log(path: str, data: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as crash_file:
        crash_file.write(data)


class TuiMainScreen(TuiBase):
    """TUI implementation that renders into the terminal's main screen and scrollback."""

    mode = "regular"

    def __init__(self, terminal, show_hardware_cursor: bool | None = None, log_directory: str | None = None) -> None:
        super().__init__(terminal, show_hardware_cursor, log_directory)
        self._previous_lines: list[str] = []
        self._previous_kitty_image_ids: set[int] = set()
        self._previous_width = 0
        self._previous_height = 0
        self._cursor_row = 0  # Logical cursor row (end of rendered content)
        self._hardware_cursor_row = 0  # Actual terminal cursor row (may differ due to IME positioning)
        self._max_lines_rendered = 0  # Track terminal's working area (max lines ever rendered)
        self._previous_viewport_top = 0  # Track previous viewport top for resize-aware cursor moves

    def capture_render_state(self) -> dict:
        """The differential state, so a re-created renderer can pick it up.

        pi's ``TuiMainScreenRenderState`` — a plain record here, keyed like pi's
        fields, because the only consumer is ``restore_render_state``.
        """
        return {
            "previousLines": list(self._previous_lines),
            "previousWidth": self._previous_width,
            "previousHeight": self._previous_height,
            "cursorRow": self._cursor_row,
            "hardwareCursorRow": self._hardware_cursor_row,
            "maxLinesRendered": self._max_lines_rendered,
            "previousViewportTop": self._previous_viewport_top,
        }

    def restore_render_state(self, state: dict) -> None:
        # Image lines are dropped: their kitty placements belong to the previous
        # renderer's uploads, so this renderer must repaint them from scratch.
        self._previous_lines = ["" if is_image_line(line) else line for line in state["previousLines"]]
        self._previous_kitty_image_ids = set()
        self._previous_width = state["previousWidth"]
        self._previous_height = state["previousHeight"]
        self._cursor_row = state["cursorRow"]
        self._hardware_cursor_row = state["hardwareCursorRow"]
        self._max_lines_rendered = state["maxLinesRendered"]
        self._previous_viewport_top = state["previousViewportTop"]

    def _reset_render_state(self) -> None:
        self._previous_lines = []
        self._previous_width = -1  # -1 triggers width_changed, forcing a full clear
        self._previous_height = -1  # -1 triggers height_changed, forcing a full clear
        self._cursor_row = 0
        self._hardware_cursor_row = 0
        self._max_lines_rendered = 0
        self._previous_viewport_top = 0

    async def _before_terminal_stop(self, options: dict) -> None:
        # Move cursor to the end of the content to prevent overwriting/artifacts on exit
        if options.get("preserveScreen") or not self._previous_lines:
            return
        # Overwrite the inverted cursor with a normal space to clear the artifact
        await self.terminal.write(" ")
        target_row = len(self._previous_lines)  # Line after the last content
        line_diff = target_row - self._hardware_cursor_row
        if line_diff > 0:
            await self.terminal.write(f"\x1b[{line_diff}B")
        elif line_diff < 0:
            await self.terminal.write(f"\x1b[{-line_diff}A")
        await self.terminal.write("\r\n")

    def _collect_kitty_image_ids(self, lines: list[str]) -> set[int]:
        ids: set[int] = set()
        for line in lines:
            for image_id in _extract_kitty_image_ids(line):
                ids.add(image_id)
        return ids

    def _delete_kitty_images(self, ids) -> str:
        buffer = ""
        for image_id in ids:
            buffer += delete_kitty_image(image_id)
        return buffer

    def _get_kitty_image_reserved_rows(self, lines: list[str], index: int, max_index: int | None = None) -> int:
        if max_index is None:
            max_index = len(lines) - 1
        rows = _extract_kitty_image_rows(lines[index] if index < len(lines) else "")
        if rows <= 1:
            return 1

        max_rows = min(rows, max_index - index + 1, len(lines) - index)
        reserved_rows = 1
        while reserved_rows < max_rows:
            line = lines[index + reserved_rows] if index + reserved_rows < len(lines) else ""
            if is_image_line(line) or visible_width(line) > 0:
                break
            reserved_rows += 1
        return reserved_rows

    def _expand_changed_range_for_kitty_images(
        self, first_changed: int, last_changed: int, new_lines: list[str]
    ) -> tuple[int, int]:
        expanded_first_changed = first_changed
        expanded_last_changed = last_changed

        def expand_for_lines(lines: list[str]) -> None:
            nonlocal expanded_first_changed, expanded_last_changed
            for i, line in enumerate(lines):
                if not _extract_kitty_image_ids(line):
                    continue
                block_end = i + self._get_kitty_image_reserved_rows(lines, i) - 1
                if i >= first_changed or (i <= last_changed and block_end >= first_changed):
                    expanded_first_changed = min(expanded_first_changed, i)
                    expanded_last_changed = max(expanded_last_changed, block_end)

        expand_for_lines(self._previous_lines)
        expand_for_lines(new_lines)
        return expanded_first_changed, expanded_last_changed

    def _delete_changed_kitty_images(self, first_changed: int, last_changed: int) -> str:
        if first_changed < 0 or last_changed < first_changed:
            return ""

        ids: set[int] = set()
        max_line = min(last_changed, len(self._previous_lines) - 1)
        for i in range(first_changed, max_line + 1):
            for image_id in _extract_kitty_image_ids(self._previous_lines[i]):
                ids.add(image_id)

        return self._delete_kitty_images(ids)

    # ------------------------------------------------------------------
    # Differential rendering
    # ------------------------------------------------------------------

    async def _do_render(self) -> None:  # noqa: C901
        if self._stopped:
            return
        width = self.terminal.columns
        height = self.terminal.rows
        width_changed = self._previous_width != 0 and self._previous_width != width
        height_changed = self._previous_height != 0 and self._previous_height != height
        previous_buffer_length = (
            self._previous_viewport_top + self._previous_height if self._previous_height > 0 else height
        )
        prev_viewport_top = max(0, previous_buffer_length - height) if height_changed else self._previous_viewport_top
        viewport_top = prev_viewport_top
        hardware_cursor_row = self._hardware_cursor_row

        def compute_line_diff(target_row: int) -> int:
            current_screen_row = hardware_cursor_row - prev_viewport_top
            target_screen_row = target_row - viewport_top
            return target_screen_row - current_screen_row

        # Render all components to get new lines
        new_lines = self.render(width)

        # Composite overlays into the rendered lines (before differential compare)
        if self._overlay_stack:
            new_lines = self._composite_overlays(new_lines, width, height)

        # Extract cursor position before applying line resets (marker must be found first)
        cursor_pos = self._extract_cursor_position(new_lines, height)

        new_lines = self._apply_line_resets(new_lines)

        # Helper to clear scrollback and viewport and render all new lines
        async def full_render(clear: bool) -> None:
            self._full_redraw_count += 1
            buffer = "\x1b[?2026h"  # Begin synchronized output
            if clear:
                buffer += self._delete_kitty_images(self._previous_kitty_image_ids)
                buffer += "\x1b[2J\x1b[H\x1b[3J"  # Clear screen, home, then clear scrollback
            i = 0
            while i < len(new_lines):
                if i > 0:
                    buffer += "\r\n"
                line = new_lines[i]
                is_image = is_image_line(line)
                image_reserved_rows = self._get_kitty_image_reserved_rows(new_lines, i) if is_image else 1
                if image_reserved_rows > 1 and image_reserved_rows <= height:
                    buffer += "\r\n" * (image_reserved_rows - 1)
                    buffer += f"\x1b[{image_reserved_rows - 1}A"
                    buffer += line
                    buffer += f"\x1b[{image_reserved_rows - 1}B"
                    i += image_reserved_rows
                    continue
                buffer += line
                i += 1
            buffer += "\x1b[?2026l"  # End synchronized output
            self._emit(buffer)
            self._cursor_row = max(0, len(new_lines) - 1)
            self._hardware_cursor_row = self._cursor_row
            # Reset max lines when clearing, otherwise track growth
            if clear:
                self._max_lines_rendered = len(new_lines)
            else:
                self._max_lines_rendered = max(self._max_lines_rendered, len(new_lines))
            buffer_length = max(height, len(new_lines))
            self._previous_viewport_top = max(0, buffer_length - height)
            await self._position_hardware_cursor(cursor_pos, len(new_lines))
            self._previous_lines = new_lines
            self._previous_kitty_image_ids = self._collect_kitty_image_ids(new_lines)
            self._previous_width = width
            self._previous_height = height

        redraw_log_directory = self._log_directory if os.environ.get("PIDREI_TUI_DEBUG_REDRAW") == "1" else None

        async def log_redraw(reason: str) -> None:
            if redraw_log_directory is None:
                return
            log_path = os.path.join(redraw_log_directory, "pidrei-tui-debug.log")
            timestamp = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            msg = (
                f"[{timestamp}] fullRender: {reason} "
                f"(prev={len(self._previous_lines)}, new={len(new_lines)}, height={height})\n"
            )
            await tonio.spawn_blocking(_append_debug_log, log_path, msg)

        # First render - just output everything without clearing (assumes clean screen)
        if not self._previous_lines and not width_changed and not height_changed:
            await log_redraw("first render")
            await full_render(False)
            return

        # Width changes always need a full re-render because wrapping changes.
        if width_changed:
            await log_redraw(f"terminal width changed ({self._previous_width} -> {width})")
            await full_render(True)
            return

        # Height changes normally need a full re-render to keep the visible viewport aligned,
        # but Termux changes height when the software keyboard shows or hides.
        # In that environment, a full redraw causes the entire history to replay on every toggle.
        if height_changed and not _is_termux_session():
            await log_redraw(f"terminal height changed ({self._previous_height} -> {height})")
            await full_render(True)
            return

        # Content shrunk below the working area and no overlays - re-render to clear empty rows
        # (overlays need the padding, so only do this when no overlays are active)
        if self._clear_on_shrink and len(new_lines) < self._max_lines_rendered and not self._overlay_stack:
            await log_redraw(f"clearOnShrink (maxLinesRendered={self._max_lines_rendered})")
            await full_render(True)
            return

        # Find first and last changed lines
        first_changed = -1
        last_changed = -1
        max_lines = max(len(new_lines), len(self._previous_lines))
        for i in range(max_lines):
            old_line = self._previous_lines[i] if i < len(self._previous_lines) else ""
            new_line = new_lines[i] if i < len(new_lines) else ""

            if old_line != new_line:
                if first_changed == -1:
                    first_changed = i
                last_changed = i
        appended_lines = len(new_lines) > len(self._previous_lines)
        if appended_lines:
            if first_changed == -1:
                first_changed = len(self._previous_lines)
            last_changed = len(new_lines) - 1
        if first_changed != -1:
            first_changed, last_changed = self._expand_changed_range_for_kitty_images(
                first_changed, last_changed, new_lines
            )
        append_start = appended_lines and first_changed == len(self._previous_lines) and first_changed > 0

        # No changes - but still need to update hardware cursor position if it moved
        if first_changed == -1:
            await self._position_hardware_cursor(cursor_pos, len(new_lines))
            self._previous_viewport_top = prev_viewport_top
            self._previous_height = height
            return

        # All changes are in deleted lines (nothing to render, just clear)
        if first_changed >= len(new_lines):
            if len(self._previous_lines) > len(new_lines):
                buffer = "\x1b[?2026h"
                buffer += self._delete_changed_kitty_images(first_changed, last_changed)
                # Move to end of new content (clamp to 0 for empty content)
                target_row = max(0, len(new_lines) - 1)
                if target_row < prev_viewport_top:
                    await log_redraw(f"deleted lines moved viewport up ({target_row} < {prev_viewport_top})")
                    await full_render(True)
                    return
                line_diff = compute_line_diff(target_row)
                if line_diff > 0:
                    buffer += f"\x1b[{line_diff}B"
                elif line_diff < 0:
                    buffer += f"\x1b[{-line_diff}A"
                buffer += "\r"
                # Clear extra lines without scrolling
                extra_lines = len(self._previous_lines) - len(new_lines)
                if extra_lines > height:
                    await log_redraw(f"extraLines > height ({extra_lines} > {height})")
                    await full_render(True)
                    return
                clear_start_offset = 0 if len(new_lines) == 0 else 1
                if extra_lines > 0 and clear_start_offset > 0:
                    buffer += f"\x1b[{clear_start_offset}B"
                for i in range(extra_lines):
                    buffer += "\r\x1b[2K"
                    if i < extra_lines - 1:
                        buffer += "\x1b[1B"
                move_back = max(0, extra_lines - 1 + clear_start_offset)
                if move_back > 0:
                    buffer += f"\x1b[{move_back}A"
                buffer += "\x1b[?2026l"
                self._emit(buffer)
                self._cursor_row = target_row
                self._hardware_cursor_row = target_row
            await self._position_hardware_cursor(cursor_pos, len(new_lines))
            self._previous_lines = new_lines
            self._previous_kitty_image_ids = self._collect_kitty_image_ids(new_lines)
            self._previous_width = width
            self._previous_height = height
            self._previous_viewport_top = prev_viewport_top
            return

        # Differential rendering can only touch what was actually visible.
        # If the first changed line is above the previous viewport, we need a full redraw.
        if first_changed < prev_viewport_top:
            await log_redraw(f"firstChanged < viewportTop ({first_changed} < {prev_viewport_top})")
            await full_render(True)
            return

        # Render from first changed line to end
        # Build buffer with all updates wrapped in synchronized output
        buffer = "\x1b[?2026h"  # Begin synchronized output
        buffer += self._delete_changed_kitty_images(first_changed, last_changed)
        prev_viewport_bottom = prev_viewport_top + height - 1
        move_target_row = first_changed - 1 if append_start else first_changed
        if move_target_row > prev_viewport_bottom:
            current_screen_row = max(0, min(height - 1, hardware_cursor_row - prev_viewport_top))
            move_to_bottom = height - 1 - current_screen_row
            if move_to_bottom > 0:
                buffer += f"\x1b[{move_to_bottom}B"
            scroll = move_target_row - prev_viewport_bottom
            buffer += "\r\n" * scroll
            prev_viewport_top += scroll
            viewport_top += scroll
            hardware_cursor_row = move_target_row

        # Move cursor to first changed line (use hardware_cursor_row for actual position)
        line_diff = compute_line_diff(move_target_row)
        if line_diff > 0:
            buffer += f"\x1b[{line_diff}B"  # Move down
        elif line_diff < 0:
            buffer += f"\x1b[{-line_diff}A"  # Move up

        buffer += "\r\n" if append_start else "\r"  # Move to column 0

        # Only render changed lines (first_changed to last_changed), not all lines to end
        # This reduces flicker when only a single line changes (e.g., spinner animation)
        render_end = min(last_changed, len(new_lines) - 1)
        i = first_changed
        while i <= render_end:
            if i > first_changed:
                buffer += "\r\n"
            line = new_lines[i]
            is_image = is_image_line(line)
            image_reserved_rows = self._get_kitty_image_reserved_rows(new_lines, i, render_end) if is_image else 1
            if image_reserved_rows > 1:
                image_start_screen_row = i - viewport_top
                if image_start_screen_row < 0 or image_start_screen_row + image_reserved_rows > height:
                    await log_redraw(
                        f"kitty image pre-clear would scroll ({image_start_screen_row} + {image_reserved_rows} > {height})"
                    )
                    await full_render(True)
                    return

                buffer += "\x1b[2K"
                buffer += "\r\n\x1b[2K" * (image_reserved_rows - 1)
                buffer += f"\x1b[{image_reserved_rows - 1}A"
                buffer += line
                buffer += f"\x1b[{image_reserved_rows - 1}B"
                i += image_reserved_rows
                continue

            buffer += "\x1b[2K"  # Clear current line
            if not is_image and visible_width(line) > width:
                # Log all lines to crash file for debugging
                crash_log_path = os.path.join(
                    self._log_directory if self._log_directory is not None else tempfile.gettempdir(),
                    "pidrei-tui-crash.log",
                )
                timestamp = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
                crash_data = "\n".join(
                    [
                        f"Crash at {timestamp}",
                        f"Terminal width: {width}",
                        f"Line {i} visible width: {visible_width(line)}",
                        "",
                        "=== All rendered lines ===",
                        *[f"[{idx}] (w={visible_width(l)}) {l}" for idx, l in enumerate(new_lines)],
                        "",
                    ]
                )
                await tonio.spawn_blocking(_write_crash_log, crash_log_path, crash_data)

                # Terminal cleanup happens in the caller's shutdown path; pi
                # calls the sync stop() here, but stop() is async in the port
                # and _do_render runs inside an owner render job.
                raise Exception(
                    "\n".join(
                        [
                            f"Rendered line {i} exceeds terminal width ({visible_width(line)} > {width}).",
                            "",
                            "This is likely caused by a custom TUI component not truncating its output.",
                            "Use visible_width() to measure and truncate_to_width() to truncate lines.",
                            "",
                            f"Debug log written to: {crash_log_path}",
                        ]
                    )
                )
            buffer += line
            i += 1

        # Track where cursor ended up after rendering
        final_cursor_row = render_end

        # If we had more lines before, clear them and move cursor back
        if len(self._previous_lines) > len(new_lines):
            # Move to end of new content first if we stopped before it
            if render_end < len(new_lines) - 1:
                move_down = len(new_lines) - 1 - render_end
                buffer += f"\x1b[{move_down}B"
                final_cursor_row = len(new_lines) - 1
            extra_lines = len(self._previous_lines) - len(new_lines)
            buffer += "\r\n\x1b[2K" * extra_lines
            # Move cursor back to end of new content
            buffer += f"\x1b[{extra_lines}A"

        buffer += "\x1b[?2026l"  # End synchronized output

        if os.environ.get("PIDREI_TUI_DEBUG") == "1":
            debug_dir = "/tmp/tui"  # noqa: S108
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f"render-{_time.time_ns() // 1_000_000}-{secrets.token_hex(6)}.log")
            debug_data = "\n".join(
                [
                    f"firstChanged: {first_changed}",
                    f"viewportTop: {viewport_top}",
                    f"cursorRow: {self._cursor_row}",
                    f"height: {height}",
                    f"lineDiff: {line_diff}",
                    f"hardwareCursorRow: {hardware_cursor_row}",
                    f"renderEnd: {render_end}",
                    f"finalCursorRow: {final_cursor_row}",
                    f"cursorPos: {cursor_pos!r}",
                    f"newLines.length: {len(new_lines)}",
                    f"previousLines.length: {len(self._previous_lines)}",
                    "",
                    "=== newLines ===",
                    repr(new_lines),
                    "",
                    "=== previousLines ===",
                    repr(self._previous_lines),
                    "",
                    "=== buffer ===",
                    repr(buffer),
                ]
            )
            await tonio.spawn_blocking(_write_crash_log, debug_path, debug_data)

        # Write entire buffer at once
        self._emit(buffer)

        # Track cursor position for next render
        # cursor_row tracks end of content (for viewport calculation)
        # hardware_cursor_row tracks actual terminal cursor position (for movement)
        self._cursor_row = max(0, len(new_lines) - 1)
        self._hardware_cursor_row = final_cursor_row
        # Track terminal's working area (grows but doesn't shrink unless cleared)
        self._max_lines_rendered = max(self._max_lines_rendered, len(new_lines))
        self._previous_viewport_top = max(prev_viewport_top, final_cursor_row - height + 1)

        # Position hardware cursor for IME
        await self._position_hardware_cursor(cursor_pos, len(new_lines))

        self._previous_lines = new_lines
        self._previous_kitty_image_ids = self._collect_kitty_image_ids(new_lines)
        self._previous_width = width
        self._previous_height = height

    async def _position_hardware_cursor(self, cursor_pos: dict | None, total_lines: int) -> None:
        """Position the hardware cursor for IME candidate window.

        Port deviation: pi ends with `terminal.showCursor()/hideCursor()` as
        separate writes; here the cursor sequence is part of the positioning
        buffer so the frame tail is one write from the render job.
        """
        if not cursor_pos or total_lines <= 0:
            self._emit("\x1b[?25l")
            return

        # Clamp cursor position to valid range
        target_row = max(0, min(cursor_pos["row"], total_lines - 1))
        target_col = max(0, cursor_pos["col"])

        # Move cursor from current position to target
        row_delta = target_row - self._hardware_cursor_row
        buffer = ""
        if row_delta > 0:
            buffer += f"\x1b[{row_delta}B"  # Move down
        elif row_delta < 0:
            buffer += f"\x1b[{-row_delta}A"  # Move up
        # Move to absolute column (1-indexed)
        buffer += f"\x1b[{target_col + 1}G"
        buffer += "\x1b[?25h" if self._show_hardware_cursor else "\x1b[?25l"
        self._emit(buffer)
        self._hardware_cursor_row = target_row
