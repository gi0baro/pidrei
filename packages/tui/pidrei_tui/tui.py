"""Mirror of pi tui src/tui.ts.

Minimal TUI implementation with differential rendering.

Port deviations (documented once here):

- Render coalescing: pi chains ``process.nextTick`` + a 16ms ``setTimeout``
  throttle; here ``start()`` spawns a single render-loop task that parks on
  an Event, applies the same 16ms throttle against the last render time, and
  calls ``_do_render``. ``request_render()`` stays sync (callable from input
  handlers); ``force=True`` resets the differential state synchronously and
  skips the throttle for the next render, but does not interrupt an
  in-flight throttle sleep.
- ``start``/``stop`` are async (they drive the async terminal driver and the
  render-loop lifecycle). The loop exits cooperatively on ``stop()`` — no
  task abort involved.
- ``query_terminal_background_color``/``query_terminal_color_scheme`` are
  async methods; their pending-state transitions take a sync lock because
  the input pump and the querying task may run on different tonio workers.
- ``CURSOR_MARKER`` is an APC sequence pi brands "pi:c" — renamed to
  "pidrei:c" (pi naming itself).
- Env renames: PI_HARDWARE_CURSOR → PIDREI_HARDWARE_CURSOR,
  PI_CLEAR_ON_SHRINK → PIDREI_CLEAR_ON_SHRINK, PI_DEBUG_REDRAW →
  PIDREI_DEBUG_REDRAW, PI_TUI_DEBUG → PIDREI_TUI_DEBUG,
  PI_CODING_AGENT_DIR → PIDREI_CODING_AGENT_DIR (log dir default
  ~/.pidrei/agent); log files pidrei-debug.log / pidrei-crash.log.
"""

import math
import os
import re
import secrets
import threading
import time as _time
from datetime import UTC, datetime
from typing import Any, Protocol

import tonio.colored as tonio

from .keys import is_key_release, matches_key
from .terminal_colors import (
    is_osc11_background_color_response,
    parse_osc11_background_color,
    parse_terminal_color_scheme_report,
)
from .terminal_image import delete_kitty_image, get_capabilities, is_image_line, set_cell_dimensions
from .utils import extract_segments, normalize_terminal_output, slice_by_column, slice_with_width, visible_width


KITTY_SEQUENCE_PREFIX = "\x1b_G"

_CELL_SIZE_RESPONSE_RE = re.compile(r"^\x1b\[6;(\d+);(\d+)t$")
_PERCENT_RE = re.compile(r"^(\d+(?:\.\d+)?)%$")


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


class Component(Protocol):
    """Component interface - all components must implement this."""

    def render(self, width: int) -> list[str]:
        """Render the component to lines for the given viewport width."""
        ...

    def invalidate(self) -> None:
        """Invalidate any cached rendering state.

        Called when theme changes or when component needs to re-render from
        scratch.
        """
        ...

    # Optional: handle_input(data) for keyboard input when focused;
    # wants_key_release = True to receive Kitty key release events.


class Focusable(Protocol):
    """Interface for components that can receive focus and display a hardware cursor.

    When focused, the component should emit CURSOR_MARKER at the cursor
    position in its render output. TUI will find this marker and position the
    hardware cursor there for proper IME candidate window positioning.
    """

    focused: bool


def is_focusable(component) -> bool:
    """Check if a component implements Focusable (pi: `"focused" in component`)."""
    return component is not None and hasattr(component, "focused")


# Cursor position marker - APC (Application Program Command) sequence.
# This is a zero-width escape sequence that terminals ignore.
# Components emit this at the cursor position when focused.
# TUI finds and strips this marker, then positions the hardware cursor there.
CURSOR_MARKER = "\x1b_pidrei:c\x07"

# OverlayAnchor: "center" | "top-left" | "top-right" | "bottom-left" |
# "bottom-right" | "top-center" | "bottom-center" | "left-center" | "right-center"


def _parse_size_value(value, reference_size: int) -> int | None:
    """Parse a SizeValue (int or "50%" string) into an absolute value."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    match = _PERCENT_RE.match(value) if isinstance(value, str) else None
    if match:
        return math.floor(reference_size * float(match.group(1)) / 100)
    return None


def _is_termux_session() -> bool:
    return bool(os.environ.get("TERMUX_VERSION"))


class _OverlayStackEntry:
    __slots__ = ("component", "focus_order", "hidden", "options", "pre_focus")

    def __init__(self, component, options, pre_focus, focus_order) -> None:
        self.component = component
        self.options: dict = options or {}
        self.pre_focus = pre_focus
        self.hidden = False
        self.focus_order = focus_order


class OverlayHandle:
    """Handle returned by show_overlay for controlling the overlay."""

    __slots__ = ("focus", "hide", "is_focused", "is_hidden", "set_hidden", "unfocus")

    def __init__(self, *, hide, set_hidden, is_hidden, focus, unfocus, is_focused) -> None:
        self.hide = hide
        self.set_hidden = set_hidden
        self.is_hidden = is_hidden
        self.focus = focus
        self.unfocus = unfocus
        self.is_focused = is_focused


class _PendingOsc11Query:
    __slots__ = ("event", "result", "settled")

    def __init__(self) -> None:
        self.settled = False
        self.result = None
        self.event = tonio.Event()


class Container:
    """A component that contains other components."""

    def __init__(self) -> None:
        self.children: list = []

    def add_child(self, component) -> None:
        self.children.append(component)

    def remove_child(self, component) -> None:
        try:
            self.children.remove(component)
        except ValueError:
            pass

    def clear(self) -> None:
        self.children = []

    def invalidate(self) -> None:
        for child in self.children:
            invalidate = getattr(child, "invalidate", None)
            if invalidate is not None:
                invalidate()

    def render(self, width: int) -> list[str]:
        lines: list[str] = []
        for child in self.children:
            lines.extend(child.render(width))
        return lines


_MIN_RENDER_INTERVAL_S = 0.016

SEGMENT_RESET = "\x1b[0m\x1b]8;;\x07"


class TUI(Container):
    """Main class for managing terminal UI with differential rendering."""

    def __init__(self, terminal, show_hardware_cursor: bool | None = None, log_directory: str | None = None) -> None:
        super().__init__()
        self.terminal = terminal
        self._log_directory = (
            log_directory
            if log_directory is not None
            else os.environ.get("PIDREI_CODING_AGENT_DIR") or os.path.join(os.path.expanduser("~"), ".pidrei", "agent")
        )
        self._previous_lines: list[str] = []
        self._previous_kitty_image_ids: set[int] = set()
        self._previous_width = 0
        self._previous_height = 0
        self._focused_component = None
        self._input_listeners: list = []

        # Global callback for debug key (Shift+Ctrl+D). Called before input is
        # forwarded to the focused component.
        self.on_debug = None

        self._render_signal = tonio.Event()
        self._render_force = False
        self._last_render_at = 0.0
        self._render_scope = None
        self._cursor_row = 0  # Logical cursor row (end of rendered content)
        self._hardware_cursor_row = 0  # Actual terminal cursor row (may differ due to IME positioning)
        self._show_hardware_cursor = os.environ.get("PIDREI_HARDWARE_CURSOR") == "1"
        # Clear empty rows when content shrinks (default: off)
        self._clear_on_shrink = os.environ.get("PIDREI_CLEAR_ON_SHRINK") == "1"
        self._max_lines_rendered = 0  # Track terminal's working area (max lines ever rendered)
        self._previous_viewport_top = 0  # Track previous viewport top for resize-aware cursor moves
        self._full_redraw_count = 0
        self._stopped = False
        self._query_lock = threading.Lock()
        self._pending_osc11_replies = 0
        self._pending_osc11_queries: list[_PendingOsc11Query] = []
        self._color_scheme_listeners: list = []
        self._color_scheme_notifications_enabled = False

        # Overlay stack for modal components rendered on top of base content
        self._focus_order_counter = 0
        self._overlay_stack: list[_OverlayStackEntry] = []
        self._overlay_focus_restore: dict = {"status": "inactive"}

        if show_hardware_cursor is not None:
            self._show_hardware_cursor = show_hardware_cursor

    @property
    def full_redraws(self) -> int:
        return self._full_redraw_count

    def get_show_hardware_cursor(self) -> bool:
        return self._show_hardware_cursor

    def set_show_hardware_cursor(self, enabled: bool) -> None:
        if self._show_hardware_cursor == enabled:
            return
        self._show_hardware_cursor = enabled
        if not enabled:
            self.terminal.hide_cursor()
        self.request_render()

    def get_clear_on_shrink(self) -> bool:
        return self._clear_on_shrink

    def set_clear_on_shrink(self, enabled: bool) -> None:
        """Set whether to trigger full re-render when content shrinks.

        When enabled, empty rows are cleared when content shrinks. When
        disabled, empty rows remain (reduces redraws on slower terminals).
        """
        self._clear_on_shrink = enabled

    # ------------------------------------------------------------------
    # Focus and overlay focus-restore machinery
    # ------------------------------------------------------------------

    def set_focus(self, component) -> None:
        self._set_focus_internal(component, overlay_focus_restore="clear")

    def _set_focus_internal(self, component, *, overlay_focus_restore: str) -> None:
        previous_focus = self._focused_component
        next_focus = component
        previous_focused_overlay = None
        if previous_focus is not None:
            previous_focused_overlay = next(
                (
                    entry
                    for entry in self._overlay_stack
                    if entry.component is previous_focus and self._is_overlay_visible(entry)
                ),
                None,
            )
        next_focus_is_overlay = (
            any(entry.component is next_focus for entry in self._overlay_stack) if next_focus is not None else False
        )
        restore_state = self._get_visible_overlay_focus_restore()
        if next_focus is not None and not next_focus_is_overlay:
            if restore_state["status"] == "blocked" and restore_state["blockedBy"] is previous_focus:
                if restore_state["resume"]["status"] == "focus-target" or not self._is_component_mounted(
                    restore_state["blockedBy"]
                ):
                    next_focus = self._resolve_blocked_overlay_focus_resume(restore_state)
                else:
                    self._overlay_focus_restore = {
                        "status": "blocked",
                        "overlay": restore_state["overlay"],
                        "blockedBy": next_focus,
                        "resume": restore_state["resume"],
                    }
            elif (
                previous_focused_overlay is not None
                and restore_state["status"] != "inactive"
                and restore_state["overlay"] is previous_focused_overlay
                and not self._is_overlay_focus_ancestor(previous_focused_overlay, next_focus)
            ):
                self._overlay_focus_restore = {
                    "status": "blocked",
                    "overlay": previous_focused_overlay,
                    "blockedBy": next_focus,
                    "resume": {"status": "restore-overlay"},
                }
        elif next_focus is None:
            if restore_state["status"] == "blocked" and restore_state["blockedBy"] is previous_focus:
                next_focus = self._resolve_blocked_overlay_focus_resume(restore_state)
            elif overlay_focus_restore == "clear":
                self._clear_overlay_focus_restore()

        if is_focusable(self._focused_component):
            self._focused_component.focused = False

        self._focused_component = next_focus

        if is_focusable(next_focus):
            next_focus.focused = True

        focused_overlay = None
        if next_focus is not None:
            focused_overlay = next(
                (
                    entry
                    for entry in self._overlay_stack
                    if entry.component is next_focus and self._is_overlay_visible(entry)
                ),
                None,
            )
        if focused_overlay is not None:
            self._overlay_focus_restore = {"status": "eligible", "overlay": focused_overlay}

    def _clear_overlay_focus_restore(self) -> None:
        self._overlay_focus_restore = {"status": "inactive"}

    def _clear_overlay_focus_restore_for(self, overlay: _OverlayStackEntry) -> None:
        if self._overlay_focus_restore["status"] != "inactive" and self._overlay_focus_restore["overlay"] is overlay:
            self._clear_overlay_focus_restore()

    def _resolve_blocked_overlay_focus_resume(self, restore_state: dict):
        if restore_state["resume"]["status"] == "restore-overlay":
            return restore_state["overlay"].component
        self._clear_overlay_focus_restore()
        return restore_state["resume"]["target"]

    def _get_visible_overlay_focus_restore(self) -> dict:
        restore_state = self._overlay_focus_restore
        if restore_state["status"] == "inactive":
            return restore_state
        if restore_state["overlay"] not in self._overlay_stack or not self._is_overlay_visible(
            restore_state["overlay"]
        ):
            return {"status": "inactive"}
        return restore_state

    def _is_overlay_focus_ancestor(self, entry: _OverlayStackEntry, component) -> bool:
        visited: list = []
        current = entry.pre_focus
        while current is not None and all(current is not seen for seen in visited):
            visited.append(current)
            if current is component:
                return True
            owner = next((overlay for overlay in self._overlay_stack if overlay.component is current), None)
            current = owner.pre_focus if owner is not None else None
        return False

    def _retarget_overlay_pre_focus(self, removed: _OverlayStackEntry) -> None:
        for overlay in self._overlay_stack:
            if overlay is not removed and overlay.pre_focus is removed.component:
                overlay.pre_focus = removed.pre_focus

    def _is_component_mounted(self, component) -> bool:
        return any(self._contains_component(child, component) for child in self.children)

    def _contains_component(self, root, target) -> bool:
        if root is target:
            return True
        if not isinstance(root, Container):
            return False
        return any(self._contains_component(child, target) for child in root.children)

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def show_overlay(self, component, options: dict | None = None) -> OverlayHandle:
        """Show an overlay component with configurable positioning and sizing.

        Options record (camelCase like pi's OverlayOptions): width, minWidth,
        maxHeight, anchor, offsetX, offsetY, row, col, margin, visible,
        nonCapturing.
        """
        self._focus_order_counter += 1
        entry = _OverlayStackEntry(component, options, self._focused_component, self._focus_order_counter)
        self._overlay_stack.append(entry)
        # Only focus if overlay is actually visible
        if not entry.options.get("nonCapturing") and self._is_overlay_visible(entry):
            self.set_focus(component)
        self.terminal.hide_cursor()
        self.request_render()

        def hide() -> None:
            if entry in self._overlay_stack:
                self._clear_overlay_focus_restore_for(entry)
                self._retarget_overlay_pre_focus(entry)
                self._overlay_stack.remove(entry)
                # Restore focus if this overlay had focus
                if self._focused_component is component:
                    top_visible = self._get_topmost_visible_overlay()
                    self.set_focus(top_visible.component if top_visible is not None else entry.pre_focus)
                if not self._overlay_stack:
                    self.terminal.hide_cursor()
                self.request_render()

        def set_hidden(hidden: bool) -> None:
            if entry.hidden == hidden:
                return
            entry.hidden = hidden
            # Update focus when hiding/showing
            if hidden:
                self._clear_overlay_focus_restore_for(entry)
                # If this overlay had focus, move focus to next visible or pre_focus
                if self._focused_component is component:
                    top_visible = self._get_topmost_visible_overlay()
                    self.set_focus(top_visible.component if top_visible is not None else entry.pre_focus)
            else:
                # Restore focus to this overlay when showing (if it's actually visible)
                if not entry.options.get("nonCapturing") and self._is_overlay_visible(entry):
                    self._focus_order_counter += 1
                    entry.focus_order = self._focus_order_counter
                    self.set_focus(component)
            self.request_render()

        def is_hidden() -> bool:
            return entry.hidden

        def focus() -> None:
            if entry not in self._overlay_stack or not self._is_overlay_visible(entry):
                return
            self._focus_order_counter += 1
            entry.focus_order = self._focus_order_counter
            self.set_focus(component)
            self.request_render()

        def unfocus(unfocus_options: dict | None = None) -> None:
            is_focused_now = self._focused_component is component
            restore_state = self._overlay_focus_restore
            has_pending_restore = restore_state["status"] != "inactive" and restore_state["overlay"] is entry
            if not is_focused_now and not has_pending_restore:
                return
            if (
                restore_state["status"] == "blocked"
                and restore_state["overlay"] is entry
                and self._focused_component is restore_state["blockedBy"]
            ):
                if unfocus_options is not None:
                    self._overlay_focus_restore = {
                        "status": "blocked",
                        "overlay": entry,
                        "blockedBy": restore_state["blockedBy"],
                        "resume": {"status": "focus-target", "target": unfocus_options["target"]},
                    }
                else:
                    self._clear_overlay_focus_restore()
                self.request_render()
                return
            self._clear_overlay_focus_restore_for(entry)
            if is_focused_now or unfocus_options is not None:
                top_visible = self._get_topmost_visible_overlay()
                fallback_target = (
                    top_visible.component if top_visible is not None and top_visible is not entry else entry.pre_focus
                )
                self.set_focus(unfocus_options["target"] if unfocus_options is not None else fallback_target)
            self.request_render()

        def is_focused() -> bool:
            return self._focused_component is component

        return OverlayHandle(
            hide=hide,
            set_hidden=set_hidden,
            is_hidden=is_hidden,
            focus=focus,
            unfocus=unfocus,
            is_focused=is_focused,
        )

    def hide_overlay(self) -> None:
        """Hide the topmost overlay and restore previous focus."""
        if not self._overlay_stack:
            return
        overlay = self._overlay_stack[-1]
        self._clear_overlay_focus_restore_for(overlay)
        self._retarget_overlay_pre_focus(overlay)
        self._overlay_stack.pop()
        if self._focused_component is overlay.component:
            # Find topmost visible overlay, or fall back to pre_focus
            top_visible = self._get_topmost_visible_overlay()
            self.set_focus(top_visible.component if top_visible is not None else overlay.pre_focus)
        if not self._overlay_stack:
            self.terminal.hide_cursor()
        self.request_render()

    def has_overlay(self) -> bool:
        """Check if there are any visible overlays."""
        return any(self._is_overlay_visible(entry) for entry in self._overlay_stack)

    def _is_overlay_visible(self, entry: _OverlayStackEntry) -> bool:
        if entry.hidden:
            return False
        visible = entry.options.get("visible")
        if visible is not None:
            return visible(self.terminal.columns, self.terminal.rows)
        return True

    def _get_topmost_visible_overlay(self) -> _OverlayStackEntry | None:
        """Find the visual-frontmost visible capturing overlay, if any."""
        topmost: _OverlayStackEntry | None = None
        for overlay in self._overlay_stack:
            if overlay.options.get("nonCapturing") or not self._is_overlay_visible(overlay):
                continue
            if topmost is None or overlay.focus_order > topmost.focus_order:
                topmost = overlay
        return topmost

    def invalidate(self) -> None:
        super().invalidate()
        for overlay in self._overlay_stack:
            invalidate = getattr(overlay.component, "invalidate", None)
            if invalidate is not None:
                invalidate()

    # ------------------------------------------------------------------
    # Lifecycle and render scheduling
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._stopped = False
        await self.terminal.start(self._handle_input, self.request_render)
        self.terminal.hide_cursor()
        if self._color_scheme_notifications_enabled:
            self.terminal.write("\x1b[?2031h")
        self._query_cell_size()
        self._render_scope = tonio.scope()
        await self._render_scope.__aenter__()
        self._render_scope.spawn(self._render_loop())
        self.request_render()

    def add_input_listener(self, listener):
        if listener not in self._input_listeners:
            self._input_listeners.append(listener)

        def unsubscribe() -> None:
            self.remove_input_listener(listener)

        return unsubscribe

    def remove_input_listener(self, listener) -> None:
        if listener in self._input_listeners:
            self._input_listeners.remove(listener)

    def on_terminal_color_scheme_change(self, listener):
        if listener not in self._color_scheme_listeners:
            self._color_scheme_listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._color_scheme_listeners:
                self._color_scheme_listeners.remove(listener)

        return unsubscribe

    def set_terminal_color_scheme_notifications(self, enabled: bool) -> None:
        if self._color_scheme_notifications_enabled == enabled:
            return
        self._color_scheme_notifications_enabled = enabled
        if not self._stopped:
            self.terminal.write("\x1b[?2031h" if enabled else "\x1b[?2031l")

    def _query_cell_size(self) -> None:
        # Only query if terminal supports images (cell size is only used for image rendering)
        if not get_capabilities()["images"]:
            return
        # Query terminal for cell size in pixels: CSI 16 t
        # Response format: CSI 6 ; height ; width t
        self.terminal.write("\x1b[16t")

    async def stop(self) -> None:
        self._stopped = True
        self._render_signal.set()
        if self._render_scope is not None:
            await self._render_scope.__aexit__(None, None, None)
            self._render_scope = None
        if self._color_scheme_notifications_enabled:
            self.terminal.write("\x1b[?2031l")
        # Move cursor to the end of the content to prevent overwriting/artifacts on exit
        if self._previous_lines:
            # Overwrite the inverted cursor with a normal space to clear the artifact
            self.terminal.write(" ")
            target_row = len(self._previous_lines)  # Line after the last content
            line_diff = target_row - self._hardware_cursor_row
            if line_diff > 0:
                self.terminal.write(f"\x1b[{line_diff}B")
            elif line_diff < 0:
                self.terminal.write(f"\x1b[{-line_diff}A")
            self.terminal.write("\r\n")

        self.terminal.show_cursor()
        await self.terminal.stop()

    def request_render(self, force: bool = False) -> None:
        if force:
            self._previous_lines = []
            self._previous_width = -1  # -1 triggers width_changed, forcing a full clear
            self._previous_height = -1  # -1 triggers height_changed, forcing a full clear
            self._cursor_row = 0
            self._hardware_cursor_row = 0
            self._max_lines_rendered = 0
            self._previous_viewport_top = 0
            self._render_force = True
        self._render_signal.set()

    async def _render_loop(self) -> None:
        while True:
            await self._render_signal.wait(None)
            if self._stopped:
                return
            force = self._render_force
            self._render_force = False
            if not force:
                elapsed = _time.monotonic() - self._last_render_at
                delay = _MIN_RENDER_INTERVAL_S - elapsed
                if delay > 0:
                    await tonio.sleep(delay)
                    if self._stopped:
                        return
            # Absorb requests that arrived up to this point; requests during
            # _do_render re-arm the loop (pi: renderRequested re-check).
            self._render_signal.clear()
            self._last_render_at = _time.monotonic()
            self._do_render()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _handle_input(self, data: str) -> None:
        if self._consume_osc11_background_response(data):
            return
        if self._consume_terminal_color_scheme_report(data):
            return

        if self._input_listeners:
            current = data
            for listener in list(self._input_listeners):
                result = listener(current)
                if result and result.get("consume"):
                    return
                if result and result.get("data") is not None:
                    current = result["data"]
            if len(current) == 0:
                return
            data = current

        # Consume terminal cell size responses without blocking unrelated input.
        if self._consume_cell_size_response(data):
            return

        # Global debug key handler (Shift+Ctrl+D)
        if matches_key(data, "shift+ctrl+d") and self.on_debug is not None:
            self.on_debug()
            return

        # If focused component is an overlay, verify it's still visible
        # (visibility can change due to terminal resize or visible() callback)
        focused_overlay = next(
            (entry for entry in self._overlay_stack if entry.component is self._focused_component), None
        )
        if focused_overlay is not None and not self._is_overlay_visible(focused_overlay):
            # Focused overlay is no longer visible, redirect to topmost visible overlay
            top_visible = self._get_topmost_visible_overlay()
            if top_visible is not None:
                self.set_focus(top_visible.component)
            else:
                self._set_focus_internal(focused_overlay.pre_focus, overlay_focus_restore="preserve")

        focus_is_overlay = any(entry.component is self._focused_component for entry in self._overlay_stack)
        if not focus_is_overlay:
            restore_state = self._get_visible_overlay_focus_restore()
            if restore_state["status"] == "eligible":
                self.set_focus(restore_state["overlay"].component)
            elif restore_state["status"] == "blocked" and restore_state["blockedBy"] is not self._focused_component:
                if restore_state["resume"]["status"] == "restore-overlay":
                    self.set_focus(restore_state["overlay"].component)
                else:
                    self._clear_overlay_focus_restore()
                    self.set_focus(restore_state["resume"]["target"])

        # Pass input to focused component (including Ctrl+C)
        # The focused component can decide how to handle Ctrl+C
        focused = self._focused_component
        handle = getattr(focused, "handle_input", None) if focused is not None else None
        if handle is not None:
            # Filter out key release events unless component opts in
            if is_key_release(data) and not getattr(focused, "wants_key_release", False):
                return
            handle(data)
            self.request_render()

    def _consume_osc11_background_response(self, data: str) -> bool:
        if self._pending_osc11_replies <= 0:
            return False

        if not is_osc11_background_color_response(data):
            return False

        rgb = parse_osc11_background_color(data)
        with self._query_lock:
            self._pending_osc11_replies -= 1
            query = self._pending_osc11_queries.pop(0) if self._pending_osc11_queries else None
            if query is not None and not query.settled:
                query.settled = True
                query.result = rgb
                query.event.set()
        return True

    def _consume_terminal_color_scheme_report(self, data: str) -> bool:
        scheme = parse_terminal_color_scheme_report(data)
        if not scheme:
            return False

        for listener in list(self._color_scheme_listeners):
            listener(scheme)
        return True

    def _consume_cell_size_response(self, data: str) -> bool:
        # Response format: ESC [ 6 ; height ; width t
        match = _CELL_SIZE_RESPONSE_RE.match(data)
        if not match:
            return False

        height_px = int(match.group(1))
        width_px = int(match.group(2))
        if height_px <= 0 or width_px <= 0:
            return True

        set_cell_dimensions({"widthPx": width_px, "heightPx": height_px})
        # Invalidate all components so images re-render with correct dimensions.
        self.invalidate()
        self.request_render()
        return True

    # ------------------------------------------------------------------
    # Overlay layout and compositing
    # ------------------------------------------------------------------

    def _resolve_overlay_layout(self, options: dict, overlay_height: int, term_width: int, term_height: int) -> dict:
        """Resolve overlay layout from options: {"width", "row", "col", "maxHeight"}."""
        opt = options or {}

        # Parse margin (clamp to non-negative)
        raw_margin = opt.get("margin")
        if isinstance(raw_margin, (int, float)) and not isinstance(raw_margin, bool):
            margin = {"top": raw_margin, "right": raw_margin, "bottom": raw_margin, "left": raw_margin}
        else:
            margin = raw_margin or {}
        margin_top = max(0, margin.get("top") or 0)
        margin_right = max(0, margin.get("right") or 0)
        margin_bottom = max(0, margin.get("bottom") or 0)
        margin_left = max(0, margin.get("left") or 0)

        # Available space after margins
        avail_width = max(1, term_width - margin_left - margin_right)
        avail_height = max(1, term_height - margin_top - margin_bottom)

        # === Resolve width ===
        width = _parse_size_value(opt.get("width"), term_width)
        if width is None:
            width = min(80, avail_width)
        # Apply minWidth
        if opt.get("minWidth") is not None:
            width = max(width, opt["minWidth"])
        # Clamp to available space
        width = max(1, min(width, avail_width))

        # === Resolve maxHeight ===
        max_height = _parse_size_value(opt.get("maxHeight"), term_height)
        # Clamp to available space
        if max_height is not None:
            max_height = max(1, min(max_height, avail_height))

        # Effective overlay height (may be clamped by maxHeight)
        effective_height = min(overlay_height, max_height) if max_height is not None else overlay_height

        # === Resolve position ===
        if opt.get("row") is not None:
            if isinstance(opt["row"], str):
                # Percentage: 0% = top, 100% = bottom (overlay stays within bounds)
                match = _PERCENT_RE.match(opt["row"])
                if match:
                    max_row = max(0, avail_height - effective_height)
                    percent = float(match.group(1)) / 100
                    row = margin_top + int(max_row * percent)
                else:
                    # Invalid format, fall back to center
                    row = self._resolve_anchor_row("center", effective_height, avail_height, margin_top)
            else:
                # Absolute row position
                row = opt["row"]
        else:
            # Anchor-based (default: center)
            anchor = opt.get("anchor") or "center"
            row = self._resolve_anchor_row(anchor, effective_height, avail_height, margin_top)

        if opt.get("col") is not None:
            if isinstance(opt["col"], str):
                # Percentage: 0% = left, 100% = right (overlay stays within bounds)
                match = _PERCENT_RE.match(opt["col"])
                if match:
                    max_col = max(0, avail_width - width)
                    percent = float(match.group(1)) / 100
                    col = margin_left + int(max_col * percent)
                else:
                    # Invalid format, fall back to center
                    col = self._resolve_anchor_col("center", width, avail_width, margin_left)
            else:
                # Absolute column position
                col = opt["col"]
        else:
            # Anchor-based (default: center)
            anchor = opt.get("anchor") or "center"
            col = self._resolve_anchor_col(anchor, width, avail_width, margin_left)

        # Apply offsets
        if opt.get("offsetY") is not None:
            row += opt["offsetY"]
        if opt.get("offsetX") is not None:
            col += opt["offsetX"]

        # Clamp to terminal bounds (respecting margins)
        row = max(margin_top, min(row, term_height - margin_bottom - effective_height))
        col = max(margin_left, min(col, term_width - margin_right - width))

        return {"width": width, "row": row, "col": col, "maxHeight": max_height}

    def _resolve_anchor_row(self, anchor: str, height: int, avail_height: int, margin_top: int) -> int:
        if anchor in ("top-left", "top-center", "top-right"):
            return margin_top
        if anchor in ("bottom-left", "bottom-center", "bottom-right"):
            return margin_top + avail_height - height
        # left-center | center | right-center
        return margin_top + (avail_height - height) // 2

    def _resolve_anchor_col(self, anchor: str, width: int, avail_width: int, margin_left: int) -> int:
        if anchor in ("top-left", "left-center", "bottom-left"):
            return margin_left
        if anchor in ("top-right", "right-center", "bottom-right"):
            return margin_left + avail_width - width
        # top-center | center | bottom-center
        return margin_left + (avail_width - width) // 2

    def _composite_overlays(self, lines: list[str], term_width: int, term_height: int) -> list[str]:
        """Composite all overlays into content lines (sorted by focus_order, higher = on top)."""
        if not self._overlay_stack:
            return lines
        result = list(lines)

        # Pre-render all visible overlays and calculate positions
        rendered: list[dict] = []
        min_lines_needed = len(result)

        visible_entries = [entry for entry in self._overlay_stack if self._is_overlay_visible(entry)]
        visible_entries.sort(key=lambda entry: entry.focus_order)
        for entry in visible_entries:
            component = entry.component
            options = entry.options

            # Get layout with height=0 first to determine width and maxHeight
            # (width and maxHeight don't depend on overlay height)
            first_layout = self._resolve_overlay_layout(options, 0, term_width, term_height)
            width = first_layout["width"]
            max_height = first_layout["maxHeight"]

            # Render component at calculated width
            overlay_lines = component.render(width)

            # Apply maxHeight if specified
            if max_height is not None and len(overlay_lines) > max_height:
                overlay_lines = overlay_lines[:max_height]

            # Get final row/col with actual overlay height
            layout = self._resolve_overlay_layout(options, len(overlay_lines), term_width, term_height)

            rendered.append({"overlayLines": overlay_lines, "row": layout["row"], "col": layout["col"], "w": width})
            min_lines_needed = max(min_lines_needed, layout["row"] + len(overlay_lines))

        # Pad to at least terminal height so overlays have screen-relative positions.
        # Excludes max_lines_rendered: the historical high-water mark caused
        # self-reinforcing inflation that pushed content into scrollback on
        # terminal widen.
        working_height = max(len(result), term_height, min_lines_needed)

        # Extend result with empty lines if content is too short for overlay placement or working area
        while len(result) < working_height:
            result.append("")

        viewport_start = max(0, working_height - term_height)

        # Composite each overlay
        for item in rendered:
            overlay_lines = item["overlayLines"]
            row = item["row"]
            col = item["col"]
            w = item["w"]
            for i, overlay_line in enumerate(overlay_lines):
                idx = viewport_start + row + i
                if 0 <= idx < len(result):
                    # Defensive: truncate overlay line to declared width before compositing
                    # (components should already respect width, but this ensures it)
                    truncated_overlay_line = (
                        slice_by_column(overlay_line, 0, w, True) if visible_width(overlay_line) > w else overlay_line
                    )
                    result[idx] = self._composite_line_at(result[idx], truncated_overlay_line, col, w, term_width)

        return result

    def _apply_line_resets(self, lines: list[str]) -> list[str]:
        reset = SEGMENT_RESET
        for i, line in enumerate(lines):
            if not is_image_line(line):
                lines[i] = normalize_terminal_output(line) + reset
        return lines

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

    def _composite_line_at(
        self, base_line: str, overlay_line: str, start_col: int, overlay_width: int, total_width: int
    ) -> str:
        """Splice overlay content into a base line at a specific column. Single-pass optimized."""
        if is_image_line(base_line):
            return base_line

        # Single pass through base_line extracts both before and after segments
        after_start = start_col + overlay_width
        base = extract_segments(base_line, start_col, after_start, total_width - after_start, True)

        # Extract overlay with width tracking (strict=True to exclude wide chars at boundary)
        overlay_text, overlay_actual_width = slice_with_width(overlay_line, 0, overlay_width, True)

        # Pad segments to target widths
        before_pad = max(0, start_col - base["beforeWidth"])
        overlay_pad = max(0, overlay_width - overlay_actual_width)
        actual_before_width = max(start_col, base["beforeWidth"])
        actual_overlay_width = max(overlay_width, overlay_actual_width)
        after_target = max(0, total_width - actual_before_width - actual_overlay_width)
        after_pad = max(0, after_target - base["afterWidth"])

        # Compose result
        r = SEGMENT_RESET
        result = (
            base["before"]
            + " " * before_pad
            + r
            + overlay_text
            + " " * overlay_pad
            + r
            + base["after"]
            + " " * after_pad
        )

        # CRITICAL: Always verify and truncate to terminal width.
        # This is the final safeguard against width overflow which would crash the TUI.
        # Width tracking can drift from actual visible width due to:
        # - Complex ANSI/OSC sequences (hyperlinks, colors)
        # - Wide characters at segment boundaries
        # - Edge cases in segment extraction
        result_width = visible_width(result)
        if result_width <= total_width:
            return result
        # Truncate with strict=True to ensure we don't exceed total_width
        return slice_by_column(result, 0, total_width, True)

    def _extract_cursor_position(self, lines: list[str], height: int) -> dict | None:
        """Find and extract cursor position from rendered lines.

        Searches for CURSOR_MARKER, calculates its position, and strips it
        from the output. Only scans the bottom terminal-height lines (visible
        viewport). Returns ``{"row": int, "col": int}`` or None.
        """
        viewport_top = max(0, len(lines) - height)
        for row in range(len(lines) - 1, viewport_top - 1, -1):
            line = lines[row]
            marker_index = line.find(CURSOR_MARKER)
            if marker_index != -1:
                # Calculate visual column (width of text before marker)
                before_marker = line[:marker_index]
                col = visible_width(before_marker)

                # Strip marker from the line
                lines[row] = line[:marker_index] + line[marker_index + len(CURSOR_MARKER) :]

                return {"row": row, "col": col}
        return None

    # ------------------------------------------------------------------
    # Differential rendering
    # ------------------------------------------------------------------

    def _do_render(self) -> None:  # noqa: C901
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
        def full_render(clear: bool) -> None:
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
            self.terminal.write(buffer)
            self._cursor_row = max(0, len(new_lines) - 1)
            self._hardware_cursor_row = self._cursor_row
            # Reset max lines when clearing, otherwise track growth
            if clear:
                self._max_lines_rendered = len(new_lines)
            else:
                self._max_lines_rendered = max(self._max_lines_rendered, len(new_lines))
            buffer_length = max(height, len(new_lines))
            self._previous_viewport_top = max(0, buffer_length - height)
            self._position_hardware_cursor(cursor_pos, len(new_lines))
            self._previous_lines = new_lines
            self._previous_kitty_image_ids = self._collect_kitty_image_ids(new_lines)
            self._previous_width = width
            self._previous_height = height

        debug_redraw = os.environ.get("PIDREI_DEBUG_REDRAW") == "1"

        def log_redraw(reason: str) -> None:
            if not debug_redraw:
                return
            log_path = os.path.join(self._log_directory, "pidrei-debug.log")
            timestamp = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            msg = (
                f"[{timestamp}] fullRender: {reason} "
                f"(prev={len(self._previous_lines)}, new={len(new_lines)}, height={height})\n"
            )
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(msg)

        # First render - just output everything without clearing (assumes clean screen)
        if not self._previous_lines and not width_changed and not height_changed:
            log_redraw("first render")
            full_render(False)
            return

        # Width changes always need a full re-render because wrapping changes.
        if width_changed:
            log_redraw(f"terminal width changed ({self._previous_width} -> {width})")
            full_render(True)
            return

        # Height changes normally need a full re-render to keep the visible viewport aligned,
        # but Termux changes height when the software keyboard shows or hides.
        # In that environment, a full redraw causes the entire history to replay on every toggle.
        if height_changed and not _is_termux_session():
            log_redraw(f"terminal height changed ({self._previous_height} -> {height})")
            full_render(True)
            return

        # Content shrunk below the working area and no overlays - re-render to clear empty rows
        # (overlays need the padding, so only do this when no overlays are active)
        if self._clear_on_shrink and len(new_lines) < self._max_lines_rendered and not self._overlay_stack:
            log_redraw(f"clearOnShrink (maxLinesRendered={self._max_lines_rendered})")
            full_render(True)
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
            self._position_hardware_cursor(cursor_pos, len(new_lines))
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
                    log_redraw(f"deleted lines moved viewport up ({target_row} < {prev_viewport_top})")
                    full_render(True)
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
                    log_redraw(f"extraLines > height ({extra_lines} > {height})")
                    full_render(True)
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
                self.terminal.write(buffer)
                self._cursor_row = target_row
                self._hardware_cursor_row = target_row
            self._position_hardware_cursor(cursor_pos, len(new_lines))
            self._previous_lines = new_lines
            self._previous_kitty_image_ids = self._collect_kitty_image_ids(new_lines)
            self._previous_width = width
            self._previous_height = height
            self._previous_viewport_top = prev_viewport_top
            return

        # Differential rendering can only touch what was actually visible.
        # If the first changed line is above the previous viewport, we need a full redraw.
        if first_changed < prev_viewport_top:
            log_redraw(f"firstChanged < viewportTop ({first_changed} < {prev_viewport_top})")
            full_render(True)
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
                    log_redraw(
                        f"kitty image pre-clear would scroll ({image_start_screen_row} + {image_reserved_rows} > {height})"
                    )
                    full_render(True)
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
                crash_log_path = os.path.join(self._log_directory, "pidrei-crash.log")
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
                os.makedirs(os.path.dirname(crash_log_path), exist_ok=True)
                with open(crash_log_path, "w", encoding="utf-8") as crash_file:
                    crash_file.write(crash_data)

                # Terminal cleanup happens in the caller's shutdown path; pi
                # calls the sync stop() here, but stop() is async in the port
                # and _do_render runs inside the render loop task.
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
            with open(debug_path, "w", encoding="utf-8") as debug_file:
                debug_file.write(debug_data)

        # Write entire buffer at once
        self.terminal.write(buffer)

        # Track cursor position for next render
        # cursor_row tracks end of content (for viewport calculation)
        # hardware_cursor_row tracks actual terminal cursor position (for movement)
        self._cursor_row = max(0, len(new_lines) - 1)
        self._hardware_cursor_row = final_cursor_row
        # Track terminal's working area (grows but doesn't shrink unless cleared)
        self._max_lines_rendered = max(self._max_lines_rendered, len(new_lines))
        self._previous_viewport_top = max(prev_viewport_top, final_cursor_row - height + 1)

        # Position hardware cursor for IME
        self._position_hardware_cursor(cursor_pos, len(new_lines))

        self._previous_lines = new_lines
        self._previous_kitty_image_ids = self._collect_kitty_image_ids(new_lines)
        self._previous_width = width
        self._previous_height = height

    def _position_hardware_cursor(self, cursor_pos: dict | None, total_lines: int) -> None:
        """Position the hardware cursor for IME candidate window."""
        if not cursor_pos or total_lines <= 0:
            self.terminal.hide_cursor()
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

        if buffer:
            self.terminal.write(buffer)

        self._hardware_cursor_row = target_row
        if self._show_hardware_cursor:
            self.terminal.show_cursor()
        else:
            self.terminal.hide_cursor()

    # ------------------------------------------------------------------
    # Terminal queries
    # ------------------------------------------------------------------

    async def query_terminal_background_color(self, *, timeout_ms: float):
        """Query the terminal's default background color with OSC 11 (``ESC ] 11 ; ? BEL``).

        Returns the parsed RGB record, or None if it times out or fails to
        parse.
        """
        query = _PendingOsc11Query()
        with self._query_lock:
            self._pending_osc11_queries.append(query)
            self._pending_osc11_replies += 1
        self.terminal.write("\x1b]11;?\x07")

        await query.event.wait(timeout_ms / 1000)
        with self._query_lock:
            if not query.settled:
                # Timed out; a late reply is still consumed silently
                # (pi keeps the pending-reply bookkeeping in place too).
                query.settled = True
                return None
            return query.result

    async def query_terminal_color_scheme(self, *, timeout_ms: float):
        """Query the terminal's color-scheme preference with DSR (``CSI ? 996 n``).

        Terminals that support the color palette notification protocol reply
        with ``CSI ? 997 ; 1 n`` for dark or ``CSI ? 997 ; 2 n`` for light.
        """
        result: dict[str, Any] = {"scheme": None}
        event = tonio.Event()

        def settle(scheme) -> None:
            with self._query_lock:
                if event.is_set():
                    return
                result["scheme"] = scheme
                event.set()

        unsubscribe = self.on_terminal_color_scheme_change(settle)
        try:
            self.terminal.write("\x1b[?996n")
            await event.wait(timeout_ms / 1000)
            return result["scheme"]
        finally:
            unsubscribe()


__all__ = [  # noqa: RUF022
    "CURSOR_MARKER",
    "Component",
    "Container",
    "Focusable",
    "OverlayHandle",
    "TUI",
    "is_focusable",
    "visible_width",
]
