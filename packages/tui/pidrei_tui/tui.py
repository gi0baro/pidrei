"""Mirror of pi tui src/tui.ts.

Shared TUI machinery: the component/overlay model and ``TuiBase``, the
renderer-independent half of the TUI. The two renderers live next door —
``tui_main_screen.TuiMainScreen`` (differential rendering into the terminal's
main screen and scrollback) and ``tui_alt_screen.TuiAltScreen`` (an
application-owned viewport on the alternate screen).

Port deviations (documented once here):

- pi's ``TUI`` is a structural interface implemented by both renderers; the
  Python stand-in for that annotation is ``TuiBase`` itself, re-exported under
  the name ``TUI``. Construct a renderer, never ``TUI``.

- Render coalescing: pi chains ``process.nextTick`` + a 16ms ``setTimeout``
  throttle; here ``start()`` spawns a single render-loop task that parks on
  an Event, applies the same 16ms throttle against the end of the last
  frame, and calls ``_do_render``. ``request_render()`` stays sync (callable
  from input handlers); ``force=True`` resets the differential state and
  skips the throttle, and keyboard input takes the same throttle-skipping
  path without the reset (pi cancels its render timer for both).
- Frame output is a two-stage pipeline: ``_do_render`` computes a frame and
  hands its bytes to a writer task over a one-slot channel (``_emit``), so
  the next frame's compute overlaps the previous frame's trip to the
  terminal while a slow link (SSH) still paces the loop — the loop can be
  at most one frame ahead of the wire. pi writes synchronously from the
  same thread; the ordering that gives it is kept by routing every write
  the renderers make through ``_emit`` and by ``render_now``/``stop``
  draining the pipeline (``_flush_frames``) before anything else goes out.
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

import functools
import math
import os
import re
import threading
import time as _time
from abc import ABC, abstractmethod
from typing import Any, Protocol

import tonio.colored as tonio
from tonio.colored.sync import channel

from ._owner import OwnerTask
from ._timers import get_ui_owner, set_ui_owner
from .keys import is_key_release, matches_key
from .terminal_colors import (
    is_osc11_background_color_response,
    parse_osc11_background_color,
    parse_terminal_color_scheme_report,
)
from .terminal_image import get_capabilities, is_image_line, set_cell_dimensions
from .utils import extract_segments, normalize_terminal_output, slice_by_column, slice_with_width, visible_width


_CELL_SIZE_RESPONSE_RE = re.compile(r"^\x1b\[6;(\d+);(\d+)t$")
_PERCENT_RE = re.compile(r"^(\d+(?:\.\d+)?)%$")


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

    def set_children(self, children: list) -> None:
        """Replace the children in one step.

        `clear()` followed by `add_child()` calls is fine on pi's single event
        loop, where nothing can render in between. Here a component can be
        rebuilt from a spawned task while the render loop reads it on another
        thread, and the half-populated window renders as a component that has
        briefly vanished. Rebuilding into a local list and publishing it with
        one assignment closes that window: a concurrent `render` sees either
        the old children or the new ones.
        """
        self.children = list(children)

    def invalidate(self) -> None:
        for child in self.children:
            invalidate = getattr(child, "invalidate", None)
            if invalidate is not None:
                invalidate()

    def render(self, width: int) -> list[str]:
        lines: list[str] = []
        # Bind once: `set_children` publishes a new list, so a rebuild landing
        # mid-render cannot make this iteration see a partial one.
        for child in self.children:
            lines.extend(child.render(width))
        return lines


_MIN_RENDER_INTERVAL_S = 0.016

SEGMENT_RESET = "\x1b[0m\x1b]8;;\x07"


def composite_tui_line(base_line: str, overlay_line: str, start_col: int, overlay_width: int, total_width: int) -> str:
    """Composite overlay content into a terminal line at a fixed column."""
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
        base["before"] + " " * before_pad + r + overlay_text + " " * overlay_pad + r + base["after"] + " " * after_pad
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


# pi brands the viewport renderer with `Symbol.for(...)`; the Python stand-in
# is an attribute name nothing else would define. A ViewportTUI also has
# `set_layout_root(component | None)`.
VIEWPORT_TUI = "__pidrei_tui_viewport__"


def is_viewport_tui(tui) -> bool:
    return getattr(tui, VIEWPORT_TUI, False) is True


# TuiMode: "regular" (main screen) | "fullscreen" (alternate screen).


class TuiBase(Container, ABC):
    """Renderer-independent half of the TUI: focus, overlays, input, queries.

    Subclasses own the frame: they implement ``_do_render`` and may hook the
    terminal lifecycle through ``_before_terminal_start`` / ``_after_terminal_start``
    / ``_before_terminal_stop`` / ``_after_terminal_stop`` and reset their
    differential state in ``_reset_render_state``.
    """

    def __init__(self, terminal, show_hardware_cursor: bool | None = None, log_directory: str | None = None) -> None:
        super().__init__()
        self.terminal = terminal
        self._log_directory = (
            log_directory
            if log_directory is not None
            else os.environ.get("PIDREI_CODING_AGENT_DIR") or os.path.join(os.path.expanduser("~"), ".pidrei", "agent")
        )
        self._focused_component = None
        self._input_listeners: list = []

        # Global callback for debug key (Shift+Ctrl+D). Called before input is
        # forwarded to the focused component.
        self.on_debug = None

        self._render_signal = tonio.Event()
        self._render_immediate_signal = tonio.Event()
        self._render_force = False
        self._render_immediate = False
        self._render_force_lock = threading.Lock()
        self._pre_render_callbacks: list = []
        self._last_render_at = 0.0
        self._line_reset_memo: dict[str, str] = {}
        self._render_scope = None
        # Frame pipeline (see the module docstring): the sender side of the
        # one-slot channel while the writer task runs, else None.
        self._writer_scope = None
        self._frames: Any = None
        self._frame_parts: list[str] = []
        self._frame_writer_error: BaseException | None = None
        # Async callback invoked when a frame raises; see `_render_loop`.
        self._render_error_handler = None
        # The task that owns UI state (pi: the JS thread). A ProcessTerminal
        # brings its own — the stdin pump and the input timers already run on
        # it; for any other terminal the TUI runs one (`_owner_scope`) and
        # routes the terminal's input through it. Components post work that
        # mutates state from elsewhere (an autocomplete result, a timer) here.
        self.input_owner: OwnerTask = getattr(terminal, "input_owner", None) or OwnerTask()
        self.input_owner.on_error = self._handle_owner_error
        self._owner_scope = None
        self._show_hardware_cursor = os.environ.get("PIDREI_HARDWARE_CURSOR") == "1"
        # Clear empty rows when content shrinks (default: off)
        self._clear_on_shrink = os.environ.get("PIDREI_CLEAR_ON_SHRINK") == "1"
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

    # ------------------------------------------------------------------
    # Renderer hooks
    # ------------------------------------------------------------------

    @abstractmethod
    async def _do_render(self) -> None: ...

    def _reset_render_state(self) -> None:
        """Drop the differential state so the next frame repaints everything."""

    async def _before_terminal_start(self) -> None: ...

    async def _after_terminal_start(self) -> None: ...

    async def _before_terminal_stop(self, options: dict) -> None: ...

    async def _after_terminal_stop(self, options: dict) -> None: ...

    @property
    def has_overlay_entries(self) -> bool:
        return bool(self._overlay_stack)

    @property
    def full_redraws(self) -> int:
        return self._full_redraw_count

    def get_show_hardware_cursor(self) -> bool:
        return self._show_hardware_cursor

    def set_show_hardware_cursor(self, enabled: bool) -> None:
        if self._show_hardware_cursor == enabled:
            return
        self._show_hardware_cursor = enabled
        # pi hides the cursor right here. Every frame ends by emitting the
        # cursor state (`_position_hardware_cursor` / the alt-screen frame
        # tail), so the render loop stays the only task writing terminal
        # bytes; the requested render applies the change.
        self.request_render()

    def get_clear_on_shrink(self) -> bool:
        return self._clear_on_shrink

    def set_render_error_handler(self, handler) -> None:
        """Install the async callback that receives a render-loop exception.

        Without one, the exception propagates out of the render task (and
        the TUI looks frozen until `stop()`), so owners should install one.
        """
        self._render_error_handler = handler

    def set_clear_on_shrink(self, enabled: bool) -> None:
        """Set whether to trigger full re-render when content shrinks.

        When enabled, empty rows are cleared when content shrinks. When
        disabled, empty rows remain (reduces redraws on slower terminals).
        """
        self._clear_on_shrink = enabled

    # ------------------------------------------------------------------
    # Focus and overlay focus-restore machinery
    # ------------------------------------------------------------------

    def get_focused_component(self):
        return self._focused_component

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

    def _get_mounted_roots(self) -> list:
        return self.children

    def _is_component_mounted(self, component) -> bool:
        return any(self._contains_component(child, component) for child in self._get_mounted_roots())

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
        # No direct `hide_cursor()` here or in the hide paths below: the
        # render loop emits the cursor state with every frame (see
        # `set_show_hardware_cursor`).
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
        self.request_render()

    def has_overlay(self) -> bool:
        """Check if there are any visible overlays."""
        return any(self._is_overlay_visible(entry) for entry in self._overlay_stack)

    def _is_overlay_focused(self) -> bool:
        """Check if the focused component is a visible overlay."""
        return any(
            entry.component is self._focused_component and self._is_overlay_visible(entry)
            for entry in self._overlay_stack
        )

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
        for root in self._get_mounted_roots():
            root.invalidate()
        for overlay in self._overlay_stack:
            invalidate = getattr(overlay.component, "invalidate", None)
            if invalidate is not None:
                invalidate()

    # ------------------------------------------------------------------
    # Lifecycle and render scheduling
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._stopped = False
        await self._before_terminal_start()
        on_input = self._handle_input
        if getattr(self.terminal, "input_owner", None) is not self.input_owner:
            # The terminal does not run the owner: do it here and put the
            # terminal's input on it.
            self._owner_scope = tonio.scope()
            await self._owner_scope.__aenter__()
            self.input_owner.start(self._owner_scope)

            async def on_input(data: str) -> None:
                await self.input_owner.run(functools.partial(self._handle_input, data))

        await self.terminal.start(on_input, self.request_render)
        set_ui_owner(self.input_owner)
        await self._after_terminal_start()
        self.terminal.hide_cursor()
        if self._color_scheme_notifications_enabled:
            await self.terminal.write("\x1b[?2031h")
        await self._query_cell_size()
        self._frames, receiver = channel.channel(1)
        self._frame_writer_error = None
        self._writer_scope = tonio.scope()
        await self._writer_scope.__aenter__()
        self._writer_scope.spawn(self._frame_writer(receiver))
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

    async def set_terminal_color_scheme_notifications(self, enabled: bool) -> None:
        if self._color_scheme_notifications_enabled == enabled:
            return
        self._color_scheme_notifications_enabled = enabled
        if not self._stopped:
            await self.terminal.write("\x1b[?2031h" if enabled else "\x1b[?2031l")

    async def _query_cell_size(self) -> None:
        # Only query if terminal supports images (cell size is only used for image rendering)
        if not get_capabilities()["images"]:
            return
        # Query terminal for cell size in pixels: CSI 16 t
        # Response format: CSI 6 ; height ; width t
        await self.terminal.write("\x1b[16t")

    async def stop(self, options: dict | None = None) -> None:
        """``options`` mirrors pi's ``TuiStopOptions`` (``{"preserveScreen"?}``).

        ``preserveScreen`` leaves the renderer's output on the terminal for
        another TUI taking the same terminal over (the runtime UI-mode switch).
        """
        options = options or {}
        self._stopped = True
        self._render_immediate_signal.set()
        self._render_signal.set()
        if self._render_scope is not None:
            await self._render_scope.__aexit__(None, None, None)
            self._render_scope = None
        if self._writer_scope is not None:
            # After the loop: what it handed over still goes out, then the
            # writer stops and everything below writes in order behind it.
            frames, self._frames = self._frames, None
            await frames.send(None)
            await self._writer_scope.__aexit__(None, None, None)
            self._writer_scope = None
        if self._color_scheme_notifications_enabled:
            await self.terminal.write("\x1b[?2031l")
        await self._before_terminal_stop(options)
        self.terminal.show_cursor()
        if get_ui_owner() is self.input_owner:
            set_ui_owner(None)
        await self.terminal.stop()
        if self._owner_scope is not None:
            # After the terminal: its input no longer arrives, so the queued
            # work drains and the owner's timers are reaped with the scope.
            self.input_owner.close()
            await self._owner_scope.__aexit__(None, None, None)
            self._owner_scope = None
        await self._after_terminal_stop(options)

    def _handle_owner_error(self, error: BaseException) -> None:
        """Posted UI work (a timer tick, an autocomplete result) raised.

        pi would crash on it; the render loop's handler gets it here too, so
        the owner keeps serving input instead of dying with the exception
        surfacing only at `stop()`.
        """
        handler = self._render_error_handler
        if handler is None:
            raise error
        tonio.spawn.without_tracking(handler(error))

    async def render_now(self, force: bool = False) -> None:
        """Render one frame synchronously, bypassing the loop and its throttle."""
        if force:
            self._reset_render_state()
        with self._render_force_lock:
            self._render_force = False
            self._render_immediate = False
        self._render_signal.clear()
        self._render_immediate_signal.clear()
        self._run_pre_render_callbacks()
        await self._render_frame()
        await self._flush_frames()

    async def _render_frame(self) -> None:
        """One frame: compute, hand the bytes over, stamp the throttle clock."""
        try:
            await self._do_render()
        except BaseException:
            self._frame_parts = []  # a torn frame is never written
            raise
        await self._send_frame()
        # Throttle from the end of the frame (pi measures from its start) so
        # a frame that takes longer than the interval is still followed by a
        # pause instead of pinning the worker for as long as updates keep
        # coming.
        self._last_render_at = _time.monotonic()

    def _emit(self, data: str) -> None:
        """Add to the frame being rendered; it goes out as one write when `_do_render` returns."""
        self._frame_parts.append(data)

    async def _send_frame(self) -> None:
        """Hand the finished frame to the writer; wait only while it is a frame behind."""
        parts = self._frame_parts
        if not parts:
            return
        self._frame_parts = []
        data = "".join(parts)
        frames = self._frames
        if frames is None:
            await self.terminal.write(data)
            return
        await frames.send(data)
        if self._frame_writer_error is not None:
            raise self._frame_writer_error

    async def _flush_frames(self) -> None:
        """Return once every frame handed over so far is on the wire."""
        frames = self._frames
        if frames is None:
            return
        done = tonio.Event()
        await frames.send(done)
        await done.wait(None)
        if self._frame_writer_error is not None:
            raise self._frame_writer_error

    async def _frame_writer(self, receiver) -> None:
        while True:
            item = await receiver.receive()
            if item is None:
                return
            if isinstance(item, tonio.Event):
                item.set()
                continue
            if self._frame_writer_error is not None:
                continue  # the terminal is gone; the loop learns on its next _emit
            try:
                await self.terminal.write(item)
            except Exception as error:
                self._frame_writer_error = error

    def post_before_render(self, callback) -> None:
        """Run `callback` on the render task right before the next frame.

        For mutations of the whole component tree (e.g. a theme reload's
        `invalidate()`) that originate off the render task — running them
        here means they never overlap a frame being rendered.
        """
        with self._render_force_lock:
            self._pre_render_callbacks.append(callback)
        self.request_render()

    def _run_pre_render_callbacks(self) -> None:
        with self._render_force_lock:
            callbacks = self._pre_render_callbacks
            self._pre_render_callbacks = []
        for callback in callbacks:
            callback()

    def request_render(self, force: bool = False) -> None:
        # pi calls resetRenderState() right here. That is safe on one thread
        # and a data race on this runtime: `_do_render`'s tail writes the
        # previous-frame state after the frame goes out, so a force landing in
        # that window had its resets clobbered and the next render diffed
        # identical lines and wrote nothing. Only the render loop calls
        # `_reset_render_state` now; a force is just a flag it consumes.
        if force:
            with self._render_force_lock:
                self._render_force = True
                self._render_immediate = True
            self._render_immediate_signal.set()
        self._render_signal.set()

    def _request_immediate_render(self) -> None:
        """Render on the next loop turn, skipping the throttle (not the diff).

        pi's counterpart cancels the throttled `setTimeout`; here the throttle
        is a wait the immediate signal cuts short.
        """
        with self._render_force_lock:
            self._render_immediate = True
        self._render_immediate_signal.set()
        self._render_signal.set()

    async def _render_loop(self) -> None:
        while True:
            await self._render_signal.wait(None)
            if self._stopped:
                return
            with self._render_force_lock:
                force = self._render_force
                self._render_force = False
                immediate = self._render_immediate
                self._render_immediate = False
            if not immediate:
                elapsed = _time.monotonic() - self._last_render_at
                delay = _MIN_RENDER_INTERVAL_S - elapsed
                if delay > 0:
                    # Keyboard input is latency-sensitive: an immediate request
                    # arriving mid-throttle ends the wait instead of queueing
                    # behind it.
                    await self._render_immediate_signal.wait(delay)
                    if self._stopped:
                        return
                    with self._render_force_lock:
                        force = force or self._render_force
                        self._render_force = False
                        self._render_immediate = False
            self._render_immediate_signal.clear()
            # Absorb requests that arrived up to this point; requests during
            # _do_render re-arm the loop (pi: renderRequested re-check).
            self._render_signal.clear()
            # stop() sets _stopped and then the render signal; if that lands
            # between this loop's wakeup and the clear() above, the wakeup is
            # consumed here and the next wait() would block forever with
            # stop() stuck awaiting this task. Re-check after clearing.
            if self._stopped:
                return
            if force:
                self._reset_render_state()
            try:
                self._run_pre_render_callbacks()
                await self._render_frame()
            except Exception as error:
                # pi crashes the process on a render throw. Here the loop is a
                # scope child: letting the exception escape would only surface
                # at `stop()`, leaving a frozen UI with a live agent. Hand it
                # to the owner (interactive mode's crash handler) instead.
                handler = self._render_error_handler
                if handler is None:
                    raise
                self._stopped = True
                # Detached on purpose: the handler typically calls `stop()`,
                # which joins this task through the render scope.
                tonio.spawn.without_tracking(handler(error))
                return
            # A force that arrived after the consume above lost its signal to
            # the clear(); without this it would sit unserved until the next
            # unrelated request. Re-arming guarantees every force ends in a
            # full_render, which always writes a frame.
            if self._render_force:
                self._render_signal.set()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    async def _handle_input(self, data: str) -> None:
        if self._consume_osc11_background_response(data):
            return
        if await self._consume_terminal_color_scheme_report(data):
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
            await handle(data)
            # Keyboard input is latency-sensitive; skip the throttled path.
            self._request_immediate_render()

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

    async def _consume_terminal_color_scheme_report(self, data: str) -> bool:
        scheme = parse_terminal_color_scheme_report(data)
        if not scheme:
            return False

        for listener in list(self._color_scheme_listeners):
            # Listeners are awaitable-returning (async-only policy): reacting
            # to a scheme change can mean loading a theme from disk.
            await listener(scheme)
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
        # Memoized across frames: a line the previous frame already
        # normalized (almost all of them — components cache their output)
        # is a dict hit on its cached hash instead of a regex pass. The memo
        # is rebuilt from this frame's lines, so it holds exactly one frame.
        reset = SEGMENT_RESET
        previous = self._line_reset_memo
        memo: dict[str, str] = {}
        for i, line in enumerate(lines):
            finished = previous.get(line)
            if finished is None:
                finished = line if is_image_line(line) else normalize_terminal_output(line) + reset
            memo[line] = finished
            lines[i] = finished
        self._line_reset_memo = memo
        return lines

    def _composite_line_at(
        self, base_line: str, overlay_line: str, start_col: int, overlay_width: int, total_width: int
    ) -> str:
        return composite_tui_line(base_line, overlay_line, start_col, overlay_width, total_width)

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
        await self.terminal.write("\x1b]11;?\x07")

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

        async def settle(scheme) -> None:
            with self._query_lock:
                if event.is_set():
                    return
                result["scheme"] = scheme
                event.set()

        unsubscribe = self.on_terminal_color_scheme_change(settle)
        try:
            await self.terminal.write("\x1b[?996n")
            await event.wait(timeout_ms / 1000)
            return result["scheme"]
        finally:
            unsubscribe()


# pi's `TUI` is a structural interface both renderers implement; annotations
# that say `TUI` there say `TuiBase` here. Kept as a name so call sites read
# the same — it is not constructible (the renderers are).
TUI = TuiBase


__all__ = [  # noqa: RUF022
    "CURSOR_MARKER",
    "Component",
    "Container",
    "Focusable",
    "OverlayHandle",
    "TUI",
    "TuiBase",
    "composite_tui_line",
    "is_focusable",
    "visible_width",
]
