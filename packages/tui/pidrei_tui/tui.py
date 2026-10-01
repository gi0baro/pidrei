"""Mirror of pi tui src/tui.ts.

Shared TUI machinery: the component/overlay model and ``TuiBase``, the
renderer-independent half of the TUI. The two renderers live next door —
``tui_main_screen.TuiMainScreen`` (differential rendering into the terminal's
main screen and scrollback) and ``tui_alt_screen.TuiAltScreen`` (an
application-owned viewport on the alternate screen).

Concurrency contract (spec/ui-island.md): the UI is passive state plus
one guard, and independent loops work on it in parallel.

- **The UI state lock** (``state_lock``, a reentrant thread lock with the
  terminal's lifetime, shared by every TUI on it) guards all component and
  TUI state. Every mutation or read of it is a short synchronous section
  under the lock, from whatever task it runs on: input handling, a frame's
  tree walk, an agent event, a flow's stretch, a timer callback, a
  component's spawned work. **Nothing awaits while holding it.** A helper
  that mutates takes it itself (it is reentrant).
- **Input**: the terminal's reader reads ahead into a channel, and one
  consumer hands each item to ``_handle_input``, which routes it under the
  lock. Handlers are synchronous; work whose effect the next key must see
  is spawned and registered with ``finish_before_next_input``.
- **Rendering**: ``request_render()`` is sync, lock-free and callable from
  anywhere (a component's render included); the render loop draws one frame
  per request, holding the per-TUI render lock for the frame and the state
  lock for the tree walk only.
- **Timers** (``_timers``) know nothing about the UI: their callbacks guard
  what they touch.
- **Errors** nothing up the stack can take (the input consumer's, the
  output pump's, a terminal event's, a component's spawned work) go to
  ``report_error`` and the installed handler.

- **Grouping**: ``apply(fn)`` runs a synchronous ``fn`` under the lock and
  refuses coroutines, so nothing can be awaited inside; it is the primitive
  component code (and extensions, through their guarded wrapper) uses to
  change state from a timer or spawned work.

Publication idioms that remain (``set_children``, atomic cache tuples) are
hygiene and efficiency, not correctness.

Port deviations (documented once here):

- pi's ``TUI`` is a structural interface implemented by both renderers; the
  Python stand-in for that annotation is ``TuiBase`` itself, re-exported under
  the name ``TUI``. Construct a renderer, never ``TUI``.

- Render scheduling (spec/ui-island.md, "Rendering"): pi chains ``process.nextTick``
  + a 16ms ``setTimeout`` throttle; here a render loop (one task per start)
  receives requests from a one-slot channel and draws one frame per
  request. ``request_render()`` stays sync and touches no lock: it sends
  with ``send_nowait``, and a full channel means a request is already
  pending, so the new one is covered and dropped. The loop receives before
  it walks the tree, so a change made after the receive queues exactly one
  more frame. There is no throttle: while a frame runs, requests collapse
  into the pending one, and the frame writer paces a slow terminal. pi's
  keyboard fast path (skip the throttle) goes with it. ``force=True``
  (drop the differential state: repaint everything) is a field of the
  request object every requester sends until the loop receives it: a
  forced request sets it before its send, so a dropped send still lands in
  the pending request. The loop installs a fresh object before reading it.
- A frame is two halves: ``_compose_frame`` walks the tree and publishes
  what input reads back (layout, overlay bounds) under the UI state lock;
  ``_write_frame`` diffs and emits under the render lock only.
- Frame output is a two-stage pipeline: ``_write_frame`` hands its bytes to
  a writer task over a one-slot channel (``_emit``), so the next frame's
  compute overlaps the previous frame's
  trip to the terminal while a slow link (SSH) still paces rendering — at
  most one frame ahead of the wire. pi writes synchronously from the
  same thread; the ordering that gives it is kept by routing every write
  the renderers make through ``_emit`` and by ``render_now``/``stop``
  draining the pipeline (``_flush_frames``) before anything else goes out.
- ``start``/``stop`` are async (they drive the async terminal driver and the
  render/writer lifecycle). ``stop()`` closes the request channel, waits for
  the render loop to finish its frame, and drains the writer — no task
  abort involved.
- ``query_terminal_colors`` does not resolve with the colors: every report
  (the query completing, a timeout's partial result, a late reply) is a
  message on the terminal-event loop, whose one consumer hands it to the
  ``on_terminal_colors`` listeners in arrival order (pi's promise resolve and
  ``onLateReply`` callback, which apply the colors from three places at once).
  The call registers the query and returns an event the loop sets once the
  listeners handled the query's first report; the burst write and the
  timeout run on their own coroutine. Pending-query transitions and report
  sends take a sync lock because the terminal's input reader and that
  coroutine may run on different tonio workers.
- Input (spec/ui-island.md, "Input"): the terminal hands terminal replies to
  ``_consume_terminal_reply`` from its reader, ahead of the input order, so
  a query is answered even while input handling waits on the work that
  asked; colour reports and colour-scheme reports go to their own loop (a
  theme load stays off the key path). Every other item reaches
  ``_handle_input`` from the terminal's one input consumer (or, for a
  terminal without one, from its caller), one at a time.
- ``CURSOR_MARKER`` is an APC sequence pi brands "pi:c" — renamed to
  "pidrei:c" (pi naming itself).
- Env renames: PI_TUI_DEBUG_REDRAW → PIDREI_TUI_DEBUG_REDRAW, PI_TUI_DEBUG →
  PIDREI_TUI_DEBUG; log files pidrei-tui-debug.log / pidrei-tui-crash.log.
  The renderer reads no coding-agent configuration: the hardware cursor,
  clear-on-shrink and the log directory are constructor/setter inputs (pi
  maps PI_HARDWARE_CURSOR through its settings manager before creating it).
"""

import inspect
import math
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

import tonio.colored as tonio
from tonio.colored import sync
from tonio.colored.sync import channel

from .keys import is_key_release, matches_key
from .terminal import ProcessTerminal
from .terminal_colors import parse_osc_color_response, parse_terminal_color_scheme_report
from .terminal_image import get_capabilities, is_image_line, set_cell_dimensions
from .utils import extract_segments, normalize_terminal_output, slice_by_column, slice_with_width, visible_width


_CELL_SIZE_RESPONSE_RE = re.compile(r"^\x1b\[6;(\d+);(\d+)t$")
_PERCENT_RE = re.compile(r"^(\d+(?:\.\d+)?)%$")


type TuiMouseEventType = Literal["press", "release", "move", "drag", "click", "wheel"]
type TuiMouseButton = Literal["left", "middle", "right", "none"]


@dataclass(slots=True, frozen=True)
class TuiMouseEvent:
    """Normalized cell-based mouse event. Coordinates are zero-based."""

    type: TuiMouseEventType
    button: TuiMouseButton
    # Coordinates local to the receiving component.
    x: int
    y: int
    # Absolute terminal coordinates.
    screen_x: int
    screen_y: int
    # Current component bounds.
    width: int
    height: int
    shift: bool = False
    alt: bool = False
    ctrl: bool = False
    # Logical lines. Negative values scroll up.
    wheel_delta: int | None = None
    # Consecutive click count when type is click.
    click_count: int | None = None


@dataclass(slots=True, frozen=True)
class TuiMouseEventResult:
    # Stop propagation and suppress renderer-level fallback behavior.
    handled: bool = False
    # Route subsequent drag/release events to this component. Implies handled.
    capture: bool = False
    # Give keyboard focus to this component. Implies handled.
    focus: bool = False
    # Explicitly request or suppress a render. Move and release default to
    # False; press, click, drag, and wheel default to True.
    render: bool | None = None


@dataclass(slots=True, frozen=True)
class TuiMouseDispatchTarget:
    """Internal target metadata used by containers and alternate-screen dispatch."""

    component: Any
    origin_x: int
    origin_y: int
    width: int
    height: int


@dataclass(slots=True, frozen=True)
class TuiMouseDispatchResult:
    """Result of dispatching to a concrete component."""

    target: TuiMouseDispatchTarget
    handled: bool = True
    capture: bool = False
    focus: bool = False
    render: bool | None = None
    # Keyboard focus target, which may be a delegating parent container.
    focus_target: Any = None


def dispatch_mouse_event(component, event: TuiMouseEvent) -> TuiMouseDispatchResult | None:
    """Dispatch an event to a component and retain the exact target and coordinate
    transform. Containers use this when forwarding events to nested children.
    """
    handle_mouse = getattr(component, "handle_mouse", None)
    if handle_mouse is None:
        return None
    result = handle_mouse(event)
    if result is None:
        return None
    if isinstance(result, TuiMouseDispatchResult):
        # The component forwarded the event to a child it hosts. Like a
        # delegating container, it routes keys to that child itself, so it
        # keeps keyboard focus. Focusing the child directly would leave focus
        # on a detached component once the host removes it, e.g. a closed
        # settings submenu.
        if result.focus and getattr(component, "handle_input", None) is not None:
            return replace(result, focus_target=component)
        return result
    if not result.handled and not result.capture and not result.focus:
        return None
    return TuiMouseDispatchResult(
        handled=True,
        capture=result.capture,
        focus=result.focus,
        render=result.render,
        focus_target=component if result.focus else None,
        target=TuiMouseDispatchTarget(
            component=component,
            origin_x=event.screen_x - event.x,
            origin_y=event.screen_y - event.y,
            width=event.width,
            height=event.height,
        ),
    )


def retarget_mouse_event(event: TuiMouseEvent, target: TuiMouseDispatchTarget) -> TuiMouseEvent:
    """Recreate local coordinates for a previously dispatched mouse target."""
    return replace(
        event,
        x=event.screen_x - target.origin_x,
        y=event.screen_y - target.origin_y,
        width=target.width,
        height=target.height,
    )


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
    # handle_mouse(event) -> TuiMouseEventResult | None for normalized
    # pointer input in fullscreen mode; wants_key_release = True to receive
    # Kitty key release events.


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
    __slots__ = ("bounds", "component", "focus_order", "hidden", "options", "pre_focus")

    def __init__(self, component, options, pre_focus, focus_order) -> None:
        self.component = component
        self.options: dict = options or {}
        self.pre_focus = pre_focus
        self.hidden = False
        self.focus_order = focus_order
        # Last rendered terminal-relative rectangle (pi's OverlayBounds):
        # {"row", "col", "width", "height"}, None until rendered.
        self.bounds: dict | None = None


class OverlayHandle:
    """Handle returned by show_overlay for controlling the overlay."""

    __slots__ = ("focus", "get_bounds", "hide", "is_focused", "is_hidden", "set_hidden", "unfocus")

    def __init__(self, *, hide, set_hidden, is_hidden, focus, unfocus, is_focused, get_bounds) -> None:
        self.hide = hide
        self.set_hidden = set_hidden
        self.is_hidden = is_hidden
        self.focus = focus
        self.unfocus = unfocus
        self.is_focused = is_focused
        # The most recent rendered bounds for a visible overlay, or None.
        self.get_bounds = get_bounds


_TERMINAL_PALETTE_SIZE = 16
# OSC 10 and 11 plus OSC 4 for every palette color.
_TERMINAL_COLOR_REPLY_COUNT = 2 + _TERMINAL_PALETTE_SIZE
# Default colors, palette colors 0-15, and a trailing primary device
# attributes (DA1) request. Every terminal answers DA1 and terminals answer in
# order, so the DA1 reply marks the end of the color replies, including for
# terminals that ignore the color queries.
_TERMINAL_COLOR_QUERY = (
    "\x1b]10;?\x07\x1b]11;?\x07"
    + "".join(f"\x1b]4;{index};?\x07" for index in range(_TERMINAL_PALETTE_SIZE))
    + "\x1b[c"
)
_DEVICE_ATTRIBUTES_RESPONSE_RE = re.compile(r"^\x1b\[\?[\d;]*c$")


class _PendingTerminalColorQuery:
    """pi's PendingTerminalColorQuery; its ``deliver`` callback becomes the
    two flags. Fields change under the TUI's ``_query_lock``."""

    __slots__ = ("applied", "background", "complete", "foreground", "palette", "replied", "settled", "timed_out")

    def __init__(self) -> None:
        self.foreground = None
        self.background = None
        self.palette: list = [None] * _TERMINAL_PALETTE_SIZE
        # Targets that already replied, so duplicates do not count twice.
        self.replied: set = set()
        # Completed (on the DA1 reply or once every color replied): later
        # replies are ignored.
        self.complete = False
        # The timeout sent the first report; completing sends a late one.
        self.timed_out = False
        # Set when the first report is sent: the awaiter's timeout wait.
        self.settled = tonio.Event()
        # Set once the listeners handled the first report.
        self.applied = tonio.Event()

    def result(self) -> dict:
        palette = list(self.palette) if all(color is not None for color in self.palette) else None
        return {"foreground": self.foreground, "background": self.background, "palette": palette}


class Container:
    """A component that contains other components."""

    def __init__(self) -> None:
        self.children: list = []
        # Geometry of the last rendered frame, for hit-testing without
        # re-rendering: {"width", "children": [(component, height), ...]}.
        self._mouse_layout: dict | None = None

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

        Correctness comes from the UI state lock (mutation and the frame's
        tree walk both hold it, so a rebuild can never overlap a frame).
        The single assignment remains as hygiene: a rebuild is one
        publication instead of a clear-then-append window, which keeps any
        off-contract reader (a test, a debug probe) from seeing a
        half-populated container.
        """
        self.children = list(children)

    def invalidate(self) -> None:
        for child in self.children:
            invalidate = getattr(child, "invalidate", None)
            if invalidate is not None:
                invalidate()

    def handle_mouse(self, event: TuiMouseEvent) -> TuiMouseDispatchResult | None:
        if event.y < 0 or event.y >= event.height:
            return None
        layout = self._mouse_layout
        mouse_children = (
            layout["children"]
            if layout is not None and layout["width"] == event.width
            else [(component, len(component.render(event.width))) for component in self.children]
        )
        child_y = 0
        for child, child_height in mouse_children:
            if child_y <= event.y < child_y + child_height:
                result = dispatch_mouse_event(child, replace(event, y=event.y - child_y, height=child_height))
                if result is not None and result.focus and getattr(self, "handle_input", None) is not None:
                    return replace(result, focus_target=self)
                return result
            child_y += child_height
        return None

    def render(self, width: int) -> list[str]:
        lines: list[str] = []
        mouse_children: list[tuple] = []
        for child in self.children:
            child_lines = child.render(width)
            mouse_children.append((child, len(child_lines)))
            lines.extend(child_lines)
        self._mouse_layout = {"width": width, "children": mouse_children}
        return lines


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


def call_sync(fn):
    """Call ``fn`` and return its result, refusing an awaitable one with
    ``TypeError`` (the coroutine is closed, not left unawaited).

    What runs under the UI state lock must not await (spec/ui-island.md,
    "`ctx.ui`"): ``apply`` and the extension UI contexts call through here. This
    enforces synchronous-only; it is not a ``T | Awaitable[T]`` union.
    """
    # An `async def` only builds its coroutine here: nothing runs.
    result = fn()
    if inspect.isawaitable(result):
        close = getattr(result, "close", None)
        if close is not None:
            close()
        raise TypeError("expected a synchronous function: nothing can be awaited under the UI state lock")
    return result


def is_viewport_tui(tui) -> bool:
    return getattr(tui, VIEWPORT_TUI, False) is True


def _without(listeners: tuple, listener) -> tuple:
    return tuple(registered for registered in listeners if registered != listener)


# TuiMode: "regular" (main screen) | "fullscreen" (alternate screen).

# pi streams renders through a BoundedTerminalWriter so a full render never
# forms one string large enough to exceed V8's limit; the island's frame
# pipeline still assembles whole frames (Python strings have no such cap),
# and the observable contract — no single terminal write above this size —
# is enforced at the write seam instead. pi also avoids splitting UTF-16
# surrogate pairs; Python slices on codepoints, so no guard is needed.
MAX_RENDER_WRITE_CHARS = 1024 * 1024


class _RenderRequest:
    """The render loop's request message (see the module docstring)."""

    __slots__ = ("force",)

    def __init__(self) -> None:
        self.force = False


class TuiBase(Container, ABC):
    """Renderer-independent half of the TUI: focus, overlays, input, queries.

    Subclasses own the frame: they implement ``_compose_frame`` and
    ``_write_frame`` and may hook the
    terminal lifecycle through ``_before_terminal_start`` / ``_after_terminal_start``
    / ``_before_terminal_stop`` / ``_after_terminal_stop`` and reset their
    differential state in ``_reset_render_state``.
    """

    def __init__(self, terminal, show_hardware_cursor: bool | None = None, log_directory: str | None = None) -> None:
        super().__init__()
        self.terminal = terminal
        # Directory for debug/crash logs. When None, debug logging is disabled
        # and crash dumps fall back to the OS temp directory.
        self._log_directory = log_directory
        self._focused_component = None
        # Listener registries: (un)registered from any task (extensions, the
        # theme controller), iterated by input routing and the terminal-event
        # loop. Copy-on-write tuples under
        # the guard, so a reader's reference never changes under it.
        self._listeners_guard = threading.Lock()
        self._input_listeners: tuple = ()

        # Global callback for debug key (Shift+Ctrl+D). Called before input is
        # forwarded to the focused component.
        self.on_debug = None
        # Spawned work the current input item must finish before the next
        # item is handled (`finish_before_next_input`). Only the input
        # consumer touches it. `_routing` is True while an item is routed
        # (read and written under the state lock only): registering from
        # anywhere else is refused.
        self._input_completions: list = []
        self._routing = False

        # Render scheduling (see the module docstring). `_render_active` is a
        # plain flag set by `start()`/`stop()` and read anywhere to drop
        # requests early; the request channel's sender and the render loop
        # exist while started. `_render_request` is the object requesters
        # send; the render loop replaces it on each receive.
        self._render_active = False
        self._render_requests = None
        self._render_loop_task = None
        self._render_request = _RenderRequest()
        self._line_reset_memo: dict[str, str] = {}
        # Frame pipeline (see the module docstring): the sender side of the
        # one-slot channel while the writer task runs, else None.
        self._frame_writer_task = None
        self._frames: Any = None
        self._frame_parts: list[str] = []
        # I/O a frame needs (debug and crash logs), run once the UI state
        # lock is released: see `_defer_frame_io`.
        self._frame_io: list = []
        self._frame_writer_error: BaseException | None = None
        # Async callback invoked when a frame raises; see `_render_loop`.
        self._render_error_handler = None
        # The UI state lock (spec/ui-island.md, "The guards"): a reentrant thread
        # lock with the terminal's lifetime, shared by every TUI on that
        # terminal; a terminal without one (tests) gets the TUI's own. It is
        # held for synchronous sections only — nothing awaits under it.
        terminal_state_lock = getattr(terminal, "state_lock", None)
        self.state_lock: threading.RLock = terminal_state_lock if terminal_state_lock is not None else threading.RLock()
        # One frame at a time: guards the render-private state (previous
        # frame, diff bookkeeping, hardware cursor, the frame writer).
        self._render_lock = sync.Lock()
        self._show_hardware_cursor = False
        # Clear empty rows when content shrinks (default: off)
        self._clear_on_shrink = False
        self._full_redraw_count = 0
        self._stopped = False
        self._query_lock = threading.Lock()
        # Color queries waiting for their DA1 reply, oldest first. Terminals
        # answer in order, so color replies belong to the oldest one. Queries
        # stay here after a timeout to collect late replies.
        self._pending_terminal_color_queries: list[_PendingTerminalColorQuery] = []
        self._color_scheme_listeners: tuple = ()  # under `_listeners_guard`
        self._terminal_colors_listeners: tuple = ()  # under `_listeners_guard`
        self._color_scheme_notifications_enabled = False
        # Terminal events (colour and colour-scheme reports) from the input
        # reader and the colour queries to their own loop, off the key path
        # (spec/ui-island.md, "Input"): the sender while the TUI is started
        # (swapped under `_query_lock`), and the loop's task.
        self._terminal_events = None
        self._terminal_event_task = None

        # Overlay stack for modal components rendered on top of base content
        self._focus_order_counter = 0
        self._overlay_stack: list[_OverlayStackEntry] = []
        self._overlay_focus_restore: dict = {"status": "inactive"}
        # Where the last frame composited each visible overlay:
        # {"entry", "row", "col", "width", "height"} records.
        self._rendered_overlay_layouts: list[dict] = []

        if show_hardware_cursor is not None:
            self._show_hardware_cursor = show_hardware_cursor

    # ------------------------------------------------------------------
    # Renderer hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _compose_frame(self) -> Any:
        """The frame's first half, under the UI state lock (and the render
        lock): walk the tree, which reads live component state, and publish
        what input reads back from render. Returns what `_write_frame` needs,
        or None when there is nothing to draw. Never awaits: I/O the frame
        needs goes through `_defer_frame_io`."""

    @abstractmethod
    def _write_frame(self, frame) -> None:
        """The frame's second half, under the render lock only: diff against
        the previous frame and `_emit` the output. Reads no live component
        state (`_compose_frame` captured it)."""

    def _reset_render_state(self) -> None:
        """Drop the differential state so the next frame repaints everything.
        Called under both locks: a renderer may also drop state input reads."""

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
        with self.state_lock:
            if self._show_hardware_cursor == enabled:
                return
            self._show_hardware_cursor = enabled
        # pi hides the cursor right here. Every frame ends by emitting the
        # cursor state (`_position_hardware_cursor` / the alt-screen frame
        # tail), so rendering stays the only path writing terminal
        # bytes; the requested render applies the change.
        self.request_render()

    def get_clear_on_shrink(self) -> bool:
        return self._clear_on_shrink

    def set_render_error_handler(self, handler) -> None:
        """Install the async callback that receives the UI's uncaught errors:
        a frame that raises, and whatever `report_error` is handed.

        Without one, a frame's exception ends the render loop and resurfaces
        from `stop()`, and `report_error` re-raises, so owners should install
        one.
        """
        self._render_error_handler = handler

    def apply(self, fn):
        """Run the synchronous ``fn`` under the UI state lock, as one whole
        change, and return its result: the way to change UI state from a
        timer or spawned work. Callable from anywhere, input handling
        included (the lock is reentrant).

        Nothing may be awaited under the lock, so a coroutine function, or
        a callable returning an awaitable, is refused with ``TypeError``.
        """
        with self.state_lock:
            return call_sync(fn)

    def report_error(self, error: BaseException) -> None:
        """Hand an error that nothing up the stack can take to the installed
        handler (interactive mode's crash handler), on a task of its own:
        the input consumer's item, the output pump's write, a terminal event,
        a component's spawned work (spec/ui-island.md, "Errors"). With no handler
        installed it is re-raised to the caller."""
        handler = self._render_error_handler
        if handler is None:
            raise error
        tonio.spawn.without_tracking(handler(error))

    def set_clear_on_shrink(self, enabled: bool) -> None:
        """Set whether to trigger full re-render when content shrinks.

        When enabled, empty rows are cleared when content shrinks. When
        disabled, empty rows remain (reduces redraws on slower terminals).
        """
        with self.state_lock:
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
        # No direct `hide_cursor()` here or in the hide paths below:
        # rendering emits the cursor state with every frame (see
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

        def get_bounds() -> dict | None:
            if entry not in self._overlay_stack or not self._is_overlay_visible(entry) or entry.bounds is None:
                return None
            return dict(entry.bounds)

        return OverlayHandle(
            hide=hide,
            set_hidden=set_hidden,
            is_hidden=is_hidden,
            focus=focus,
            unfocus=unfocus,
            is_focused=is_focused,
            get_bounds=get_bounds,
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

    def _resolve_mouse_focus_target(self, component):
        """Keep overlay containers as keyboard focus owners when a nested control is clicked."""
        for overlay in reversed(self._overlay_stack):
            if self._is_overlay_visible(overlay) and self._contains_component(overlay.component, component):
                return overlay.component
        return component

    def _dispatch_mouse_to_overlay(self, event: TuiMouseEvent) -> tuple[bool, TuiMouseDispatchResult | None]:
        """Dispatch to the visually topmost overlay under the pointer: (hit, result)."""
        for layout in reversed(self._rendered_overlay_layouts):
            if (
                event.screen_x < layout["col"]
                or event.screen_x >= layout["col"] + layout["width"]
                or event.screen_y < layout["row"]
                or event.screen_y >= layout["row"] + layout["height"]
            ):
                continue
            result = dispatch_mouse_event(
                layout["entry"].component,
                replace(
                    event,
                    x=event.screen_x - layout["col"],
                    y=event.screen_y - layout["row"],
                    width=layout["width"],
                    height=layout["height"],
                ),
            )
            if result is None:
                return True, None
            return True, replace(result, focus_target=layout["entry"].component) if result.focus else result
        return False, None

    def _handle_pointer_input(self, data: str) -> bool:
        """Renderer hook for pointer input, run before the input listeners.

        pi handles the mouse inside the alternate screen's viewport input
        listener; here the pointer path is this hook instead. Returns True
        when the input was consumed.
        """
        return False

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
        # Armed before anything this session writes: the alternate-screen
        # entry comes before `terminal.start()`.
        self.terminal.arm()
        await self._before_terminal_start()
        on_input = self._handle_input
        if not isinstance(self.terminal, ProcessTerminal):
            # Only a ProcessTerminal runs an input consumer. Any other
            # terminal's caller awaits each item, so the error routing a
            # consumer does happens here.

            async def on_input(data: str) -> None:
                try:
                    await self._handle_input(data)
                except Exception as error:
                    self.report_error(error)

        sender, events = channel.unbounded()
        with self._query_lock:
            self._terminal_events = sender
        self._terminal_event_task = tonio.spawn(self._run_terminal_events(events))
        # Requests are dropped (`_render_active`) until the end of start.
        self._render_request = _RenderRequest()
        self._render_requests, requests = channel.channel(1)
        self._render_loop_task = tonio.spawn(self._render_loop(requests))
        try:
            await self.terminal.start(on_input, self.request_render, self._consume_terminal_reply, self.report_error)
            await self._after_terminal_start()
            self.terminal.hide_cursor()
            if self._color_scheme_notifications_enabled:
                await self.terminal.write("\x1b[?2031h")
            await self._query_cell_size()
            self._frames, receiver = channel.channel(1)
            self._frame_writer_error = None
            # Ended by the `None` sentinel `stop()` sends, and joined there.
            self._frame_writer_task = tonio.spawn(self._frame_writer(receiver))
        except BaseException:
            # A failed start leaves no terminal events or rendering running:
            # the loops end on their closed channels.
            self._close_terminal_events()
            self._terminal_event_task = None
            self._render_requests = self._render_loop_task = None
            requests.close()
            raise
        # Requests made earlier in start() were dropped by the
        # `_render_active` gate; this final request supersedes them all with
        # the first frame.
        self._render_active = True
        self.request_render()

    def close(self) -> Awaitable[None]:
        """App shutdown, once: ends what outlives stop/start, the terminal's
        output queue (putting out what it still holds) and input consumer."""
        return self.terminal.close()

    def finish_before_next_input(self, task) -> None:
        """Hold the next input item until ``task`` (a ``tonio.spawn()``
        handle) finishes.

        Input handlers are synchronous; a handler whose effect needs I/O
        (pi does that I/O synchronously) spawns it and registers the handle
        here, so the next key sees the effect. Awaiting the handle re-raises
        the task's error, which then takes the input error path.

        Only from input handling (``handle_input``/``handle_mouse`` and the
        callbacks they run): from anywhere else the handle would be awaited
        by an unrelated item, or never, so that raises ``RuntimeError``.
        """
        # Routing holds the (reentrant) lock, so a call from inside it
        # re-enters and sees the flag; any other caller waits for the item
        # to be routed and sees it cleared.
        with self.state_lock:
            if not self._routing:
                raise RuntimeError("finish_before_next_input() called outside input handling")
            self._input_completions.append(task)

    async def _finish_input_completions(self) -> None:
        completions, self._input_completions = self._input_completions, []
        error = None
        for completion in completions:
            try:
                await completion
            except Exception as exc:
                if error is None:
                    error = exc
        if error is None:
            return
        # A spawn handle raises the task's error inside a
        # `SpawnExceptionGroup`; the error handler gets the task's own error,
        # with the group as its cause.
        leaf = error
        while isinstance(leaf, BaseExceptionGroup) and leaf.exceptions:
            leaf = leaf.exceptions[0]
        if leaf is error:
            raise error
        raise leaf from error

    def add_input_listener(self, listener):
        with self._listeners_guard:
            if listener not in self._input_listeners:
                self._input_listeners = (*self._input_listeners, listener)

        def unsubscribe() -> None:
            self.remove_input_listener(listener)

        return unsubscribe

    def remove_input_listener(self, listener) -> None:
        with self._listeners_guard:
            self._input_listeners = _without(self._input_listeners, listener)

    def on_terminal_colors(self, listener):
        """Register an async ``listener(colors)`` for the TerminalColors
        reports of `query_terminal_colors`, in arrival order. Returns the
        unsubscribe function."""
        with self._listeners_guard:
            if listener not in self._terminal_colors_listeners:
                self._terminal_colors_listeners = (*self._terminal_colors_listeners, listener)

        def unsubscribe() -> None:
            with self._listeners_guard:
                self._terminal_colors_listeners = _without(self._terminal_colors_listeners, listener)

        return unsubscribe

    def on_terminal_color_scheme_change(self, listener):
        with self._listeners_guard:
            if listener not in self._color_scheme_listeners:
                self._color_scheme_listeners = (*self._color_scheme_listeners, listener)

        def unsubscribe() -> None:
            with self._listeners_guard:
                self._color_scheme_listeners = _without(self._color_scheme_listeners, listener)

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

        Never waits on input handling (spec/ui-island.md, "Stopping the UI
        from inside a key"), so a key's
        completion may stop the UI, as pi's key handlers do in place.
        """
        options = options or {}
        self._stopped = True
        self._render_active = False

        # Later requests are dropped; the render loop finishes the frame it
        # is on, if any, and ends, so nothing sends into a drained pipeline.
        requests, self._render_requests = self._render_requests, None
        if requests is not None:
            requests.close()
        if self._render_loop_task is not None:
            task, self._render_loop_task = self._render_loop_task, None
            await task
        # The render lock: the writer and the diff state the exit sequence
        # reads are render-private.
        async with self._render_lock:
            if self._frame_writer_task is not None:
                # After the barrier: what rendering handed over still goes out,
                # then the writer stops and everything below writes in order
                # behind it.
                frames, self._frames = self._frames, None
                await frames.send(None)
                await self._frame_writer_task
                self._frame_writer_task = None
            if self._color_scheme_notifications_enabled:
                await self.terminal.write("\x1b[?2031l")
            await self._before_terminal_stop(options)
        self.terminal.show_cursor()
        await self.terminal.stop()
        # The reader is gone: no more terminal events. The loop hands out
        # what is queued, and a listener being notified finishes first.
        self._close_terminal_events()
        if self._terminal_event_task is not None:
            task, self._terminal_event_task = self._terminal_event_task, None
            await task
        async with self._render_lock:
            await self._after_terminal_stop(options)
        # Released after the last thing this session writes: the
        # alternate-screen exit comes after `terminal.stop()`.
        await self.terminal.release()

    async def render_now(self, force: bool = False) -> None:
        """Render one frame on the caller, bypassing the render loop, and
        wait until it is on the wire. The render lock orders it
        with the loop's frames (the UI-mode switch renders a stopped regular
        screen this way)."""
        async with self._render_lock:
            if force:
                self._reset_for_full_redraw()
            await self._render_frame()
            await self._flush_frames()

    def _reset_for_full_redraw(self) -> None:
        with self.state_lock:
            self._reset_render_state()

    async def _render_frame(self) -> None:
        """One frame, under the render lock: walk the tree under the UI state
        lock, then — released — diff it, run its I/O and hand the bytes over."""
        try:
            with self.state_lock:
                frame = self._compose_frame()
            if frame is not None:
                self._write_frame(frame)
        except Exception:
            self._frame_parts = []  # a torn frame is never written
            await self._run_frame_io()  # the crash log lands before the error propagates
            raise
        await self._run_frame_io()
        await self._send_frame()

    def _defer_frame_io(self, io) -> None:
        """Queue I/O the frame being computed needs (`io` returns an
        awaitable). It runs after the UI state lock is released, in order,
        before the frame goes out — or before the frame's error propagates."""
        self._frame_io.append(io)

    async def _run_frame_io(self) -> None:
        pending, self._frame_io = self._frame_io, []
        for io in pending:
            await io()

    def _emit(self, data: str) -> None:
        """Add to the frame being rendered; it goes out as one write when `_write_frame` returns."""
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
            await self._write_bounded(data)
            return
        await frames.send(data)
        if self._frame_writer_error is not None:
            raise self._frame_writer_error

    async def _write_bounded(self, data: str) -> None:
        """Write frame data in chunks no larger than MAX_RENDER_WRITE_CHARS."""
        for offset in range(0, len(data), MAX_RENDER_WRITE_CHARS):
            await self.terminal.write(data[offset : offset + MAX_RENDER_WRITE_CHARS])

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
            try:
                item = await receiver.receive()
            except BrokenPipeError:
                return  # sender closed: nothing more will be handed over
            if item is None:
                return
            if isinstance(item, tonio.Event):
                item.set()
                continue
            if self._frame_writer_error is not None:
                continue  # the terminal is gone; the loop learns on its next _emit
            try:
                await self._write_bounded(item)
            except Exception as error:
                # A dead writer would wedge the next render job on its send;
                # the stored error resurfaces there instead.
                self._frame_writer_error = error

    def request_render(self, force: bool = False) -> None:
        # Sync, callable from any task (a component's render included), and
        # touches no lock. pi calls resetRenderState() right here; the diff
        # state is the render loop's, so `force` travels in the request.
        if not self._render_active:
            return
        request = self._render_request
        if force:
            # Before the send: if the channel is full, the pending request
            # is this same object, so the force still lands.
            request.force = True
        requests = self._render_requests
        if requests is not None:
            # Full: a pending request covers this one. Closed: stopping.
            requests.send_nowait(request)

    async def _render_loop(self, requests) -> None:
        """One frame per received request. Ends when `stop()` closes the
        channel, or after a frame raises (rendering is over; the error goes
        to the installed handler)."""
        while True:
            try:
                request = await requests.receive()
            except BrokenPipeError:
                return  # `stop()`
            # Later requests go in a fresh object; `force` is read once, after.
            # A forced request that took this one just before the swap either
            # set `force` before this read, or sends it again (the channel is
            # empty now) and forces the next frame. A plain request resending
            # it can only repeat a full repaint, never lose one.
            self._render_request = _RenderRequest()
            force = request.force
            try:
                async with self._render_lock:
                    if force:
                        self._reset_for_full_redraw()
                    await self._render_frame()
            except Exception as error:
                # pi crashes the process on a render throw. Here rendering
                # stops and the error goes to the installed handler
                # (interactive mode's crash handler).
                self._render_active = False
                handler = self._render_error_handler
                if handler is None:
                    raise
                # Detached: the handler typically calls `stop()`, which
                # waits for this loop to end.
                tonio.spawn.without_tracking(handler(error))
                return

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    async def _handle_input(self, data: str) -> None:
        try:
            with self.state_lock:
                self._routing = True
                try:
                    self._route_input(data)
                finally:
                    self._routing = False
        finally:
            await self._finish_input_completions()

    def _consume_terminal_reply(self, data: str) -> bool:
        """The terminal's ``on_reply``: runs in its input reader, ahead of the
        input order (pi checks these first in `handleInput`). A query's
        answer settles the query even while input handling waits on the
        work that asked; colour and colour-scheme reports go to the
        terminal-event loop, since reacting to them can mean loading a
        theme."""
        if self._consume_terminal_color_response(data):
            return True
        scheme = parse_terminal_color_scheme_report(data)
        if not scheme:
            return False
        with self._query_lock:
            events = self._terminal_events
            if events is not None:
                events.send(("scheme", scheme, None))
        return True

    async def _run_terminal_events(self, receiver) -> None:
        while True:
            try:
                kind, payload, applied = await receiver.receive()
            except BrokenPipeError:
                return  # `stop()`
            try:
                if kind == "colors":
                    await self._notify_terminal_colors(payload)
                else:
                    await self._notify_terminal_color_scheme(payload)
            except Exception as error:
                self.report_error(error)
            if applied is not None:
                applied.set()

    def _close_terminal_events(self) -> None:
        """End the terminal-event loop: it hands out what is queued, then
        stops. A report sent later only settles its query's `applied`."""
        with self._query_lock:
            events, self._terminal_events = self._terminal_events, None
        if events is not None:
            events.close()

    def _send_terminal_colors(self, query: _PendingTerminalColorQuery, first: bool) -> None:
        """Send one report of ``query`` to the loop. Under `_query_lock`, so
        reports go out in the order the query transitions happened."""
        applied = query.applied if first else None
        events = self._terminal_events
        if events is None:
            # Stopped: nothing handles the report, but waiters must not hang.
            if applied is not None:
                applied.set()
            return
        events.send(("colors", query.result(), applied))

    def _route_input(self, data: str) -> None:
        """The input item's synchronous handling, under the UI state lock."""
        if self._handle_pointer_input(data):
            return

        input_listeners = self._input_listeners
        if input_listeners:
            current = data
            for listener in input_listeners:
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

    def _consume_terminal_color_response(self, data: str) -> bool:
        is_device_attributes = _DEVICE_ATTRIBUTES_RESPONSE_RE.match(data) is not None
        response = None if is_device_attributes else parse_osc_color_response(data)
        if not is_device_attributes and response is None:
            return False
        with self._query_lock:
            if not self._pending_terminal_color_queries:
                return False
            query = self._pending_terminal_color_queries[0]
            if is_device_attributes:
                self._pending_terminal_color_queries.pop(0)
                self._complete_terminal_color_query(query)
                return True

            target = response["target"]
            key = str(target)
            if query.complete or key in query.replied:
                return True
            query.replied.add(key)
            if target == "foreground":
                query.foreground = response["rgb"]
            elif target == "background":
                query.background = response["rgb"]
            elif target < _TERMINAL_PALETTE_SIZE:
                query.palette[target] = response["rgb"]
            if len(query.replied) == _TERMINAL_COLOR_REPLY_COUNT:
                self._complete_terminal_color_query(query)
        return True

    def _complete_terminal_color_query(self, query: _PendingTerminalColorQuery) -> None:
        """Under `_query_lock`: the first report, or a late one after the
        timeout; once only."""
        if query.complete:
            return
        query.complete = True
        self._send_terminal_colors(query, first=not query.timed_out)
        query.settled.set()

    async def _notify_terminal_colors(self, colors: dict) -> None:
        for listener in self._terminal_colors_listeners:
            await listener(colors)

    async def _notify_terminal_color_scheme(self, scheme: str) -> None:
        for listener in self._color_scheme_listeners:
            # Listeners are awaitable-returning (async-only policy): reacting
            # to a scheme change can mean loading a theme from disk.
            await listener(scheme)

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
            self._rendered_overlay_layouts = []
            return lines
        result = list(lines)

        for entry in self._overlay_stack:
            entry.bounds = None

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
            entry.bounds = {"row": layout["row"], "col": layout["col"], "width": width, "height": len(overlay_lines)}

            rendered.append(
                {"entry": entry, "overlayLines": overlay_lines, "row": layout["row"], "col": layout["col"], "w": width}
            )
            min_lines_needed = max(min_lines_needed, layout["row"] + len(overlay_lines))
        self._rendered_overlay_layouts = [
            {
                "entry": item["entry"],
                "row": item["row"],
                "col": item["col"],
                "width": item["w"],
                "height": len(item["overlayLines"]),
            }
            for item in rendered
        ]

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

    def query_terminal_colors(self, *, timeout_ms: float) -> tonio.Event:
        """Query the terminal's theme colors: the default foreground (OSC 10),
        the default background (OSC 11), and ANSI colors 0-15 (OSC 4),
        followed by a DA1 request that marks the end of the replies.

        The query reports a TerminalColors record to the `on_terminal_colors`
        listeners when the DA1 reply or all color replies arrive, or when the
        timeout (for terminals that do not answer DA1 either) expires; colors
        the terminal did not report are None, and the palette is only set when
        all 16 arrived. Replies completing the query after the timeout, e.g.
        over slow links, are reported once more. A query whose write fails
        reports no colors.

        Returns an event set once the listeners handled the first report.
        """
        query = _PendingTerminalColorQuery()
        with self._query_lock:
            self._pending_terminal_color_queries.append(query)
        tonio.spawn.without_tracking(self._run_terminal_color_query(query, timeout_ms))
        return query.applied

    async def _run_terminal_color_query(self, query: _PendingTerminalColorQuery, timeout_ms: float) -> None:
        """The burst write, then the timeout; the reader completes the query."""
        try:
            await self.terminal.write(_TERMINAL_COLOR_QUERY)
        except Exception:
            # pi treats a failed query like a terminal that does not report
            # colors: one empty report.
            with self._query_lock:
                if query in self._pending_terminal_color_queries:
                    self._pending_terminal_color_queries.remove(query)
                query.foreground = query.background = None
                query.palette = [None] * _TERMINAL_PALETTE_SIZE
                self._complete_terminal_color_query(query)
            return
        await query.settled.wait(timeout_ms / 1000)
        with self._query_lock:
            if query.complete or query.timed_out:
                return
            # Report the replies so far, and keep collecting late replies.
            query.timed_out = True
            self._send_terminal_colors(query, first=True)
            query.settled.set()


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
