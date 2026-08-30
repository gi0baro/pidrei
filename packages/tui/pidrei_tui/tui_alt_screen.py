"""Mirror of pi tui src/tui-alt-screen.ts.

Alternate-screen TUI with a scrollable, application-owned viewport: a layout
tree is measured into the terminal box every frame, scroll regions own their
positions, and scrolling, text selection and OSC 8 link activation are handled
here instead of by the terminal emulator. ``tui_main_screen`` is the other
renderer; the machinery both share lives in ``tui``.

With no layout root set, the whole child list is wrapped in one
end-following ``ScrollView`` so the renderer behaves like the main screen.

Port deviations: ``_do_render`` and the lifecycle hooks are async (the
terminal driver is). Mouse handling runs in a sync input listener, so the
OSC 52 clipboard write it triggers is spawned rather than awaited — pi's
``terminal.write`` is sync.
"""

import base64
import math
import os
import re
import time

import tonio.colored as tonio

from ._timers import Interval
from .alt_screen_search import (
    AltScreenSearchComponent,
    find_alt_screen_search_matches,
    get_alt_screen_search_match_key,
)
from .components.alt_screen_flash import AltScreenFlashContainer
from .components.scroll_view import ScrollView
from .keybindings import get_keybindings
from .keys import is_key_release
from .layout import get_scroll_view_box, get_scroll_views_at, get_scrollbar_geometry, render_layout_frame
from .terminal_image import (
    delete_all_kitty_images,
    delete_all_kitty_placements,
    delete_kitty_image,
    get_capabilities,
    get_kitty_image_placement,
    is_image_line,
    set_capabilities,
)
from .tui import CURSOR_MARKER, VIEWPORT_TUI, TuiBase, composite_tui_line
from .utils import (
    extract_ansi_code,
    get_grapheme_cell_range,
    get_osc8_link_at_column,
    get_word_segmenter,
    js_round,
    slice_by_column,
    strip_terminal_sequences,
    visible_width,
)


ENTER_ALT_SCREEN = "\x1b[?1049h"
EXIT_ALT_SCREEN = "\x1b[?1049l"
DISABLE_AUTOWRAP = "\x1b[?7l"
ENABLE_AUTOWRAP = "\x1b[?7h"
ENABLE_BUTTON_MOTION_MOUSE = "\x1b[?1000h\x1b[?1002h\x1b[?1004h\x1b[?1006h"
ENABLE_ALL_MOTION_MOUSE = "\x1b[?1000h\x1b[?1002h\x1b[?1003h\x1b[?1004h\x1b[?1006h"
DISABLE_MOUSE = "\x1b[?1006l\x1b[?1004l\x1b[?1003l\x1b[?1002l\x1b[?1000l"
FOCUS_IN = "\x1b[I"
FOCUS_OUT = "\x1b[O"
BEGIN_SYNCHRONIZED_OUTPUT = "\x1b[?2026h"
END_SYNCHRONIZED_OUTPUT = "\x1b[?2026l"
OSC133_ZONE_PREFIX = re.compile(r"^(?:\x1b\]133;[ABC](?:\x07|\x1b\\))+")
OSC133_PROMPT_START = re.compile(r"^\x1b\]133;A(?:\x07|\x1b\\)")
PAGE_SCROLL_OVERLAP = 4
MAX_CACHED_OFFSCREEN_KITTY_IMAGES = 16
MAX_CACHED_OFFSCREEN_KITTY_TRANSMISSION_BYTES = 32 * 1024 * 1024
MAX_CACHED_OFFSCREEN_KITTY_DECODED_BYTES = 64 * 1024 * 1024
DOUBLE_CLICK_INTERVAL_S = 0.5
# Regular mode delegates double-click selection to the terminal emulator. Fullscreen owns mouse selection,
# so mirror common terminal word-selection behavior by keeping paths and kebab-case tokens whole.
TERMINAL_WORD_SELECTION_JOINERS = {"/", "-"}
_word_segmenter = get_word_segmenter()

_SGR_MOUSE_RE = re.compile(r"^\x1b\[<(\d+);(\d+);(\d+)([Mm])$")


class _ImplicitDocument:
    """Renders the TUI's own child list, for the no-layout-root case."""

    def __init__(self, tui) -> None:
        self._tui = tui

    def render(self, width: int) -> list[str]:
        return super(TuiAltScreen, self._tui).render(width)

    def invalidate(self) -> None:
        for child in self._tui.children:
            child.invalidate()


class TuiAltScreen(TuiBase):
    """Alternate-screen TUI with a scrollable, application-owned viewport.

    Options (keyword arguments, pi's ``TuiAltScreenOptions``):
    ``wheel_scroll_lines`` — logical lines moved per mouse-wheel event;
    ``mouse`` — capture mouse events for viewport scrolling and
    application-owned text selection; ``open_url`` — callback for an OSC 8
    hyperlink activated with a primary-button click; ``copy_selection`` —
    async callback copying selected text to the system clipboard, returning
    ``True`` on success (the caller flashes an error otherwise); when omitted,
    the selection is copied via an OSC 52 write.

    Selection points are ``{"row", "col", "scrollView"?}`` records; mouse
    events are ``{"button", "x", "y", "release"}``; wheel events are
    ``{"direction", "x", "y"}``.
    """

    mode = "fullscreen"

    def __init__(
        self,
        terminal,
        show_hardware_cursor: bool | None = None,
        log_directory: str | None = None,
        *,
        wheel_scroll_lines: int | None = None,
        mouse: bool | None = None,
        search_match_style=None,
        search_current_match_style=None,
        open_url=None,
        copy_on_select: bool | None = None,
        copy_selection=None,
    ) -> None:
        super().__init__(terminal, show_hardware_cursor, log_directory)
        self._previous_screen: list[str] = []
        self._last_document: list[str] = []
        self._previous_screen_width = 0
        self._previous_screen_height = 0
        self._layout_root = None
        self._current_layout = None
        self._implicit_document = _ImplicitDocument(self)
        self._implicit_scroll_view = ScrollView(self._implicit_document, {"follow": "end", "primary": True})
        self._flashes = AltScreenFlashContainer(self.request_render)
        self._alt_screen_active = False
        self._image_protocol = None
        self._saved_capabilities: dict | None = None
        # image id -> {"transmissionGeneration", "transmissionBytes",
        # "estimatedDecodedBytes"} for what this session already uploaded,
        # in least-recently-placed order
        self._uploaded_kitty_images: dict[int, dict] = {}
        self._selection_anchor: dict | None = None
        self._selection_focus: dict | None = None
        self._selection_granularity = "character"
        self._selection_initial_range: dict | None = None
        self._last_click: dict | None = None
        self._selection_drag_pointer: dict | None = None
        self._selection_auto_scroll_direction = 0
        self._selection_auto_scroll_timer: Interval | None = None
        self._selection_press_active = False
        self._scrollbar_drag: dict | None = None
        self._scrollbar_hover = None
        self._pressed_url: str | None = None
        self._selection_dragged = False
        self._active_search: dict | None = None
        self._wheel_scroll_lines = max(1, math.floor(wheel_scroll_lines if wheel_scroll_lines is not None else 1))
        self._mouse_enabled = mouse if mouse is not None else True
        self._search_match_style = search_match_style or (lambda text: f"\x1b[4m{text}\x1b[24m")
        self._search_current_match_style = search_current_match_style or (lambda text: f"\x1b[1;7m{text}\x1b[22;27m")
        self._open_url = open_url
        self._copy_on_select = copy_on_select if copy_on_select is not None else True
        self._copy_selection = copy_selection
        self.add_input_listener(self._handle_viewport_input)

    def get_copy_on_select(self) -> bool:
        return self._copy_on_select

    def set_copy_on_select(self, enabled: bool) -> None:
        self._copy_on_select = enabled

    def has_active_selection(self) -> bool:
        """Whether the fullscreen viewport has a non-empty active text selection."""
        return self._get_active_selection_text() is not None

    async def copy_active_selection_to_clipboard(self) -> bool:
        """Copy the active fullscreen text selection, if any, using the configured
        selection clipboard path."""
        text = self._get_active_selection_text()
        if not text:
            return False
        return await self._copy_text_to_clipboard(text)

    @property
    def viewport_top(self) -> int:
        return self._get_primary_scroll_view().scroll_top

    @property
    def is_following_output(self) -> bool:
        return self._get_primary_scroll_view().is_following_end

    def set_layout_root(self, component) -> None:
        if self._layout_root is component:
            return
        self._layout_root = component
        self._current_layout = None
        self.request_render()

    def render(self, width: int) -> list[str]:
        return self._layout_root.render(width) if self._layout_root is not None else super().render(width)

    def _get_mounted_roots(self) -> list:
        return [self._layout_root] if self._layout_root is not None else self.children

    def _get_primary_scroll_view(self) -> ScrollView:
        if self._current_layout is not None and self._current_layout.primary_scroll_view is not None:
            return self._current_layout.primary_scroll_view
        return self._implicit_scroll_view

    async def _before_terminal_start(self) -> None:
        self._stop_selection_auto_scroll()
        self._selection_press_active = False
        self._stop_scrollbar_hover()
        self._stop_scrollbar_drag()
        self._flashes.dispose()
        self._alt_screen_active = True
        capabilities = get_capabilities()
        self._image_protocol = capabilities["images"]
        self._uploaded_kitty_images.clear()
        if capabilities["images"] == "iterm2":
            self._saved_capabilities = capabilities
            set_capabilities({**capabilities, "images": None})
            self.invalidate()
        self._last_document = []
        self._selection_anchor = None
        self._selection_focus = None
        self._selection_granularity = "character"
        self._selection_initial_range = None
        self._last_click = None
        self._pressed_url = None
        self._selection_dragged = False
        self._reset_render_state()
        term = (os.environ.get("TERM") or "").lower()
        # Multiplexers can lag when every pointer movement is forwarded. Button-motion
        # tracking preserves clicks, wheel events, selections, and scrollbar dragging.
        in_multiplexer = (
            os.environ.get("TMUX") is not None
            or os.environ.get("ZELLIJ") is not None
            or os.environ.get("STY") is not None
            or term.startswith(("tmux", "screen"))
        )
        mouse_sequence = ENABLE_BUTTON_MOTION_MOUSE if in_multiplexer else ENABLE_ALL_MOTION_MOUSE
        await self.terminal.write(
            f"{ENTER_ALT_SCREEN}{DISABLE_AUTOWRAP}{mouse_sequence if self._mouse_enabled else ''}\x1b[2J\x1b[H\x1b[?25l"
        )

    async def _before_terminal_stop(self, _options: dict) -> None:
        self._close_search()
        self._stop_selection_auto_scroll()
        self._selection_press_active = False
        self._stop_scrollbar_hover()
        self._stop_scrollbar_drag()
        self._flashes.dispose()
        if not self._alt_screen_active:
            return
        await self.terminal.write(
            f"{BEGIN_SYNCHRONIZED_OUTPUT}{self._delete_kitty_images()}"
            f"{DISABLE_MOUSE if self._mouse_enabled else ''}{ENABLE_AUTOWRAP}{END_SYNCHRONIZED_OUTPUT}"
        )
        self._uploaded_kitty_images.clear()

    async def _after_terminal_stop(self, options: dict) -> None:
        if not self._alt_screen_active:
            return
        self._alt_screen_active = False
        if options.get("preserveScreen"):
            # The renderer taking over owns the main screen from here; leaving
            # the alt screen is all this one still has to do.
            await self.terminal.write(f"{BEGIN_SYNCHRONIZED_OUTPUT}{EXIT_ALT_SCREEN}\x1b[?25h{END_SYNCHRONIZED_OUTPUT}")
        else:
            width = max(1, self.terminal.columns)
            document_lines = [OSC133_ZONE_PREFIX.sub("", line) for line in self.render(width)]
            self._last_document = [
                line if is_image_line(line) or visible_width(line) <= width else slice_by_column(line, 0, width, True)
                for line in self._apply_line_resets([line.replace(CURSOR_MARKER, "") for line in document_lines])
            ]
            buffer = f"{BEGIN_SYNCHRONIZED_OUTPUT}{EXIT_ALT_SCREEN}{DISABLE_AUTOWRAP}"
            for row, line in enumerate(self._last_document):
                if row > 0:
                    buffer += "\r\n"
                buffer += f"\r\x1b[2K{line}"
            buffer += f"\x1b[0m{ENABLE_AUTOWRAP}\r\n\x1b[?25h{END_SYNCHRONIZED_OUTPUT}"
            await self.terminal.write(buffer)
        if self._saved_capabilities:
            set_capabilities(self._saved_capabilities)
            self._saved_capabilities = None

    def _delete_kitty_images(self) -> str:
        return delete_all_kitty_images() if self._image_protocol == "kitty" else ""

    def _prepare_kitty_screen(self, screen: list[str]) -> tuple[list[str], str]:
        """Swap already-uploaded images for placement-only commands.

        Returns the rewritten lines and a deletion sequence for the offscreen
        images evicted to stay under the cache budgets.
        """
        visible_image_ids: set[int] = set()
        lines: list[str] = []
        for line in screen:
            placement = get_kitty_image_placement(line)
            if not placement:
                lines.append(line)
                continue
            visible_image_ids.add(placement["imageId"])

            cached_image = self._uploaded_kitty_images.get(placement["imageId"])
            next_cached_image = {
                "transmissionGeneration": placement["transmissionGeneration"],
                "transmissionBytes": placement["transmissionBytes"],
                "estimatedDecodedBytes": placement["estimatedDecodedBytes"],
            }
            if cached_image:
                del self._uploaded_kitty_images[placement["imageId"]]
            self._uploaded_kitty_images[placement["imageId"]] = next_cached_image

            lines.append(
                placement["replacementLine"]
                if cached_image is not None
                and cached_image["transmissionGeneration"] == placement["transmissionGeneration"]
                else line
            )

        cached_offscreen_image_count = 0
        cached_offscreen_transmission_bytes = 0
        cached_offscreen_decoded_bytes = 0
        for image_id, cached_image in self._uploaded_kitty_images.items():
            if image_id in visible_image_ids:
                continue
            cached_offscreen_image_count += 1
            cached_offscreen_transmission_bytes += cached_image["transmissionBytes"]
            cached_offscreen_decoded_bytes += cached_image["estimatedDecodedBytes"]

        evicted_image_deletion = ""
        for image_id, cached_image in list(self._uploaded_kitty_images.items()):
            if (
                cached_offscreen_image_count <= MAX_CACHED_OFFSCREEN_KITTY_IMAGES
                and cached_offscreen_transmission_bytes <= MAX_CACHED_OFFSCREEN_KITTY_TRANSMISSION_BYTES
                and cached_offscreen_decoded_bytes <= MAX_CACHED_OFFSCREEN_KITTY_DECODED_BYTES
            ):
                break
            if image_id in visible_image_ids:
                continue
            evicted_image_deletion += delete_kitty_image(image_id)
            del self._uploaded_kitty_images[image_id]
            cached_offscreen_image_count -= 1
            cached_offscreen_transmission_bytes -= cached_image["transmissionBytes"]
            cached_offscreen_decoded_bytes -= cached_image["estimatedDecodedBytes"]
        return lines, evicted_image_deletion

    def _reset_render_state(self) -> None:
        self._previous_screen = []
        self._previous_screen_width = 0
        self._previous_screen_height = 0
        self._current_layout = None

    def scroll_by(self, lines: int) -> None:
        self._get_primary_scroll_view().scroll_by(lines)
        self.request_render()

    def scroll_to_top(self) -> None:
        self._get_primary_scroll_view().scroll_to_start()
        self.request_render()

    def scroll_to_bottom(self) -> None:
        self._get_primary_scroll_view().scroll_to_end()
        self.request_render()

    def _scroll_to_prompt(self, direction: int) -> None:
        if self._current_layout is None:
            return
        scroll_view = self._get_primary_scroll_view()
        box = get_scroll_view_box(self._current_layout, scroll_view)
        lines = box.scroll_content_lines if box is not None else None
        if lines is None:
            return

        row = scroll_view.scroll_top + direction
        while 0 <= row < len(lines):
            if OSC133_PROMPT_START.match(lines[row]):
                scroll_view.scroll_to(row)
                self.request_render()
                return
            row += direction

    def _open_search(self) -> None:
        if self._active_search is not None:
            overlay = self._active_search.get("overlay")
            if overlay is not None:
                overlay.focus()
            return
        component = AltScreenSearchComponent(self._update_search_query)
        search: dict = {
            "component": component,
            "overlay": None,
            "query": "",
            "matches": [],
            "selectedIndex": -1,
            "selectedKey": None,
            "anchorRow": self._get_primary_scroll_view().scroll_top,
            # "query" | "retain" | "next" | "previous"
            "selectionMode": "query",
        }
        self._active_search = search
        search["overlay"] = self.show_overlay(
            component,
            {"anchor": "top-right", "width": "40%", "minWidth": 24, "margin": 1},
        )

    def _close_search(self) -> None:
        search = self._active_search
        if search is None:
            return
        self._active_search = None
        overlay = search.get("overlay")
        if overlay is not None:
            overlay.hide()
        self.request_render()

    def _update_search_query(self, query: str) -> None:
        search = self._active_search
        if search is None or query == search["query"]:
            return
        selected = (
            search["matches"][search["selectedIndex"]]
            if 0 <= search["selectedIndex"] < len(search["matches"])
            else None
        )
        search["anchorRow"] = (
            selected.segments[0].row
            if selected is not None and selected.segments
            else self._get_primary_scroll_view().scroll_top
        )
        search["query"] = query
        search["selectionMode"] = "query"
        search["component"].set_result(-1, 0)
        self.request_render()

    def _navigate_search(self, direction: int) -> None:
        search = self._active_search
        if search is None or not search["query"]:
            return
        search["selectionMode"] = "previous" if direction < 0 else "next"
        self.request_render()

    def _refresh_search(self, layout) -> bool:
        search = self._active_search
        if search is None:
            return False
        scroll_view = (
            layout.primary_scroll_view if layout.primary_scroll_view is not None else self._implicit_scroll_view
        )
        box = get_scroll_view_box(layout, scroll_view)
        lines = box.scroll_content_lines if box is not None else None
        if not lines or not search["query"].strip():
            search["matches"] = []
            search["selectedIndex"] = -1
            search["selectedKey"] = None
            search["selectionMode"] = "retain"
            search["component"].set_result(-1, 0)
            return False

        should_reveal_selection = search["selectionMode"] != "retain"
        matches = find_alt_screen_search_matches(lines, search["query"])
        exact_index = -1
        if search["selectedKey"] is not None:
            exact_index = next(
                (
                    index
                    for index, match in enumerate(matches)
                    if get_alt_screen_search_match_key(match) == search["selectedKey"]
                ),
                -1,
            )
        selected_index = -1
        if matches:
            if search["selectionMode"] == "query":
                selected_index = next(
                    (
                        index
                        for index, match in enumerate(matches)
                        if (match.segments[0].row if match.segments else 0) >= search["anchorRow"]
                    ),
                    -1,
                )
                selected_index = max(selected_index, 0)
            elif search["selectionMode"] == "next":
                base_index = exact_index if exact_index >= 0 else min(search["selectedIndex"], len(matches) - 1)
                selected_index = 0 if base_index < 0 else (base_index + 1) % len(matches)
            elif search["selectionMode"] == "previous":
                base_index = exact_index if exact_index >= 0 else min(search["selectedIndex"], len(matches) - 1)
                selected_index = len(matches) - 1 if base_index < 0 else (base_index - 1 + len(matches)) % len(matches)
            else:
                selected_index = (
                    exact_index if exact_index >= 0 else min(max(0, search["selectedIndex"]), len(matches) - 1)
                )

        search["matches"] = matches
        search["selectedIndex"] = selected_index
        search["selectedKey"] = (
            get_alt_screen_search_match_key(matches[selected_index]) if selected_index >= 0 else None
        )
        search["selectionMode"] = "retain"
        search["component"].set_result(selected_index, len(matches))
        if not should_reveal_selection:
            return False

        selected = matches[selected_index] if 0 <= selected_index < len(matches) else None
        first_segment = selected.segments[0] if selected is not None and selected.segments else None
        last_segment = selected.segments[-1] if selected is not None and selected.segments else None
        if box is None or first_segment is None or last_segment is None or scroll_view.viewport_height <= 0:
            return False
        before = scroll_view.scroll_top
        visible_bottom = before + scroll_view.viewport_height - 1
        target = before
        if first_segment.row < before or last_segment.row > visible_bottom:
            target = first_segment.row - scroll_view.viewport_height // 3
        scroll_view.scroll_to(target, {"disableFollow": True})
        return scroll_view.scroll_top != before

    def flash(self, message: str, duration_ms: float | None = None) -> None:
        """Show a transient message in the alternate-screen flash stack."""
        self._flashes.flash(message, duration_ms)

    def _should_defer_viewport_input_to_overlay(self) -> bool:
        search = self._active_search
        search_overlay = search.get("overlay") if search is not None else None
        search_focused = search_overlay is not None and search_overlay.is_focused()
        return self._is_overlay_focused() and not search_focused

    def _handle_viewport_input(self, data: str) -> dict | None:
        if data == FOCUS_OUT:
            had_active_selection = self._selection_press_active
            had_non_empty_active_selection = had_active_selection and self._get_selection_bounds() is not None
            self._selection_press_active = False
            self._stop_selection_auto_scroll()
            self._stop_scrollbar_hover()
            self._stop_scrollbar_drag()
            self._pressed_url = None
            self._selection_dragged = False
            if had_active_selection:
                self._selection_anchor = None
                self._selection_focus = None
                self._selection_granularity = "character"
                self._selection_initial_range = None
                if had_non_empty_active_selection:
                    self.request_render()
            self._last_click = None
            return {"consume": True}
        if data == FOCUS_IN:
            return {"consume": True}

        wheel_event = self._parse_wheel_event(data)
        if wheel_event:
            if self._should_defer_viewport_input_to_overlay():
                return None
            self._route_wheel(wheel_event)
            return {"consume": True}
        mouse_event = self._parse_sgr_mouse_event(data)
        if mouse_event:
            handled = self._handle_scrollbar_mouse_event(mouse_event)
            if self._scrollbar_drag is None:
                self._update_scrollbar_hover(mouse_event["x"], mouse_event["y"])
            if not handled:
                self._handle_selection_mouse_event(mouse_event)
            return {"consume": True}
        if self._is_mouse_sequence(data):
            return {"consume": True}

        keybindings = get_keybindings()
        is_release = is_key_release(data)
        if keybindings.matches(data, "tui.altScreen.search"):
            if not is_release:
                self._open_search()
            return {"consume": True}
        active_search = self._active_search
        search_overlay = active_search.get("overlay") if active_search is not None else None
        if search_overlay is not None and search_overlay.is_focused():
            if keybindings.matches(data, "tui.altScreen.searchNext"):
                if not is_release:
                    self._navigate_search(1)
                return {"consume": True}
            if keybindings.matches(data, "tui.altScreen.searchPrevious"):
                if not is_release:
                    self._navigate_search(-1)
                return {"consume": True}
            if keybindings.matches(data, "tui.altScreen.searchClose"):
                if not is_release:
                    self._close_search()
                return {"consume": True}
        if self._should_defer_viewport_input_to_overlay():
            return None
        if keybindings.matches(data, "tui.altScreen.pageUp"):
            if not is_release:
                self.scroll_by(-max(1, self._get_primary_scroll_view().viewport_height - PAGE_SCROLL_OVERLAP))
            return {"consume": True}
        if keybindings.matches(data, "tui.altScreen.pageDown"):
            if not is_release:
                self.scroll_by(max(1, self._get_primary_scroll_view().viewport_height - PAGE_SCROLL_OVERLAP))
            return {"consume": True}
        if keybindings.matches(data, "tui.altScreen.halfPageUp"):
            if not is_release:
                self.scroll_by(-max(1, self._get_primary_scroll_view().viewport_height // 2))
            return {"consume": True}
        if keybindings.matches(data, "tui.altScreen.halfPageDown"):
            if not is_release:
                self.scroll_by(max(1, self._get_primary_scroll_view().viewport_height // 2))
            return {"consume": True}
        if keybindings.matches(data, "tui.altScreen.lineUp"):
            if not is_release:
                self.scroll_by(-1)
            return {"consume": True}
        if keybindings.matches(data, "tui.altScreen.lineDown"):
            if not is_release:
                self.scroll_by(1)
            return {"consume": True}
        if keybindings.matches(data, "tui.altScreen.previousPrompt"):
            if not is_release:
                self._scroll_to_prompt(-1)
            return {"consume": True}
        if keybindings.matches(data, "tui.altScreen.nextPrompt"):
            if not is_release:
                self._scroll_to_prompt(1)
            return {"consume": True}
        if keybindings.matches(data, "tui.altScreen.top"):
            if not is_release:
                self.scroll_to_top()
            return {"consume": True}
        if keybindings.matches(data, "tui.altScreen.bottom"):
            if not is_release:
                self.scroll_to_bottom()
            return {"consume": True}
        return None

    def _parse_wheel_event(self, data: str) -> dict | None:
        sgr = _SGR_MOUSE_RE.match(data)
        if sgr:
            button = int(sgr.group(1))
            if (button & 64) == 0:
                return None
            direction = button & 3
            if direction not in (0, 1):
                return None
            return {
                "direction": -1 if direction == 0 else 1,
                "x": int(sgr.group(2)) - 1,
                "y": int(sgr.group(3)) - 1,
            }
        if len(data) == 6 and data.startswith("\x1b[M"):
            button = ord(data[3]) - 32
            if (button & 64) == 0:
                return None
            direction = button & 3
            if direction not in (0, 1):
                return None
            return {"direction": -1 if direction == 0 else 1, "x": ord(data[4]) - 33, "y": ord(data[5]) - 33}
        return None

    def _route_wheel(self, event: dict) -> None:
        remaining = event["direction"] * self._wheel_scroll_lines
        seen: list = []
        scroll_views = (
            get_scroll_views_at(self._current_layout, event["x"], event["y"])
            if self._current_layout is not None
            else []
        )
        for scroll_view in scroll_views:
            seen.append(scroll_view)
            remaining = scroll_view.scroll_by(remaining)
            if remaining == 0 or scroll_view.overscroll == "contain":
                break
        primary = self._get_primary_scroll_view()
        if remaining != 0 and all(scroll_view is not primary for scroll_view in seen):
            primary.scroll_by(remaining)
        self._update_scrollbar_hover(event["x"], event["y"])
        self.request_render()

    def _parse_sgr_mouse_event(self, data: str) -> dict | None:
        match = _SGR_MOUSE_RE.match(data)
        if not match:
            return None
        return {
            "button": int(match.group(1)),
            "x": int(match.group(2)) - 1,
            "y": int(match.group(3)) - 1,
            "release": match.group(4) == "m",
        }

    def _get_scrollbar_target_at(self, x: int, y: int) -> dict | None:
        """The scroll view whose scrollbar thumb covers (x, y): {"scrollView", "geometry"}."""
        if self.has_overlay() or self._current_layout is None:
            return None
        for scroll_view in get_scroll_views_at(self._current_layout, x, y):
            box = get_scroll_view_box(self._current_layout, scroll_view)
            geometry = get_scrollbar_geometry(box) if box is not None else None
            if (
                geometry
                and x == geometry["column"]
                and geometry["thumbTop"] <= y < geometry["thumbTop"] + geometry["thumbHeight"]
            ):
                return {"scrollView": scroll_view, "geometry": geometry}
        return None

    def _set_scrollbar_hover(self, scroll_view) -> None:
        if scroll_view is self._scrollbar_hover:
            return
        if self._scrollbar_hover is not None:
            self._scrollbar_hover.set_scrollbar_active(False)
        self._scrollbar_hover = scroll_view
        if self._scrollbar_hover is not None:
            self._scrollbar_hover.set_scrollbar_active(True)

    def _update_scrollbar_hover(self, x: int, y: int) -> None:
        target = self._get_scrollbar_target_at(x, y)
        self._set_scrollbar_hover(target["scrollView"] if target else None)

    def _stop_scrollbar_hover(self) -> None:
        self._set_scrollbar_hover(None)

    def _handle_scrollbar_mouse_event(self, event: dict) -> bool:
        if self._scrollbar_drag is not None:
            if event["release"]:
                self._stop_scrollbar_drag()
                return True
            box = (
                get_scroll_view_box(self._current_layout, self._scrollbar_drag["scrollView"])
                if self._current_layout is not None
                else None
            )
            geometry = get_scrollbar_geometry(box) if box is not None else None
            if geometry:
                max_thumb_offset = geometry["trackHeight"] - geometry["thumbHeight"]
                thumb_offset = max(
                    0,
                    min(max_thumb_offset, event["y"] - geometry["trackTop"] - self._scrollbar_drag["grabOffset"]),
                )
                scroll_top = (
                    0 if max_thumb_offset == 0 else js_round(thumb_offset / max_thumb_offset * geometry["maxScrollTop"])
                )
                self._scrollbar_drag["scrollView"].scroll_to(scroll_top)
            return True

        if event["release"] or (event["button"] & 32) != 0 or (event["button"] & 3) != 0:
            return False
        target = self._get_scrollbar_target_at(event["x"], event["y"])
        if not target:
            return False
        self._stop_selection_auto_scroll()
        self._selection_press_active = False
        self._selection_anchor = None
        self._selection_focus = None
        self._selection_granularity = "character"
        self._selection_initial_range = None
        self._last_click = None
        self._pressed_url = None
        self._selection_dragged = False
        self._set_scrollbar_hover(target["scrollView"])
        self._scrollbar_drag = {
            "scrollView": target["scrollView"],
            "grabOffset": event["y"] - target["geometry"]["thumbTop"],
        }
        return True

    def _stop_scrollbar_drag(self) -> None:
        self._scrollbar_drag = None

    def _get_scroll_selection_point(self, scroll_view, x: int, y: int) -> dict | None:
        if self._current_layout is None:
            return None
        box = get_scroll_view_box(self._current_layout, scroll_view)
        if box is None or box.rect.height <= 0 or box.clip.height <= 0:
            return None
        visible_top = max(0, box.rect.y, box.clip.y)
        visible_bottom = min(
            self.terminal.rows - 1,
            box.rect.y + box.rect.height - 1,
            box.clip.y + box.clip.height - 1,
        )
        if visible_bottom < visible_top:
            return None
        pointer_row = max(visible_top, min(visible_bottom, y))
        max_content_row = max(0, len(box.scroll_content_lines or [""]) - 1)
        return {
            "row": max(0, min(max_content_row, scroll_view.scroll_top + pointer_row - box.rect.y)),
            "col": max(0, min(box.rect.width - 1, x - box.rect.x)),
            "scrollView": scroll_view,
        }

    def _get_selection_point(self, event: dict, scroll_view=None) -> dict:
        if scroll_view is not None:
            point = self._get_scroll_selection_point(scroll_view, event["x"], event["y"])
            if point:
                return point
        return {
            "row": max(0, min(self.terminal.rows - 1, event["y"])),
            "col": max(0, min(self.terminal.columns - 1, event["x"])),
            "scrollView": None,
        }

    def _get_selection_source_line(self, point: dict) -> str:
        scroll_view = point.get("scrollView")
        if scroll_view is not None and self._current_layout is not None:
            box = get_scroll_view_box(self._current_layout, scroll_view)
            lines = box.scroll_content_lines if box is not None else None
            if lines is not None:
                return lines[point["row"]] if point["row"] < len(lines) else ""
        return self._previous_screen[point["row"]] if point["row"] < len(self._previous_screen) else ""

    def _get_word_selection(self, point: dict) -> dict | None:
        """Selection range {"start", "end"} covering the word under `point`."""
        line = strip_terminal_sequences(self._get_selection_source_line(point))
        segments: list[dict] = []
        start = 0
        for segment in _word_segmenter.segment(line):
            end = start + visible_width(segment["segment"])
            joiner = segment["segment"] in TERMINAL_WORD_SELECTION_JOINERS
            segments.append(
                {"start": start, "end": end, "selectable": segment["isWordLike"] is True or joiner, "joiner": joiner}
            )
            start = end
        clicked_segment_index = next(
            (i for i, segment in enumerate(segments) if segment["start"] <= point["col"] < segment["end"]), -1
        )
        if clicked_segment_index < 0:
            return None

        def can_join(left: dict, right: dict) -> bool:
            return left["selectable"] and right["selectable"] and (left["joiner"] or right["joiner"])

        selection_start = segments[clicked_segment_index]["start"]
        selection_end = segments[clicked_segment_index]["end"]
        index = clicked_segment_index
        while index > 0 and can_join(segments[index - 1], segments[index]):
            selection_start = segments[index - 1]["start"]
            index -= 1
        index = clicked_segment_index
        while index < len(segments) - 1 and can_join(segments[index], segments[index + 1]):
            selection_end = segments[index + 1]["end"]
            index += 1
        return {
            "start": {**point, "col": selection_start},
            "end": {**point, "col": selection_end, "boundary": True},
        }

    def _get_line_selection(self, point: dict) -> dict:
        return {
            "start": {**point, "col": 0},
            "end": {**point, "col": visible_width(self._get_selection_source_line(point)), "boundary": True},
        }

    def _update_selection_focus(self, point: dict) -> None:
        if self._selection_granularity == "character" or not self._selection_initial_range:
            self._selection_focus = point
            return
        selection_range = (
            self._get_word_selection(point)
            if self._selection_granularity == "word"
            else self._get_line_selection(point)
        )
        if not selection_range:
            return
        initial = self._selection_initial_range
        target_before_initial = selection_range["start"]["row"] < initial["start"]["row"] or (
            selection_range["start"]["row"] == initial["start"]["row"]
            and selection_range["start"]["col"] < initial["start"]["col"]
        )
        if target_before_initial:
            self._selection_anchor = initial["end"]
            self._selection_focus = selection_range["start"]
        else:
            self._selection_anchor = initial["start"]
            self._selection_focus = selection_range["end"]

    def _get_click_count(self, point: dict, word: dict | None) -> int:
        """pi uses `Date.now()`; a monotonic clock is the right primitive for an
        interval and cannot be moved by a wall-clock adjustment."""
        now = time.monotonic()
        previous = self._last_click
        if (
            word
            and previous
            and now - previous["timestamp"] <= DOUBLE_CLICK_INTERVAL_S
            and previous["row"] == point["row"]
            and previous["scrollView"] is point.get("scrollView")
            and previous["wordStart"] == word["start"]["col"]
            and previous["wordEnd"] == word["end"]["col"]
        ):
            count = (previous["count"] % 3) + 1
        else:
            count = 1
        self._last_click = (
            {
                "timestamp": now,
                "count": count,
                "row": point["row"],
                "scrollView": point.get("scrollView"),
                "wordStart": word["start"]["col"],
                "wordEnd": word["end"]["col"],
            }
            if word
            else None
        )
        return count

    def _update_selection_auto_scroll(self, event: dict) -> None:
        scroll_view = self._selection_anchor.get("scrollView") if self._selection_anchor else None
        if scroll_view is None or self._current_layout is None:
            self._stop_selection_auto_scroll()
            return
        box = get_scroll_view_box(self._current_layout, scroll_view)
        if box is None or box.rect.height <= 0 or box.clip.height <= 0:
            self._stop_selection_auto_scroll()
            return
        visible_top = max(0, box.rect.y, box.clip.y)
        visible_bottom = min(
            self.terminal.rows - 1,
            box.rect.y + box.rect.height - 1,
            box.clip.y + box.clip.height - 1,
        )
        self._selection_drag_pointer = {"x": event["x"], "y": event["y"]}
        if event["y"] <= visible_top:
            self._selection_auto_scroll_direction = -1
        elif event["y"] >= visible_bottom:
            self._selection_auto_scroll_direction = 1
        else:
            self._selection_auto_scroll_direction = 0
        if self._selection_auto_scroll_direction == 0:
            self._stop_selection_auto_scroll()
            return
        if self._selection_auto_scroll_timer is not None:
            return
        self._selection_auto_scroll_timer = Interval(50, self._auto_scroll_selection)

    async def _auto_scroll_selection(self) -> None:
        scroll_view = self._selection_anchor.get("scrollView") if self._selection_anchor else None
        pointer = self._selection_drag_pointer
        direction = self._selection_auto_scroll_direction
        if scroll_view is None or pointer is None or direction == 0:
            self._stop_selection_auto_scroll()
            return
        remaining = scroll_view.scroll_by(direction)
        if remaining == direction:
            self._stop_selection_auto_scroll()
            return
        point = self._get_scroll_selection_point(scroll_view, pointer["x"], pointer["y"])
        if point:
            self._update_selection_focus(point)
        self.request_render()

    def _stop_selection_auto_scroll(self) -> None:
        if self._selection_auto_scroll_timer is not None:
            self._selection_auto_scroll_timer.cancel()
            self._selection_auto_scroll_timer = None
        self._selection_auto_scroll_direction = 0
        self._selection_drag_pointer = None

    def _handle_selection_mouse_event(self, event: dict) -> None:
        button = event["button"] & 3
        if button != 0 and not (event["release"] and button == 3):
            return
        anchor_scroll_view = self._selection_anchor.get("scrollView") if self._selection_anchor else None
        point = self._get_selection_point(event, anchor_scroll_view)
        if event["release"]:
            if not self._selection_press_active:
                return
            self._selection_press_active = False
            self._stop_selection_auto_scroll()
            if not self._selection_anchor:
                return
            self._update_selection_focus(point)
            clicked_url = (
                self._pressed_url
                if (
                    not self._selection_dragged
                    and self._selection_anchor.get("scrollView") is point.get("scrollView")
                    and self._selection_anchor["row"] == point["row"]
                    and self._selection_anchor["col"] == point["col"]
                )
                else None
            )
            self._pressed_url = None
            if clicked_url and self._open_url:
                self._selection_anchor = None
                self._selection_focus = None
                try:
                    self._open_url(clicked_url)
                except Exception:
                    # URL activation is best-effort.
                    pass
                self.request_render()
                return
            if self._copy_on_select:
                tonio.spawn.without_tracking(self._copy_selection_to_clipboard())
            self.request_render()
            return
        if (event["button"] & 32) != 0:
            if not self._selection_press_active or not self._selection_anchor:
                return
            self._selection_dragged = True
            self._last_click = None
            self._pressed_url = None
            self._update_selection_focus(point)
            self._update_selection_auto_scroll(event)
            self.request_render()
            return
        self._stop_selection_auto_scroll()
        self._selection_press_active = True
        scroll_view = None
        if not self.has_overlay() and self._current_layout is not None:
            hits = get_scroll_views_at(self._current_layout, event["x"], event["y"])
            scroll_view = hits[0] if hits else None
        anchor = self._get_selection_point(event, scroll_view)
        word = self._get_word_selection(anchor)
        click_count = self._get_click_count(anchor, word)
        selection_range = word if click_count == 2 else self._get_line_selection(anchor) if click_count == 3 else None
        self._selection_granularity = ("word" if click_count == 2 else "line") if selection_range else "character"
        self._selection_initial_range = selection_range
        self._selection_anchor = selection_range["start"] if selection_range else anchor
        self._selection_focus = selection_range["end"] if selection_range else anchor
        self._selection_dragged = False
        if selection_range:
            self._pressed_url = None
        else:
            screen_row = max(0, min(self.terminal.rows - 1, event["y"]))
            self._pressed_url = get_osc8_link_at_column(
                self._previous_screen[screen_row] if screen_row < len(self._previous_screen) else "",
                max(0, min(self.terminal.columns - 1, event["x"])),
            )
        self.request_render()

    def _get_selection_bounds(self) -> dict | None:
        if not self._selection_anchor or not self._selection_focus:
            return None
        anchor = self._selection_anchor
        focus = self._selection_focus
        if anchor.get("scrollView") is not focus.get("scrollView"):
            return None
        anchor_before_focus = anchor["row"] < focus["row"] or (
            anchor["row"] == focus["row"] and anchor["col"] < focus["col"]
        )
        if anchor["row"] == focus["row"] and anchor["col"] == focus["col"]:
            return None
        return {"start": anchor, "end": focus} if anchor_before_focus else {"start": focus, "end": anchor}

    def _get_selection_columns(
        self, line: str, row: int, selection: dict, min_column: int = 0, max_column: int | None = None
    ) -> tuple[int, int]:
        line_width = visible_width(line)
        if max_column is None:
            max_column = line_width
        start = max(0, min_column)
        end = min(line_width, max_column)
        if row == selection["start"]["row"]:
            cell_range = get_grapheme_cell_range(line, selection["start"]["col"])
            start = cell_range[0] if cell_range else min(selection["start"]["col"], line_width)
        if row == selection["end"]["row"]:
            if selection["end"].get("boundary"):
                end = min(selection["end"]["col"], line_width)
            else:
                cell_range = get_grapheme_cell_range(line, selection["end"]["col"])
                end = cell_range[1] if cell_range else min(selection["end"]["col"] + 1, line_width)
        return max(min_column, start), min(max_column, end)

    def _get_active_selection_text(self) -> str | None:
        selection = self._get_selection_bounds()
        if not selection:
            return None
        source_lines: list[str] = self._previous_screen
        if selection["start"].get("scrollView") is not None:
            if self._current_layout is None:
                return None
            box = get_scroll_view_box(self._current_layout, selection["start"]["scrollView"])
            if box is None or box.scroll_content_lines is None:
                return None
            source_lines = box.scroll_content_lines
        lines: list[str] = []
        for row in range(selection["start"]["row"], selection["end"]["row"] + 1):
            line = source_lines[row] if row < len(source_lines) else ""
            start, end = self._get_selection_columns(line, row, selection)
            lines.append(strip_terminal_sequences(slice_by_column(line, start, max(0, end - start), True)).rstrip())
        text = "\n".join(lines)
        return None if len(text) == 0 else text

    async def _copy_selection_to_clipboard(self) -> bool:
        text = self._get_active_selection_text()
        if text is None:
            return False
        return await self._copy_text_to_clipboard(text)

    async def _copy_text_to_clipboard(self, text: str) -> bool:
        # Prefer an injected clipboard implementation (native clipboard +
        # platform tools with a verified success path) when the host app
        # provides one. A bare OSC 52 write can show "Copied!" while leaving
        # the system clipboard untouched (e.g. macOS Terminal.app, tmux
        # without OSC 52 clipboard passthrough), so only report success when
        # it actually copies.
        if self._copy_selection is not None:
            ok = await self._copy_selection(text)
            self.flash("Copied!" if ok else "Copy failed")
            return ok
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        await self.terminal.write(f"\x1b]52;c;{encoded}\x07")
        self.flash("Copied!")
        return True

    def _apply_search_text_highlight(self, text: str, current: bool) -> str:
        style = self._search_current_match_style if current else self._search_match_style
        result = ""
        plain_start = 0
        index = 0
        while index < len(text):
            ansi = extract_ansi_code(text, index)
            if not ansi:
                index += 1
                continue
            if index > plain_start:
                result += style(text[plain_start:index])
            result += ansi["code"]
            index += ansi["length"]
            plain_start = index
        if plain_start < len(text):
            result += style(text[plain_start:])
        return result

    def _apply_search_highlights(self, screen: list[str], layout) -> list[str]:
        search = self._active_search
        if search is None or search["selectedIndex"] < 0 or not search["matches"]:
            return screen
        scroll_view = (
            layout.primary_scroll_view if layout.primary_scroll_view is not None else self._implicit_scroll_view
        )
        box = get_scroll_view_box(layout, scroll_view)
        if box is None:
            return screen

        ranges_by_row: dict[int, list[dict]] = {}
        scrollbar_geometry = get_scrollbar_geometry(box)
        scrollbar_column = scrollbar_geometry["column"] if scrollbar_geometry is not None else None
        min_row = max(0, box.rect.y, box.clip.y)
        max_row = min(len(screen), box.rect.y + box.rect.height, box.clip.y + box.clip.height)
        min_column = max(0, box.rect.x, box.clip.x)
        max_column = min(
            self.terminal.columns,
            box.rect.x + box.rect.width,
            box.clip.x + box.clip.width,
            scrollbar_column if scrollbar_column is not None else math.inf,
        )
        for match_index, match in enumerate(search["matches"]):
            for segment in match.segments:
                row = box.rect.y + segment.row - scroll_view.scroll_top
                if row < min_row or row >= max_row:
                    continue
                start_col = max(min_column, box.rect.x + segment.start_col)
                end_col = min(max_column, box.rect.x + segment.end_col)
                if end_col <= start_col:
                    continue
                ranges_by_row.setdefault(row, []).append(
                    {"startCol": start_col, "endCol": end_col, "current": match_index == search["selectedIndex"]}
                )

        result = list(screen)
        for row, ranges in ranges_by_row.items():
            line = result[row] if row < len(result) else ""
            if is_image_line(line):
                continue
            line_width = visible_width(line)
            for range_ in sorted(ranges, key=lambda item: -item["startCol"]):
                start_col = int(min(range_["startCol"], line_width))
                end_col = int(min(range_["endCol"], line_width))
                if end_col <= start_col:
                    continue
                before = slice_by_column(line, 0, start_col, True)
                highlighted = slice_by_column(line, start_col, end_col - start_col, True)
                after = slice_by_column(line, end_col, max(0, line_width - end_col), True)
                line = f"{before}{self._apply_search_text_highlight(highlighted, range_['current'])}{after}"
            result[row] = line
        return result

    def _apply_selection_highlight(self, text: str) -> str:
        result = "\x1b[7m"
        index = 0
        while index < len(text):
            ansi = extract_ansi_code(text, index)
            if not ansi:
                result += text[index]
                index += 1
                continue
            result += ansi["code"]
            if ansi["code"].endswith("m"):
                result += "\x1b[7m"
            index += ansi["length"]
        return f"{result}\x1b[27m"

    def _apply_selection(self, screen: list[str], layout=None) -> list[str]:
        if layout is None:
            layout = self._current_layout
        selection = self._get_selection_bounds()
        if not selection:
            return screen
        screen_selection = selection
        min_row = 0
        max_row = len(screen) - 1
        min_column = 0
        max_column = self.terminal.columns
        if selection["start"].get("scrollView") is not None:
            if layout is None:
                return screen
            scroll_view = selection["start"]["scrollView"]
            box = get_scroll_view_box(layout, scroll_view)
            if box is None:
                return screen
            min_row = max(0, box.rect.y, box.clip.y)
            max_row = min(len(screen) - 1, box.rect.y + box.rect.height - 1, box.clip.y + box.clip.height - 1)
            min_column = max(0, box.rect.x, box.clip.x)
            max_column = min(self.terminal.columns, box.rect.x + box.rect.width, box.clip.x + box.clip.width)
            screen_selection = {
                "start": {
                    **selection["start"],
                    "row": box.rect.y + selection["start"]["row"] - scroll_view.scroll_top,
                    "col": box.rect.x + selection["start"]["col"],
                },
                "end": {
                    **selection["end"],
                    "row": box.rect.y + selection["end"]["row"] - scroll_view.scroll_top,
                    "col": box.rect.x + selection["end"]["col"],
                },
            }
        result: list[str] = []
        for row, line in enumerate(screen):
            if (
                row < min_row
                or row > max_row
                or row < screen_selection["start"]["row"]
                or row > screen_selection["end"]["row"]
                or is_image_line(line)
            ):
                result.append(line)
                continue
            line_width = visible_width(line)
            start, end = self._get_selection_columns(line, row, screen_selection, min_column, max_column)
            if end <= start:
                result.append(line)
                continue
            before = slice_by_column(line, 0, start, True)
            selected = slice_by_column(line, start, end - start, True)
            after = slice_by_column(line, end, max(0, line_width - end), True)
            result.append(f"{before}{self._apply_selection_highlight(selected)}{after}")
        return result

    def _is_mouse_sequence(self, data: str) -> bool:
        return bool(_SGR_MOUSE_RE.match(data)) or (len(data) == 6 and data.startswith("\x1b[M"))

    def _composite_flashes(self, screen: list[str], width: int, height: int) -> list[str]:
        flash_lines = self._flashes.render(width)[-height:]
        if not flash_lines:
            return screen
        result = list(screen)
        while len(result) < height:
            result.append("")
        for row, line in enumerate(flash_lines):
            flash_width = visible_width(line)
            if flash_width == 0:
                continue
            result[row] = composite_tui_line(result[row], line, width - flash_width, flash_width, width)
        return result

    async def _do_render(self) -> None:
        if self._stopped or not self._alt_screen_active:
            return
        width = max(1, self.terminal.columns)
        height = max(1, self.terminal.rows)
        root = self._layout_root if self._layout_root is not None else self._implicit_scroll_view
        next_layout = render_layout_frame(root, width, height, self.request_render)
        if self._refresh_search(next_layout):
            next_layout = render_layout_frame(root, width, height, self.request_render)
        screen = [OSC133_ZONE_PREFIX.sub("", line) for line in next_layout.lines]
        screen = self._apply_search_highlights(screen, next_layout)
        screen = self._composite_overlays(screen, width, height)
        if len(screen) > height:
            screen = screen[len(screen) - height :]
        screen = self._apply_selection(screen, next_layout)
        screen = self._composite_flashes(screen, width, height)

        cursor_pos = self._extract_cursor_position(screen, height)
        screen = [
            line if is_image_line(line) or visible_width(line) <= width else slice_by_column(line, 0, width, True)
            for line in self._apply_line_resets(screen)
        ]

        full_redraw = (
            not self._previous_screen or self._previous_screen_width != width or self._previous_screen_height != height
        )
        images_need_redraw = any(
            line != self._previous_of(row) and (is_image_line(line) or is_image_line(self._previous_of(row)))
            for row, line in enumerate(screen)
        )

        redraw_images = full_redraw or images_need_redraw
        had_uploaded_kitty_images = bool(self._uploaded_kitty_images)
        if redraw_images and self._image_protocol == "kitty":
            prepared_lines, evicted_image_deletion = self._prepare_kitty_screen(screen)
        else:
            prepared_lines, evicted_image_deletion = screen, ""

        buffer = BEGIN_SYNCHRONIZED_OUTPUT
        if full_redraw:
            self._full_redraw_count += 1
            clear_images = (
                delete_all_kitty_placements()
                if self._image_protocol == "kitty" and had_uploaded_kitty_images
                else self._delete_kitty_images()
            )
            buffer += f"{clear_images}\x1b[2J"
        elif images_need_redraw:
            if self._image_protocol == "iterm2":
                buffer += "\x1b[2J"
            elif self._image_protocol == "kitty":
                buffer += delete_all_kitty_placements()
        buffer += evicted_image_deletion

        for row in range(height):
            if not full_redraw and not images_need_redraw and screen[row] == self._previous_of(row):
                continue
            buffer += f"\x1b[{row + 1};1H\x1b[2K{prepared_lines[row] if row < len(prepared_lines) else ''}"

        if cursor_pos:
            buffer += f"\x1b[{cursor_pos['row'] + 1};{min(width, cursor_pos['col']) + 1}H"
            buffer += "\x1b[?25h" if self.get_show_hardware_cursor() else "\x1b[?25l"
        else:
            buffer += "\x1b[?25l"
        buffer += END_SYNCHRONIZED_OUTPUT
        # Publish the layout before the frame goes out: readers that wake on
        # the write (`_get_primary_scroll_view`, input handlers keyed on the
        # primary scroll view) must see the layout that frame was drawn from.
        # pi assigns after the write; on one thread nothing can look between.
        self._current_layout = next_layout
        self._emit(buffer)

        self._previous_screen = screen
        self._previous_screen_width = width
        self._previous_screen_height = height

    def _previous_of(self, row: int) -> str:
        """pi indexes `previousScreen[row]` past the end and gets undefined."""
        return self._previous_screen[row] if row < len(self._previous_screen) else ""


setattr(TuiAltScreen, VIEWPORT_TUI, True)
