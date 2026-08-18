"""Overlay QA tests.

Comprehensive overlay positioning and edge case tests:

    /overlay-animation  - Real-time animation demo (~30 FPS, proves game-like rendering works)
    /overlay-anchors    - Cycle through all 9 anchor positions
    /overlay-margins    - Test margin and offset options
    /overlay-stack      - Test stacked overlays
    /overlay-overflow   - Test width overflow with streaming process output
    /overlay-edge       - Test overlay positioned at terminal edge
    /overlay-percent    - Test percentage-based positioning
    /overlay-maxheight  - Test maxHeight truncation
    /overlay-sidepanel  - Responsive sidepanel (hides when terminal < 100 cols)
    /overlay-toggle     - Toggle visibility demo (demonstrates OverlayHandle.set_hidden)
    /overlay-passive    - Non-capturing overlay demo (passive info panel alongside active overlay)
    /overlay-focus      - Focus cycling, input routing, dismissal, and rendering order with overlays
    /overlay-streaming  - Multiple input panels with simulated streaming (Tab to cycle focus)

Start pidrei with this extension:
    pidrei -e ./examples/extensions/overlay_qa_tests.py
"""

import colorsys
import math
import subprocess
import time

import tonio.colored as tonio

from pidrei_tui import Input, matches_key, truncate_to_width, visible_width
from pidrei_tui._timers import Interval, Timeout


ANCHORS = [
    "top-left",
    "top-center",
    "top-right",
    "left-center",
    "center",
    "right-center",
    "bottom-left",
    "bottom-center",
    "bottom-right",
]


class BaseOverlay:
    """Base overlay component with common box rendering."""

    def __init__(self, theme) -> None:
        self._theme = theme

    def box(self, lines: list[str], width: int, title: str | None = None) -> list[str]:
        th = self._theme
        inner_w = max(1, width - 2)
        result: list[str] = []

        title_str = truncate_to_width(f" {title} ", inner_w) if title else ""
        title_w = visible_width(title_str)
        top_left = "─" * ((inner_w - title_w) // 2)
        top_right = "─" * max(0, inner_w - title_w - len(top_left))
        result.append(th.fg("border", f"╭{top_left}") + th.fg("accent", title_str) + th.fg("border", f"{top_right}╮"))

        for line in lines:
            result.append(th.fg("border", "│") + truncate_to_width(line, inner_w, "...", True) + th.fg("border", "│"))

        result.append(th.fg("border", f"╰{'─' * inner_w}╯"))
        return result

    def invalidate(self) -> None:
        pass

    def dispose(self) -> None:
        pass


# Anchor position test
class AnchorTestComponent(BaseOverlay):
    def __init__(self, theme, anchor: str, done) -> None:
        super().__init__(theme)
        self._anchor = anchor
        self._done = done

    async def handle_input(self, data: str) -> None:
        if matches_key(data, "escape") or matches_key(data, "ctrl+c"):
            self._done("cancel")
        elif matches_key(data, "return"):
            self._done("confirm")
        elif matches_key(data, "space") or matches_key(data, "right"):
            self._done("next")

    def render(self, width: int) -> list[str]:
        th = self._theme
        return self.box(
            [
                "",
                f" Current: {th.fg('accent', self._anchor)}",
                "",
                f" {th.fg('dim', 'Space/→ = next anchor')}",
                f" {th.fg('dim', 'Enter = confirm')}",
                f" {th.fg('dim', 'Esc = cancel')}",
                "",
            ],
            width,
            "Anchor Test",
        )


# Margin/offset test
class MarginTestComponent(BaseOverlay):
    def __init__(self, theme, config: dict, done) -> None:
        super().__init__(theme)
        self._config = config
        self._done = done

    async def handle_input(self, data: str) -> None:
        if matches_key(data, "escape") or matches_key(data, "ctrl+c"):
            self._done("close")
        elif matches_key(data, "space") or matches_key(data, "right"):
            self._done("next")

    def render(self, width: int) -> list[str]:
        th = self._theme
        return self.box(
            [
                "",
                f" {th.fg('accent', self._config['name'])}",
                "",
                f" {th.fg('dim', 'Space/→ = next config')}",
                f" {th.fg('dim', 'Esc = close')}",
                "",
            ],
            width,
            "Margin Test",
        )


# Stacked overlay test
class StackOverlayComponent(BaseOverlay):
    def __init__(self, theme, num: int, position: str, done) -> None:
        super().__init__(theme)
        self._num = num
        self._position = position
        self._done = done

    async def handle_input(self, data: str) -> None:
        if matches_key(data, "escape") or matches_key(data, "ctrl+c") or matches_key(data, "return"):
            self._done(f"Overlay {self._num}")

    def render(self, width: int) -> list[str]:
        th = self._theme
        # Use different colors for each overlay to show stacking
        color = ("error", "success", "accent")[(self._num - 1) % 3]
        inner_w = max(1, width - 2)
        border = lambda char: th.fg(color, char)
        pad_line = lambda s: truncate_to_width(s, inner_w, "...", True)
        lines: list[str] = []

        lines.append(border(f"╭{'─' * inner_w}╮"))
        lines.append(border("│") + pad_line(f" Overlay {th.fg('accent', f'#{self._num}')}") + border("│"))
        lines.append(border("│") + pad_line(f" Layer: {th.fg(color, self._position)}") + border("│"))
        lines.append(border("│") + pad_line("") + border("│"))
        # Add extra lines to make it taller
        for _ in range(5):
            lines.append(border("│") + pad_line(f" {'░' * (inner_w - 2)} ") + border("│"))
        lines.append(border("│") + pad_line("") + border("│"))
        lines.append(border("│") + pad_line(th.fg("dim", " Press Enter/Esc to close")) + border("│"))
        lines.append(border(f"╰{'─' * inner_w}╯"))

        return lines


# Streaming overflow test - spawns a real process with colored output (the
# original crash scenario). Produces many lines with ANSI colors, OSC 8
# hyperlinks, and long paths that exceed the overlay width.
_OVERFLOW_SCRIPT = r"""
echo "Starting streaming overflow test (30+ seconds)..."
echo "This simulates subagent output with colors, hyperlinks, and long paths"
echo ""
for i in $(seq 1 100); do
    # Simulate long file paths with OSC 8 hyperlinks (clickable) - tests width overflow
    DIR="/home/user/development/pidrei/packages/pidrei/pidrei/modes/interactive"
    FILE="${DIR}/components/very-long-component-name-that-exceeds-width-${i}.py"
    echo -e "\033]8;;file://${FILE}\007▶ read: ${FILE}\033]8;;\007"

    # Add some colored status messages with long text
    if [ $((i % 5)) -eq 0 ]; then
        echo -e "  \033[32m✓ Successfully processed ${i} files in /home/user/development/pidrei\033[0m"
    fi
    if [ $((i % 7)) -eq 0 ]; then
        echo -e "  \033[33m⚠ Warning: potential issue detected at line ${i} in very-long-component-name-that-exceeds-width.py\033[0m"
    fi
    if [ $((i % 11)) -eq 0 ]; then
        echo -e "  \033[31m✗ Error: file not found /some/really/long/path/that/definitely/exceeds/the/overlay/width/limit/file-${i}.py\033[0m"
    fi
    sleep 0.3
done
echo ""
echo -e "\033[32m✓ Complete - 100 files processed in 30 seconds\033[0m"
echo "Press Esc to close"
"""


class StreamingOverflowComponent(BaseOverlay):
    def __init__(self, tui, theme, done) -> None:
        super().__init__(theme)
        self._tui = tui
        self._done = done
        self._lines: list[str] = []
        self._process = None
        self._scroll_offset = 0
        self._max_visible_lines = 15
        self._finished = False
        self._disposed = False
        tonio.spawn.without_tracking(self._run_process())

    async def _run_process(self) -> None:
        self._process = await tonio.open_process(
            ["bash", "-c", _OVERFLOW_SCRIPT],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Drain both streams concurrently, then mark finished
        await tonio.spawn.without_results(
            self._pump(self._process.stdout, is_error=False),
            self._pump(self._process.stderr, is_error=True),
        )
        await self._process.wait()
        if self._disposed:  # Guard against callbacks after dispose
            return
        self._finished = True
        self._tui.request_render()

    async def _pump(self, stream, *, is_error: bool) -> None:
        buffer = ""
        with stream as source:
            while True:
                chunk = await source.receive_some()
                if not chunk:
                    break
                if self._disposed:  # Guard against callbacks after dispose
                    return
                buffer += bytes(chunk).decode("utf-8", "replace")
                *complete, buffer = buffer.split("\n")
                for line in complete:
                    if line:
                        self._lines.append(self._theme.fg("error", line.strip()) if is_error else line)
                # Auto-scroll to bottom
                self._scroll_offset = max(0, len(self._lines) - self._max_visible_lines)
                self._tui.request_render()

    async def handle_input(self, data: str) -> None:
        if matches_key(data, "escape") or matches_key(data, "ctrl+c"):
            self._kill_process()
            self._done(None)
        elif matches_key(data, "up"):
            self._scroll_offset = max(0, self._scroll_offset - 1)
            self._tui.request_render()  # Trigger re-render after scroll
        elif matches_key(data, "down"):
            max_offset = max(0, len(self._lines) - self._max_visible_lines)
            self._scroll_offset = min(max_offset, self._scroll_offset + 1)
            self._tui.request_render()  # Trigger re-render after scroll

    def render(self, width: int) -> list[str]:
        th = self._theme
        inner_w = max(1, width - 2)
        pad_line = lambda s: truncate_to_width(s, inner_w, "...", True)
        border = lambda c: th.fg("border", c)

        result: list[str] = []
        title = truncate_to_width(f" Streaming Output ({len(self._lines)} lines) ", inner_w)
        title_pad = max(0, inner_w - visible_width(title))
        result.append(border("╭") + th.fg("accent", title) + border(f"{'─' * title_pad}╮"))

        # Scroll indicators
        can_scroll_up = self._scroll_offset > 0
        can_scroll_down = self._scroll_offset < len(self._lines) - self._max_visible_lines
        below = max(0, len(self._lines) - self._max_visible_lines - self._scroll_offset)
        scroll_info = f"↑{self._scroll_offset} | ↓{below}"

        scroll_line = th.fg("dim", f" {scroll_info}") if can_scroll_up or can_scroll_down else ""
        result.append(border("│") + pad_line(scroll_line) + border("│"))

        # Visible lines - truncate long lines to fit within border
        visible_lines = self._lines[self._scroll_offset : self._scroll_offset + self._max_visible_lines]
        for line in visible_lines:
            result.append(border("│") + pad_line(f" {line}") + border("│"))

        # Pad to max visible lines
        for _ in range(len(visible_lines), self._max_visible_lines):
            result.append(border("│") + pad_line("") + border("│"))

        status = th.fg("success", "✓ Done") if self._finished else th.fg("warning", "● Running")
        result.append(border("│") + pad_line(f" {status} {th.fg('dim', '| ↑↓ scroll | Esc close')}") + border("│"))
        result.append(border(f"╰{'─' * inner_w}╯"))

        return result

    def _kill_process(self) -> None:
        if self._process is not None and self._process.returncode is None:
            self._process.kill()

    def dispose(self) -> None:
        self._disposed = True
        self._kill_process()


# Edge position test
class EdgeTestComponent(BaseOverlay):
    def __init__(self, theme, done) -> None:
        super().__init__(theme)
        self._done = done

    async def handle_input(self, data: str) -> None:
        if matches_key(data, "escape") or matches_key(data, "ctrl+c"):
            self._done(None)

    def render(self, width: int) -> list[str]:
        th = self._theme
        return self.box(
            [
                "",
                " This overlay is at the",
                " right edge of terminal.",
                "",
                f" {th.fg('dim', 'Verify right border')}",
                f" {th.fg('dim', 'aligns with edge.')}",
                "",
                f" {th.fg('dim', 'Press Esc to close')}",
                "",
            ],
            width,
            "Edge Test",
        )


# Percentage positioning test
class PercentTestComponent(BaseOverlay):
    def __init__(self, theme, config: dict, done) -> None:
        super().__init__(theme)
        self._config = config
        self._done = done

    async def handle_input(self, data: str) -> None:
        if matches_key(data, "escape") or matches_key(data, "ctrl+c"):
            self._done("close")
        elif matches_key(data, "space") or matches_key(data, "right"):
            self._done("next")

    def render(self, width: int) -> list[str]:
        th = self._theme
        return self.box(
            [
                "",
                f" {th.fg('accent', self._config['name'])}",
                "",
                f" {th.fg('dim', 'Space/→ = next')}",
                f" {th.fg('dim', 'Esc = close')}",
                "",
            ],
            width,
            "Percent Test",
        )


# MaxHeight test - renders 21 lines, truncated to 10 by maxHeight
class MaxHeightTestComponent(BaseOverlay):
    def __init__(self, theme, done) -> None:
        super().__init__(theme)
        self._done = done

    async def handle_input(self, data: str) -> None:
        if matches_key(data, "escape") or matches_key(data, "ctrl+c"):
            self._done(None)

    def render(self, width: int) -> list[str]:
        th = self._theme
        # Intentionally render 21 lines - maxHeight: 10 will truncate to first 10
        # You should see header + lines 1-6, with bottom border cut off
        content_lines = [
            th.fg("warning", " ⚠ Rendering 21 lines, maxHeight: 10"),
            th.fg("dim", " Lines 11-21 truncated (no bottom border)"),
            "",
        ]

        for i in range(1, 15):
            content_lines.append(f" Line {i} of 14")

        content_lines.extend(["", th.fg("dim", " Press Esc to close")])

        return self.box(content_lines, width, "MaxHeight Test")


# Responsive sidepanel - demonstrates percentage width and visibility callback
class SidepanelComponent(BaseOverlay):
    def __init__(self, tui, theme, done) -> None:
        super().__init__(theme)
        self._tui = tui
        self._done = done
        self._items = ["Dashboard", "Messages", "Settings", "Help", "About"]
        self._selected_index = 0

    async def handle_input(self, data: str) -> None:
        if matches_key(data, "escape") or matches_key(data, "ctrl+c"):
            self._done(None)
        elif matches_key(data, "up"):
            self._selected_index = max(0, self._selected_index - 1)
            self._tui.request_render()
        elif matches_key(data, "down"):
            self._selected_index = min(len(self._items) - 1, self._selected_index + 1)
            self._tui.request_render()
        elif matches_key(data, "return"):
            # Could trigger an action here
            self._tui.request_render()

    def render(self, width: int) -> list[str]:
        th = self._theme
        inner_w = max(1, width - 2)
        pad_line = lambda s: truncate_to_width(s, inner_w, "...", True)
        border = lambda c: th.fg("border", c)
        lines: list[str] = []

        # Header
        lines.append(border(f"╭{'─' * inner_w}╮"))
        lines.append(border("│") + pad_line(th.fg("accent", " Responsive Sidepanel")) + border("│"))
        lines.append(border("├") + border("─" * inner_w) + border("┤"))

        # Menu items
        for i, item in enumerate(self._items):
            is_selected = i == self._selected_index
            prefix = th.fg("accent", "→ ") if is_selected else "  "
            text = th.fg("accent", item) if is_selected else item
            lines.append(border("│") + pad_line(f"{prefix}{text}") + border("│"))

        # Footer with responsive behavior info
        lines.append(border("├") + border("─" * inner_w) + border("┤"))
        lines.append(border("│") + pad_line(th.fg("warning", " ⚠ Resize terminal < 100 cols")) + border("│"))
        lines.append(border("│") + pad_line(th.fg("warning", "   to see panel auto-hide")) + border("│"))
        lines.append(border("│") + pad_line(th.fg("dim", " Uses visible: cols >= 100")) + border("│"))
        lines.append(border("│") + pad_line(th.fg("dim", " ↑↓ navigate | Esc close")) + border("│"))
        lines.append(border(f"╰{'─' * inner_w}╯"))

        return lines


# Animation demo - proves overlays can handle real-time game-like updates
class AnimationDemoComponent(BaseOverlay):
    def __init__(self, tui, theme, done) -> None:
        super().__init__(theme)
        self._tui = tui
        self._done = done
        self._frame = 0
        self._interval: Interval | None = None
        self._fps = 0
        self._last_fps_update = time.monotonic()
        self._frames_since_last_fps = 0
        self._start_animation()

    def _start_animation(self) -> None:
        # Run at ~30 FPS
        async def on_tick() -> None:
            self._frame += 1
            self._frames_since_last_fps += 1

            # Update FPS counter every second
            now = time.monotonic()
            if now - self._last_fps_update >= 1.0:
                self._fps = self._frames_since_last_fps
                self._frames_since_last_fps = 0
                self._last_fps_update = now

            self._tui.request_render()

        self._interval = Interval(1000 / 30, on_tick)

    async def handle_input(self, data: str) -> None:
        if matches_key(data, "escape") or matches_key(data, "ctrl+c"):
            self.dispose()
            self._done(None)

    def render(self, width: int) -> list[str]:
        th = self._theme
        inner_w = max(1, width - 2)
        pad_line = lambda s: truncate_to_width(s, inner_w, "...", True)
        border = lambda c: th.fg("border", c)

        lines: list[str] = []
        lines.append(border(f"╭{'─' * inner_w}╮"))
        lines.append(border("│") + pad_line(th.fg("accent", " Animation Demo (~30 FPS)")) + border("│"))
        lines.append(border("│") + pad_line("") + border("│"))
        lines.append(border("│") + pad_line(f" Frame: {th.fg('accent', str(self._frame))}") + border("│"))
        lines.append(border("│") + pad_line(f" FPS: {th.fg('success', str(self._fps))}") + border("│"))
        lines.append(border("│") + pad_line("") + border("│"))

        # Animated content - bouncing bar
        bar_width = max(12, inner_w - 4)  # Ensure enough space for bar
        pos = max(0, int((math.sin(self._frame / 10) + 1) * (bar_width - 10) / 2))
        bar = " " * pos + th.fg("accent", "██████████") + " " * max(0, bar_width - 10 - pos)
        lines.append(border("│") + pad_line(f" {bar}") + border("│"))

        # Spinning character
        spin = "◐◓◑◒"[self._frame % 4]
        lines.append(border("│") + pad_line(f" Spinner: {th.fg('warning', spin)}") + border("│"))

        # Color cycling (colorsys is pi's hand-rolled hslToRgb)
        hue = (self._frame * 3) % 360
        r, g, b = (round(c * 255) for c in colorsys.hls_to_rgb(hue / 360, 0.5, 0.8))
        color_block = f"\x1b[48;2;{r};{g};{b}m{'  ' * 10}\x1b[0m"
        lines.append(border("│") + pad_line(f" Color: {color_block}") + border("│"))

        lines.append(border("│") + pad_line("") + border("│"))
        lines.append(border("│") + pad_line(th.fg("dim", " This proves overlays can handle")) + border("│"))
        lines.append(border("│") + pad_line(th.fg("dim", " real-time game-like rendering.")) + border("│"))
        lines.append(border("│") + pad_line("") + border("│"))
        lines.append(border("│") + pad_line(th.fg("dim", " Press Esc to close")) + border("│"))
        lines.append(border(f"╰{'─' * inner_w}╯"))

        return lines

    def dispose(self) -> None:
        if self._interval is not None:
            self._interval.cancel()
            self._interval = None


# Toggle demo - demonstrates OverlayHandle.set_hidden() via the onHandle callback
class ToggleDemoComponent(BaseOverlay):
    def __init__(self, tui, theme, toggle_state: dict, done) -> None:
        super().__init__(theme)
        self._tui = tui
        self._toggle_state = toggle_state
        self._done = done
        self._toggle_count = 0
        self._is_toggling = False

    async def handle_input(self, data: str) -> None:
        if matches_key(data, "escape") or matches_key(data, "ctrl+c"):
            self._done(None)
        elif matches_key(data, "t") and self._toggle_state["handle"] is not None and not self._is_toggling:
            # Demonstrate toggle by hiding for 1 second then showing again
            # (In real usage, a global keybinding would control visibility)
            self._is_toggling = True
            self._toggle_count += 1
            self._toggle_state["handle"].set_hidden(True)

            # Auto-restore after 1 second to demonstrate the API
            async def restore() -> None:
                handle = self._toggle_state["handle"]
                if handle is not None:
                    handle.set_hidden(False)
                    self._is_toggling = False
                    self._tui.request_render()

            Timeout(1000, restore)

    def render(self, width: int) -> list[str]:
        th = self._theme
        return self.box(
            [
                "",
                th.fg("accent", " Toggle Demo"),
                "",
                " This overlay demonstrates the",
                " onHandle callback API.",
                "",
                f" Toggle count: {th.fg('accent', str(self._toggle_count))}",
                "",
                th.fg("dim", " Press 't' to hide for 1 second"),
                th.fg("dim", " (demonstrates set_hidden API)"),
                "",
                th.fg("dim", " In real usage, a global keybinding"),
                th.fg("dim", " would toggle visibility externally."),
                "",
                th.fg("dim", " Press Esc to close"),
                "",
            ],
            width,
            "Toggle Demo",
        )


# === Non-capturing passive overlay demo ===


class PassiveDemoController(BaseOverlay):
    def __init__(self, tui, theme, done) -> None:
        super().__init__(theme)
        self.focused = False
        self._tui = tui
        self._done = done
        self._typed = ""
        self._input_count = 0
        self._last_input_debug = ""
        self._timer_component = TimerPanel(theme)
        self._timer_handle = tui.show_overlay(
            self._timer_component,
            {"nonCapturing": True, "anchor": "top-right", "width": 22, "margin": {"top": 1, "right": 2}},
        )

        async def on_tick() -> None:
            self._timer_component.tick()
            self._tui.request_render()

        self._interval: Interval | None = Interval(1000, on_tick)

    async def handle_input(self, data: str) -> None:
        self._input_count += 1
        self._last_input_debug = f"len={len(data)} c0={ord(data[0])}"
        if matches_key(data, "escape") or matches_key(data, "ctrl+c"):
            self._cleanup()
            self._done(None)
        elif matches_key(data, "backspace"):
            self._typed = self._typed[:-1]
        elif len(data) == 1 and ord(data) >= 32:
            self._typed += data

    def render(self, width: int) -> list[str]:
        th = self._theme
        display = self._typed if self._typed else th.fg("dim", "(type here)")
        return self.box(
            [
                "",
                f" {th.fg('dim', f'focused={self.focused} inputs={self._input_count}')}",
                f" {th.fg('dim', f'last: {self._last_input_debug or "none"}')}",
                "",
                f" > {display}",
                "",
                th.fg("dim", " Type to prove input goes here."),
                th.fg("dim", " Press Esc to close both."),
                "",
            ],
            width,
            "Non-Capturing Demo",
        )

    def _cleanup(self) -> None:
        if self._interval is not None:
            self._interval.cancel()
            self._interval = None
        if self._timer_handle is not None:
            self._timer_handle.hide()
            self._timer_handle = None

    def dispose(self) -> None:
        self._cleanup()


class TimerPanel(BaseOverlay):
    def __init__(self, theme) -> None:
        super().__init__(theme)
        self._seconds = 0

    def tick(self) -> None:
        self._seconds += 1

    def render(self, width: int) -> list[str]:
        th = self._theme
        mins, secs = divmod(self._seconds, 60)
        return self.box(
            [f" {th.fg('accent', f'{mins:02d}:{secs:02d}')}", th.fg("dim", " nonCapturing: True")],
            width,
            "Timer",
        )


# === Focus cycling demo ===

FOCUS_PANEL_CONFIGS = [
    {"label": "Alpha", "color": "error", "options": {"row": 2, "col": 4, "width": 34}},
    {"label": "Beta", "color": "success", "options": {"row": 5, "col": 28, "width": 34}},
    {"label": "Gamma", "color": "accent", "options": {"row": 8, "col": 52, "width": 34}},
]


class FocusDemoController(BaseOverlay):
    def __init__(self, tui, theme, done) -> None:
        super().__init__(theme)
        self._tui = tui
        self._done = done
        self._entries: list[dict] = []
        self._closed = False

        for config in FOCUS_PANEL_CONFIGS:
            panel = FocusPanel(theme=theme, config=config, controller=self)
            handle = tui.show_overlay(panel, {"nonCapturing": True, **config["options"]})
            self._entries.append({"panel": panel, "handle": handle})

        self._focus_first_open_panel()

    def focus_next(self, current, direction: int = 1) -> None:
        open_entries = self._open_entries()
        current_position = next((i for i, e in enumerate(open_entries) if e["panel"] is current), -1)
        if current_position == -1:
            raise RuntimeError(f"Panel {current.label} is not open")
        next_position = (current_position + direction) % len(open_entries)
        self._focus_entry_at(open_entries, next_position)

    def dismiss(self, panel) -> None:
        open_entries = self._open_entries()
        current_position = next((i for i, e in enumerate(open_entries) if e["panel"] is panel), -1)
        if current_position == -1:
            return
        entry = open_entries[current_position]
        remaining_entries = [e for e in open_entries if e["panel"] is not panel]

        entry["panel"].closed = True
        entry["handle"].hide()
        if not remaining_entries:
            self.close()
            return

        self._focus_entry_at(remaining_entries, current_position % len(remaining_entries))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._hide_panels()
        self._done(None)

    async def handle_input(self, data: str) -> None:
        if matches_key(data, "escape") or matches_key(data, "ctrl+c"):
            self.close()
        elif matches_key(data, "tab"):
            self._focus_first_open_panel()

    def render(self, width: int) -> list[str]:
        th = self._theme
        focused = next((e["panel"].label for e in self._entries if e["handle"].is_focused()), "Controller")
        return self.box(
            [
                "",
                f" Current focus: {th.fg('accent', focused)}",
                "",
                " Three overlapping panels above are",
                f" {th.fg('accent', 'nonCapturing')} overlays controlled with",
                " raw OverlayHandle.focus()/hide().",
                "",
                " Type in the focused panel's input.",
                " Focused panel renders on top.",
                "",
                th.fg("dim", " Tab/Shift+Tab = cycle panels"),
                th.fg("dim", " Esc/Ctrl+D = dismiss panel"),
                th.fg("dim", " Ctrl+C = close all"),
                "",
            ],
            width,
            "Focus + Input Demo",
        )

    def dispose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._hide_panels()

    def _focus_first_open_panel(self) -> None:
        open_entries = self._open_entries()
        if open_entries:
            open_entries[0]["handle"].focus()
            self._tui.request_render()

    def _focus_entry_at(self, entries: list[dict], index: int) -> None:
        entries[index]["handle"].focus()
        self._tui.request_render()

    def _hide_panels(self) -> None:
        for entry in self._entries:
            if not entry["panel"].closed:
                entry["panel"].closed = True
                entry["handle"].hide()
        self._entries = []

    def _open_entries(self) -> list[dict]:
        return [entry for entry in self._entries if not entry["panel"].closed]


class FocusPanel(BaseOverlay):
    def __init__(self, *, theme, config: dict, controller) -> None:
        super().__init__(theme)
        self.focused = False
        self.closed = False
        self.label = config["label"]
        self._color = config["color"]
        self._controller = controller
        self._input = Input()
        self._inputs: list[str] = []

    async def handle_input(self, data: str) -> None:
        if matches_key(data, "tab"):
            self._controller.focus_next(self)
        elif matches_key(data, "shift+tab"):
            self._controller.focus_next(self, -1)
        elif matches_key(data, "escape") or matches_key(data, "ctrl+d"):
            self._controller.dismiss(self)
        elif matches_key(data, "ctrl+c"):
            self._controller.close()
        elif matches_key(data, "return"):
            self._inputs.append("Enter")
        elif matches_key(data, "up"):
            self._inputs.append("↑")
        elif matches_key(data, "down"):
            self._inputs.append("↓")
        elif matches_key(data, "left"):
            await self._input.handle_input(data)
            self._inputs.append("←")
        elif matches_key(data, "right"):
            await self._input.handle_input(data)
            self._inputs.append("→")
        elif matches_key(data, "backspace"):
            await self._input.handle_input(data)
            self._inputs.append("Backspace")
        else:
            await self._input.handle_input(data)
            self._inputs.append(repr(data))

    def render(self, width: int) -> list[str]:
        th = self._theme
        inner_w = max(1, width - 2)
        border = lambda c: th.fg(self._color if self.focused else "dim", c)
        pad_line = lambda s: truncate_to_width(s, inner_w, "...", True)
        recent = " ".join(self._inputs[-6:]) if self._inputs else "(none)"
        lines: list[str] = []

        self._input.focused = self.focused
        input_lines = self._input.render(max(1, inner_w - 8))
        input_line = input_lines[0] if input_lines else ""
        status = th.fg("success", "FOCUSED") if self.focused else th.fg("dim", "visible")
        lines.append(border(f"╭{'─' * inner_w}╮"))
        lines.append(border("│") + pad_line(f" {th.fg(self._color, self.label)} {status}") + border("│"))
        lines.append(border("│") + pad_line("") + border("│"))
        lines.append(border("│") + pad_line(f" Input: {input_line}") + border("│"))
        lines.append(border("│") + pad_line(f" Keys: {recent}") + border("│"))
        lines.append(border("│") + pad_line(th.fg("dim", " Tab/Shift+Tab focus")) + border("│"))
        lines.append(border("│") + pad_line(th.fg("dim", " Esc/Ctrl+D dismiss")) + border("│"))
        lines.append(border(f"╰{'─' * inner_w}╯"))

        return lines


# === Streaming input panel test (/overlay-streaming) ===


class StreamingInputController(BaseOverlay):
    def __init__(self, tui, theme, done) -> None:
        super().__init__(theme)
        self._tui = tui
        self._done = done
        self._panels: list = []
        self._handles: list = []
        self._focus_index = -1  # -1 = controller focused, 0-2 = panel focused
        self._stream_lines: list[str] = []
        self._line_count = 0

        # Create 3 input panels as non-capturing overlays
        colors = ["error", "success", "accent"]
        labels = ["Panel A", "Panel B", "Panel C"]

        for i in range(3):
            panel = StreamingInputPanel(theme, labels[i], colors[i], self._cycle_focus, self._close)
            handle = tui.show_overlay(panel, {"nonCapturing": True, "row": 1 + i * 9, "col": 2, "width": 35})
            panel.handle = handle
            self._panels.append(panel)
            self._handles.append(handle)

        # Start with controller focused (focus_index = -1)

        # Start simulated streaming
        async def on_tick() -> None:
            self._line_count += 1
            timestamp = time.strftime("%H:%M:%S")
            self._stream_lines.append(f"[{timestamp}] Streaming line {self._line_count}...")
            if len(self._stream_lines) > 8:
                self._stream_lines.pop(0)
            self._tui.request_render()

        self._stream_interval: Interval | None = Interval(500, on_tick)

    def _cycle_focus(self) -> None:
        # Unfocus current panel if any
        if 0 <= self._focus_index < len(self._handles):
            self._handles[self._focus_index].unfocus()

        # Cycle: -1 (controller) → 0 → 1 → 2 → -1 ...
        self._focus_index += 1
        if self._focus_index >= len(self._handles):
            self._focus_index = -1  # Back to controller

        # Focus new panel if any
        if self._focus_index >= 0:
            self._handles[self._focus_index].focus()

        self._tui.request_render()

    def _close(self) -> None:
        if self._stream_interval is not None:
            self._stream_interval.cancel()
            self._stream_interval = None
        for handle in self._handles:
            handle.hide()
        self._handles = []
        self._panels = []
        self._done(None)

    async def handle_input(self, data: str) -> None:
        if matches_key(data, "escape") or matches_key(data, "ctrl+c"):
            self._close()
        elif matches_key(data, "tab"):
            self._cycle_focus()

    def render(self, width: int) -> list[str]:
        th = self._theme
        if self._focus_index == -1:
            focused_label = th.fg("success", "Controller (this panel)")
        else:
            focused_label = self._panels[self._focus_index].label if self._panels else "?"

        lines = [
            "",
            f" Current focus: {th.fg('accent', focused_label)}",
            "",
            " Simulated streaming output:",
            th.fg("dim", " ─" * ((width - 2) // 2)),
        ]

        for line in self._stream_lines:
            lines.append(f" {th.fg('dim', line)}")

        while len(lines) < 12:
            lines.append("")

        lines.append(th.fg("dim", " ─" * ((width - 2) // 2)))
        lines.append("")
        lines.append(f" Three {th.fg('accent', 'nonCapturing')} input panels on the left.")
        lines.append(" Tab cycles: Controller → Panel A → B → C → Controller")
        lines.append(" Type in each panel to test input routing.")
        lines.append("")
        lines.append(th.fg("dim", " Tab = cycle focus | Esc = close all"))
        lines.append("")

        return self.box(lines, width, "Streaming + Input Test")

    def dispose(self) -> None:
        self._close()


class StreamingInputPanel:
    def __init__(self, theme, label: str, color: str, on_tab, on_close) -> None:
        self._theme = theme
        self.label = label
        self.handle = None
        self._color = color
        self._on_tab = on_tab
        self._on_close = on_close
        self._typed = ""

    async def handle_input(self, data: str) -> None:
        if matches_key(data, "tab"):
            self._on_tab()
        elif matches_key(data, "escape") or matches_key(data, "ctrl+c"):
            self._on_close()
        elif matches_key(data, "backspace"):
            self._typed = self._typed[:-1]
        elif len(data) == 1 and ord(data) >= 32:
            self._typed += data

    def render(self, width: int) -> list[str]:
        th = self._theme
        focused = self.handle.is_focused() if self.handle is not None else False
        inner_w = max(1, width - 2)
        border = lambda c: th.fg(self._color, c)
        pad_line = lambda s: s + " " * max(0, inner_w - visible_width(s))

        input_display = self._typed if self._typed else th.fg("dim", "(type here)")
        truncated_input = truncate_to_width(f" > {input_display}", inner_w, "...", True)

        lines: list[str] = []
        lines.append(border(f"╭{'─' * inner_w}╮"))
        lines.append(border("│") + pad_line(f" {th.fg('accent', self.label)}") + border("│"))
        lines.append(border("│") + pad_line("") + border("│"))
        if focused:
            lines.append(border("│") + pad_line(th.fg("success", " ● FOCUSED")) + border("│"))
            lines.append(border("│") + pad_line(th.fg("dim", " (receiving input)")) + border("│"))
        else:
            lines.append(border("│") + pad_line(th.fg("dim", " ○ unfocused")) + border("│"))
            lines.append(border("│") + pad_line("") + border("│"))
        lines.append(border("│") + pad_line(truncated_input) + border("│"))
        lines.append(border("│") + pad_line("") + border("│"))
        lines.append(border("│") + pad_line(th.fg("dim", " Tab | Esc")) + border("│"))
        lines.append(border(f"╰{'─' * inner_w}╯"))

        return lines

    def invalidate(self) -> None:
        pass


def extension(pi):
    # Handle for the toggle demo, shared between the command's onHandle
    # callback and the component (kept in the factory closure, not a module
    # global, so it survives /reload semantics cleanly)
    toggle_state: dict = {"handle": None}

    # Animation demo - proves overlays can handle real-time updates
    async def overlay_animation(_args: str, ctx) -> None:
        await ctx.ui.custom(
            lambda tui, theme, _kb, done: AnimationDemoComponent(tui, theme, done),
            {"overlay": True, "overlayOptions": {"anchor": "center", "width": 50, "maxHeight": 20}},
        )

    # Test all 9 anchor positions
    async def overlay_anchors(_args: str, ctx) -> None:
        index = 0
        while True:
            anchor = ANCHORS[index]
            result = await ctx.ui.custom(
                lambda _tui, theme, _kb, done, anchor=anchor: AnchorTestComponent(theme, anchor, done),
                {"overlay": True, "overlayOptions": {"anchor": anchor, "width": 40}},
            )

            if result == "next":
                index = (index + 1) % len(ANCHORS)
                continue
            if result == "confirm":
                ctx.ui.notify(f"Selected: {ANCHORS[index]}", "info")
            break

    # Test margins and offsets
    async def overlay_margins(_args: str, ctx) -> None:
        configs = [
            {"name": "No margin (top-left)", "options": {"anchor": "top-left", "width": 35}},
            {"name": "Margin: 3 all sides", "options": {"anchor": "top-left", "width": 35, "margin": 3}},
            {
                "name": "Margin: top=5, left=10",
                "options": {"anchor": "top-left", "width": 35, "margin": {"top": 5, "left": 10}},
            },
            {
                "name": "Center + offset (10, -3)",
                "options": {"anchor": "center", "width": 35, "offsetX": 10, "offsetY": -3},
            },
            {"name": "Bottom-right, margin: 2", "options": {"anchor": "bottom-right", "width": 35, "margin": 2}},
        ]

        index = 0
        while True:
            config = configs[index]
            result = await ctx.ui.custom(
                lambda _tui, theme, _kb, done, config=config: MarginTestComponent(theme, config, done),
                {"overlay": True, "overlayOptions": config["options"]},
            )

            if result == "next":
                index = (index + 1) % len(configs)
                continue
            break

    # Test stacked overlays
    async def overlay_stack(_args: str, ctx) -> None:
        # Three large overlays that overlap in the center area
        # Each offset slightly so you can see the stacking

        def show(num: int, position: str, offset_x: int, offset_y: int):
            # tonio.spawn starts the coroutine eagerly, so the overlay shows
            # now and the join handle is awaited later (JS Promise.all shape)
            return tonio.spawn(
                ctx.ui.custom(
                    lambda _tui, theme, _kb, done: StackOverlayComponent(theme, num, position, done),
                    {
                        "overlay": True,
                        "overlayOptions": {
                            "anchor": "center",
                            "width": 50,
                            "offsetX": offset_x,
                            "offsetY": offset_y,
                            "maxHeight": 15,
                        },
                    },
                )
            )

        ctx.ui.notify("Showing overlay 1 (back)...", "info")
        t1 = show(1, "back (red border)", -8, -4)
        await tonio.sleep(0.4)

        ctx.ui.notify("Showing overlay 2 (middle)...", "info")
        t2 = show(2, "middle (green border)", 0, 0)
        await tonio.sleep(0.4)

        ctx.ui.notify("Showing overlay 3 (front)...", "info")
        t3 = show(3, "front (blue border)", 8, 4)

        # Wait for all to close
        results = [await t1, await t2, await t3]
        ctx.ui.notify(f"Closed in order: {', '.join(results)}", "info")

    # Test width overflow scenarios (original crash case) - streams real process output
    async def overlay_overflow(_args: str, ctx) -> None:
        await ctx.ui.custom(
            lambda tui, theme, _kb, done: StreamingOverflowComponent(tui, theme, done),
            {"overlay": True, "overlayOptions": {"anchor": "center", "width": 90, "maxHeight": 20}},
        )

    # Test overlay at terminal edge
    async def overlay_edge(_args: str, ctx) -> None:
        await ctx.ui.custom(
            lambda _tui, theme, _kb, done: EdgeTestComponent(theme, done),
            {"overlay": True, "overlayOptions": {"anchor": "right-center", "width": 40, "margin": {"right": 0}}},
        )

    # Test percentage-based positioning
    async def overlay_percent(_args: str, ctx) -> None:
        configs = [
            {"name": "row: 0% (top)", "row": 0, "col": 50},
            {"name": "row: 50% (middle)", "row": 50, "col": 50},
            {"name": "row: 100% (bottom)", "row": 100, "col": 50},
            {"name": "col: 0% (left)", "row": 50, "col": 0},
            {"name": "col: 100% (right)", "row": 50, "col": 100},
        ]

        index = 0
        while True:
            config = configs[index]
            result = await ctx.ui.custom(
                lambda _tui, theme, _kb, done, config=config: PercentTestComponent(theme, config, done),
                {
                    "overlay": True,
                    "overlayOptions": {"width": 30, "row": f"{config['row']}%", "col": f"{config['col']}%"},
                },
            )

            if result == "next":
                index = (index + 1) % len(configs)
                continue
            break

    # Test maxHeight
    async def overlay_maxheight(_args: str, ctx) -> None:
        await ctx.ui.custom(
            lambda _tui, theme, _kb, done: MaxHeightTestComponent(theme, done),
            {"overlay": True, "overlayOptions": {"anchor": "center", "width": 50, "maxHeight": 10}},
        )

    # Test responsive sidepanel - only shows when terminal is wide enough
    async def overlay_sidepanel(_args: str, ctx) -> None:
        await ctx.ui.custom(
            lambda tui, theme, _kb, done: SidepanelComponent(tui, theme, done),
            {
                "overlay": True,
                "overlayOptions": {
                    "anchor": "right-center",
                    "width": "25%",
                    "minWidth": 30,
                    "margin": {"right": 1},
                    # Only show when terminal is wide enough (pidrei passes
                    # columns and rows to the visibility callback)
                    "visible": lambda cols, _rows: cols >= 100,
                },
            },
        )

    # Test toggle overlay - demonstrates OverlayHandle.set_hidden() via onHandle
    async def overlay_toggle(_args: str, ctx) -> None:
        def on_handle(handle) -> None:
            # The onHandle callback provides access to the OverlayHandle for
            # visibility control
            toggle_state["handle"] = handle

        await ctx.ui.custom(
            lambda tui, theme, _kb, done: ToggleDemoComponent(tui, theme, toggle_state, done),
            {
                "overlay": True,
                "overlayOptions": {"anchor": "center", "width": 50},
                "onHandle": on_handle,
            },
        )
        toggle_state["handle"] = None

    # Non-capturing overlay demo - passive info panel that doesn't steal focus
    async def overlay_passive(_args: str, ctx) -> None:
        ctx.ui.set_editor_text("")
        await ctx.ui.custom(
            lambda tui, theme, _kb, done: PassiveDemoController(tui, theme, done),
            {"overlay": True, "overlayOptions": {"anchor": "center", "width": 48}},
        )

    # Focus cycling demo - demonstrates focus(), input routing, per-panel
    # dismissal, and rendering order
    async def overlay_focus(_args: str, ctx) -> None:
        ctx.ui.set_editor_text("")
        await ctx.ui.custom(
            lambda tui, theme, _kb, done: FocusDemoController(tui, theme, done),
            {"overlay": True, "overlayOptions": {"anchor": "bottom-center", "width": 55, "margin": {"bottom": 1}}},
        )

    # Test multiple input panels with simulated streaming
    async def overlay_streaming(_args: str, ctx) -> None:
        ctx.ui.set_editor_text("")
        await ctx.ui.custom(
            lambda tui, theme, _kb, done: StreamingInputController(tui, theme, done),
            {"overlay": True, "overlayOptions": {"anchor": "bottom-center", "width": 60, "margin": {"bottom": 1}}},
        )

    pi.register_command(
        "overlay-animation", handler=overlay_animation, description="Test real-time animation in overlay (~30 FPS)"
    )
    pi.register_command("overlay-anchors", handler=overlay_anchors, description="Cycle through all anchor positions")
    pi.register_command("overlay-margins", handler=overlay_margins, description="Test margin and offset options")
    pi.register_command("overlay-stack", handler=overlay_stack, description="Test stacked overlays")
    pi.register_command(
        "overlay-overflow", handler=overlay_overflow, description="Test width overflow with streaming process output"
    )
    pi.register_command("overlay-edge", handler=overlay_edge, description="Test overlay positioned at terminal edge")
    pi.register_command("overlay-percent", handler=overlay_percent, description="Test percentage-based positioning")
    pi.register_command("overlay-maxheight", handler=overlay_maxheight, description="Test maxHeight truncation")
    pi.register_command(
        "overlay-sidepanel",
        handler=overlay_sidepanel,
        description="Test responsive sidepanel (hides when terminal < 100 cols)",
    )
    pi.register_command(
        "overlay-toggle", handler=overlay_toggle, description="Test overlay toggle (press 't' to toggle visibility)"
    )
    pi.register_command(
        "overlay-passive",
        handler=overlay_passive,
        description="Test non-capturing overlay (passive info panel alongside active overlay)",
    )
    pi.register_command(
        "overlay-focus",
        handler=overlay_focus,
        description="Test focus cycling, input routing, dismissal, and rendering order with overlays",
    )
    pi.register_command(
        "overlay-streaming",
        handler=overlay_streaming,
        description="Multiple input panels with simulated streaming (Tab to cycle focus)",
    )
