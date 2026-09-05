"""Mirror of pi tui test/tui-alt-screen.test.ts.

Viewport expectations are right-stripped (see virtual_terminal.py). pi's
`waitForRender()` awaits the next frame; here the frame counter is sampled
before each trigger and passed to `wait_for_render` so the assertions never
read a stale viewport.
"""

import base64
import os
import re
import time
from contextlib import contextmanager

import pytest
import tonio.colored as tonio

from pidrei_tui.components.h_stack import HStack
from pidrei_tui.components.image import Image
from pidrei_tui.components.mouse_region import MouseRegion
from pidrei_tui.components.scroll_view import ScrollView
from pidrei_tui.components.select_list import SelectList
from pidrei_tui.components.text import Text
from pidrei_tui.components.v_stack import VStack
from pidrei_tui.keybindings import (
    TUI_KEYBINDINGS,
    KeybindingsManager,
    get_keybindings,
    set_keybindings,
)
from pidrei_tui.terminal_image import (
    encode_kitty,
    hyperlink,
    register_kitty_image_metadata,
    reset_capabilities_cache,
    set_capabilities,
)
from pidrei_tui.tui import TuiMouseEventResult
from pidrei_tui.tui_alt_screen import TuiAltScreen

from .virtual_terminal import VirtualTerminal, poll_until


OSC133_ZONE_START = "\x1b]133;A\x07"


class RecordingTerminal(VirtualTerminal):
    """VirtualTerminal that records start/write/stop in order."""

    def __init__(self, columns: int = 80, rows: int = 24) -> None:
        super().__init__(columns, rows)
        self.events: list[dict] = []

    async def start(self, on_input, on_resize) -> None:
        self.events.append({"type": "start"})
        await super().start(on_input, on_resize)

    async def write(self, data: str) -> None:
        self.events.append({"type": "write", "data": data})
        await super().write(data)

    async def stop(self) -> None:
        self.events.append({"type": "stop"})
        await super().stop()

    def index_of_write(self, needle: str) -> int:
        for index, event in enumerate(self.events):
            if event["type"] == "write" and needle in event["data"]:
                return index
        return -1

    def index_of_type(self, kind: str) -> int:
        for index, event in enumerate(self.events):
            if event["type"] == kind:
                return index
        return -1

    def writes_containing(self, needle: str) -> list[str]:
        return [event["data"] for event in self.events if event["type"] == "write" and needle in event["data"]]

    async def wait_for_write(self, needle: str, count: int = 1, timeout: float = 2.0) -> list[str]:
        """Poll until `count` writes contain `needle`, bounded so a miss fails.

        `wait_for_render(since)` waits for *a* frame after `since`, which is not
        the same as the frame that reflects the input just sent: the render loop
        is a separate throttled task, so a frame already computed before the
        input can be the one that satisfies the wait. With several inputs per
        wait that is a real window, and it is how the mouse-selection tests here
        failed on the macOS CI runners while passing everywhere else. Use this
        whenever the assertion is that a write *did* happen; `wait_for_render`
        still fits assertions that nothing further happens.
        """
        deadline = time.monotonic() + timeout
        while True:
            found = self.writes_containing(needle)
            if len(found) >= count or time.monotonic() >= deadline:
                return found
            await tonio.sleep(0.005)


class RenderComponent:
    """Component that renders a fixed list of lines."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def render(self, width: int) -> list[str]:
        return self.lines

    def invalidate(self) -> None:
        pass


@contextmanager
def capabilities(caps: dict):
    set_capabilities(caps)
    try:
        yield
    finally:
        reset_capabilities_cache()


def _osc52(text: str) -> str:
    return f"\x1b]52;c;{base64.b64encode(text.encode('utf-8')).decode('ascii')}\x07"


def _lines(count: int) -> str:
    return "\n".join(f"line {index + 1}" for index in range(count))


def _viewport(terminal: VirtualTerminal) -> list[str]:
    return [line.rstrip() for line in terminal.get_viewport()]


async def _wait_until(predicate, timeout: float = 2.0) -> bool:
    """Poll `predicate` until true, bounded so a miss fails.

    `wait_for_render(since)` returns on *a* frame, which is not necessarily the
    frame reflecting the input just sent (the render loop is a separate
    throttled task) — assertions on state or viewport content driven by an
    input must poll the condition itself. Same rationale as
    `RecordingTerminal.wait_for_write`.
    """
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        await tonio.sleep(0.005)


async def _wait_for_viewport_text(terminal: VirtualTerminal, needle: str, timeout: float = 2.0) -> bool:
    """Poll until `needle` shows in the viewport, bounded so a miss fails.

    Selection copy runs as a detached task (pi: `void copySelectionToClipboard()`),
    so its "Copied!"/"Copy failed" flash can land after the frame that
    `wait_for_render` observed.
    """
    return await _wait_until(
        lambda: any(needle in line for line in terminal.get_viewport()),
        timeout,
    )


@pytest.mark.tonio
async def test_renders_a_terminal_height_viewport_and_preserves_manual_scroll_position():
    terminal = VirtualTerminal(20, 4)
    tui = TuiAltScreen(terminal)
    text = Text(_lines(10), 0, 0)
    tui.add_child(text)
    await tui.start()
    await terminal.wait_for_render()

    assert _viewport(terminal) == ["line 7", "line 8", "line 9", "line 10"]
    assert tui.is_following_output is True

    since = terminal.frames
    await terminal.send_input("\x1b[<64;1;1M")
    await terminal.wait_for_render(since)
    assert _viewport(terminal) == ["line 6", "line 7", "line 8", "line 9"]
    assert tui.viewport_top == 5
    assert tui.is_following_output is False

    text.set_text(_lines(12))
    since = terminal.frames
    tui.request_render()
    await terminal.wait_for_render(since)
    assert _viewport(terminal) == ["line 6", "line 7", "line 8", "line 9"]

    await tui.stop()


@pytest.mark.tonio
async def test_shows_a_clickable_jump_to_end_indicator_on_the_transcripts_last_row_while_scrolled_up():
    terminal = VirtualTerminal(30, 6)
    tui = TuiAltScreen(terminal, None, None, scroll_to_end_indicator=lambda: "\x1b[7m ↓ Jump to end \x1b[27m")
    transcript = ScrollView(Text(_lines(8), 0, 0), {"follow": "end", "primary": True})
    tui.set_layout_root(
        VStack(
            [
                {"component": transcript, "basis": 0, "grow": 1, "minSize": 1},
                {"component": Text("editor\nfooter", 0, 0), "basis": "auto", "minSize": 1},
            ]
        )
    )
    await tui.start()
    await terminal.wait_for_render()
    assert not any("Jump to end" in line for line in terminal.get_viewport())

    await terminal.send_input("\x1b[<64;1;1M")
    assert await _wait_until(lambda: transcript.is_following_end is False)
    # The virtual terminal trims trailing blanks; pi compares the padded row.
    assert await _wait_until(lambda: terminal.get_viewport()[3].rstrip() == "line 7  ↓ Jump to end")
    assert terminal.get_viewport()[4].rstrip() == "editor"

    # Pressing next to the label starts a selection instead of jumping.
    await terminal.send_input("\x1b[<0;2;4M")
    await terminal.send_input("\x1b[<0;2;4m")
    await terminal.wait_for_render()
    assert transcript.is_following_end is False

    await terminal.send_input("\x1b[<0;15;4M")
    await terminal.send_input("\x1b[<0;15;4m")
    assert await _wait_until(lambda: transcript.is_following_end is True)
    assert await _wait_until(
        lambda: _viewport(terminal) == ["line 5", "line 6", "line 7", "line 8", "editor", "footer"]
    )
    await tui.stop()


@pytest.mark.tonio
async def test_leaves_the_scrollbar_clickable_when_the_jump_to_end_indicator_spans_the_transcript():
    terminal = VirtualTerminal(30, 6)
    tui = TuiAltScreen(terminal, None, None, scroll_to_end_indicator=lambda: "↓" * 30)
    transcript = ScrollView(Text(_lines(12), 0, 0), {"follow": "end", "primary": True, "scrollbar": "always"})
    tui.set_layout_root(
        VStack(
            [
                {"component": transcript, "basis": 0, "grow": 1, "minSize": 1},
                {"component": Text("editor\nfooter", 0, 0), "basis": "auto", "minSize": 1},
            ]
        )
    )
    await tui.start()
    await terminal.wait_for_render()

    await terminal.send_input("\x1b[<64;1;1M")
    assert await _wait_until(lambda: transcript.is_following_end is False)

    # The indicator must not intercept a press on the scrollbar's last column.
    await terminal.send_input("\x1b[<0;30;4M")
    await terminal.send_input("\x1b[<0;30;4m")
    await terminal.wait_for_render()
    assert transcript.is_following_end is False
    await tui.stop()


@pytest.mark.tonio
async def test_never_shows_the_jump_to_end_indicator_for_a_primary_scroll_view_without_follow_end():
    terminal = VirtualTerminal(30, 3)
    tui = TuiAltScreen(terminal, None, None, scroll_to_end_indicator=lambda: " ↓ Jump to end ")
    transcript = ScrollView(Text("one\ntwo\nthree\nfour\nfive", 0, 0), {"primary": True})
    tui.set_layout_root(transcript)
    await tui.start()
    await terminal.wait_for_render()

    assert transcript.is_following_end is False
    assert not any("Jump to end" in line for line in terminal.get_viewport())
    await tui.stop()


@pytest.mark.tonio
async def test_keeps_an_explicit_dock_fixed_while_the_transcript_scrolls():
    terminal = VirtualTerminal(20, 6)
    tui = TuiAltScreen(terminal)
    transcript_text = Text(_lines(8), 0, 0)
    transcript = ScrollView(transcript_text, {"follow": "end", "primary": True})
    dock = VStack([Text("editor", 0, 0), Text("footer", 0, 0)])
    tui.set_layout_root(
        VStack(
            [
                {"component": transcript, "basis": 0, "grow": 1, "minSize": 1},
                {"component": dock, "basis": "auto", "minSize": 1},
            ]
        )
    )
    await tui.start()
    await terminal.wait_for_render()

    assert _viewport(terminal) == ["line 5", "line 6", "line 7", "line 8", "editor", "footer"]

    # Wheel over the dock falls back to the primary transcript scroll view.
    since = terminal.frames
    await terminal.send_input("\x1b[<64;1;6M")
    await terminal.wait_for_render(since)
    assert _viewport(terminal) == ["line 4", "line 5", "line 6", "line 7", "editor", "footer"]
    assert transcript.is_following_end is False

    transcript_text.set_text(_lines(10))
    since = terminal.frames
    tui.request_render()
    await terminal.wait_for_render(since)
    assert _viewport(terminal) == ["line 4", "line 5", "line 6", "line 7", "editor", "footer"]

    since = terminal.frames
    tui.scroll_to_bottom()
    await terminal.wait_for_render(since)
    assert _viewport(terminal) == ["line 7", "line 8", "line 9", "line 10", "editor", "footer"]
    await tui.stop()


@pytest.mark.tonio
async def test_invalidates_overlays_with_an_explicit_layout_root():
    tui = TuiAltScreen(VirtualTerminal())
    overlay = Text("overlay", 0, 0)
    invalidated: list[bool] = []
    overlay.invalidate = lambda: invalidated.append(True)
    tui.set_layout_root(Text("root", 0, 0))
    tui.show_overlay(overlay)

    tui.invalidate()

    assert invalidated == [True]
    await tui.stop()


@pytest.mark.tonio
async def test_routes_wheel_input_to_the_scroll_view_under_the_pointer():
    terminal = VirtualTerminal(20, 4)
    tui = TuiAltScreen(terminal)
    left = ScrollView(Text("a1\na2\na3\na4\na5\na6\na7", 0, 0), {"follow": "end", "primary": True})
    right = ScrollView(Text("b1\nb2\nb3\nb4\nb5\nb6\nb7", 0, 0), {"follow": "end"})
    tui.set_layout_root(
        HStack([{"component": left, "basis": 10, "shrink": 0}, {"component": right, "basis": 10, "shrink": 0}])
    )
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    await terminal.send_input("\x1b[<64;15;1M")
    await terminal.wait_for_render(since)
    assert left.scroll_top == 3
    assert right.scroll_top == 2
    assert _viewport(terminal) == ["a4        b3", "a5        b4", "a6        b5", "a7        b6"]
    await tui.stop()


@pytest.mark.tonio
async def test_uses_button_motion_tracking_inside_terminal_multiplexers():
    environment_keys = ("TMUX", "ZELLIJ", "STY", "TERM")
    previous_environment = {key: os.environ.get(key) for key in environment_keys}
    try:
        for key in environment_keys:
            os.environ.pop(key, None)
        os.environ["TERM"] = "xterm-256color"
        direct_terminal = RecordingTerminal()
        direct_tui = TuiAltScreen(direct_terminal)
        await direct_tui.start()
        assert direct_terminal.writes_containing("\x1b[?1003h")
        await direct_tui.stop()

        multiplexers = [
            ("tmux environment", {"TMUX": "/tmp/tmux/default,1,0"}),
            ("tmux TERM", {"TERM": "tmux-256color"}),
            ("Zellij environment", {"ZELLIJ": "0"}),
            ("Screen environment", {"STY": "123.session"}),
            ("Screen TERM", {"TERM": "screen-256color"}),
        ]
        for name, environment in multiplexers:
            for key in environment_keys:
                os.environ.pop(key, None)
            os.environ.update(environment)
            terminal = RecordingTerminal()
            tui = TuiAltScreen(terminal)
            await tui.start()
            assert terminal.writes_containing("\x1b[?1002h"), f"{name} should enable button-motion tracking"
            assert not terminal.writes_containing("\x1b[?1003h"), f"{name} should not enable all-motion tracking"
            assert terminal.writes_containing("\x1b[?1006h"), f"{name} should enable SGR mouse encoding"
            await tui.stop()
    finally:
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.mark.tonio
async def test_reveals_an_auto_scrollbar_when_the_pointer_enters_its_hidden_track():
    terminal = RecordingTerminal(10, 5)
    tui = TuiAltScreen(terminal)
    scroll_view = ScrollView(Text(_lines(20), 0, 0), {"primary": True, "scrollbar": "auto", "scrollbarHideDelayMs": 20})
    tui.set_layout_root(scroll_view)
    await tui.start()
    await terminal.wait_for_render()
    assert scroll_view.is_scrollbar_visible is False

    await terminal.send_input("\x1b[<35;10;3M")
    assert await _wait_until(lambda: scroll_view.is_scrollbar_visible is True)
    assert scroll_view.is_scrollbar_active is True
    assert await _wait_until(lambda: any(re.search(r"[│█]", line) for line in terminal.get_viewport()))

    await terminal.send_input("\x1b[<35;9;3M")
    assert await _wait_until(lambda: scroll_view.is_scrollbar_visible is False)
    await tui.stop()


@pytest.mark.tonio
async def test_jumps_to_a_scrollbar_track_position_and_continues_dragging_from_there():
    terminal = RecordingTerminal(10, 10)
    tui = TuiAltScreen(terminal)
    scroll_view = ScrollView(Text(_lines(50), 0, 0), {"primary": True, "scrollbar": "always"})
    tui.set_layout_root(scroll_view)
    await tui.start()
    await terminal.wait_for_render()
    assert scroll_view.scroll_top == 0

    await terminal.send_input("\x1b[<0;10;6M")
    assert await _wait_until(lambda: scroll_view.scroll_top == 20)

    await terminal.send_input("\x1b[<32;10;10M")
    assert await _wait_until(lambda: scroll_view.scroll_top == 40)

    await terminal.send_input("\x1b[<0;10;10m")
    await terminal.wait_for_render()
    assert not terminal.writes_containing("\x1b]52;c;")
    await tui.stop()


@pytest.mark.tonio
async def test_chains_unused_wheel_delta_to_an_outer_scroll_view():
    terminal = VirtualTerminal(20, 4)
    tui = TuiAltScreen(terminal, wheel_scroll_lines=3)
    inner = ScrollView(Text("i1\ni2\ni3\ni4\ni5\ni6", 0, 0))
    outer = ScrollView(
        VStack([{"component": inner, "basis": 2}, Text("tail1\ntail2\ntail3\ntail4\ntail5", 0, 0)]),
        {"primary": True},
    )
    tui.set_layout_root(outer)
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    await terminal.send_input("\x1b[<65;1;1M")
    await terminal.wait_for_render(since)
    assert inner.scroll_top == 3
    assert outer.scroll_top == 0

    since = terminal.frames
    await terminal.send_input("\x1b[<65;1;1M")
    await terminal.wait_for_render(since)
    assert inner.scroll_top == 4
    assert outer.scroll_top == 2
    await tui.stop()


@pytest.mark.tonio
async def test_supports_configurable_keyboard_viewport_navigation_with_four_rows_of_page_overlap():
    terminal = VirtualTerminal(20, 8)
    tui = TuiAltScreen(terminal)
    tui.add_child(Text(_lines(12), 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    await terminal.send_input("\x1b[57421u")  # pageUp
    await terminal.send_input("\x1b[57421;1:3u")  # pageUp release
    await terminal.wait_for_render(since)
    assert _viewport(terminal) == [f"line {index}" for index in range(1, 9)]

    since = terminal.frames
    await terminal.send_input("\x1b[57422u")  # pageDown
    await terminal.send_input("\x1b[57422;1:3u")  # pageDown release
    await terminal.wait_for_render(since)
    assert _viewport(terminal) == [f"line {index}" for index in range(5, 13)]

    since = terminal.frames
    await terminal.send_input("\x1bOH")  # home
    await terminal.wait_for_render(since)
    assert _viewport(terminal) == [f"line {index}" for index in range(1, 9)]

    since = terminal.frames
    await terminal.send_input("\x1bOF")  # end
    await terminal.wait_for_render(since)
    assert _viewport(terminal) == [f"line {index}" for index in range(5, 13)]

    await tui.stop()


def test_searches_normalized_rendered_transcript_text_across_rows():
    from pidrei_tui.alt_screen_search import (
        AltScreenSearchMatch,
        AltScreenSearchSegment,
        find_alt_screen_search_matches,
    )

    assert find_alt_screen_search_matches(["alpha QUICK", "brown fox"], "quick brown") == [
        AltScreenSearchMatch(
            segments=[
                AltScreenSearchSegment(row=0, start_col=6, end_col=11),
                AltScreenSearchSegment(row=1, start_col=0, end_col=5),
            ]
        )
    ]


def test_maps_normalized_ascii_and_unicode_search_matches_back_to_rendered_columns():
    from pidrei_tui.alt_screen_search import (
        AltScreenSearchMatch,
        AltScreenSearchSegment,
        find_alt_screen_search_matches,
    )

    assert find_alt_screen_search_matches(["\x1b[31mfoo  bar\x1b[0m", "A界🙂éZ"], "oo   bar\nA界🙂é") == [
        AltScreenSearchMatch(
            segments=[
                AltScreenSearchSegment(row=0, start_col=1, end_col=3),
                AltScreenSearchSegment(row=0, start_col=5, end_col=8),
                AltScreenSearchSegment(row=1, start_col=0, end_col=6),
            ]
        )
    ]


def test_reuses_indexed_transcript_matches_until_the_query_or_rendered_lines_change():
    from pidrei_tui.alt_screen_search import AltScreenSearchIndex, AltScreenSearchSegment

    index = AltScreenSearchIndex()
    initial = index.search(["alpha needle", "omega"], "needle")
    assert initial.changed is True
    assert len(initial.matches) == 1

    cached = index.search(["alpha needle", "omega"], "needle")
    assert cached.changed is False
    assert cached.matches is initial.matches

    changed_query = index.search(["alpha needle", "omega"], "omega")
    assert changed_query.changed is True
    assert changed_query.matches is not initial.matches
    assert changed_query.matches[0].segments == [AltScreenSearchSegment(row=1, start_col=0, end_col=5)]

    changed_lines = index.search(["alpha needle", "no match"], "omega")
    assert changed_lines.changed is True
    assert changed_lines.matches == []


def test_renders_transcript_search_with_a_muted_placeholder_and_right_aligned_controls():
    from pidrei_tui.alt_screen_search import AltScreenSearchComponent
    from pidrei_tui.utils import strip_terminal_sequences, visible_width

    set_keybindings(KeybindingsManager(TUI_KEYBINDINGS))
    component = AltScreenSearchComponent(lambda _query: None)
    rendered = component.render(48)
    lines = [strip_terminal_sequences(line) for line in rendered]

    assert len(lines) == 3
    assert all(visible_width(line) == 48 for line in lines)
    assert re.match(r"^┌─+┐$", lines[0])
    assert re.match(r"^│ Find in transcript +│$", lines[1])
    assert "\x1b[2m" in rendered[1]
    assert re.match(r"^└─+ ↑ Shift\+Enter · ↓ Enter ─┘$", lines[2])
    controls = lines[2]
    assert component.get_navigation_direction_at(2, controls.index("↑")) == -1
    assert component.get_navigation_direction_at(2, controls.index("Shift+Enter") + 5) == -1
    assert component.get_navigation_direction_at(2, controls.index("·")) is None
    assert component.get_navigation_direction_at(2, controls.index("↓")) == 1
    assert component.get_navigation_direction_at(2, controls.rindex("Enter") + 2) == 1


@pytest.mark.tonio
async def test_populates_transcript_search_results_next_to_the_query():
    from pidrei_tui.alt_screen_search import AltScreenSearchComponent
    from pidrei_tui.utils import strip_terminal_sequences

    set_keybindings(KeybindingsManager(TUI_KEYBINDINGS))
    component = AltScreenSearchComponent(lambda _query: None)
    component.render(48)

    await component.handle_input("n")
    component.set_result(0, 2)
    populated_render = component.render(48)
    populated = [strip_terminal_sequences(line) for line in populated_render]
    assert "n" in populated[1]
    assert "1/2" in populated[1]
    assert "\x1b[2m 1/2 \x1b[22m" in populated_render[1]
    assert not any("Find in transcript" in line for line in populated)


@pytest.mark.tonio
async def test_navigates_transcript_search_with_hoverable_arrow_buttons_and_toggles_it_with_its_shortcut():
    terminal = RecordingTerminal(120, 6)
    tui = TuiAltScreen(
        terminal,
        None,
        None,
        search_navigation_button_style=lambda text, hovered: f"{'\x1b[45m' if hovered else '\x1b[44m'}{text}\x1b[49m",
    )
    tui.add_child(Text("needle one\nmiddle\nneedle two\nend", 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    await terminal.send_input("\x1b[102;6u")
    await terminal.send_input("needle")
    assert await _wait_for_viewport_text(terminal, "1/2")
    assert await _wait_for_viewport_text(terminal, "↑ Shift+Enter · ↓ Enter")

    viewport = terminal.get_viewport()
    arrow_row = next(index for index, line in enumerate(viewport) if "↑" in line and "↓" in line)
    arrow_column = viewport[arrow_row].rindex("Enter")
    await terminal.send_input(f"\x1b[<35;{arrow_column + 1};{arrow_row + 1}M")
    assert await terminal.wait_for_write("\x1b[45m↓ Enter\x1b[49m")
    await terminal.send_input(f"\x1b[<0;{arrow_column + 1};{arrow_row + 1}M")
    assert await _wait_for_viewport_text(terminal, "2/2")
    assert any("↑ Shift+Enter · ↓ Enter" in line for line in terminal.get_viewport())

    viewport = terminal.get_viewport()
    arrow_row = next(index for index, line in enumerate(viewport) if "↑" in line and "↓" in line)
    arrow_column = viewport[arrow_row].index("Shift+Enter") + 3
    await terminal.send_input(f"\x1b[<0;{arrow_column + 1};{arrow_row + 1}M")
    assert await _wait_for_viewport_text(terminal, "1/2")
    assert any("↑ Shift+Enter · ↓ Enter" in line for line in terminal.get_viewport())

    await terminal.send_input("\x1b[102;6u")
    assert await _wait_until(lambda: not any("↑ Shift+Enter · ↓ Enter" in line for line in terminal.get_viewport()))
    await tui.stop()


@pytest.mark.tonio
async def test_does_not_treat_transcript_box_drawing_as_search_navigation_buttons():
    terminal = VirtualTerminal(80, 10)
    tui = TuiAltScreen(terminal)
    transcript_lines = [
        "needle one",
        "middle",
        "needle two",
        "filler",
        "┌────────────────────────────────────────┐",
        "│ box                                    │",
        "└────────────────────────────────────────┘",
        "end",
    ]
    tui.add_child(Text("\n".join(transcript_lines), 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    await terminal.send_input("\x1b[102;6u")
    await terminal.send_input("needle")
    assert await _wait_for_viewport_text(terminal, "1/2")
    assert not any("2/2" in line for line in terminal.get_viewport())

    box_bottom_row = next(index for index, line in enumerate(terminal.get_viewport()) if line.startswith("└"))
    await terminal.send_input(f"\x1b[<0;24;{box_bottom_row + 1}M")
    await terminal.wait_for_render()

    viewport = terminal.get_viewport()
    assert any("1/2" in line for line in viewport)
    assert not any("2/2" in line for line in viewport)
    await tui.stop()


@pytest.mark.tonio
async def test_uses_configured_styles_for_current_and_non_current_search_matches():
    terminal = RecordingTerminal(60, 4)
    tui = TuiAltScreen(
        terminal,
        None,
        None,
        search_match_style=lambda text: f"\x1b[41m{text}\x1b[49m",
        search_current_match_style=lambda text: f"\x1b[42m{text}\x1b[49m",
    )
    tui.add_child(Text("needle first\nmiddle\nneedle second\nend", 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    await terminal.send_input("\x1b[102;6u")
    await terminal.send_input("needle")
    await terminal.wait_for_render(since)

    assert await terminal.wait_for_write("\x1b[42mneedle\x1b[49m")
    assert await terminal.wait_for_write("\x1b[41mneedle\x1b[49m")
    await tui.stop()


@pytest.mark.tonio
async def test_searches_the_transcript_with_ctrl_shift_f_and_restores_editor_focus_on_close():
    terminal = RecordingTerminal(60, 8)
    tui = TuiAltScreen(terminal)
    transcript_lines = []
    for index in range(12):
        if index == 4:
            transcript_lines.append("line 5 needle one")
        elif index == 9:
            transcript_lines.append("line 10 needle two")
        else:
            transcript_lines.append(f"line {index + 1}")
    transcript_text = Text("\n".join(transcript_lines), 0, 0)
    transcript = ScrollView(transcript_text, {"follow": "end", "primary": True})
    editor_inputs: list[str] = []

    class _Editor:
        focused = False

        def render(self, _width):
            return ["editor"]

        def invalidate(self):
            pass

        async def handle_input(self, data):
            editor_inputs.append(data)

    editor = _Editor()
    tui.set_layout_root(
        VStack(
            [
                {"component": transcript, "basis": 0, "grow": 1, "minSize": 1},
                {"component": editor, "basis": 1, "shrink": 0},
            ]
        )
    )
    tui.set_focus(editor)
    await tui.start()
    await terminal.wait_for_render()

    await terminal.send_input("\x1b[102;6u")
    await terminal.send_input("needle")
    assert await _wait_until(lambda: transcript.is_following_end is False)
    assert await _wait_for_viewport_text(terminal, "2/2")
    assert any("↑ Shift+Enter · ↓ Enter" in line for line in terminal.get_viewport())
    assert any("line 10 needle two" in line for line in terminal.get_viewport())
    assert editor_inputs == []
    assert await terminal.wait_for_write("\x1b[1;7mneedle\x1b[22;27m")

    for _ in range(6):
        await terminal.send_input("\x1b[<64;1;4M")
    assert await _wait_until(lambda: transcript.scroll_top == 0)
    assert await _wait_until(lambda: any("needle" in line and "2/2" in line for line in terminal.get_viewport()))

    await terminal.send_input("\x07")
    assert await _wait_for_viewport_text(terminal, "1/2")
    assert any("line 5 needle one" in line for line in terminal.get_viewport())

    await terminal.send_input("\x1b[103;6u")
    assert await _wait_for_viewport_text(terminal, "2/2")
    assert any("line 10 needle two" in line for line in terminal.get_viewport())

    await terminal.send_input("\x1b")
    await terminal.send_input("x")
    assert await _wait_until(lambda: not any("↑ Shift+Enter · ↓ Enter" in line for line in terminal.get_viewport()))
    assert await _wait_until(lambda: editor_inputs == ["x"])

    await tui.stop()


@pytest.mark.tonio
async def test_scrolls_the_transcript_by_half_a_page_with_custom_bindings():
    original_keybindings = get_keybindings()
    terminal = VirtualTerminal(20, 10)
    tui = TuiAltScreen(terminal)
    set_keybindings(
        KeybindingsManager(
            TUI_KEYBINDINGS,
            {"tui.altScreen.halfPageUp": "ctrl+u", "tui.altScreen.halfPageDown": "ctrl+d"},
        )
    )
    try:
        tui.add_child(Text(_lines(30), 0, 0))
        await tui.start()
        await terminal.wait_for_render()
        assert tui.viewport_top == 20

        since = terminal.frames
        await terminal.send_input("\x15")
        await terminal.wait_for_render(since)
        assert tui.viewport_top == 15

        since = terminal.frames
        await terminal.send_input("\x04")
        await terminal.wait_for_render(since)
        assert tui.viewport_top == 20
    finally:
        await tui.stop()
        set_keybindings(original_keybindings)


@pytest.mark.tonio
async def test_scrolls_the_transcript_by_one_line_with_custom_bindings():
    original_keybindings = get_keybindings()
    terminal = VirtualTerminal(20, 10)
    tui = TuiAltScreen(terminal)
    set_keybindings(
        KeybindingsManager(
            TUI_KEYBINDINGS,
            {"tui.altScreen.lineUp": "ctrl+y", "tui.altScreen.lineDown": "ctrl+e"},
        )
    )
    try:
        tui.add_child(Text(_lines(30), 0, 0))
        await tui.start()
        await terminal.wait_for_render()
        assert tui.viewport_top == 20

        since = terminal.frames
        await terminal.send_input("\x19")
        await terminal.wait_for_render(since)
        assert tui.viewport_top == 19

        since = terminal.frames
        await terminal.send_input("\x05")
        await terminal.wait_for_render(since)
        assert tui.viewport_top == 20
    finally:
        await tui.stop()
        set_keybindings(original_keybindings)


@pytest.mark.tonio
async def test_routes_ctrl_modified_viewport_navigation_to_the_focused_component():
    terminal = VirtualTerminal(20, 6)
    tui = TuiAltScreen(terminal)
    transcript = ScrollView(Text(_lines(12), 0, 0), {"follow": "end", "primary": True})
    editor_inputs: list[str] = []

    class _Editor:
        focused = False

        def render(self, width: int) -> list[str]:
            return ["editor"]

        def invalidate(self) -> None:
            pass

        async def handle_input(self, data: str) -> None:
            editor_inputs.append(data)

    editor = _Editor()
    tui.set_layout_root(
        VStack(
            [
                {"component": transcript, "basis": 0, "grow": 1, "minSize": 1},
                {"component": editor, "basis": 1, "shrink": 0},
            ]
        )
    )
    tui.set_focus(editor)
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    await terminal.send_input("\x1bOH")
    await terminal.wait_for_render(since)
    assert transcript.scroll_top == 0
    assert editor_inputs == []

    modified_inputs = ["\x1b[1;5H", "\x1b[1;5F", "\x1b[5;5~", "\x1b[6;5~", "\x1b[57423;5u"]
    for data in modified_inputs:
        await terminal.send_input(data)
    await terminal.send_input("\x1b[57423;5:3u")
    await terminal.wait_for_render()
    assert transcript.scroll_top == 0
    assert editor_inputs == modified_inputs

    since = terminal.frames
    await terminal.send_input("\x1b[6~")
    await terminal.wait_for_render(since)
    assert transcript.scroll_top == 1
    assert editor_inputs == modified_inputs

    await tui.stop()


@pytest.mark.tonio
async def test_jumps_between_osc133_semantic_prompt_markers():
    terminal = VirtualTerminal(20, 3)
    tui = TuiAltScreen(terminal)
    tui.add_child(
        Text(
            "\n".join(line for message in (1, 2, 3, 4) for line in (f"{OSC133_ZONE_START}message {message}", "detail")),
            0,
            0,
        )
    )
    await tui.start()
    await terminal.wait_for_render()
    assert tui.viewport_top == 5

    since = terminal.frames
    await terminal.send_input("\x1b[57419;6u")  # ctrl+shift+up
    await terminal.send_input("\x1b[57419;6:3u")
    await terminal.wait_for_render(since)
    assert tui.viewport_top == 4
    assert _viewport(terminal)[0] == "message 3"

    since = terminal.frames
    await terminal.send_input("\x1b[1;6A")
    await terminal.wait_for_render(since)
    assert tui.viewport_top == 2
    assert _viewport(terminal)[0] == "message 2"

    since = terminal.frames
    await terminal.send_input("\x1b[57420;6u")  # ctrl+shift+down
    await terminal.send_input("\x1b[57420;6:3u")
    await terminal.wait_for_render(since)
    assert tui.viewport_top == 4
    assert _viewport(terminal)[0] == "message 3"

    since = terminal.frames
    await terminal.send_input("\x1b[1;6B")
    await terminal.wait_for_render(since)
    assert tui.viewport_top == 5
    assert _viewport(terminal)[1] == "message 4"
    assert tui.is_following_output is True

    await tui.stop()


@pytest.mark.tonio
async def test_does_not_emit_kitty_graphics_or_osc133_zones_in_iterm2():
    with capabilities({"images": "iterm2", "trueColor": True, "hyperlinks": True}):
        terminal = RecordingTerminal(20, 3)
        tui = TuiAltScreen(terminal)
        tui.add_child(RenderComponent(["\x1b]133;B\x07\x1b]133;C\x07\x1b]133;A\x07content"]))
        tui.add_child(
            Image(
                "AAAA",
                "image/png",
                {"fallbackColor": lambda value: value},
                {"filename": "example.png"},
                {"widthPx": 10, "heightPx": 10},
            )
        )
        await tui.start()
        await terminal.wait_for_render()
        await tui.stop()

        writes = [event["data"] for event in terminal.events if event["type"] == "write"]
        assert all("\x1b_G" not in data for data in writes)
        assert all("\x1b]133;" not in data for data in writes)
        assert all("\x1b]1337;File=" not in data for data in writes)
        assert any("[Image:" in data for data in writes)


@pytest.mark.tonio
async def test_clears_stale_iterm2_image_placements_when_they_leave_the_viewport():
    with capabilities({"images": "iterm2", "trueColor": True, "hyperlinks": True}):
        terminal = RecordingTerminal(20, 3)
        tui = TuiAltScreen(terminal)
        image_line = "\x1b]1337;File=inline=1;width=2;height=auto:AAAA\x07"
        tui.add_child(RenderComponent([image_line, "", "", "after", "more", "end"]))
        await tui.start()
        await terminal.wait_for_render()
        since = terminal.frames
        tui.scroll_to_top()
        await terminal.wait_for_render(since)
        event_count = len(terminal.events)

        since = terminal.frames
        tui.scroll_by(1)
        await terminal.wait_for_render(since)
        assert any(event["type"] == "write" and "\x1b[2J" in event["data"] for event in terminal.events[event_count:])
        await tui.stop()


@pytest.mark.tonio
async def test_crops_a_kitty_image_whose_first_line_is_above_the_viewport():
    terminal = RecordingTerminal(20, 3)
    tui = TuiAltScreen(terminal)
    image_id = 123
    image_line = encode_kitty("AAAA", columns=2, rows=3, image_id=image_id, move_cursor=False)
    register_kitty_image_metadata({"imageId": image_id, "columns": 2, "rows": 3, "widthPx": 100, "heightPx": 100})
    tui.add_child(RenderComponent(["before", image_line, "", "", "after", "end"]))
    await tui.start()
    await terminal.wait_for_render()

    assert tui.viewport_top == 3
    assert any(
        event["type"] == "write" and "i=123" in event["data"] and "y=66,h=34,r=1" in event["data"]
        for event in terminal.events
    )

    await tui.stop()


@pytest.mark.tonio
async def test_reuses_moved_kitty_images_without_dropping_hstack_siblings():
    with capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True}):
        terminal = RecordingTerminal(20, 6)
        tui = TuiAltScreen(terminal)
        label = Text("left", 0, 0)
        image = Image(
            "A" * 8192,
            "image/png",
            {"fallbackColor": lambda value: value},
            {},
            {"widthPx": 100, "heightPx": 100},
        )
        header = Text("header", 0, 0)
        row = HStack([{"component": label, "basis": 10}, {"component": image, "basis": 10}])
        tui.set_layout_root(VStack([{"component": header, "basis": "auto"}, {"component": row, "basis": 4}]))
        await tui.start()
        await terminal.wait_for_render()
        assert await terminal.wait_for_write("\x1b_Ga=T")

        event_count = len(terminal.events)
        label.set_text("changed")
        header.set_text("header\nsecond")
        since = terminal.frames
        tui.request_render()
        await terminal.wait_for_render(since)
        redraw_writes = "".join(event["data"] for event in terminal.events[event_count:] if event["type"] == "write")
        placement_index = redraw_writes.find("\x1b_Ga=p,q=2")
        assert "\x1b_Ga=d,d=a,q=2\x1b\\" in redraw_writes
        assert placement_index > redraw_writes.find("changed")
        assert "\x1b_Ga=T" not in redraw_writes
        assert len(redraw_writes) < 2000, f"expected placement-only redraw, got {len(redraw_writes)} bytes"
        assert any(line.rstrip() == "changed" for line in terminal.get_viewport())
        await tui.stop()


def _image_scroll_view(image_lines: list[str]) -> ScrollView:
    return ScrollView(RenderComponent(image_lines), {"primary": True})


@pytest.mark.tonio
async def test_retains_recently_offscreen_kitty_images_for_placement_only_reuse():
    with capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True}):
        terminal = RecordingTerminal(20, 1)
        tui = TuiAltScreen(terminal)
        image_id = 321
        image_line = encode_kitty("AAAA", columns=2, rows=1, image_id=image_id, move_cursor=False)
        register_kitty_image_metadata({"imageId": image_id, "columns": 2, "rows": 1, "widthPx": 100, "heightPx": 50})
        tui.set_layout_root(_image_scroll_view([image_line, "after"]))
        await tui.start()
        await terminal.wait_for_render()
        assert await terminal.wait_for_write("\x1b_Ga=T")

        event_count = len(terminal.events)
        since = terminal.frames
        tui.scroll_by(1)
        await terminal.wait_for_render(since)
        since = terminal.frames
        tui.scroll_by(-1)
        await terminal.wait_for_render(since)
        reentry_writes = "".join(event["data"] for event in terminal.events[event_count:] if event["type"] == "write")
        assert "\x1b_Ga=p,q=2" in reentry_writes
        assert "\x1b_Ga=T" not in reentry_writes
        assert f"\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\" not in reentry_writes
        await tui.stop()


@pytest.mark.tonio
async def test_evicts_the_least_recently_visible_kitty_image_when_the_cache_is_full():
    with capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True}):
        terminal = RecordingTerminal(20, 1)
        tui = TuiAltScreen(terminal)
        first_image_id = 500
        image_lines = []
        for index in range(18):
            image_id = first_image_id + index
            register_kitty_image_metadata(
                {"imageId": image_id, "columns": 2, "rows": 1, "widthPx": 100, "heightPx": 50}
            )
            image_lines.append(encode_kitty("AAAA", columns=2, rows=1, image_id=image_id, move_cursor=False))
        tui.set_layout_root(_image_scroll_view(image_lines))
        await tui.start()
        await terminal.wait_for_render()
        for _ in range(1, len(image_lines)):
            since = terminal.frames
            tui.scroll_by(1)
            await terminal.wait_for_render(since)
        assert await terminal.wait_for_write(f"\x1b_Ga=d,d=I,i={first_image_id},q=2\x1b\\")

        event_count = len(terminal.events)
        since = terminal.frames
        tui.scroll_to_top()
        await terminal.wait_for_render(since)
        reentry_writes = "".join(event["data"] for event in terminal.events[event_count:] if event["type"] == "write")
        assert "\x1b_Ga=T" in reentry_writes
        await tui.stop()


@pytest.mark.tonio
async def test_evicts_offscreen_kitty_images_when_decoded_raster_memory_exceeds_the_cache_quota():
    with capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True}):
        terminal = RecordingTerminal(20, 1)
        tui = TuiAltScreen(terminal)
        first_image_id = 600
        image_lines = []
        for index in range(4):
            image_id = first_image_id + index
            register_kitty_image_metadata(
                {"imageId": image_id, "columns": 2, "rows": 1, "widthPx": 3840, "heightPx": 2160}
            )
            image_lines.append(encode_kitty("AAAA", columns=2, rows=1, image_id=image_id, move_cursor=False))
        tui.set_layout_root(_image_scroll_view(image_lines))
        await tui.start()
        await terminal.wait_for_render()
        for _ in range(1, len(image_lines)):
            since = terminal.frames
            tui.scroll_by(1)
            await terminal.wait_for_render(since)
        assert await terminal.wait_for_write(f"\x1b_Ga=d,d=I,i={first_image_id},q=2\x1b\\")
        await tui.stop()


@pytest.mark.tonio
async def test_opens_an_osc8_hyperlink_with_specific_or_generic_release_codes_but_not_on_drag():
    terminal = RecordingTerminal(20, 3)
    opened_urls: list[str] = []
    url = "https://example.com/path?q=1"
    bel_url = "https://example.com/bel"
    emoji_url = "https://example.com/emoji"
    tui = TuiAltScreen(terminal, None, None, open_url=opened_urls.append)
    tui.add_child(
        Text(
            f"{hyperlink('link', url)}\n\x1b]8;;{bel_url}\x07link\x1b]8;;\x07\n{hyperlink('🙂', emoji_url)}",
            0,
            0,
        )
    )
    await tui.start()
    # The press handler hit-tests the last *published* frame
    # (`_previous_screen`), which lands after the terminal write — the no-arg
    # settle-wait can lose the race against the first paint, so wait until
    # the frame carrying the links is actually there.
    assert await poll_until(lambda: any(url in line for line in tui._previous_screen))

    for column, row, release_button in ((2, 1, 3), (2, 2, 0), (2, 3, 0)):
        since = terminal.frames
        await terminal.send_input(f"\x1b[<0;{column};{row}M")
        await terminal.send_input(f"\x1b[<{release_button};{column};{row}m")
        await terminal.wait_for_render(since)
    assert opened_urls == [url, bel_url, emoji_url]

    since = terminal.frames
    await terminal.send_input("\x1b[<0;2;1M")
    await terminal.send_input("\x1b[<32;4;1M")
    await terminal.send_input("\x1b[<0;4;1m")
    await terminal.wait_for_render(since)
    assert opened_urls == [url, bel_url, emoji_url]

    await tui.stop()


@pytest.mark.tonio
async def test_selects_visible_text_with_the_mouse_and_copies_it_with_osc52_after_a_generic_release():
    terminal = RecordingTerminal(20, 4)
    tui = TuiAltScreen(terminal)
    tui.add_child(Text("\x1b[1mal\x1b[0mpha\nbeta\ngamma\ndelta", 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    await terminal.send_input("\x1b[<0;1;1M")
    await terminal.send_input("\x1b[<32;4;2M")
    await terminal.send_input("\x1b[<3;4;2m")
    await terminal.wait_for_render(since)

    assert await terminal.wait_for_write(_osc52("alpha\nbeta"))
    assert await terminal.wait_for_write("\x1b[7m")
    assert await terminal.wait_for_write("al\x1b[0m\x1b[7mpha"), (
        "selection inverse must be reapplied after a reset inside the selection"
    )
    assert await _wait_for_viewport_text(terminal, "Copied!")

    await tui.stop()


@pytest.mark.tonio
async def test_uses_an_injected_copy_selection_handler_instead_of_osc52_and_reports_success():
    terminal = RecordingTerminal(20, 4)
    copied: list[str] = []

    async def copy_selection(text: str) -> bool:
        copied.append(text)
        return True

    tui = TuiAltScreen(terminal, None, None, copy_selection=copy_selection)
    tui.add_child(Text("alpha\nbeta\ngamma\ndelta", 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    await terminal.send_input("\x1b[<0;1;1M")
    await terminal.send_input("\x1b[<32;4;2M")
    await terminal.send_input("\x1b[<0;4;2m")
    await terminal.wait_for_render(since)

    assert await _wait_for_viewport_text(terminal, "Copied!")
    assert copied == ["alpha\nbeta"]
    assert not terminal.writes_containing("\x1b]52;c;"), (
        "must not emit OSC 52 when a copy_selection handler is provided"
    )

    await tui.stop()


@pytest.mark.tonio
async def test_leaves_selections_visible_without_copying_when_copy_on_select_is_disabled():
    terminal = RecordingTerminal(20, 4)
    copied: list[str] = []

    async def copy_selection(text: str) -> bool:
        copied.append(text)
        return True

    tui = TuiAltScreen(terminal, None, None, copy_on_select=False, copy_selection=copy_selection)
    tui.add_child(Text("alpha\nbeta\ngamma\ndelta", 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    await terminal.send_input("\x1b[<0;1;1M")
    await terminal.send_input("\x1b[<32;4;2M")
    await terminal.send_input("\x1b[<0;4;2m")
    await terminal.wait_for_render(since)

    assert copied == []
    assert tui.has_active_selection() is True
    assert terminal.writes_containing("\x1b[7m")
    assert all("Copied!" not in line for line in terminal.get_viewport())

    await tui.stop()


@pytest.mark.tonio
async def test_copies_an_active_selection_programmatically():
    terminal = RecordingTerminal(20, 4)
    copied: list[str] = []

    async def copy_selection(text: str) -> bool:
        copied.append(text)
        return True

    tui = TuiAltScreen(terminal, None, None, copy_selection=copy_selection)
    tui.add_child(Text("alpha\nbeta\ngamma\ndelta", 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    assert tui.has_active_selection() is False
    assert await tui.copy_active_selection_to_clipboard() is False

    since = terminal.frames
    await terminal.send_input("\x1b[<0;1;1M")
    await terminal.send_input("\x1b[<32;4;2M")
    await terminal.send_input("\x1b[<0;4;2m")
    await terminal.wait_for_render(since)

    assert await _wait_for_viewport_text(terminal, "Copied!")
    assert copied == ["alpha\nbeta"]
    assert tui.has_active_selection() is True

    copied.clear()
    assert await tui.copy_active_selection_to_clipboard() is True

    assert copied == ["alpha\nbeta"]
    assert await _wait_for_viewport_text(terminal, "Copied!")

    await tui.stop()


@pytest.mark.tonio
async def test_flashes_an_error_when_the_injected_copy_selection_handler_fails():
    terminal = RecordingTerminal(20, 4)

    async def copy_selection(_text: str) -> bool:
        return False

    tui = TuiAltScreen(terminal, None, None, copy_selection=copy_selection)
    tui.add_child(Text("alpha\nbeta\ngamma\ndelta", 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    await terminal.send_input("\x1b[<0;1;1M")
    await terminal.send_input("\x1b[<32;4;2M")
    await terminal.send_input("\x1b[<0;4;2m")
    await terminal.wait_for_render(since)

    assert await _wait_for_viewport_text(terminal, "Copy failed")
    assert not terminal.writes_containing("\x1b]52;c;"), (
        "must not emit OSC 52 when a copy_selection handler is provided"
    )

    await tui.stop()


@pytest.mark.tonio
async def test_does_not_append_whitespace_to_double_click_word_highlighting():
    terminal = RecordingTerminal(20, 1)
    tui = TuiAltScreen(terminal)
    tui.add_child(Text("foo  bar", 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    await terminal.send_input("\x1b[<0;1;1M")
    await terminal.send_input("\x1b[<0;1;1m")
    await terminal.send_input("\x1b[<0;3;1M")

    assert await terminal.wait_for_write("foo\x1b[27m")
    await tui.stop()


@pytest.mark.tonio
async def test_coalesces_slash_and_hyphen_separated_segments_for_double_click_word_selection():
    for line, needle in [
        ("extensions/starline/fixed-editor/compositor.ts", "starline"),
        ("earendil-works/pi-tui", "works"),
    ]:
        copied: list[str] = []
        terminal = RecordingTerminal(80, 1)

        async def copy_selection(text: str, copied=copied) -> bool:
            copied.append(text)
            return True

        tui = TuiAltScreen(terminal, None, None, copy_selection=copy_selection)
        tui.add_child(Text(line, 0, 0))
        await tui.start()
        await terminal.wait_for_render()

        one_based_click_column = line.index(needle) + 1
        since = terminal.frames
        await terminal.send_input(f"\x1b[<0;{one_based_click_column};1M")
        await terminal.send_input(f"\x1b[<0;{one_based_click_column};1m")
        await terminal.send_input(f"\x1b[<0;{one_based_click_column};1M")
        await terminal.send_input(f"\x1b[<0;{one_based_click_column};1m")
        await terminal.wait_for_render(since)

        assert copied == [line]
        await tui.stop()


@pytest.mark.tonio
async def test_highlights_a_complete_whitespace_segment_during_a_word_drag():
    terminal = RecordingTerminal(20, 1)
    tui = TuiAltScreen(terminal)
    tui.add_child(Text("foo  bar", 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    await terminal.send_input("\x1b[<0;1;1M")
    await terminal.send_input("\x1b[<0;1;1m")
    await terminal.send_input("\x1b[<0;2;1M")
    await terminal.send_input("\x1b[<32;4;1M")
    await terminal.wait_for_render(since)

    assert await terminal.wait_for_write("foo  \x1b[27m")
    await tui.stop()


@pytest.mark.tonio
async def test_selects_whole_words_on_double_click_extends_word_drags_and_selects_lines_on_triple_click():
    terminal = RecordingTerminal(20, 2)
    tui = TuiAltScreen(terminal)
    tui.add_child(Text("zero alpha beta\ngamma delta", 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    # The second click lands on a different character in alpha.
    since = terminal.frames
    await terminal.send_input("\x1b[<0;6;1M")
    await terminal.send_input("\x1b[<0;6;1m")
    await terminal.send_input("\x1b[<0;10;1M")
    await terminal.send_input("\x1b[<0;10;1m")
    await terminal.wait_for_render(since)
    assert await terminal.wait_for_write(_osc52("alpha"))

    # A double-click drag includes each word touched, rather than partial words.
    since = terminal.frames
    await terminal.send_input("\x1b[<0;12;1M")
    await terminal.send_input("\x1b[<0;12;1m")
    await terminal.send_input("\x1b[<0;14;1M")
    await terminal.send_input("\x1b[<32;3;2M")
    await terminal.send_input("\x1b[<0;3;2m")
    await terminal.wait_for_render(since)
    assert await terminal.wait_for_write(_osc52("beta\ngamma"))

    since = terminal.frames
    await terminal.send_input("\x1b[<0;7;2M")
    await terminal.send_input("\x1b[<0;7;2m")
    await terminal.send_input("\x1b[<0;9;2M")
    await terminal.send_input("\x1b[<0;9;2m")
    await terminal.send_input("\x1b[<0;11;2M")
    await terminal.send_input("\x1b[<0;11;2m")
    await terminal.wait_for_render(since)
    assert await terminal.wait_for_write(_osc52("gamma delta"))

    await tui.stop()


@pytest.mark.tonio
async def test_does_not_repaint_idle_or_zero_width_selections_on_focus_loss():
    terminal = RecordingTerminal(20, 4)
    tui = TuiAltScreen(terminal)
    tui.add_child(Text("alpha\nbeta\ngamma\ndelta", 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    def write_count() -> int:
        return len([event for event in terminal.events if event["type"] == "write"])

    def clipboard_write_count() -> int:
        return len(terminal.writes_containing("\x1b]52;c;"))

    # pi asserts "no repaint" by comparing write counts after waitForRender;
    # wait_for_render here blocks until a NEW frame, so absence is asserted
    # after a settle sleep instead.
    idle_write_count = write_count()
    await terminal.send_input("\x1b[O")
    await terminal.send_input("\x1b[I")
    await tonio.sleep(0.1)
    assert write_count() == idle_write_count

    # A completed click leaves a zero-width anchor, but later orphaned
    # drag/release events must not extend it.
    since = terminal.frames
    await terminal.send_input("\x1b[<0;1;1M")
    await terminal.send_input("\x1b[<0;1;1m")
    await terminal.send_input("\x1b[<32;4;2M")
    await terminal.send_input("\x1b[<0;4;2m")
    await terminal.wait_for_render(since)
    assert clipboard_write_count() == 0

    # Losing focus after a press without a drag cancels the press without repainting.
    since = terminal.frames
    await terminal.send_input("\x1b[<0;1;3M")
    await terminal.wait_for_render(since)
    pressed_write_count = write_count()
    await terminal.send_input("\x1b[O")
    await terminal.send_input("\x1b[I")
    await tonio.sleep(0.1)
    assert write_count() == pressed_write_count
    await terminal.send_input("\x1b[<32;4;2M")
    await terminal.send_input("\x1b[<0;4;2m")
    await tonio.sleep(0.1)
    assert clipboard_write_count() == 0
    assert terminal.writes_containing("\x1b[?1004h")

    await tui.stop()
    assert terminal.writes_containing("\x1b[?1004l")


@pytest.mark.tonio
async def test_clears_an_active_visible_selection_on_focus_loss_and_ignores_orphan_events():
    terminal = RecordingTerminal(20, 4)
    tui = TuiAltScreen(terminal)
    tui.add_child(Text("alpha\nbeta\ngamma\ndelta", 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    await terminal.send_input("\x1b[<0;1;1M")
    await terminal.send_input("\x1b[<32;4;2M")
    await terminal.wait_for_render(since)
    focus_loss_event_count = len(terminal.events)
    since = terminal.frames
    await terminal.send_input("\x1b[O")
    await terminal.send_input("\x1b[I")
    await terminal.wait_for_render(since)
    focus_loss_writes = "".join(
        event["data"] for event in terminal.events[focus_loss_event_count:] if event["type"] == "write"
    )
    assert "alpha" in focus_loss_writes
    assert "beta" in focus_loss_writes
    assert "\x1b[7m" not in focus_loss_writes

    await terminal.send_input("\x1b[<32;4;2M")
    await terminal.send_input("\x1b[<0;4;2m")
    await tonio.sleep(0.1)
    assert not terminal.writes_containing("\x1b]52;c;")
    await tui.stop()


@pytest.mark.tonio
async def test_retains_a_completed_visible_selection_across_focus_changes():
    terminal = RecordingTerminal(20, 4)
    tui = TuiAltScreen(terminal)
    tui.add_child(Text("alpha\nbeta\ngamma\ndelta", 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    await terminal.send_input("\x1b[<0;1;1M")
    await terminal.send_input("\x1b[<32;4;2M")
    await terminal.send_input("\x1b[<0;4;2m")
    await terminal.wait_for_render(since)
    # The release-triggered copy runs as a detached task (pi voids the
    # promise, but its OSC 52 path is synchronous inside the handler), so
    # drain its whole lifecycle — OSC 52 write, "Copied!" flash, flash expiry
    # repaint — before opening the no-repaint window; any of those would
    # otherwise land inside it as a "write".
    assert await terminal.wait_for_write(_osc52("alpha\nbeta"))
    assert await _wait_for_viewport_text(terminal, "Copied!")
    assert await _wait_until(
        lambda: not any("Copied!" in line for line in terminal.get_viewport()),
        timeout=3.0,
    )
    # One settle beat: the expiry frame we just observed is the last requested
    # render, but its write may still be draining when the count is sampled.
    await tonio.sleep(0.05)
    completed_write_count = len([event for event in terminal.events if event["type"] == "write"])
    await terminal.send_input("\x1b[O")
    await terminal.send_input("\x1b[I")
    await tonio.sleep(0.1)
    assert len([event for event in terminal.events if event["type"] == "write"]) == completed_write_count

    redraw_event_count = len(terminal.events)
    since = terminal.frames
    await tui.render_now(True)
    await terminal.wait_for_render(since)
    redraw_writes = "".join(event["data"] for event in terminal.events[redraw_event_count:] if event["type"] == "write")
    assert "alpha" in redraw_writes
    assert "beta" in redraw_writes
    assert "\x1b[7m" in redraw_writes
    await tui.stop()


@pytest.mark.tonio
async def test_stacks_flash_messages_and_collapses_them_as_they_expire():
    terminal = VirtualTerminal(20, 4)
    tui = TuiAltScreen(terminal)
    tui.add_child(Text("one\ntwo\nthree\nfour", 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    tui.flash("First", 80)
    tui.flash("Second", 500)
    await terminal.wait_for_render(since)
    viewport = terminal.get_viewport()
    assert viewport[0].rstrip().endswith(" First")
    assert viewport[1].rstrip().endswith(" Second")

    await tonio.sleep(0.1)
    await terminal.wait_for_render()
    viewport = terminal.get_viewport()
    assert viewport[0].rstrip().endswith(" Second")
    assert not any("First" in line for line in viewport)

    await tui.stop()


@pytest.mark.tonio
async def test_auto_scrolls_and_extends_a_drag_selection_held_at_the_viewport_edge():
    terminal = RecordingTerminal(20, 4)
    tui = TuiAltScreen(terminal)
    tui.add_child(Text(_lines(10), 0, 0))
    await tui.start()
    await terminal.wait_for_render()
    assert tui.viewport_top == 6

    await terminal.send_input("\x1b[<0;1;3M")
    await terminal.send_input("\x1b[<32;1;1M")
    await tonio.sleep(0.13)
    await terminal.wait_for_render()

    selection_top = tui.viewport_top
    assert selection_top < 6, f"expected auto-scroll above row 6, got {selection_top}"
    since = terminal.frames
    await terminal.send_input("\x1b[<0;1;1m")
    await terminal.wait_for_render(since)

    selected_lines = [f"line {selection_top + index + 1}" for index in range(8 - selection_top)]
    selected_lines.append("l")
    assert await terminal.wait_for_write(_osc52("\n".join(selected_lines)))
    await tui.stop()


@pytest.mark.tonio
async def test_snaps_mouse_selection_to_cjk_emoji_and_combining_grapheme_boundaries():
    terminal = RecordingTerminal(20, 2)
    tui = TuiAltScreen(terminal)
    tui.add_child(Text("A界🙂éZ", 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    wide_selection = _osc52("界🙂")
    since = terminal.frames
    await terminal.send_input("\x1b[<0;3;1M")
    await terminal.send_input("\x1b[<32;4;1M")
    await terminal.send_input("\x1b[<0;4;1m")
    await terminal.wait_for_render(since)
    assert len(await terminal.wait_for_write(wide_selection)) == 1

    since = terminal.frames
    await terminal.send_input("\x1b[<0;5;1M")
    await terminal.send_input("\x1b[<32;2;1M")
    await terminal.send_input("\x1b[<0;2;1m")
    await terminal.wait_for_render(since)
    assert len(await terminal.wait_for_write(wide_selection, count=2)) == 2

    since = terminal.frames
    await terminal.send_input("\x1b[<0;6;1M")
    await terminal.send_input("\x1b[<32;7;1M")
    await terminal.send_input("\x1b[<0;7;1m")
    await terminal.wait_for_render(since)
    assert await terminal.wait_for_write(_osc52("éZ"))

    await tui.stop()


@pytest.mark.tonio
async def test_ignores_horizontal_trackpad_wheel_events():
    terminal = VirtualTerminal(20, 4)
    tui = TuiAltScreen(terminal)
    tui.add_child(Text(_lines(8), 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    await terminal.send_input("\x1b[<66;1;1M")
    await terminal.send_input("\x1b[<67;1;1M")
    await terminal.wait_for_render()
    assert tui.viewport_top == 4
    assert _viewport(terminal) == ["line 5", "line 6", "line 7", "line 8"]

    await tui.stop()


@pytest.mark.tonio
async def test_restores_keyboard_state_before_leaving_alt_mode_and_prints_the_full_document():
    terminal = RecordingTerminal(20, 3)
    tui = TuiAltScreen(terminal)
    tui.add_child(Text("first\nsecond\nthird\nfourth\nfifth\nsixth", 0, 0))
    await tui.start()
    await terminal.wait_for_render()
    await tui.stop()

    start_index = terminal.index_of_type("start")
    alt_screen_enter_index = terminal.index_of_write("\x1b[?1049h")
    stop_index = terminal.index_of_type("stop")
    mouse_disable_index = terminal.index_of_write("\x1b[?1006l")
    main_screen_restore_index = terminal.index_of_write("\x1b[?1049l")
    assert alt_screen_enter_index >= 0
    assert alt_screen_enter_index < start_index
    assert mouse_disable_index >= 0
    assert mouse_disable_index < stop_index
    assert main_screen_restore_index > stop_index

    restore_data = terminal.events[main_screen_restore_index]["data"]
    for word in ("first", "second", "third", "fourth", "fifth", "sixth"):
        assert word in restore_data
    assert restore_data.index("first") < restore_data.index("sixth")


class InputOverlay:
    def __init__(self) -> None:
        self.focused = False
        self.inputs: list[str] = []

    async def handle_input(self, data: str) -> None:
        self.inputs.append(data)

    def render(self, _width: int) -> list[str]:
        return ["overlay"]

    def invalidate(self) -> None:
        pass


@pytest.mark.tonio
async def test_gives_wheel_and_viewport_keys_to_a_focused_overlay():
    terminal = VirtualTerminal(20, 6)
    tui = TuiAltScreen(terminal)
    tui.add_child(Text(_lines(12), 0, 0))
    overlay = InputOverlay()
    await tui.start()
    await terminal.wait_for_render()
    top_before = tui.viewport_top
    handle = tui.show_overlay(overlay)
    await terminal.wait_for_render()
    assert overlay.focused is True

    wheel = "\x1b[<64;10;3M"
    keys = ["\x1b[5~", "\x1b[6~", "\x1bOH", "\x1bOF", wheel]
    since = terminal.frames
    for key in keys:
        await terminal.send_input(key)
    await terminal.wait_for_render(since)

    assert overlay.inputs == keys
    assert tui.viewport_top == top_before

    handle.hide()
    await terminal.wait_for_render()
    since = terminal.frames
    await terminal.send_input("\x1b[5~")
    await terminal.wait_for_render(since)
    assert tui.viewport_top < top_before
    await tui.stop()


@pytest.mark.tonio
async def test_keeps_viewport_scrolling_when_an_overlay_is_not_focused():
    terminal = VirtualTerminal(20, 6)
    tui = TuiAltScreen(terminal)
    editor = InputOverlay()
    tui.add_child(Text(_lines(12), 0, 0))
    tui.set_focus(editor)
    await tui.start()
    await terminal.wait_for_render()
    top_before = tui.viewport_top

    hidden = tui.show_overlay(InputOverlay())
    hidden.set_hidden(True)
    non_capturing = InputOverlay()
    tui.show_overlay(non_capturing, {"nonCapturing": True})
    unfocused = InputOverlay()
    unfocused_handle = tui.show_overlay(unfocused)
    unfocused_handle.unfocus()
    await terminal.wait_for_render()
    assert non_capturing.focused is False
    assert unfocused.focused is False

    since = terminal.frames
    await terminal.send_input("\x1b[5~")
    await terminal.send_input("\x1b[<64;10;3M")
    await terminal.wait_for_render(since)
    assert tui.viewport_top < top_before
    assert non_capturing.inputs == []
    assert unfocused.inputs == []
    await tui.stop()


@pytest.mark.tonio
async def test_keeps_viewport_scrolling_while_transcript_search_is_focused():
    terminal = VirtualTerminal(20, 6)
    tui = TuiAltScreen(terminal)
    tui.add_child(Text(_lines(12), 0, 0))
    await tui.start()
    await terminal.wait_for_render()
    top_before = tui.viewport_top

    await terminal.send_input("\x1b[102;6u")
    assert await _wait_for_viewport_text(terminal, "↑ ↓")

    since = terminal.frames
    await terminal.send_input("\x1b[5~")
    await terminal.send_input("\x1b[<64;1;4M")
    await terminal.wait_for_render(since)
    assert await _wait_until(lambda: tui.viewport_top < top_before)
    assert any("↑ ↓" in line for line in terminal.get_viewport())
    await tui.stop()


# --- component mouse dispatch (pi 0.85.0) ------------------------------------


class _MouseComponent:
    """A dict-literal component in pi; render/invalidate/handle_mouse here."""

    def __init__(self, on_mouse, lines: list[str] | None = None) -> None:
        self._on_mouse = on_mouse
        self._lines = lines if lines is not None else ["control"]
        self.render_count = 0

    def render(self, _width: int) -> list[str]:
        self.render_count += 1
        return self._lines

    def invalidate(self) -> None:
        pass

    async def handle_mouse(self, event):
        return self._on_mouse(event)


@pytest.mark.tonio
async def test_does_not_vertically_redispatch_misses_through_horizontal_layout_containers():
    terminal = VirtualTerminal(20, 2)
    tui = TuiAltScreen(terminal)
    selections = [0]
    select_list = SelectList(
        [{"value": "a", "label": "A"}, {"value": "b", "label": "B"}],
        2,
        {
            "selectedPrefix": lambda text: text,
            "selectedText": lambda text: text,
            "description": lambda text: text,
            "scrollInfo": lambda text: text,
            "noMatch": lambda text: text,
        },
    )

    async def on_select(_item) -> None:
        selections[0] += 1

    select_list.on_select = on_select
    tui.set_layout_root(
        HStack([{"component": select_list, "basis": 10}, {"component": Text("plain", 0, 0), "basis": 10}])
    )
    await tui.start()
    await terminal.wait_for_render()

    await terminal.send_input("\x1b[<0;15;1M")
    await terminal.send_input("\x1b[<0;15;1m")
    await terminal.wait_for_render()
    assert selections[0] == 0
    await tui.stop()


@pytest.mark.tonio
async def test_dispatches_clicks_to_nested_mouse_regions_without_breaking_drag_selection():
    terminal = RecordingTerminal(20, 2)
    tui = TuiAltScreen(terminal)
    clicks = [0]

    def on_mouse(event):
        if event.type != "click":
            return None
        clicks[0] += 1
        return TuiMouseEventResult(handled=True)

    tui.add_child(MouseRegion(Text("clickable\nselectable", 0, 0), on_mouse))
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    await terminal.send_input("\x1b[<0;2;1M")
    await terminal.send_input("\x1b[<0;2;1m")
    await terminal.wait_for_render(since)
    assert clicks[0] == 1

    since = terminal.frames
    await terminal.send_input("\x1b[<0;1;1M")
    await terminal.send_input("\x1b[<32;4;2M")
    await terminal.send_input("\x1b[<0;4;2m")
    await terminal.wait_for_render(since)
    assert clicks[0] == 1
    assert await terminal.wait_for_write("\x1b]52;c;")
    await tui.stop()


@pytest.mark.tonio
async def test_focuses_and_captures_drag_gestures_for_mouse_aware_components():
    terminal = VirtualTerminal(20, 2)
    tui = TuiAltScreen(terminal)
    events: list[str] = []

    def on_mouse(event):
        events.append(event.type)
        if event.type == "press":
            return TuiMouseEventResult(handled=True, capture=True, focus=True)
        return TuiMouseEventResult(handled=True)

    component = _MouseComponent(on_mouse)
    tui.add_child(component)
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    await terminal.send_input("\x1b[<0;1;1M")
    await terminal.send_input("\x1b[<32;5;2M")
    await terminal.send_input("\x1b[<0;5;2m")
    await terminal.wait_for_render(since)

    assert events == ["press", "drag", "release"]
    assert tui.get_focused_component() is component
    await tui.stop()


@pytest.mark.tonio
async def test_reports_consecutive_click_counts_to_component_owned_controls():
    terminal = VirtualTerminal(20, 1)
    tui = TuiAltScreen(terminal)
    click_counts: list[int] = []

    def on_mouse(event):
        if event.type == "press":
            return TuiMouseEventResult(handled=True)
        if event.type == "click":
            click_counts.append(event.click_count if event.click_count is not None else 0)
            return TuiMouseEventResult(handled=True)
        return None

    tui.add_child(_MouseComponent(on_mouse))
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    for _ in range(3):
        await terminal.send_input("\x1b[<0;1;1M")
        await terminal.send_input("\x1b[<0;1;1m")
    await terminal.wait_for_render(since)
    assert click_counts == [1, 2, 3]
    await tui.stop()


@pytest.mark.tonio
async def test_does_not_rerender_for_handled_no_op_pointer_motion():
    terminal = RecordingTerminal(20, 2)
    tui = TuiAltScreen(terminal)
    component = _MouseComponent(
        lambda event: TuiMouseEventResult(handled=True) if event.type == "move" else None,
        ["hover target"],
    )
    tui.add_child(component)
    await tui.start()
    await terminal.wait_for_render()
    rendered_before_motion = component.render_count
    writes_before_motion = len([event for event in terminal.events if event["type"] == "write"])

    await terminal.send_input("\x1b[<35;1;1M")
    await terminal.wait_for_render()
    assert component.render_count == rendered_before_motion
    assert len([event for event in terminal.events if event["type"] == "write"]) == writes_before_motion
    await tui.stop()


@pytest.mark.tonio
async def test_lets_mouse_aware_components_consume_wheel_events_before_viewport_scrolling():
    terminal = VirtualTerminal(20, 3)
    tui = TuiAltScreen(terminal)
    wheel_events = [0]

    def on_mouse(event):
        if event.type != "wheel":
            return None
        wheel_events[0] += 1
        return TuiMouseEventResult(handled=True)

    tui.add_child(MouseRegion(Text(_lines(8), 0, 0), on_mouse))
    await tui.start()
    await terminal.wait_for_render()
    viewport_top = tui.viewport_top

    since = terminal.frames
    await terminal.send_input("\x1b[<64;1;1M")
    await terminal.wait_for_render(since)
    assert wheel_events[0] == 1
    assert tui.viewport_top == viewport_top
    await tui.stop()
