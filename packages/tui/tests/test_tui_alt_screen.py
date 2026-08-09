"""Mirror of pi tui test/tui-alt-screen.test.ts.

Viewport expectations are right-stripped (see virtual_terminal.py). pi's
`waitForRender()` awaits the next frame; here the frame counter is sampled
before each trigger and passed to `wait_for_render` so the assertions never
read a stale viewport.
"""

import base64
import os
from contextlib import contextmanager

import pytest
import tonio.colored as tonio

from pidrei_tui.components.h_stack import HStack
from pidrei_tui.components.image import Image
from pidrei_tui.components.scroll_view import ScrollView
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
from pidrei_tui.tui_alt_screen import TuiAltScreen

from .virtual_terminal import VirtualTerminal


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
async def test_drags_a_visible_scrollbar_thumb_and_keeps_it_visible_until_release():
    terminal = RecordingTerminal(10, 5)
    tui = TuiAltScreen(terminal)
    scroll_view = ScrollView(
        Text(_lines(20), 0, 0),
        {"primary": True, "scrollbar": "auto", "scrollbarHideDelayMs": 50},
    )
    tui.set_layout_root(scroll_view)
    await tui.start()
    await terminal.wait_for_render()
    assert scroll_view.is_scrollbar_visible is False

    since = terminal.frames
    await terminal.send_input("\x1b[<65;10;1M")
    await terminal.wait_for_render(since)
    assert scroll_view.scroll_top == 1
    assert scroll_view.is_scrollbar_visible is True

    await terminal.send_input("\x1b[<0;10;1M")
    await terminal.wait_for_render()
    await tonio.sleep(0.07)
    assert scroll_view.is_scrollbar_visible is True

    since = terminal.frames
    await terminal.send_input("\x1b[<32;10;4M")
    await terminal.wait_for_render(since)
    assert scroll_view.scroll_top == 15
    assert _viewport(terminal) == ["line 16", "line 17", "line 18", "line 19", "line 20"]

    await terminal.send_input("\x1b[<0;10;4m")
    await terminal.wait_for_render()
    assert scroll_view.is_scrollbar_visible is True
    await tonio.sleep(0.07)
    assert scroll_view.is_scrollbar_visible is True
    await terminal.send_input("\x1b[<35;9;4M")
    await tonio.sleep(0.07)
    assert scroll_view.is_scrollbar_visible is False

    since = terminal.frames
    await terminal.send_input("\x1b[<64;10;5M")
    await terminal.wait_for_render(since)
    assert scroll_view.scroll_top == 14
    await tonio.sleep(0.07)
    assert scroll_view.is_scrollbar_visible is True
    await terminal.send_input("\x1b[<35;9;5M")
    await tonio.sleep(0.07)
    assert scroll_view.is_scrollbar_visible is False

    assert not terminal.writes_containing("\x1b]52;c;")
    await tui.stop()


@pytest.mark.tonio
async def test_keeps_the_scrollbar_column_selectable_while_the_thumb_is_hidden():
    terminal = RecordingTerminal(10, 2)
    tui = TuiAltScreen(terminal)
    scroll_view = ScrollView(Text("123456789A\nabcdefghij\nmore\nlines", 0, 0), {"scrollbar": "auto"})
    tui.set_layout_root(scroll_view)
    await tui.start()
    await terminal.wait_for_render()
    assert scroll_view.is_scrollbar_visible is False

    since = terminal.frames
    await terminal.send_input("\x1b[<0;10;1M")
    await terminal.send_input("\x1b[<32;10;2M")
    await terminal.send_input("\x1b[<0;10;2m")
    await terminal.wait_for_render(since)

    assert terminal.writes_containing(_osc52("A\nabcdefghij"))
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
        assert terminal.writes_containing("\x1b_Ga=T")

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
        assert terminal.writes_containing("\x1b_Ga=T")

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
        assert terminal.writes_containing(f"\x1b_Ga=d,d=I,i={first_image_id},q=2\x1b\\")

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
        assert terminal.writes_containing(f"\x1b_Ga=d,d=I,i={first_image_id},q=2\x1b\\")
        await tui.stop()


@pytest.mark.tonio
async def test_opens_an_osc8_hyperlink_on_click_but_not_on_drag():
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
    await terminal.wait_for_render()

    for column, row in ((2, 1), (2, 2), (2, 3)):
        since = terminal.frames
        await terminal.send_input(f"\x1b[<0;{column};{row}M")
        await terminal.send_input(f"\x1b[<0;{column};{row}m")
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
async def test_selects_visible_text_with_the_mouse_and_copies_it_with_osc52():
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

    assert terminal.writes_containing(_osc52("alpha\nbeta"))
    assert terminal.writes_containing("\x1b[7m")
    assert terminal.writes_containing("\x1b[7m\x1b[0m\x1b[7m"), (
        "selection inverse must be reapplied after layout segment resets"
    )
    assert any("Copied!" in line for line in terminal.get_viewport())

    await tui.stop()


@pytest.mark.tonio
async def test_does_not_append_whitespace_to_double_click_word_highlighting():
    terminal = RecordingTerminal(20, 1)
    tui = TuiAltScreen(terminal)
    tui.add_child(Text("foo  bar", 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    since = terminal.frames
    await terminal.send_input("\x1b[<0;1;1M")
    await terminal.send_input("\x1b[<0;1;1m")
    await terminal.send_input("\x1b[<0;3;1M")
    await terminal.wait_for_render(since)

    assert terminal.writes_containing("foo\x1b[27m")
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

    assert terminal.writes_containing("foo  \x1b[27m")
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
    assert terminal.writes_containing(_osc52("alpha"))

    # A double-click drag includes each word touched, rather than partial words.
    since = terminal.frames
    await terminal.send_input("\x1b[<0;12;1M")
    await terminal.send_input("\x1b[<0;12;1m")
    await terminal.send_input("\x1b[<0;14;1M")
    await terminal.send_input("\x1b[<32;3;2M")
    await terminal.send_input("\x1b[<0;3;2m")
    await terminal.wait_for_render(since)
    assert terminal.writes_containing(_osc52("beta\ngamma"))

    since = terminal.frames
    await terminal.send_input("\x1b[<0;7;2M")
    await terminal.send_input("\x1b[<0;7;2m")
    await terminal.send_input("\x1b[<0;9;2M")
    await terminal.send_input("\x1b[<0;9;2m")
    await terminal.send_input("\x1b[<0;11;2M")
    await terminal.send_input("\x1b[<0;11;2m")
    await terminal.wait_for_render(since)
    assert terminal.writes_containing(_osc52("gamma delta"))

    await tui.stop()


@pytest.mark.tonio
async def test_ignores_orphan_selection_events_and_cancels_an_active_selection_on_focus_loss():
    terminal = RecordingTerminal(20, 4)
    tui = TuiAltScreen(terminal)
    tui.add_child(Text("alpha\nbeta\ngamma\ndelta", 0, 0))
    await tui.start()
    await terminal.wait_for_render()

    def clipboard_write_count() -> int:
        return len(terminal.writes_containing("\x1b]52;c;"))

    # A completed click leaves a zero-width anchor, but later orphaned
    # drag/release events must not extend it.
    since = terminal.frames
    await terminal.send_input("\x1b[<0;1;1M")
    await terminal.send_input("\x1b[<0;1;1m")
    await terminal.send_input("\x1b[<32;4;2M")
    await terminal.send_input("\x1b[<0;4;2m")
    await terminal.wait_for_render(since)
    assert clipboard_write_count() == 0

    # Losing focus also cancels a press whose matching release never arrived.
    since = terminal.frames
    await terminal.send_input("\x1b[<0;1;1M")
    await terminal.send_input("\x1b[O")
    await terminal.send_input("\x1b[I")
    await terminal.send_input("\x1b[<32;4;2M")
    await terminal.send_input("\x1b[<0;4;2m")
    await terminal.wait_for_render(since)
    assert clipboard_write_count() == 0
    assert terminal.writes_containing("\x1b[?1004h")

    await tui.stop()
    assert terminal.writes_containing("\x1b[?1004l")


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
    assert terminal.writes_containing(_osc52("\n".join(selected_lines)))
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
    assert len(terminal.writes_containing(wide_selection)) == 1

    since = terminal.frames
    await terminal.send_input("\x1b[<0;5;1M")
    await terminal.send_input("\x1b[<32;2;1M")
    await terminal.send_input("\x1b[<0;2;1m")
    await terminal.wait_for_render(since)
    assert len(terminal.writes_containing(wide_selection)) == 2

    since = terminal.frames
    await terminal.send_input("\x1b[<0;6;1M")
    await terminal.send_input("\x1b[<32;7;1M")
    await terminal.send_input("\x1b[<0;7;1m")
    await terminal.wait_for_render(since)
    assert terminal.writes_containing(_osc52("éZ"))

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
