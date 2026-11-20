"""Mirror of pi tui test/overlay-terminal-image.test.ts.

Regression tests for #11584: overlays left a hole where they intersected a
terminal image.

pi resets its recorded `writes` string between steps; here each step marks
the write log and reads what came after the mark. Overlays are shown and
hidden through `tui.apply`, under the UI state lock, since the render loop
walks the overlay stack in parallel.
"""

import math
from contextlib import contextmanager

import pytest

from pidrei_tui.terminal_image import (
    delete_kitty_image,
    encode_iterm2,
    encode_kitty,
    get_image_line_extent,
    register_kitty_image_metadata,
    reset_capabilities_cache,
    set_capabilities,
)
from pidrei_tui.tui_alt_screen import TuiAltScreen
from pidrei_tui.tui_main_screen import TuiMainScreen

from .virtual_terminal import VirtualTerminal


class RecordingTerminal(VirtualTerminal):
    def __init__(self, columns: int = 80, rows: int = 24) -> None:
        super().__init__(columns, rows)
        self._writes: list[str] = []

    async def write(self, data: str) -> None:
        self._writes.append(data)
        await super().write(data)

    def mark(self) -> int:
        return len(self._writes)

    def writes_since(self, mark: int) -> str:
        return "".join(self._writes[mark:])


class StaticComponent:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def render(self, width: int) -> list[str]:
        return self.lines

    def invalidate(self) -> None:
        pass


def _panel() -> StaticComponent:
    return StaticComponent(["PANEL"])


@contextmanager
def capabilities(caps: dict):
    set_capabilities(caps)
    try:
        yield
    finally:
        reset_capabilities_cache()


def _show(tui, component, options: dict):
    return tui.apply(lambda: tui.show_overlay(component, options))


async def _step(terminal: RecordingTerminal, change) -> tuple[str, object]:
    """Apply `change`, wait for the frame it requests, and return what was written since and `change`'s result."""
    mark = terminal.mark()
    since = terminal.frames
    result = change()
    await terminal.wait_for_render(since)
    return terminal.writes_since(mark), result


@pytest.mark.tonio
async def test_deletes_a_kitty_image_under_an_overlay_and_redraws_it_after_the_overlay_closes():
    with capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True}):
        terminal = RecordingTerminal(20, 6)
        tui = TuiMainScreen(terminal)
        image = encode_kitty("AAAA", columns=4, rows=4, image_id=11584, move_cursor=False)
        tui.add_child(StaticComponent(["top", image, "", "", "", "bottom"]))
        await tui.start()
        await terminal.wait_for_render()

        # Row 3 is the third image row, below the image line itself.
        writes, overlay = await _step(terminal, lambda: _show(tui, _panel(), {"row": 3, "col": 2, "width": 8}))
        assert delete_kitty_image(11584) in writes, "the covered image is deleted"
        assert "\x1b_Ga=T" not in writes, "the covered image is not drawn again"
        assert terminal.get_viewport()[3].rstrip() == "  PANEL"

        writes, _ = await _step(terminal, lambda: tui.apply(overlay.hide))
        assert image in writes, "the image is drawn again once uncovered"
        await tui.stop()


@pytest.mark.tonio
async def test_keeps_images_whose_rows_the_overlay_does_not_touch():
    with capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True}):
        terminal = RecordingTerminal(20, 6)
        tui = TuiMainScreen(terminal)
        image = encode_kitty("AAAA", columns=4, rows=2, image_id=11585, move_cursor=False)
        tui.add_child(StaticComponent([image, "", "text", "", "", ""]))
        await tui.start()
        await terminal.wait_for_render()

        writes, _ = await _step(terminal, lambda: _show(tui, _panel(), {"row": 4, "col": 0, "width": 8}))
        assert delete_kitty_image(11585) not in writes
        assert terminal.get_viewport()[4].rstrip() == "PANEL"
        await tui.stop()


@pytest.mark.tonio
async def test_keeps_an_image_beside_an_overlay_that_shares_its_rows():
    with capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True}):
        terminal = RecordingTerminal(20, 6)
        tui = TuiMainScreen(terminal)
        image = encode_kitty("AAAA", columns=4, rows=4, image_id=11588, move_cursor=False)
        tui.add_child(StaticComponent(["top", image, "", "", "", "bottom"]))
        await tui.start()
        await terminal.wait_for_render()

        # A sidebar at columns 10-17 next to the image at columns 0-3.
        sidebar = StaticComponent(["SIDE", "SIDE", "SIDE", "SIDE", "SIDE", "SIDE"])
        writes, _ = await _step(terminal, lambda: _show(tui, sidebar, {"row": 0, "col": 10, "width": 8}))
        # The main screen redraws an image whenever a row it covers changes, so it may delete it first.
        assert writes.rfind(image) > writes.rfind(delete_kitty_image(11588)), "the image beside the sidebar is drawn"
        viewport = terminal.get_viewport()
        assert viewport[2][10:].rstrip() == "SIDE", "rows below the image line show the sidebar"
        # The image line itself still passes through, so the sidebar skips that row.
        assert viewport[1].strip() == ""
        await tui.stop()


@pytest.mark.tonio
async def test_keeps_a_fullscreen_image_beside_an_overlay_that_shares_its_rows():
    with capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True}):
        terminal = RecordingTerminal(20, 6)
        tui = TuiAltScreen(terminal)
        image_id = 11591
        image = encode_kitty("AAAA", columns=4, rows=3, image_id=image_id, move_cursor=False)
        register_kitty_image_metadata({"imageId": image_id, "columns": 4, "rows": 3, "widthPx": 100, "heightPx": 100})
        tui.set_layout_root(StaticComponent(["top", image, "", "", "after", ""]))
        await tui.start()
        await terminal.wait_for_render()

        sidebar = StaticComponent(["SIDE", "SIDE", "SIDE", "SIDE"])
        writes, _ = await _step(terminal, lambda: _show(tui, sidebar, {"row": 1, "col": 10, "width": 8}))
        assert "\x1b_Ga=p,q=2" in writes, "the image is placed again after the redraw"
        assert f"i={image_id}" in writes
        assert terminal.get_viewport()[2][10:].rstrip() == "SIDE"
        await tui.stop()


@pytest.mark.tonio
async def test_hides_an_image_of_unknown_width_under_any_overlay_sharing_its_rows():
    with capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True}):
        terminal = RecordingTerminal(20, 6)
        tui = TuiMainScreen(terminal)
        # No `c=` and no registered metadata: the width cannot be read.
        image = encode_kitty("AAAA", rows=2, image_id=11589, move_cursor=False)
        tui.add_child(StaticComponent(["top", image, "", "", "", ""]))
        await tui.start()
        await terminal.wait_for_render()

        writes, _ = await _step(terminal, lambda: _show(tui, _panel(), {"row": 2, "col": 15, "width": 5}))
        assert delete_kitty_image(11589) in writes
        await tui.stop()


def test_reads_the_cells_an_image_line_covers():
    kitty = f"  {encode_kitty('AAAA', columns=6, rows=3, image_id=11590, move_cursor=False)}"
    assert get_image_line_extent(kitty) == {"firstRow": 0, "lastRow": 2, "startCol": 2, "endCol": 8}
    iterm2 = f"\x1b[2A{encode_iterm2('AAAA', width=5, height='auto')}"
    assert get_image_line_extent(iterm2) == {"firstRow": -2, "lastRow": 0, "startCol": 0, "endCol": 5}
    auto_width = encode_iterm2("AAAA", width="50%", height="auto")
    assert get_image_line_extent(auto_width)["endCol"] == math.inf
    assert get_image_line_extent("plain text") is None


@pytest.mark.tonio
async def test_hides_an_image_inside_an_overlay_when_a_later_overlay_covers_it():
    with capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True}):
        terminal = RecordingTerminal(20, 6)
        tui = TuiMainScreen(terminal)
        image = encode_kitty("AAAA", columns=4, rows=3, image_id=11587, move_cursor=False)
        tui.add_child(StaticComponent(["", "", "", "", "", ""]))
        await tui.start()
        _show(tui, StaticComponent([image, "", ""]), {"row": 0, "col": 0, "width": 10})
        await terminal.wait_for_render()
        assert image in terminal.writes_since(0), "the overlay's own image is drawn"

        writes, _ = await _step(terminal, lambda: _show(tui, _panel(), {"row": 2, "col": 0, "width": 8}))
        assert delete_kitty_image(11587) in writes, "the image under the top overlay is deleted"
        assert terminal.get_viewport()[2].rstrip() == "PANEL"
        await tui.stop()


@pytest.mark.tonio
async def test_hides_an_iterm2_image_when_an_overlay_covers_rows_above_its_image_line():
    with capabilities({"images": "iterm2", "trueColor": True, "hyperlinks": True}):
        terminal = RecordingTerminal(20, 6)
        tui = TuiMainScreen(terminal)
        image = f"\x1b[2A{encode_iterm2('AAAA', width=4, height='auto')}"
        tui.add_child(StaticComponent(["top", "", "", image, "", "bottom"]))
        await tui.start()
        await terminal.wait_for_render()

        writes, overlay = await _step(terminal, lambda: _show(tui, _panel(), {"row": 1, "col": 0, "width": 8}))
        assert "\x1b]1337;File=" not in writes, "the covered image is not drawn again"
        # The virtual terminal ignores the iTerm2 sequence but applies its cursor-up prefix, so check writes, not
        # the viewport.
        assert "PANEL" in writes

        writes, _ = await _step(terminal, lambda: tui.apply(overlay.hide))
        assert image in writes, "the image is drawn again once uncovered"
        await tui.stop()


@pytest.mark.tonio
async def test_removes_kitty_placements_in_fullscreen_mode_while_an_overlay_covers_them():
    with capabilities({"images": "kitty", "trueColor": True, "hyperlinks": True}):
        terminal = RecordingTerminal(20, 6)
        tui = TuiAltScreen(terminal)
        image_id = 11586
        image = encode_kitty("AAAA", columns=4, rows=3, image_id=image_id, move_cursor=False)
        register_kitty_image_metadata({"imageId": image_id, "columns": 4, "rows": 3, "widthPx": 100, "heightPx": 100})
        tui.set_layout_root(StaticComponent(["top", image, "", "", "after", ""]))
        await tui.start()
        await terminal.wait_for_render()

        writes, overlay = await _step(terminal, lambda: _show(tui, _panel(), {"row": 2, "col": 1, "width": 8}))
        assert "\x1b_Ga=d,d=a,q=2\x1b\\" in writes, "visible placements are deleted"
        assert f"i={image_id}" not in writes, "the covered image is not placed again"
        assert terminal.get_viewport()[2].rstrip() == " PANEL"

        writes, _ = await _step(terminal, lambda: tui.apply(overlay.hide))
        assert "\x1b_Ga=p,q=2" in writes, "the uploaded image is placed again"
        assert "\x1b_Ga=T" not in writes, "the image data is not uploaded again"
        await tui.stop()
