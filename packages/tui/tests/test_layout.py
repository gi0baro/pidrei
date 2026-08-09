"""Mirror of pi tui test/layout.test.ts."""

import pytest
import tonio.colored as tonio

from pidrei_tui.components.h_stack import HStack
from pidrei_tui.components.scroll_view import ScrollView
from pidrei_tui.components.text import Text
from pidrei_tui.components.v_stack import VStack
from pidrei_tui.layout import render_layout_frame
from pidrei_tui.terminal_image import encode_kitty, register_kitty_image_metadata
from pidrei_tui.utils import strip_terminal_sequences


class RenderComponent:
    """Component that renders a fixed list of lines."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def render(self, width: int) -> list[str]:
        return self.lines

    def invalidate(self) -> None:
        pass


def visible_lines(lines: list[str]) -> list[str]:
    return [strip_terminal_sequences(line).rstrip() for line in lines]


def _noop() -> None:
    pass


def test_allocates_vertical_grow_space_deterministically():
    frame = render_layout_frame(
        VStack(
            [
                {"component": Text("top", 0, 0), "basis": 1, "shrink": 0},
                {"component": Text("body", 0, 0), "basis": 0, "grow": 1},
            ]
        ),
        10,
        4,
        _noop,
    )

    assert [child.rect.height for child in frame.root.children] == [1, 3]
    assert visible_lines(frame.lines) == ["top", "body", "", ""]


def test_does_not_render_fixed_basis_scroll_content_during_stack_measurement():
    render_count = 0

    class CountingContent:
        def render(self, width: int) -> list[str]:
            nonlocal render_count
            render_count += 1
            return ["one", "two", "three"]

        def invalidate(self) -> None:
            pass

    transcript = ScrollView(CountingContent())
    root = VStack(
        [
            {"component": transcript, "basis": 0, "grow": 1},
            {"component": Text("dock", 0, 0), "basis": "auto"},
        ]
    )
    render_layout_frame(root, 10, 3, _noop)
    assert render_count == 1


def test_paints_only_clipped_rows_from_very_large_scroll_content():
    line_count = 1_000_000_000

    class HugeLines:
        """pi builds a sparse 1e9-entry array; Python cannot, so the content is
        a lazy sequence — scanning it row by row would hang the test."""

        def __len__(self) -> int:
            return line_count

        def __getitem__(self, index: int) -> str:
            offset = line_count - index
            return f"visible {4 - offset}" if 1 <= offset <= 3 else "before" if offset == 4 else ""

    lines = HugeLines()

    class HugeContent:
        def render(self, width: int):
            return lines

        def invalidate(self) -> None:
            pass

    transcript = ScrollView(HugeContent(), {"follow": "end"})

    frame = render_layout_frame(transcript, 10, 3, _noop)
    assert visible_lines(frame.lines) == ["visible 1", "visible 2", "visible 3"]


def test_shrinks_entries_to_their_minimum_sizes():
    frame = render_layout_frame(
        VStack(
            [
                {"component": Text("a1\na2\na3", 0, 0), "shrink": 1, "minSize": 1},
                {"component": Text("b1\nb2\nb3", 0, 0), "shrink": 0},
            ]
        ),
        10,
        4,
        _noop,
    )

    assert [child.rect.height for child in frame.root.children] == [1, 3]
    assert visible_lines(frame.lines) == ["a1", "b1", "b2", "b3"]


def test_includes_nested_minimum_sizes_in_intrinsic_stack_measurement():
    dock = VStack(
        [
            Text("top1\ntop2\ntop3", 0, 0),
            {"component": Text("selector", 0, 0), "minSize": 3},
            Text("below", 0, 0),
            {"component": Text("footer", 0, 0), "minSize": 1},
        ]
    )
    frame = render_layout_frame(
        VStack(
            [
                {"component": Text("body", 0, 0), "basis": 0, "grow": 1, "minSize": 1},
                {"component": dock, "basis": "auto", "minSize": 1},
            ]
        ),
        10,
        9,
        _noop,
    )

    assert visible_lines(frame.lines) == [
        "body",
        "top1",
        "top2",
        "top3",
        "selector",
        "",
        "",
        "below",
        "footer",
    ]


def test_omits_gaps_around_invisible_entries():
    stack = VStack(
        [
            Text("one", 0, 0),
            {"component": Text("hidden", 0, 0), "visible": lambda _viewport: False},
            Text("two", 0, 0),
        ],
        {"gap": 1},
    )
    assert [line.rstrip() for line in stack.render(10)] == ["one", "", "two"]


def test_crops_kitty_images_at_a_scroll_views_lower_boundary():
    image_id = 124
    image_line = encode_kitty("AAAA", columns=2, rows=3, image_id=image_id, move_cursor=False)
    register_kitty_image_metadata({"imageId": image_id, "columns": 2, "rows": 3, "widthPx": 100, "heightPx": 100})
    transcript = ScrollView(RenderComponent(["one", "two", image_line, "", ""]))
    frame = render_layout_frame(
        VStack([{"component": transcript, "basis": 0, "grow": 1}, Text("dock", 0, 0)]),
        20,
        4,
        _noop,
    )

    assert "y=0,h=34,r=1" in frame.lines[2]


def test_composes_horizontal_children_at_allocated_widths():
    frame = render_layout_frame(
        HStack(
            [
                {"component": Text("left", 0, 0), "basis": 6, "shrink": 0},
                {"component": Text("right", 0, 0), "basis": 6, "shrink": 0},
            ]
        ),
        12,
        1,
        _noop,
    )
    assert visible_lines(frame.lines) == ["left  right"]


def test_does_not_paint_zero_width_horizontal_children():
    frame = render_layout_frame(
        HStack(
            [
                {"component": Text("hidden", 0, 0), "basis": 0, "shrink": 0},
                {"component": Text("shown", 0, 0), "basis": 0, "grow": 1},
            ]
        ),
        5,
        1,
        _noop,
    )
    assert visible_lines(frame.lines) == ["shown"]


def test_tracks_follow_end_state_and_returns_unused_scroll_delta():
    scroll_view = ScrollView(Text("1\n2\n3\n4\n5\n6", 0, 0), {"follow": "end", "primary": True})
    render_layout_frame(scroll_view, 10, 3, _noop)
    assert scroll_view.scroll_top == 3
    assert scroll_view.is_following_end is True

    assert scroll_view.scroll_by(-2) == 0
    assert scroll_view.scroll_top == 1
    assert scroll_view.is_following_end is False
    assert scroll_view.scroll_by(-3) == -2
    assert scroll_view.scroll_top == 0
    assert scroll_view.scroll_by(10) == 7
    assert scroll_view.scroll_top == 3
    assert scroll_view.is_following_end is True


@pytest.mark.tonio
async def test_renders_a_transient_proportional_scrollbar_without_replacing_cell_content():
    source_lines = ["abcd界", "abcde2", "abcde3", "abcde4", "abcde5", "abcde6", "abcde7", "abcde8"]
    content_background = "\x1b[42m"
    scrollbar_background = "\x1b[48;5;1m"

    def scrollbar_style(text: str) -> str:
        return f"{scrollbar_background}{text}\x1b[49m"

    content = Text("\n".join(source_lines), 0, 0, lambda text: f"{content_background}{text}\x1b[49m")
    scroll_view = ScrollView(
        content, {"scrollbar": "auto", "scrollbarStyle": scrollbar_style, "scrollbarHideDelayMs": 10}
    )

    def render() -> list[str]:
        return render_layout_frame(scroll_view, 6, 4, _noop).lines

    def thumb_rows(lines: list[str]) -> list[bool]:
        return [scrollbar_background in line for line in lines]

    lines = render()
    assert thumb_rows(lines) == [False, False, False, False]
    assert [strip_terminal_sequences(line) for line in lines] == source_lines[:4]

    scroll_view.scroll_by(2)
    lines = render()
    assert thumb_rows(lines) == [False, True, True, False]
    assert [strip_terminal_sequences(line) for line in lines] == source_lines[2:6]
    assert lines[1].rfind(content_background) < lines[1].rfind(scrollbar_background)

    await tonio.sleep(0.03)
    lines = render()
    assert thumb_rows(lines) == [False, False, False, False]

    scroll_view.scroll_to_end()
    lines = render()
    assert thumb_rows(lines) == [False, False, True, True]
    assert [strip_terminal_sequences(line) for line in lines] == source_lines[4:]

    followed_content = Text("\n".join(source_lines), 0, 0)
    followed = ScrollView(followed_content, {"follow": "end", "scrollbar": "auto", "scrollbarStyle": scrollbar_style})
    render_layout_frame(followed, 6, 4, _noop)
    assert followed.scroll_top == 4
    followed_content.set_text("\n".join([*source_lines, "abcde9"]))
    growth_frame = render_layout_frame(followed, 6, 4, _noop)
    assert followed.scroll_top == 5
    assert all(scrollbar_background not in line for line in growth_frame.lines)

    fitting_content = Text("1\n2", 0, 0)
    automatic = ScrollView(fitting_content, {"scrollbar": "auto", "scrollbarStyle": scrollbar_style})
    render_layout_frame(automatic, 6, 4, _noop)
    automatic.scroll_by(1)
    assert all(scrollbar_background not in line for line in render_layout_frame(automatic, 6, 4, _noop).lines)

    always_fitting = ScrollView(fitting_content, {"scrollbar": "always", "scrollbarStyle": scrollbar_style})
    always_fitting_frame = render_layout_frame(always_fitting, 6, 4, _noop)
    assert always_fitting_frame.root.children[0].rect.width == 5
    assert all(scrollbar_background in line for line in always_fitting_frame.lines)

    always_overflowing = ScrollView(content, {"scrollbar": "always", "scrollbarStyle": scrollbar_style})
    always_overflowing_frame = render_layout_frame(always_overflowing, 6, 4, _noop)
    assert always_overflowing_frame.root.children[0].rect.width == 5
    assert len([line for line in always_overflowing_frame.lines if scrollbar_background in line]) == 2

    def thumb_height_for(content_height: int) -> int:
        sized = ScrollView(
            Text("\n".join(["x"] * content_height), 0, 0),
            {"scrollbar": "auto", "scrollbarStyle": scrollbar_style},
        )
        render_layout_frame(sized, 6, 20, _noop)
        sized.scroll_by(1)
        return len([line for line in render_layout_frame(sized, 6, 20, _noop).lines if scrollbar_background in line])

    assert thumb_height_for(21) == 19
    assert thumb_height_for(40) == 10
    assert thumb_height_for(100) == 4
    assert thumb_height_for(400) == 2


def test_updates_reserved_scrollbar_layout_at_runtime():
    scroll_view = ScrollView(Text("123456", 0, 0), {"scrollbar": "always"})

    def render():
        return render_layout_frame(HStack([scroll_view], {"align": "start"}), 6, 2, _noop)

    always = render()
    assert visible_lines(always.lines) == ["12345", "6"]
    assert always.root.children[0].rect.width == 6
    assert always.root.children[0].children[0].rect.width == 5

    scroll_view.set_scrollbar("hidden")
    assert render().root.children[0].children[0].rect.width == 6
    assert scroll_view.is_scrollbar_visible is False


def test_measures_nested_scroll_content_from_constrained_child_geometry():
    inner = ScrollView(Text("1\n2\n3\n4\n5\n6", 0, 0))
    outer = ScrollView(VStack([{"component": inner, "basis": 2}, Text("tail", 0, 0)]))
    render_layout_frame(outer, 10, 2, _noop)

    assert inner.viewport_height == 2
    assert outer.scroll_by(10) == 9
    assert outer.scroll_top == 1


def test_rebuilds_geometry_after_content_changes():
    text = Text("one", 0, 0)
    root = VStack([text])
    first = render_layout_frame(root, 10, 4, _noop)
    text.set_text("one\ntwo\nthree")
    second = render_layout_frame(root, 10, 4, _noop)

    assert len(first.root.children[0].lines) == 1
    assert len(second.root.children[0].lines) == 3
