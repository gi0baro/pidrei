"""Mirror of pi tui src/layout.ts.

Measures and positions a component tree into a fixed viewport, then paints it
into a screen buffer. Only the alternate-screen renderer uses this: the main
screen renders a document that the terminal scrolls, while here the
application owns both the box tree and the scroll positions.

``LayoutRect``/``LayoutBox``/``LayoutFrame`` are small mutable records rather
than dicts — the layout pass translates and re-clips boxes in place.
"""

import math
import re

from .components.stack import allocate_stack_sizes, visible_stack_entries
from .layout_node import get_layout_node
from .terminal_image import crop_kitty_image_line, get_kitty_image_metadata, is_image_line
from .tui import CURSOR_MARKER, composite_tui_line
from .utils import (
    extract_ansi_code,
    get_active_background_ansi,
    get_grapheme_cell_range,
    js_round,
    slice_by_column,
    visible_width,
)


OSC133_ZONE_PREFIX_RE = re.compile(r"^(?:\x1b\]133;[ABC](?:\x07|\x1b\\))+")


class LayoutRect:
    __slots__ = ("height", "width", "x", "y")

    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class LayoutBox:
    __slots__ = (
        "children",
        "clip",
        "component",
        "layer",
        "line_offset",
        "lines",
        "parent",
        "rect",
        "scroll_content_lines",
        "scroll_view",
    )

    def __init__(self, component, rect: LayoutRect, clip: LayoutRect) -> None:
        self.component = component
        self.rect = rect
        self.clip = clip
        self.children: list[LayoutBox] = []
        self.parent: LayoutBox | None = None
        self.lines: list[str] | None = None
        self.line_offset = 0
        self.scroll_view = None
        self.scroll_content_lines: list[str] | None = None
        self.layer = 0


class LayoutFrame:
    __slots__ = ("height", "lines", "primary_scroll_view", "root", "width")

    def __init__(self, root: LayoutBox, width: int, height: int, lines: list[str], primary_scroll_view) -> None:
        self.root = root
        self.width = width
        self.height = height
        self.lines = lines
        self.primary_scroll_view = primary_scroll_view


class _LayoutContext:
    __slots__ = ("primary_scroll_view", "render_cache", "request_render", "viewport")

    def __init__(self, viewport: dict, request_render) -> None:
        self.viewport = viewport
        self.render_cache: dict[int, tuple[object, dict[int, list[str]]]] = {}
        self.request_render = request_render
        self.primary_scroll_view = None


def _intersect(a: LayoutRect, b: LayoutRect) -> LayoutRect:
    x = max(a.x, b.x)
    y = max(a.y, b.y)
    right = min(a.x + a.width, b.x + b.width)
    bottom = min(a.y + a.height, b.y + b.height)
    return LayoutRect(x, y, max(0, right - x), max(0, bottom - y))


def _render_cached(context: _LayoutContext, component, width: int) -> list[str]:
    safe_width = max(1, math.floor(width))
    # Components are not hashable by contract (pi keys a Map by identity), so
    # the cache is keyed by id() with the component kept alive alongside it.
    entry = context.render_cache.get(id(component))
    if entry is None:
        entry = (component, {})
        context.render_cache[id(component)] = entry
    widths = entry[1]
    lines = widths.get(safe_width)
    if lines is None:
        lines = component.render(safe_width)
        widths[safe_width] = lines
    return lines


def _measure_height(context: _LayoutContext, component, width: int) -> int:
    return len(_render_cached(context, component, width))


def _measure_width(context: _LayoutContext, component, width: int) -> int:
    return max((visible_width(line) for line in _render_cached(context, component, width)), default=0)


def _with_parent(box: LayoutBox, parent: LayoutBox) -> LayoutBox:
    box.parent = parent
    return box


def _translate_box(box: LayoutBox, delta_y: int) -> None:
    box.rect.y += delta_y
    for child in box.children:
        _translate_box(child, delta_y)


def _update_clips(box: LayoutBox, parent_clip: LayoutRect) -> None:
    box.clip = _intersect(parent_clip, box.rect)
    for child in box.children:
        _update_clips(child, box.clip)


def _layout_component(
    context: _LayoutContext, component, x: int, y: int, width: int, height: int | None, clip: LayoutRect
) -> LayoutBox:
    safe_width = max(1, math.floor(width))
    node = get_layout_node(component)
    if not node:
        lines = _render_cached(context, component, safe_width)
        allocated_height = len(lines) if height is None else max(0, math.floor(height))
        line_offset = 0
        if len(lines) > allocated_height > 0:
            cursor_line = next((index for index, line in enumerate(lines) if CURSOR_MARKER in line), -1)
            if cursor_line >= allocated_height:
                line_offset = cursor_line - allocated_height + 1
        rect = LayoutRect(x, y, safe_width, allocated_height)
        box = LayoutBox(component, rect, _intersect(clip, LayoutRect(x, y, safe_width, allocated_height)))
        box.lines = lines
        box.line_offset = line_offset
        return box

    if node["type"] == "scroll":
        state = node["state"]
        previous_scroll_top = state.scroll_top
        content_width = state.get_content_width(safe_width)
        child_box = _layout_component(context, node["component"], x, y - previous_scroll_top, content_width, None, clip)
        content_height = child_box.rect.height
        viewport_height = content_height if height is None else max(0, math.floor(height))
        state.update_layout(content_height, viewport_height, context.request_render)
        _translate_box(child_box, previous_scroll_top - state.scroll_top)
        if state.primary or context.primary_scroll_view is None:
            context.primary_scroll_view = state
        rect = LayoutRect(x, y, safe_width, viewport_height)
        child_clip = _intersect(clip, rect)
        box = LayoutBox(component, rect, child_clip)
        box.children = [child_box]
        box.scroll_view = state
        box.scroll_content_lines = _render_cached(context, node["component"], content_width)
        child_box.parent = box
        _update_clips(child_box, child_clip)
        return box

    entries = visible_stack_entries(node["entries"], context.viewport)
    gap_total = max(0, len(entries) - 1) * node["gap"]
    if node["type"] == "vstack":
        intrinsic_heights = [
            entry["basis"]
            if isinstance(entry.get("basis"), int) and not isinstance(entry.get("basis"), bool)
            else _measure_height(context, entry["component"], safe_width)
            for entry in entries
        ]
        sizes = allocate_stack_sizes(entries, intrinsic_heights, height, node["gap"])
        natural_height = sum(sizes) + gap_total
        allocated_height = natural_height if height is None else max(0, math.floor(height))
        rect = LayoutRect(x, y, safe_width, allocated_height)
        box = LayoutBox(component, rect, _intersect(clip, rect))
        child_y = y
        for index, entry in enumerate(entries):
            box.children.append(
                _with_parent(
                    _layout_component(context, entry["component"], x, child_y, safe_width, sizes[index], box.clip),
                    box,
                )
            )
            child_y += sizes[index] + node["gap"]
        return box

    intrinsic_widths = [
        entry["basis"]
        if isinstance(entry.get("basis"), int) and not isinstance(entry.get("basis"), bool)
        else _measure_width(context, entry["component"], safe_width)
        for entry in entries
    ]
    widths = allocate_stack_sizes(entries, intrinsic_widths, safe_width, node["gap"])
    intrinsic_heights = [
        _measure_height(context, entry["component"], max(1, widths[index])) for index, entry in enumerate(entries)
    ]
    allocated_height = max(intrinsic_heights, default=0) if height is None else max(0, height)
    rect = LayoutRect(x, y, safe_width, allocated_height)
    box = LayoutBox(component, rect, _intersect(clip, rect))
    child_x = x
    for index, entry in enumerate(entries):
        natural_child_height = intrinsic_heights[index]
        child_height = allocated_height if node["align"] == "stretch" else min(allocated_height, natural_child_height)
        child_y = y
        if node["align"] == "center":
            child_y += math.floor((allocated_height - child_height) / 2)
        elif node["align"] == "end":
            child_y += allocated_height - child_height
        child_width = widths[index]
        if child_width == 0:
            empty = LayoutBox(
                entry["component"],
                LayoutRect(child_x, child_y, 0, child_height),
                LayoutRect(child_x, child_y, 0, 0),
            )
            empty.parent = box
            box.children.append(empty)
        else:
            box.children.append(
                _with_parent(
                    _layout_component(
                        context, entry["component"], child_x, child_y, child_width, child_height, box.clip
                    ),
                    box,
                )
            )
        child_x += child_width + node["gap"]
    return box


def _replace_scrollbar_cell(
    line: str, column: int, total_width: int, replacement: str, preserve_target_background: bool
) -> str:
    if is_image_line(line):
        return line

    grapheme_range = get_grapheme_cell_range(line, column)
    start = grapheme_range[0] if grapheme_range else column
    end = grapheme_range[1] if grapheme_range else column + 1
    before = slice_by_column(line, 0, start, True)
    target = slice_by_column(line, start, end - start, True)
    after = slice_by_column(line, end, max(0, total_width - end), True)

    target_prefix = ""
    target_index = 0
    while target_index < len(target):
        ansi = extract_ansi_code(target, target_index)
        if not ansi:
            break
        target_prefix += ansi["code"]
        target_index += ansi["length"]
    before_padding = " " * max(0, start - visible_width(before))
    cell_padding_before = " " * max(0, column - start)
    cell_padding_after = " " * max(0, end - column - 1)
    target_style = "\x1b[0m\x1b]8;;\x07" + (
        get_active_background_ansi(target_prefix) if preserve_target_background else ""
    )
    return f"{before}{before_padding}{target_style}{cell_padding_before}{replacement}{cell_padding_after}{after}"


def get_scrollbar_geometry(box: LayoutBox, include_hidden_auto: bool = False) -> dict | None:
    """Scrollbar geometry for a scroll box: {"column", "trackTop", "trackHeight",
    "thumbTop", "thumbHeight", "maxScrollTop"}, or None when it is not shown.

    `include_hidden_auto` also describes a hidden `auto` scrollbar whose content
    overflows, so hovering its track can reveal it."""
    if box.scroll_view is None or box.rect.width <= 0 or box.rect.height <= 0:
        return None

    if box.children:
        content_height = box.children[0].rect.height
    else:
        content_height = len(box.scroll_content_lines or [])
    track_height = box.rect.height
    can_reveal_hidden_auto = (
        include_hidden_auto and box.scroll_view.scrollbar == "auto" and content_height > track_height
    )
    if not box.scroll_view.is_scrollbar_visible and not can_reveal_hidden_auto:
        return None

    min_thumb_height = min(2, track_height)
    thumb_height = max(min_thumb_height, min(track_height, js_round(track_height * track_height / content_height)))
    max_scroll_top = max(0, content_height - track_height)
    max_thumb_top = track_height - thumb_height
    thumb_offset = 0 if max_scroll_top == 0 else js_round(box.scroll_view.scroll_top / max_scroll_top * max_thumb_top)
    column = box.rect.x + box.rect.width - 1
    if column < box.clip.x or column >= box.clip.x + box.clip.width:
        return None

    return {
        "column": column,
        "trackTop": box.rect.y,
        "trackHeight": track_height,
        "thumbTop": box.rect.y + thumb_offset,
        "thumbHeight": thumb_height,
        "maxScrollTop": max_scroll_top,
    }


def _paint_scrollbar(box: LayoutBox, screen: list[str], total_width: int) -> None:
    geometry = get_scrollbar_geometry(box)
    if not geometry or box.scroll_view is None:
        return

    scroll_view = box.scroll_view
    for offset in range(geometry["trackHeight"]):
        row = geometry["trackTop"] + offset
        if row < box.clip.y or row >= box.clip.y + box.clip.height or row < 0 or row >= len(screen):
            continue
        is_thumb = geometry["thumbTop"] <= row < geometry["thumbTop"] + geometry["thumbHeight"]
        replacement = (
            scroll_view.scrollbar_thumb_style("█" if scroll_view.is_scrollbar_active else "┃")
            if is_thumb
            else scroll_view.scrollbar_track_style("│")
        )
        screen[row] = _replace_scrollbar_cell(
            screen[row], geometry["column"], total_width, replacement, scroll_view.scrollbar != "always"
        )


def _paint_box(box: LayoutBox, screen: list[str], total_width: int) -> None:
    if box.lines is not None:
        offset = box.line_offset
        first_row = max(box.rect.y, box.clip.y, 0)
        last_row = min(box.rect.y + box.rect.height, box.clip.y + box.clip.height, len(screen))
        for row in range(first_row, last_row):
            source_index = offset + row - box.rect.y
            if source_index < 0 or source_index >= len(box.lines):
                continue
            line = OSC133_ZONE_PREFIX_RE.sub("", box.lines[source_index])
            image_metadata = get_kitty_image_metadata(line)
            if image_metadata:
                clip_bottom = min(len(screen), box.clip.y + box.clip.height)
                visible_rows = min(image_metadata["rows"], clip_bottom - row)
                if visible_rows < image_metadata["rows"]:
                    line = crop_kitty_image_line(line, 0, visible_rows)
            # Fast path: a full-width box painting onto an untouched row can use the
            # source line reference directly. Compositing here would rebuild the row
            # string through ANSI/grapheme segmentation every frame; padding is
            # unnecessary because rows are written with erase-line and the final
            # width clamp still truncates over-wide lines.
            if box.rect.x == 0 and box.rect.width >= total_width and (is_image_line(line) or not screen[row]):
                screen[row] = line
            else:
                screen[row] = composite_tui_line(screen[row], line, box.rect.x, box.rect.width, total_width)
    for child in box.children:
        _paint_box(child, screen, total_width)

    if box.scroll_view and box.scroll_content_lines and box.scroll_view.scroll_top > 0 and box.rect.height > 0:
        for image_row in range(box.scroll_view.scroll_top - 1, -1, -1):
            image_line = box.scroll_content_lines[image_row] if image_row < len(box.scroll_content_lines) else ""
            metadata = get_kitty_image_metadata(image_line)
            if metadata:
                hidden_rows = box.scroll_view.scroll_top - image_row
                if hidden_rows < metadata["rows"]:
                    visible_rows = min(box.rect.height, metadata["rows"] - hidden_rows)
                    cropped = crop_kitty_image_line(image_line, hidden_rows, visible_rows)
                    if box.rect.x == 0 and box.rect.width >= total_width:
                        screen[box.rect.y] = cropped
                break
            if image_line != "":
                break

    _paint_scrollbar(box, screen, total_width)


def render_layout_frame(root, width: int, height: int, request_render) -> LayoutFrame:
    safe_width = max(1, math.floor(width))
    safe_height = max(1, math.floor(height))
    context = _LayoutContext({"width": safe_width, "height": safe_height}, request_render)
    root_box = _layout_component(
        context, root, 0, 0, safe_width, safe_height, LayoutRect(0, 0, safe_width, safe_height)
    )
    lines = [""] * safe_height
    _paint_box(root_box, lines, safe_width)
    return LayoutFrame(root_box, safe_width, safe_height, lines, context.primary_scroll_view)


def _contains_point(rect: LayoutRect, x: int, y: int) -> bool:
    return rect.x <= x < rect.x + rect.width and rect.y <= y < rect.y + rect.height


def get_layout_boxes_at(frame: LayoutFrame, x: int, y: int) -> list[LayoutBox]:
    """Return the visual hit path from the deepest component to the layout root."""
    result: list[tuple[LayoutBox, int]] = []

    def visit(box: LayoutBox, depth: int) -> None:
        if not _contains_point(box.clip, x, y):
            return
        result.append((box, depth))
        for child in box.children:
            visit(child, depth + 1)

    visit(frame.root, 0)
    result.sort(key=lambda entry: (-entry[0].layer, -entry[1]))
    return [box for box, _ in result]


def get_scroll_view_box(frame: LayoutFrame, scroll_view) -> LayoutBox | None:
    def visit(box: LayoutBox) -> LayoutBox | None:
        if box.scroll_view is scroll_view:
            return box
        for child in box.children:
            match = visit(child)
            if match:
                return match
        return None

    return visit(frame.root)


def get_scroll_views_at(frame: LayoutFrame, x: int, y: int) -> list:
    result: list[tuple] = []

    def visit(box: LayoutBox, depth: int) -> None:
        if not _contains_point(box.clip, x, y):
            return
        if box.scroll_view and _contains_point(box.rect, x, y):
            result.append((depth, box.scroll_view))
        for child in box.children:
            visit(child, depth + 1)

    visit(frame.root, 0)
    # Innermost first; pi's sort is stable, so same-depth views keep tree order.
    result.sort(key=lambda entry: -entry[0])
    return [scroll_view for _, scroll_view in result]
