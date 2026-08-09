"""Mirror of pi tui src/components/stack.ts.

The flexbox-ish sizing shared by ``VStack``/``HStack`` and the layout engine:
entries carry basis/grow/shrink/min/max, and ``allocate_stack_sizes`` grows or
shrinks them into the available space.

Entry options are camelCase records like pi's ``StackEntryOptions``:
``{"basis"?, "grow"?, "shrink"?, "minSize"?, "maxSize"?, "visible"?}``. A
child is either a bare component or ``{"component": ..., **options}``.
"""

import math

from ..layout_node import LAYOUT_NODE
from ..tui import Container


# JS Number.MAX_SAFE_INTEGER — the "no maximum" sentinel in pi's records.
MAX_SIZE = 2**53 - 1


def _is_stack_entry(child) -> bool:
    # pi: `!("render" in child)` — a plain options record, not a component.
    return isinstance(child, dict)


def _normalize_size(value, fallback: int) -> int:
    if value is None or not math.isfinite(value):
        return fallback
    return max(0, math.floor(value))


class Stack(Container):
    """Base class for the two stack directions. Subclasses set `_layout_type`."""

    _layout_type = "vstack"

    def __init__(self, children: list | None = None, options: dict | None = None) -> None:
        super().__init__()
        options = options or {}
        self._entries: list[dict] = []
        self._gap = _normalize_size(options.get("gap"), 0)
        self._align = options.get("align") or "stretch"
        for child in children or []:
            if _is_stack_entry(child):
                entry_options = {key: value for key, value in child.items() if key != "component"}
                self.add_child(child["component"], entry_options)
            else:
                self.add_child(child)

    def add_child(self, component, options: dict | None = None) -> None:
        super().add_child(component)
        options = options or {}
        entry: dict = {"component": component}
        if options.get("basis") is not None:
            entry["basis"] = options["basis"]
        if options.get("grow") is not None:
            entry["grow"] = _normalize_size(options["grow"], 0)
        if options.get("shrink") is not None:
            entry["shrink"] = _normalize_size(options["shrink"], 1)
        if options.get("minSize") is not None:
            entry["minSize"] = _normalize_size(options["minSize"], 0)
        if options.get("maxSize") is not None:
            entry["maxSize"] = _normalize_size(options["maxSize"], MAX_SIZE)
        if options.get("visible") is not None:
            entry["visible"] = options["visible"]
        self._entries.append(entry)

    def remove_child(self, component) -> None:
        super().remove_child(component)
        for index, entry in enumerate(self._entries):
            if entry["component"] is component:
                del self._entries[index]
                return

    def clear(self) -> None:
        super().clear()
        self._entries = []

    def _layout_node(self) -> dict:
        return {
            "type": self._layout_type,
            "entries": self._entries,
            "gap": self._gap,
            "align": self._align,
        }


setattr(Stack, LAYOUT_NODE, Stack._layout_node)


def visible_stack_entries(entries, viewport: dict) -> list[dict]:
    return [entry for entry in entries if entry.get("visible") is None or entry["visible"](viewport)]


def _clamp_size(size, entry: dict) -> int:
    minimum = max(0, math.floor(entry.get("minSize") or 0))
    maximum = max(minimum, math.floor(entry["maxSize"] if entry.get("maxSize") is not None else MAX_SIZE))
    return max(minimum, min(maximum, max(0, math.floor(size))))


def _entry_grow(entry: dict) -> int:
    return entry["grow"] if entry.get("grow") is not None else 0


def _entry_shrink(entry: dict) -> int:
    return entry["shrink"] if entry.get("shrink") is not None else 1


def _entry_min(entry: dict) -> int:
    return entry["minSize"] if entry.get("minSize") is not None else 0


def _entry_max(entry: dict) -> int:
    return entry["maxSize"] if entry.get("maxSize") is not None else MAX_SIZE


def _distribute(sizes: list[int], entries, amount: int, mode: str) -> None:
    remaining = amount
    while remaining > 0:
        if mode == "grow":
            candidates = [
                (entry, index)
                for index, entry in enumerate(entries)
                if _entry_grow(entry) > 0 and sizes[index] < _entry_max(entry)
            ]
        else:
            candidates = [
                (entry, index)
                for index, entry in enumerate(entries)
                if _entry_shrink(entry) > 0 and sizes[index] > _entry_min(entry)
            ]
        if not candidates:
            return

        def weight_of(entry: dict, index: int) -> int:
            if mode == "grow":
                return _entry_grow(entry)
            return _entry_shrink(entry) * max(1, sizes[index])

        total_weight = sum(weight_of(entry, index) for entry, index in candidates)
        distributed = 0
        for entry, index in candidates:
            if remaining <= 0:
                break
            weight = weight_of(entry, index)
            proposed = max(1, math.floor(remaining * weight / total_weight))
            capacity = _entry_max(entry) - sizes[index] if mode == "grow" else sizes[index] - _entry_min(entry)
            delta = min(remaining, proposed, capacity)
            if delta <= 0:
                continue
            sizes[index] = sizes[index] + (delta if mode == "grow" else -delta)
            remaining -= delta
            distributed += delta
        if distributed == 0:
            return


def allocate_stack_sizes(entries, intrinsic_sizes, available_size: int | None, gap: int) -> list[int]:
    def intrinsic_of(index: int) -> int:
        return intrinsic_sizes[index] if index < len(intrinsic_sizes) else 0

    sizes = [
        _clamp_size(
            intrinsic_of(index) if entry.get("basis") is None or entry["basis"] == "auto" else entry["basis"],
            entry,
        )
        for index, entry in enumerate(entries)
    ]
    if available_size is None:
        return sizes

    content_size = max(0, math.floor(available_size) - max(0, len(entries) - 1) * gap)
    total = sum(sizes)
    if total < content_size:
        _distribute(sizes, entries, content_size - total, "grow")
    elif total > content_size:
        _distribute(sizes, entries, total - content_size, "shrink")
    return sizes
