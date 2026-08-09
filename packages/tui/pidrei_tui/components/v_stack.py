"""Mirror of pi tui src/components/v-stack.ts."""

from .stack import Stack, allocate_stack_sizes, visible_stack_entries


# JS Number.MAX_SAFE_INTEGER: "unbounded height" when measuring visibility.
_UNBOUNDED_HEIGHT = 2**53 - 1


class VStack(Stack):
    """Stacks children vertically, with an optional gap between them."""

    _layout_type = "vstack"

    def render(self, width: int) -> list[str]:
        viewport = {"width": max(1, width), "height": _UNBOUNDED_HEIGHT}
        entries = visible_stack_entries(self._entries, viewport)
        rendered = [entry["component"].render(viewport["width"]) for entry in entries]
        sizes = allocate_stack_sizes(entries, [len(lines) for lines in rendered], None, self._gap)
        lines: list[str] = []
        for index in range(len(entries)):
            if index > 0:
                lines.extend([""] * self._gap)
            child_lines = rendered[index][: sizes[index]]
            lines.extend(child_lines)
            lines.extend([""] * (sizes[index] - len(child_lines)))
        return lines
