"""Mirror of pi coding-agent src/modes/interactive/components/visual-truncate.ts.

Shared utility for truncating text to visual lines (accounting for line
wrapping). Used by tool_execution and bash_execution for consistent behavior.
"""

from pidrei_tui import Text


def truncate_to_visual_lines(text: str, max_visual_lines: int, width: int, padding_x: int = 0) -> dict:
    """Truncate text to a maximum number of visual lines (from the end).

    ``padding_x`` is 0 when the result goes into a Box (which pads itself)
    and 1 when placed in a plain Container. Returns
    ``{"visualLines", "skippedCount"}``.
    """
    if not text:
        return {"visualLines": [], "skippedCount": 0}

    # Create a temporary Text component to render and get visual lines
    temp_text = Text(text, padding_x, 0)
    all_visual_lines = temp_text.render(width)

    if len(all_visual_lines) <= max_visual_lines:
        return {"visualLines": all_visual_lines, "skippedCount": 0}

    # Take the last N visual lines
    truncated_lines = all_visual_lines[-max_visual_lines:]
    skipped_count = len(all_visual_lines) - max_visual_lines

    return {"visualLines": truncated_lines, "skippedCount": skipped_count}
