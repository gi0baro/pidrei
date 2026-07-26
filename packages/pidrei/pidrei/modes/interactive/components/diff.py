"""Mirror of pi coding-agent src/modes/interactive/components/diff.ts.

Deviation: pi uses jsdiff's ``diffWords`` for intra-line highlighting; here
``difflib.SequenceMatcher`` runs over word/whitespace tokens. The visible
result (inverse video on changed tokens, leading indentation not
highlighted) matches; exact token grouping can differ between the engines.
"""

import difflib
import re

from ..theme import theme


_DIFF_LINE_RE = re.compile(r"^([+\-\s])(\s*\d*)\s(.*)$")
_TOKEN_RE = re.compile(r"\S+|\s+")
_LEADING_WS_RE = re.compile(r"^(\s*)")


def _parse_diff_line(line: str) -> dict | None:
    """Parse "+123 content" / "-123 content" / " 123 content" lines."""
    match = _DIFF_LINE_RE.match(line)
    if not match:
        return None
    return {"prefix": match.group(1), "lineNum": match.group(2), "content": match.group(3)}


def _replace_tabs(text: str) -> str:
    """Replace tabs with spaces for consistent rendering."""
    return text.replace("\t", "   ")


def _render_intra_line_diff(old_content: str, new_content: str) -> dict:
    """Word-level diff rendered with inverse on changed parts.

    Strips leading whitespace from inverse to avoid highlighting indentation.
    """
    old_tokens = _TOKEN_RE.findall(old_content)
    new_tokens = _TOKEN_RE.findall(new_content)
    matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens, autojunk=False)

    removed_line = ""
    added_line = ""
    is_first_removed = True
    is_first_added = True

    def emit_removed(value: str) -> str:
        nonlocal removed_line, is_first_removed
        if is_first_removed:
            leading_ws = _LEADING_WS_RE.match(value).group(1)
            value = value[len(leading_ws) :]
            removed_line += leading_ws
            is_first_removed = False
        if value:
            removed_line += theme.inverse(value)
        return value

    def emit_added(value: str) -> str:
        nonlocal added_line, is_first_added
        if is_first_added:
            leading_ws = _LEADING_WS_RE.match(value).group(1)
            value = value[len(leading_ws) :]
            added_line += leading_ws
            is_first_added = False
        if value:
            added_line += theme.inverse(value)
        return value

    for op, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if op == "equal":
            common = "".join(old_tokens[old_start:old_end])
            removed_line += common
            added_line += common
        else:
            removed = "".join(old_tokens[old_start:old_end])
            added = "".join(new_tokens[new_start:new_end])
            if removed:
                emit_removed(removed)
            if added:
                emit_added(added)

    return {"removedLine": removed_line, "addedLine": added_line}


def render_diff(diff_text: str, options: dict | None = None) -> str:
    """Render a diff string with colored lines and intra-line highlighting.

    - Context lines: dim/gray
    - Removed lines: red, with inverse on changed tokens
    - Added lines: green, with inverse on changed tokens
    """
    lines = diff_text.split("\n")
    result: list = []

    i = 0
    while i < len(lines):
        line = lines[i]
        parsed = _parse_diff_line(line)

        if parsed is None:
            result.append(theme.fg("toolDiffContext", line))
            i += 1
            continue

        if parsed["prefix"] == "-":
            # Collect consecutive removed lines
            removed_lines: list = []
            while i < len(lines):
                p = _parse_diff_line(lines[i])
                if p is None or p["prefix"] != "-":
                    break
                removed_lines.append(p)
                i += 1

            # Collect consecutive added lines
            added_lines: list = []
            while i < len(lines):
                p = _parse_diff_line(lines[i])
                if p is None or p["prefix"] != "+":
                    break
                added_lines.append(p)
                i += 1

            # Only do intra-line diffing when there's exactly one removed and
            # one added line (a single line modification). Otherwise, show
            # lines as-is.
            if len(removed_lines) == 1 and len(added_lines) == 1:
                removed = removed_lines[0]
                added = added_lines[0]

                intra = _render_intra_line_diff(_replace_tabs(removed["content"]), _replace_tabs(added["content"]))

                result.append(theme.fg("toolDiffRemoved", f"-{removed['lineNum']} {intra['removedLine']}"))
                result.append(theme.fg("toolDiffAdded", f"+{added['lineNum']} {intra['addedLine']}"))
            else:
                # Show all removed lines first, then all added lines
                for removed in removed_lines:
                    result.append(
                        theme.fg("toolDiffRemoved", f"-{removed['lineNum']} {_replace_tabs(removed['content'])}")
                    )
                for added in added_lines:
                    result.append(theme.fg("toolDiffAdded", f"+{added['lineNum']} {_replace_tabs(added['content'])}"))
        elif parsed["prefix"] == "+":
            # Standalone added line
            result.append(theme.fg("toolDiffAdded", f"+{parsed['lineNum']} {_replace_tabs(parsed['content'])}"))
            i += 1
        else:
            # Context line
            result.append(theme.fg("toolDiffContext", f" {parsed['lineNum']} {_replace_tabs(parsed['content'])}"))
            i += 1

    return "\n".join(result)
