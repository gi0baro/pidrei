"""Mirror of pi coding-agent test/edit-diff-window.test.ts.

pi's mulberry32 generator becomes a seeded `random.Random`, and jsdiff's
`applyPatch` a small unified-patch applier that honours the "No newline at
end of file" marker (`_apply_patch`).

Not mirrored:
- "diffs a small change in a 20 MB file with a 96 MB heap": it runs the diff
  in a Node child process with a capped V8 heap, so that a whole-file diff
  runs out of memory. There is no such heap cap to run under here.
- "handles a replacement of many lines": it guards the JS call-stack limit of
  `push(...lines)` with 150,000 lines; `list.extend` has no such limit.
"""

import random
import re

from pidrei.core.tools.edit_diff import (
    TextReplacement,
    generate_diff_string,
    generate_edit_diff,
    generate_unified_patch,
)


_NO_NEWLINE_MARKER = "\\ No newline at end of file\n"


def _apply_replacements(content: str, replacements: list[TextReplacement]) -> str:
    result = ""
    position = 0
    for replacement in replacements:
        result += content[position : replacement.match_index] + replacement.new_text
        position = replacement.match_index + replacement.match_length
    return result + content[position:]


def _split_lines_with_endings(text: str) -> list[str]:
    return re.findall(r"[^\n]*\n|[^\n]+", text)


def _apply_patch(content: str, patch: str) -> str:
    """Apply a unified patch, checking every context and removed line against `content`."""
    old_lines = _split_lines_with_endings(content)
    patch_lines = _split_lines_with_endings(patch)
    output: list[str] = []
    cursor = 0
    index = 0
    while index < len(patch_lines):
        line = patch_lines[index]
        index += 1
        if line.startswith(("--- ", "+++ ")):
            continue
        if line.startswith("@@"):
            header = re.match(r"@@ -(\d+)(?:,(\d+))? ", line)
            old_start, old_length = int(header.group(1)), int(header.group(2) or 1)
            # An empty old range names the line after which the hunk inserts.
            target = old_start if old_length == 0 else old_start - 1
            assert target >= cursor, f"hunk at line {old_start} overlaps the previous one"
            output.extend(old_lines[cursor:target])
            cursor = target
            continue
        text = line[1:]
        if index < len(patch_lines) and patch_lines[index] == _NO_NEWLINE_MARKER:
            text = text.removesuffix("\n")
            index += 1
        if line.startswith("+"):
            output.append(text)
            continue
        assert old_lines[cursor] == text, f"line {cursor + 1}: {old_lines[cursor]!r} != {text!r}"
        cursor += 1
        if line.startswith(" "):
            output.append(text)
    output.extend(old_lines[cursor:])
    return "".join(output)


def _random_case(rng: random.Random, distinct: bool):
    """Random content and sorted, disjoint, non-empty replacements; `distinct` makes every line unique."""

    def rand(maximum: int) -> int:
        return int(rng.random() * maximum)

    line_count = rand(400)
    lines = [f"line {i}" if distinct else ["a", "b", "", "}"][rand(4)] for i in range(line_count)]
    content = "\n".join(lines) + ("\n" if rand(2) == 0 else "")
    replacements: list[TextReplacement] = []
    position = 0
    count = 1 + rand(4)
    for i in range(count):
        if position >= len(content):
            break
        match_index = position + rand(min(len(content) - position, 300))
        if match_index >= len(content):
            break
        match_length = 1 + rand(min(len(content) - match_index, 120))
        new_lines = [f"new {i} {j}" for j in range(rand(4))]
        new_text = "" if rand(3) == 0 else "\n".join(new_lines) + ("\n" if rand(2) == 0 else "")
        replacements.append(TextReplacement(match_index=match_index, match_length=match_length, new_text=new_text))
        position = match_index + match_length
    return content, replacements, _apply_replacements(content, replacements)


def test_matches_the_whole_file_diff_and_patch_when_lines_are_distinct():
    rng = random.Random(1)  # noqa: S311 - a seeded test generator
    for run in range(2000):
        content, replacements, new_content = _random_case(rng, True)
        if content == new_content:
            continue
        windowed = generate_edit_diff("f.txt", content, new_content, replacements)
        whole_diff, whole_first_changed_line = generate_diff_string(content, new_content)
        assert (windowed.diff, windowed.patch, windowed.first_changed_line) == (
            whole_diff,
            generate_unified_patch("f.txt", content, new_content),
            whole_first_changed_line,
        ), f"run {run}"


def test_produces_a_patch_that_applies_when_lines_repeat():
    rng = random.Random(2)  # noqa: S311 - a seeded test generator
    for run in range(2000):
        content, replacements, new_content = _random_case(rng, False)
        if content == new_content:
            continue
        windowed = generate_edit_diff("f.txt", content, new_content, replacements)
        assert _apply_patch(content, windowed.patch) == new_content, f"run {run}"
