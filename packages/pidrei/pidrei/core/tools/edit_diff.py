"""Mirror of pi coding-agent src/core/tools/edit-diff.ts.

The diff/apply engine is the Phase 2 port in pidrei-agent (byte-identical in
pi between the two packages); this module re-exports it and adds the
coding-agent-only preview helpers (compute_edits_diff / compute_edit_diff).
"""

import errno
from dataclasses import dataclass

from tonio.colored import fs

from pidrei_agent.harness.tools.edit_diff import (
    Edit,
    apply_edits_to_normalized_content,
    detect_line_ending,
    generate_diff_string,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)

from .path_utils import resolve_to_cwd


@dataclass(slots=True)
class EditDiffResult:
    diff: str
    first_changed_line: int | None


@dataclass(slots=True)
class EditDiffError:
    error: str


def _error_code_message(error: Exception) -> str:
    code = _errno_code(error)
    return f"Error code: {code}" if code else str(error)


def _errno_code(error: Exception) -> str | None:
    if isinstance(error, OSError) and error.errno is not None:
        return errno.errorcode.get(error.errno)
    return None


async def compute_edits_diff(path: str, edits: list[Edit], cwd: str) -> EditDiffResult | EditDiffError:
    """Compute the diff for one or more edit operations without applying them.
    Used for preview rendering before the tool executes."""
    absolute_path = resolve_to_cwd(path, cwd)

    try:
        # Check if file exists and is readable
        try:
            raw_content = await fs.Path(absolute_path).read_text(encoding="utf-8", newline="")
        except OSError as error:
            return EditDiffError(error=f"Could not edit file: {path}. {_error_code_message(error)}.")

        # Strip BOM before matching (LLM won't include invisible BOM in oldText)
        _bom, content = strip_bom(raw_content)
        normalized_content = normalize_to_lf(content)
        applied = apply_edits_to_normalized_content(normalized_content, edits, path)

        # Generate the diff
        diff, first_changed_line = generate_diff_string(applied.base_content, applied.new_content)
        return EditDiffResult(diff=diff, first_changed_line=first_changed_line)
    except Exception as error:
        return EditDiffError(error=str(error))


async def compute_edit_diff(path: str, old_text: str, new_text: str, cwd: str) -> EditDiffResult | EditDiffError:
    """Compute the diff for a single edit operation without applying it."""
    return await compute_edits_diff(path, [Edit(old_text=old_text, new_text=new_text)], cwd)


__all__ = [
    "Edit",
    "EditDiffError",
    "EditDiffResult",
    "apply_edits_to_normalized_content",
    "compute_edit_diff",
    "compute_edits_diff",
    "detect_line_ending",
    "generate_diff_string",
    "generate_unified_patch",
    "normalize_to_lf",
    "restore_line_endings",
    "strip_bom",
]
