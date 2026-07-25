"""Mirror of pi coding-agent src/core/tools/edit.ts (execute path; renderers Phase 4)."""

import errno
import json
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent

from ..extensions.types import ToolDefinition
from .edit_diff import (
    Edit,
    apply_edits_to_normalized_content,
    detect_line_ending,
    generate_diff_string,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from .file_mutation_queue import with_file_mutation_queue
from .path_utils import resolve_to_cwd
from .tool_definition_wrapper import WrappedDefinitionTool, wrap_tool_definition


@dataclass(slots=True)
class EditToolDetails:
    # Display-oriented diff of the changes made
    diff: str
    # Standard unified patch of the changes made
    patch: str
    # Line number of the first change in the new file (for editor navigation)
    first_changed_line: int | None = None


EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
        "edits": {
            "type": "array",
            "description": (
                "One or more targeted replacements. Each edit is matched against the original file, not "
                "incrementally. Do not include overlapping or nested edits. If two changes touch the same "
                "block or nearby lines, merge them into one edit instead."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "oldText": {
                        "type": "string",
                        "description": (
                            "Exact text for one targeted replacement. It must be unique in the original file "
                            "and must not overlap with any other edits[].oldText in the same call."
                        ),
                    },
                    "newText": {"type": "string", "description": "Replacement text for this targeted edit."},
                },
                "required": ["oldText", "newText"],
            },
        },
    },
    "required": ["path", "edits"],
}


class LocalEditOperations:
    async def read_file(self, absolute_path: str) -> bytes:
        def read() -> bytes:
            with open(absolute_path, "rb") as f:
                return f.read()

        return await tonio.spawn_blocking(read)

    async def write_file(self, absolute_path: str, content: str) -> None:
        def write() -> None:
            with open(absolute_path, "w", encoding="utf-8", newline="") as f:
                f.write(content)

        await tonio.spawn_blocking(write)

    async def access(self, absolute_path: str) -> None:
        """Check the file is readable and writable (raise if not)."""

        def check() -> None:
            with open(absolute_path, "rb"):
                pass
            with open(absolute_path, "r+b"):
                pass

        await tonio.spawn_blocking(check)


def prepare_edit_arguments(input: Any) -> Any:
    if not input or not isinstance(input, dict):
        return input

    args = dict(input)

    # Some models (Opus 4.6, GLM-5.1) send edits as a JSON string instead of an array
    if isinstance(args.get("edits"), str):
        try:
            parsed = json.loads(args["edits"])
            if isinstance(parsed, list):
                args["edits"] = parsed
        except Exception:
            pass

    if not isinstance(args.get("oldText"), str) or not isinstance(args.get("newText"), str):
        return args

    edits = list(args["edits"]) if isinstance(args.get("edits"), list) else []
    edits.append({"oldText": args["oldText"], "newText": args["newText"]})
    rest = {key: value for key, value in args.items() if key not in ("oldText", "newText")}
    return {**rest, "edits": edits}


def _validate_edit_input(params: dict) -> tuple[str, list[Edit]]:
    edits = params.get("edits")
    if not isinstance(edits, list) or len(edits) == 0:
        raise Exception("Edit tool input is invalid. edits must contain at least one replacement.")
    return params["path"], [Edit(old_text=edit["oldText"], new_text=edit["newText"]) for edit in edits]


def _errno_code(error: Exception) -> str | None:
    if isinstance(error, OSError) and error.errno is not None:
        return errno.errorcode.get(error.errno)
    return None


def edit_access_error_message(error: Exception) -> str:
    code = _errno_code(error)
    return f"Error code: {code}" if code else f"{type(error).__name__}: {error}"


def create_edit_tool_definition(cwd: str, *, operations: Any = None) -> ToolDefinition:
    ops = operations if operations is not None else LocalEditOperations()

    async def execute(_tool_call_id, params, cancel=None, _on_update=None, _ctx=None):
        path, edits = _validate_edit_input(params)
        absolute_path = resolve_to_cwd(path, cwd)

        async def run():
            # Do not release the mutation queue while an in-flight filesystem
            # operation may still finish: check cancellation after each await.
            def throw_if_aborted() -> None:
                if cancel is not None and cancel.cancelled:
                    raise Exception("Operation aborted")

            throw_if_aborted()

            # Check if file exists.
            try:
                await ops.access(absolute_path)
            except Exception as error:
                throw_if_aborted()
                raise Exception(f"Could not edit file: {path}. {edit_access_error_message(error)}.")
            throw_if_aborted()

            # Read the file.
            buffer = await ops.read_file(absolute_path)
            raw_content = buffer.decode("utf-8")
            throw_if_aborted()

            # Strip BOM before matching. The model will not include an invisible BOM in oldText.
            bom, content = strip_bom(raw_content)
            original_ending = detect_line_ending(content)
            normalized_content = normalize_to_lf(content)
            applied = apply_edits_to_normalized_content(normalized_content, edits, path)
            throw_if_aborted()

            final_content = bom + restore_line_endings(applied.new_content, original_ending)
            await ops.write_file(absolute_path, final_content)
            throw_if_aborted()

            diff, first_changed_line = generate_diff_string(applied.base_content, applied.new_content)
            patch = generate_unified_patch(path, applied.base_content, applied.new_content)
            return AgentToolResult(
                content=[TextContent(text=f"Successfully replaced {len(edits)} block(s) in {path}.")],
                details=EditToolDetails(diff=diff, patch=patch, first_changed_line=first_changed_line),
            )

        return await with_file_mutation_queue(absolute_path, run)

    return ToolDefinition(
        name="edit",
        label="edit",
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
            "non-overlapping region of the original file. If two changes affect the same block or nearby "
            "lines, merge them into one edit instead of emitting overlapping edits. Do not include large "
            "unchanged regions just to connect distant changes."
        ),
        prompt_snippet=(
            "Make precise file edits with exact text replacement, including multiple disjoint edits in one call"
        ),
        prompt_guidelines=[
            "Use edit for precise changes (edits[].oldText must match exactly)",
            (
                "When changing multiple separate locations in one file, use one edit call with multiple entries "
                "in edits[] instead of multiple edit calls"
            ),
            (
                "Each edits[].oldText is matched against the original file, not after earlier edits are applied. "
                "Do not emit overlapping or nested edits. Merge nearby changes into one edit."
            ),
            (
                "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with "
                "large unchanged regions."
            ),
        ],
        parameters=EDIT_SCHEMA,
        render_shell="self",
        prepare_arguments=prepare_edit_arguments,
        execute=execute,
    )


def create_edit_tool(cwd: str, **options) -> WrappedDefinitionTool:
    return wrap_tool_definition(create_edit_tool_definition(cwd, **options))
