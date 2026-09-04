"""Edit tool (port of pi `harness/tools/edit.ts`)."""

import json
from dataclasses import dataclass
from typing import Any

from pidrei_ai.types import TextContent
from pidrei_ai.utils.cancel import CancelToken

from ...types import AgentToolResult, AgentToolUpdateCallback
from ..types import AgentHarnessTool, FileError
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
from .path_utils import resolve_tool_path
from .tool_context import ExecutionToolContext


_REPLACE_EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "oldText": {
            "type": "string",
            "description": (
                "Exact text for one targeted replacement. It must be unique in the original file and must "
                "not overlap with any other edits[].oldText in the same call."
            ),
        },
        "newText": {"type": "string", "description": "Replacement text for this targeted edit."},
    },
    "required": ["oldText", "newText"],
}

_EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
        "edits": {
            "type": "array",
            "items": _REPLACE_EDIT_SCHEMA,
            "description": (
                "One or more targeted replacements. Each edit is matched against the original file, not "
                "incrementally. Do not include overlapping or nested edits. If two changes touch the same "
                "block or nearby lines, merge them into one edit instead."
            ),
        },
    },
    "required": ["path", "edits"],
}


@dataclass(slots=True)
class EditToolDetails:
    diff: str
    patch: str
    first_changed_line: int | None = None


def _is_single_edit_input(value: Any) -> bool:
    """pi's `isSingleEditInput`: a bare `{oldText, newText}` sent instead of a one-element array."""
    if not isinstance(value, dict):
        return False
    return isinstance(value.get("oldText"), str) and isinstance(value.get("newText"), str)


def prepare_edit_arguments(input_value: Any) -> Any:
    if not isinstance(input_value, dict):
        return input_value
    args = input_value
    if isinstance(args.get("edits"), str):
        try:
            parsed = json.loads(args["edits"])
            if isinstance(parsed, list):
                args["edits"] = parsed
            elif _is_single_edit_input(parsed):
                args["edits"] = [parsed]
        except TypeError, ValueError:
            pass
    elif _is_single_edit_input(args.get("edits")):
        args["edits"] = [args["edits"]]

    if not isinstance(args.get("oldText"), str) or not isinstance(args.get("newText"), str):
        return args
    edits = list(args["edits"]) if isinstance(args.get("edits"), list) else []
    edits.append({"oldText": args["oldText"], "newText": args["newText"]})
    rest = {key: value for key, value in args.items() if key not in ("oldText", "newText")}
    return {**rest, "edits": edits}


def _validate_edit_input(params: dict[str, Any]) -> tuple[str, list[Edit]]:
    edits = params.get("edits")
    if not isinstance(edits, list) or len(edits) == 0:
        raise Exception("Edit tool input is invalid. edits must contain at least one replacement.")
    return params["path"], [Edit(old_text=entry["oldText"], new_text=entry["newText"]) for entry in edits]


def _edit_access_error(path: str, error: FileError) -> Exception:
    exception = Exception(f"Could not edit file: {path}. Error code: {error.code}.")
    exception.__cause__ = error
    return exception


class EditTool(AgentHarnessTool[ExecutionToolContext, EditToolDetails | None]):
    name = "edit"
    label = "edit"
    description = (
        "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
        "non-overlapping region of the original file. If two changes affect the same block or nearby lines, "
        "merge them into one edit instead of emitting overlapping edits. Do not include large unchanged "
        "regions just to connect distant changes."
    )
    parameters = _EDIT_SCHEMA

    def __init__(self) -> None:
        self.prepare_arguments = prepare_edit_arguments

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        on_update: AgentToolUpdateCallback[EditToolDetails | None] | None,
        tool_context: ExecutionToolContext,
        cancel: CancelToken | None = None,
    ) -> AgentToolResult[EditToolDetails | None]:
        path, edits = _validate_edit_input(params)
        env = tool_context.env
        absolute_path = await resolve_tool_path(env, path, cancel)

        async def mutation() -> AgentToolResult[EditToolDetails | None]:
            if cancel is not None and cancel.cancelled:
                raise Exception("Operation aborted")
            info = await env.file_info(absolute_path, cancel)
            if not info.ok:
                raise _edit_access_error(path, info.error)
            if info.value.kind not in ("file", "symlink"):
                raise Exception(f"Could not edit file: {path}. Path is not a file.")

            read_result = await env.read_text_file(absolute_path, cancel)
            if not read_result.ok:
                raise _edit_access_error(path, read_result.error)
            if cancel is not None and cancel.cancelled:
                raise Exception("Operation aborted")

            bom, content = strip_bom(read_result.value)
            original_ending = detect_line_ending(content)
            normalized_content = normalize_to_lf(content)
            applied = apply_edits_to_normalized_content(normalized_content, edits, path)
            if cancel is not None and cancel.cancelled:
                raise Exception("Operation aborted")

            final_content = bom + restore_line_endings(applied.new_content, original_ending)
            write_result = await env.write_file(absolute_path, final_content, cancel)
            if not write_result.ok:
                raise _edit_access_error(path, write_result.error)
            if cancel is not None and cancel.cancelled:
                raise Exception("Operation aborted")

            diff, first_changed_line = generate_diff_string(applied.base_content, applied.new_content)
            return AgentToolResult(
                content=[TextContent(text=f"Successfully replaced {len(edits)} block(s) in {path}.")],
                details=EditToolDetails(
                    diff=diff,
                    patch=generate_unified_patch(path, applied.base_content, applied.new_content),
                    first_changed_line=first_changed_line,
                ),
            )

        return await with_file_mutation_queue(env, absolute_path, mutation, cancel)


def create_edit_tool() -> EditTool:
    return EditTool()
