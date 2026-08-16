"""Mirror of pi coding-agent src/core/tools/edit.ts."""

import errno
import json
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio
from tonio.colored import fs

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent
from pidrei_tui import Box, Container, Spacer, Text

from ...modes.interactive.components.diff import render_diff
from ..experimental import get_experimental_tool_sampling
from ..extensions.types import ToolDefinition
from .edit_diff import (
    Edit,
    EditDiffError,
    EditDiffResult,
    apply_edits_to_normalized_content,
    compute_edits_diff,
    detect_line_ending,
    generate_diff_string,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from .file_mutation_queue import resolve_mutation_queue_key, with_file_mutation_queue
from .path_utils import resolve_to_cwd
from .render_utils import render_tool_path, str_or_none
from .tool_definition_wrapper import WrappedDefinitionTool, wrap_tool_definition


class EditCallRenderComponent(Box):
    """Box carrying edit preview state (pi's Object.assign'd Box)."""

    def __init__(self) -> None:
        super().__init__(1, 1, lambda text: text)
        self.preview = None  # EditDiffResult | EditDiffError | None
        self.preview_args_key: str | None = None
        self.preview_pending = False
        self.settled_error = False


def _get_edit_call_render_component(state: dict, last_component) -> EditCallRenderComponent:
    if isinstance(last_component, Box):
        state["callComponent"] = last_component
        return last_component
    if state.get("callComponent") is not None:
        return state["callComponent"]
    component = EditCallRenderComponent()
    state["callComponent"] = component
    return component


def _get_renderable_preview_input(args: dict | None) -> dict | None:
    if not args:
        return None

    path = args.get("path") if isinstance(args.get("path"), str) else None
    if path is None:
        path = args.get("file_path") if isinstance(args.get("file_path"), str) else None
    if not path:
        return None

    edits = args.get("edits")
    if (
        isinstance(edits, list)
        and edits
        and all(
            isinstance(e, dict) and isinstance(e.get("oldText"), str) and isinstance(e.get("newText"), str)
            for e in edits
        )
    ):
        return {"path": path, "edits": edits}

    if isinstance(args.get("oldText"), str) and isinstance(args.get("newText"), str):
        return {"path": path, "edits": [{"oldText": args["oldText"], "newText": args["newText"]}]}

    return None


def _preview_is_error(preview) -> bool:
    return isinstance(preview, EditDiffError)


def _set_edit_preview(component: EditCallRenderComponent, preview, args_key: str | None) -> bool:
    current = component.preview
    if current is None:
        changed = True
    elif _preview_is_error(current) and _preview_is_error(preview):
        changed = current.error != preview.error
    elif _preview_is_error(current) != _preview_is_error(preview):
        changed = True
    else:
        changed = current.diff != preview.diff or current.first_changed_line != preview.first_changed_line
    component.preview = preview
    component.preview_args_key = args_key
    component.preview_pending = False
    return changed


def _format_edit_call(args: dict | None, theme, cwd: str) -> str:
    args = args or {}
    path_display = render_tool_path(str_or_none(args.get("file_path", args.get("path"))), theme, cwd)
    return f"{theme.fg('toolTitle', theme.bold('edit'))} {path_display}"


def _get_edit_header_bg(preview, settled_error: bool, theme):
    if preview is not None:
        if _preview_is_error(preview):
            return lambda text: theme.bg("toolErrorBg", text)
        return lambda text: theme.bg("toolSuccessBg", text)
    if settled_error:
        return lambda text: theme.bg("toolErrorBg", text)
    return lambda text: theme.bg("toolPendingBg", text)


def _build_edit_call_component(component: EditCallRenderComponent, args: dict | None, theme, cwd: str):
    component.set_bg_fn(_get_edit_header_bg(component.preview, component.settled_error, theme))
    component.clear()
    component.add_child(Text(_format_edit_call(args, theme, cwd), 0, 0))

    if component.preview is None:
        return component

    if _preview_is_error(component.preview):
        body = theme.fg("error", component.preview.error)
    else:
        body = render_diff(component.preview.diff)
    component.add_child(Spacer(1))
    component.add_child(Text(body, 0, 0))
    return component


def _result_content_text(result) -> str:
    content = result["content"] if isinstance(result, dict) else result.content
    return "\n".join(
        (c.get("text") if isinstance(c, dict) else getattr(c, "text", None)) or ""
        for c in content
        if (c.get("type") if isinstance(c, dict) else getattr(c, "type", None)) == "text"
    )


def _result_details(result):
    return result.get("details") if isinstance(result, dict) else getattr(result, "details", None)


def _details_get(details, key: str, snake_key: str | None = None):
    # Detail records may be dataclasses (live results) or camelCase dicts.
    if details is None:
        return None
    if isinstance(details, dict):
        return details.get(key)
    return getattr(details, snake_key or key, None)


def _format_edit_result(args: dict | None, preview, result, theme, is_error: bool) -> str | None:
    preview_diff = preview.diff if preview is not None and not _preview_is_error(preview) else None
    preview_error = preview.error if preview is not None and _preview_is_error(preview) else None
    if is_error:
        error_text = _result_content_text(result)
        if not error_text or error_text == preview_error:
            return None
        return theme.fg("error", error_text)

    details = _result_details(result)
    result_diff = _details_get(details, "diff")
    if result_diff and result_diff != preview_diff:
        raw_path = str_or_none((args or {}).get("file_path", (args or {}).get("path")))
        return render_diff(result_diff, {"filePath": raw_path})

    return None


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

EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION: dict[str, Any] = {
    "snippet": "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
    "guidelines": (
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
    ),
}


class LocalEditOperations:
    async def read_file(self, absolute_path: str) -> bytes:
        return await fs.Path(absolute_path).read_bytes()

    async def write_file(self, absolute_path: str, content: str) -> None:
        await fs.Path(absolute_path).write_text(content, encoding="utf-8", newline="")

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

        queue_key = await resolve_mutation_queue_key(absolute_path)
        return await with_file_mutation_queue(absolute_path, run, queue_key=queue_key)

    def render_call(args, theme, context):
        component = _get_edit_call_render_component(context["state"], context.get("lastComponent"))
        preview_input = _get_renderable_preview_input(args)
        args_key = (
            json.dumps({"path": preview_input["path"], "edits": preview_input["edits"]}, separators=(",", ":"))
            if preview_input
            else None
        )

        if component.preview_args_key != args_key:
            component.preview = None
            component.preview_args_key = args_key
            component.preview_pending = False
            component.settled_error = False

        if context["argsComplete"] and preview_input and component.preview is None and not component.preview_pending:
            component.preview_pending = True
            request_key = args_key
            edits = [Edit(old_text=e["oldText"], new_text=e["newText"]) for e in preview_input["edits"]]

            async def compute_preview() -> None:
                preview = await compute_edits_diff(preview_input["path"], edits, context["cwd"])
                if component.preview_args_key == request_key:
                    _set_edit_preview(component, preview, request_key)
                    context["invalidate"]()

            tonio.spawn.without_tracking(compute_preview())

        return _build_edit_call_component(component, args, theme, context["cwd"])

    def render_result(result, _options, theme, context):
        call_component = context["state"].get("callComponent")
        preview_input = _get_renderable_preview_input(context["args"])
        args_key = (
            json.dumps({"path": preview_input["path"], "edits": preview_input["edits"]}, separators=(",", ":"))
            if preview_input
            else None
        )
        details = _result_details(result)
        result_diff = _details_get(details, "diff") if not context["isError"] else None
        changed = False
        if call_component is not None:
            if isinstance(result_diff, str):
                changed = (
                    _set_edit_preview(
                        call_component,
                        EditDiffResult(
                            diff=result_diff,
                            first_changed_line=_details_get(details, "firstChangedLine", "first_changed_line"),
                        ),
                        args_key,
                    )
                    or changed
                )
            if call_component.settled_error != context["isError"]:
                call_component.settled_error = context["isError"]
                changed = True
            if changed:
                _build_edit_call_component(call_component, context["args"], theme, context["cwd"])

        output = _format_edit_result(
            context["args"],
            call_component.preview if call_component is not None else None,
            result,
            theme,
            context["isError"],
        )
        component = context["lastComponent"] if isinstance(context.get("lastComponent"), Container) else Container()
        component.clear()
        if not output:
            return component
        component.add_child(Spacer(1))
        component.add_child(Text(output, 1, 0))
        return component

    return ToolDefinition(
        name="edit",
        label="edit",
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
            "non-overlapping region of the original file. If two changes affect the same block or nearby "
            "lines, merge them into one edit instead of emitting overlapping edits. Do not include large "
            "unchanged regions just to connect distant changes."
        ),
        prompt_snippet=EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION["snippet"],
        prompt_guidelines=list(EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION["guidelines"]),
        parameters=EDIT_SCHEMA,
        constrained_sampling=get_experimental_tool_sampling(),
        render_shell="self",
        render_call=render_call,
        render_result=render_result,
        prepare_arguments=prepare_edit_arguments,
        execute=execute,
    )


def create_edit_tool(cwd: str, **options) -> WrappedDefinitionTool:
    return wrap_tool_definition(create_edit_tool_definition(cwd, **options))
