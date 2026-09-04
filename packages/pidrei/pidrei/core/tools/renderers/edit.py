"""Mirror of pi coding-agent src/core/tools/renderers/edit.ts.

Presentation for the edit tool.

Renderers live apart from the implementation so a process that only displays
tool output does not load the execution path or its parameter schema.
`edit.py` spreads these into its definition, so the tool's public shape is
unchanged.
"""

import json

import tonio.colored as tonio

from pidrei_tui import Box, Container, Spacer, Text

from ....modes.interactive.components.diff import render_diff
from ..edit_diff import Edit, EditDiffError, EditDiffResult, compute_edits_diff
from ..render_utils import render_tool_path, str_or_none
from .types import ToolRenderers


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


def _render_call(args, theme, context):
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


def _render_result(result, _options, theme, context):
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


edit_renderers = ToolRenderers(render_call=_render_call, render_result=_render_result)
