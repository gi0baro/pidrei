"""Mirror of pi coding-agent src/extensions/mcp/tools.ts: adapts MCP tools to
pidrei tool definitions.

Results map onto pidrei's model-facing content (text and images). Text over
20KB keeps its start and end with the middle cut out, like Codex does, and the
full text is saved to a temp file the model can read. Binary resources other
than images are saved to temp files too, and resource links name the
`read_mcp_resource` tool. Codemode scripts receive the whole `CallToolResult`
without `_meta` (`content` blocks as sent by the server,
`structuredContent`, `isError`), never truncated: it is the tool's
`structured_content`, and every MCP tool declares a `CallToolResult` output
schema. MCP errors (`isError`) are error results for the model, but scripts
still receive the result.

Results stay wire dicts (`pidrei_mcp`'s `TypedDict`s), and so does the
model-facing content up to the output limit (pidrei-mcp's
`limit_mcp_content` works on its `LlmContent`); after it, the content becomes
`TextContent`/`ImageContent` (`from_llm_content`), where pi hands the same
objects on.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, NotRequired, Protocol, TypedDict

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import ImageContent, TextContent
from pidrei_mcp import (
    CallToolResult,
    LlmContent,
    McpModelContentOptions,
    McpOutputSaver,
    Tool,
    create_mcp_result_schema,
    limit_mcp_content,
    normalize_mcp_input_schema,
    to_mcp_model_content,
)
from pidrei_tui import Container, Spacer, Text
from pidrei_utils.cancel import CancelToken

from ...core.extensions.types import ToolAnnotations, ToolDefinition, ToolExposure, ToolNamespace, ToolRenderers
from ...core.mcp_servers import READ_MCP_RESOURCE_TOOL
from ...core.tools.render_utils import format_tool_call_with_args, get_text_output, replace_tabs
from ...modes.interactive.components.keybinding_hints import key_hint
from ...modes.interactive.components.visual_truncate import VisualLinePreview
from ...utils.output_files import write_output_file
from .config import McpExposure


def to_tool_exposure(exposure: McpExposure) -> ToolExposure:
    """Tool exposure of an MCP exposure. `codemode` and `deferred` both leave
    tools out of the codemode description; they differ only in which tool the
    MCP extension activates to reach them."""
    return "deferred" if exposure == "codemode" else exposure


# Visual (wrapped) result lines shown before the output is expanded.
_OUTPUT_PREVIEW_LINES = 5

type ModelContent = TextContent | ImageContent


class McpToolDetails(TypedDict):
    server: str
    tool: str
    # Temp file with the full text output, when the model-facing text was truncated.
    fullOutputPath: NotRequired[str]


def save_to_temp_file(data: str | bytes, extension: str) -> Awaitable[str]:
    """Saves the full text of a truncated result, or a binary resource, to a temp file and returns its path."""
    return write_output_file("pidrei-mcp", extension, data)


class McpToolCaller(Protocol):
    async def call_tool(
        self,
        name: str,
        args: dict[str, Any],
        *,
        cancel: CancelToken | None = None,
        timeout_ms: float | None = None,
        on_progress: Callable[[Any], Awaitable[None]] | None = None,
    ) -> CallToolResult: ...


def _text_of(content: Sequence[LlmContent]) -> str:
    return "\n".join(block["text"] for block in content if block["type"] == "text")


@dataclass(frozen=True, slots=True)
class ConvertMcpResultOptions:
    # Saves truncated text and binary resources. Default: a temp file.
    save_output: McpOutputSaver | None = None
    # Whether the server's resources can be read with `read_mcp_resource`,
    # which resource links then name.
    readable_resources: bool = False


def from_llm_content(blocks: Sequence[LlmContent]) -> list[ModelContent]:
    """pidrei-mcp's content blocks as the model-facing content of a tool result."""
    return [
        TextContent(text=block["text"])
        if block["type"] == "text"
        else ImageContent(data=block["data"], mime_type=block["mimeType"])
        for block in blocks
    ]


async def convert_mcp_result(
    server: str, tool: str, result: CallToolResult, options: ConvertMcpResultOptions | None = None
) -> AgentToolResult[McpToolDetails]:
    """Convert an MCP result. `isError` results become error results that keep the structured result."""
    options = options or ConvertMcpResultOptions()
    save_output = options.save_output or save_to_temp_file
    converted = await to_mcp_model_content(
        server,
        result,
        McpModelContentOptions(
            save_output=save_output,
            read_resource_tool=READ_MCP_RESOURCE_TOOL if options.readable_resources else None,
        ),
    )
    if result.get("isError") and _text_of(converted) == "":
        converted.append({"type": "text", "text": f"MCP tool {server}/{tool} returned an error"})
    limited, full_output_path = await limit_mcp_content(converted, save_output)
    content = from_llm_content(limited)
    script_result = {key: value for key, value in result.items() if key != "_meta"}
    details: McpToolDetails = {"server": server, "tool": tool}
    if full_output_path:
        details["fullOutputPath"] = full_output_path
    return AgentToolResult(
        content=content,
        details=details,
        structured_content=script_result,
        is_error=True if result.get("isError") else None,
    )


_ANNOTATION_HINTS = (
    ("readOnlyHint", "read_only_hint"),
    ("destructiveHint", "destructive_hint"),
    ("idempotentHint", "idempotent_hint"),
    ("openWorldHint", "open_world_hint"),
)


def _to_tool_annotations(tool: Tool) -> ToolAnnotations | None:
    """The boolean hints of an MCP tool's annotations, or None when it has none."""
    annotations = tool.get("annotations") or {}
    hints = {field: annotations[hint] for hint, field in _ANNOTATION_HINTS if isinstance(annotations.get(hint), bool)}
    return ToolAnnotations(**hints) if hints else None


def _details_of(result: Any) -> Any:
    return result.get("details") if isinstance(result, dict) else result.details


def create_mcp_tool_definition(
    *,
    server: str,
    tool: Tool,
    name: str,
    exposure: McpExposure,
    namespace: ToolNamespace,
    timeout_ms: float,
    get_client: Callable[[], Awaitable[McpToolCaller]],
    # Whether `read_mcp_resource` can read the server's resources.
    readable_resources: Callable[[], bool] | None = None,
) -> ToolDefinition:
    annotations_title = (tool.get("annotations") or {}).get("title")
    title = tool.get("title") if tool.get("title") is not None else annotations_title
    annotations = _to_tool_annotations(tool)
    tool_name = tool["name"]
    label = f"{server}/{tool_name}"
    renderers = create_mcp_tool_renderers(label)

    async def execute(_tool_call_id, params, cancel, on_update, *_rest):
        client = await get_client()

        async def on_progress(progress: Any) -> None:
            total = "" if progress.get("total") is None else f"/{_js_number(progress['total'])}"
            message = progress.get("message")
            text = message if message is not None else f"Progress {_js_number(progress['progress'])}{total}"
            if on_update is not None:
                on_update(
                    AgentToolResult(content=[TextContent(text=text)], details={"server": server, "tool": tool_name})
                )

        result = await client.call_tool(
            tool_name,
            params if params is not None else {},
            cancel=cancel,
            timeout_ms=timeout_ms,
            on_progress=on_progress,
        )
        return await convert_mcp_result(
            server,
            tool_name,
            result,
            ConvertMcpResultOptions(readable_resources=readable_resources() if readable_resources else False),
        )

    description = (tool.get("description") or "").strip() or title or f"MCP tool {tool_name} from server {server}"
    return ToolDefinition(
        name=name,
        label=label,
        description=description,
        parameters=normalize_mcp_input_schema(tool["inputSchema"]),
        output_schema=create_mcp_result_schema(tool.get("outputSchema")),
        exposure=to_tool_exposure(exposure),
        namespace=namespace,
        annotations=annotations,
        render_call=renderers.render_call,
        render_result=renderers.render_result,
        execute=execute,
    )


def create_mcp_tool_renderers(label: str) -> ToolRenderers:
    """Renderers of calls to an MCP tool, labeled `server/tool`, also used before the tool is registered."""

    def render_call(args, theme, context):
        last = context.get("lastComponent")
        component = last if isinstance(last, Text) else Text("", 0, 0)
        component.set_text(format_tool_call_with_args(label, args, theme, context["expanded"]))
        return component

    def render_result(result, options, theme, context):
        last = context.get("lastComponent")
        component = last if isinstance(last, Container) else Container()
        component.clear()
        output = get_text_output(result, context["showImages"]).strip()
        if not output:
            return component
        color = "error" if context["isError"] else "toolOutput"
        styled = "\n".join(theme.fg(color, line) for line in replace_tabs(output).split("\n"))
        component.add_child(Spacer(1))
        if options.get("expanded"):
            component.add_child(Text(styled, 0, 0))
        else:
            # Limit wrapped lines, not logical ones: MCP results are often one long JSON line.
            component.add_child(
                VisualLinePreview(
                    text=styled,
                    max_visual_lines=_OUTPUT_PREVIEW_LINES,
                    keep="start",
                    format_hint=lambda hidden: (
                        f"{theme.fg('muted', f'... ({hidden} more lines,')} "
                        f"{key_hint('app.tools.expand', 'to expand')}{theme.fg('muted', ')')}"
                    ),
                )
            )
            details = _details_of(result)
            full_output_path = details.get("fullOutputPath") if isinstance(details, dict) else None
            if full_output_path:
                component.add_child(Text(theme.fg("muted", f"Full output: {full_output_path}"), 0, 0))
        return component

    return ToolRenderers(render_call=render_call, render_result=render_result)


def _js_number(value: float) -> str:
    """A number as JS template strings show it: `3`, not `3.0`."""
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
