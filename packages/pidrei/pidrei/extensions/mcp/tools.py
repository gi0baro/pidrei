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

Results stay wire dicts (`pidrei_mcp`'s `TypedDict`s); only the model-facing
content becomes `TextContent`/`ImageContent`.
"""

import base64
import hashlib
import re
import urllib.parse
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, NotRequired, Protocol, TypedDict

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import ImageContent, TextContent
from pidrei_ai.utils.tasks import gather
from pidrei_mcp import CallToolResult, ContentBlock, LlmContent, Tool, to_llm_content
from pidrei_tui import Container, Spacer, Text
from pidrei_utils.cancel import CancelToken

from ...core.extensions.types import ToolAnnotations, ToolDefinition, ToolExposure, ToolNamespace, ToolRenderers
from ...core.mcp_servers import READ_MCP_RESOURCE_TOOL
from ...core.tools.render_utils import format_tool_call_with_args, get_text_output, replace_tabs
from ...core.tools.truncate import format_size, truncate_middle
from ...modes.interactive.components.keybinding_hints import key_hint
from ...modes.interactive.components.visual_truncate import VisualLinePreview
from ...utils.output_files import write_output_file
from .config import McpExposure


def to_tool_exposure(exposure: McpExposure) -> ToolExposure:
    """Tool exposure of an MCP exposure. `codemode` and `deferred` both leave
    tools out of the codemode description; they differ only in which tool the
    MCP extension activates to reach them."""
    return "deferred" if exposure == "codemode" else exposure


# Provider tool names are limited to 64 characters of `[A-Za-z0-9_-]`.
_MAX_TOOL_NAME_LENGTH = 64
# Model-facing text of an MCP result beyond this is cut in the middle.
MCP_OUTPUT_MAX_BYTES = 20 * 1024
# Visual (wrapped) result lines shown before the output is expanded.
_OUTPUT_PREVIEW_LINES = 5

type ModelContent = TextContent | ImageContent


class McpToolDetails(TypedDict):
    server: str
    tool: str
    # Temp file with the full text output, when the model-facing text was truncated.
    fullOutputPath: NotRequired[str]


# Saves the full text of a truncated result, or a binary resource, and returns
# the file path. `extension` includes the dot, for example `.txt`.
type McpOutputSaver = Callable[[str | bytes, str], Awaitable[str]]


def save_to_temp_file(data: str | bytes, extension: str) -> Awaitable[str]:
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


def create_mcp_tool_name(server: str, tool: str, is_taken: Callable[[str], bool] = lambda _name: False) -> str:
    """`mcp__<server>__<tool>`, sanitized and shortened with a hash suffix
    when too long. Like Codex, everything but `[A-Za-z0-9_]` becomes `_`, so
    the name is also the identifier codemode scripts call it by. `is_taken`
    reports names used by a different MCP tool: sanitizing can map two tools
    to one name (`a-b` and `a_b`), which then get the hash suffix."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", f"mcp__{server}__{tool}")
    if len(name) <= _MAX_TOOL_NAME_LENGTH and not is_taken(name):
        return name
    digest = hashlib.sha256(f"{server}\0{tool}".encode()).hexdigest()[:8]
    return f"{name[: _MAX_TOOL_NAME_LENGTH - len(digest) - 1]}_{digest}"


def _text_of(content: Sequence[ModelContent]) -> str:
    return "\n".join(block.text for block in content if isinstance(block, TextContent))


def create_mcp_result_schema(structured_content_schema: dict[str, Any] | None) -> dict[str, Any]:
    """Output schema of every MCP tool: the `CallToolResult` scripts receive,
    with the tool's own output schema as `structuredContent`. Codemode detects
    this shape to render `CallToolResult[T]` declarations."""
    return {
        "type": "object",
        "properties": {
            "content": {"type": "array", "items": {"type": "object"}},
            **({"structuredContent": structured_content_schema} if structured_content_schema else {}),
            "isError": {"type": "boolean"},
            "_meta": {"type": "object"},
        },
        "required": ["content"],
    }


class LimitedContent(NamedTuple):
    content: list[ModelContent]
    full_output_path: str | None = None


async def limit_mcp_content(
    content: list[ModelContent], save_output: McpOutputSaver = save_to_temp_file
) -> LimitedContent:
    """Keep model-facing text within `MCP_OUTPUT_MAX_BYTES`. Longer text
    becomes one text block in Codex's truncation format, followed by the path
    of the file with the full text; images follow it."""
    combined = _text_of(content)
    truncation = truncate_middle(combined, MCP_OUTPUT_MAX_BYTES)
    if not truncation.truncated:
        return LimitedContent(content)
    full_output_path: str | None = None
    try:
        full_output_path = await save_output(combined, ".txt")
        where = f"[Full output: {full_output_path} (read it with offset/limit)]"
    except Exception as error:
        where = f"[Could not save the full output: {error}]"
    tokens = -(-truncation.total_bytes // 4)
    text = (
        f"Warning: truncated output (original token count: {tokens})\n"
        f"Total output lines: {truncation.total_lines}\n\n{truncation.content}\n\n{where}"
    )
    images = [block for block in content if isinstance(block, ImageContent)]
    return LimitedContent([TextContent(text=text), *images], full_output_path)


@dataclass(frozen=True, slots=True)
class ConvertMcpResultOptions:
    # Saves truncated text and binary resources. Default: a temp file.
    save_output: McpOutputSaver | None = None
    # Whether the server's resources can be read with `read_mcp_resource`,
    # which resource links then name.
    readable_resources: bool = False


def _extension_of(uri: str) -> str:
    """File extension for a saved binary resource: the one its URI ends in, else `.bin`."""
    try:
        parts = urllib.parse.urlsplit(uri)
    except ValueError:
        parts = None
    path = parts.path if parts is not None and parts.scheme else uri
    match = re.search(r"\.[A-Za-z0-9]{1,8}$", path)
    return match.group(0) if match else ".bin"


def _is_text_mime_type(mime_type: str | None) -> bool:
    """Blobs of these types are shown as text."""
    if not mime_type:
        return False
    kind = mime_type.split(";", 1)[0].strip().lower()
    return kind.startswith("text/") or kind == "application/json" or kind.endswith(("+json", "+xml"))


def _from_llm_content(blocks: Sequence[LlmContent]) -> list[ModelContent]:
    return [
        TextContent(text=block["text"])
        if block["type"] == "text"
        else ImageContent(data=block["data"], mime_type=block["mimeType"])
        for block in blocks
    ]


async def _block_to_content(server: str, block: ContentBlock, options: ConvertMcpResultOptions) -> list[ModelContent]:
    """Model-facing content of one block of `server`'s result."""
    if block.get("type") == "resource_link":
        size = block.get("size")
        details = [detail for detail in (block.get("mimeType"), None if size is None else format_size(size)) if detail]
        read = f'. Read it with {READ_MCP_RESOURCE_TOOL} (server "{server}")' if options.readable_resources else ""
        description = f": {block['description']}" if block.get("description") else ""
        title = block.get("title")
        label = title if title is not None else block["name"]
        listed = f" ({', '.join(details)})" if details else ""
        return [TextContent(text=f'[Resource {block["uri"]} "{label}"{listed}{description}{read}]')]
    resource = block.get("resource")
    if (
        block.get("type") == "resource"
        and isinstance(resource, dict)
        and "blob" in resource
        and not (resource.get("mimeType") or "").startswith("image/")
    ):
        uri, mime_type = resource["uri"], resource.get("mimeType")
        data = base64.b64decode(resource["blob"])
        if _is_text_mime_type(mime_type):
            return [TextContent(text=data.decode("utf-8", errors="replace"))]
        kind = f"{mime_type or 'unknown type'}, {format_size(len(data))}"
        try:
            path = await (options.save_output or save_to_temp_file)(data, _extension_of(uri))
        except Exception as error:
            return [TextContent(text=f"[Binary resource {uri} ({kind}) could not be saved: {error}]")]
        return [TextContent(text=f"[Binary resource {uri} ({kind}) saved to {path}]")]
    return _from_llm_content(to_llm_content({"content": [block]}))


async def to_model_content(
    server: str, blocks: Sequence[ContentBlock], options: ConvertMcpResultOptions | None = None
) -> list[ModelContent]:
    """Model-facing content of `server`'s content blocks, before the output limit."""
    options = options or ConvertMcpResultOptions()
    converted = await gather(*(_block_to_content(server, block, options) for block in blocks))
    return [item for content in converted for item in content]


async def convert_mcp_result(
    server: str, tool: str, result: CallToolResult, options: ConvertMcpResultOptions | None = None
) -> AgentToolResult[McpToolDetails]:
    """Convert an MCP result. `isError` results become error results that keep the structured result."""
    options = options or ConvertMcpResultOptions()
    # Without content blocks, to_llm_content falls back to the structured content as JSON.
    converted = (
        await to_model_content(server, result["content"], options)
        if result["content"]
        else _from_llm_content(to_llm_content(result))
    )
    if result.get("isError") and _text_of(converted) == "":
        converted.append(TextContent(text=f"MCP tool {server}/{tool} returned an error"))
    content, full_output_path = await limit_mcp_content(converted, options.save_output or save_to_temp_file)
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


def _to_parameters(schema: dict[str, Any]) -> dict[str, Any]:
    """Tool input schemas must be objects. MCP servers may omit `type`, and
    some providers reject object schemas without `properties`."""
    return {
        **schema,
        "type": schema["type"] if schema.get("type") is not None else "object",
        **({"properties": {}} if "properties" not in schema else {}),
    }


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
        parameters=_to_parameters(tool["inputSchema"]),
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
