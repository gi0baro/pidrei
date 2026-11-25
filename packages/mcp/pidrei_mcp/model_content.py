"""Mirror of pi mcp src/model-content.ts: the model-facing content of an MCP
result as pidrei's coding agent shows it: text and images, embedded text
resources as text, binary resources saved where the model can read them, and
resource links that name the tool reading them.

pi converts the blocks with `Promise.all`; here they run concurrently as one
`tonio.spawn`, whose `SpawnExceptionGroup` is unwrapped to the failing
block's own error (pi's `Promise.all` rejects with it).
"""

import base64
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio

from .output_limit import McpOutputSaver
from .protocol.content import ContentBlock, LlmContent, to_llm_content


@dataclass(frozen=True, slots=True)
class McpModelContentOptions:
    """How `to_mcp_model_content` handles what the model cannot read inline."""

    # Saves binary resources other than images where the model can read them. Without it they are left out.
    save_output: McpOutputSaver | None = None
    # The tool that reads the server's resources, which resource links then name.
    read_resource_tool: str | None = None


async def to_mcp_model_content(
    server: str, result: dict[str, Any], options: McpModelContentOptions | None = None
) -> list[LlmContent]:
    """Model-facing content of `server`'s result (its `content` and
    `structuredContent`), before the output limit (`limit_mcp_content`). A
    result without content blocks becomes its structured content as JSON, as
    in `to_llm_content`."""
    options = options or McpModelContentOptions()
    blocks = result["content"]
    if not blocks:
        return to_llm_content(result)
    coros = [_block_to_content(server, block, options) for block in blocks]
    try:
        converted = [await tonio.spawn(coros[0])] if len(coros) == 1 else await tonio.spawn(*coros)
    except BaseExceptionGroup as group:
        leaf = group
        while isinstance(leaf, BaseExceptionGroup) and leaf.exceptions:
            leaf = leaf.exceptions[0]
        if leaf is group:
            raise
        raise leaf from group
    return [item for content in converted for item in content]


def _text(text: str) -> LlmContent:
    return {"type": "text", "text": text}


async def _block_to_content(server: str, block: ContentBlock, options: McpModelContentOptions) -> list[LlmContent]:
    """Model-facing content of one block of `server`'s result."""
    if block.get("type") == "resource_link":
        size = block.get("size")
        details = [detail for detail in (block.get("mimeType"), None if size is None else _format_size(size)) if detail]
        read = f'. Read it with {options.read_resource_tool} (server "{server}")' if options.read_resource_tool else ""
        description = f": {block['description']}" if block.get("description") else ""
        title = block.get("title")
        label = title if title is not None else block["name"]
        listed = f" ({', '.join(details)})" if details else ""
        return [_text(f'[Resource {block["uri"]} "{label}"{listed}{description}{read}]')]
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
            return [_text(data.decode("utf-8", errors="replace"))]
        if not options.save_output:
            return to_llm_content({"content": [block]})
        kind = f"{mime_type or 'unknown type'}, {_format_size(len(data))}"
        try:
            path = await options.save_output(data, _extension_of(uri))
        except Exception as error:
            return [_text(f"[Binary resource {uri} ({kind}) could not be saved: {error}]")]
        return [_text(f"[Binary resource {uri} ({kind}) saved to {path}]")]
    return to_llm_content({"content": [block]})


def _format_size(size_bytes: float) -> str:
    """A byte count as pidrei's coding agent shows sizes: `512B`, `2.0KB`, `1.5MB`."""
    if size_bytes < 1024:
        # :g mirrors JS number stringification ("12B", "12.5B").
        return f"{size_bytes:g}B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    return f"{size_bytes / (1024 * 1024):.1f}MB"


def _is_text_mime_type(mime_type: str | None) -> bool:
    """Blobs of these types are shown as text."""
    if not mime_type:
        return False
    kind = mime_type.split(";", 1)[0].strip().lower()
    return kind.startswith("text/") or kind == "application/json" or kind.endswith(("+json", "+xml"))


def _extension_of(uri: str) -> str:
    """File extension for a saved binary resource: the one its URI ends in, else `.bin`."""
    try:
        parts = urllib.parse.urlsplit(uri)
    except ValueError:
        parts = None
    path = parts.path if parts is not None and parts.scheme else uri
    match = re.search(r"\.[A-Za-z0-9]{1,8}$", path)
    return match.group(0) if match else ".bin"
