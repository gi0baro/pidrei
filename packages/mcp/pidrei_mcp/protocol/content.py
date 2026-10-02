"""Mirror of pi mcp src/protocol/content.ts."""

import json
from typing import Any, Literal, NotRequired, TypedDict


class ContentAnnotations(TypedDict, total=False):
    audience: list[Literal["user", "assistant"]]
    priority: float
    lastModified: str


class TextContent(TypedDict):
    type: Literal["text"]
    text: str
    annotations: NotRequired[ContentAnnotations]
    _meta: NotRequired[dict[str, Any]]


class ImageContent(TypedDict):
    type: Literal["image"]
    data: str
    mimeType: str
    annotations: NotRequired[ContentAnnotations]
    _meta: NotRequired[dict[str, Any]]


class AudioContent(TypedDict):
    type: Literal["audio"]
    data: str
    mimeType: str
    annotations: NotRequired[ContentAnnotations]
    _meta: NotRequired[dict[str, Any]]


class ResourceLinkContent(TypedDict):
    type: Literal["resource_link"]
    uri: str
    name: str
    title: NotRequired[str]
    description: NotRequired[str]
    mimeType: NotRequired[str]
    size: NotRequired[int]
    annotations: NotRequired[ContentAnnotations]
    _meta: NotRequired[dict[str, Any]]


class TextResourceContents(TypedDict):
    uri: str
    mimeType: NotRequired[str]
    text: str
    _meta: NotRequired[dict[str, Any]]


class BlobResourceContents(TypedDict):
    uri: str
    mimeType: NotRequired[str]
    blob: str
    _meta: NotRequired[dict[str, Any]]


class EmbeddedResourceContent(TypedDict):
    type: Literal["resource"]
    resource: TextResourceContents | BlobResourceContents
    annotations: NotRequired[ContentAnnotations]
    _meta: NotRequired[dict[str, Any]]


type ContentBlock = TextContent | ImageContent | AudioContent | ResourceLinkContent | EmbeddedResourceContent


class CallToolResult(TypedDict):
    content: list[ContentBlock]
    structuredContent: NotRequired[dict[str, Any]]
    isError: NotRequired[bool]
    _meta: NotRequired[dict[str, Any]]


class LlmTextContent(TypedDict):
    type: Literal["text"]
    text: str


class LlmImageContent(TypedDict):
    type: Literal["image"]
    data: str
    mimeType: str


# Tool result content in the shape LLM APIs accept: text and base64 images.
# Matches the `TextContent` and `ImageContent` shapes of pi-ai.
type LlmContent = LlmTextContent | LlmImageContent


def _block_to_llm_content(block: dict[str, Any]) -> LlmContent:
    match block.get("type"):
        case "text":
            return {"type": "text", "text": block["text"]}
        case "image":
            return {"type": "image", "data": block["data"], "mimeType": block["mimeType"]}
        case "audio":
            return {"type": "text", "text": f"[audio {block['mimeType']} omitted]"}
        case "resource_link":
            return {"type": "text", "text": f"{block['name']}: {block['uri']}"}
        case "resource":
            resource = block["resource"]
            if "text" in resource:
                return {"type": "text", "text": resource["text"]}
            mime_type = resource.get("mimeType")
            if isinstance(mime_type, str) and mime_type.startswith("image/"):
                return {"type": "image", "data": resource["blob"], "mimeType": mime_type}
            return {
                "type": "text",
                "text": f"[binary resource {resource['uri']} ({mime_type if mime_type is not None else 'unknown type'}) omitted]",
            }
        case other:
            return {"type": "text", "text": f"[unsupported MCP content {other}]"}


def to_llm_content(result: dict[str, Any]) -> list[LlmContent]:
    """Convert a tool result to text and image content for a model. Text and
    images pass through, embedded text resources become text, embedded image
    resources become images, and other blocks (audio, resource links, binary
    resources) become a short text placeholder. A result without content
    blocks but with `structuredContent` becomes its JSON, since servers
    should, but do not always, mirror structured results as text."""
    content = [_block_to_llm_content(block) for block in result.get("content") or []]
    if not content and "structuredContent" in result:
        content.append({"type": "text", "text": json.dumps(result["structuredContent"], indent=2, ensure_ascii=False)})
    return content
