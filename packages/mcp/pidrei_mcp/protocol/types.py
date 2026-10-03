"""Mirror of pi mcp src/protocol/types.ts."""

from typing import Any, Literal, NotRequired, TypedDict

from .content import BlobResourceContents, ContentAnnotations, TextResourceContents
from .jsonrpc import JsonRpcId


LATEST_PROTOCOL_VERSION = "2025-11-25"
# Versions the client accepts from a server. Servers that do not support the
# requested version answer with their own latest one, so older versions stay
# accepted for servers built on older SDKs.
SUPPORTED_PROTOCOL_VERSIONS = (LATEST_PROTOCOL_VERSION, "2025-06-18", "2025-03-26", "2024-11-05")
type SupportedProtocolVersion = Literal["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]


class Implementation(TypedDict):
    name: str
    version: str
    title: NotRequired[str]


class Root(TypedDict):
    uri: str
    name: NotRequired[str]


class _ListChanged(TypedDict, total=False):
    listChanged: bool


class _ResourcesCapability(TypedDict, total=False):
    subscribe: bool
    listChanged: bool


class ClientCapabilities(TypedDict, total=False):
    experimental: dict[str, Any]
    roots: _ListChanged
    sampling: dict[str, Any]
    elicitation: dict[str, Any]


class ServerCapabilities(TypedDict, total=False):
    experimental: dict[str, Any]
    logging: dict[str, Any]
    prompts: _ListChanged
    resources: _ResourcesCapability
    tools: _ListChanged
    completions: dict[str, Any]


class InitializeParams(TypedDict):
    protocolVersion: str
    capabilities: ClientCapabilities
    clientInfo: Implementation


class InitializeResult(TypedDict):
    protocolVersion: str
    capabilities: ServerCapabilities
    serverInfo: Implementation
    instructions: NotRequired[str]


class ProgressNotification(TypedDict):
    progressToken: str | int
    progress: float
    total: NotRequired[float]
    message: NotRequired[str]


class CancelledNotification(TypedDict):
    requestId: JsonRpcId
    reason: NotRequired[str]


class ToolAnnotations(TypedDict, total=False):
    title: str
    readOnlyHint: bool
    destructiveHint: bool
    idempotentHint: bool
    openWorldHint: bool


class ToolExecution(TypedDict, total=False):
    taskSupport: Literal["forbidden", "optional", "required"]


class Tool(TypedDict):
    name: str
    title: NotRequired[str]
    description: NotRequired[str]
    inputSchema: dict[str, Any]
    outputSchema: NotRequired[dict[str, Any]]
    annotations: NotRequired[ToolAnnotations]
    execution: NotRequired[ToolExecution]
    _meta: NotRequired[dict[str, Any]]


class ListToolsResult(TypedDict):
    tools: list[Tool]
    nextCursor: NotRequired[str]
    _meta: NotRequired[dict[str, Any]]


class Resource(TypedDict):
    """A resource a server lists in `resources/list`."""

    uri: str
    name: str
    title: NotRequired[str]
    description: NotRequired[str]
    mimeType: NotRequired[str]
    size: NotRequired[int]
    annotations: NotRequired[ContentAnnotations]
    _meta: NotRequired[dict[str, Any]]


class ResourceTemplate(TypedDict):
    """A family of resources, addressed by an RFC 6570 URI template, from
    `resources/templates/list`."""

    uriTemplate: str
    name: str
    title: NotRequired[str]
    description: NotRequired[str]
    mimeType: NotRequired[str]
    annotations: NotRequired[ContentAnnotations]
    _meta: NotRequired[dict[str, Any]]


class ListResourcesResult(TypedDict):
    resources: list[Resource]
    nextCursor: NotRequired[str]
    _meta: NotRequired[dict[str, Any]]


class ListResourceTemplatesResult(TypedDict):
    resourceTemplates: list[ResourceTemplate]
    nextCursor: NotRequired[str]
    _meta: NotRequired[dict[str, Any]]


class ReadResourceResult(TypedDict):
    contents: list[TextResourceContents | BlobResourceContents]
    _meta: NotRequired[dict[str, Any]]
