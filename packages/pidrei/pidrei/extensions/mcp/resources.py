"""Mirror of pi coding-agent src/extensions/mcp/resources.ts: MCP resources,
through the tools Codex and opencode use: `list_mcp_resources`,
`list_mcp_resource_templates`, and `read_mcp_resource`. They take a `server`
argument and cover every connected server with resources, so models trained on
those tools use them unchanged.

Listings are JSON, as in Codex: `{ server?, resources: [{ server, ...resource }], nextCursor? }`.
With a `server`, one page is listed and `cursor` continues it; without, every
page of every server. MCP App resources (`ui://` URIs and `profile=mcp-app`
HTML) are left out, since they are user interfaces for hosts that render them,
and so are icons. Read resources become text and images for the model; binary
resources are saved to temp files. Scripts get the JSON payloads.
"""

import json
import re
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from typing import Any, Literal, Protocol

import tonio.colored as tonio

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent
from pidrei_mcp import ListResourcesResult, ListResourceTemplatesResult, ReadResourceResult, Resource, ResourceTemplate
from pidrei_utils.cancel import CancelToken

from ...core.extensions.types import ToolAnnotations, ToolDefinition
from ...core.mcp_servers import LIST_MCP_RESOURCE_TEMPLATES_TOOL, LIST_MCP_RESOURCES_TOOL, READ_MCP_RESOURCE_TOOL
from .config import McpExposure, locale_order
from .tools import McpToolDetails, limit_mcp_content, to_model_content, to_tool_exposure


__all__ = [
    "LIST_MCP_RESOURCES_TOOL",
    "LIST_MCP_RESOURCE_TEMPLATES_TOOL",
    "READ_MCP_RESOURCE_TOOL",
    "McpResourceServer",
    "create_mcp_resource_tool_definitions",
    "is_mcp_app_resource",
]


class McpResourceServer(Protocol):
    """A connected server that offers resources."""

    @property
    def name(self) -> str: ...

    @property
    def timeout_ms(self) -> float: ...

    async def resources_page(
        self, cursor: str | None, *, cancel: CancelToken | None = None, timeout_ms: float | None = None
    ) -> ListResourcesResult: ...

    async def resource_templates_page(
        self, cursor: str | None, *, cancel: CancelToken | None = None, timeout_ms: float | None = None
    ) -> ListResourceTemplatesResult: ...

    async def all_resources(
        self, *, cancel: CancelToken | None = None, timeout_ms: float | None = None
    ) -> list[Resource]: ...

    async def all_resource_templates(
        self, *, cancel: CancelToken | None = None, timeout_ms: float | None = None
    ) -> list[ResourceTemplate]: ...

    async def read_resource(
        self, uri: str, *, cancel: CancelToken | None = None, timeout_ms: float | None = None
    ) -> ReadResourceResult: ...


_MCP_APP_MIME_TYPE = re.compile(r';\s*profile\s*=\s*"?mcp-app"?', re.IGNORECASE)


def is_mcp_app_resource(item: dict[str, Any]) -> bool:
    """MCP App user interfaces, which only hosts that render them can use."""
    uri = item.get("uri")
    if uri is None:
        uri = item.get("uriTemplate")
    return (uri or "").startswith("ui://") or _MCP_APP_MIME_TYPE.search(item.get("mimeType") or "") is not None


def _listed(server: str, item: dict[str, Any]) -> dict[str, Any]:
    """A listed resource or template without `_meta` and icons, tagged with its server."""
    return {"server": server, **{key: value for key, value in item.items() if key not in ("_meta", "icons")}}


def _string_property(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


_SERVER_FILTER = _string_property("MCP server name. Omit to list every server with resources.")
_CURSOR = _string_property("Opaque cursor from a previous call with the same server; omit for the first page.")
_LIST_PARAMETERS = {
    "type": "object",
    "properties": {"server": _SERVER_FILTER, "cursor": _CURSOR},
    "additionalProperties": False,
}
_READ_PARAMETERS = {
    "type": "object",
    "properties": {
        "server": _string_property(
            "MCP server name exactly as configured. Must match the 'server' field returned by list_mcp_resources."
        ),
        "uri": _string_property("Resource URI to read. Must be one of the URIs returned by list_mcp_resources."),
    },
    "required": ["server", "uri"],
    "additionalProperties": False,
}

_OPTIONAL_STRING = {"type": "string"}
_LISTING_ERRORS = {
    "type": "array",
    "description": "Servers that could not be listed",
    "items": {
        "type": "object",
        "properties": {"server": {"type": "string"}, "error": {"type": "string"}},
        "required": ["server", "error"],
    },
}
_LIST_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "server": _OPTIONAL_STRING,
        "resources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "server": {"type": "string"},
                    "uri": {"type": "string"},
                    "name": {"type": "string"},
                    "title": _OPTIONAL_STRING,
                    "description": _OPTIONAL_STRING,
                    "mimeType": _OPTIONAL_STRING,
                    "size": {"type": "number"},
                },
                "required": ["server", "uri", "name"],
            },
        },
        "nextCursor": _OPTIONAL_STRING,
        "errors": _LISTING_ERRORS,
    },
    "required": ["resources"],
}
_LIST_TEMPLATES_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "server": _OPTIONAL_STRING,
        "resourceTemplates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "server": {"type": "string"},
                    "uriTemplate": {"type": "string", "description": "RFC 6570 URI template"},
                    "name": {"type": "string"},
                    "title": _OPTIONAL_STRING,
                    "description": _OPTIONAL_STRING,
                    "mimeType": _OPTIONAL_STRING,
                },
                "required": ["server", "uriTemplate", "name"],
            },
        },
        "nextCursor": _OPTIONAL_STRING,
        "errors": _LISTING_ERRORS,
    },
    "required": ["resourceTemplates"],
}
_READ_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "server": {"type": "string"},
        "uri": {"type": "string"},
        "contents": {
            "type": "array",
            "items": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "uri": {"type": "string"},
                            "mimeType": _OPTIONAL_STRING,
                            "text": {"type": "string"},
                        },
                        "required": ["uri", "text"],
                    },
                    {
                        "type": "object",
                        "properties": {
                            "uri": {"type": "string"},
                            "mimeType": _OPTIONAL_STRING,
                            "blob": {"type": "string", "description": "base64"},
                        },
                        "required": ["uri", "blob"],
                    },
                ]
            },
        },
    },
    "required": ["server", "uri", "contents"],
}


def _string_argument(params: Any, key: str) -> str | None:
    value = params.get(key) if isinstance(params, dict) else None
    if value is None:
        return None
    if not isinstance(value, str):
        raise Exception(f"{key} must be a string")  # noqa: TRY004 - pi throws a plain Error
    return value.strip() or None


async def _json_result(tool: str, server: str | None, payload: dict[str, Any]) -> AgentToolResult[McpToolDetails]:
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    content, full_output_path = await limit_mcp_content([TextContent(text=text)])
    details: McpToolDetails = {"server": server or "", "tool": tool}
    if full_output_path:
        details["fullOutputPath"] = full_output_path
    return AgentToolResult(content=content, details=details, structured_content=payload)


async def _all_settled(operations: Sequence[Coroutine[Any, Any, Any]]) -> list[tuple[bool, Any]]:
    """pi's `Promise.allSettled`: run the operations concurrently; each
    outcome is `(True, value)` or `(False, error)`, in order."""

    async def settle(operation: Coroutine[Any, Any, Any]) -> tuple[bool, Any]:
        try:
            return True, await operation
        except Exception as error:
            return False, error

    handles = [tonio.spawn(settle(operation)) for operation in operations]
    return [await handle for handle in handles]


type _Page = Callable[[McpResourceServer, str | None, CancelToken | None], Awaitable[tuple[list[Any], str | None]]]
type _All = Callable[[McpResourceServer, CancelToken | None], Awaitable[list[Any]]]


def create_mcp_resource_tool_definitions(
    *, exposure: McpExposure, servers: Callable[[], Sequence[McpResourceServer]]
) -> list[ToolDefinition]:
    """The three resource tools. `servers` returns the servers whose resources they reach, at call time."""
    read_only = ToolAnnotations(read_only_hint=True)

    def find_server(name: str) -> McpResourceServer:
        candidates = servers()
        server = next((candidate for candidate in candidates if candidate.name == name), None)
        if server is not None:
            return server
        available = ", ".join(candidate.name for candidate in candidates)
        raise Exception(
            f'MCP server "{name}" has no resources' + (f". Servers with resources: {available}" if available else "")
        )

    async def list_items(
        params: Any,
        cancel: CancelToken | None,
        key: Literal["resources", "resourceTemplates"],
        page: _Page,
        all_items: _All,
    ) -> dict[str, Any]:
        """One page of one server, or every page of every server."""
        server_name = _string_argument(params, "server")
        cursor = _string_argument(params, "cursor")
        if server_name:
            server = find_server(server_name)
            items, next_cursor = await page(server, cursor, cancel)
            listed: dict[str, Any] = {
                "server": server.name,
                key: [_listed(server.name, item) for item in items if not is_mcp_app_resource(item)],
            }
            if next_cursor is not None:
                listed["nextCursor"] = next_cursor
            return listed
        if cursor:
            raise Exception("cursor can only be used when a server is specified")
        reached = sorted(servers(), key=lambda server: locale_order(server.name))
        results = await _all_settled([all_items(server, cancel) for server in reached])
        collected: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for server, (ok, value) in zip(reached, results, strict=True):
            if ok:
                collected.extend(_listed(server.name, item) for item in value if not is_mcp_app_resource(item))
            else:
                errors.append({"server": server.name, "error": str(value)})
        return {key: collected, **({"errors": errors} if errors else {})}

    async def resources_page(server: McpResourceServer, cursor: str | None, cancel: CancelToken | None):
        result = await server.resources_page(cursor, cancel=cancel, timeout_ms=server.timeout_ms)
        return result["resources"], result.get("nextCursor")

    async def templates_page(server: McpResourceServer, cursor: str | None, cancel: CancelToken | None):
        result = await server.resource_templates_page(cursor, cancel=cancel, timeout_ms=server.timeout_ms)
        return result["resourceTemplates"], result.get("nextCursor")

    async def list_resources(_tool_call_id, params, cancel, *_rest):
        payload = await list_items(
            params,
            cancel,
            "resources",
            resources_page,
            lambda server, cancel: server.all_resources(cancel=cancel, timeout_ms=server.timeout_ms),
        )
        return await _json_result(LIST_MCP_RESOURCES_TOOL, _string_argument(params, "server"), payload)

    async def list_templates(_tool_call_id, params, cancel, *_rest):
        payload = await list_items(
            params,
            cancel,
            "resourceTemplates",
            templates_page,
            lambda server, cancel: server.all_resource_templates(cancel=cancel, timeout_ms=server.timeout_ms),
        )
        return await _json_result(LIST_MCP_RESOURCE_TEMPLATES_TOOL, _string_argument(params, "server"), payload)

    async def read_resource(_tool_call_id, params, cancel, *_rest):
        server_name = _string_argument(params, "server")
        uri = _string_argument(params, "uri")
        if not server_name:
            raise Exception("server must be provided")
        if not uri:
            raise Exception("uri must be provided")
        server = find_server(server_name)
        result = await server.read_resource(uri, cancel=cancel, timeout_ms=server.timeout_ms)
        contents_list = result["contents"]
        # Several contents (for example a directory) are labeled with their URIs.
        blocks: list[Any] = []
        for contents in contents_list:
            if len(contents_list) > 1:
                blocks.append({"type": "text", "text": f"{contents['uri']}:"})
            blocks.append({"type": "resource", "resource": contents})
        converted = await to_model_content(server.name, blocks)
        content, full_output_path = await limit_mcp_content(
            converted if converted else [TextContent(text=f"Resource {uri} is empty.")]
        )
        details: McpToolDetails = {"server": server.name, "tool": READ_MCP_RESOURCE_TOOL}
        if full_output_path:
            details["fullOutputPath"] = full_output_path
        contents = [{key: value for key, value in item.items() if key != "_meta"} for item in contents_list]
        return AgentToolResult(
            content=content,
            details=details,
            structured_content={"server": server.name, "uri": uri, "contents": contents},
        )

    tool_exposure = to_tool_exposure(exposure)
    return [
        ToolDefinition(
            name=LIST_MCP_RESOURCES_TOOL,
            label=LIST_MCP_RESOURCES_TOOL,
            description=(
                "Lists resources provided by MCP servers. Resources allow servers to share data that provides "
                "context to language models, such as files, database schemas, or application-specific information. "
                "Prefer resources over web search when possible."
            ),
            parameters=_LIST_PARAMETERS,
            output_schema=_LIST_OUTPUT_SCHEMA,
            exposure=tool_exposure,
            annotations=read_only,
            execute=list_resources,
        ),
        ToolDefinition(
            name=LIST_MCP_RESOURCE_TEMPLATES_TOOL,
            label=LIST_MCP_RESOURCE_TEMPLATES_TOOL,
            description=(
                "Lists resource templates provided by MCP servers. Parameterized resource templates allow servers to "
                "share data that takes parameters and provides context to language models, such as files, database "
                "schemas, or application-specific information. Prefer resource templates over web search when possible."
            ),
            parameters=_LIST_PARAMETERS,
            output_schema=_LIST_TEMPLATES_OUTPUT_SCHEMA,
            exposure=tool_exposure,
            annotations=read_only,
            execute=list_templates,
        ),
        ToolDefinition(
            name=READ_MCP_RESOURCE_TOOL,
            label=READ_MCP_RESOURCE_TOOL,
            description="Read a specific resource from an MCP server given the server name and resource URI.",
            parameters=_READ_PARAMETERS,
            output_schema=_READ_OUTPUT_SCHEMA,
            exposure=tool_exposure,
            annotations=read_only,
            execute=read_resource,
        ),
    ]
