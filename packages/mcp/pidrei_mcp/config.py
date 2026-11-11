"""Mirror of pi mcp src/config.ts.

The `mcpServers` configuration shape that MCP clients share, so
configurations can be copied between them: stdio servers (`command`,
`args`, `env`, `cwd`), Streamable HTTP servers (`url`, `headers`, `oauth`),
and `enabled`, `description` and `timeout` for both::

    {
      "mcpServers": {
        "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]},
        "docs": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${DOCS_TOKEN}"}}
      }
    }

Applications add fields of their own: `validate_mcp_server_config` keeps
fields it does not know for the caller to check afterwards. Entries keep the
JSON shape (camelCase keys); a field pi checks for `undefined` is checked
for presence here.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

from .url import can_parse, parse_url


class _McpServerConfigBase(TypedDict, total=False):
    """Fields every server entry may have."""

    # What the server offers, in a sentence.
    description: str
    # Set to false to keep the entry without connecting. Default: true.
    enabled: bool
    # Per-request timeout in seconds.
    timeout: float


class McpStdioServerConfig(_McpServerConfigBase):
    """A server started as a local program that speaks MCP over stdin and stdout."""

    type: NotRequired[Literal["stdio"]]
    command: str
    args: NotRequired[list[str]]
    # Added to the environment. The application decides whether values may reference variables (`${NAME}`).
    env: NotRequired[dict[str, str]]
    # The application decides what relative paths resolve against.
    cwd: NotRequired[str]


class McpOAuthConfig(TypedDict, total=False):
    """OAuth client settings for servers that do not support dynamic client registration."""

    # Pre-registered client id. Without it, the application registers a client with the authorization server.
    clientId: str
    # As configured; the application decides whether it may reference variables.
    clientSecret: str
    # Port of the loopback callback server, for clients registered with a
    # fixed redirect URI. Without `callbackUrl`, the redirect URI is
    # `http://127.0.0.1:<port>/callback`.
    callbackPort: int
    # Redirect URI registered for `clientId`, for example
    # `http://localhost:8080/oauth/callback`. It must be an `http` URI on
    # `localhost`, `127.0.0.1`, or `[::1]`. Without a port, the callback
    # server listens on `callbackPort` or a free port, which is added to the
    # URI (RFC 8252).
    callbackUrl: str
    # Scopes to request, separated by spaces. Default: the scopes the server advertises.
    scope: str
    # `client_name` sent with dynamic client registration. Default: the application's name.
    clientName: str
    # How the application identifies itself without `clientId`. `auto`
    # (default): its Client ID Metadata Document when the authorization
    # server supports them for public clients, otherwise dynamic client
    # registration. `cimd`: always the document, and sign-in fails without
    # support. `dcr`: always dynamic client registration.
    clientRegistration: Literal["auto", "dcr", "cimd"]
    # Authorization server metadata document (RFC 8414 or OpenID Connect
    # discovery) to use instead of discovery through the server, for servers
    # that advertise a wrong authorization server or none. The document is
    # trusted as configured. Must use https, except on loopback hosts.
    authServerMetadataUrl: str


class McpHttpServerConfig(_McpServerConfigBase):
    """A server reached over Streamable HTTP."""

    type: NotRequired[Literal["http"]]
    url: str
    # Sent with every request. The application decides whether values may reference variables.
    headers: NotRequired[dict[str, str]]
    oauth: NotRequired[McpOAuthConfig]


# One entry of the `mcpServers` shape: a stdio or a Streamable HTTP server.
type McpServerConfig = McpStdioServerConfig | McpHttpServerConfig


class _StdioResult(TypedDict):
    kind: Literal["stdio"]
    config: dict[str, Any]


class _HttpResult(TypedDict):
    kind: Literal["http"]
    config: dict[str, Any]


class _InvalidResult(TypedDict):
    kind: Literal["invalid"]
    error: str


# A validated server entry, tagged with its transport, or why it is invalid.
# The config is the entry as given: fields the validator does not know are
# kept, unchecked, for the caller to check.
type McpServerConfigResult = _StdioResult | _HttpResult | _InvalidResult


@dataclass(frozen=True, slots=True)
class McpServerConfigOptions:
    """What the application adds to the validation of server entries."""

    # Checks the `oauth.callbackUrl` (None: none configured) of an entry with
    # `oauth.clientRegistration: "cimd"`: the application's Client ID Metadata
    # Documents must list it. Returns an error message. Without this check,
    # `"cimd"` is refused.
    check_metadata_document_callback: Callable[[str | None], str | None] | None = None


_SERVER_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_mcp_server_config(
    name: str, raw: Any, options: McpServerConfigOptions | None = None
) -> McpServerConfigResult:
    """Validate one entry of the `mcpServers` shape (configuration files are a
    trust boundary). Server names are letters, digits, `_` and `-`, which
    tool names and namespaces are built from."""
    options = options or McpServerConfigOptions()

    def invalid(error: str) -> _InvalidResult:
        return {"kind": "invalid", "error": error}

    # `fullmatch`: Python's `$` also matches before a trailing newline, JavaScript's does not.
    if not _SERVER_NAME.fullmatch(name):
        return invalid(f'invalid server name "{name}" (use letters, digits, "_" and "-")')
    if not isinstance(raw, dict):
        return invalid(f'server "{name}" must be an object')
    fields_error = _validate_base_fields(raw)
    if fields_error:
        return invalid(f'server "{name}": {fields_error}')
    server_type, url, command = raw.get("type"), raw.get("url"), raw.get("command")
    # pi's `type === undefined`: the key is absent. A `"type": null` names no transport.
    no_type = "type" not in raw
    if server_type == "sse":
        return invalid(f'server "{name}": legacy SSE transport is not supported; use the streamable HTTP URL')
    if isinstance(url, str) and (no_type or server_type in ("http", "streamable-http")):
        error = _validate_http_fields(raw, url, options)
        if error:
            return invalid(f'server "{name}": {error}')
        return {"kind": "http", "config": raw}
    if isinstance(command, str) and (no_type or server_type == "stdio"):
        error = _validate_stdio_fields(raw)
        if error:
            return invalid(f'server "{name}": {error}')
        return {"kind": "stdio", "config": raw}
    return invalid(f'server "{name}" needs either "command" (stdio) or "url" (streamable HTTP)')


def _validate_base_fields(entry: dict[str, Any]) -> str | None:
    if "enabled" in entry and not isinstance(entry["enabled"], bool):
        return "enabled must be a boolean"
    if "description" in entry and not isinstance(entry["description"], str):
        return "description must be a string"
    if "timeout" in entry:
        timeout = entry["timeout"]
        if isinstance(timeout, bool) or not isinstance(timeout, int | float) or not timeout > 0:
            return "timeout must be a positive number of seconds"
    return None


def _validate_http_fields(entry: dict[str, Any], url: str, options: McpServerConfigOptions) -> str | None:
    if not can_parse(url) or parse_url(url).protocol not in ("http:", "https:"):
        return "url must be an http or https URL"
    if "headers" in entry and not _is_string_record(entry["headers"]):
        return "headers must map names to strings"
    # pi skips only an absent `oauth`; `"oauth": null` is not an object.
    return _validate_oauth(entry["oauth"], options) if "oauth" in entry else None


def _validate_stdio_fields(entry: dict[str, Any]) -> str | None:
    if "args" in entry and not (isinstance(entry["args"], list) and all(isinstance(arg, str) for arg in entry["args"])):
        return "args must be an array of strings"
    if "env" in entry and not _is_string_record(entry["env"]):
        return "env must map names to strings"
    if "cwd" in entry and not isinstance(entry["cwd"], str):
        return "cwd must be a string"
    return None


def _is_string_record(value: Any) -> bool:
    return isinstance(value, dict) and all(isinstance(entry, str) for entry in value.values())


def _is_port(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 65535


_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]")


def _validate_oauth(value: Any, options: McpServerConfigOptions) -> str | None:
    if not isinstance(value, dict):
        return "oauth must be an object"
    if "clientId" in value and not isinstance(value["clientId"], str):
        return "oauth.clientId must be a string"
    if "clientSecret" in value and not isinstance(value["clientSecret"], str):
        return "oauth.clientSecret must be a string"
    port = value.get("callbackPort")
    if "callbackPort" in value and not _is_port(port):
        return "oauth.callbackPort must be a port number"
    if "callbackUrl" in value:
        callback_url = value["callbackUrl"]
        if not isinstance(callback_url, str) or not _is_loopback_redirect_uri(callback_url):
            return "oauth.callbackUrl must be an http URI on localhost, 127.0.0.1, or [::1] without query or fragment"
        # `new URL(...).port`: empty when the URL names none, or names its scheme's default.
        url_port = parse_url(callback_url).port
        if url_port and "callbackPort" in value and int(url_port) != port:
            return "oauth.callbackUrl and oauth.callbackPort name different ports"
    if "scope" in value and not isinstance(value["scope"], str):
        return "oauth.scope must be a string"
    if "clientName" in value and (not isinstance(value["clientName"], str) or not value["clientName"].strip()):
        return "oauth.clientName must be a non-empty string"
    if "clientRegistration" in value and value["clientRegistration"] not in ("auto", "dcr"):
        if value["clientRegistration"] != "cimd":
            return 'oauth.clientRegistration must be "auto", "dcr", or "cimd"'
        if "clientId" in value or "clientName" in value:
            return 'oauth.clientRegistration "cimd" cannot be combined with oauth.clientId or oauth.clientName'
        check = options.check_metadata_document_callback
        if check is None:
            return 'oauth.clientRegistration "cimd" is not supported: there are no Client ID Metadata Documents'
        callback_url = value.get("callbackUrl")
        callback_error = check(callback_url if isinstance(callback_url, str) else None)
        if callback_error:
            return callback_error
    if "authServerMetadataUrl" in value:
        metadata_url = value["authServerMetadataUrl"]
        url = parse_url(metadata_url) if isinstance(metadata_url, str) and can_parse(metadata_url) else None
        if url is None or not (
            url.protocol == "https:" or (url.protocol == "http:" and url.hostname in _LOOPBACK_HOSTS)
        ):
            return "oauth.authServerMetadataUrl must be an https URL, or http on localhost, 127.0.0.1, or [::1]"
    return None


def _is_loopback_redirect_uri(value: str) -> bool:
    """Whether a redirect URI can be served by a loopback callback server."""
    if not can_parse(value):
        return False
    url = parse_url(value)
    return url.protocol == "http:" and url.hostname in _LOOPBACK_HOSTS and url.search == "" and not url.fragment


def is_mcp_server_override(entry: dict[str, Any]) -> bool:
    """Whether an entry overrides a server defined elsewhere instead of
    defining one: it has no `command`, `url` or `type`. Applications let a
    less trusted file (a project's) override a few fields, such as `enabled`,
    of a server a trusted file defines; which fields is theirs to decide."""
    return "command" not in entry and "url" not in entry and "type" not in entry
