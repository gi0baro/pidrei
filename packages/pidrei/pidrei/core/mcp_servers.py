"""Mirror of pi coding-agent src/core/mcp-servers.ts.

MCP server configuration and the servers extensions register with
`pi.register_mcp_server()`.

The core only validates and stores registrations. An MCP extension (one that
handles `mcp_servers_change`) connects them. pidrei's built-in MCP extension
and client port later, with codemode; until then a registration is reported
as unhandled unless an extension handles the event.

Configs keep the JSON shape of an `mcpServers` entry in `mcp.json`
(camelCase keys), like pi's objects.
"""

import copy
import re
import threading
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal


# - "codemode": tools are callable from codemode scripts but neither declared
#   to the model nor listed in the codemode description, which lists only the
#   server's namespace. Scripts find them with `searchTools()`.
#   "codemode-deferred" is accepted as an alias.
# - "deferred": not declared to the model until the `tool_search` tool loads
#   them; the model then calls them directly. Does not need codemode.
# - "direct": tools are declared to the model like any other tool (and callable from codemode).
# - "hidden": tools are registered but unreachable.
type McpExposure = Literal["codemode", "deferred", "direct", "hidden"]

_MCP_EXPOSURES: tuple[str, ...] = ("codemode", "deferred", "direct", "hidden")

# Older exposure names, accepted in configs and replaced by their current name when validated.
_MCP_EXPOSURE_ALIASES: dict[str, McpExposure] = {"codemode-deferred": "codemode"}

# A server config: `{"command", "args"?, "env"?, "cwd"?}` (stdio) or
# `{"url", "headers"?, "oauth"?, "auth"?}` (streamable HTTP), plus the common
# `exposure`, `toolExposure`, `description`, `enabled` and `timeout`.
#
# - `description`: what the server offers, in a sentence. The `mcp_servers`
#   system prompt section lists the server with it, tool search ranks the
#   server's tools by it, and codemode's `describeNamespace()` returns it.
# - `oauth.clientName`: `client_name` sent with dynamic client registration,
#   for servers that only accept known clients. Default: `pidrei`.
# - `oauth.authServerMetadataUrl`: authorization server metadata document
#   (RFC 8414 or OpenID Connect discovery) to use instead of discovery through
#   the server, for servers that advertise a wrong authorization server or
#   none. The document is trusted as configured. Must use https, except on
#   loopback hosts.
# - `auth`: `{"provider": "<provider>"}` sends the token of a provider login
#   (`/login <provider>`) instead of using OAuth. Not allowed in project
#   `mcp.json` files, and requires https except on loopback hosts, since it
#   sends the credential to `url`.
type McpServerConfig = dict[str, Any]

_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]")
_SERVER_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def mcp_namespace(server: str) -> str:
    """Namespace of a server's tools: `mcp__<server>` with `-` replaced by `_`, like the tool names."""
    return f"mcp__{server.replace('-', '_')}"


def _parse_url(value: str) -> urllib.parse.SplitResult | None:
    """pi: `URL.canParse(value) ? new URL(value) : undefined`, for absolute URLs."""
    try:
        url = urllib.parse.urlsplit(value)
        url.port  # noqa: B018 - raises for an invalid port, as URL parsing does
    except ValueError:
        return None
    if not url.scheme or not url.netloc:
        return None
    return url


def _host_of(url: urllib.parse.SplitResult) -> str:
    """The URL's hostname as `new URL(...).hostname` spells it (IPv6 in brackets)."""
    hostname = url.hostname or ""
    return f"[{hostname}]" if ":" in hostname else hostname


def is_loopback_redirect_uri(value: str) -> bool:
    """Whether a redirect URI can be served by pidrei's loopback callback server."""
    url = _parse_url(value)
    if url is None:
        return False
    return url.scheme == "http" and _host_of(url) in _LOOPBACK_HOSTS and not url.query and not url.fragment


def _is_string_record(value: Any) -> bool:
    return isinstance(value, dict) and all(isinstance(entry, str) for entry in value.values())


def _is_port(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 65535


def _validate_oauth(value: Any) -> str | None:
    if value is None:
        return None
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
        if not isinstance(callback_url, str) or not is_loopback_redirect_uri(callback_url):
            return "oauth.callbackUrl must be an http URI on localhost, 127.0.0.1, or [::1] without query or fragment"
        url_port = _parse_url(callback_url).port  # type: ignore[union-attr]
        if url_port is not None and "callbackPort" in value and url_port != port:
            return "oauth.callbackUrl and oauth.callbackPort name different ports"
    if "scope" in value and not isinstance(value["scope"], str):
        return "oauth.scope must be a string"
    if "clientName" in value and (not isinstance(value["clientName"], str) or not value["clientName"].strip()):
        return "oauth.clientName must be a non-empty string"
    if "authServerMetadataUrl" in value:
        metadata_url = value["authServerMetadataUrl"]
        url = _parse_url(metadata_url) if isinstance(metadata_url, str) else None
        if url is None or not (url.scheme == "https" or (url.scheme == "http" and _host_of(url) in _LOOPBACK_HOSTS)):
            return "oauth.authServerMetadataUrl must be an https URL, or http on localhost, 127.0.0.1, or [::1]"
    return None


def _is_exposure(value: Any) -> bool:
    return isinstance(value, str) and value in _MCP_EXPOSURES


def _resolve_exposure_alias(value: Any) -> Any:
    """The exposure an alias stands for; other values are returned unchanged."""
    return _MCP_EXPOSURE_ALIASES.get(value, value) if isinstance(value, str) else value


def _resolve_exposure_aliases(value: dict[str, Any]) -> dict[str, Any]:
    """A copy of the server entry with exposure aliases replaced by their current names."""
    resolved = dict(value)
    if "exposure" in value:
        resolved["exposure"] = _resolve_exposure_alias(value["exposure"])
    tool_exposure = value.get("toolExposure")
    if isinstance(tool_exposure, dict):
        resolved["toolExposure"] = {tool: _resolve_exposure_alias(entry) for tool, entry in tool_exposure.items()}
    return resolved


def _tool_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile("^" + ".*".join(re.escape(part) for part in pattern.split("*")) + "$")


def get_mcp_tool_exposure(config: McpServerConfig, tool_name: str) -> McpExposure:
    """Exposure of one tool of a server: its `toolExposure` entry, else the
    server's `exposure`. An exact name wins over patterns; among patterns the
    first match in the object wins."""
    overrides: dict[str, McpExposure] = config.get("toolExposure") or {}
    exact = overrides.get(tool_name)
    if exact is not None:
        return exact
    for pattern, exposure in overrides.items():
        if "*" in pattern and _tool_pattern(pattern).match(tool_name):
            return exposure
    return config.get("exposure") or "codemode"


def validate_mcp_server_config(name: str, raw: Any) -> McpServerConfig | str:
    """Validate one server entry of the `mcpServers` shape. Returns a copy of
    the config with exposure aliases resolved, or an error message."""
    if not _SERVER_NAME.match(name):
        return f'invalid server name "{name}" (use letters, digits, "_" and "-")'
    if not isinstance(raw, dict):
        return f'server "{name}" must be an object'
    value = _resolve_exposure_aliases(raw)
    server_type = value.get("type")
    exposures = ", ".join(f'"{exposure}"' for exposure in _MCP_EXPOSURES)
    if "exposure" in value and not _is_exposure(value["exposure"]):
        return f'server "{name}": exposure must be one of {exposures}'
    if "toolExposure" in value:
        tool_exposure = value["toolExposure"]
        if not isinstance(tool_exposure, dict):
            return f'server "{name}": toolExposure must map tool names to exposures'
        for tool, exposure in tool_exposure.items():
            if not _is_exposure(exposure):
                return f'server "{name}": toolExposure "{tool}" must be one of {exposures}'
    if "enabled" in value and not isinstance(value["enabled"], bool):
        return f'server "{name}": enabled must be a boolean'
    if "description" in value and not isinstance(value["description"], str):
        return f'server "{name}": description must be a string'
    if "timeout" in value:
        timeout = value["timeout"]
        if isinstance(timeout, bool) or not isinstance(timeout, int | float) or not timeout > 0:
            return f'server "{name}": timeout must be a positive number of seconds'
    if server_type == "sse":
        return f'server "{name}": legacy SSE transport is not supported; use the streamable HTTP URL'

    if isinstance(value.get("url"), str) and server_type in (None, "http", "streamable-http"):
        url = _parse_url(value["url"])
        if url is None or url.scheme not in ("http", "https"):
            return f'server "{name}": url must be an http or https URL'
        if "headers" in value and not _is_string_record(value["headers"]):
            return f'server "{name}": headers must map names to strings'
        oauth_error = _validate_oauth(value.get("oauth"))
        if oauth_error:
            return f'server "{name}": {oauth_error}'
        if "auth" in value:
            auth = value["auth"]
            if not isinstance(auth, dict) or not isinstance(auth.get("provider"), str) or not auth["provider"]:
                return f'server "{name}": auth.provider must be a provider name'
            if url.scheme != "https" and _host_of(url) not in _LOOPBACK_HOSTS:
                return f'server "{name}": auth requires an https URL, or http on localhost, 127.0.0.1, or [::1]'
        return value
    if isinstance(value.get("command"), str) and server_type in (None, "stdio"):
        if "args" in value and not (
            isinstance(value["args"], list) and all(isinstance(arg, str) for arg in value["args"])
        ):
            return f'server "{name}": args must be an array of strings'
        if "env" in value and not _is_string_record(value["env"]):
            return f'server "{name}": env must map names to strings'
        if "cwd" in value and not isinstance(value["cwd"], str):
            return f'server "{name}": cwd must be a string'
        return value
    return f'server "{name}" needs either "command" (stdio) or "url" (streamable HTTP)'


@dataclass(slots=True, frozen=True)
class RegisteredMcpServer:
    """A server an extension registered with `pi.register_mcp_server()`."""

    name: str
    config: McpServerConfig
    # Path of the extension that registered the server.
    extension_path: str


class McpServerRegistry:
    """Servers registered by the extensions of one runtime.

    Registrations come from any coroutine an extension runs on, so the map is
    changed under a guard; the change listener runs after it is released.
    """

    def __init__(self) -> None:
        self._servers: dict[str, RegisteredMcpServer] = {}
        self._guard = threading.Lock()
        self._change_listener: Callable[[], None] | None = None

    def register(self, server: RegisteredMcpServer) -> None:
        """Register or replace a server. The caller checks ownership."""
        with self._guard:
            self._servers[server.name] = server
            listener = self._change_listener
        if listener is not None:
            listener()

    def claim(self, server: RegisteredMcpServer) -> RegisteredMcpServer | None:
        """Register a server, or replace the one its extension registered
        earlier. A name another extension owns, or a server whose name differs
        only in `-` and `_` (they would share a namespace), is left alone and
        that registration returned: the checks and the registration are one
        step, for extensions registering in parallel."""
        with self._guard:
            conflict = self._conflict_of(server)
            if conflict is not None:
                return conflict
            self._servers[server.name] = server
            listener = self._change_listener
        if listener is not None:
            listener()
        return None

    def conflict_of(self, server: RegisteredMcpServer) -> RegisteredMcpServer | None:
        """The registration `claim()` would refuse `server` for, if any."""
        with self._guard:
            return self._conflict_of(server)

    def _conflict_of(self, server: RegisteredMcpServer) -> RegisteredMcpServer | None:
        registered = self._servers.get(server.name)
        if registered is not None and registered.extension_path != server.extension_path:
            return registered
        namespace = mcp_namespace(server.name)
        return next(
            (
                other
                for other in self._servers.values()
                if other.name != server.name and mcp_namespace(other.name) == namespace
            ),
            None,
        )

    def unregister(self, name: str, extension_path: str) -> None:
        """Remove a server registered by `extension_path`. Servers of other extensions are left alone."""
        with self._guard:
            server = self._servers.get(name)
            if server is None or server.extension_path != extension_path:
                return
            del self._servers[name]
            listener = self._change_listener
        if listener is not None:
            listener()

    def get(self, name: str) -> RegisteredMcpServer | None:
        with self._guard:
            return self._servers.get(name)

    def list(self) -> list[RegisteredMcpServer]:
        """Copies of the registered servers, in registration order."""
        with self._guard:
            servers = list(self._servers.values())
        return [
            RegisteredMcpServer(
                name=server.name, config=copy.deepcopy(server.config), extension_path=server.extension_path
            )
            for server in servers
        ]

    def set_change_listener(self, listener: Callable[[], None] | None) -> None:
        """Called after every change. The runner sets it when it binds, to emit `mcp_servers_change`."""
        with self._guard:
            self._change_listener = listener
