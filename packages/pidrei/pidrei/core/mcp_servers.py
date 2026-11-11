"""Mirror of pi coding-agent src/core/mcp-servers.ts.

MCP server configuration and the servers extensions register with
`pi.register_mcp_server()`.

The core only validates and stores registrations. An MCP extension (one that
handles `mcp_servers_change`) connects them: the built-in MCP extension, or
another one that replaced it. A registration nothing handles is reported.

Configs keep the JSON shape of an `mcpServers` entry in `mcp.json`
(camelCase keys), like pi's objects. The `mcpServers` shape itself (stdio
and HTTP fields, `oauth`, `enabled`, `description`, `timeout`) is
pidrei-mcp's; this file adds pidrei's fields to it.
"""

import copy
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pidrei_mcp import McpServerConfigOptions, mcp_namespace, validate_mcp_server_config as validate_mcp_server_fields
from pidrei_mcp.url import parse_url


def create_tool_name_matcher(entries: list[str] | tuple[str, ...]) -> Callable[[str], bool]:
    """Whether a tool name matches any of the entries, each an exact name or a pattern."""
    names = {entry for entry in entries if "*" not in entry}
    patterns = [_tool_pattern(entry) for entry in entries if "*" in entry]
    return lambda name: name in names or any(pattern.match(name) for pattern in patterns)


# MCP resource tools, which reach every server with resources.
LIST_MCP_RESOURCES_TOOL = "list_mcp_resources"
LIST_MCP_RESOURCE_TEMPLATES_TOOL = "list_mcp_resource_templates"
READ_MCP_RESOURCE_TOOL = "read_mcp_resource"

_MCP_RESOURCE_TOOLS: frozenset[str] = frozenset(
    (LIST_MCP_RESOURCES_TOOL, LIST_MCP_RESOURCE_TEMPLATES_TOOL, READ_MCP_RESOURCE_TOOL)
)


def is_mcp_tool_name(name: str) -> bool:
    """Whether a tool comes from MCP: a server tool (`mcp__<server>__<tool>`) or a resource tool."""
    return name.startswith("mcp__") or name in _MCP_RESOURCE_TOOLS


# - "codemode": tools are callable from codemode scripts but neither declared
#   to the model nor listed in the codemode description, which lists only the
#   server's namespace. Scripts find them with `search_tools()`.
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
#   server's tools by it, and codemode's `describe_namespace()` returns it.
# - `oauth.clientName`: `client_name` sent with dynamic client registration,
#   for servers that only accept known clients. Default: `pidrei`.
# - `oauth.clientRegistration`: how pidrei identifies itself without
#   `clientId`. `"auto"` (default): pi's Client ID Metadata Document on pi.dev
#   when the authorization server supports it for public clients, otherwise
#   dynamic client registration; with `clientName` or a `callbackUrl` other
#   than `/callback` on localhost or 127.0.0.1, always dynamic client
#   registration. pidrei is not pi, so `"auto"` always registers dynamically
#   (`_pi_settings` in `extensions/mcp/oauth.py`).
#   `"cimd"`: always the document, and sign-in fails without support.
#   `"dcr"`: always dynamic client registration.
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


def is_client_metadata_document_callback(callback_url: str | None) -> bool:
    """Whether pi's Client ID Metadata Documents list a redirect URI for the
    loopback `callback_url`: they list `/callback` on localhost and 127.0.0.1,
    on any port. Without `callback_url`, pidrei uses such a URI."""
    if callback_url is None:
        return True
    try:
        url = parse_url(callback_url)
    except ValueError:
        return False
    return url.hostname != "[::1]" and url.pathname == "/callback"


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
    """Regular expression for a tool name pattern where `*` matches any characters, as
    `toolExposure`, `--tools`, and `--exclude-tools` accept them."""
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


def _check_exposures(entry: dict[str, Any]) -> str | None:
    """Checks `exposure` and `toolExposure`, after aliases were resolved. Returns an error message."""
    exposures = ", ".join(f'"{exposure}"' for exposure in _MCP_EXPOSURES)
    if "exposure" in entry and not _is_exposure(entry["exposure"]):
        return f"exposure must be one of {exposures}"
    if "toolExposure" in entry:
        tool_exposure = entry["toolExposure"]
        if not isinstance(tool_exposure, dict):
            return "toolExposure must map tool names to exposures"
        for tool, exposure in tool_exposure.items():
            if not _is_exposure(exposure):
                return f'toolExposure "{tool}" must be one of {exposures}'
    return None


_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]")


def _check_auth(entry: dict[str, Any], url: str) -> str | None:
    """Checks `auth` of an HTTP server, which sends a provider's credential to `url`. Returns an error message."""
    if "auth" not in entry:
        return None
    auth = entry["auth"]
    if not isinstance(auth, dict) or not isinstance(auth.get("provider"), str) or not auth["provider"]:
        return "auth.provider must be a provider name"
    parsed = parse_url(url)
    if parsed.protocol != "https:" and parsed.hostname not in _LOOPBACK_HOSTS:
        return "auth requires an https URL, or http on localhost, 127.0.0.1, or [::1]"
    return None


def _check_metadata_document_callback(callback_url: str | None) -> str | None:
    if is_client_metadata_document_callback(callback_url):
        return None
    return 'oauth.clientRegistration "cimd" requires oauth.callbackUrl on localhost or 127.0.0.1 with path /callback'


_FIELD_OPTIONS = McpServerConfigOptions(check_metadata_document_callback=_check_metadata_document_callback)


def validate_mcp_server_config(name: str, raw: Any) -> McpServerConfig | str:
    """Validate one server entry of the `mcpServers` shape: pidrei-mcp's
    shared fields and pidrei's own. Returns a copy of the config with exposure
    aliases resolved, or an error message."""
    result = validate_mcp_server_fields(
        name, _resolve_exposure_aliases(raw) if isinstance(raw, dict) else raw, _FIELD_OPTIONS
    )
    if result["kind"] == "invalid":
        return result["error"]
    config = result["config"]
    error = _check_exposures(config)
    if error is None and result["kind"] == "http":
        error = _check_auth(config, config["url"])
    if error:
        return f'server "{name}": {error}'
    return config


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
