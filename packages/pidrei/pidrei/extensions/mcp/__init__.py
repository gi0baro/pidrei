"""Mirror of pi coding-agent src/extensions/mcp/index.ts: the built-in MCP
integration.

Connects the servers from `mcp.json` and the servers extensions register with
`pi.register_mcp_server()` when a session starts, and servers registered later
right away. A server in `mcp.json` takes precedence over a registered server of
the same name. Connections run in the background: the first prompt waits only
for servers with `direct` tools, and codemode scripts, `tool_search`, and the
resource tools wait for the servers they need when they run. Tools are
registered as `mcp__<server>__<tool>`. By default (`"exposure": "codemode"`)
the tools are only callable from codemode scripts, which keeps MCP tools out of
the model's tool declarations and the codemode description: scripts find the
tools with `search_tools()` and the server instructions with
`describe_namespace()`. The codemode tool is activated for that unless
`autoEnableCodemode` is false. `"deferred"` declares the tools to the model once
the `tool_search` tool loads them, and activates `tool_search` instead of
codemode. `"exposure": "direct"` declares them to the model right away, and
`"hidden"` makes them unreachable. `toolExposure` overrides the exposure of
single tools. Servers with resources are reached through Codex's
`list_mcp_resources`, `list_mcp_resource_templates`, and `read_mcp_resource`
tools (resources.py).

Every call runs through pidrei's tool pipeline, so `tool_call`/`tool_result`
hooks and permission extensions apply to MCP tools the same way they do to
built-in tools.

Problems found at startup (config errors, failed connections, servers that
need a sign-in) are reported once. `/mcp` opens a manager to sign in,
reconnect, enable or disable servers, and change their exposure; the last two
are saved to the `mcp.json` that defines the server, or apply to the current
session for registered servers.

What diverges from pi's shape:
- The extension's state (the server list, tool-name assignments, sign-in
  snapshots, flags) is behind one thread lock, held for synchronous stretches
  only. The server list is a tuple, rebound whole; readers pin one read. Lock
  order: this lock, then the session's tool loadout guard (`register_tools`,
  `update_active_tools`); `update_active_tools` callbacks never take it.
- Each server update registers its tools (and the resource tools, and the
  active-tools change that follows) in one `register_tools` call, so the
  session never sees part of it, as pi's synchronous loop guarantees.
- pi's read-then-set of the active tools goes through `update_active_tools`
  (recipe `update-active-tools`).
- A server's `ready` is an Event set when the connection started for it
  connected or failed. Background work (startup connections, the startup
  report) runs on detached coroutines; the first-prompt wait bounds each
  `ready` by one deadline, a tool call's wait ends early when the run is
  cancelled.
- Each connection attempt is a fresh `object()` compared by identity (pi's
  `Symbol()`), and a disabled server's cleanup is a `closing` Event a
  re-enable waits for. pi's per-session `AbortController` is a `CancelToken`;
  work it tracks (manager actions, sign-ins) registers an Event that
  shutdown joins.
- The MCP client is imported with the extension (pi loads it lazily), so the
  "MCP failed to load" reports have nothing to report and are not ported.
- The manager (ui.py) changes its view only through `tui.apply`; its menu
  builders read this state without the lock, so the lock is never taken under
  the UI lock.
"""

import json
import os
import re
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import tonio.colored as tonio

from pidrei_ai.utils.tasks import gather
from pidrei_utils import clock
from pidrei_utils.cancel import CancelToken, combine_cancel_tokens

from ...config import get_agent_dir
from ...core.extensions.types import ToolDefinition, ToolNamespace
from ...core.mcp_servers import mcp_namespace
from ...utils.open_browser import open_browser
from ..codemode.tool import CODEMODE_TOOL_NAME, is_codemode_tool
from ..tool_search.tool import TOOL_SEARCH_TOOL_NAME, is_tool_search_tool
from .config import (
    LoadedMcpConfig,
    McpExposure,
    McpServerConfigPatch,
    McpServerEntry,
    get_mcp_tool_exposure,
    load_mcp_config,
    locale_order,
    update_mcp_server_config,
)
from .oauth import McpSignInPrompt
from .resources import (
    LIST_MCP_RESOURCE_TEMPLATES_TOOL,
    LIST_MCP_RESOURCES_TOOL,
    READ_MCP_RESOURCE_TOOL,
    create_mcp_resource_tool_definitions,
)
from .runtime import (
    ConnectionView,
    McpOAuthCredentialStore,
    McpServerConnection,
    McpServerLog,
    McpSignInCancelledError,
    McpTransportFactory,
    create_default_transport,
    sign_in_mcp_server,
)
from .tools import create_mcp_tool_definition, create_mcp_tool_name, create_mcp_tool_renderers
from .ui import McpManagerView, McpMenu, Subscribe, show_mcp_manager


__all__ = [
    "MAX_SERVERS_SECTION_CHARS",
    "MCP_SERVERS_SECTION",
    "McpServerListing",
    "McpTransportFactory",
    "create_mcp_extension",
    "extension",
    "render_servers_section",
]

_DEFAULT_STARTUP_WAIT_MS = 10_000

_RESOURCE_TOOL_NAMES = frozenset((LIST_MCP_RESOURCES_TOOL, LIST_MCP_RESOURCE_TEMPLATES_TOOL, READ_MCP_RESOURCE_TOOL))


class _McpServer:
    """A configured server. Disabled servers have no connection. Its fields
    are written under the extension's state lock."""

    __slots__ = ("attempt", "closing", "connection", "entry", "message", "ready", "registered_config")

    def __init__(self, entry: McpServerEntry, registered_config: str | None = None) -> None:
        self.entry = entry
        self.connection: McpServerConnection | None = None
        # For servers extensions registered: the config as registered, to detect re-registrations.
        self.registered_config = registered_config
        # Result of the last `/mcp` action that failed.
        self.message: str | None = None
        # Identifies the current connection attempt (pi's `Symbol()`, compared by identity);
        # disabling or replacing it invalidates earlier work.
        self.attempt: object | None = None
        # Set when the connection started for the server connected or failed.
        self.ready: tonio.Event | None = None
        # Set when a detached connection's cleanup ended; a replacement waits for it before opening a transport.
        self.closing: tonio.Event | None = None


def _first_line(text: str) -> str:
    return text.split("\n", 1)[0]


def _is_enabled(server: Any) -> bool:
    return server.entry.config.get("enabled") is not False


def _exposure_of(entry: McpServerEntry) -> McpExposure:
    exposure = entry.config.get("exposure")
    return exposure if exposure is not None else "codemode"


def _configured_exposures(entry: McpServerEntry) -> set[McpExposure]:
    """Exposures the server's tools can have, known from its config before it connects."""
    return {_exposure_of(entry), *(entry.config.get("toolExposure") or {}).values()}


def _has_direct_tools(entry: McpServerEntry) -> bool:
    """Whether some of the server's tools are declared to the model, so the first prompt waits for them."""
    return "direct" in _configured_exposures(entry)


def _has_indirect_tools(entry: McpServerEntry) -> bool:
    """Whether some of the server's tools are reached through codemode or tool_search."""
    exposures = _configured_exposures(entry)
    return "codemode" in exposures or "deferred" in exposures


# Name of the system prompt section that lists the servers whose tools are not declared.
MCP_SERVERS_SECTION = "mcp_servers"
# Characters of a server description in the section, as Codex allows for deferred namespaces.
_MAX_SERVER_DESCRIPTION_CHARS = 250
# Characters of the whole section. Descriptions shrink to fit; when the server
# lines alone do not fit, the last servers are left out and counted in a
# closing line.
MAX_SERVERS_SECTION_CHARS = 4096


def _servers_section_intro(reaches: set[str]) -> str:
    """The section's first line. It explains only the ways of reaching tools that the listed servers use."""
    intro = "MCP servers whose tools are not declared to you."
    if "codemode" in reaches:
        intro += " Call the tools of `codemode` servers from codemode scripts."
    if "tool_search" in reaches:
        intro += " Load the tools of `tool_search` servers with `tool_search`."
    return intro


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return "" if limit <= 1 else f"{text[: limit - 1].rstrip()}…"


@dataclass(frozen=True, slots=True)
class McpServerListing:
    """What the `mcp_servers` section needs of a server: its entry and, once
    connected, its connection (for `instructions`)."""

    entry: McpServerEntry
    connection: Any = None


def _server_summary(server: Any) -> str:
    """First line of the configured description, or of the server instructions once connected."""
    description = (server.entry.config.get("description") or "").strip()
    connection = server.connection
    instructions = connection.instructions if connection is not None else None
    return _first_line(description or instructions or "").strip()


def render_servers_section(servers: Sequence[Any]) -> str | None:
    """The `mcp_servers` section: every enabled server with codemode or
    deferred tools, with how its tools are reached and a one-line summary. The
    model learns of the servers from it, since neither codemode nor
    tool_search lists them. None when there are no such servers."""
    listed = sorted(
        (server for server in servers if _is_enabled(server) and _has_indirect_tools(server.entry)),
        key=lambda server: locale_order(server.entry.name),
    )
    if not listed:
        return None
    reaches = ["codemode" if "codemode" in _configured_exposures(server.entry) else "tool_search" for server in listed]
    intro = _servers_section_intro(set(reaches))
    heads = [f"- {mcp_namespace(server.entry.name)} ({reach})" for server, reach in zip(listed, reaches, strict=True)]

    def omitted(count: int) -> list[str]:
        if count <= 0:
            return []
        return [f"- … {count} more server{'' if count == 1 else 's'}; find their tools with search_tools()"]

    def size(kept: int) -> int:
        """Characters of the intro, the first `kept` server lines without descriptions, and the omission line."""
        return len("\n".join([intro, *heads[:kept], *omitted(len(listed) - kept)]))

    kept = len(listed)
    while kept > 0 and size(kept) > MAX_SERVERS_SECTION_CHARS:
        kept -= 1
    # Each description also takes a ": " separator.
    per_server = (
        0 if kept == 0 else min(_MAX_SERVER_DESCRIPTION_CHARS, (MAX_SERVERS_SECTION_CHARS - size(kept)) // kept - 2)
    )
    lines = []
    for server, head in zip(listed[:kept], heads, strict=False):
        summary = _truncate(_server_summary(server), per_server) if per_server > 0 else ""
        lines.append(f"{head}: {summary}" if summary else head)
    return "\n".join([intro, *lines, *omitted(len(listed) - kept)])


# Script API names that search, enumerate or describe tools or namespaces, which
# may name a server in other forms (`has_tool` reads `ALL_TOOLS` in the prelude).
_SCRIPT_DISCOVERY = re.compile(r"\b(search_tools|describe_namespace|describe_tool|ALL_TOOLS|call_tool|has_tool)\b")


def _script_needs_server(code: str, server: str) -> bool:
    """Whether a codemode script needs the server: it names the server's
    namespace, or searches, enumerates, or describes tools or namespaces,
    which may name the server in other forms."""
    if _SCRIPT_DISCOVERY.search(code):
        return True
    return mcp_namespace(server) in code


def _view_of(server: _McpServer) -> ConnectionView | None:
    """The snapshot of the server's connection, None without one. Two reads
    can see two snapshots: a reader of more than one field takes it once."""
    connection = server.connection
    return connection.view if connection is not None else None


def _in_state(server: _McpServer, *states: str) -> bool:
    """Whether the server has a connection in one of `states`."""
    view = _view_of(server)
    return view is not None and view.state in states


def _describe_state(server: _McpServer, with_error: bool = True) -> str:
    """Short state for lists and the startup report. `with_error` appends the first line of a failure."""
    return _describe_view(server, _view_of(server), with_error)


def _describe_view(server: _McpServer, view: ConnectionView | None, with_error: bool = True) -> str:
    """`_describe_state` of a snapshot the caller took."""
    if not _is_enabled(server):
        return "disabled"
    if view is None:
        return "starting"
    match view.state:
        case "needs-auth":
            return "needs sign-in"
        case "failed":
            return f"failed: {_first_line(view.error or 'unknown error')}" if with_error else "failed"
        case "connected":
            tools, count = view.tools, len(view.resources)
            resource_count = f" · {count} resource{'' if count == 1 else 's'}" if count > 0 else ""
            return f"connected · {len(tools)} tool{'' if len(tools) == 1 else 's'}{resource_count}"
        case "connecting":
            return "connecting…"
        case state:
            return state


_EXPOSURE_DESCRIPTIONS: dict[str, str] = {
    "codemode": "called from codemode scripts, which find them with search_tools()",
    "deferred": "not declared until tool_search loads them, then called directly; no codemode needed",
    "direct": "declared to the model like built-in tools",
}


def _attention_rank(server: _McpServer, view: ConnectionView | None) -> int:
    """Servers that need the user first."""
    if not _is_enabled(server):
        return 5
    match view.state if view is not None else None:
        case "needs-auth":
            return 0
        case "failed":
            return 1
        case "disconnected":
            return 2
        case "connected":
            return 4
        case _:
            return 3


def _describe_transport(entry: McpServerEntry) -> str:
    config = entry.config
    if "url" in config:
        return config["url"]
    return " ".join([config["command"], *(config.get("args") or [])])


_MCP_USAGE = "Usage: /mcp, /mcp login [server], /mcp logout [server], /mcp reconnect [server]"


async def _wait_all(operations: Sequence[Awaitable[Any]]) -> None:
    """pi's `Promise.all` for operations whose results are not needed: run them concurrently."""

    async def run(operation: Awaitable[Any]) -> None:
        await operation

    handles = [tonio.spawn(run(operation)) for operation in operations]
    for handle in handles:
        await handle


async def _settle_all(operations: Sequence[Awaitable[Any]]) -> None:
    """pi's `Promise.allSettled`, outcomes ignored."""

    async def settle(operation: Awaitable[Any]) -> None:
        try:
            await operation
        except Exception:
            pass

    await _wait_all([settle(operation) for operation in operations])


def _default_load_config(ctx: Any) -> Awaitable[LoadedMcpConfig]:
    return load_mcp_config(agent_dir=get_agent_dir(), cwd=ctx.cwd, project_trusted=ctx.is_project_trusted())


def _config_hint() -> str:
    path = os.path.normpath(os.path.join(get_agent_dir(), "mcp.json"))
    return f"No MCP servers configured. Add them to {path} or .pidrei/mcp.json."


def create_mcp_extension(
    *,
    # Defaults to reading `mcp.json` from the agent directory and the trusted project.
    load_config: Callable[[Any], Awaitable[LoadedMcpConfig]] | None = None,
    # Defaults to stdio and streamable HTTP transports built from the server config.
    create_transport: McpTransportFactory | None = None,
    # Defaults to `mcp-auth.json` in the agent directory.
    credentials: McpOAuthCredentialStore | None = None,
    # File server log messages are appended to. Defaults to `mcp.log` in the agent directory.
    log_path: str | None = None,
    # Opens the OAuth authorization URL. Defaults to the platform browser.
    open_url: Callable[[str], None] | None = None,
    # Saves `/mcp` changes to the server's config file: its project `override`
    # when set, else its `source`. Defaults to editing that `mcp.json`.
    update_config: Callable[[McpServerEntry, McpServerConfigPatch], Awaitable[None]] | None = None,
    # How long the first prompt waits for servers with `direct` tools that are
    # still connecting at startup, in milliseconds. Their tools become
    # available when they connect. Other servers are waited for when a script
    # or search needs them. Default: 10000.
    startup_wait_ms: float | None = None,
) -> Callable[[Any], Awaitable[None]]:
    async def extension(pi: Any) -> None:
        _McpExtension(
            pi,
            load_config=load_config or _default_load_config,
            create_transport=create_transport or create_default_transport,
            credentials=credentials,
            log_path=log_path,
            open_url=open_url or open_browser,
            update_config=update_config or _default_update_config,
            startup_wait_ms=startup_wait_ms if startup_wait_ms is not None else _DEFAULT_STARTUP_WAIT_MS,
        )

    return extension


# pi's `/^mcp__(.+?)__(.+)$/`, used with `fullmatch` (JS `$` is the end of input).
_MCP_TOOL_NAME = re.compile(r"mcp__(.+?)__(.+)")


def _resolve_mcp_tool_renderers(tool_name: str, next_renderers: Callable[[], Any]) -> Any:
    """Calls to `mcp__<server>__<tool>` keep the renderers others choose, else get the MCP ones."""
    renderers = next_renderers()
    if renderers is not None:
        return renderers
    match = _MCP_TOOL_NAME.fullmatch(tool_name)
    return create_mcp_tool_renderers(f"{match[1]}/{match[2]}") if match else None


def _default_update_config(entry: McpServerEntry, patch: McpServerConfigPatch) -> Awaitable[None]:
    return update_mcp_server_config(
        entry.override if entry.override is not None else entry.source,
        entry.name,
        patch,
        override=entry.override is not None,
    )


class _McpExtension:
    """pi's `createMcpExtension` closure: one per extension instance."""

    def __init__(
        self,
        pi: Any,
        *,
        load_config: Callable[[Any], Awaitable[LoadedMcpConfig]],
        create_transport: McpTransportFactory,
        credentials: McpOAuthCredentialStore | None,
        log_path: str | None,
        open_url: Callable[[str], None],
        update_config: Callable[[McpServerEntry, McpServerConfigPatch], Awaitable[None]],
        startup_wait_ms: float,
    ) -> None:
        self._pi = pi
        self._load_config = load_config
        self._create_transport = create_transport
        self._log_path = log_path
        self._open_url = open_url
        self._update_config = update_config
        self._startup_wait_ms = startup_wait_ms
        self._lock = threading.Lock()
        self._servers: tuple[_McpServer, ...] = ()
        # Servers from `mcp.json`, which take precedence over registered servers of the same name.
        self._configured_entries: list[McpServerEntry] = []
        self._config_errors: list[str] = []
        # The trusted project's `mcp.json`, where `/mcp` saves project overrides of global servers.
        self._project_config: str | None = None
        # Registered servers that `mcp.json` overrides, shown in `/mcp`.
        self._overridden: list[str] = []
        # Lifetime of the current session: cancelled on session_shutdown, and before the first
        # session_start (registrations before that are read on session_start). Work captures it when
        # it starts and drops late results once it fired. Work that outlives the command that started
        # it also stops on it and is tracked (`_track`), so shutdown can wait for its cleanup.
        self._session = CancelToken()
        self._session.cancel()
        # Tracked work (manager actions opening or closing connections, sign-ins): each sets its Event when it ends.
        self._background_actions: set[tonio.Event] = set()
        self._auto_enable_codemode = True
        # Whether the "codemode tools unreachable" warning was shown since the session started.
        self._warned_unreachable = False
        # Set once the startup connections settled and were reported.
        self._pending: tonio.Event | None = None
        # Whether a prompt already waited for the startup connections since the session started.
        self._waited_for_startup = False
        # Working directory of the session, for stdio servers.
        self._session_cwd = os.getcwd()
        self._credentials = credentials
        # The session's model registry, which resolves `auth.provider` tokens.
        self._model_registry: Any = None
        self._server_log: McpServerLog | None = None
        self._listeners: list[Callable[[], None]] = []
        # Tool name to the `<server>\0<tool>` it was assigned to, so names stay unique and stable.
        self._tool_owners: dict[str, str] = {}
        # Tool names currently offered by each server (insertion-ordered sets).
        self._server_tools: dict[str, dict[str, None]] = {}
        # Last definition registered under each tool name, to re-register withdrawn tools as hidden.
        self._definitions: dict[str, ToolDefinition] = {}
        # Exposure the resource tools were last registered with; None until a server has resources.
        self._resource_tools_exposure: McpExposure | None = None
        # Stored tokens of servers waiting for a sign-in, as they were when the
        # sign-in was needed. `pidrei mcp login` in another process (for
        # example run by the agent) changes them.
        self._tokens_at_sign_in: dict[McpServerConnection, str] = {}

        # A resumed session renders calls to MCP tools before their server connected, if it ever does.
        pi.register_tool_renderer(_resolve_mcp_tool_renderers)
        pi.on("session_start", self._on_session_start)
        pi.on("before_agent_start", self._on_before_agent_start)
        pi.on("tool_call", self._on_tool_call)
        pi.on("turn_start", self._on_turn_start)
        pi.on("mcp_servers_change", self._on_mcp_servers_change)
        pi.on("session_shutdown", self._on_session_shutdown)
        pi.register_command(
            "mcp",
            description="Manage MCP servers: sign in, reconnect, enable or disable, and change exposure",
            get_argument_completions=self._argument_completions,
            handler=self._command,
        )

    # -----------------------------------------------------------------------
    # State
    # -----------------------------------------------------------------------

    def _emit_change(self) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            listener()

    def _track(self) -> tonio.Event:
        """pi's `track`: register work that shutdown waits for. Call `_untrack`
        with the returned Event when the work ended, also when it failed."""
        done = tonio.Event()
        with self._lock:
            self._background_actions.add(done)
        return done

    def _untrack(self, done: tonio.Event) -> None:
        """Synchronous, so it can end a `finally` reached by cancellation."""
        with self._lock:
            self._background_actions.discard(done)
        done.set()

    def _subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def _connections(self) -> list[McpServerConnection]:
        return [connection for server in self._servers if (connection := server.connection) is not None]

    def _find_server(self, name: str) -> _McpServer | None:
        return next((server for server in self._servers if server.entry.name == name), None)

    def _registered_servers_locked(self) -> tuple[list[_McpServer], list[str]]:
        """Servers extensions registered, except names `mcp.json` defines, which take precedence."""
        registered: list[_McpServer] = []
        overridden: list[str] = []
        for server in self._pi.get_mcp_servers():
            configured = next(
                (
                    entry
                    for entry in self._configured_entries
                    if mcp_namespace(entry.name) == mcp_namespace(server.name)
                ),
                None,
            )
            if configured is not None:
                overridden.append(
                    f'"{server.name}" registered by {server.extension_path} is overridden by '
                    f'"{configured.name}" in {configured.source}'
                )
                continue
            registered.append(
                _McpServer(
                    McpServerEntry(
                        name=server.name, config=server.config, source=server.extension_path, scope="extension"
                    ),
                    registered_config=json.dumps(server.config),
                )
            )
        return registered, overridden

    def _get_credentials(self) -> McpOAuthCredentialStore:
        with self._lock:
            return self._credentials_locked()

    def _credentials_locked(self) -> McpOAuthCredentialStore:
        if self._credentials is None:
            self._credentials = McpOAuthCredentialStore()
        return self._credentials

    def _server_log_locked(self) -> McpServerLog:
        if self._server_log is None:
            self._server_log = McpServerLog(self._log_path or os.path.join(get_agent_dir(), "mcp.log"))
        return self._server_log

    # -----------------------------------------------------------------------
    # Tools
    # -----------------------------------------------------------------------

    def _register_tools(self, connection: McpServerConnection) -> None:
        with self._lock:
            definitions = self._server_tool_definitions_locked(connection)
            if definitions is not None:
                self._publish_tools_locked(definitions)

    def _server_tool_definitions_locked(self, connection: McpServerConnection) -> list[ToolDefinition] | None:
        """The tool definitions of a connection's server, recorded as its tools:
        the server's tools with their configured exposure, and the ones it
        dropped as hidden. None for a connection that is no longer the server's."""
        server = connection.entry.name
        found = self._find_server(server)
        # The connection checks that it is current before calling this, but
        # not in the same step: one replaced, disabled or removed in
        # between had its tools hidden, and registers nothing.
        if found is None or found.connection is not connection:
            return None
        entry = found.entry
        view = connection.view
        description = (entry.config.get("description") or "").strip()
        namespace = ToolNamespace(
            name=mcp_namespace(server), description=description or None, instructions=view.instructions
        )
        previous = self._server_tools.get(server, {})
        current: dict[str, None] = {}
        tools = view.tools
        # Like Codex, all tools whose names sanitize to the same name get the
        # hash suffix, so which one would keep the plain name does not depend
        # on the order of the list.
        plain = [create_mcp_tool_name(server, name) for name in dict.fromkeys(tool["name"] for tool in tools)]

        def assign_name(tool: str, owner: str) -> str:
            def is_taken(candidate: str) -> bool:
                existing = self._tool_owners.get(candidate)
                return (
                    (existing is not None and existing != owner) or candidate in current or plain.count(candidate) > 1
                )

            name = create_mcp_tool_name(server, tool, is_taken)
            self._tool_owners[name] = owner
            current[name] = None
            return name

        def readable_resources() -> bool:
            return any(current.entry.name == server for current, _connection in self._servers_with_resources())

        definitions: list[ToolDefinition] = []
        for tool in tools:
            definition = create_mcp_tool_definition(
                server=server,
                tool=tool,
                name=assign_name(tool["name"], f"{server}\0{tool['name']}"),
                exposure=get_mcp_tool_exposure(entry.config, tool["name"]),
                namespace=namespace,
                timeout_ms=connection.timeout_ms,
                get_client=self._client_resolver(server, tool["name"]),
                readable_resources=readable_resources,
            )
            self._definitions[definition.name] = definition
            definitions.append(definition)
        self._server_tools[server] = current
        # Tools cannot be unregistered, so tools the server dropped are
        # re-registered as hidden. When the server offers them again they are
        # registered with their configured exposure above.
        for name in previous:
            definition = self._definitions.get(name)
            if name not in current and definition is not None:
                definitions.append(replace(definition, exposure="hidden"))
        return definitions

    def _client_resolver(self, server: str, tool: str) -> Callable[[], Awaitable[McpServerConnection]]:
        """A tool's `get_client`. A prepared call can outlive the connection its
        definition was made from while the readiness hook waits for a disable or
        re-enable, so the server's current connection is resolved when the call
        executes."""

        async def get_client() -> McpServerConnection:
            with self._lock:
                current = self._find_server(server)
                enabled = current is not None and _is_enabled(current)
                client = current.connection if current is not None else None
                hidden = current is not None and get_mcp_tool_exposure(current.entry.config, tool) == "hidden"
            if not enabled:
                raise RuntimeError(f'MCP server "{server}" is disabled.')
            if client is None:
                raise RuntimeError(f'MCP server "{server}" is still starting.')
            view = client.view
            if hidden or (view.state == "connected" and all(offered["name"] != tool for offered in view.tools)):
                raise RuntimeError(f'MCP tool "{server}/{tool}" is no longer available.')
            return client

        return get_client

    def _hide_tools_locked(self, server: str) -> None:
        """Make a removed server's tools unreachable."""
        definitions = [
            replace(definition, exposure="hidden")
            for name in self._server_tools.get(server, {})
            if (definition := self._definitions.get(name)) is not None
        ]
        self._server_tools[server] = {}
        self._publish_tools_locked(definitions)

    def _publish_tools_locked(
        self,
        definitions: list[ToolDefinition],
        update_active: Callable[[list[str]], list[str] | None] | None = None,
    ) -> None:
        """Register `definitions` and the resource tools' update in one step, so the
        session never sees part of a server's tools, then apply the deactivations
        and `update_active` in the same step."""
        resource_definitions, deactivate = self._resource_tool_updates_locked()
        updates = [update for update in (deactivate, update_active) if update is not None]
        if not definitions and not resource_definitions and not updates:
            return

        def update_active_tools(active: list[str]) -> list[str] | None:
            changed = False
            for update in updates:
                result = update(active)
                if result is not None:
                    active, changed = result, True
            return active if changed else None

        self._pi.register_tools(
            [*definitions, *resource_definitions], update_active=update_active_tools if updates else None
        )

    def _servers_with_resources(self) -> list[tuple[_McpServer, McpServerConnection]]:
        """Enabled servers with resources whose exposure is not `hidden`, which
        the resource tools reach, each with the connection it was read with."""
        return [
            (server, connection)
            for server in self._servers
            if (connection := server.connection) is not None
            and connection.has_resources
            and _is_enabled(server)
            and _exposure_of(server.entry) != "hidden"
        ]

    def _resource_servers(self) -> list[McpServerConnection]:
        return [connection for _server, connection in self._servers_with_resources()]

    def _resource_tool_updates_locked(
        self,
    ) -> tuple[list[ToolDefinition], Callable[[list[str]], list[str] | None] | None]:
        """The resource tools to register with the widest exposure of the servers
        they reach (`direct` when one of them is direct, and so on; hidden when no
        server has resources), and the update that deactivates them when they stop
        being direct. Nothing when their exposure is unchanged."""
        exposures = {_exposure_of(server.entry) for server, _connection in self._servers_with_resources()}
        exposure = next((candidate for candidate in ("direct", "codemode", "deferred") if candidate in exposures), None)
        target: McpExposure = exposure if exposure is not None else "hidden"
        if target == self._resource_tools_exposure or (self._resource_tools_exposure is None and target == "hidden"):
            return [], None
        was_direct = self._resource_tools_exposure == "direct"
        self._resource_tools_exposure = target
        definitions = create_mcp_resource_tool_definitions(exposure=target, servers=self._resource_servers)
        if not was_direct:
            return definitions, None
        names = {definition.name for definition in definitions}
        return definitions, lambda active: [name for name in active if name not in names]

    def _ensure_discovery_active(self, ctx: Any) -> None:
        """Tools that are not declared to the model are reached through the
        codemode tool (scripts call them) or the tool_search tool (it declares
        them). Either reaches every such tool. Activate the one the tools'
        exposure asks for: codemode for `codemode` unless `autoEnableCodemode`
        is false, tool_search for `deferred`."""
        # From the config, so the tool is active before the servers connect.
        # Resource tools share their server's exposure.
        exposures: set[McpExposure] = set()
        for server in self._servers:
            if _is_enabled(server):
                exposures |= _configured_exposures(server.entry)
        needs_codemode = "codemode" in exposures
        needs_tool_search = "deferred" in exposures
        if not needs_codemode and not needs_tool_search:
            return
        # Other extensions' tools of the same names cannot reach MCP tools, so never activate them.
        tools = self._pi.get_all_tools()
        has_codemode = any(is_codemode_tool(tool) for tool in tools)
        has_tool_search = any(is_tool_search_tool(tool) for tool in tools)
        auto_enable_codemode = self._auto_enable_codemode
        reachable: list[str] = []

        def update(active: list[str]) -> list[str] | None:
            activate: list[str] = []
            if needs_codemode and has_codemode and auto_enable_codemode and CODEMODE_TOOL_NAME not in active:
                activate.append(CODEMODE_TOOL_NAME)
            if needs_tool_search and has_tool_search and TOOL_SEARCH_TOOL_NAME not in active:
                activate.append(TOOL_SEARCH_TOOL_NAME)
            reachable.extend([*active, *activate])
            return [*active, *activate] if activate else None

        self._pi.update_active_tools(update)
        if has_codemode and CODEMODE_TOOL_NAME in reachable:
            return
        if has_tool_search and TOOL_SEARCH_TOOL_NAME in reachable:
            return
        with self._lock:
            if self._warned_unreachable:
                return
            self._warned_unreachable = True
        reason = (
            " (autoEnableCodemode is false)" if needs_codemode and has_codemode and not auto_enable_codemode else ""
        )
        ctx.ui.notify(
            "MCP tools are only reachable from the codemode or tool_search tool, but neither is active"
            f"{reason}; they cannot be called.",
            "warning",
        )

    # -----------------------------------------------------------------------
    # Connections
    # -----------------------------------------------------------------------

    async def _stored_tokens(self, connection: McpServerConnection) -> str:
        url = connection.oauth_url
        credentials = self._credentials
        if not url or credentials is None:
            return "null"
        return json.dumps(await credentials.tokens(connection.name, url))

    async def _on_connection_change(self, connection: McpServerConnection) -> None:
        if connection.state != "needs-auth":
            with self._lock:
                self._tokens_at_sign_in.pop(connection, None)
        else:
            with self._lock:
                known = connection in self._tokens_at_sign_in
            if not known:
                tokens = await self._stored_tokens(connection)
                with self._lock:
                    # The connection may have moved on while the tokens were read.
                    if connection.state == "needs-auth" and connection not in self._tokens_at_sign_in:
                        self._tokens_at_sign_in[connection] = tokens
        self._emit_change()

    async def _reconnect_signed_in(self, ctx: Any) -> None:
        """Reconnect servers that need a sign-in when their credentials were stored since."""
        with self._lock:
            waiting = list(self._tokens_at_sign_in.items())
        stored = await gather(*(self._stored_tokens(connection) for connection, _tokens in waiting))
        signed_in = [
            connection for (connection, tokens), current in zip(waiting, stored, strict=True) if current != tokens
        ]
        if not signed_in:
            return
        with self._lock:
            for connection in signed_in:
                self._tokens_at_sign_in.pop(connection, None)
        await _settle_all([connection.reconnect() for connection in signed_in])
        self._ensure_discovery_active(ctx)

    async def _provider_token(self, provider: str) -> str | None:
        registry = self._model_registry
        return await registry.get_api_key_for_provider(provider) if registry is not None else None

    def _create_connection_locked(
        self, server: _McpServer, is_current_locked: Callable[[], bool]
    ) -> McpServerConnection:
        """The server's connection for one attempt. Once the attempt is no longer
        current (disabled, replaced, or the session ended), its tool updates do
        nothing and its state changes only drop its sign-in snapshot."""

        def on_tools(connection: McpServerConnection) -> None:
            with self._lock:
                if not is_current_locked():
                    return
                definitions = self._server_tool_definitions_locked(connection)
                if definitions is not None:
                    self._publish_tools_locked(definitions)

        async def on_change(connection: McpServerConnection) -> None:
            with self._lock:
                current = is_current_locked()
                if not current:
                    self._tokens_at_sign_in.pop(connection, None)
            if current:
                await self._on_connection_change(connection)

        connection = McpServerConnection(
            entry=server.entry,
            cwd=self._session_cwd,
            create_transport=self._create_transport,
            credentials=self._credentials_locked(),
            provider_token=self._provider_token,
            log=self._server_log_locked(),
            on_tools=on_tools,
            on_change=on_change,
        )
        server.connection = connection
        return connection

    def _start_connection(
        self,
        server: _McpServer,
        after: tonio.Event | None = None,
        on_attempt: Callable[[object | None], None] | None = None,
    ) -> tonio.Event:
        """Connect the server in the background. The returned Event (also
        `server.ready`) is set when it connected or failed; failures show in
        its state. It starts once `after` is set, unless the server was
        disabled, replaced (a newer attempt), or removed, or the session ended
        meanwhile: nothing would close its connection. `on_attempt` receives
        the new attempt, under the lock that sets it."""
        ready = tonio.Event()
        attempt = object()
        with self._lock:
            session = self._session
            server.attempt = attempt
            server.ready = ready
            if on_attempt is not None:
                on_attempt(attempt)

        def is_current_locked() -> bool:
            # By identity: a new session and a re-registration list new servers.
            return (
                not session.cancelled
                and server.attempt is attempt
                and _is_enabled(server)
                and any(candidate is server for candidate in self._servers)
            )

        async def connect() -> None:
            try:
                if after is not None:
                    await after.wait()
                with self._lock:
                    if not is_current_locked():
                        return
                    connection = self._create_connection_locked(server, is_current_locked)
                self._emit_change()
                try:
                    await connection.get_client()
                except Exception:
                    # Shown in the connection's state.
                    pass
            finally:
                ready.set()

        tonio.spawn.without_tracking(connect())
        return ready

    async def _wait_for_servers(self, waiting: Sequence[_McpServer], cancel: Any) -> None:
        """Wait for the latest connection attempts of `waiting`, including ones
        queued while waiting (a reconnect, a re-enable), until they settle or
        `cancel` fires."""
        while True:
            with self._lock:
                attempts = [
                    (server, ready)
                    for server in waiting
                    if _is_enabled(server)
                    and any(candidate is server for candidate in self._servers)
                    and (ready := server.ready) is not None
                ]
            if not attempts or (cancel is not None and cancel.cancelled):
                return
            for _server, ready in attempts:
                if cancel is None:
                    await ready.wait()
                else:
                    await tonio.Waiter.any(ready, cancel.event)
                    if cancel.cancelled:
                        return
            with self._lock:
                if all(
                    not _is_enabled(server)
                    or not any(candidate is server for candidate in self._servers)
                    or ready is server.ready
                    for server, ready in attempts
                ):
                    return

    def _report_problems(self, ctx: Any, only: Sequence[_McpServer] | None = None) -> None:
        """One message for everything that needs the user after startup, or
        only for `only`, servers that connected later."""
        lines = [] if only is not None else [f"config: {error}" for error in self._config_errors]
        for server in only if only is not None else self._servers:
            view = _view_of(server)
            if view is not None and view.state in ("needs-auth", "failed"):
                lines.append(f"{server.entry.name}: {_describe_view(server, view)}")
        if not lines:
            return
        body = "\n".join(f"  {line}" for line in lines)
        ctx.ui.notify(f"MCP servers need attention:\n{body}\nRun /mcp to fix.", "warning")

    # -----------------------------------------------------------------------
    # Sign-in, sign-out, reconnect
    # -----------------------------------------------------------------------

    async def _sign_in(
        self, server: _McpServer, prompt: McpSignInPrompt, cancel: CancelToken | None = None
    ) -> str | None:
        """Sign in, then reconnect. Returns an error message on failure. `cancel`
        or the session's end stops the sign-in, which shutdown waits for."""
        connection = server.connection
        url = connection.oauth_url if connection is not None else None
        if connection is None or not url:
            return f'MCP server "{server.entry.name}" does not use OAuth.'
        settings = await connection.oauth_settings()
        with self._lock:
            session = self._session
        combined = combine_cancel_tokens(session, cancel)
        done = self._track()
        try:
            await sign_in_mcp_server(
                server_url=url,
                store=self._get_credentials().for_server(server.entry.name, url),
                settings=settings,
                challenge=connection.challenge,
                prompt=prompt,
                cancel=combined.token,
            )
        except McpSignInCancelledError:
            return "Sign-in cancelled."
        except Exception as error:
            return f"Sign-in failed: {error}"
        finally:
            combined.cleanup()
            self._untrack(done)
        # The challenge that asked for this sign-in (for example for more scope) is answered.
        connection.challenge = None
        try:
            await connection.reconnect()
        except Exception as error:
            return f"Signed in, but {error}"
        return None

    async def _sign_out(self, server: _McpServer) -> bool:
        connection = server.connection
        url = connection.oauth_url if connection is not None else None
        if connection is None or not url:
            return False
        removed = await self._get_credentials().remove(server.entry.name, url)
        await connection.sign_out()
        return removed

    async def _reconnect(self, server: _McpServer) -> str | None:
        connection = server.connection
        if connection is None:
            return f'MCP server "{server.entry.name}" is disabled.'
        # Queued behind the previous attempt, and the server's readiness until it ends.
        ready = tonio.Event()
        with self._lock:
            previous = server.ready
            server.ready = ready
        try:
            if previous is not None:
                await previous.wait()
            await connection.reconnect()
        except Exception as error:
            return str(error)
        finally:
            ready.set()
        return None

    async def _save_config(
        self, server: _McpServer, patch: McpServerConfigPatch, in_project: bool = False
    ) -> str | None:
        """Save a config change; returns an error message when the file could
        not be updated. Changes to registered servers only apply to the
        current session. `in_project` adds a project override."""
        override = self._project_config if in_project else server.entry.override
        entry = replace(server.entry, override=override) if override else server.entry
        if entry.scope != "extension":
            try:
                await self._update_config(entry, patch)
            except Exception as error:
                return f"Could not update {entry.override if entry.override is not None else entry.source}: {error}"
        with self._lock:
            # Re-read: the entry may have changed while the file was written.
            current = replace(server.entry, override=override) if override else server.entry
            server.entry = replace(current, config={**current.config, **patch.as_config()})
        return None

    async def _set_enabled(
        self,
        server: _McpServer,
        enabled: bool,
        in_project: bool = False,
        on_attempt: Callable[[object | None], None] | None = None,
    ) -> str | None:
        """Returns an error message when the config could not be saved; connection errors show in the state.
        `on_attempt` receives the attempt it sets (None when disabling), under the lock that sets it."""
        failed = await self._save_config(server, McpServerConfigPatch(enabled=enabled), in_project)
        if failed:
            return failed
        if not enabled:
            with self._lock:
                connection = server.connection
                server.attempt = None
                if on_attempt is not None:
                    on_attempt(None)
                server.connection = None
                if connection is not None:
                    # A reconnect may already have detached its old client while awaiting transport
                    # shutdown. Closing the connection alone does not wait for that reconnect.
                    server.closing = self._close_detached(connection, server.ready)
                closing = server.closing
                self._hide_tools_locked(server.entry.name)
            self._emit_change()
            try:
                if closing is not None:
                    await closing.wait()
            finally:
                with self._lock:
                    if server.closing is closing:
                        server.closing = None
            return None
        with self._lock:
            after = server.closing
        await self._start_connection(server, after, on_attempt).wait()
        return None

    @staticmethod
    def _close_detached(connection: McpServerConnection, ready: tonio.Event | None) -> tonio.Event:
        """Close a connection that left its server, and wait for its attempt's
        readiness (pi: `Promise.all([connection.close(), server.ready])`). The
        returned Event is set once both ended. Detached, so a re-enable can wait
        for it after the disabling call is gone."""
        closed = tonio.Event()

        async def close() -> None:
            try:
                await connection.close()
                if ready is not None:
                    await ready.wait()
            finally:
                closed.set()

        tonio.spawn.without_tracking(close())
        return closed

    async def _set_exposure(self, server: _McpServer, exposure: McpExposure) -> str | None:
        failed = await self._save_config(server, McpServerConfigPatch(exposure=exposure))
        if failed:
            return failed
        connection = server.connection
        with self._lock:
            definitions = (
                self._server_tool_definitions_locked(connection)
                if connection is not None and connection.state == "connected"
                else None
            )
            tools = set(self._server_tools.get(server.entry.name, {}))

            def drop_indirect(active: list[str]) -> list[str]:
                # Tools no longer exposed directly leave the declared set; direct tools are activated on
                # registration. Runs after the registration, in the same step.
                indirect = {tool.name for tool in self._pi.get_all_tools() if tool.exposure != "direct"}
                return [name for name in active if name not in tools or name not in indirect]

            self._publish_tools_locked(definitions or [], drop_indirect)
        self._emit_change()
        return None

    # -----------------------------------------------------------------------
    # Manager (`/mcp` in the TUI)
    # -----------------------------------------------------------------------

    def _notices(self) -> list[str]:
        return [
            *(f"config: {error}" for error in self._config_errors),
            *(f"overridden: {line}" for line in self._overridden),
        ]

    def _servers_menu(self) -> McpMenu:
        # Each server is ranked and described from one snapshot.
        servers = sorted(
            ((server, _view_of(server)) for server in self._servers),
            key=lambda listed: (_attention_rank(*listed), locale_order(listed[0].entry.name)),
        )
        return McpMenu(
            title="MCP servers",
            error="\n".join(self._notices()) or None,
            items=[
                {
                    "value": server.entry.name,
                    "label": server.entry.name,
                    "description": (
                        f"{_describe_view(server, view)} · {_exposure_of(server.entry)} · "
                        f"{'global, project override' if server.entry.override else (server.entry.scope or server.entry.source)}"
                    ),
                }
                for server, view in servers
            ],
            empty=_config_hint(),
            confirm_label="manage",
            cancel_label="close",
        )

    def _server_menu(self, name: str) -> McpMenu:
        server = self._find_server(name)
        if server is None:
            return McpMenu(
                title=name,
                items=[],
                empty="This server is no longer configured.",
                confirm_label="",
                cancel_label="back",
            )
        entry, connection = server.entry, server.connection
        view = connection.view if connection is not None else None
        if entry.scope == "extension":
            saved = "for this session"
        elif entry.override:
            saved = "saved to the project mcp.json"
        elif entry.scope:
            saved = f"saved to the {entry.scope} mcp.json"
        else:
            saved = "saved to mcp.json"
        # Global servers without an override can be turned on or off for the trusted project alone.
        in_project = entry.scope == "global" and not entry.override and self._project_config is not None
        in_project_saved = "saved to the project mcp.json"
        items: list[dict[str, str]] = []
        if not _is_enabled(server):
            items.append({"value": "enable", "label": "Enable", "description": saved})
            if in_project:
                items.append(
                    {"value": "enable-project", "label": "Enable in this project", "description": in_project_saved}
                )
        else:
            state = view.state if view is not None else None
            if state == "needs-auth":
                items.append({"value": "signin", "label": "Sign in", "description": "opens the browser"})
            if state == "connected" and view is not None:
                items.append({"value": "tools", "label": "Tools", "description": f"{len(view.tools)} offered"})
            if state in ("failed", "disconnected", "connected", "needs-auth"):
                items.append({"value": "reconnect", "label": "Reconnect"})
            if state == "connected" and connection is not None and connection.oauth_url:
                items.append({"value": "signout", "label": "Sign out", "description": "deletes the stored credentials"})
            items.append({"value": "exposure", "label": "Exposure", "description": _exposure_of(entry)})
            items.append({"value": "disable", "label": "Disable", "description": saved})
            if in_project:
                items.append(
                    {"value": "disable-project", "label": "Disable in this project", "description": in_project_saved}
                )
        details = [
            _describe_transport(entry),
            f"{entry.scope or 'config'}: {entry.source}",
            *([f"project override: {entry.override}"] if entry.override else []),
            f"State: {_describe_view(server, view, False)}",
        ]
        errors = [
            line
            for line in (
                server.message,
                view.error if view is not None and view.state != "connected" else None,
            )
            if line is not None
        ]
        return McpMenu(
            title=f"MCP server {name}",
            details="\n".join(details),
            error="\n".join(errors) or None,
            items=items,
            selected=items[0]["value"] if items else None,
            confirm_label="select",
            cancel_label="back",
        )

    async def _show_tools(self, ui: McpManagerView, server: _McpServer) -> None:
        exposure = _exposure_of(server.entry)
        overridden = bool(server.entry.config.get("toolExposure"))
        reach = "unreachable" if exposure == "hidden" else _EXPOSURE_DESCRIPTIONS[exposure]
        note = "\nSome tools override it with toolExposure." if overridden else ""

        def build() -> McpMenu:
            view = _view_of(server)
            tools = view.tools if view is not None else []
            items = []
            for tool in tools:
                tool_exposure = get_mcp_tool_exposure(server.entry.config, tool["name"])
                description = _first_line(tool.get("description") or "")
                items.append(
                    {
                        "value": tool["name"],
                        "label": tool["name"],
                        "description": description if tool_exposure == exposure else f"[{tool_exposure}] {description}",
                    }
                )
            return McpMenu(
                title=f"Tools of {server.entry.name}",
                details=f"Exposure {exposure}: {reach}{note}",
                items=items,
                empty="The server offers no tools.",
                confirm_label="back",
                cancel_label="back",
            )

        await ui.menu(build)

    async def _choose_exposure(self, ui: McpManagerView, server: _McpServer) -> str | None:
        current = _exposure_of(server.entry)
        details = (
            f"Applies to this session; the server is registered by {server.entry.source}."
            if server.entry.scope == "extension"
            else f"Saved to {server.entry.override if server.entry.override is not None else server.entry.source}."
        )
        choice = await ui.menu(
            lambda: McpMenu(
                title=f"Exposure of {server.entry.name}",
                details=details,
                items=[
                    {
                        "value": exposure,
                        "label": f"{'✓ ' if exposure == current else '  '}{exposure}",
                        "description": description,
                    }
                    for exposure, description in _EXPOSURE_DESCRIPTIONS.items()
                ],
                selected=current,
                confirm_label="save",
                cancel_label="back",
            )
        )
        if not choice or choice == current:
            return None
        return await self._set_exposure(server, choice)  # type: ignore[arg-type]

    def _sign_in_with_ui(self, ui: McpManagerView, server: _McpServer) -> Awaitable[str | None]:
        """Sign in with the manager view's sign-in screen, which shows the URL
        with a copy key and cancels with the cancel key."""
        title = f"Sign in to {server.entry.name}"
        cancel = CancelToken()
        authorization_url = [""]
        open_url = self._open_url

        def status(message: str) -> None:
            # The cancel key arrives on the input path; `CancelToken.cancel` is synchronous.
            ui.status(title, message, cancel.cancel)

        class Prompt:
            def show_authorization_url(self, url: str) -> None:
                authorization_url[0] = url
                open_url(url)

            async def prompt_for_redirect_url(self, redirect_cancel: Any) -> str | None:
                value = await ui.redirect_url(title, authorization_url[0], redirect_cancel)
                status("Connecting…")
                return value

        status("Contacting the authorization server…")
        return self._sign_in(server, Prompt(), cancel)

    def _run_in_background(
        self,
        ctx: Any,
        server: _McpServer,
        operation: Callable[[Callable[[object | None], None]], Awaitable[str | None]],
    ) -> None:
        """Keep the subscribed menu usable while a connection opens or closes:
        the operation runs detached and tracked, and its result goes to the
        server's line unless the server was disabled, replaced or removed, or
        the session ended, meanwhile.

        "Replaced" is judged against the attempt the operation itself set: it
        reports it through the callback it receives, under the lock. pi reads
        `server.attempt` after `operation()` ran up to its first await, which
        includes that change since pi saves the config synchronously; here the
        save is awaited first. An operation that sets no attempt (reconnect)
        keeps the one current when it started."""
        with self._lock:
            attempt = server.attempt
            session = self._session
        done = self._track()

        def on_attempt(new: object | None) -> None:
            # Called under the lock; read under the lock below.
            nonlocal attempt
            attempt = new

        async def run() -> None:
            try:
                try:
                    message = await operation(on_attempt)
                except Exception as error:
                    message = str(error)
                with self._lock:
                    current = (
                        not session.cancelled
                        and server.attempt is attempt
                        and any(candidate is server for candidate in self._servers)
                    )
                    if current:
                        server.message = message
                if current:
                    self._ensure_discovery_active(ctx)
                    self._emit_change()
            finally:
                self._untrack(done)

        tonio.spawn.without_tracking(run())

    async def _run_action(self, ui: McpManagerView, ctx: Any, server: _McpServer, action: str) -> None:
        with self._lock:
            session = self._session
        message: str | None = None
        match action:
            case "signin":
                message = await self._sign_in_with_ui(ui, server)
            case "reconnect":

                async def reconnect(_on_attempt: Callable[[object | None], None]) -> None:
                    # Connection state and error already report failures, including required sign-ins.
                    await self._reconnect(server)

                self._run_in_background(ctx, server, reconnect)
            case "signout":
                await self._sign_out(server)
            case "tools":
                await self._show_tools(ui, server)
            case "exposure":
                message = await self._choose_exposure(ui, server)
            case "enable" | "disable" | "enable-project" | "disable-project":
                enable = action.startswith("enable")
                in_project = action.endswith("-project")
                self._run_in_background(
                    ctx, server, lambda on_attempt: self._set_enabled(server, enable, in_project, on_attempt)
                )
        # The session ended meanwhile (for example during a sign-in), which made ctx stale.
        if session.cancelled:
            return
        with self._lock:
            server.message = message
        self._ensure_discovery_active(ctx)
        self._emit_change()

    async def _manage(self, ui: McpManagerView, ctx: Any) -> None:
        subscribe: Subscribe = self._subscribe
        while True:
            name = await ui.menu(self._servers_menu, subscribe)
            if not name:
                return
            while True:
                action = await ui.menu(lambda name=name: self._server_menu(name), subscribe)
                server = self._find_server(name)
                if not action or server is None:
                    break
                await self._run_action(ui, ctx, server, action)

    # -----------------------------------------------------------------------
    # Subcommands and plain status
    # -----------------------------------------------------------------------

    def _format_status(self) -> str:
        servers = self._servers
        if not servers and not self._config_errors and not self._overridden:
            return _config_hint()
        lines = []
        for server in servers:
            name = server.entry.name
            exposure = _exposure_of(server.entry)
            view = _view_of(server)
            if view is not None and view.state == "needs-auth":
                lines.append(f"{name}: needs sign-in, run /mcp login {name} ({exposure})")
                continue
            tools = f", {len(view.tools)} tools" if view is not None and view.state == "connected" else ""
            if not _is_enabled(server):
                state = "disabled"
            elif view is not None and view.state == "disconnected":
                state = "disconnected, reconnects on next call"
            else:
                state = view.state if view is not None else "starting"
            error = (
                "\n    " + "\n    ".join(view.error.split("\n"))
                if view is not None and view.error and view.state != "connected"
                else ""
            )
            lines.append(f"{name}: {state}{tools} ({exposure}){error}")
        lines.extend(f"config error: {error}" for error in self._config_errors)
        lines.extend(f"overridden: {line}" for line in self._overridden)
        return "\n".join(lines)

    async def _pick_server(
        self,
        name: str | None,
        ctx: Any,
        *,
        eligible: Callable[[_McpServer], bool],
        preferred: Callable[[_McpServer], bool],
        none: str,
    ) -> _McpServer | None:
        """Resolve the server for a subcommand, asking when the name is omitted and ambiguous."""
        if name:
            server = self._find_server(name)
            if server is None:
                ctx.ui.notify(f'No MCP server named "{name}".', "error")
            elif not eligible(server):
                ctx.ui.notify(none, "error")
            return server if server is not None and eligible(server) else None
        candidates = [server for server in self._servers if eligible(server)]
        if not candidates:
            ctx.ui.notify(none, "info")
            return None
        preferred_candidates = [server for server in candidates if preferred(server)]
        if len(candidates) == 1:
            return candidates[0]
        if len(preferred_candidates) == 1:
            return preferred_candidates[0]
        choice = await ctx.ui.select("MCP server", [server.entry.name for server in candidates])
        return next((server for server in candidates if server.entry.name == choice), None)

    @staticmethod
    def _uses_oauth(server: _McpServer) -> bool:
        connection = server.connection
        return connection is not None and connection.oauth_url is not None

    def _pick_oauth_server(self, name: str | None, ctx: Any) -> Awaitable[_McpServer | None]:
        return self._pick_server(
            name,
            ctx,
            eligible=self._uses_oauth,
            preferred=lambda server: _in_state(server, "needs-auth"),
            none="No enabled MCP server uses OAuth. Only HTTP servers without an Authorization header do.",
        )

    async def _login_command(self, server: _McpServer, ctx: Any) -> None:
        name = server.entry.name
        if not ctx.has_ui:
            ctx.ui.notify(f'Signing in to MCP server "{name}" requires interactive mode.', "error")
            return
        with self._lock:
            session = self._session
        failure: str | None = None
        if ctx.mode == "tui":

            async def manage(ui: McpManagerView) -> None:
                nonlocal failure
                failure = await self._sign_in_with_ui(ui, server)

            await show_mcp_manager(ctx, manage)
        else:
            open_url = self._open_url

            class Prompt:
                def show_authorization_url(self, url: str) -> None:
                    ctx.ui.notify(f'Sign in to MCP server "{name}" in your browser:\n{url}', "info")
                    open_url(url)

                def prompt_for_redirect_url(self, cancel: Any) -> Awaitable[str | None]:
                    return ctx.ui.input(
                        f'Waiting for sign-in to "{name}". If the browser cannot reach this machine, '
                        "paste the URL it was redirected to.",
                        "http://127.0.0.1:.../callback?code=...",
                        {"signal": cancel},
                    )

            failure = await self._sign_in(server, Prompt())
        # The session ended meanwhile, which cancelled the sign-in and made ctx stale.
        if session.cancelled:
            return
        if failure:
            ctx.ui.notify(failure, "info" if failure == "Sign-in cancelled." else "error")
            return
        self._ensure_discovery_active(ctx)
        view = _view_of(server)
        tools = len(view.tools) if view is not None else 0
        ctx.ui.notify(f'Signed in to MCP server "{name}" ({tools} tools).', "info")

    async def _argument_completions(self, prefix: str) -> list[dict[str, str]] | None:
        # As `split(/\s+/)`: a trailing space leaves an empty last word.
        words = re.split(r"\s+", prefix.lstrip())
        action = words[0]
        server = words[1] if len(words) > 1 else None
        if len(words) > 2:
            return None
        if server is None:
            return [
                {"value": f"{item} ", "label": item}
                for item in ("login", "logout", "reconnect")
                if item.startswith(action)
            ]
        if action not in ("login", "logout", "reconnect"):
            return None
        items = [
            {
                "value": f"{action} {candidate.entry.name}",
                "label": candidate.entry.name,
                "description": _describe_state(candidate),
            }
            for candidate in self._servers
            if (candidate.connection is not None if action == "reconnect" else self._uses_oauth(candidate))
            and candidate.entry.name.startswith(server)
        ]
        return items or None

    async def _wait_for_startup(self) -> None:
        pending = self._pending
        if pending is not None:
            await pending.wait()

    async def _command(self, args: str, ctx: Any) -> None:
        words = args.split()
        action = words[0] if words else None
        name = words[1] if len(words) > 1 else None
        if action is None:
            # The manager opens before the startup connections finish; it shows them as they settle.
            if ctx.mode == "tui":
                await show_mcp_manager(ctx, lambda ui: self._manage(ui, ctx))
            else:
                await self._wait_for_startup()
                ctx.ui.notify(self._format_status(), "info")
            return
        if len(words) > 2:
            ctx.ui.notify(_MCP_USAGE, "warning")
            return
        await self._wait_for_startup()
        match action:
            case "login":
                server = await self._pick_oauth_server(name, ctx)
                if server is not None:
                    await self._login_command(server, ctx)
            case "logout":
                server = await self._pick_oauth_server(name, ctx)
                if server is None:
                    return
                removed = await self._sign_out(server)
                ctx.ui.notify(
                    f'Signed out of MCP server "{server.entry.name}".'
                    if removed
                    else f'No stored credentials for MCP server "{server.entry.name}".',
                    "info",
                )
            case "reconnect":
                server = await self._pick_server(
                    name,
                    ctx,
                    eligible=lambda candidate: candidate.connection is not None,
                    preferred=lambda candidate: _in_state(candidate, "failed", "disconnected"),
                    none="No enabled MCP server to reconnect.",
                )
                if server is None:
                    return
                failure = await self._reconnect(server)
                if failure:
                    ctx.ui.notify(failure, "error")
                else:
                    self._ensure_discovery_active(ctx)
                    ctx.ui.notify(
                        f'Reconnected to MCP server "{server.entry.name}" ({_describe_state(server)}).', "info"
                    )
            case _:
                ctx.ui.notify(_MCP_USAGE, "warning")

    # -----------------------------------------------------------------------
    # Session events
    # -----------------------------------------------------------------------

    async def _on_session_start(self, _event: Any, ctx: Any) -> None:
        loaded = await self._load_config(ctx)
        with self._lock:
            self._config_errors = loaded.errors
            self._project_config = loaded.project_config
            self._auto_enable_codemode = loaded.auto_enable_codemode is not False
            self._warned_unreachable = False
            self._waited_for_startup = False
            self._session_cwd = ctx.cwd
            self._model_registry = ctx.model_registry
            self._session = session = CancelToken()
            self._configured_entries = loaded.servers
            registered, overridden = self._registered_servers_locked()
            self._overridden = overridden
            self._servers = (*(_McpServer(entry) for entry in loaded.servers), *registered)
            servers = self._servers
        self._emit_change()
        # Codemode or tool_search is activated from the config: the first prompt
        # does not wait for servers whose tools are not declared to the model,
        # and scripts or searches wait for them.
        self._ensure_discovery_active(ctx)
        enabled = [server for server in servers if _is_enabled(server)]
        if not enabled:
            self._report_problems(ctx)
            return

        readies = [self._start_connection(server) for server in enabled]
        pending = tonio.Event()
        with self._lock:
            self._pending = pending

        async def report_when_ready() -> None:
            try:
                for ready in readies:
                    await ready.wait()
                if not session.cancelled:
                    self._report_problems(ctx)
            except Exception:
                # The session may have been disposed meanwhile, which makes ctx stale.
                pass
            finally:
                pending.set()

        tonio.spawn.without_tracking(report_when_ready())

    async def _wait_for_direct_servers(self, ctx: Any) -> None:
        """The first prompt waits for servers whose tools are declared to the
        model, so they are declared in its first request, but not indefinitely:
        a slow or hanging server must not hold up the prompt. Other servers are
        waited for when a script or search needs them."""
        with self._lock:
            if self._waited_for_startup:
                return
            self._waited_for_startup = True
            ready = [
                server.ready
                for server in self._servers
                if _is_enabled(server) and _has_direct_tools(server.entry) and server.ready is not None
            ]
        if not ready:
            return
        deadline = clock.monotonic() + self._startup_wait_ms / 1000
        for event in ready:
            await event.wait(max(0.0, deadline - clock.monotonic()))
        if not all(event.is_set() for event in ready):
            ctx.ui.notify("MCP servers are still connecting; their tools become available once connected.", "info")

    async def _on_before_agent_start(self, event: Any, ctx: Any) -> None:
        """Every prompt lists the servers in the `mcp_servers` section as they
        are when it starts. pidrei appends the section to the conversation when
        it changed, for example after a server connected."""
        await self._wait_for_direct_servers(ctx)
        sections = event["systemPromptOptions"].sections
        section = render_servers_section(self._servers)
        if section:
            sections[MCP_SERVERS_SECTION] = section
        else:
            sections.pop(MCP_SERVERS_SECTION, None)

    async def _on_tool_call(self, event: Any, ctx: Any) -> None:
        """A codemode script waits for the servers it names, or for every
        server when it searches or enumerates tools, so their tools are
        registered before the script runs. tool_search and the resource tools
        reach every server, so they wait for all of them. Direct MCP calls wait
        only for their own server, including a reconnect still closing its old
        transport."""
        tool = next((candidate for candidate in self._pi.get_all_tools() if candidate.name == event["toolName"]), None)
        if tool is None:
            return
        # Readiness can be pending while the display state is still connected: a reconnect
        # first closes the old transport, then opens a new one. Set Events return at once.
        ready_servers = [server for server in self._servers if _is_enabled(server) and server.ready is not None]
        if not ready_servers:
            return
        waiting: list[_McpServer] = []
        if is_codemode_tool(tool):
            code = (event.get("input") or {}).get("code")
            source = code if isinstance(code, str) else ""
            waiting = [server for server in ready_servers if _script_needs_server(source, server.entry.name)]
        elif is_tool_search_tool(tool) or tool.name in _RESOURCE_TOOL_NAMES:
            waiting = ready_servers
        else:
            with self._lock:
                owner = self._tool_owners.get(event["toolName"], "").split("\0", 1)[0]
            waiting = [server for server in ready_servers if server.entry.name == owner]
        await self._wait_for_servers(waiting, ctx.signal)

    async def _on_turn_start(self, _event: Any, ctx: Any) -> None:
        """Pick up sign-ins done outside the session, such as `pidrei mcp login` run by the agent."""
        if self._tokens_at_sign_in:
            await self._reconnect_signed_in(ctx)

    async def _on_mcp_servers_change(self, _event: Any, ctx: Any) -> None:
        """Servers registered or unregistered during the session connect or disconnect right away."""
        with self._lock:
            session = self._session
            if session.cancelled:
                return
            registered, overridden = self._registered_servers_locked()
            self._overridden = overridden
            upcoming = {server.entry.name: server.registered_config for server in registered}
            # Unregistered servers and re-registered ones with a new config are
            # dropped; the latter come back below.
            removed = [
                server
                for server in self._servers
                if server.entry.scope == "extension" and upcoming.get(server.entry.name) != server.registered_config
            ]
            kept = [server for server in self._servers if server not in removed]
            names = {server.entry.name for server in kept}
            added = [server for server in registered if server.entry.name not in names]
            self._servers = (*kept, *added)
            for server in removed:
                self._hide_tools_locked(server.entry.name)
        self._emit_change()
        self._ensure_discovery_active(ctx)
        await _wait_all([connection.close() for server in removed if (connection := server.connection) is not None])
        connecting = [server for server in added if _is_enabled(server)]
        if session.cancelled or not connecting:
            return
        readies = [self._start_connection(server) for server in connecting]
        for ready in readies:
            await ready.wait()
        if session.cancelled:
            await _wait_all(
                [connection.close() for server in connecting if (connection := server.connection) is not None]
            )
            return
        self._report_problems(ctx, connecting)

    async def _on_session_shutdown(self, _event: Any, _ctx: Any) -> None:
        with self._lock:
            session = self._session
            closing = self._connections()
            self._servers = ()
        # Outside the lock: cancelling runs the token's callbacks (sign-ins in flight) synchronously.
        session.cancel()
        self._emit_change()
        # Tracked work (aborted sign-ins, manager actions) is joined after the cancel, so it is cleaning up.
        with self._lock:
            tracked = list(self._background_actions)
        await _wait_all([*(done.wait() for done in tracked), *(connection.close() for connection in closing)])


extension = create_mcp_extension()
