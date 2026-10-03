"""Mirror of pi coding-agent src/extensions/mcp/runtime.ts: the part of the
MCP integration that talks to servers: connections, transports, and OAuth
sign-in.

pi loads this module lazily (`runtime.lazy.ts`) so sessions without servers
never load the MCP client; here it is imported with the extension, since a
lazy import would move the import itself onto a runtime worker.

What diverges from pi's shape, for a runtime where a connection's calls,
refreshes and close run in parallel:

- What the extension reads of a connection (state, error, tools, resources,
  instructions) is one frozen snapshot, replaced whole under the connection's
  lock; readers take one copy.
- pi shares one `opening` promise between callers. Here the open runs on its
  own detached coroutine and callers join it through an Event, so a caller
  that is cancelled stops waiting while the open completes for the others.
- `close()` also closes the client still connecting, and wakes a retry
  delay: pi leaves a connect in flight running until it fails or times out,
  which here would be work still running after shutdown.
- `on_change` is async and awaited where the state changes, so the extension
  can read what it needs (the stored tokens) at that moment.
- Factories and settings that resolve config values (`!command`, `${VAR}`)
  are async: resolving runs a command or reads the environment.
"""

import os
import pathlib
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import tonio.colored as tonio

from pidrei_http.http import TransportError
from pidrei_mcp import (
    JSON_RPC_ERROR_CODES,
    AuthProvider,
    CallToolResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    McpAuthRequiredError,
    McpClient,
    McpError,
    McpHttpError,
    McpSessionExpiredError,
    McpTransport,
    ReadResourceResult,
    Resource,
    ResourceTemplate,
    StdioTransport,
    StreamableHttpTransport,
    Tool,
)
from pidrei_mcp.oauth import McpOAuthAuthorizationRequiredError, OAuthChallenge
from pidrei_utils.cancel import CancelToken

from ...config import VERSION
from ...core.resolve_config_value import resolve_config_value_or_throw, resolve_headers_or_throw
from .config import McpServerEntry
from .log import McpServerLog
from .oauth import (
    McpOAuthCredentialStore,
    McpOAuthSettings,
    McpSignInCancelledError,
    create_mcp_auth_provider,
    sign_in_mcp_server,
)
from .resources import is_mcp_app_resource


__all__ = [
    "McpOAuthCredentialStore",
    "McpServerConnection",
    "McpServerLog",
    "McpSignInCancelledError",
    "McpTransportFactory",
    "create_default_transport",
    "sign_in_mcp_server",
]

_DEFAULT_TIMEOUT_SECONDS = 60
_STDERR_TAIL_CHARS = 2_000
# Delays between attempts to connect to an HTTP server that failed with a transient error.
_CONNECT_RETRY_DELAYS_MS = (250, 1_000)

# `disconnected`: the connection dropped (for example the stdio server
# exited); the next call reconnects.
type ServerState = Literal["connecting", "connected", "disconnected", "needs-auth", "failed", "closed"]

type McpTransportFactory = Callable[[McpServerEntry, str, AuthProvider | None], Awaitable[McpTransport]]


def _is_transient_error(error: BaseException) -> bool:
    """Network failures and overloaded or restarting servers, which are worth another attempt."""
    if isinstance(error, McpHttpError):
        return error.status in (408, 429) or (error.status >= 500 and error.status != 501)
    # fetch's TypeError: the exchange itself failed.
    return isinstance(error, TransportError)


def _sign_in_required_message(entry: McpServerEntry) -> str:
    provider = (entry.config.get("auth") or {}).get("provider") if "url" in entry.config else None
    run = f"/login {provider}" if provider else "/mcp"
    return f'MCP server "{entry.name}" requires sign-in. Run {run} to sign in.'


def _uses_oauth(entry: McpServerEntry) -> bool:
    """HTTP servers authenticate with OAuth unless the config supplies an `Authorization` header or `auth`."""
    config = entry.config
    if "url" not in config or config.get("auth"):
        return False
    return not any(header.lower() == "authorization" for header in config.get("headers") or {})


def _expand_home(value: str) -> str:
    """`~` and `~/…` name the home directory, like in a shell."""
    if value == "~":
        return os.path.expanduser("~")
    if value.startswith("~/"):
        return os.path.join(os.path.expanduser("~"), value[2:])
    return value


async def create_default_transport(entry: McpServerEntry, cwd: str, auth_provider: AuthProvider | None) -> McpTransport:
    config, name = entry.config, entry.name
    if "url" in config:
        return StreamableHttpTransport(
            config["url"],
            headers=await resolve_headers_or_throw(config.get("headers"), f'MCP server "{name}"'),
            auth_provider=auth_provider,
        )
    env: dict[str, str] = {}
    for key, value in (config.get("env") or {}).items():
        env[key] = await resolve_config_value_or_throw(value, f'MCP server "{name}" env "{key}"')
    args = config.get("args")
    return StdioTransport(
        _expand_home(config["command"]),
        args=[_expand_home(arg) for arg in args] if args is not None else None,
        cwd=os.path.normpath(os.path.join(cwd, _expand_home(config.get("cwd") or "."))),
        env=env,
        stderr="pipe",
    )


async def _without_templates[T](list_templates: Callable[[], Awaitable[T]], empty: T) -> T:
    """Servers that do not implement `resources/templates/list` have no templates."""
    try:
        return await list_templates()
    except McpError as error:
        if error.code == JSON_RPC_ERROR_CODES.method_not_found:
            return empty
        raise


def _list_templates(
    client: McpClient, cancel: CancelToken | None = None, timeout_ms: float | None = None
) -> Awaitable[list[ResourceTemplate]]:
    return _without_templates(lambda: client.list_resource_templates(cancel=cancel, timeout_ms=timeout_ms), [])


async def _quietly[T](operation: Awaitable[list[T]]) -> list[T]:
    try:
        return await operation
    except Exception:
        return []


async def _fetch_resources(client: McpClient) -> tuple[list[Resource], list[ResourceTemplate]]:
    """Resources and templates at connect time, for the counts in `/mcp` and
    `pidrei mcp list`. A server whose lists fail still connects: the resource
    tools list and read its resources on demand."""
    templates = tonio.spawn(_quietly(_list_templates(client)))
    resources = await _quietly(client.list_resources())
    resource_templates = await templates
    return (
        [resource for resource in resources if not is_mcp_app_resource(resource)],
        [template for template in resource_templates if not is_mcp_app_resource(template)],
    )


@dataclass(frozen=True, slots=True)
class _ConnectionView:
    """Published whole; the lists are never changed after publication."""

    state: ServerState = "connecting"
    error: str | None = None
    tools: list[Tool] = field(default_factory=list)
    # Whether the server offers resources. The lists below are what it listed
    # at the last connect or change, without MCP App resources.
    has_resources: bool = False
    resources: list[Resource] = field(default_factory=list)
    resource_templates: list[ResourceTemplate] = field(default_factory=list)
    # Server instructions from `initialize`, describing its tools as a group.
    instructions: str | None = None


class _Opening:
    """An open in flight: `client` or `error` is set before `done`."""

    __slots__ = ("client", "done", "error")

    def __init__(self) -> None:
        self.done = tonio.Event()
        self.client: McpClient | None = None
        self.error: Exception | None = None

    async def join(self) -> McpClient:
        await self.done.wait()
        if self.error is not None:
            raise self.error
        return self.client  # type: ignore[return-value]


class McpServerConnection:
    """One configured server. Reconnects lazily when a call finds the connection gone."""

    def __init__(
        self,
        *,
        entry: McpServerEntry,
        cwd: str,
        create_transport: McpTransportFactory,
        credentials: McpOAuthCredentialStore,
        on_tools: Callable[[McpServerConnection], None],
        # The current token of a pidrei provider, for servers with `auth.provider`.
        provider_token: Callable[[str], Awaitable[str | None]] | None = None,
        # Called when `state`, `error`, or `tools` change.
        on_change: Callable[[McpServerConnection], Awaitable[None]] | None = None,
        # Receives the server's log messages (`notifications/message`).
        log: McpServerLog | None = None,
    ) -> None:
        self.entry = entry
        self._cwd = cwd
        self._create_transport = create_transport
        self._on_tools = on_tools
        self._on_change = on_change
        self._log = log
        self._lock = threading.Lock()
        self._view = _ConnectionView()
        # Last OAuth challenge from the server; sign-in uses its resource
        # metadata URL and scope. One value, rebound whole.
        self.challenge: OAuthChallenge | None = None
        self._client: McpClient | None = None
        # The client of the connect in flight, which `close()` closes too.
        self._connecting: McpClient | None = None
        self._opening: _Opening | None = None
        self._closed = False
        self._closed_event = tonio.Event()
        # Stderr of the last stdio server that failed to connect.
        self._stderr_tail: str | None = None
        self._auth_settled: Callable[[], Awaitable[None]] | None = None
        url = self.oauth_url
        provider = (entry.config.get("auth") or {}).get("provider") if "url" in entry.config else None
        self._auth_provider: AuthProvider | None = None
        if url is not None:
            oauth_provider = create_mcp_auth_provider(
                server_url=url,
                store=credentials.for_server(entry.name, url),
                settings=self.oauth_settings,
                on_challenge=self._set_challenge,
            )
            self._auth_provider = oauth_provider
            self._auth_settled = oauth_provider.settled
        elif provider:

            async def token() -> str | None:
                # Read on every request, so the provider's refreshes apply; MCP stores no copy.
                return await provider_token(provider) if provider_token is not None else None

            self._auth_provider = AuthProvider(token=token)

    @property
    def name(self) -> str:
        return self.entry.name

    @property
    def timeout_ms(self) -> float:
        timeout = self.entry.config.get("timeout")
        return (timeout if timeout is not None else _DEFAULT_TIMEOUT_SECONDS) * 1000

    @property
    def oauth_url(self) -> str | None:
        """Server URL when the server authenticates with OAuth."""
        return self.entry.config["url"] if _uses_oauth(self.entry) else None

    @property
    def state(self) -> ServerState:
        return self._view.state

    @property
    def error(self) -> str | None:
        return self._view.error

    @property
    def tools(self) -> list[Tool]:
        return self._view.tools

    @property
    def has_resources(self) -> bool:
        return self._view.has_resources

    @property
    def resources(self) -> list[Resource]:
        return self._view.resources

    @property
    def resource_templates(self) -> list[ResourceTemplate]:
        return self._view.resource_templates

    @property
    def instructions(self) -> str | None:
        return self._view.instructions

    def _set_challenge(self, challenge: OAuthChallenge) -> None:
        self.challenge = challenge

    async def oauth_settings(self) -> McpOAuthSettings:
        oauth = self.entry.config.get("oauth") if "url" in self.entry.config else None
        if not oauth:
            return McpOAuthSettings()
        client_secret = oauth.get("clientSecret")
        return McpOAuthSettings(
            client_id=oauth.get("clientId"),
            client_secret=(
                None
                if client_secret is None
                else await resolve_config_value_or_throw(
                    client_secret, f'MCP server "{self.entry.name}" oauth.clientSecret'
                )
            ),
            callback_port=oauth.get("callbackPort"),
            callback_url=oauth.get("callbackUrl"),
            scope=oauth.get("scope"),
            client_name=oauth.get("clientName"),
            auth_server_metadata_url=oauth.get("authServerMetadataUrl") or None,
        )

    def _publish_locked(self, **changes: Any) -> None:
        self._view = replace(self._view, **changes)

    async def _changed(self) -> None:
        if self._on_change is not None:
            await self._on_change(self)

    async def get_client(self) -> McpClient:
        start = False
        with self._lock:
            if self._closed:
                raise Exception(f'MCP server "{self.entry.name}" is shut down')
            client = self._client
            if client is not None and client.connection_state == "connected":
                return client
            opening = self._opening
            if opening is None:
                opening = self._opening = _Opening()
                start = True
        if start:
            tonio.spawn.without_tracking(self._run_open(opening))
        return await opening.join()

    async def _run_open(self, opening: _Opening) -> None:
        try:
            opening.client = await self._open()
        except Exception as error:
            opening.error = error
        finally:
            with self._lock:
                if self._opening is opening:
                    self._opening = None
            opening.done.set()

    def call_tool(
        self,
        name: str,
        args: dict[str, Any],
        *,
        cancel: CancelToken | None = None,
        timeout_ms: float | None = None,
        on_progress: Callable[[Any], Awaitable[None]] | None = None,
    ) -> Awaitable[CallToolResult]:
        return self._with_client(
            lambda client: client.call_tool(name, args, cancel=cancel, timeout_ms=timeout_ms, on_progress=on_progress)
        )

    def read_resource(
        self, uri: str, *, cancel: CancelToken | None = None, timeout_ms: float | None = None
    ) -> Awaitable[ReadResourceResult]:
        return self._with_client(lambda client: client.read_resource(uri, cancel=cancel, timeout_ms=timeout_ms), True)

    def resources_page(
        self, cursor: str | None, *, cancel: CancelToken | None = None, timeout_ms: float | None = None
    ) -> Awaitable[ListResourcesResult]:
        return self._with_client(
            lambda client: client.list_resources_page(cursor, cancel=cancel, timeout_ms=timeout_ms), True
        )

    def resource_templates_page(
        self, cursor: str | None, *, cancel: CancelToken | None = None, timeout_ms: float | None = None
    ) -> Awaitable[ListResourceTemplatesResult]:
        return self._with_client(
            lambda client: _without_templates(
                lambda: client.list_resource_templates_page(cursor, cancel=cancel, timeout_ms=timeout_ms),
                {"resourceTemplates": []},
            ),
            True,
        )

    def all_resources(
        self, *, cancel: CancelToken | None = None, timeout_ms: float | None = None
    ) -> Awaitable[list[Resource]]:
        return self._with_client(lambda client: client.list_resources(cancel=cancel, timeout_ms=timeout_ms), True)

    def all_resource_templates(
        self, *, cancel: CancelToken | None = None, timeout_ms: float | None = None
    ) -> Awaitable[list[ResourceTemplate]]:
        return self._with_client(lambda client: _list_templates(client, cancel, timeout_ms), True)

    async def _with_client[T](self, run: Callable[[McpClient], Awaitable[T]], read_only: bool = False) -> T:
        """Run a request, reconnecting when needed. `read_only` requests are
        retried once after a transient HTTP error; tool calls are not, since
        they may have run."""
        attempt = 1
        while True:
            client = await self.get_client()
            try:
                return await run(client)
            except Exception as error:
                if read_only and attempt == 1 and isinstance(error, McpHttpError) and _is_transient_error(error):
                    # A close cuts the delay short; the next attempt then reports the shutdown.
                    await self._closed_event.wait(_CONNECT_RETRY_DELAYS_MS[0] / 1000)
                    attempt += 1
                    continue
                if isinstance(error, McpSessionExpiredError) and attempt == 1:
                    # The server no longer knows the session (restart, deploy), so it
                    # did not run the request. Retry once on a new session. The old
                    # client is detached but not closed: closing would fail its other
                    # in-flight calls, which instead get the same 404 and retry the
                    # same way.
                    with self._lock:
                        if self._client is client:
                            self._client = None
                    attempt += 1
                    continue
                if not self._needs_sign_in(error):
                    raise
                await self._drop_client(client)
                await self._mark_needs_auth()
                raise Exception(_sign_in_required_message(self.entry)) from None

    async def _wait_for_opening(self) -> None:
        with self._lock:
            opening = self._opening
        if opening is not None:
            await opening.done.wait()

    async def reconnect(self) -> None:
        """Connect again with fresh credentials, for example after signing in."""
        await self._wait_for_opening()
        client = self._client
        if client is not None:
            await self._drop_client(client)
        await self.get_client()

    async def sign_out(self) -> None:
        """Disconnect after the stored credentials were removed."""
        await self._wait_for_opening()
        client = self._client
        if client is not None:
            await self._drop_client(client)
        if not self._closed:
            await self._mark_needs_auth()

    def _needs_sign_in(self, error: BaseException) -> bool:
        """OAuth servers that still reject the request after a refresh need the user to sign in again."""
        return isinstance(error, McpOAuthAuthorizationRequiredError) or (
            self._auth_provider is not None and isinstance(error, McpAuthRequiredError)
        )

    async def _mark_needs_auth(self) -> None:
        with self._lock:
            self._publish_locked(state="needs-auth", error=None)
        await self._changed()

    async def _drop_client(self, client: McpClient) -> None:
        with self._lock:
            if self._client is client:
                self._client = None
        await _close_quietly(client)

    async def _open(self) -> McpClient:
        with self._lock:
            self._publish_locked(state="connecting")
        await self._changed()
        retries = _CONNECT_RETRY_DELAYS_MS if "url" in self.entry.config else ()
        attempt = 0
        while True:
            self._stderr_tail = None
            try:
                return await self._connect_once()
            except Exception as error:
                delay = retries[attempt] if attempt < len(retries) else None
                if self._closed or delay is None or not _is_transient_error(error):
                    raise await self._connect_failed(error) from None
                await self._closed_event.wait(delay / 1000)
                if self._closed:
                    raise await self._connect_failed(error) from None
            attempt += 1

    async def _connect_once(self) -> McpClient:
        client = McpClient(
            name="pidrei",
            version=VERSION,
            request_timeout_ms=self.timeout_ms,
            roots=[{"uri": pathlib.Path(self._cwd).as_uri(), "name": os.path.basename(self._cwd)}],
        )
        log = self._log
        name = self.entry.name
        if log is not None:

            async def on_message(params: Any) -> None:
                await log.write(name, params)

            client.on_notification("notifications/message", on_message)
        with self._lock:
            if self._closed:
                raise Exception("shut down while connecting")
            self._connecting = client
        transport: McpTransport | None = None
        try:
            transport = await self._create_transport(self.entry, self._cwd, self._auth_provider)
            await client.connect(transport)

            async def on_tools_changed(_params: Any) -> None:
                tonio.spawn.without_tracking(self._refresh_tools(client))

            async def on_resources_changed(_params: Any) -> None:
                tonio.spawn.without_tracking(self._refresh_resources(client))

            client.on_notification("notifications/tools/list_changed", on_tools_changed)
            client.on_notification("notifications/resources/list_changed", on_resources_changed)
            stdio = transport if isinstance(transport, StdioTransport) else None

            async def on_close() -> None:
                await self._handle_client_close(client, stdio)

            client.on_close(on_close)
            capabilities = client.server_capabilities or {}
            has_resources = capabilities.get("resources") is not None
            # Servers without the tools capability (prompts or resources only) do not answer tools/list.
            fetching = tonio.spawn(_fetch_resources(client)) if has_resources else None
            # pi tests the capability object, which is truthy even when empty (`"tools": {}`).
            tools = await client.list_tools() if capabilities.get("tools") is not None else []
            resources, resource_templates = await fetching if fetching is not None else ([], [])
            with self._lock:
                if self._closed:
                    raise Exception("shut down while connecting")
                if client.connection_state != "connected":
                    raise Exception("connection closed during setup")
                self._client = client
                self._connecting = None
                self._publish_locked(
                    state="connected",
                    error=None,
                    tools=tools,
                    has_resources=has_resources,
                    resources=resources,
                    resource_templates=resource_templates,
                    instructions=(client.instructions or "").strip() or None,
                )
            self._on_tools(self)
            await self._changed()
            return client
        except Exception:
            with self._lock:
                if self._connecting is client:
                    self._connecting = None
            await _close_quietly(client)
            if isinstance(transport, StdioTransport):
                self._stderr_tail = transport.stderr.strip()[-_STDERR_TAIL_CHARS:] or None
            raise

    async def _connect_failed(self, error: Exception) -> Exception:
        if self._needs_sign_in(error) and not self._closed:
            await self._mark_needs_auth()
            return Exception(_sign_in_required_message(self.entry))
        message = str(error)
        with self._lock:
            text = f"{message}\n{self._stderr_tail}" if self._stderr_tail else message
            self._publish_locked(state="closed" if self._closed else "failed", error=text)
        await self._changed()
        return Exception(f'MCP server "{self.entry.name}" failed to connect: {text}')

    async def _handle_client_close(self, client: McpClient, stdio: StdioTransport | None) -> None:
        """The transport dropped. The next call reconnects; until then the status shows why."""
        with self._lock:
            if self._client is not client or self._closed:
                return
            self._client = None
            stderr = stdio.stderr.strip()[-_STDERR_TAIL_CHARS:] if stdio is not None else ""
            self._publish_locked(
                state="disconnected", error=f"Connection closed\n{stderr}" if stderr else "Connection closed"
            )
        await self._changed()

    async def _refresh_tools(self, client: McpClient) -> None:
        try:
            tools = await client.list_tools()
            with self._lock:
                if self._client is not client or self._closed:
                    return
                self._publish_locked(tools=tools)
            self._on_tools(self)
        except Exception as error:
            with self._lock:
                self._publish_locked(error=f"Failed to refresh tools: {error}")
        await self._changed()

    async def _refresh_resources(self, client: McpClient) -> None:
        resources, resource_templates = await _fetch_resources(client)
        with self._lock:
            if self._client is not client or self._closed:
                return
            self._publish_locked(resources=resources, resource_templates=resource_templates)
        self._on_tools(self)
        await self._changed()

    async def close(self) -> None:
        with self._lock:
            self._closed = True
            self._publish_locked(state="closed")
            clients = [client for client in (self._client, self._connecting) if client is not None]
            self._client = None
            self._connecting = None
        # Wakes a retry delay; the open then fails as closed.
        self._closed_event.set()
        await self._changed()
        for client in clients:
            await _close_quietly(client)
        # A refresh the server already answered may have rotated the refresh
        # token; exiting before the new tokens are saved would lose the grant.
        if self._auth_settled is not None:
            await self._auth_settled()


async def _close_quietly(client: McpClient) -> None:
    try:
        await client.close()
    except Exception:
        pass
