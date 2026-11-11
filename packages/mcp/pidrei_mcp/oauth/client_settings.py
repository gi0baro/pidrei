"""Mirror of pi mcp src/oauth/client-settings.ts.

The OAuth client that sign-in and refreshes use for one MCP server: its
settings, how it identifies itself (a pre-registered client, a Client ID
Metadata Document, or dynamic client registration), and the
`McpOAuthProvider` built from them.

pi's `timedFetch` adds an `AbortSignal.timeout` to each request; here each
request gets the fetch's own `timeout_ms`.
"""

import base64
import hashlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from ..fetch import McpFetch, McpResponse, default_fetch
from ..url import parse_url
from .flow import OAuthClientMetadataDocument
from .provider import McpOAuthProvider, McpOAuthStateStore
from .types import AuthorizationServerMetadata, OAuthClientInformationMixed


@dataclass(frozen=True, slots=True)
class McpOAuthSettings:
    """Client settings of one MCP server, usually from the application's configuration."""

    # Pre-registered client id. Without it, sign-in uses a Client ID Metadata Document or registers a client.
    client_id: str | None = None
    # Already resolved.
    client_secret: str | None = None
    # Port of the loopback callback server, for clients registered with a
    # fixed redirect URI. Without `callback_url`, the redirect URI is
    # `http://127.0.0.1:<port>/callback`. Only `listen_for_loopback_sign_in`
    # and the redirect URI of refreshes use it; `start_mcp_sign_in` and
    # `complete_mcp_sign_in` take theirs from the caller.
    callback_port: int | None = None
    # Loopback redirect URI registered for `client_id`, for example
    # `http://localhost:8080/oauth/callback`. Without a port, the callback
    # server listens on `callback_port` or a free port, which is added to the
    # URI (RFC 8252). Default: `http://127.0.0.1/callback`. Like
    # `callback_port`, only `listen_for_loopback_sign_in` and refreshes use it.
    callback_url: str | None = None
    # Scopes to request, separated by spaces. Default: the scopes the server advertises.
    scope: str | None = None
    # `client_name` for dynamic client registration. Default: the application's `client_name`.
    client_name: str | None = None
    # How to identify without `client_id`. `auto` (default): a Client ID
    # Metadata Document when the authorization server supports them for
    # public clients and the documents list the redirect URI, otherwise
    # dynamic client registration; with `client_name`, always dynamic client
    # registration. `cimd`: always the document, and sign-in fails without
    # support. `dcr`: always dynamic client registration.
    client_registration: str | None = None
    # Authorization server metadata document to use instead of discovery
    # through the server, for servers that advertise a wrong authorization
    # server or none. It is trusted as configured.
    auth_server_metadata_url: str | None = None


@dataclass(frozen=True, slots=True)
class McpClientMetadataDocuments:
    """The Client ID Metadata Documents an application publishes:
    `<base_url>/client.json`, and for each MCP server
    `<base_url>/<callback ID>/client.json`, whose redirect URIs carry the path
    `<path>/<callback ID>`."""

    # For example `https://pi.dev/oauth`.
    base_url: str
    # Whether the documents list a redirect URI. Loopback redirect URIs match on any port (RFC 8252).
    lists_redirect_url: Callable[[str], bool]


class McpOAuthServerStore(McpOAuthStateStore, Protocol):
    """State store of one MCP server, with a lock that keeps other processes
    from refreshing at the same time."""

    def with_refresh_lock[T](self, fn: Callable[[], Awaitable[T]]) -> Awaitable[T]:
        """Run `fn` while no other process refreshes the server's tokens. Not
        reentrant: `start_mcp_sign_in` and `create_mcp_auth_provider` take it
        themselves, so never call them from `fn`. Raise
        `McpRefreshLockBusyError` when another process holds the lock for the
        whole wait: `start_mcp_sign_in` then goes to the browser step. Any
        other error fails the sign-in or refresh."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class McpOAuthClientOptions:
    """The OAuth client of one MCP server: where its state lives, its
    settings, and the application's identity."""

    server_url: str
    store: McpOAuthServerStore
    settings: McpOAuthSettings
    # `client_name` for dynamic client registration unless `settings.client_name` names another: the application.
    client_name: str
    # None: dynamic client registration only.
    client_metadata_documents: McpClientMetadataDocuments | None = None


async def _ignore_redirect(_url: str) -> None:
    pass


def create_oauth_client_provider(
    options: McpOAuthClientOptions,
    redirect_url: str,
    on_redirect: Callable[[str], Awaitable[None]] = _ignore_redirect,
) -> McpOAuthProvider:
    """An `McpOAuthProvider` for the client, redirecting to `redirect_url` or the document's variant of it."""
    server_url, settings, documents = options.server_url, options.settings, options.client_metadata_documents

    def client_metadata_document(metadata: AuthorizationServerMetadata | None) -> OAuthClientMetadataDocument | None:
        if documents is None or not documents.lists_redirect_url(redirect_url):
            raise RuntimeError(
                'oauth.clientRegistration "cimd" needs Client ID Metadata Documents that list the redirect URI '
                f"{redirect_url}"
            )
        document = _client_metadata_document(server_url, redirect_url, metadata, documents.base_url)
        if document is None and settings.client_registration == "cimd":
            raise RuntimeError(
                "The authorization server does not support Client ID Metadata Documents for public clients; "
                'remove oauth.clientRegistration "cimd"'
            )
        return document

    return McpOAuthProvider(
        server_url=server_url,
        redirect_url=redirect_url,
        client_metadata={
            "client_name": settings.client_name if settings.client_name is not None else options.client_name
        },
        client_metadata_document=(
            client_metadata_document if offers_client_metadata_document(settings, documents, redirect_url) else None
        ),
        client_id=settings.client_id,
        client_secret=settings.client_secret,
        store=options.store,
        on_redirect=on_redirect,
    )


def offers_client_metadata_document(
    settings: McpOAuthSettings, documents: McpClientMetadataDocuments | None, redirect_url: str
) -> bool:
    """Whether the client may identify with a Client ID Metadata Document
    instead of registering. `auto` registers when the user asked for a client
    name or the documents do not list the redirect URI."""
    registration = settings.client_registration if settings.client_registration is not None else "auto"
    if registration != "auto":
        return registration == "cimd"
    return (
        not settings.client_id
        and not settings.client_name
        and documents is not None
        and documents.lists_redirect_url(redirect_url)
    )


def _client_metadata_document(
    server_url: str, redirect_url: str, metadata: AuthorizationServerMetadata | None, base_url: str
) -> OAuthClientMetadataDocument | None:
    """The Client ID Metadata Document to use, chosen like Codex chooses its
    own, or None when the authorization server does not support documents for
    public clients. Without the `iss` parameter in authorization responses
    (RFC 9207), the redirect URI and the document are specific to the MCP
    server, so a response cannot be mixed up with one from another
    authorization server (RFC 9700 section 4.4.2.2)."""
    if (
        metadata is None
        or not metadata.get("client_id_metadata_document_supported")
        or "none" not in (metadata.get("token_endpoint_auth_methods_supported") or [])
    ):
        return None
    if metadata.get("authorization_response_iss_parameter_supported"):
        return OAuthClientMetadataDocument(url=f"{base_url}/client.json", redirect_url=redirect_url)
    redirect = parse_url(redirect_url)
    redirect = replace(redirect, pathname=server_specific_path(redirect.pathname, server_url))
    return OAuthClientMetadataDocument(
        url=f"{base_url}/{_callback_id(server_url)}/client.json", redirect_url=redirect.href
    )


def server_specific_path(path: str, server_url: str) -> str:
    """The path of the redirect URI of a server-specific Client ID Metadata Document: `<path>/<callback ID>`."""
    return f"{path.removesuffix('/')}/{_callback_id(server_url)}"


def _callback_id(server_url: str) -> str:
    """12 characters identifying an MCP server URL in callback paths, computed like Codex does."""
    href = parse_url(server_url).without_fragment().href
    return base64.urlsafe_b64encode(hashlib.sha256(href.encode()).digest()[:9]).decode().rstrip("=")


def registered_redirect_urls(client: OAuthClientInformationMixed | None) -> list[str]:
    """Redirect URIs a stored client was registered with."""
    return list(client.get("redirect_uris") or []) if client else []


# Bounds each request to the authorization server, so an unresponsive one cannot hold the refresh lock,
# delay shutdown, or keep a sign-in waiting for long.
OAUTH_REQUEST_TIMEOUT_MS = 15_000


def timed_fetch(fetch: McpFetch | None = None) -> McpFetch:
    """`fetch` with `OAUTH_REQUEST_TIMEOUT_MS` per request."""
    inner = fetch if fetch is not None else default_fetch

    def timed(
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        body: bytes | None = None,
        timeout_ms: float | None = None,
    ) -> Awaitable[McpResponse]:
        return inner(url, method=method, headers=headers, body=body, timeout_ms=OAUTH_REQUEST_TIMEOUT_MS)

    return timed


_CALLBACK_HOST = "127.0.0.1"
_CALLBACK_PATH = "/callback"

# Redirect URI when the settings name none and no client is stored.
DEFAULT_REDIRECT_URL = f"http://{_CALLBACK_HOST}{_CALLBACK_PATH}"


@dataclass(frozen=True, slots=True)
class LoopbackSettings:
    """Where the loopback callback server listens and the redirect URI it
    serves, from `callback_url` and `callback_port`."""

    # The redirect URI, without the port unless one is configured.
    url: str
    # Address to listen on.
    host: str
    # Host name in the redirect URI.
    redirect_host: str
    port: int | None
    path: str
    # The exact redirect URI, when the port is fixed.
    fixed_redirect_url: str | None


def loopback_settings(settings: McpOAuthSettings) -> LoopbackSettings:
    """The loopback callback settings of `settings`."""
    configured = settings.callback_url if settings.callback_url is not None else DEFAULT_REDIRECT_URL
    url = parse_url(configured)
    address = url.hostname.removeprefix("[").removesuffix("]")
    port = int(url.port) if url.port else settings.callback_port
    fixed_redirect_url: str | None = None
    # A configured URI with a port is sent exactly as written, since servers compare it as a string.
    if url.port:
        fixed_redirect_url = settings.callback_url
    elif port is not None:
        fixed_redirect_url = parse_url(f"{url.scheme}://{url.hostname}:{port}{url.pathname}").href
    return LoopbackSettings(
        url=configured,
        # `localhost` is served on 127.0.0.1; browsers fall back to it when ::1 refuses.
        host=_CALLBACK_HOST if address == "localhost" else address,
        redirect_host=address,
        port=port,
        path=url.pathname,
        fixed_redirect_url=fixed_redirect_url,
    )
