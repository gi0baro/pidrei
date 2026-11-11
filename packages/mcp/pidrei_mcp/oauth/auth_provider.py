"""Mirror of pi mcp src/oauth/auth-provider.ts.

The auth provider for connections to an OAuth-protected MCP server: it
sends the stored access token and refreshes it, but never starts a browser
flow. When the user has to sign in, it fails with
`McpOAuthAuthorizationRequiredError` and the application runs
`start_mcp_sign_in`.

Where pi shares one refresh promise, a refresh here runs on its own detached
coroutine and callers join it through an Event, so a caller that is
cancelled stops waiting while the refresh finishes and saves the tokens for
the others. `on_unauthorized` starts the refresh at the call, as pi's runs up
to its first await there. `settings` is async: resolving a secret may run a
command.
"""

import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import tonio.colored as tonio

from pidrei_utils import clock

from ..auth_provider import AuthProvider, UnauthorizedContext
from ..fetch import McpFetch, default_fetch
from .client_settings import (
    DEFAULT_REDIRECT_URL,
    McpClientMetadataDocuments,
    McpOAuthClientOptions,
    McpOAuthServerStore,
    McpOAuthSettings,
    create_oauth_client_provider,
    loopback_settings,
    registered_redirect_urls,
    timed_fetch,
)
from .discovery import parse_www_authenticate
from .errors import McpOAuthAuthorizationRequiredError
from .flow import OAuthFlowOptions, OAuthFlowResult, authorize_mcp
from .provider import McpOAuthState
from .types import OAuthChallenge


@dataclass(frozen=True, slots=True, kw_only=True)
class McpAuthProvider(AuthProvider):
    """An `AuthProvider` that can say when its refreshes have finished."""

    # Resolves when no refresh is running, so shutdown does not drop rotated
    # tokens before they are saved.
    settled: Callable[[], Awaitable[None]]


# Access tokens this close to expiry are refreshed before they are sent.
_REFRESH_SKEW_MS = 30_000


class _Refresh:
    """One refresh in flight. `error` is set before `done`."""

    __slots__ = ("done", "error")

    def __init__(self) -> None:
        self.done = tonio.Event()
        self.error: Exception | None = None


def create_mcp_auth_provider(
    *,
    server_url: str,
    store: McpOAuthServerStore,
    settings: Callable[[], Awaitable[McpOAuthSettings]],
    on_challenge: Callable[[OAuthChallenge], None],
    client_name: str,
    client_metadata_documents: McpClientMetadataDocuments | None = None,
    redirect_url: str | None = None,
) -> McpAuthProvider:
    """Auth provider for MCP connections: sends the stored access token and
    refreshes it when it is about to expire or after a 401. Raises
    `McpOAuthAuthorizationRequiredError` when the user has to sign in,
    including when the server asks for more scope (`insufficient_scope`).
    `on_challenge` receives the server's `WWW-Authenticate` challenge so
    sign-in can use its resource metadata URL and scope. `settings` is only
    called when a refresh is needed, so a secret that fails to resolve fails
    the refresh instead of the whole connection setup.

    `redirect_url` is a redirect URI the application signs in with, such as
    its https callback route. Refreshing tokens from a sign-in with a Client
    ID Metadata Document needs a redirect URI the documents list; a refresh
    never redirects, so any listed one works. Default: the loopback redirect
    URI of `settings`.

    Many servers rotate refresh tokens, so two refreshes with the same refresh
    token lose the grant. Requests in this process share one refresh, and
    other processes are kept out by the store's refresh lock, held from
    reading the tokens to saving new ones. Tokens that changed meanwhile
    (another process refreshed them, or the user signed in) are used without
    refreshing."""
    guard = threading.Lock()
    current: list[_Refresh] = []

    async def run_refresh(
        stale_token: str | None, fetch: McpFetch, challenge: OAuthChallenge | None, refresh: _Refresh
    ) -> None:
        async def locked() -> None:
            state = await store.load()
            tokens = (state or {}).get("tokens") or {}
            if tokens.get("access_token") != stale_token:
                return
            if state is None or not tokens.get("refresh_token"):
                raise McpOAuthAuthorizationRequiredError()
            client = McpOAuthClientOptions(
                server_url=server_url,
                store=store,
                settings=await settings(),
                client_name=client_name,
                client_metadata_documents=client_metadata_documents,
            )
            result = await refresh_stored_tokens(
                client, state, fetch=fetch, challenge=challenge, redirect_url=redirect_url
            )
            if result == "REDIRECT":
                raise McpOAuthAuthorizationRequiredError()

        try:
            await store.with_refresh_lock(locked)
        except Exception as error:
            refresh.error = error
        finally:
            with guard:
                current.clear()
            refresh.done.set()

    def start_refresh(stale_token: str | None, fetch: McpFetch, challenge: OAuthChallenge | None = None) -> _Refresh:
        """Replace `stale_token`, the access token that expired or was rejected."""
        with guard:
            if current:
                return current[0]
            refresh = _Refresh()
            current.append(refresh)
        tonio.spawn.without_tracking(run_refresh(stale_token, fetch, challenge, refresh))
        return refresh

    async def join(refresh: _Refresh) -> None:
        await refresh.done.wait()
        if refresh.error is not None:
            raise refresh.error

    async def settled() -> None:
        with guard:
            running = current[0] if current else None
        if running is not None:
            await running.done.wait()

    async def token() -> str | None:
        await settled()
        state = await store.load()
        token = ((state or {}).get("tokens") or {}).get("access_token")
        expires_at = (state or {}).get("tokensExpireAt")
        expired = expires_at is not None and expires_at - _REFRESH_SKEW_MS <= clock.now_ms()
        if not expired or not ((state or {}).get("tokens") or {}).get("refresh_token"):
            return token
        # Failures fall through: the request goes out with the old token and a 401 decides what happens.
        try:
            await join(start_refresh(token, default_fetch))
        except Exception:
            pass
        return (((await store.load()) or {}).get("tokens") or {}).get("access_token")

    def on_unauthorized(context: UnauthorizedContext) -> Awaitable[None]:
        # The refresh starts at the call, as pi's runs up to its first await
        # there: a `settled()` called right after sees it.
        challenge = parse_www_authenticate(context.response.headers.get("www-authenticate"))
        on_challenge(challenge)
        # A refresh keeps the granted scope, so more scope needs a new sign-in.
        if challenge.get("error") == "insufficient_scope":
            raise McpOAuthAuthorizationRequiredError()
        return join(start_refresh(context.token, context.fetch, challenge))

    return McpAuthProvider(token=token, on_unauthorized=on_unauthorized, settled=settled)


async def refresh_stored_tokens(
    client: McpOAuthClientOptions,
    state: McpOAuthState,
    *,
    fetch: McpFetch | None = None,
    challenge: OAuthChallenge | None = None,
    redirect_url: str | None = None,
) -> OAuthFlowResult:
    """Refresh the tokens of `state` with the stored client (or the document
    it signed in with), or report `REDIRECT` when that is not possible.
    `redirect_url` is a redirect URI the application signs in with, tried
    before the loopback ones of the settings and the stored client. Never
    registers a client or prepares an authorization request, so a pending
    sign-in stays usable. Call while holding the store's refresh lock. Not
    cancellable: a refresh the server answered may have rotated the refresh
    token, so its answer must be saved."""
    # Refreshing never redirects the user; the redirect URI only decides whether tokens without a stored
    # client can be refreshed with a Client ID Metadata Document, so any URI the documents list works.
    registered = registered_redirect_urls(state.get("clientInformation"))
    candidates = [
        url
        for url in (
            redirect_url,
            loopback_settings(client.settings).fixed_redirect_url,
            registered[0] if registered else None,
        )
        if url is not None
    ]
    documents = client.client_metadata_documents
    chosen = next(
        (url for url in candidates if documents is not None and documents.lists_redirect_url(url)),
        DEFAULT_REDIRECT_URL,
    )
    return await authorize_mcp(
        create_oauth_client_provider(client, chosen),
        OAuthFlowOptions(
            server_url=client.server_url,
            resource_metadata_url=(challenge or {}).get("resourceMetadataUrl"),
            authorization_server_metadata_url=client.settings.auth_server_metadata_url,
            fetch=timed_fetch(fetch),
            refresh_only=True,
        ),
    )
