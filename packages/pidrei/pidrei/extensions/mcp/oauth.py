"""Mirror of pi coding-agent src/extensions/mcp/oauth.ts: OAuth sign-in for
remote MCP servers.

Connections never start a browser flow on their own. They send the stored
access token and, after a 401, try the stored refresh token. When that is not
possible they fail with `McpOAuthAuthorizationRequiredError`, and the user
signs in through `/mcp`, which runs the authorization code flow (PKCE, dynamic
client registration) against a loopback callback.

Credentials live in `<agent-dir>/mcp-auth.json`, keyed by server name and URL.

Where pi shares one refresh promise, a refresh here runs on its own detached
coroutine and callers join it through an Event, so a caller that is cancelled
stops waiting while the refresh finishes and saves the tokens for the others.
Each request of a refresh is capped by the fetch's own `timeout_ms` (pi's
`AbortSignal.timeout`). The refresh lock is a `FileLock` that renews itself
while held, as proper-lockfile's does.
"""

import hashlib
import json
import os
import sys
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import tonio.colored as tonio
from tonio.colored import fs
from tonio.colored.exceptions import CancelledError

from pidrei_ai.utils.oauth_page import oauth_error_html, oauth_success_html
from pidrei_mcp import AuthProvider, McpFetch, McpResponse, UnauthorizedContext, default_fetch
from pidrei_mcp.oauth import (
    McpOAuthAuthorizationRequiredError,
    McpOAuthProvider,
    McpOAuthState,
    McpOAuthStateStore,
    OAuthCallback,
    OAuthCallbackPage,
    OAuthCallbackServer,
    OAuthChallenge,
    OAuthClientInformationMixed,
    OAuthFlowOptions,
    authorize_mcp,
    parse_www_authenticate,
    step_up_scope,
)
from pidrei_mcp.url import parse_url
from pidrei_utils import clock
from pidrei_utils.cancel import CancelToken

from ...config import APP_NAME, get_agent_dir
from ...core.auth_storage import FileAuthStorageBackend
from ...core.mcp_servers import mcp_namespace
from ...utils.lockfile import FileLock


_CALLBACK_HOST = "127.0.0.1"
_CALLBACK_PATH = "/callback"
# Redirect URI for refreshes when none is stored. Refreshing never redirects the user.
_FALLBACK_REDIRECT_URL = f"http://{_CALLBACK_HOST}{_CALLBACK_PATH}"
# Access tokens this close to expiry are refreshed before they are sent.
_REFRESH_SKEW_MS = 30_000
# Bounds each request of a refresh, so it cannot hold the refresh lock or delay shutdown for long.
_REFRESH_REQUEST_TIMEOUT_MS = 15_000
# A refresh lock that its holder stopped renewing (the process was killed) is taken over after this.
_REFRESH_LOCK_STALE_MS = 20_000
# How long to wait for another process's refresh: longer than a stale lock lives.
_REFRESH_LOCK_WAIT_MS = 25_000
_REFRESH_LOCK_RETRY_MS = 100


@dataclass(frozen=True, slots=True)
class McpOAuthSettings:
    client_id: str | None = None
    # Already resolved.
    client_secret: str | None = None
    callback_port: int | None = None
    # Loopback redirect URI; see `McpOAuthConfig.callbackUrl`.
    callback_url: str | None = None
    # Scopes to request, separated by spaces.
    scope: str | None = None
    # `client_name` for dynamic client registration. Default: `APP_NAME`.
    client_name: str | None = None
    # See `McpOAuthConfig.authServerMetadataUrl`.
    auth_server_metadata_url: str | None = None


@dataclass(frozen=True, slots=True)
class _CallbackSettings:
    """Where the loopback callback server listens and the redirect URI it serves."""

    # Address to listen on.
    host: str
    # Host name in the redirect URI.
    redirect_host: str
    port: int | None
    path: str
    # The exact redirect URI, when the port is fixed.
    fixed_redirect_url: str | None


def _callback_settings(settings: McpOAuthSettings) -> _CallbackSettings:
    url = parse_url(settings.callback_url if settings.callback_url is not None else _FALLBACK_REDIRECT_URL)
    address = url.hostname.removeprefix("[").removesuffix("]")
    port = int(url.port) if url.port else settings.callback_port
    fixed_redirect_url: str | None = None
    # A configured URI with a port is sent exactly as written, since servers compare it as a string.
    if url.port:
        fixed_redirect_url = settings.callback_url
    elif port is not None:
        fixed_redirect_url = parse_url(f"{url.scheme}://{url.hostname}:{port}{url.pathname}").href
    return _CallbackSettings(
        # `localhost` is served on 127.0.0.1; browsers fall back to it when ::1 refuses.
        host=_CALLBACK_HOST if address == "localhost" else address,
        redirect_host=address,
        port=port,
        path=url.pathname,
        fixed_redirect_url=fixed_redirect_url,
    )


def _merge_scopes(*scopes: str | None) -> str | None:
    """Scopes of both lists, each once."""
    merged = list(dict.fromkeys(item for scope in scopes if scope for item in scope.split()))
    return " ".join(merged) if merged else None


type _StoredStates = dict[str, McpOAuthState]


def _parse_states(content: str | None) -> _StoredStates:
    if not content or not content.strip():
        return {}
    parsed = json.loads(content)
    return parsed if isinstance(parsed, dict) else {}


def _serialize_states(states: _StoredStates) -> str:
    return f"{json.dumps(states, indent=2, ensure_ascii=False)}\n"


def _store_keys(name: str, server_url: str) -> tuple[str, str]:
    """Keys of a server's state: by name and URL, so servers sharing a URL keep
    separate accounts, and the legacy key by URL alone, written by older
    versions."""
    legacy_key = parse_url(server_url).href
    return f"{mcp_namespace(name)}|{legacy_key}", legacy_key


class McpOAuthServerStore(McpOAuthStateStore, Protocol):
    async def with_refresh_lock[T](self, fn: Callable[[], Awaitable[T]]) -> T:
        """Run `fn` while no other process refreshes the server's tokens."""
        ...


class AuthStorageBackend(Protocol):
    async def with_lock_async(
        self, fn: Callable[[str | None], Awaitable[tuple[Any, str | None]]], options: Any = None
    ) -> Any: ...


class _ServerStore:
    __slots__ = ("_credentials", "_key", "_legacy_key")

    def __init__(self, credentials: McpOAuthCredentialStore, key: str, legacy_key: str) -> None:
        self._credentials = credentials
        self._key = key
        self._legacy_key = legacy_key

    async def load(self) -> McpOAuthState | None:
        key, legacy_key = self._key, self._legacy_key

        # The first server to load legacy state takes it over; others with the same URL sign in again.
        async def take_over(current: str | None) -> tuple[McpOAuthState | None, str | None]:
            states = _parse_states(current)
            if states.get(key) or not states.get(legacy_key):
                return states.get(key), None
            states[key] = states.pop(legacy_key)
            return states[key], _serialize_states(states)

        return await self._credentials.backend.with_lock_async(take_over)

    async def save(self, state: McpOAuthState) -> None:
        key = self._key

        def update(states: _StoredStates) -> None:
            states[key] = state

        await self._credentials.write(update)

    async def with_refresh_lock[T](self, fn: Callable[[], Awaitable[T]]) -> T:
        return await self._credentials.with_refresh_lock(self._key, fn)


class McpOAuthCredentialStore:
    """Per-server OAuth state (client registration, tokens, pending PKCE
    verifier) in `mcp-auth.json`."""

    def __init__(self, backend: AuthStorageBackend | None = None, lock_dir: str | None = None) -> None:
        self.backend: AuthStorageBackend = (
            backend if backend is not None else FileAuthStorageBackend(os.path.join(get_agent_dir(), "mcp-auth.json"))
        )
        # Directory for the refresh lock files. Without one, refreshes are only serialized in this process.
        self._lock_dir = lock_dir if backend is not None else get_agent_dir()

    def for_server(self, name: str, server_url: str) -> McpOAuthServerStore:
        key, legacy_key = _store_keys(name, server_url)
        return _ServerStore(self, key, legacy_key)

    async def with_refresh_lock[T](self, key: str, fn: Callable[[], Awaitable[T]]) -> T:
        """A lock file per server. A holder renews it while it runs; when the
        process is killed, the lock goes stale because it is no longer renewed,
        and the next process takes it over."""
        if not self._lock_dir:
            return await fn()
        await fs.Path(self._lock_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        lock = FileLock(
            os.path.join(self._lock_dir, f"mcp-auth-refresh-{digest}"),
            stale=_REFRESH_LOCK_STALE_MS / 1000,
            max_attempts=_REFRESH_LOCK_WAIT_MS // _REFRESH_LOCK_RETRY_MS + 1,
            delay=_REFRESH_LOCK_RETRY_MS / 1000,
            renew=True,
        )
        await lock.acquire()
        try:
            return await fn()
        finally:
            if isinstance(sys.exception(), CancelledError):
                # Nothing awaited on a cancelled chain runs: release on a coroutine of its own.
                tonio.spawn.without_tracking(lock.release())
            else:
                await lock.release()

    async def tokens(self, name: str, server_url: str) -> Mapping[str, Any] | None:
        """The stored tokens of a server, for noticing sign-ins done by another
        process. Does not take over legacy state."""
        key, legacy_key = _store_keys(name, server_url)

        async def read(current: str | None) -> tuple[_StoredStates, None]:
            return _parse_states(current), None

        states = await self.backend.with_lock_async(read)
        state = states.get(key) or states.get(legacy_key)
        return state.get("tokens") if state else None

    async def remove(self, name: str, server_url: str) -> bool:
        """Returns whether credentials were stored for the server. Removes
        legacy state the server would take over."""
        key, legacy_key = _store_keys(name, server_url)

        async def drop(current: str | None) -> tuple[bool, str | None]:
            states = _parse_states(current)
            stored = key if key in states else legacy_key if legacy_key in states else None
            if stored is None:
                return False, None
            del states[stored]
            return True, _serialize_states(states)

        return await self.backend.with_lock_async(drop)

    async def write(self, update: Callable[[_StoredStates], None]) -> None:
        async def apply(current: str | None) -> tuple[None, str]:
            states = _parse_states(current)
            update(states)
            return None, _serialize_states(states)

        await self.backend.with_lock_async(apply)


def _registered_redirect_urls(client: OAuthClientInformationMixed | None) -> list[str]:
    return list(client.get("redirect_uris") or []) if client else []


async def _ignore_redirect(_url: str) -> None:
    pass


def _create_provider(
    server_url: str,
    store: McpOAuthStateStore,
    settings: McpOAuthSettings,
    redirect_url: str,
    on_redirect: Callable[[str], Awaitable[None]],
) -> McpOAuthProvider:
    return McpOAuthProvider(
        server_url=server_url,
        redirect_url=redirect_url,
        client_metadata={"client_name": settings.client_name if settings.client_name is not None else APP_NAME},
        client_id=settings.client_id,
        client_secret=settings.client_secret,
        store=store,
        on_redirect=on_redirect,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class McpAuthProvider(AuthProvider):
    # Resolves when no refresh is running, so shutdown does not drop rotated
    # tokens before they are saved.
    settled: Callable[[], Awaitable[None]]


class _Refresh:
    """One refresh in flight. `error` is set before `done`."""

    __slots__ = ("done", "error")

    def __init__(self) -> None:
        self.done = tonio.Event()
        self.error: Exception | None = None


def _with_timeout(fetch: McpFetch) -> McpFetch:
    def timed(
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        body: bytes | None = None,
        timeout_ms: float | None = None,
    ) -> Awaitable[McpResponse]:
        return fetch(url, method=method, headers=headers, body=body, timeout_ms=_REFRESH_REQUEST_TIMEOUT_MS)

    return timed


def create_mcp_auth_provider(
    *,
    server_url: str,
    store: McpOAuthServerStore,
    settings: Callable[[], Awaitable[McpOAuthSettings]],
    on_challenge: Callable[[OAuthChallenge], None],
) -> McpAuthProvider:
    """Auth provider for MCP connections: sends the stored access token and
    refreshes it when it is about to expire or after a 401. Raises
    `McpOAuthAuthorizationRequiredError` when the user has to sign in,
    including when the server asks for more scope (`insufficient_scope`).
    `on_challenge` receives the server's `WWW-Authenticate` challenge so
    sign-in can use its resource metadata URL and scope. `settings` is only
    called when a refresh is needed, so a secret that fails to resolve fails
    the refresh instead of the whole connection setup.

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
            if not tokens.get("refresh_token"):
                raise McpOAuthAuthorizationRequiredError()
            resolved = await settings()
            fixed = _callback_settings(resolved).fixed_redirect_url
            registered = _registered_redirect_urls((state or {}).get("clientInformation"))
            redirect_url = fixed if fixed is not None else registered[0] if registered else _FALLBACK_REDIRECT_URL
            provider = _create_provider(server_url, store, resolved, redirect_url, _ignore_redirect)
            # Refreshes the tokens, or reports that a new sign-in is needed.
            result = await authorize_mcp(
                provider,
                OAuthFlowOptions(
                    server_url=server_url,
                    resource_metadata_url=(challenge or {}).get("resourceMetadataUrl"),
                    authorization_server_metadata_url=resolved.auth_server_metadata_url,
                    scope=(challenge or {}).get("scope"),
                    fetch=_with_timeout(fetch),
                ),
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


class McpSignInPrompt(Protocol):
    def show_authorization_url(self, url: str) -> None:
        """Show the authorization URL to the user and open it in a browser."""
        ...

    def prompt_for_redirect_url(self, cancel: CancelToken) -> Awaitable[str | None]:
        """Ask for the redirect URL from the browser address bar, for when the
        browser cannot reach the loopback callback (for example over SSH).
        Cancelled once the callback arrives. Resolves to None or an empty
        string when the user cancels."""
        ...


class McpSignInCancelledError(Exception):
    def __init__(self) -> None:
        super().__init__("Sign-in cancelled")


type _AuthorizationResponse = tuple[str, str | None]


def _response_from_redirect_url(text: str, state: str) -> _AuthorizationResponse:
    """`(code, iss)` from a pasted redirect URL."""
    try:
        url = parse_url(text.strip())
    except ValueError:
        raise Exception("Expected the full redirect URL from the browser address bar") from None
    error = url.search_param("error")
    if error:
        description = url.search_param("error_description")
        raise Exception(description if description is not None else error)
    if url.search_param("state") != state:
        raise Exception("The redirect URL belongs to a different sign-in")
    code = url.search_param("code")
    if not code:
        raise Exception("The redirect URL does not contain an authorization code")
    return code, url.search_param("iss")


class _FirstOutcome:
    """The first of several racing outcomes; later ones are dropped."""

    __slots__ = ("_guard", "done", "error", "value")

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self.done = tonio.Event()
        self.value: Any = None
        self.error: Exception | None = None

    def settle(self, value: Any = None, error: Exception | None = None) -> None:
        with self._guard:
            if self.done.is_set():
                return
            self.value, self.error = value, error
            self.done.set()


async def _wait_for_authorization_response(
    from_browser: Awaitable[OAuthCallback], state: str, prompt: McpSignInPrompt
) -> _AuthorizationResponse:
    """Wait for the browser callback or a pasted redirect URL, whichever comes
    first. `from_browser` is the callback wait, registered by the caller before
    the browser was sent to the authorization page."""
    cancel = CancelToken()
    outcome = _FirstOutcome()

    async def browser() -> None:
        try:
            result = await from_browser
            outcome.settle((result.code, result.iss))
        except Exception as error:
            outcome.settle(error=error)

    async def user() -> None:
        try:
            text = await prompt.prompt_for_redirect_url(cancel)
            if not text or not text.strip():
                raise McpSignInCancelledError()
            outcome.settle(_response_from_redirect_url(text, state))
        except Exception as error:
            outcome.settle(error=error)

    tonio.spawn.without_tracking(browser(), user())
    try:
        await outcome.done.wait()
    finally:
        # The losing side ends once the prompt is cancelled or the callback server closes.
        cancel.cancel()
    if outcome.error is not None:
        raise outcome.error
    return outcome.value


def _render_page(page: OAuthCallbackPage) -> str:
    if page.ok:
        return oauth_success_html("Signed in to the MCP server. You may now close this page.")
    return oauth_error_html(page.message or "", page.details)


async def _listen_for_callback(settings: _CallbackSettings, port: int | None, required: bool) -> OAuthCallbackServer:
    """Listen on `port`, or on a free port when it is taken and not `required`."""

    def server(port: int | None) -> OAuthCallbackServer:
        return OAuthCallbackServer(
            host=settings.host,
            redirect_host=settings.redirect_host,
            path=settings.path,
            port=port,
            render_page=_render_page,
        )

    try:
        return await server(port if port is not None else 0)
    except Exception:
        if required or port is None:
            raise
        return await server(None)


async def sign_in_mcp_server(
    *,
    server_url: str,
    store: McpOAuthStateStore,
    settings: McpOAuthSettings,
    prompt: McpSignInPrompt,
    challenge: OAuthChallenge | None = None,
) -> None:
    """Sign in to an MCP server. Uses the stored refresh token when possible;
    otherwise runs the browser authorization code flow. Tokens are saved to
    `store`."""
    stored = await store.load()
    step_up = (challenge or {}).get("error") == "insufficient_scope"
    callback_options = _callback_settings(settings)
    # Reuse the port of the registered redirect URI so the registered client stays valid.
    registered = _registered_redirect_urls((stored or {}).get("clientInformation"))
    preferred_port = callback_options.port
    if preferred_port is None and registered:
        registered_port = parse_url(registered[0]).port
        preferred_port = int(registered_port) if registered_port and int(registered_port) else None
    callback = await _listen_for_callback(callback_options, preferred_port, callback_options.port is not None)
    redirect_url = (
        callback_options.fixed_redirect_url
        if callback_options.fixed_redirect_url is not None
        else callback.redirect_url
    )
    try:
        if stored:
            # Every sign-in gets a fresh `state` parameter.
            next_state: McpOAuthState = {key: value for key, value in stored.items() if key != "oauthState"}  # type: ignore[assignment]
            # A registered client cannot use another redirect URI, and its tokens belong to it.
            if not settings.client_id and redirect_url not in registered:
                for key in ("clientInformation", "tokens", "tokensExpireAt"):
                    next_state.pop(key, None)  # type: ignore[misc]
            await store.save(next_state)

        authorization_urls: list[str] = []

        async def on_redirect(url: str) -> None:
            authorization_urls.append(url)

        provider = _create_provider(server_url, store, settings, redirect_url, on_redirect)
        # A server asking for more scope gets it on top of the configured scope
        # and, since the challenge may list only the missing scopes, on top of
        # the scope granted so far.
        challenged = (challenge or {}).get("scope")
        granted = ((stored or {}).get("tokens") or {}).get("scope")
        flow = {
            "server_url": server_url,
            "resource_metadata_url": (challenge or {}).get("resourceMetadataUrl"),
            "authorization_server_metadata_url": settings.auth_server_metadata_url,
            "scope": _merge_scopes(settings.scope, step_up_scope(granted, challenged) if step_up else challenged),
        }
        # A refresh keeps the granted scope; a server asking for more needs the browser flow.
        if await authorize_mcp(provider, OAuthFlowOptions(**flow, skip_refresh=step_up)) == "AUTHORIZED":
            return
        if not authorization_urls:
            raise Exception("OAuth flow did not produce an authorization URL")

        state = await provider.state()
        # Registered before the browser is sent to the authorization page: the
        # browser runs on its own and may reach the callback before this
        # coroutine runs again (pi registers it in the same turn).
        from_browser = callback.wait_for_callback(state)
        prompt.show_authorization_url(authorization_urls[-1])
        code, iss = await _wait_for_authorization_response(from_browser, state, prompt)
        await authorize_mcp(provider, OAuthFlowOptions(**flow, authorization_code=code, iss=iss))
    finally:
        callback.close()
