"""Mirror of pi coding-agent src/extensions/mcp/oauth.ts: OAuth sign-in for
remote MCP servers, PiDrei's credential store and its side of the sign-in.

Connections never start a browser flow on their own. They send the stored
access token and, after a 401, try the stored refresh token. When that is not
possible they fail with `McpOAuthAuthorizationRequiredError`, and the user
signs in through `/mcp` or `pidrei mcp login`, which run pidrei-mcp's sign-in
(PKCE, pi's Client ID Metadata Document or dynamic client registration)
against a loopback callback or a pasted redirect URL.

Credentials live in `<agent-dir>/mcp-auth.json`, keyed by server name and URL.
The refresh lock is a `FileLock` that renews itself while held, as
proper-lockfile's does.
"""

import hashlib
import json
import os
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from typing import Any, Protocol

import tonio.colored as tonio
from tonio.colored import fs
from tonio.colored.exceptions import CancelledError

from pidrei_ai.utils.oauth_page import oauth_error_html, oauth_success_html
from pidrei_mcp import mcp_namespace
from pidrei_mcp.oauth import (
    McpAuthProvider,
    McpClientMetadataDocuments,
    McpOAuthServerStore,
    McpOAuthSettings,
    McpOAuthState,
    McpRefreshLockBusyError,
    McpSignInOptions,
    OAuthCallbackPage,
    OAuthChallenge,
    complete_mcp_sign_in,
    create_mcp_auth_provider as _create_auth_provider,
    listen_for_loopback_sign_in,
    start_mcp_sign_in,
    wait_for_loopback_sign_in,
)
from pidrei_mcp.url import parse_url
from pidrei_utils.cancel import CancelToken

from ...config import APP_NAME, get_agent_dir
from ...core.auth_storage import FileAuthStorageBackend
from ...core.mcp_servers import is_client_metadata_document_callback
from ...utils.lockfile import FileLock, LockedError


# Where pi.dev serves pi's Client ID Metadata Documents: `client.json` and `<callback ID>/client.json`.
# The one place where PiDrei presents itself as pi, deliberately: with `clientRegistration: "cimd"` the
# client ID is pi's document (pi's name and logo on the consent screen), since PiDrei has no document of
# its own. Not renamed by rename passes; dynamic client registration keeps registering as `APP_NAME`.
CLIENT_METADATA_BASE_URL = "https://pi.dev/oauth"
# A refresh lock that its holder stopped renewing (the process was killed) is taken over after this.
_REFRESH_LOCK_STALE_MS = 20_000
# How long to wait for another process's refresh: longer than a stale lock lives.
_REFRESH_LOCK_WAIT_MS = 25_000
_REFRESH_LOCK_RETRY_MS = 100


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

    def with_refresh_lock[T](self, fn: Callable[[], Awaitable[T]]) -> Awaitable[T]:
        return self._credentials.with_refresh_lock(self._key, fn)


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
        and the next process takes it over. Raises `McpRefreshLockBusyError`
        when another process holds the lock for the whole wait."""
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
        try:
            await lock.acquire()
        except LockedError:
            # The lock is held by another process (`ELOCKED`).
            raise McpRefreshLockBusyError() from None
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


# pi's Client ID Metadata Documents on pi.dev: `client.json` and `<callback ID>/client.json`.
PI_CLIENT_METADATA_DOCUMENTS = McpClientMetadataDocuments(
    base_url=CLIENT_METADATA_BASE_URL, lists_redirect_url=is_client_metadata_document_callback
)


def _pi_settings(settings: McpOAuthSettings) -> McpOAuthSettings:
    """`settings` for pidrei-mcp: a rebranded build would show up as pi with
    pi's documents, so `auto` registers. PiDrei is such a build (`APP_NAME`
    is not pi's), so `auto` always registers; only an explicit `cimd` uses
    pi's document."""
    auto = (settings.client_registration if settings.client_registration is not None else "auto") == "auto"
    return replace(settings, client_registration="dcr") if auto and APP_NAME != "pi" else settings


def create_mcp_auth_provider(
    *,
    server_url: str,
    store: McpOAuthServerStore,
    settings: Callable[[], Awaitable[McpOAuthSettings]],
    on_challenge: Callable[[OAuthChallenge], None],
) -> McpAuthProvider:
    """pidrei-mcp's `create_mcp_auth_provider` with PiDrei's client identity."""

    async def resolved() -> McpOAuthSettings:
        return _pi_settings(await settings())

    return _create_auth_provider(
        server_url=server_url,
        store=store,
        settings=resolved,
        on_challenge=on_challenge,
        client_name=APP_NAME,
        client_metadata_documents=PI_CLIENT_METADATA_DOCUMENTS,
    )


class McpSignInPrompt(Protocol):
    def show_authorization_url(self, url: str) -> None:
        """Show the authorization URL to the user and open it in a browser."""
        ...

    def prompt_for_redirect_url(self, cancel: CancelToken) -> Awaitable[str | None]:
        """Ask for the redirect URL from the browser address bar, for when the
        browser cannot reach the loopback callback (for example over SSH).
        Cancelled once the callback arrives or the sign-in is cancelled.
        Resolves to None or an empty string when the user cancels."""
        ...


def _render_page(page: OAuthCallbackPage) -> str:
    if page.ok:
        return oauth_success_html("Signed in to the MCP server. You may now close this page.")
    return oauth_error_html(page.message or "", page.details)


async def sign_in_mcp_server(
    *,
    server_url: str,
    store: McpOAuthServerStore,
    settings: McpOAuthSettings,
    prompt: McpSignInPrompt,
    challenge: OAuthChallenge | None = None,
    cancel: CancelToken | None = None,
) -> None:
    """Sign in to an MCP server. Uses the stored refresh token when possible;
    otherwise runs the browser authorization code flow against a loopback
    callback or a pasted redirect URL. Tokens are saved to `store`. `cancel`
    stops the sign-in at any step with `McpSignInCancelledError`; requests to
    the authorization server are also time-limited."""
    settings = _pi_settings(settings)
    loopback = await listen_for_loopback_sign_in(
        server_url=server_url,
        store=store,
        settings=settings,
        client_metadata_documents=PI_CLIENT_METADATA_DOCUMENTS,
        render_page=_render_page,
    )
    try:
        sign_in = McpSignInOptions(
            server_url=server_url,
            store=store,
            settings=settings,
            redirect_url=loopback.redirect_url,
            challenge=challenge,
            cancel=cancel,
            client_name=APP_NAME,
            client_metadata_documents=PI_CLIENT_METADATA_DOCUMENTS,
        )
        authorization = await start_mcp_sign_in(sign_in)
        if authorization["kind"] == "authorized":
            return
        # Registered before the browser is sent to the authorization page: the
        # browser runs on its own and may reach the callback before this
        # coroutine runs again (pi registers it in the same turn).
        waiting = wait_for_loopback_sign_in(
            callback=loopback.callback,
            authorization=authorization,
            prompt_for_redirect_url=prompt.prompt_for_redirect_url,
            cancel=cancel,
        )
        try:
            prompt.show_authorization_url(authorization["authorizationUrl"])
        except BaseException:
            waiting.close()
            raise
        response = await waiting
        await complete_mcp_sign_in(sign_in, authorization=authorization, response=response)
    finally:
        loopback.callback.close()
