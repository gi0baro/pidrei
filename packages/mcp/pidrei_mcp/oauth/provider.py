"""Mirror of pi mcp src/oauth/provider.ts.

The store is read and written in call order under a TonIO lock (pi chains
its writes on one promise). `state()` reads and, when none is stored, writes
in one locked step: two concurrent callers get the same state.
"""

import copy
import secrets
from collections.abc import Awaitable, Callable
from typing import NotRequired, Protocol, TypedDict

from tonio.colored import sync

from pidrei_utils import clock

from ..url import parse_url
from .flow import CredentialsKind, OAuthClientMetadataDocument, OAuthClientProvider
from .types import (
    AuthorizationServerMetadata,
    OAuthClientInformationMixed,
    OAuthClientMetadata,
    OAuthDiscoveryState,
    OAuthTokens,
)


class McpOAuthState(TypedDict):
    serverUrl: str
    clientInformation: NotRequired[OAuthClientInformationMixed]
    tokens: NotRequired[OAuthTokens]
    # When the access token expires, in milliseconds since the epoch, from `expires_in` at the time it was saved.
    tokensExpireAt: NotRequired[float]
    codeVerifier: NotRequired[str]
    oauthState: NotRequired[str]
    discovery: NotRequired[OAuthDiscoveryState]


class McpOAuthStateStore(Protocol):
    async def load(self) -> McpOAuthState | None: ...
    async def save(self, state: McpOAuthState) -> None: ...


class MemoryOAuthStateStore:
    def __init__(self) -> None:
        self._value: McpOAuthState | None = None

    async def load(self) -> McpOAuthState | None:
        value = self._value
        return None if value is None else copy.deepcopy(value)

    async def save(self, state: McpOAuthState) -> None:
        self._value = copy.deepcopy(state)


class McpOAuthProvider(OAuthClientProvider):
    """Default stateful provider for one exact MCP server URL. Applications
    inject durable storage if needed."""

    def __init__(
        self,
        *,
        server_url: str,
        redirect_url: str,
        client_metadata: OAuthClientMetadata,
        on_redirect: Callable[[str], Awaitable[None]],
        client_metadata_document: (
            Callable[[AuthorizationServerMetadata | None], OAuthClientMetadataDocument | None] | None
        ) = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        store: McpOAuthStateStore | None = None,
    ) -> None:
        self._server_url = parse_url(server_url).href
        self.redirect_url = redirect_url
        # See `OAuthClientProvider.client_metadata_document`.
        self.client_metadata_document = client_metadata_document

        def given(key: str, default: object) -> object:
            value = client_metadata.get(key)
            return default if value is None else value

        self.client_metadata = {
            **client_metadata,
            "redirect_uris": given("redirect_uris", [redirect_url]),
            "grant_types": given("grant_types", ["authorization_code", "refresh_token"]),
            "response_types": given("response_types", ["code"]),
            "token_endpoint_auth_method": given(
                "token_endpoint_auth_method", "client_secret_post" if client_secret else "none"
            ),
        }  # type: ignore[assignment]
        self._configured_client: OAuthClientInformationMixed | None = None
        if client_id:
            self._configured_client = {"client_id": client_id}
            if client_secret:
                self._configured_client["client_secret"] = client_secret
        self._store: McpOAuthStateStore = store if store is not None else MemoryOAuthStateStore()
        self._on_redirect = on_redirect
        self._lock = sync.Lock()

    async def state(self) -> str:  # type: ignore[override]
        async with self._lock:
            value = self._own(await self._store.load())
            existing = value.get("oauthState")
            if existing:
                return existing
            state = secrets.token_hex(32)
            await self._store.save({**value, "oauthState": state})
            return state

    async def client_information(self) -> OAuthClientInformationMixed | None:
        if self._configured_client is not None:
            return self._configured_client
        return (await self._load()).get("clientInformation")

    async def save_client_information(self, information: OAuthClientInformationMixed) -> None:  # type: ignore[override]
        if self._configured_client is not None:
            return
        await self._update(lambda value: {**value, "clientInformation": information})

    async def tokens(self) -> OAuthTokens | None:
        return (await self._load()).get("tokens")

    async def save_tokens(self, tokens: OAuthTokens) -> None:
        expires_in = tokens.get("expires_in")
        expires_at = None if expires_in is None else clock.now_ms() + expires_in * 1000

        def update(value: McpOAuthState) -> McpOAuthState:
            updated: McpOAuthState = {**value, "tokens": tokens}
            if expires_at is None:
                updated.pop("tokensExpireAt", None)
            else:
                updated["tokensExpireAt"] = expires_at
            return updated

        await self._update(update)

    def redirect_to_authorization(self, url: str) -> Awaitable[None]:
        return self._on_redirect(url)

    def save_code_verifier(self, verifier: str) -> Awaitable[None]:
        return self._update(lambda value: {**value, "codeVerifier": verifier})

    async def code_verifier(self) -> str:
        verifier = (await self._load()).get("codeVerifier")
        if not verifier:
            raise RuntimeError("No OAuth PKCE code verifier is stored")
        return verifier

    async def invalidate_credentials(self, kind: CredentialsKind) -> None:  # type: ignore[override]
        def update(value: McpOAuthState) -> McpOAuthState:
            updated: McpOAuthState = {**value}
            if kind in ("all", "client"):
                updated.pop("clientInformation", None)
            if kind in ("all", "tokens"):
                updated.pop("tokens", None)
                updated.pop("tokensExpireAt", None)
            if kind in ("all", "verifier"):
                updated.pop("codeVerifier", None)
            if kind in ("all", "discovery"):
                updated.pop("discovery", None)
            if kind == "all":
                updated.pop("oauthState", None)
            return updated

        await self._update(update)

    def save_discovery_state(self, discovery: OAuthDiscoveryState) -> Awaitable[None]:  # type: ignore[override]
        return self._update(lambda value: {**value, "discovery": discovery})

    async def discovery_state(self) -> OAuthDiscoveryState | None:  # type: ignore[override]
        return (await self._load()).get("discovery")

    async def _load(self) -> McpOAuthState:
        async with self._lock:
            return self._own(await self._store.load())

    async def _update(self, update: Callable[[McpOAuthState], McpOAuthState]) -> None:
        async with self._lock:
            await self._store.save(update(self._own(await self._store.load())))

    def _own(self, state: McpOAuthState | None) -> McpOAuthState:
        """Stored state for another server URL is ignored so credentials never leak across servers."""
        if state is not None and state.get("serverUrl") == self._server_url:
            return state
        return {"serverUrl": self._server_url}
