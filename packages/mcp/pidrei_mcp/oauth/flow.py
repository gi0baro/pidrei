"""Mirror of pi mcp src/oauth/flow.ts.

Adapted from modelcontextprotocol/typescript-sdk v1.29.0 src/client/auth.ts.
Copyright (c) 2024 Anthropic, PBC. Licensed under MIT; see LICENSES/.
Modified to remove SDK/Zod dependencies and use WebCrypto for PKCE.
"""

import base64
import threading
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import tonio.colored as tonio

from pidrei_http.pkce import generate_pkce
from pidrei_utils.cancel import CancelToken, run_cancellable

from ..auth_provider import AuthProvider, UnauthorizedContext
from ..fetch import McpFetch, McpResponse, default_fetch
from ..protocol.jsonrpc import is_object, parse_json, stringify
from ..url import Url, can_parse, form_encode, parse_url
from .discovery import (
    discover_authorization_server_metadata,
    discover_oauth_server_info,
    parse_www_authenticate,
    select_resource,
)
from .errors import (
    McpOAuthAuthorizationRequiredError,
    OAuthError,
    OAuthInsecureEndpointError,
    OAuthIssuerMismatchError,
    OAuthRegistrationError,
)
from .types import (
    AuthorizationServerMetadata,
    OAuthClientInformationFull,
    OAuthClientInformationMixed,
    OAuthClientMetadata,
    OAuthDiscoveryState,
    OAuthServerInfo,
    OAuthTokens,
    parse_client_information,
    parse_oauth_tokens,
)


# `(headers, params, url, metadata)`: both mappings are the request's own,
# changed in place, as pi hands over its `Headers` and `URLSearchParams`.
type AddClientAuthentication = Callable[
    [dict[str, str], dict[str, str], str, AuthorizationServerMetadata | None], Awaitable[None]
]
type CredentialsKind = Literal["all", "client", "tokens", "verifier", "discovery"]
type OAuthFlowResult = Literal["AUTHORIZED", "REDIRECT"]
type _ClientAuthMethod = Literal["client_secret_basic", "client_secret_post", "none"]


@dataclass(frozen=True, slots=True)
class OAuthClientMetadataDocument:
    """A Client ID Metadata Document: an https URL used as `client_id`, and a redirect URI it lists."""

    url: str
    redirect_url: str


class OAuthClientProvider(ABC):
    """pi's optional members are the attributes that default to `None`: a
    provider that has one overrides it (with a method, for the callables),
    and the flow skips what a provider does not have, as pi does."""

    redirect_url: str
    client_metadata: OAuthClientMetadata
    # Client ID Metadata Document to identify as instead of registering
    # dynamically, or None (returned) to register. Called when no client
    # information is stored; the document is not stored. `metadata` is None
    # when the authorization server has none; check
    # `client_id_metadata_document_supported`.
    client_metadata_document: (
        Callable[[AuthorizationServerMetadata | None], OAuthClientMetadataDocument | None] | None
    ) = None
    state: Callable[[], Awaitable[str]] | None = None
    save_client_information: Callable[[OAuthClientInformationMixed], Awaitable[None]] | None = None
    add_client_authentication: AddClientAuthentication | None = None
    invalidate_credentials: Callable[[CredentialsKind], Awaitable[None]] | None = None
    save_discovery_state: Callable[[OAuthDiscoveryState], Awaitable[None]] | None = None
    discovery_state: Callable[[], Awaitable[OAuthDiscoveryState | None]] | None = None

    @abstractmethod
    async def client_information(self) -> OAuthClientInformationMixed | None: ...

    @abstractmethod
    async def tokens(self) -> OAuthTokens | None: ...

    @abstractmethod
    async def save_tokens(self, tokens: OAuthTokens) -> None: ...

    @abstractmethod
    async def redirect_to_authorization(self, url: str) -> None: ...

    @abstractmethod
    async def save_code_verifier(self, verifier: str) -> None: ...

    @abstractmethod
    async def code_verifier(self) -> str: ...


@dataclass(frozen=True, slots=True)
class OAuthFlowOptions:
    server_url: str
    authorization_code: str | None = None
    # `iss` parameter of the authorization response that delivered `authorization_code` (RFC 9207).
    iss: str | None = None
    scope: str | None = None
    resource_metadata_url: str | None = None
    # Authorization server metadata document to use instead of discovery, for
    # servers that advertise a wrong authorization server or none. It is
    # trusted as configured. Must use https, except on loopback.
    authorization_server_metadata_url: str | None = None
    fetch: McpFetch | None = None
    # Stops every request of the flow (pi: `signal`, handed to each request).
    # Requests have no time limit of their own; combine with a timeout as needed.
    cancel: CancelToken | None = None
    skip_issuer_validation: bool = False
    # Go straight to the authorization redirect instead of refreshing stored
    # tokens, for example when the server asks for scopes the current grant
    # lacks (a refresh keeps the old scope).
    skip_refresh: bool = False


@dataclass(frozen=True, slots=True)
class TokenRequestOptions:
    client_information: OAuthClientInformationMixed
    metadata: AuthorizationServerMetadata | None = None
    resource: str | None = None
    add_client_authentication: AddClientAuthentication | None = None
    fetch: McpFetch | None = None


@dataclass(frozen=True, slots=True)
class AuthorizationStart:
    authorization_url: str
    code_verifier: str


def _loopback(hostname: str) -> bool:
    return hostname in ("localhost", "127.0.0.1", "[::1]", "::1")


def _application_type(redirect_uris: list[str]) -> str:
    """The OpenID Connect `application_type` for `redirect_uris` (MCP SEP-837). Without one, OpenID Connect servers
    assume `web`, which rejects http loopback redirect URIs. Loopback hosts and custom schemes are native (RFC 8252)."""

    def native(uri: str) -> bool:
        if not can_parse(uri):
            return False
        url = parse_url(uri)
        return url.protocol not in ("http:", "https:") or _loopback(url.hostname)

    return "native" if any(native(uri) for uri in redirect_uris) else "web"


def _secure_endpoint(value: str) -> Url:
    url = parse_url(value)
    if url.protocol != "https:" and not _loopback(url.hostname):
        raise OAuthInsecureEndpointError(url.href)
    return url


def _select_client_auth_method(information: OAuthClientInformationMixed, supported: list[str]) -> _ClientAuthMethod:
    hinted = information.get("token_endpoint_auth_method")
    if (
        hinted
        and hinted in ("client_secret_basic", "client_secret_post", "none")
        and (not supported or hinted in supported)
    ):
        return hinted  # type: ignore[return-value]
    secret = information.get("client_secret")
    if not supported:
        return "client_secret_basic" if secret else "none"
    if secret and "client_secret_basic" in supported:
        return "client_secret_basic"
    if secret and "client_secret_post" in supported:
        return "client_secret_post"
    if "none" in supported:
        return "none"
    return "client_secret_post" if secret else "none"


def _apply_client_authentication(
    method: _ClientAuthMethod,
    information: OAuthClientInformationMixed,
    headers: dict[str, str],
    params: dict[str, str],
) -> None:
    secret = information.get("client_secret")
    if method == "client_secret_basic":
        if not secret:
            raise RuntimeError("client_secret_basic requires a client secret")
        credentials = base64.b64encode(f"{information['client_id']}:{secret}".encode()).decode("ascii")
        headers["Authorization"] = f"Basic {credentials}"
    else:
        params["client_id"] = information["client_id"]
        if method == "client_secret_post" and secret:
            params["client_secret"] = secret


def start_authorization(
    authorization_server_url: str,
    *,
    client_information: OAuthClientInformationMixed,
    redirect_url: str,
    metadata: AuthorizationServerMetadata | None = None,
    scope: str | None = None,
    state: str | None = None,
    resource: str | None = None,
) -> AuthorizationStart:
    if metadata is not None and "code" not in metadata["response_types_supported"]:
        raise RuntimeError("Authorization server does not support authorization codes")
    methods = metadata.get("code_challenge_methods_supported") if metadata is not None else None
    if methods is not None and "S256" not in methods:
        raise RuntimeError("Authorization server does not support PKCE S256")
    if metadata is not None:
        url = parse_url(metadata["authorization_endpoint"])
    else:
        url = parse_url("/authorize", authorization_server_url)
    pkce = generate_pkce()
    url = url.with_search_param("response_type", "code")
    url = url.with_search_param("client_id", client_information["client_id"])
    url = url.with_search_param("code_challenge", pkce.challenge)
    url = url.with_search_param("code_challenge_method", "S256")
    url = url.with_search_param("redirect_uri", redirect_url)
    if state:
        url = url.with_search_param("state", state)
    if scope:
        url = url.with_search_param("scope", scope)
    if scope and "offline_access" in scope.split():
        url = url.with_search_param("prompt", "consent")
    if resource:
        url = url.with_search_param("resource", resource)
    return AuthorizationStart(authorization_url=url.href, code_verifier=pkce.verifier)


async def _token_request(
    authorization_server_url: str, options: TokenRequestOptions, params: dict[str, str]
) -> OAuthTokens:
    metadata = options.metadata
    url = _secure_endpoint(
        metadata["token_endpoint"] if metadata is not None else parse_url("/token", authorization_server_url).href
    )
    headers = {"Accept": "application/json", "content-type": "application/x-www-form-urlencoded"}
    if options.resource:
        params["resource"] = options.resource
    if options.add_client_authentication is not None:
        await options.add_client_authentication(headers, params, url.href, metadata)
    else:
        supported = (metadata or {}).get("token_endpoint_auth_methods_supported") or []
        _apply_client_authentication(
            _select_client_auth_method(options.client_information, supported),
            options.client_information,
            headers,
            params,
        )
    fetch = options.fetch if options.fetch is not None else default_fetch
    response = await fetch(url.href, method="POST", headers=headers, body=form_encode(list(params.items())).encode())
    text = (await response.read()).decode("utf-8", "replace")
    try:
        value: Any = parse_json(text)
    except ValueError:
        value = None
    # Servers may report OAuth errors with any status, so check the body before the status.
    if is_object(value) and isinstance(value.get("error"), str):
        description = value.get("error_description")
        error_uri = value.get("error_uri")
        raise OAuthError(
            value["error"],
            description if isinstance(description, str) else value["error"],
            error_uri if isinstance(error_uri, str) else None,
        )
    if not 200 <= response.status < 300:
        raise OAuthError("server_error", f"HTTP {response.status}: {text}")
    return parse_oauth_tokens(value)


async def register_client(
    authorization_server_url: str,
    *,
    client_metadata: OAuthClientMetadata,
    metadata: AuthorizationServerMetadata | None = None,
    scope: str | None = None,
    fetch: McpFetch | None = None,
) -> OAuthClientInformationFull:
    endpoint = metadata.get("registration_endpoint") if metadata is not None else None
    if metadata is not None and not endpoint:
        raise RuntimeError("Authorization server does not support dynamic client registration")
    url = parse_url(endpoint) if endpoint else parse_url("/register", authorization_server_url)
    application_type = client_metadata.get("application_type")
    body = {
        **client_metadata,
        "application_type": application_type
        if application_type is not None
        else _application_type(client_metadata["redirect_uris"]),
        **({"scope": scope} if scope else {}),
    }
    response = await (fetch if fetch is not None else default_fetch)(
        url.href,
        method="POST",
        headers={"Accept": "application/json", "content-type": "application/json"},
        body=stringify(body).encode(),
    )
    payload = await response.read()
    if not 200 <= response.status < 300:
        raise OAuthRegistrationError(response.status, payload.decode("utf-8", "replace"))
    return parse_client_information(parse_json(payload))


def exchange_authorization_code(
    authorization_server_url: str,
    options: TokenRequestOptions,
    *,
    code: str,
    code_verifier: str,
    redirect_url: str,
) -> Awaitable[OAuthTokens]:
    return _token_request(
        authorization_server_url,
        options,
        {
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": redirect_url,
        },
    )


async def refresh_authorization(
    authorization_server_url: str, options: TokenRequestOptions, *, refresh_token: str
) -> OAuthTokens:
    tokens = await _token_request(
        authorization_server_url, options, {"grant_type": "refresh_token", "refresh_token": refresh_token}
    )
    return {"refresh_token": refresh_token, **tokens}  # type: ignore[typeddict-item]


def _with_scope(tokens: OAuthTokens, scope: str | None) -> OAuthTokens:
    return {**tokens, "scope": scope} if "scope" not in tokens and scope else tokens  # type: ignore[typeddict-item]


def step_up_scope(granted: str | None, challenged: str | None) -> str | None:
    """Scopes for a step-up authorization: the challenged scopes plus the
    ones granted so far, since a challenge may list only the missing scopes
    and a token with just those would lose access the old one had
    (SEP-2350). Without challenged scopes, None lets the flow pick its
    default."""
    if not challenged:
        return None
    scopes = [item for scope in (granted, challenged) if scope for item in scope.split()]
    return " ".join(dict.fromkeys(scopes))


def _cancellable(fetch: McpFetch, cancel: CancelToken | None) -> McpFetch:
    """`fetch` with every request run under `cancel`, where pi hands each
    request the flow's `signal`. Only the requests are torn: the provider's
    saves between them always complete."""
    if cancel is None:
        return fetch

    def fetching(
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        body: bytes | None = None,
        timeout_ms: float | None = None,
    ) -> Awaitable[McpResponse]:
        return run_cancellable(
            fetch(url, method=method, headers=headers, body=body, timeout_ms=timeout_ms),  # type: ignore[arg-type]
            cancel,
        )

    return fetching


async def _run_flow(provider: OAuthClientProvider, options: OAuthFlowOptions) -> OAuthFlowResult:
    fetch = _cancellable(options.fetch if options.fetch is not None else default_fetch, options.cancel)
    metadata_url = (
        _secure_endpoint(options.authorization_server_metadata_url)
        if options.authorization_server_metadata_url
        else None
    )
    # With a configured metadata URL, discovery is not cached, so changing the URL applies at once.
    cached = None
    if metadata_url is None and provider.discovery_state is not None:
        cached = await provider.discovery_state()
    discovered: OAuthServerInfo
    if cached is not None and cached.get("authorizationServerUrl"):
        cached_metadata = cached.get("authorizationServerMetadata")
        if cached_metadata is None:
            cached_metadata = await discover_authorization_server_metadata(
                cached["authorizationServerUrl"],
                fetch=fetch,
                skip_issuer_validation=options.skip_issuer_validation,
            )
        discovered = {"authorizationServerUrl": cached["authorizationServerUrl"]}
        if cached_metadata is not None:
            discovered["authorizationServerMetadata"] = cached_metadata
        if cached.get("resourceMetadata") is not None:
            discovered["resourceMetadata"] = cached["resourceMetadata"]
    else:
        discovered = await discover_oauth_server_info(
            options.server_url,
            resource_metadata_url=options.resource_metadata_url,
            authorization_server_metadata_url=metadata_url.href if metadata_url is not None else None,
            fetch=fetch,
            skip_issuer_validation=options.skip_issuer_validation,
        )
    if metadata_url is None and provider.save_discovery_state is not None:
        state: OAuthDiscoveryState = {**discovered}  # type: ignore[typeddict-item]
        if options.resource_metadata_url:
            state["resourceMetadataUrl"] = parse_url(options.resource_metadata_url).href
        await provider.save_discovery_state(state)
    metadata = discovered.get("authorizationServerMetadata")
    resource_metadata = discovered.get("resourceMetadata")
    resource = select_resource(options.server_url, resource_metadata)
    # `or`: an empty scope (for example from `scopes_supported: []`) falls through to the next source.
    scope = (
        options.scope
        or " ".join((resource_metadata or {}).get("scopes_supported") or [])
        or provider.client_metadata.get("scope")
        or None
    )
    stored = await provider.client_information()
    client_document = (
        None
        if stored is not None or provider.client_metadata_document is None
        else provider.client_metadata_document(metadata)
    )
    if client_document is not None:
        url = parse_url(client_document.url)
        if url.protocol != "https:" or url.pathname == "/":
            raise RuntimeError("Invalid OAuth client metadata URL")
    client = stored if stored is not None else ({"client_id": client_document.url} if client_document else None)
    if client is None:
        if options.authorization_code:
            raise RuntimeError("OAuth client information is missing during code exchange")
        if provider.save_client_information is None:
            raise RuntimeError("OAuth client information cannot be persisted")
        client = await register_client(
            discovered["authorizationServerUrl"],
            metadata=metadata,
            client_metadata=provider.client_metadata,
            scope=scope,
            fetch=fetch,
        )
        await provider.save_client_information(client)
    # The document's redirect URI may differ from the provider's, for example by a server-specific path.
    redirect_url = client_document.redirect_url if client_document is not None else provider.redirect_url
    token_options = TokenRequestOptions(
        client_information=client,
        metadata=metadata,
        resource=resource,
        add_client_authentication=provider.add_client_authentication,
        fetch=fetch,
    )
    if options.authorization_code:
        # RFC 9207: never send a code from another authorization server to this one.
        iss = options.iss
        promised = (
            iss is not None or metadata is not None and metadata.get("authorization_response_iss_parameter_supported")
        )
        if metadata is not None and promised and iss != metadata["issuer"]:
            raise OAuthIssuerMismatchError(metadata["issuer"], iss)
        tokens = await exchange_authorization_code(
            discovered["authorizationServerUrl"],
            token_options,
            code=options.authorization_code,
            code_verifier=await provider.code_verifier(),
            redirect_url=redirect_url,
        )
        # A response without `scope` grants the requested scope (RFC 6749
        # §5.1). Recorded so a step-up can keep it. Callers pass the options
        # of the authorization request, so `scope` is what was requested.
        await provider.save_tokens(_with_scope(tokens, scope))
        return "AUTHORIZED"
    existing = None if options.skip_refresh else await provider.tokens()
    if existing and existing.get("refresh_token"):
        try:
            tokens = await refresh_authorization(
                discovered["authorizationServerUrl"], token_options, refresh_token=existing["refresh_token"]
            )
            # A refresh without `scope` keeps the scope of the grant (RFC 6749 §6).
            await provider.save_tokens(_with_scope(tokens, existing.get("scope")))
            return "AUTHORIZED"
        except OAuthInsecureEndpointError:
            raise
        except OAuthError as error:
            if (options.cancel is not None and options.cancel.cancelled) or error.code != "server_error":
                raise
        except Exception:
            # A cancelled refresh never falls back to a new authorization.
            if options.cancel is not None and options.cancel.cancelled:
                raise
            # Any other failure falls through to a new authorization, as in pi.
    state_value = await provider.state() if provider.state is not None else None
    authorization = start_authorization(
        discovered["authorizationServerUrl"],
        metadata=metadata,
        client_information=client,
        redirect_url=redirect_url,
        scope=scope,
        state=state_value,
        resource=resource,
    )
    await provider.save_code_verifier(authorization.code_verifier)
    await provider.redirect_to_authorization(authorization.authorization_url)
    return "REDIRECT"


async def authorize_mcp(provider: OAuthClientProvider, options: OAuthFlowOptions) -> OAuthFlowResult:
    try:
        return await _run_flow(provider, options)
    except OAuthError as error:
        if error.code in ("invalid_client", "unauthorized_client"):
            if provider.invalidate_credentials is not None:
                await provider.invalidate_credentials("all")
            return await _run_flow(provider, options)
        if error.code == "invalid_grant":
            if provider.invalidate_credentials is not None:
                await provider.invalidate_credentials("tokens")
            return await _run_flow(provider, options)
        raise


class _SharedRefresh:
    """One refresh, awaited by every request that hit a 401 meanwhile."""

    __slots__ = ("done", "error")

    def __init__(self) -> None:
        self.done = tonio.Event()
        self.error: Exception | None = None


def adapt_oauth_provider(provider: OAuthClientProvider) -> AuthProvider:
    """Auth provider for `StreamableHttpTransport`. After a 401 it refreshes
    the tokens, or raises `McpOAuthAuthorizationRequiredError` when the user
    has to authorize (again). Concurrent 401s share one refresh, and a
    request whose token was already replaced is just retried: with rotating
    refresh tokens, a second refresh with the old refresh token would fail
    and discard the new grant.

    The shared refresh runs detached and stores its outcome: a request that
    is cancelled while waiting for it does not stop it."""
    lock = threading.Lock()
    in_flight: _SharedRefresh | None = None

    async def token() -> str | None:
        tokens = await provider.tokens()
        return tokens.get("access_token") if tokens else None

    async def refresh(shared: _SharedRefresh, context: UnauthorizedContext, challenge: Any, insufficient: bool) -> None:
        nonlocal in_flight
        error: Exception | None = None
        try:
            granted = await provider.tokens() if insufficient else None
            result = await authorize_mcp(
                provider,
                OAuthFlowOptions(
                    server_url=context.server_url,
                    resource_metadata_url=challenge.get("resourceMetadataUrl"),
                    scope=(
                        step_up_scope((granted or {}).get("scope"), challenge.get("scope"))
                        if insufficient
                        else challenge.get("scope")
                    ),
                    fetch=context.fetch,
                    skip_refresh=insufficient,
                ),
            )
            if result == "REDIRECT":
                raise McpOAuthAuthorizationRequiredError()
        except Exception as caught:
            error = caught
        with lock:
            if in_flight is shared:
                in_flight = None
        shared.error = error
        shared.done.set()

    async def on_unauthorized(context: UnauthorizedContext) -> None:
        nonlocal in_flight
        challenge = parse_www_authenticate(context.response.headers.get("www-authenticate"))
        insufficient = challenge.get("error") == "insufficient_scope"
        with lock:
            refreshing = in_flight is not None
        if not insufficient and not refreshing and context.token is not None:
            current = await token()
            if current is not None and current != context.token:
                return
        with lock:
            shared = in_flight
            start = shared is None
            if shared is None:
                shared = in_flight = _SharedRefresh()
        if start:
            tonio.spawn.without_tracking(refresh(shared, context, challenge, insufficient))
        await shared.done.wait()
        if shared.error is not None:
            raise shared.error

    return AuthProvider(token=token, on_unauthorized=on_unauthorized)
