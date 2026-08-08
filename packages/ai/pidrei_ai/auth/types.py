"""Port of pi's auth types (packages/ai/src/auth/types.ts).

Interfaces with behavior (`ApiKeyAuth`, `OAuthAuth`) become dataclasses with
callable fields — provider factories assemble them from functions, exactly like
pi's object literals. Method inputs collapse to positional args:
pi `resolve({ctx, credential})` → `resolve(ctx, credential)`.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pidrei_ai.types import ProviderEnv, ProviderHeaders
from pidrei_ai.utils.cancel import CancelToken


@dataclass(slots=True)
class ModelAuth:
    """Request auth for a single model request.

    If a value cannot be expressed as `api_key`, `headers`, or `base_url`, it
    is provider config, not auth.
    """

    api_key: str | None = None
    headers: ProviderHeaders | None = None
    base_url: str | None = None


@dataclass(slots=True)
class ApiKeyCredential:
    """Stored api-key credential; `env` holds provider-scoped config values."""

    key: str | None = None
    env: ProviderEnv | None = None
    type: Literal["api_key"] = "api_key"


@dataclass(slots=True)
class OAuthCredential:
    """Stored canonical OAuth credential.

    pi's shape has an index signature for extra provider fields; those live in
    `extra` and are flattened back at the auth.json serialization boundary.
    """

    refresh: str
    access: str
    expires: int  # Unix timestamp in milliseconds
    extra: dict[str, Any] = field(default_factory=dict)
    type: Literal["oauth"] = "oauth"


# One type-tagged credential per provider — the shape of today's auth.json.
type Credential = ApiKeyCredential | OAuthCredential


@dataclass(slots=True)
class CredentialInfo:
    """Non-secret credential metadata for account/status enumeration."""

    provider_id: str
    type: Literal["api_key", "oauth"]


@dataclass(slots=True)
class AuthOperationOptions:
    """Optional cancellation for public auth and credential operations."""

    cancel: CancelToken | None = None


class CredentialStore(Protocol):
    """App-owned credential storage, keyed by provider id, one credential per
    provider. `modify` is the only write path (serialized read-modify-write);
    OAuth refresh runs inside `modify` so concurrent requests cannot
    double-refresh a rotated token.

    `read` returns None for missing entries; methods raise only on storage
    failure (wrapped by Models in `ModelsError` code "auth").
    """

    async def read(self, provider_id: str, options: AuthOperationOptions | None = None) -> Credential | None: ...

    async def list(self, options: AuthOperationOptions | None = None) -> list[CredentialInfo]: ...

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None: ...

    async def delete(self, provider_id: str, options: AuthOperationOptions | None = None) -> None: ...


class AuthContext(Protocol):
    """Environment access for auth resolution. Injectable for tests."""

    async def env(self, name: str) -> str | None: ...

    # Check whether a file exists. Supports a leading `~`.
    async def file_exists(self, path: str) -> bool: ...


@dataclass(slots=True)
class AuthResult:
    """Result of resolving auth for a model."""

    auth: ModelAuth
    # Provider-scoped environment/config values resolved from credentials and ambient context.
    env: ProviderEnv | None = None
    # Human-readable label for status UI: "ANTHROPIC_API_KEY", "OAuth", "~/.aws/credentials".
    source: str | None = None


@dataclass(slots=True)
class AuthCheck:
    type: Literal["api_key", "oauth"]
    source: str | None = None


type AuthType = Literal["api_key", "oauth"]


@dataclass(slots=True)
class AuthPromptOption:
    id: str
    label: str
    description: str | None = None


@dataclass(slots=True)
class AuthPrompt:
    """Prompt shown to the user during login.

    `cancel` lets the flow cancel a pending prompt when an out-of-band event
    resolves the step (e.g. a manual_code prompt raced against a callback
    server).
    """

    type: Literal["text", "secret", "select", "manual_code"]
    message: str
    placeholder: str | None = None
    options: list[AuthPromptOption] | None = None  # select only
    cancel: CancelToken | None = None


@dataclass(slots=True)
class AuthInfoLink:
    url: str
    label: str | None = None


@dataclass(slots=True)
class AuthEvent:
    type: Literal["info", "auth_url", "device_code", "progress"]
    message: str | None = None
    links: list[AuthInfoLink] | None = None
    url: str | None = None
    instructions: str | None = None
    user_code: str | None = None
    verification_uri: str | None = None
    interval_seconds: float | None = None
    expires_in_seconds: float | None = None


class AuthInteraction(Protocol):
    """Login interaction callbacks serving both api-key and OAuth flows.

    `prompt()` returns the entered/selected string (`select` returns the
    option id) and raises on cancel/abort.
    """

    cancel: CancelToken | None

    async def prompt(self, prompt: AuthPrompt) -> str: ...

    def notify(self, event: AuthEvent) -> None: ...


class ProviderAuthInteraction(Protocol):
    """Normalized interaction passed to provider login implementations:
    `cancel` is always present (pi's `AuthInteraction & { signal: AbortSignal }`)."""

    cancel: CancelToken

    async def prompt(self, prompt: AuthPrompt) -> str: ...

    def notify(self, event: AuthEvent) -> None: ...


@dataclass(slots=True)
class ApiKeyAuth:
    """Api-key auth: stored key/provider env plus ambient sources.

    `resolve(ctx, credential, cancel)` merges per field and returns None when
    the provider is not configured. Ambient-only providers omit `login`.
    `check` is an optional side-effect-free availability probe for providers
    whose `resolve` may execute commands.
    """

    name: str
    resolve: Callable[[AuthContext, ApiKeyCredential | None, CancelToken], Awaitable[AuthResult | None]]
    login: Callable[[ProviderAuthInteraction], Awaitable[ApiKeyCredential]] | None = None
    check: Callable[[AuthContext, ApiKeyCredential | None, CancelToken], Awaitable[AuthCheck | None]] | None = None


@dataclass(slots=True)
class OAuthAuth:
    """OAuth auth. The `refresh`/`to_auth` split lets Models own the locked
    refresh pattern: `refresh` produces a credential, `to_auth` derives request
    auth from whatever credential ends up stored.
    """

    name: str
    login: Callable[[ProviderAuthInteraction], Awaitable[OAuthCredential]]
    # Exchange the refresh token; network call, raises on failure. Runs under the store lock.
    refresh: Callable[[OAuthCredential, CancelToken], Awaitable[OAuthCredential]]
    # Side-effect-free derivation of request auth from a valid credential.
    to_auth: Callable[[OAuthCredential], Awaitable[ModelAuth]]
    # Whether access through this auth method is backed by a provider subscription.
    is_subscription: bool | None = None
    # Selector label for the OAuth login option.
    login_label: str | None = None


@dataclass(slots=True)
class ProviderAuth:
    """At least one of `api_key`/`oauth` must be present: even ambient-credential
    providers and keyless local servers provide `api_key` auth whose `resolve()`
    reports whether the provider is configured.
    """

    api_key: ApiKeyAuth | None = None
    oauth: OAuthAuth | None = None
