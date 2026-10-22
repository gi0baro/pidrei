"""Client identity (port of pi `utils/identity.ts`).

How requests identify the calling app to providers: the `User-Agent` and the
provider attribution headers (OpenRouter, NVIDIA NIM, OpenCode). The types
(`ClientIdentity`, `ClientAttribution`, `UserAgentMode`) live in `types.py`,
next to the request options that carry them.

The identity is pidrei's, not pi's: the User-Agent names this program and
`PIDREI_ATTRIBUTION` points at this project. Presenting as pi to a provider
would be misattribution, whatever a backend may or may not gate on. pi's
browser branch (`pi/<version> (browser)`) has no Python counterpart.

PiDrei's header dicts do not fold case, so adapters that build plain dicts
apply the identity through `set_default_identity_headers`, which skips a
header the model or caller already set under any casing.
"""

import platform
import sys
from collections.abc import Awaitable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlparse

from pidrei_ai.types import (
    AnyModel,
    AssistantImages,
    AssistantMessage,
    ClassifierContext,
    ClassifierModel,
    ClassifierOptions,
    ClassifierResult,
    ClientAttribution,
    ClientIdentity,
    Context,
    DeferredCancelOptions,
    DeferredFetchOptions,
    DeferredHandle,
    ImageModel,
    ImagesContext,
    ImagesOptions,
    Model,
    ModelType,
    ProviderRequestOptions,
    SimpleStreamOptions,
    StreamOptions,
    TranscriptContext,
)
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.utils.version import PIDREI_AI_VERSION


if TYPE_CHECKING:
    from pidrei_ai.auth.resolve import AuthResolutionOverrides
    from pidrei_ai.auth.types import (
        AuthCheck,
        AuthInteraction,
        AuthOperationOptions,
        AuthResult,
        AuthType,
        Credential,
        LoginOptions,
    )
    from pidrei_ai.registry import Models, ModelsRefreshOptions, ModelsRefreshResult, Provider


CLIENT_NAME = "pidrei"

# The `originator` the Codex adapter sends on every request and in the OAuth
# authorize URL, and the xAI OAuth referrer. Codex expects an originator; it is
# not attribution, so no identity changes it.
ORIGINATOR = CLIENT_NAME

#: pidrei's attribution, used by pidrei-agent and the coding agent.
PIDREI_ATTRIBUTION: ClientAttribution = {
    "name": CLIENT_NAME,
    "url": "https://github.com/gi0baro/pidrei",
    "categories": ("cli-agent",),
}

# The project's casing for its own name; NVIDIA's billing origin uses it.
_DISPLAY_NAME = "PiDrei"

_NVIDIA_NIM_HOST = "integrate.api.nvidia.com"
_OPENCODE_HOST = "opencode.ai"

# Node's `process.platform`/`process.arch` report "x64"/"arm64" where
# `platform.machine()` reports "x86_64"/"aarch64"; the raw Python value is sent.
_RUNTIME_DESCRIPTION = f"{sys.platform}; python/{platform.python_version()}; {platform.machine()}"


class ClientIdentityTarget(Protocol):
    """The provider and endpoint a request goes to."""

    @property
    def provider(self) -> str: ...

    @property
    def base_url(self) -> str: ...


_IDENTITY_FIELDS = ("user_agent", "attribution", "version")


def merge_client_identity(*sources: ClientIdentity | None) -> ClientIdentity | None:
    """Merge identities field by field; later sources win and absent fields are skipped."""
    merged: ClientIdentity | None = None
    for source in sources:
        if source is None:
            continue
        merged = ClientIdentity(**(merged or {}))
        for name in _IDENTITY_FIELDS:
            if name in source:
                merged[name] = source[name]
    return merged


def get_client_version(identity: ClientIdentity | None = None) -> str:
    """The version in the detailed `User-Agent`."""
    return identity.get("version", PIDREI_AI_VERSION) if identity is not None else PIDREI_AI_VERSION


def format_user_agent(identity: ClientIdentity | None = None) -> str:
    """The `User-Agent` value of an identity."""
    mode = identity.get("user_agent", "detailed") if identity is not None else "detailed"
    if isinstance(mode, dict):
        return mode["custom"]
    if mode == "short":
        return CLIENT_NAME
    return f"{CLIENT_NAME}/{get_client_version(identity)} ({_RUNTIME_DESCRIPTION})"


def _matches_host(base_url: str, expected_host: str) -> bool:
    try:
        return urlparse(base_url).hostname == expected_host
    except ValueError:
        return False


def _is_openrouter(target: ClientIdentityTarget) -> bool:
    return target.provider == "openrouter" or "openrouter.ai" in target.base_url


def _is_nvidia_nim(target: ClientIdentityTarget) -> bool:
    return target.provider == "nvidia" or _matches_host(target.base_url, _NVIDIA_NIM_HOST)


def _is_opencode(target: ClientIdentityTarget) -> bool:
    return (
        target.provider == "opencode"
        or target.provider == "opencode-go"
        or _matches_host(target.base_url, _OPENCODE_HOST)
    )


def _capitalized(name: str) -> str:
    """pi upper-cases the first letter; pidrei's own name keeps the project's casing."""
    return _DISPLAY_NAME if name == CLIENT_NAME else name[:1].upper() + name[1:]


def get_attribution_headers(target: ClientIdentityTarget, identity: ClientIdentity | None = None) -> dict[str, str]:
    """Provider attribution headers of an identity for a request target. Empty when attribution is off."""
    attribution = identity.get("attribution") if identity is not None else None
    if not attribution:
        return {}
    name = attribution["name"]
    if _is_openrouter(target):
        categories = attribution.get("categories")
        return {
            "HTTP-Referer": attribution["url"],
            "X-OpenRouter-Title": name,
            **({"X-OpenRouter-Categories": ",".join(categories)} if categories else {}),
        }
    # NVIDIA expects a capitalized origin, such as `PiDrei`.
    if _is_nvidia_nim(target):
        return {"X-BILLING-INVOKE-ORIGIN": _capitalized(name)}
    if _is_opencode(target):
        return {"x-opencode-client": name}
    return {}


def get_opencode_session_headers(target: ClientIdentityTarget, session_id: str | None) -> dict[str, str] | None:
    """OpenCode's per-conversation routing header for requests to OpenCode,
    including custom providers pointing at it. None without a session or for
    other targets."""
    return {"x-opencode-session": session_id} if session_id and _is_opencode(target) else None


def get_identity_headers(target: ClientIdentityTarget, identity: ClientIdentity | None = None) -> dict[str, str]:
    """Default headers an identity adds to a request: `User-Agent` and provider
    attribution. APIs apply them below model, provider, and request headers, so
    configured headers override them and None request headers remove them."""
    return {"User-Agent": format_user_agent(identity), **get_attribution_headers(target, identity)}


def set_default_identity_headers(
    headers: dict[str, Any], target: ClientIdentityTarget, identity: ClientIdentity | None = None
) -> None:
    """Add the identity headers the model or caller did not set.

    pi spreads `getIdentityHeaders()` in front of the other header sources and
    lets the provider SDK fold the names case-insensitively. A Python dict does
    not fold, so an override under any casing would otherwise reach the wire
    alongside the default; the check happens here instead. A None value is the
    caller's delete sentinel and counts as an override, as it does upstream.
    """
    present = {name.lower() for name in headers}
    for name, value in get_identity_headers(target, identity).items():
        if name.lower() not in present:
            headers[name] = value


class ModelsWithIdentity:
    """A `Models` whose requests use a default identity (see `with_client_identity`).

    Every member is listed: request methods merge the identity into their
    options, the rest delegate unchanged. A request method added to `Models`
    and not listed here fails loudly instead of silently sending no identity.
    The provider mutators delegate too, so a mutable `Models` stays mutable.
    """

    def __init__(self, models: Models, identity: ClientIdentity):
        self._models = models
        self._identity = identity

    # -- delegated unchanged ---------------------------------------------------

    def set_provider(self, provider: Provider) -> None:
        self._models.set_provider(provider)

    def delete_provider(self, id: str) -> None:
        self._models.delete_provider(id)

    def clear_providers(self) -> None:
        self._models.clear_providers()

    def get_providers(self) -> list[Provider]:
        return self._models.get_providers()

    def get_provider(self, id: str) -> Provider | None:
        return self._models.get_provider(id)

    def get_models(self, provider: str | None = None) -> list[Model]:
        return self._models.get_models(provider)

    def get_model(self, provider: str, id: str) -> Model | None:
        return self._models.get_model(provider, id)

    def get_all_models(self, provider: str | None = None) -> list[AnyModel]:
        return self._models.get_all_models(provider)

    def get_models_of_type(self, type: ModelType, provider: str | None = None) -> list[AnyModel]:
        return self._models.get_models_of_type(type, provider)

    def get_model_of_type(self, type: ModelType, provider: str, id: str) -> AnyModel | None:
        return self._models.get_model_of_type(type, provider, id)

    def refresh(self, options: ModelsRefreshOptions | None = None) -> Awaitable[ModelsRefreshResult]:
        return self._models.refresh(options)

    def check_auth(self, provider_id: str, options: AuthOperationOptions | None = None) -> Awaitable[AuthCheck | None]:
        return self._models.check_auth(provider_id, options)

    def get_available(
        self, provider_id: str | None = None, options: AuthOperationOptions | None = None
    ) -> Awaitable[list[Model]]:
        return self._models.get_available(provider_id, options)

    def get_available_of_type(
        self, type: ModelType, provider_id: str | None = None, options: AuthOperationOptions | None = None
    ) -> Awaitable[list[AnyModel]]:
        return self._models.get_available_of_type(type, provider_id, options)

    def get_all_available(
        self, provider_id: str | None = None, options: AuthOperationOptions | None = None
    ) -> Awaitable[list[AnyModel]]:
        return self._models.get_all_available(provider_id, options)

    def get_auth(
        self, provider_or_model: str | AnyModel, overrides: AuthResolutionOverrides | None = None
    ) -> Awaitable[AuthResult | None]:
        return self._models.get_auth(provider_or_model, overrides)

    def get_auth_for_provider(
        self,
        provider: Provider,
        provider_or_model: str | AnyModel,
        overrides: AuthResolutionOverrides | None = None,
    ) -> Awaitable[AuthResult | None]:
        return self._models.get_auth_for_provider(provider, provider_or_model, overrides)

    def login(
        self, provider_id: str, type: AuthType, interaction: AuthInteraction, options: LoginOptions | None = None
    ) -> Awaitable[Credential]:
        return self._models.login(provider_id, type, interaction, options)

    def logout(self, provider_id: str, options: AuthOperationOptions | None = None) -> Awaitable[None]:
        return self._models.logout(provider_id, options)

    # -- requests: the identity merges into the options ------------------------

    def _with_identity[TOptions: ProviderRequestOptions](
        self, options: TOptions | None, default: type[TOptions]
    ) -> TOptions:
        base = options if options is not None else default()
        return replace(base, identity=merge_client_identity(self._identity, base.identity))

    def stream(
        self, model: Model, context: Context | TranscriptContext, options: StreamOptions | None = None
    ) -> AssistantMessageEventStream:
        return self._models.stream(model, context, self._with_identity(options, StreamOptions))

    def complete(self, model: Model, context: Context, options: StreamOptions | None = None):
        return self._models.complete(model, context, self._with_identity(options, StreamOptions))

    def stream_simple(
        self, model: Model, context: Context | TranscriptContext, options: SimpleStreamOptions | None = None
    ) -> AssistantMessageEventStream:
        return self._models.stream_simple(model, context, self._with_identity(options, SimpleStreamOptions))

    def complete_simple(self, model: Model, context: Context, options: SimpleStreamOptions | None = None):
        return self._models.complete_simple(model, context, self._with_identity(options, SimpleStreamOptions))

    def stream_deferred(
        self, model: Model, handle: DeferredHandle, options: DeferredFetchOptions | None = None
    ) -> AssistantMessageEventStream:
        return self._models.stream_deferred(model, handle, self._with_identity(options, DeferredFetchOptions))

    def fetch_deferred(
        self, model: Model, handle: DeferredHandle, options: DeferredFetchOptions | None = None
    ) -> Awaitable[AssistantMessage]:
        return self._models.fetch_deferred(model, handle, self._with_identity(options, DeferredFetchOptions))

    def cancel_deferred(
        self, model: Model, handle: DeferredHandle, options: DeferredCancelOptions | None = None
    ) -> Awaitable[None]:
        return self._models.cancel_deferred(model, handle, self._with_identity(options, ProviderRequestOptions))

    def generate_images(
        self, model: ImageModel, context: ImagesContext, options: ImagesOptions | None = None
    ) -> Awaitable[AssistantImages]:
        return self._models.generate_images(model, context, self._with_identity(options, ImagesOptions))

    def classify(
        self, model: ClassifierModel, context: ClassifierContext, options: ClassifierOptions | None = None
    ) -> Awaitable[ClassifierResult]:
        return self._models.classify(model, context, self._with_identity(options, ClassifierOptions))


def with_client_identity(models: Models, identity: ClientIdentity) -> ModelsWithIdentity:
    """`models` with `identity` as the default identity of its requests: a
    request's `identity` overrides it field by field, and it overrides the
    identity `models` was created with. Everything else is delegated unchanged;
    a mutable `Models` stays mutable."""
    return ModelsWithIdentity(models, identity)
