"""Mirror of pi coding-agent src/core/model-registry.ts.

Synchronous compatibility facade exposed to extensions. Coding-agent
internals use ModelRuntime directly.
"""

from collections.abc import Awaitable
from dataclasses import dataclass

from pidrei_ai.auth.types import AuthOperationOptions, AuthResult
from pidrei_ai.registry import Provider
from pidrei_ai.types import (
    AnyModel,
    ClassifierContext,
    ClassifierModel,
    ClassifierOptions,
    ClassifierResult,
    Context,
    Model,
    ModelType,
    SimpleStreamOptions,
    StreamOptions,
)

from .model_runtime import ModelRuntime
from .provider_composer import AuthStatus, ProviderConfigInput, clear_api_key_cache
from .virtual_models import VirtualModelDefinition


__all__ = ["ModelRegistry", "ProviderConfigInput", "ResolvedRequestAuth", "clear_api_key_cache"]


@dataclass(slots=True)
class ResolvedRequestAuth:
    ok: bool
    api_key: str | None = None
    # None header values are deletion markers and are preserved (pi #7030);
    # pi-ai streams strip them at request time.
    headers: dict[str, str | None] | None = None
    # Credential-resolved endpoint (e.g. GitHub Copilot Business/Enterprise).
    base_url: str | None = None
    env: dict[str, str] | None = None
    error: str | None = None


class ModelRegistry:
    def __init__(self, runtime: ModelRuntime):
        self._runtime = runtime

    async def refresh(self, options=None):
        """Reload models.json asynchronously. Await before making synchronous registry reads."""
        return await self._runtime.refresh(options)

    def get_error(self) -> str | None:
        return self._runtime.get_error()

    def get_all(self) -> list[Model]:
        return list(self._runtime.get_models())

    def get_available(self) -> list[Model]:
        return list(self._runtime.get_available_snapshot())

    def find(self, provider: str, model_id: str) -> Model | None:
        return self._runtime.get_model(provider, model_id)

    def find_of_type(self, type: ModelType, provider: str, model_id: str) -> AnyModel | None:
        """Find a model of a non-chat type, e.g. `find_of_type("classifier", "typesafe", "jev-latest")`."""
        return self._runtime.get_model_of_type(type, provider, model_id)

    def has_configured_auth(self, model: Model) -> bool:
        return self._runtime.has_configured_auth(model.provider)

    async def get_api_key_and_headers(self, model: Model) -> ResolvedRequestAuth:
        try:
            resolution = await self._runtime.get_auth(model)
            if resolution is None:
                compatibility = await self._runtime.get_compatibility_request_config(model)
                if compatibility.auth_header:
                    return ResolvedRequestAuth(ok=False, error=f'No API key found for "{model.provider}"')
                return ResolvedRequestAuth(
                    ok=True, headers=dict(compatibility.headers) if compatibility.headers is not None else None
                )
            return ResolvedRequestAuth(
                ok=True,
                api_key=resolution.auth.api_key,
                headers=dict(resolution.auth.headers) if resolution.auth.headers is not None else None,
                base_url=resolution.auth.base_url,
                env=dict(resolution.env) if resolution.env is not None else None,
            )
        except Exception as error:
            cause = error.__cause__
            message = str(cause) if isinstance(cause, Exception) else str(error)
            if message == "authHeader requires a resolved API key":
                message = f'No API key found for "{model.provider}"'
            return ResolvedRequestAuth(ok=False, error=message)

    def get_provider_auth_status(self, provider: str) -> AuthStatus:
        return self._runtime.get_provider_auth_status(provider)

    def get_provider(self, provider: str) -> Provider | None:
        return self._runtime.get_provider(provider)

    def stream(self, model: Model, context: Context, options: StreamOptions | None = None):
        """Stream through the configured provider with request-time authentication."""
        return self._runtime.stream(model, context, options)

    def stream_simple(self, model: Model, context: Context, options: SimpleStreamOptions | None = None):
        """Stream with provider-neutral options and request-time authentication."""
        return self._runtime.stream_simple(model, context, options)

    def complete(self, model: Model, context: Context, options: StreamOptions | None = None):
        return self._runtime.complete(model, context, options)

    def get_models_of_type(self, type: ModelType, provider: str | None = None) -> list[AnyModel]:
        """Every known model of a type (chat, image, classifier), optionally for one provider."""
        return self._runtime.get_models_of_type(type, provider)

    def get_available_of_type(
        self, type: ModelType, provider: str | None = None, options: AuthOperationOptions | None = None
    ) -> Awaitable[list[AnyModel]]:
        """Models of a type whose provider has working credentials."""
        return self._runtime.get_available_of_type(type, provider, options)

    def get_model_of_type(self, type: ModelType, provider: str, model_id: str) -> AnyModel | None:
        return self._runtime.get_model_of_type(type, provider, model_id)

    def classify(
        self, model: ClassifierModel, context: ClassifierContext, options: ClassifierOptions | None = None
    ) -> Awaitable[ClassifierResult]:
        """Classify structured state with request-time authentication. Never raises."""
        return self._runtime.classify(model, context, options)

    def get_provider_display_name(self, provider: str) -> str:
        entry = self._runtime.get_provider(provider)
        return entry.name if entry is not None and entry.name is not None else provider

    async def get_provider_auth(self, provider: str) -> AuthResult | None:
        return await self._runtime.get_auth(provider)

    async def get_api_key_for_provider(self, provider: str) -> str | None:
        try:
            resolution = await self._runtime.get_auth(provider)
            return resolution.auth.api_key if resolution is not None else None
        except Exception:
            return None

    def is_using_oauth(self, model: Model) -> bool:
        return self._runtime.is_using_oauth(model.provider)

    def register_provider(self, provider_or_name: Provider | str, config: ProviderConfigInput | None = None) -> None:
        if isinstance(provider_or_name, str):
            if not config:
                raise Exception("Provider config is required when registering by name")
            self._runtime.register_provider(provider_or_name, config)
            return
        self._runtime.register_native_provider(provider_or_name)

    def unregister_provider(self, provider_name: str) -> None:
        self._runtime.unregister_provider(provider_name)

    def register_virtual_model(self, definition: VirtualModelDefinition) -> None:
        self._runtime.register_virtual_model(definition)

    def unregister_virtual_model(self, provider_name: str, model_id: str) -> None:
        self._runtime.unregister_virtual_model(provider_name, model_id)

    def get_registered_provider_config(self, provider_name: str) -> ProviderConfigInput | None:
        return self._runtime.get_registered_provider_config(provider_name)

    def get_registered_native_provider(self, provider_name: str) -> Provider | None:
        return self._runtime.get_registered_native_provider(provider_name)

    def get_registered_provider_ids(self) -> list[str]:
        return self._runtime.get_registered_provider_ids()
