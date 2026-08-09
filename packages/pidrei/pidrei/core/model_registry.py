"""Mirror of pi coding-agent src/core/model-registry.ts.

Synchronous compatibility facade exposed to extensions. Coding-agent
internals use ModelRuntime directly.
"""

from dataclasses import dataclass

from pidrei_ai.auth.types import AuthResult
from pidrei_ai.registry import Provider
from pidrei_ai.types import Context, Model, StreamOptions

from .model_runtime import ModelRuntime
from .provider_composer import AuthStatus, ProviderConfigInput, clear_api_key_cache


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

    def complete(self, model: Model, context: Context, options: StreamOptions | None = None):
        return self._runtime.complete(model, context, options)

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

    def get_registered_provider_config(self, provider_name: str) -> ProviderConfigInput | None:
        return self._runtime.get_registered_provider_config(provider_name)

    def get_registered_native_provider(self, provider_name: str) -> Provider | None:
        return self._runtime.get_registered_native_provider(provider_name)

    def get_registered_provider_ids(self) -> list[str]:
        return self._runtime.get_registered_provider_ids()
