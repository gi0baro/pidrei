"""Port of pi's images-models.ts: the image-side counterpart of `registry.Models`."""

import threading
import time
from dataclasses import replace
from typing import Any, Protocol

import tonio.colored as tonio

from pidrei_ai.auth.context import default_provider_auth_context
from pidrei_ai.auth.credential_store import InMemoryCredentialStore
from pidrei_ai.auth.resolve import AuthResolutionOverrides, ModelsError, resolve_provider_auth
from pidrei_ai.auth.types import AuthResult, ProviderAuth
from pidrei_ai.types import AssistantImages, ImagesContext, ImagesModel, ImagesOptions


class ImagesProvider(Protocol):
    """An image-generation provider: the image-side counterpart of `Provider`."""

    id: str
    name: str
    auth: ProviderAuth

    def get_models(self) -> list[ImagesModel]: ...

    async def generate_images(
        self, model: ImagesModel, context: ImagesContext, options: ImagesOptions | None = None
    ) -> AssistantImages: ...


class ImagesModels:
    """Runtime collection of image providers plus auth application."""

    def __init__(self, credentials=None, auth_context=None):
        self._providers: dict[str, Any] = {}
        self._credentials = credentials or InMemoryCredentialStore()
        self._auth_context = auth_context or default_provider_auth_context()
        # pi relies on JavaScript's single thread; provider mutation can race
        # with a read from another turn here.
        self._guard = threading.Lock()

    # -- provider collection ---------------------------------------------------

    def set_provider(self, provider: Any) -> None:
        with self._guard:
            self._providers[provider.id] = provider

    def delete_provider(self, id: str) -> None:
        with self._guard:
            self._providers.pop(id, None)

    def clear_providers(self) -> None:
        with self._guard:
            self._providers.clear()

    def get_providers(self) -> list[Any]:
        with self._guard:
            return list(self._providers.values())

    def get_provider(self, id: str) -> Any | None:
        with self._guard:
            return self._providers.get(id)

    # -- models ----------------------------------------------------------------

    def get_models(self, provider: str | None = None) -> list[ImagesModel]:
        """Best-effort: a provider whose `get_models()` raises yields no models."""
        if provider is not None:
            entry = self.get_provider(provider)
            if entry is None:
                return []
            try:
                return list(entry.get_models())
            except Exception:
                return []

        models: list[ImagesModel] = []
        for entry in self.get_providers():
            try:
                models.extend(entry.get_models())
            except Exception:
                # Best-effort: ill-behaved providers yield no models.
                pass
        return models

    def get_model(self, provider: str, id: str) -> ImagesModel | None:
        return next((model for model in self.get_models(provider) if model.id == id), None)

    async def refresh(self, provider: str | None = None) -> None:
        if provider is not None:
            entry = self.get_provider(provider)
            refresh_models = getattr(entry, "refresh_models", None) if entry is not None else None
            if refresh_models is None:
                return
            try:
                await refresh_models()
            except ModelsError:
                raise
            except Exception as error:
                raise ModelsError("model_source", f"Model refresh failed for {provider}") from error
            return

        # Cannot raise: every provider's refresh is isolated, as pi's
        # `Promise.allSettled` is.
        async def _safe(entry) -> None:
            refresh_models = getattr(entry, "refresh_models", None)
            if refresh_models is None:
                return
            try:
                await refresh_models()
            except Exception:
                pass

        entries = self.get_providers()
        if entries:
            await tonio.spawn(*[_safe(entry) for entry in entries])

    # -- auth / generation -----------------------------------------------------

    async def get_auth(self, provider_or_model: Any, overrides=None) -> AuthResult | None:
        provider_id = provider_or_model if isinstance(provider_or_model, str) else provider_or_model.provider
        provider = self.get_provider(provider_id)
        if provider is None:
            return None
        return await resolve_provider_auth(provider, self._credentials, self._auth_context, overrides)

    async def generate_images(
        self, model: ImagesModel, context: ImagesContext, options: ImagesOptions | None = None
    ) -> AssistantImages:
        """Never raises; failures come back as an error `AssistantImages`."""
        try:
            provider = self.get_provider(model.provider)
            if provider is None:
                raise ModelsError("provider", f"Unknown provider: {model.provider}")

            resolution = await self.get_auth(
                model,
                AuthResolutionOverrides(
                    api_key=options.api_key if options else None,
                    env=options.env if options else None,
                ),
            )
            auth = resolution.auth if resolution is not None else None
            if not auth:
                return await provider.generate_images(model, context, options)

            request_model = replace(model, base_url=auth.base_url) if auth.base_url else model

            # Explicit request options win per field; headers/env merge per key.
            api_key = (options.api_key if options else None) or auth.api_key
            headers = None
            if auth.headers or (options and options.headers):
                headers = {**(auth.headers or {}), **((options.headers if options else None) or {})}
            env = None
            if (resolution is not None and resolution.env) or (options and options.env):
                env = {
                    **((resolution.env if resolution else None) or {}),
                    **((options.env if options else None) or {}),
                }

            merged = (
                replace(options, api_key=api_key, headers=headers, env=env)
                if options
                else ImagesOptions(api_key=api_key, headers=headers, env=env)
            )
            return await provider.generate_images(request_model, context, merged)
        except Exception as error:
            return AssistantImages(
                api=model.api,
                provider=model.provider,
                model=model.id,
                output=[],
                stop_reason="error",
                error_message=str(error),
                timestamp=int(time.time() * 1000),
            )


def create_images_models(credentials=None, auth_context=None) -> ImagesModels:
    return ImagesModels(credentials=credentials, auth_context=auth_context)


class _ImagesProviderImpl:
    __slots__ = ("_api", "_guard", "_inflight", "_models", "auth", "id", "name", "refresh_models")

    def __init__(self, id, name, auth, models, api):
        self.id = id
        self.name = name
        self.auth = auth
        self._models = list(models)
        self._api = api
        self._guard = threading.Lock()
        self._inflight: tuple[Any, list] | None = None

    def get_models(self) -> list[ImagesModel]:
        with self._guard:
            return list(self._models)

    async def generate_images(self, model, context, options=None):
        return await self._api.generate_images(model, context, options)


def create_images_provider(
    *,
    id: str,
    name: str | None = None,
    auth: ProviderAuth,
    models: list[ImagesModel],
    api: Any,
    refresh_models=None,
) -> Any:
    """Builds an image-generation provider from parts."""
    provider = _ImagesProviderImpl(id, name or id, auth, models, api)

    if refresh_models is not None:

        async def _refresh() -> None:
            """Concurrent calls share one fetch, as pi's `inflightRefresh ??=` does.

            pi can hand out the same promise; here the followers wait on an event
            the leader sets, and the leader's outcome is replayed to them so a
            failure is not silently seen as success.
            """
            with provider._guard:
                inflight = provider._inflight
                leader = inflight is None
                if leader:
                    inflight = (tonio.Event(), [])
                    provider._inflight = inflight

            event, outcome = inflight
            if not leader:
                await event.wait()
                if outcome and isinstance(outcome[0], BaseException):
                    raise outcome[0]
                return

            try:
                fetched = await refresh_models()
                with provider._guard:
                    provider._models = list(fetched)
                outcome.append(None)
            except BaseException as error:
                outcome.append(error)
                raise
            finally:
                with provider._guard:
                    provider._inflight = None
                event.set()

        provider.refresh_models = _refresh  # type: ignore[attr-defined]

    return provider
