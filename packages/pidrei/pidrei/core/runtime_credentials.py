"""Mirror of pi coding-agent src/core/runtime-credentials.ts."""

import threading
from collections.abc import Awaitable, Callable

from pidrei_ai.auth.types import ApiKeyCredential, AuthOperationOptions, Credential, CredentialInfo, CredentialStore


class RuntimeCredentials(CredentialStore):
    """Async credential store overlay for non-persistent runtime API keys."""

    def __init__(self, store: CredentialStore):
        self._store = store
        # Immutable snapshot swapped on write (never mutated in place):
        # readers pin one attribute read and take no lock; the guard
        # serializes writers only (PROPER_MT_DESIGN.md step 3).
        self._overrides: dict[str, str] = {}
        self._guard = threading.Lock()

    def set_runtime_api_key(self, provider_id: str, api_key: str) -> None:
        with self._guard:
            self._overrides = {**self._overrides, provider_id: api_key}

    def remove_runtime_api_key(self, provider_id: str) -> None:
        with self._guard:
            overrides = dict(self._overrides)
            overrides.pop(provider_id, None)
            self._overrides = overrides

    def has_runtime_api_key(self, provider_id: str) -> bool:
        return provider_id in self._overrides

    async def read(self, provider_id: str, options: AuthOperationOptions | None = None) -> Credential | None:
        if options is not None and options.cancel is not None:
            options.cancel.raise_if_cancelled()
        override = self._overrides.get(provider_id)
        if override is not None:
            return ApiKeyCredential(key=override)
        return await self._store.read(provider_id, options)

    async def list(self, options: AuthOperationOptions | None = None) -> list[CredentialInfo]:
        entries = {entry.provider_id: entry for entry in await self._store.list(options)}
        if options is not None and options.cancel is not None:
            options.cancel.raise_if_cancelled()
        for provider_id in self._overrides:
            entries[provider_id] = CredentialInfo(provider_id=provider_id, type="api_key")
        return list(entries.values())

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None:
        return await self._store.modify(provider_id, fn, options)

    async def delete(self, provider_id: str, options: AuthOperationOptions | None = None) -> None:
        if options is not None and options.cancel is not None:
            options.cancel.raise_if_cancelled()
        await self._store.delete(provider_id, options)
        with self._guard:
            overrides = dict(self._overrides)
            overrides.pop(provider_id, None)
            self._overrides = overrides
