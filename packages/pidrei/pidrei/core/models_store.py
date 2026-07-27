"""Mirror of pi coding-agent src/core/models-store.ts."""

import json
import os
from collections.abc import Awaitable
from typing import Any

import tonio.colored as tonio

from pidrei_ai.models_store import ModelsStore, ModelsStoreEntry

from ..config import get_agent_dir
from .model_wire import model_to_dict, parse_model_dict


class InMemoryCodingAgentModelsStore(ModelsStore):
    def __init__(self) -> None:
        self._entries: dict[str, ModelsStoreEntry] = {}

    async def read(self, provider_id: str) -> ModelsStoreEntry | None:
        return self._entries.get(provider_id)

    async def write(self, provider_id: str, entry: ModelsStoreEntry) -> None:
        self._entries[provider_id] = entry

    async def delete(self, provider_id: str) -> None:
        self._entries.pop(provider_id, None)


def _entry_to_dict(entry: ModelsStoreEntry) -> dict[str, Any]:
    raw: dict[str, Any] = {"models": [model_to_dict(model) for model in entry.models]}
    if entry.last_modified is not None:
        raw["lastModified"] = entry.last_modified
    if entry.checked_at is not None:
        raw["checkedAt"] = entry.checked_at
    return raw


def _entry_from_dict(raw: dict[str, Any]) -> ModelsStoreEntry:
    return ModelsStoreEntry(
        models=[parse_model_dict(model) for model in raw.get("models", [])],
        last_modified=raw.get("lastModified"),
        checked_at=raw.get("checkedAt"),
    )


class FileModelsStore(ModelsStore):
    """Locked JSON-backed storage for dynamically refreshed provider catalogs."""

    def __init__(self, path: str | None = None):
        # lazy: import cycle within core
        from .auth_storage import FileAuthStorageBackend

        if path is None:
            path = os.path.join(get_agent_dir(), "models-store.json")
        self._storage = FileAuthStorageBackend(path)

    def _parse(self, content: str | None) -> dict[str, Any]:
        return json.loads(content) if content else {}

    def read(self, provider_id: str) -> Awaitable[ModelsStoreEntry | None]:
        """Read one provider's catalog.

        The callback is pure, so the whole lock-read-parse cycle is one blocking
        unit handed to the pool — the same shape as `ProjectTrustStore.get_entry`,
        and cheaper than `with_lock_async`, which cannot offload as a unit
        because it has to await its callback.

        Sync def returning the awaitable, per the standing rule; `ModelsStore` is
        a Protocol, so this satisfies `async def read`.
        """

        def under_lock(content: str | None) -> tuple[ModelsStoreEntry | None, None]:
            raw = self._parse(content).get(provider_id)
            return (_entry_from_dict(raw) if raw is not None else None), None

        return tonio.spawn_blocking(self._storage.with_lock, under_lock)

    async def write(self, provider_id: str, entry: ModelsStoreEntry) -> None:
        async def under_lock(content: str | None) -> tuple[None, str]:
            current = self._parse(content)
            current[provider_id] = _entry_to_dict(entry)
            return None, json.dumps(current, indent=2)

        await self._storage.with_lock_async(under_lock)

    async def delete(self, provider_id: str) -> None:
        async def under_lock(content: str | None) -> tuple[None, str]:
            current = self._parse(content)
            current.pop(provider_id, None)
            return None, json.dumps(current, indent=2)

        await self._storage.with_lock_async(under_lock)
