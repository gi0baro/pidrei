"""Mirror of pi coding-agent src/core/models-store.ts."""

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any

import tonio.colored as tonio
from tonio.colored import sync

from pidrei_ai.auth.types import AuthOperationOptions
from pidrei_ai.models_store import ModelsStore, ModelsStoreEntry, ModelsStoreOperationOptions

from ..config import get_agent_dir
from ..utils.paths import get_file_revision, normalize_path
from ..utils.text import strip_bom
from .model_wire import model_to_dict, parse_model_dict


def _auth_options(options: ModelsStoreOperationOptions | None) -> AuthOperationOptions | None:
    return AuthOperationOptions(cancel=options.cancel) if options is not None else None


@dataclass(slots=True)
class _ModelsFileReadState:
    data: dict[str, Any] = field(default_factory=dict)
    revision: str | None = None
    lock: sync.Lock = field(default_factory=sync.Lock)


# Per-path map instead of pi's single shared slot — see the auth-storage
# counterpart for the pytest-shared-process rationale.
_shared_models_file_read_states: dict[str, _ModelsFileReadState] = {}


class InMemoryCodingAgentModelsStore(ModelsStore):
    def __init__(self) -> None:
        self._entries: dict[str, ModelsStoreEntry] = {}

    async def read(
        self, provider_id: str, options: ModelsStoreOperationOptions | None = None
    ) -> ModelsStoreEntry | None:
        if options is not None and options.cancel is not None:
            options.cancel.raise_if_cancelled()
        entry = self._entries.get(provider_id)
        return copy.deepcopy(entry) if entry is not None else None

    async def write(
        self, provider_id: str, entry: ModelsStoreEntry, options: ModelsStoreOperationOptions | None = None
    ) -> None:
        if options is not None and options.cancel is not None:
            options.cancel.raise_if_cancelled()
        self._entries[provider_id] = copy.deepcopy(entry)

    async def delete(self, provider_id: str, options: ModelsStoreOperationOptions | None = None) -> None:
        if options is not None and options.cancel is not None:
            options.cancel.raise_if_cancelled()
        self._entries.pop(provider_id, None)


def _entry_to_dict(entry: ModelsStoreEntry) -> dict[str, Any]:
    raw: dict[str, Any] = {"models": [model_to_dict(model) for model in entry.models]}
    if entry.last_modified is not None:
        raw["lastModified"] = entry.last_modified
    if entry.checked_at is not None:
        raw["checkedAt"] = entry.checked_at
    if entry.etag is not None:
        raw["etag"] = entry.etag
    return raw


def _entry_from_dict(raw: dict[str, Any]) -> ModelsStoreEntry:
    return ModelsStoreEntry(
        models=[parse_model_dict(model) for model in raw.get("models", [])],
        last_modified=raw.get("lastModified"),
        checked_at=raw.get("checkedAt"),
        etag=raw.get("etag"),
    )


class FileModelsStore(ModelsStore):
    """Locked JSON-backed storage for dynamically refreshed provider catalogs."""

    def __init__(self, path: str | None = None):
        # lazy: import cycle within core
        from .auth_storage import FileAuthStorageBackend

        if path is None:
            path = os.path.join(get_agent_dir(), "models-store.json")
        self._path = normalize_path(path)
        self._storage = FileAuthStorageBackend(self._path)
        self._read_state = _shared_models_file_read_states.setdefault(self._path, _ModelsFileReadState())

    def _parse(self, content: str | None) -> dict[str, Any]:
        return json.loads(strip_bom(content)) if content else {}

    def _update_read_state(self, data: dict[str, Any], revision: str | None = None) -> None:
        self._read_state.data = data
        self._read_state.revision = revision

    async def _reload_from_storage(self, options: ModelsStoreOperationOptions | None = None) -> dict[str, Any]:
        async def under_lock(content: str | None) -> tuple[dict[str, Any], None]:
            data = self._parse(content)
            revision = await tonio.spawn_blocking(get_file_revision, self._path)
            self._update_read_state(data, revision)
            return data, None

        return await self._storage.with_lock_async(under_lock, _auth_options(options))

    async def _read_latest(self, options: ModelsStoreOperationOptions | None = None) -> dict[str, Any]:
        cancel = options.cancel if options is not None else None
        if cancel is not None:
            cancel.raise_if_cancelled()
        state = self._read_state
        revision = await tonio.spawn_blocking(get_file_revision, self._path)
        if revision is not None and revision == state.revision:
            return state.data

        # pi dedups concurrent reloads with a shared in-flight promise; that
        # check-then-set is not atomic on worker threads. A FIFO lock with a
        # revision re-check gives the same "one reload per revision" result:
        # readers queued behind the reloader see its revision and return.
        async with state.lock:
            if cancel is not None:
                cancel.raise_if_cancelled()
            revision = await tonio.spawn_blocking(get_file_revision, self._path)
            if revision is not None and revision == state.revision:
                return state.data
            return await self._reload_from_storage(options)

    async def read(
        self, provider_id: str, options: ModelsStoreOperationOptions | None = None
    ) -> ModelsStoreEntry | None:
        raw = (await self._read_latest(options)).get(provider_id)
        if options is not None and options.cancel is not None:
            options.cancel.raise_if_cancelled()
        return _entry_from_dict(raw) if raw is not None else None

    async def write(
        self, provider_id: str, entry: ModelsStoreEntry, options: ModelsStoreOperationOptions | None = None
    ) -> None:
        latest: list[dict[str, Any] | None] = [None]

        async def under_lock(content: str | None) -> tuple[None, str]:
            current = self._parse(content)
            current[provider_id] = _entry_to_dict(entry)
            latest[0] = current
            return None, json.dumps(current, indent=2)

        await self._storage.with_lock_async(under_lock, _auth_options(options))
        if latest[0] is not None:
            self._update_read_state(latest[0])

    async def delete(self, provider_id: str, options: ModelsStoreOperationOptions | None = None) -> None:
        latest: list[dict[str, Any] | None] = [None]

        async def under_lock(content: str | None) -> tuple[None, str]:
            current = self._parse(content)
            current.pop(provider_id, None)
            latest[0] = current
            return None, json.dumps(current, indent=2)

        await self._storage.with_lock_async(under_lock, _auth_options(options))
        if latest[0] is not None:
            self._update_read_state(latest[0])
