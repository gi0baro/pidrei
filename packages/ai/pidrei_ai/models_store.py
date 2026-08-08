"""Port of pi's models store (packages/ai/src/models-store.ts)."""

import copy
from dataclasses import dataclass
from typing import Protocol

from pidrei_ai.types import Model
from pidrei_ai.utils.cancel import CancelToken


@dataclass(slots=True)
class ModelsStoreEntry:
    models: list[Model]
    # Unix timestamp from the remote catalog's Last-Modified header.
    last_modified: int | None = None
    # Unix timestamp of the last completed remote check.
    checked_at: int | None = None
    # Opaque validator from the remote catalog's ETag header, stored verbatim
    # (quotes included) and echoed back as If-None-Match.
    etag: str | None = None


@dataclass(slots=True)
class ModelsStoreOperationOptions:
    cancel: CancelToken | None = None


class ModelsStore(Protocol):
    """Persistent model catalogs keyed by provider ID."""

    async def read(
        self, provider_id: str, options: ModelsStoreOperationOptions | None = None
    ) -> ModelsStoreEntry | None: ...

    async def write(
        self, provider_id: str, entry: ModelsStoreEntry, options: ModelsStoreOperationOptions | None = None
    ) -> None: ...

    async def delete(self, provider_id: str, options: ModelsStoreOperationOptions | None = None) -> None: ...


class InMemoryModelsStore(ModelsStore):
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
