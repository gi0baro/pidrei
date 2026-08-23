"""Mirror of pi coding-agent src/core/remote-catalog-provider.ts.

Adds a persisted pi.dev catalog overlay to a static built-in provider. The
HTTP transport is injectable for tests; the default uses the shared punkreq
client (pidrei-ai's HTTP seam).
"""

import json
import time
import urllib.parse
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

from pidrei_ai.api.lazy import call_stream_into
from pidrei_ai.models_store import ModelsStoreEntry
from pidrei_ai.registry import ModelsPublication, Provider, RefreshModelsContext
from pidrei_ai.types import Model
from pidrei_ai.utils.abort import run_cancellable

from ..config import VERSION
from ..utils.management_http import fetch_with_retry
from ..utils.user_agent import get_pidrei_user_agent
from .model_wire import parse_model_dict


DEFAULT_CATALOG_BASE_URL = "https://pi.dev"
REMOTE_CATALOG_REFRESH_INTERVAL_MS = 4 * 60 * 60 * 1000


@dataclass(slots=True)
class CatalogResponse:
    status: int
    headers: dict[str, str]
    body: str


async def _default_fetch(url: str, headers: dict[str, str], cancel: Any) -> CatalogResponse:
    async def _request() -> Any:
        response = await fetch_with_retry(url, headers=headers)
        return response, await response.read()

    # The refresh watchdog (`model_runtime`) cancels this token; the request
    # is unwound wherever it is parked instead of running to completion.
    response, body = await run_cancellable(_request(), cancel)
    return CatalogResponse(
        status=response.status_code,
        headers={name.lower(): value for name, value in response.headers.items()},
        body=body.decode("utf-8", "replace") if isinstance(body, bytes) else body,
    )


def _now_ms() -> int:
    return int(time.time() * 1000)


def _merge_models(baseline: list[Model], dynamic: list[Model]) -> list[Model]:
    merged = list(baseline)
    for model in dynamic:
        index = next((i for i, entry in enumerate(merged) if entry.id == model.id), -1)
        if index >= 0:
            merged[index] = model
        else:
            merged.append(model)
    return merged


def _parse_catalog(provider_id: str, value: Any) -> list[Model]:
    if isinstance(value, list):
        entries = value
    elif isinstance(value, dict) and isinstance(value.get("models"), list):
        entries = value["models"]
    elif isinstance(value, dict):
        entries = list(value.values())
    else:
        raise Exception(f'Invalid model catalog for provider "{provider_id}"')  # noqa: TRY004
    return [
        parse_model_dict({**entry, "provider": provider_id})
        for entry in entries
        if isinstance(entry, dict) and "id" in entry
    ]


def _remote_models(entry: ModelsStoreEntry | None, local_generated_at: int | None) -> list[Model]:
    if entry is None:
        return []
    if local_generated_at is not None and (entry.last_modified is None or entry.last_modified <= local_generated_at):
        return []
    return list(entry.models)


def _parse_last_modified(value: str | None) -> int:
    if not value:
        return 0
    try:
        return int(parsedate_to_datetime(value).timestamp() * 1000)
    except TypeError, ValueError:
        return 0


class RemoteCatalogProvider:
    """Provider wrapper: static base catalog plus the persisted remote overlay."""

    def __init__(self, provider: Provider, catalog_base_url: str, local_generated_at: int | None, fetch: Any):
        self._provider = provider
        self._catalog_base_url = catalog_base_url
        self._local_generated_at = local_generated_at
        self._fetch = fetch
        self._dynamic_models: list[Model] = []

        self.id = provider.id
        self.name = provider.name
        self.base_url = provider.base_url
        self.headers = provider.headers
        self.auth = provider.auth
        self.filter_models = provider.filter_models

    @property
    def has_dynamic_models(self) -> bool:
        return True

    def get_models(self) -> list[Model]:
        return _merge_models(self._provider.get_models(), self._dynamic_models)

    def stream(self, model: Model, context: Any, options: Any = None, *, into: Any = None) -> Any:
        if into is None:
            return self._provider.stream(model, context, options)
        return call_stream_into(self._provider.stream, model, context, options, into=into)

    def stream_simple(self, model: Model, context: Any, options: Any = None, *, into: Any = None) -> Any:
        if into is None:
            return self._provider.stream_simple(model, context, options)
        return call_stream_into(self._provider.stream_simple, model, context, options, into=into)

    async def refresh_models(self, context: RefreshModelsContext) -> None:
        stored = context.stored
        restored = [
            model for model in _remote_models(stored, self._local_generated_at) if model.provider == self._provider.id
        ]

        def _apply_restored() -> None:
            self._dynamic_models = restored

        if not await context.publish(ModelsPublication(update=_apply_restored)):
            return
        if not context.allow_network or context.cancel.cancelled:
            return
        if (
            not context.force
            and stored is not None
            and stored.checked_at is not None
            and stored.last_modified is not None
            and _now_ms() - stored.checked_at < REMOTE_CATALOG_REFRESH_INTERVAL_MS
        ):
            return

        # Only revalidate when a cached body backs the validator, so a 304 can
        # never leave the overlay empty.
        validator = stored.etag if stored is not None and stored.models else None
        url = urllib.parse.urljoin(
            self._catalog_base_url, f"/api/models/providers/{urllib.parse.quote(self._provider.id, safe='')}"
        )
        headers = {"accept": "application/json", "User-Agent": get_pidrei_user_agent(VERSION)}
        if validator is not None:
            headers["if-none-match"] = validator
        response = await self._fetch(url, headers, context.cancel)
        if context.cancel.cancelled:
            return
        checked_at = _now_ms()
        stored_models = list(stored.models) if stored is not None else []
        # Unchanged: dynamic_models already holds the stored overlay, so only
        # the freshness window moves.
        if response.status == 304 and stored is not None:
            await context.publish(
                ModelsPublication(
                    persist=ModelsStoreEntry(
                        models=stored_models,
                        checked_at=checked_at,
                        last_modified=stored.last_modified,
                        etag=stored.etag,
                    )
                )
            )
            return
        if response.status in (404, 501):
            await context.publish(
                ModelsPublication(
                    persist=ModelsStoreEntry(models=stored_models, checked_at=checked_at, last_modified=0)
                )
            )
            return
        if not (200 <= response.status < 300):
            # Transient failure: the cached body and its validator stay valid,
            # so keep the etag and let the next refresh revalidate instead of
            # downloading the catalog.
            await context.publish(
                ModelsPublication(
                    persist=ModelsStoreEntry(
                        models=stored_models,
                        checked_at=checked_at,
                        last_modified=stored.last_modified if stored is not None else None,
                        etag=stored.etag if stored is not None else None,
                    )
                )
            )
            raise Exception(f"Model catalog request failed for {self._provider.id}: {response.status}")
        refreshed = _parse_catalog(self._provider.id, json.loads(response.body))
        last_modified = _parse_last_modified(response.headers.get("last-modified"))
        if context.cancel.cancelled:
            return
        entry = ModelsStoreEntry(
            models=refreshed,
            checked_at=checked_at,
            last_modified=last_modified,
            etag=response.headers.get("etag"),
        )
        published = _remote_models(entry, self._local_generated_at)

        def _apply_refreshed() -> None:
            self._dynamic_models = published

        await context.publish(ModelsPublication(persist=entry, update=_apply_refreshed))


def with_remote_catalog(
    provider: Provider,
    catalog_base_url: str | None = None,
    local_generated_at: int | None = None,
    *,
    fetch: Any = None,
) -> RemoteCatalogProvider:
    return RemoteCatalogProvider(
        provider,
        catalog_base_url if catalog_base_url is not None else DEFAULT_CATALOG_BASE_URL,
        local_generated_at,
        fetch if fetch is not None else _default_fetch,
    )
