"""Mirror of pi coding-agent test/remote-catalog-provider.test.ts.

fetch is injected through with_remote_catalog's `fetch` seam instead of
vitest-spying on global fetch.
"""

import json
from datetime import UTC, datetime
from email.utils import format_datetime

import pytest

from pidrei.config import VERSION
from pidrei.core.model_wire import model_to_dict
from pidrei.core.remote_catalog import CatalogResponse, with_remote_catalog
from pidrei_ai.auth.types import ApiKeyAuth, ApiKeyCredential, AuthResult, ModelAuth, ProviderAuth
from pidrei_ai.models_store import InMemoryModelsStore
from pidrei_ai.registry import RefreshModelsContext, create_provider
from tests.model_runtime_helpers import make_model


def model_dict(id: str) -> dict:
    return model_to_dict(make_model("test-provider", id))


def make_provider(local_generated_at=None, *, fetch):
    async def resolve(_ctx, _credential):
        return AuthResult(auth=ModelAuth())

    provider = create_provider(
        id="test-provider",
        auth=ProviderAuth(api_key=ApiKeyAuth(name="Test", resolve=resolve)),
        models=[make_model("test-provider", "static")],
        api={},
    )
    return with_remote_catalog(provider, "https://pi.dev", local_generated_at, fetch=fetch)


class ScopedStore:
    def __init__(self, store: InMemoryModelsStore):
        self._store = store

    async def read(self):
        return await self._store.read("test-provider")

    async def write(self, entry):
        await self._store.write("test-provider", entry)

    async def delete(self):
        await self._store.delete("test-provider")


def refresh_context(store, **overrides):
    defaults = {
        "credential": ApiKeyCredential(key="key"),
        "store": ScopedStore(store),
        "allow_network": True,
    }
    defaults.update(overrides)
    return RefreshModelsContext(**defaults)


@pytest.mark.tonio
async def test_parses_keyed_catalogs_sends_version_headers_observes_ttl_and_supports_forced_refreshes():
    calls = []

    async def fetch(url, headers, _cancel):
        calls.append((url, headers))
        return CatalogResponse(status=200, headers={}, body=json.dumps({"dynamic": model_dict("dynamic")}))

    provider = make_provider(fetch=fetch)
    store = InMemoryModelsStore()
    await provider.refresh_models(refresh_context(store))
    await provider.refresh_models(refresh_context(store))
    await provider.refresh_models(refresh_context(store, force=True))

    assert [entry.id for entry in provider.get_models()] == ["static", "dynamic"]
    stored = await store.read(provider.id)
    assert [entry.id for entry in stored.models] == ["dynamic"]
    assert len(calls) == 2
    assert f"pidrei/{VERSION}" in calls[0][1]["User-Agent"]


@pytest.mark.tonio
async def test_prefers_the_newer_of_the_generated_and_remote_catalogs():
    local_generated_at = int(datetime(2026, 7, 23, 10, 0, 0, tzinfo=UTC).timestamp() * 1000)
    newer_header = format_datetime(datetime.fromtimestamp((local_generated_at + 60_000) / 1000, tz=UTC), usegmt=True)
    older_header = format_datetime(datetime.fromtimestamp((local_generated_at - 60_000) / 1000, tz=UTC), usegmt=True)
    responses = [
        CatalogResponse(
            status=200, headers={"last-modified": older_header}, body=json.dumps({"old": model_dict("old")})
        ),
        CatalogResponse(
            status=200, headers={"last-modified": newer_header}, body=json.dumps({"newer": model_dict("newer")})
        ),
    ]

    async def fetch(_url, _headers, _cancel):
        return responses.pop(0)

    provider = make_provider(local_generated_at, fetch=fetch)
    store = InMemoryModelsStore()

    await provider.refresh_models(refresh_context(store))
    assert [entry.id for entry in provider.get_models()] == ["static"]

    await provider.refresh_models(refresh_context(store, force=True))
    assert [entry.id for entry in provider.get_models()] == ["static", "newer"]
    stored = await store.read(provider.id)
    assert stored.last_modified == local_generated_at + 60_000


@pytest.mark.tonio
async def test_treats_unimplemented_catalog_routes_as_an_unavailable_overlay():
    async def fetch(_url, _headers, _cancel):
        return CatalogResponse(status=501, headers={}, body="not implemented")

    provider = make_provider(fetch=fetch)
    store = InMemoryModelsStore()

    await provider.refresh_models(refresh_context(store))
    assert [entry.id for entry in provider.get_models()] == ["static"]
    stored = await store.read(provider.id)
    assert stored.models == []
    assert isinstance(stored.checked_at, int)
