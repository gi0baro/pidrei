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
from pidrei_ai.utils.cancel import CancelToken
from tests.model_runtime_helpers import make_model


def model_dict(id: str) -> dict:
    return model_to_dict(make_model("test-provider", id))


def make_provider(local_generated_at=None, *, fetch):
    async def resolve(_ctx, _credential, _cancel):
        return AuthResult(auth=ModelAuth())

    provider = create_provider(
        id="test-provider",
        auth=ProviderAuth(api_key=ApiKeyAuth(name="Test", resolve=resolve)),
        models=[make_model("test-provider", "static")],
        api={},
    )
    return with_remote_catalog(provider, "https://pi.dev", local_generated_at, fetch=fetch)


async def refresh_provider(provider, store: InMemoryModelsStore, **overrides) -> None:
    """pi's `refreshProvider` helper: an always-current publication backed by
    the store, with the stored snapshot read up-front."""

    async def publish(publication) -> bool:
        if publication.persist is None:
            await store.delete(provider.id)
        elif publication.persist is not ...:
            await store.write(provider.id, publication.persist)
        if publication.update is not None:
            publication.update()
        return True

    await provider.refresh_models(
        RefreshModelsContext(
            credential=ApiKeyCredential(key="key"),
            stored=await store.read(provider.id),
            publish=publish,
            allow_network=overrides.get("allow_network", True),
            force=overrides.get("force"),
            cancel=overrides.get("cancel") or CancelToken(),
        )
    )


@pytest.mark.tonio
async def test_parses_keyed_catalogs_sends_version_headers_observes_ttl_and_supports_forced_refreshes():
    calls = []

    async def fetch(url, headers, _cancel):
        calls.append((url, headers))
        return CatalogResponse(status=200, headers={}, body=json.dumps({"dynamic": model_dict("dynamic")}))

    provider = make_provider(fetch=fetch)
    store = InMemoryModelsStore()
    await refresh_provider(provider, store)
    await refresh_provider(provider, store)
    await refresh_provider(provider, store, force=True)

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

    await refresh_provider(provider, store)
    assert [entry.id for entry in provider.get_models()] == ["static"]

    await refresh_provider(provider, store, force=True)
    assert [entry.id for entry in provider.get_models()] == ["static", "newer"]
    stored = await store.read(provider.id)
    assert stored.last_modified == local_generated_at + 60_000


@pytest.mark.tonio
async def test_revalidates_a_stored_catalog_with_its_etag_and_keeps_the_overlay_on_304():
    responses = [
        CatalogResponse(
            status=200, headers={"etag": '"catalog-1"'}, body=json.dumps({"dynamic": model_dict("dynamic")})
        ),
        CatalogResponse(status=304, headers={"etag": '"catalog-1"'}, body=""),
    ]
    calls = []

    async def fetch(url, headers, _cancel):
        calls.append((url, headers))
        return responses.pop(0)

    provider = make_provider(fetch=fetch)
    store = InMemoryModelsStore()

    await refresh_provider(provider, store)
    assert "if-none-match" not in calls[0][1]
    assert (await store.read(provider.id)).etag == '"catalog-1"'

    checked_at = (await store.read(provider.id)).checked_at
    await refresh_provider(provider, store, force=True)

    assert calls[1][1]["if-none-match"] == '"catalog-1"'
    assert [entry.id for entry in provider.get_models()] == ["static", "dynamic"]
    stored = await store.read(provider.id)
    assert [entry.id for entry in stored.models] == ["dynamic"]
    assert stored.etag == '"catalog-1"'
    assert stored.checked_at >= (checked_at or 0)


@pytest.mark.tonio
async def test_drops_a_stale_etag_when_the_overlay_becomes_unavailable():
    responses = [
        CatalogResponse(
            status=200, headers={"etag": '"catalog-1"'}, body=json.dumps({"dynamic": model_dict("dynamic")})
        ),
        CatalogResponse(status=501, headers={}, body="not implemented"),
    ]

    async def fetch(_url, _headers, _cancel):
        return responses.pop(0)

    provider = make_provider(fetch=fetch)
    store = InMemoryModelsStore()

    await refresh_provider(provider, store)
    await refresh_provider(provider, store, force=True)

    assert (await store.read(provider.id)).etag is None


@pytest.mark.tonio
async def test_keeps_the_etag_and_overlay_after_a_transient_failure():
    responses = [
        CatalogResponse(
            status=200, headers={"etag": '"catalog-1"'}, body=json.dumps({"dynamic": model_dict("dynamic")})
        ),
        CatalogResponse(status=429, headers={}, body="rate limited"),
        CatalogResponse(status=304, headers={"etag": '"catalog-1"'}, body=""),
    ]
    calls = []

    async def fetch(url, headers, _cancel):
        calls.append((url, headers))
        return responses.pop(0)

    provider = make_provider(fetch=fetch)
    store = InMemoryModelsStore()

    await refresh_provider(provider, store)
    with pytest.raises(Exception, match="429"):
        await refresh_provider(provider, store, force=True)

    stored = await store.read(provider.id)
    assert stored.etag == '"catalog-1"'
    assert [entry.id for entry in stored.models] == ["dynamic"]

    await refresh_provider(provider, store, force=True)
    assert calls[2][1]["if-none-match"] == '"catalog-1"'
    assert [entry.id for entry in provider.get_models()] == ["static", "dynamic"]


@pytest.mark.tonio
async def test_treats_unimplemented_catalog_routes_as_an_unavailable_overlay():
    async def fetch(_url, _headers, _cancel):
        return CatalogResponse(status=501, headers={}, body="not implemented")

    provider = make_provider(fetch=fetch)
    store = InMemoryModelsStore()

    await refresh_provider(provider, store)
    assert [entry.id for entry in provider.get_models()] == ["static"]
    stored = await store.read(provider.id)
    assert stored.models == []
    assert isinstance(stored.checked_at, int)
