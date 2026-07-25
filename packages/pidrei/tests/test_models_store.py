"""Mirror of pi coding-agent test/models-store.test.ts."""

import pytest

from pidrei.core.models_store import FileModelsStore
from pidrei_ai.models_store import ModelsStoreEntry
from tests.model_runtime_helpers import make_model


@pytest.mark.tonio
async def test_persists_provider_catalogs_without_replacing_unrelated_providers(tmp_dir):
    path = str(tmp_dir / "models-store.json")
    store = FileModelsStore(path)

    await store.write("one", ModelsStoreEntry(models=[make_model("one", "m1")], checked_at=100))
    await store.write("two", ModelsStoreEntry(models=[make_model("two", "m2")], checked_at=200))

    reloaded = FileModelsStore(path)
    entry_one = await reloaded.read("one")
    assert [model.id for model in entry_one.models] == ["m1"]
    assert entry_one.checked_at == 100
    entry_two = await reloaded.read("two")
    assert [model.id for model in entry_two.models] == ["m2"]

    await reloaded.delete("one")
    assert await reloaded.read("one") is None
    assert [model.id for model in (await reloaded.read("two")).models] == ["m2"]
