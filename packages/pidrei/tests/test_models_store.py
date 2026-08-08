"""Mirror of pi coding-agent test/models-store.test.ts."""

import json

import pytest
import tonio.colored as tonio

from pidrei.core.models_store import FileModelsStore
from pidrei.utils import lockfile
from pidrei_ai.models_store import ModelsStoreEntry, ModelsStoreOperationOptions
from pidrei_ai.utils.cancel import AbortError, CancelToken
from tests.model_runtime_helpers import make_model


@pytest.mark.tonio
async def test_cancels_a_catalog_write_waiting_for_a_held_file_lock_without_writing_later(tmp_dir):
    path = str(tmp_dir / "models-store.json")
    entry_one = ModelsStoreEntry(models=[make_model("one", "existing")])
    store = FileModelsStore(path)
    await store.write("one", entry_one)
    release = lockfile.lock_sync(path, stale=30.0)
    controller = CancelToken()
    outcome: dict = {}

    async def run_pending() -> None:
        try:
            await store.write(
                "two",
                ModelsStoreEntry(models=[make_model("two", "cancelled")]),
                ModelsStoreOperationOptions(cancel=controller),
            )
            outcome["error"] = None
        except BaseException as error:
            outcome["error"] = error

    async def drive() -> None:
        await tonio.time.sleep(0.01)
        controller.cancel()

    await tonio.spawn(run_pending(), drive())
    assert isinstance(outcome["error"], AbortError)
    release()
    await tonio.time.sleep(0.15)

    stored = json.loads((tmp_dir / "models-store.json").read_text(encoding="utf-8"))
    assert "one" in stored
    assert "two" not in stored


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
