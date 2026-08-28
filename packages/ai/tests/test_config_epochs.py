"""Regression net for PROPER_MT_DESIGN.md step 3 (config epochs), ai package.

Not a pi mirror. Pins the snapshot-swap contract: published state is
immutable and rebound wholesale, so a reader that pinned a snapshot can
never observe a later write through it. Each test fails on the old
mutate-in-place shape.
"""

import pytest

from pidrei_ai.auth.types import ProviderAuth
from pidrei_ai.models_store import InMemoryModelsStore, ModelsStoreEntry
from pidrei_ai.registry import create_models, create_provider


def _provider(id: str):
    return create_provider(id=id, auth=ProviderAuth(), models=[], api={})


def test_the_provider_map_is_swapped_not_mutated():
    """Old shape: `set_provider`/`delete_provider` mutated `_providers` in
    place under a lock every reader also took; now readers pin the map."""
    models = create_models()
    first = _provider("first")
    models.set_provider(first)
    pinned = models._providers
    second = _provider("second")
    models.set_provider(second)
    assert "second" not in pinned
    models.delete_provider("first")
    assert "first" in pinned
    assert models.get_provider("first") is None
    # Swaps rebind the map but never clone the providers inside it.
    assert models.get_provider("second") is second


@pytest.mark.tonio
async def test_in_memory_models_store_swaps_entries_instead_of_mutating():
    store = InMemoryModelsStore()
    await store.write("first", ModelsStoreEntry(models=[]))
    pinned = store._entries
    await store.write("second", ModelsStoreEntry(models=[]))
    assert "second" not in pinned
    await store.delete("first")
    assert "first" in pinned
    assert await store.read("first") is None
    assert await store.read("second") is not None
