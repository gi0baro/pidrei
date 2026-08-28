"""Regression net for PROPER_MT_DESIGN.md step 3 (config epochs).

Not a pi mirror. Pins the snapshot-swap contract of the config services:
published state (settings dicts, the auth read-state pair, the runtime
credential overrides, the model-runtime composition epoch) is immutable and
rebound wholesale, so a reader that pinned a snapshot can never observe a
later write through it, and a torn multi-field read is unrepresentable.
Each test fails on the old mutate-in-place shape.
"""

import pytest

from pidrei.core.auth_storage import AuthStorage
from pidrei.core.model_runtime import ModelRuntime
from pidrei.core.models_store import InMemoryCodingAgentModelsStore
from pidrei.core.runtime_credentials import RuntimeCredentials
from pidrei.core.settings_manager import SettingsManager
from pidrei_ai.auth.types import ApiKeyCredential
from pidrei_ai.models_store import ModelsStoreEntry


def test_a_pinned_settings_snapshot_never_changes_under_a_later_set():
    """Old shape: nested setters mutated `_global_settings["compaction"]` in
    place, and that dict was aliased into the published merged snapshot — a
    pinned `_settings` changed under its reader."""
    manager = SettingsManager.in_memory({})
    manager.set_compaction_enabled(True)
    pinned = manager._settings
    pinned_compaction = pinned.get("compaction")
    manager.set_compaction_enabled(False)
    assert pinned_compaction == {"enabled": True}
    assert pinned.get("compaction") is pinned_compaction
    assert manager.get_compaction_enabled() is False


def test_a_pinned_global_scope_snapshot_survives_scalar_sets():
    manager = SettingsManager.in_memory({})
    manager.set_theme("dark")
    pinned = manager._global_settings
    manager.set_theme("light")
    assert pinned["theme"] == "dark"
    assert manager.get_theme() == "light"


def test_runtime_credential_overrides_swap_instead_of_mutating():
    creds = RuntimeCredentials(AuthStorage.in_memory())
    creds.set_runtime_api_key("first", "key-1")
    pinned = creds._overrides
    creds.set_runtime_api_key("second", "key-2")
    assert "second" not in pinned
    assert creds.has_runtime_api_key("second")
    creds.remove_runtime_api_key("first")
    assert "first" in pinned
    assert not creds.has_runtime_api_key("first")


@pytest.mark.tonio
async def test_auth_read_state_publishes_one_immutable_snapshot_per_reload(tmp_dir):
    """The (data, revision) pair is one frozen object rebound wholesale: a
    reader can never combine the data of one reload with the revision of
    another, and a pinned snapshot's data never grows later entries."""
    path = str(tmp_dir / "auth.json")
    store = await AuthStorage.create(path)

    async def put_first(_current):
        return ApiKeyCredential(key="key-1")

    await store.modify("first", put_first)
    pinned = store._read_state.snapshot

    async def put_second(_current):
        return ApiKeyCredential(key="key-2")

    await store.modify("second", put_second)
    assert store._read_state.snapshot is not pinned
    assert "second" not in pinned.data
    assert "first" in pinned.data


@pytest.mark.tonio
async def test_in_memory_models_store_swaps_entries_instead_of_mutating():
    store = InMemoryCodingAgentModelsStore()
    await store.write("first", ModelsStoreEntry(models=[]))
    pinned = store._entries
    await store.write("second", ModelsStoreEntry(models=[]))
    assert "second" not in pinned
    await store.delete("first")
    assert "first" in pinned
    assert await store.read("first") is None


@pytest.mark.tonio
async def test_a_pinned_composition_epoch_survives_provider_registration():
    """Old shape: `_extension_providers` and `_composition_errors` were
    mutated in place, so `get_error`'s iteration could race a recompose on
    another thread and a pinned view changed mid-operation."""
    runtime = await ModelRuntime.create(credentials=AuthStorage.in_memory(), models_path=None)
    pinned = runtime._composition
    runtime.register_provider(
        "epoch-provider",
        {
            "baseUrl": "https://example.test/v1",
            "apiKey": "key",
            "api": "openai-completions",
            "models": [
                {
                    "id": "epoch-model",
                    "name": "epoch-model",
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 10000,
                    "maxTokens": 1000,
                }
            ],
        },
    )
    assert "epoch-provider" not in pinned.extension_providers
    current = runtime._composition
    assert current is not pinned
    assert "epoch-provider" in current.extension_providers
    runtime.unregister_provider("epoch-provider")
    assert "epoch-provider" in current.extension_providers
    assert "epoch-provider" not in runtime._composition.extension_providers
