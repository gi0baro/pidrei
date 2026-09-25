"""Mirror of pi coding-agent test/settings-manager-compaction.test.ts.

Regression coverage for #8133. pi's `String(value)` in the error messages is
JS's; the settings manager renders invalid values the same way.
"""

import json
import math
import re

import pytest

from pidrei.core.settings_manager import InMemorySettingsStorage, SettingsManager


class _Model:
    def __init__(self, provider: str, model_id: str) -> None:
        self.provider = provider
        self.id = model_id


MODEL = _Model("provider", "family/model")
MODEL_KEY = "provider/family/model"
DEFAULTS = {"enabled": True, "reserve_tokens": 16384, "keep_recent_tokens": 20000}

# pi's `[null, -1, 1.5, "400000", true, {}, [], Number.MAX_SAFE_INTEGER + 1]`
# with the `String(value)` each renders to.
INVALID_TOKEN_VALUES = [
    (None, "null"),
    (-1, "-1"),
    (1.5, "1.5"),
    ("400000", "400000"),
    (True, "true"),
    ({}, "[object Object]"),
    ([], ""),
    (2**53, "9007199254740992"),
]
NON_FINITE_VALUES = [(math.nan, "NaN"), (math.inf, "Infinity"), (-math.inf, "-Infinity")]


def _from_global(settings: dict) -> SettingsManager:
    storage = InMemorySettingsStorage()
    storage.with_lock("global", lambda _current: json.dumps(settings))
    return SettingsManager.from_storage(storage)


def test_uses_defaults_without_compaction_settings():
    manager = SettingsManager.in_memory()
    assert manager.get_compaction_settings() == DEFAULTS
    assert manager.get_compaction_settings(MODEL) == DEFAULTS


def test_resolves_each_field_independently_and_keeps_individual_getters_consistent():
    manager = SettingsManager.in_memory(
        {
            "compaction": {
                "reserveTokens": 8192,
                "keepRecentTokens": 10000,
                "modelOverrides": {MODEL_KEY: {"reserveTokens": 400000}},
            }
        }
    )
    assert manager.get_compaction_settings(MODEL) == {
        "enabled": True,
        "reserve_tokens": 400000,
        "keep_recent_tokens": 10000,
    }
    assert manager.get_compaction_reserve_tokens(MODEL) == 400000
    assert manager.get_compaction_keep_recent_tokens(MODEL) == 10000
    assert manager.get_compaction_settings() == {"enabled": True, "reserve_tokens": 8192, "keep_recent_tokens": 10000}

    manager.apply_overrides({"compaction": {"modelOverrides": {MODEL_KEY: {"keepRecentTokens": 30000}}}})
    assert manager.get_compaction_keep_recent_tokens(MODEL) == 30000
    assert manager.get_compaction_reserve_tokens(MODEL) == 400000


def test_falls_back_to_built_in_defaults_for_missing_fields():
    manager = SettingsManager.in_memory({"compaction": {"modelOverrides": {MODEL_KEY: {"keepRecentTokens": 1024}}}})
    assert manager.get_compaction_settings(MODEL) == {**DEFAULTS, "keep_recent_tokens": 1024}


def test_matches_exact_provider_model_ids_including_ids_containing_slashes():
    manager = SettingsManager.in_memory(
        {
            "compaction": {
                "modelOverrides": {
                    MODEL_KEY: {"reserveTokens": 400000},
                    "provider/*": {"reserveTokens": 1},
                    "family/model": {"reserveTokens": 2},
                }
            }
        }
    )
    assert manager.get_compaction_reserve_tokens(MODEL) == 400000
    for other in (_Model("other", MODEL.id), _Model(MODEL.provider, "other"), _Model(MODEL.provider, "family/Model")):
        assert manager.get_compaction_settings(other) == DEFAULTS


@pytest.mark.tonio
async def test_merges_project_model_overrides_per_field_before_resolving_fallbacks():
    storage = InMemorySettingsStorage()
    storage.with_lock(
        "global",
        lambda _current: json.dumps(
            {
                "compaction": {
                    "reserveTokens": 8192,
                    "modelOverrides": {
                        MODEL_KEY: {"reserveTokens": 400000, "keepRecentTokens": 30000},
                        "provider/other": {"keepRecentTokens": 4096},
                    },
                }
            }
        ),
    )
    storage.with_lock(
        "project",
        lambda _current: json.dumps(
            {"compaction": {"reserveTokens": 1024, "modelOverrides": {MODEL_KEY: {"keepRecentTokens": 2000}}}}
        ),
    )
    manager = SettingsManager.from_storage(storage)
    assert manager.get_compaction_settings(MODEL) == {
        "enabled": True,
        "reserve_tokens": 400000,
        "keep_recent_tokens": 2000,
    }
    assert manager.get_compaction_settings(_Model("provider", "other")) == {
        "enabled": True,
        "reserve_tokens": 1024,
        "keep_recent_tokens": 4096,
    }
    await manager.reload()
    assert manager.get_compaction_keep_recent_tokens(MODEL) == 2000
    manager.set_project_trusted(False)
    assert manager.get_compaction_keep_recent_tokens(MODEL) == 30000


@pytest.mark.tonio
async def test_keeps_enabled_global_and_preserves_overrides_when_saving_the_toggle():
    storage = InMemorySettingsStorage()
    storage.with_lock(
        "global",
        lambda _current: json.dumps(
            {"compaction": {"modelOverrides": {MODEL_KEY: {"enabled": False, "reserveTokens": 400000}}}}
        ),
    )
    manager = SettingsManager.from_storage(storage)
    assert manager.get_compaction_settings(MODEL)["enabled"] is True
    manager.set_compaction_enabled(False)
    await manager.flush()
    await manager.reload()
    assert manager.get_compaction_settings(MODEL) == {**DEFAULTS, "enabled": False, "reserve_tokens": 400000}


@pytest.mark.parametrize("field", ["reserveTokens", "keepRecentTokens"])
@pytest.mark.parametrize(("value", "rendered"), INVALID_TOKEN_VALUES)
def test_model_override_reports_invalid_token_values(field, value, rendered):
    manager = _from_global({"compaction": {"modelOverrides": {MODEL_KEY: {field: value}}}})
    with pytest.raises(
        Exception,
        match=re.escape(
            f'Invalid compaction.modelOverrides["{MODEL_KEY}"].{field} setting: {rendered}. '
            "Expected a non-negative safe integer."
        ),
    ):
        manager.get_compaction_settings(MODEL)
    assert manager.get_compaction_settings() == DEFAULTS
    assert manager.get_compaction_settings(_Model("other", MODEL.id)) == DEFAULTS


@pytest.mark.parametrize("field", ["reserveTokens", "keepRecentTokens"])
@pytest.mark.parametrize(("value", "rendered"), NON_FINITE_VALUES)
def test_model_override_reports_non_finite_runtime_values(field, value, rendered):
    manager = SettingsManager.in_memory()
    manager.apply_overrides({"compaction": {"modelOverrides": {MODEL_KEY: {field: value}}}})
    with pytest.raises(
        Exception, match=re.escape(f'Invalid compaction.modelOverrides["{MODEL_KEY}"].{field} setting: {rendered}')
    ):
        manager.get_compaction_settings(MODEL)


@pytest.mark.parametrize("field", ["reserveTokens", "keepRecentTokens"])
@pytest.mark.parametrize(("value", "rendered"), INVALID_TOKEN_VALUES)
def test_ordinary_setting_reports_invalid_values_even_when_a_valid_model_override_exists(field, value, rendered):
    manager = _from_global({"compaction": {field: value, "modelOverrides": {MODEL_KEY: {field: 4096}}}})
    error = re.escape(f"Invalid compaction.{field} setting: {rendered}. Expected a non-negative safe integer.")
    with pytest.raises(Exception, match=error):
        manager.get_compaction_settings()
    with pytest.raises(Exception, match=error):
        manager.get_compaction_settings(MODEL)


@pytest.mark.parametrize("field", ["reserveTokens", "keepRecentTokens"])
@pytest.mark.parametrize(("value", "rendered"), NON_FINITE_VALUES)
def test_ordinary_setting_reports_non_finite_runtime_values(field, value, rendered):
    manager = SettingsManager.in_memory()
    manager.apply_overrides({"compaction": {field: value}})
    with pytest.raises(Exception, match=re.escape(f"Invalid compaction.{field} setting: {rendered}")):
        manager.get_compaction_settings()


@pytest.mark.parametrize(
    ("entry", "rendered"), [(None, "null"), (False, "false"), (42, "42"), ("invalid", "invalid"), ([], "")]
)
def test_reports_malformed_model_entries(entry, rendered):
    manager = _from_global({"compaction": {"modelOverrides": {MODEL_KEY: entry}}})
    with pytest.raises(
        Exception,
        match=re.escape(f'Invalid compaction.modelOverrides["{MODEL_KEY}"] setting: {rendered}. Expected an object.'),
    ):
        manager.get_compaction_settings(MODEL)


def test_accepts_zero_in_ordinary_settings_and_model_overrides():
    manager = SettingsManager.in_memory({"compaction": {"reserveTokens": 0, "keepRecentTokens": 0}})
    assert manager.get_compaction_settings(MODEL) == {"enabled": True, "reserve_tokens": 0, "keep_recent_tokens": 0}
    manager.apply_overrides(
        {
            "compaction": {
                "reserveTokens": 1000,
                "keepRecentTokens": 1000,
                "modelOverrides": {MODEL_KEY: {"reserveTokens": 0, "keepRecentTokens": 0}},
            }
        }
    )
    assert manager.get_compaction_settings(MODEL) == {"enabled": True, "reserve_tokens": 0, "keep_recent_tokens": 0}
