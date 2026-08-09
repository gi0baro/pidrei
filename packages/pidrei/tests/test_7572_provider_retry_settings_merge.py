"""Mirror of pi's suite/regressions/7572-provider-retry-settings-merge.test.ts."""

import json

from pidrei.core.settings_manager import InMemorySettingsStorage, SettingsManager


def test_preserves_global_provider_settings_not_overridden_by_the_project():
    storage = InMemorySettingsStorage()
    storage.with_lock(
        "global", lambda _current: json.dumps({"retry": {"provider": {"timeoutMs": 30000, "maxRetryDelayMs": 45000}}})
    )
    storage.with_lock("project", lambda _current: json.dumps({"retry": {"provider": {"maxRetries": 2}}}))

    settings_manager = SettingsManager.from_storage(storage)

    assert settings_manager.get_provider_retry_settings() == {
        "timeout_ms": 30000,
        "max_retries": 2,
        "max_retry_delay_ms": 45000,
    }
