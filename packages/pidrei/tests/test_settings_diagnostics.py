"""Mirror of pi coding-agent test/settings-diagnostics.test.ts."""

import pytest

from pidrei.core.agent_session_services import AgentSessionRuntimeDiagnostic
from pidrei.core.settings_diagnostics import collect_settings_diagnostics, deduplicate_diagnostics
from pidrei.core.settings_manager import SettingsManager


@pytest.mark.tonio
async def test_includes_the_settings_file_path_for_file_backed_storage(tmp_path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True)
    settings_path = agent_dir / "settings.json"
    settings_path.write_text("{", encoding="utf-8")

    diagnostics = collect_settings_diagnostics(await SettingsManager(str(tmp_path), str(agent_dir)))

    assert len(diagnostics) == 1
    assert diagnostics[0].type == "warning"
    assert f"Invalid settings file {settings_path}:" in diagnostics[0].message


@pytest.mark.tonio
async def test_falls_back_to_the_settings_scope_for_storage_without_file_paths():
    class _Storage:
        async def with_lock_async(self, scope, fn):
            if scope == "global":
                raise Exception("backend failed")
            fn(None)

    diagnostics = collect_settings_diagnostics(await SettingsManager(storage=_Storage()))

    assert diagnostics == [
        AgentSessionRuntimeDiagnostic(type="warning", message="Invalid global settings: backend failed")
    ]


def test_deduplicates_diagnostics_by_type_and_message():
    warning = AgentSessionRuntimeDiagnostic(type="warning", message="Invalid settings file /tmp/settings.json")
    error = AgentSessionRuntimeDiagnostic(type="error", message=warning.message)

    assert deduplicate_diagnostics([warning, warning, error]) == [warning, error]
