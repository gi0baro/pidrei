"""Mirror of pi coding-agent test/settings-manager-bug.test.ts.

Tests for the fix to a bug where external file changes to arrays were
overwritten. The bug scenario was:
1. pidrei starts with settings.json containing packages: ["npm:some-pkg"]
2. User externally edits file to packages: []
3. User changes an unrelated setting (e.g., theme) via UI
4. save() would overwrite packages back to ["npm:some-pkg"] from stale in-memory state

The fix tracks which fields were explicitly modified during the session, and
only those fields override file values during save().
"""

import json

import pytest

from pidrei.core.settings_manager import SettingsManager


@pytest.fixture
def dirs(tmp_path):
    agent_dir = tmp_path / "agent"
    project_dir = tmp_path / "project"
    agent_dir.mkdir(parents=True)
    (project_dir / ".pidrei").mkdir(parents=True)
    return agent_dir, project_dir


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_preserves_file_changes_to_packages_array_when_changing_unrelated_setting(dirs):
    agent_dir, project_dir = dirs
    settings_path = agent_dir / "settings.json"

    settings_path.write_text(json.dumps({"theme": "dark", "packages": ["npm:pi-mcp-adapter"]}), encoding="utf-8")

    manager = SettingsManager.create(str(project_dir), str(agent_dir))

    assert manager.get_packages() == ["npm:pi-mcp-adapter"]

    current_settings = read_json(settings_path)
    current_settings["packages"] = []
    settings_path.write_text(json.dumps(current_settings, indent=2), encoding="utf-8")

    assert read_json(settings_path)["packages"] == []

    manager.set_theme("light")
    manager.flush()

    saved_settings = read_json(settings_path)

    assert saved_settings["packages"] == []
    assert saved_settings["theme"] == "light"


def test_preserves_file_changes_to_extensions_array_when_changing_unrelated_setting(dirs):
    agent_dir, project_dir = dirs
    settings_path = agent_dir / "settings.json"

    settings_path.write_text(json.dumps({"theme": "dark", "extensions": ["/old/extension.py"]}), encoding="utf-8")

    manager = SettingsManager.create(str(project_dir), str(agent_dir))

    current_settings = read_json(settings_path)
    current_settings["extensions"] = ["/new/extension.py"]
    settings_path.write_text(json.dumps(current_settings, indent=2), encoding="utf-8")

    manager.set_default_thinking_level("high")
    manager.flush()

    saved_settings = read_json(settings_path)

    assert saved_settings["extensions"] == ["/new/extension.py"]


def test_preserves_external_project_settings_changes_when_updating_unrelated_project_field(dirs):
    agent_dir, project_dir = dirs
    project_settings_path = project_dir / ".pidrei" / "settings.json"
    project_settings_path.write_text(
        json.dumps({"extensions": ["./old-extension.py"], "prompts": ["./old-prompt.md"]}), encoding="utf-8"
    )

    manager = SettingsManager.create(str(project_dir), str(agent_dir))

    current_project_settings = read_json(project_settings_path)
    current_project_settings["prompts"] = ["./new-prompt.md"]
    project_settings_path.write_text(json.dumps(current_project_settings, indent=2), encoding="utf-8")

    manager.set_project_extension_paths(["./updated-extension.py"])
    manager.flush()

    saved_project_settings = read_json(project_settings_path)
    assert saved_project_settings["prompts"] == ["./new-prompt.md"]
    assert saved_project_settings["extensions"] == ["./updated-extension.py"]


def test_lets_in_memory_project_changes_override_external_changes_for_the_same_project_field(dirs):
    agent_dir, project_dir = dirs
    project_settings_path = project_dir / ".pidrei" / "settings.json"
    project_settings_path.write_text(json.dumps({"extensions": ["./initial-extension.py"]}), encoding="utf-8")

    manager = SettingsManager.create(str(project_dir), str(agent_dir))

    current_project_settings = read_json(project_settings_path)
    current_project_settings["extensions"] = ["./external-extension.py"]
    project_settings_path.write_text(json.dumps(current_project_settings, indent=2), encoding="utf-8")

    manager.set_project_extension_paths(["./in-memory-extension.py"])
    manager.flush()

    saved_project_settings = read_json(project_settings_path)
    assert saved_project_settings["extensions"] == ["./in-memory-extension.py"]
