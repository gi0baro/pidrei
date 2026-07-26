"""Mirror of pi coding-agent test/settings-manager.test.ts.

The externalEditor platform-fallback assertions for win32 (notepad) are not
ported (POSIX-only); the POSIX fallback (nano) is covered.
"""

import json
import os

import pytest

from pidrei.core.http_config import DEFAULT_HTTP_IDLE_TIMEOUT_MS
from pidrei.core.settings_manager import SettingsManager


HOME = os.path.expanduser("~")


@pytest.fixture
def dirs(tmp_path):
    agent_dir = tmp_path / "agent"
    project_dir = tmp_path / "project"
    agent_dir.mkdir(parents=True)
    (project_dir / ".pidrei").mkdir(parents=True)
    return agent_dir, project_dir


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


class TestPreservesExternallyAddedSettings:
    def test_preserves_enabled_models_when_changing_thinking_level(self, dirs):
        agent_dir, project_dir = dirs
        settings_path = agent_dir / "settings.json"
        write_json(settings_path, {"theme": "dark", "defaultModel": "claude-sonnet"})

        manager = SettingsManager.create(str(project_dir), str(agent_dir))

        current_settings = read_json(settings_path)
        current_settings["enabledModels"] = ["claude-opus-4-5", "gpt-5.2-codex"]
        settings_path.write_text(json.dumps(current_settings, indent=2), encoding="utf-8")

        manager.set_default_thinking_level("high")
        manager.flush()

        saved_settings = read_json(settings_path)
        assert saved_settings["enabledModels"] == ["claude-opus-4-5", "gpt-5.2-codex"]
        assert saved_settings["defaultThinkingLevel"] == "high"
        assert saved_settings["theme"] == "dark"
        assert saved_settings["defaultModel"] == "claude-sonnet"

    def test_preserves_custom_settings_when_changing_theme(self, dirs):
        agent_dir, project_dir = dirs
        settings_path = agent_dir / "settings.json"
        write_json(settings_path, {"defaultModel": "claude-sonnet"})

        manager = SettingsManager.create(str(project_dir), str(agent_dir))

        current_settings = read_json(settings_path)
        current_settings["shellPath"] = "/bin/zsh"
        current_settings["extensions"] = ["/path/to/extension.py"]
        settings_path.write_text(json.dumps(current_settings, indent=2), encoding="utf-8")

        manager.set_theme("light")
        manager.flush()

        saved_settings = read_json(settings_path)
        assert saved_settings["shellPath"] == "/bin/zsh"
        assert saved_settings["extensions"] == ["/path/to/extension.py"]
        assert saved_settings["theme"] == "light"

    def test_lets_in_memory_changes_override_file_changes_for_same_key(self, dirs):
        agent_dir, project_dir = dirs
        settings_path = agent_dir / "settings.json"
        write_json(settings_path, {"theme": "dark"})

        manager = SettingsManager.create(str(project_dir), str(agent_dir))

        current_settings = read_json(settings_path)
        current_settings["defaultThinkingLevel"] = "low"
        settings_path.write_text(json.dumps(current_settings, indent=2), encoding="utf-8")

        manager.set_default_thinking_level("high")
        manager.flush()

        saved_settings = read_json(settings_path)
        assert saved_settings["defaultThinkingLevel"] == "high"


class TestPackagesMigration:
    def test_keeps_local_only_extensions_in_extensions_array(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"extensions": ["/local/ext.py", "./relative/ext.py"]})

        manager = SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_packages() == []
        assert manager.get_extension_paths() == ["/local/ext.py", "./relative/ext.py"]

    def test_handles_packages_with_filtering_objects(self, dirs):
        agent_dir, project_dir = dirs
        write_json(
            agent_dir / "settings.json",
            {
                "packages": [
                    "npm:simple-pkg",
                    {"source": "npm:shitty-extensions", "extensions": ["extensions/oracle.py"], "skills": []},
                ]
            },
        )

        manager = SettingsManager.create(str(project_dir), str(agent_dir))

        packages = manager.get_packages()
        assert len(packages) == 2
        assert packages[0] == "npm:simple-pkg"
        assert packages[1] == {
            "source": "npm:shitty-extensions",
            "extensions": ["extensions/oracle.py"],
            "skills": [],
        }


class TestReload:
    def test_reloads_global_settings_from_disk(self, dirs):
        agent_dir, project_dir = dirs
        settings_path = agent_dir / "settings.json"
        write_json(settings_path, {"theme": "dark", "extensions": ["/before.py"]})

        manager = SettingsManager.create(str(project_dir), str(agent_dir))

        write_json(settings_path, {"theme": "light", "extensions": ["/after.py"], "defaultModel": "claude-sonnet"})

        manager.reload()

        assert manager.get_theme() == "light"
        assert manager.get_extension_paths() == ["/after.py"]
        assert manager.get_default_model() == "claude-sonnet"

    def test_keeps_previous_settings_when_file_is_invalid(self, dirs):
        agent_dir, project_dir = dirs
        settings_path = agent_dir / "settings.json"
        write_json(settings_path, {"theme": "dark"})

        manager = SettingsManager.create(str(project_dir), str(agent_dir))

        settings_path.write_text("{ invalid json", encoding="utf-8")
        manager.reload()

        assert manager.get_theme() == "dark"


class TestThemeSetting:
    def test_stores_slash_separated_automatic_theme_settings_separately_from_fixed_theme_names(self, dirs):
        agent_dir, project_dir = dirs
        settings_path = agent_dir / "settings.json"
        write_json(settings_path, {"theme": "light/dark"})

        manager = SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_theme() is None
        assert manager.get_theme_setting() == "light/dark"

        manager.set_theme("solarized-light/tokyo-night")
        manager.flush()

        saved_settings = read_json(settings_path)
        assert saved_settings["theme"] == "solarized-light/tokyo-night"


class TestErrorTracking:
    def test_collects_and_clears_load_errors_via_drain_errors(self, dirs):
        agent_dir, project_dir = dirs
        (agent_dir / "settings.json").write_text("{ invalid global json", encoding="utf-8")
        (project_dir / ".pidrei" / "settings.json").write_text("{ invalid project json", encoding="utf-8")

        manager = SettingsManager.create(str(project_dir), str(agent_dir))
        errors = manager.drain_errors()

        assert len(errors) == 2
        assert sorted(error.scope for error in errors) == ["global", "project"]
        assert manager.drain_errors() == []


class TestProjectTrust:
    def test_skips_project_settings_when_project_is_not_trusted(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"theme": "global"})
        write_json(project_dir / ".pidrei" / "settings.json", {"theme": "project"})

        manager = SettingsManager.create(str(project_dir), str(agent_dir), project_trusted=False)

        assert manager.is_project_trusted() is False
        assert manager.get_theme() == "global"
        assert manager.get_project_settings() == {}

    def test_reloads_project_settings_after_trust_changes_to_true(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"theme": "global"})
        write_json(project_dir / ".pidrei" / "settings.json", {"theme": "project"})
        manager = SettingsManager.create(str(project_dir), str(agent_dir), project_trusted=False)

        manager.set_project_trusted(True)

        assert manager.is_project_trusted() is True
        assert manager.get_theme() == "project"

    def test_fails_project_settings_writes_when_project_is_not_trusted(self, dirs):
        agent_dir, project_dir = dirs
        project_settings_path = project_dir / ".pidrei" / "settings.json"
        write_json(project_settings_path, {"packages": ["npm:existing"]})
        manager = SettingsManager.create(str(project_dir), str(agent_dir), project_trusted=False)

        with pytest.raises(Exception, match="Project is not trusted; refusing to write project settings"):
            manager.set_project_packages(["npm:new"])
        manager.flush()

        assert manager.get_project_settings() == {}
        assert read_json(project_settings_path) == {"packages": ["npm:existing"]}

    def test_reads_default_project_trust_from_global_settings_only(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"defaultProjectTrust": "always"})
        write_json(project_dir / ".pidrei" / "settings.json", {"defaultProjectTrust": "never"})

        manager = SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_default_project_trust() == "always"

    def test_defaults_invalid_project_trust_settings_to_ask(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"defaultProjectTrust": "sometimes"})

        manager = SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_default_project_trust() == "ask"


class TestProjectSettingsDirectoryCreation:
    def test_does_not_create_config_folder_when_only_reading_project_settings(self, dirs, tmp_path):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"theme": "dark"})

        (project_dir / ".pidrei").rmdir()

        manager = SettingsManager.create(str(project_dir), str(agent_dir))

        assert not (project_dir / ".pidrei").exists()
        assert manager.get_theme() == "dark"

    def test_creates_config_folder_when_writing_project_settings(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"theme": "dark"})

        (project_dir / ".pidrei").rmdir()

        manager = SettingsManager.create(str(project_dir), str(agent_dir))

        assert not (project_dir / ".pidrei").exists()

        manager.set_project_packages([{"source": "npm:test-pkg"}])
        manager.flush()

        assert (project_dir / ".pidrei").exists()
        assert (project_dir / ".pidrei" / "settings.json").exists()


class TestHttpIdleTimeoutMs:
    def test_defaults_to_5_minutes(self, dirs):
        agent_dir, project_dir = dirs
        manager = SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_http_idle_timeout_ms() == DEFAULT_HTTP_IDLE_TIMEOUT_MS

    def test_uses_merged_global_and_project_settings(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"httpIdleTimeoutMs": 300000})
        write_json(project_dir / ".pidrei" / "settings.json", {"httpIdleTimeoutMs": 0})

        manager = SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_http_idle_timeout_ms() == 0

    def test_rejects_invalid_timeout_values(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"httpIdleTimeoutMs": -1})
        manager = SettingsManager.create(str(project_dir), str(agent_dir))

        with pytest.raises(Exception, match="Invalid httpIdleTimeoutMs setting"):
            manager.get_http_idle_timeout_ms()


class TestExternalEditor:
    def test_resolves_editor_commands_by_precedence(self, monkeypatch):
        monkeypatch.setenv("VISUAL", "vim")
        monkeypatch.setenv("EDITOR", "nano")
        assert SettingsManager.in_memory({"externalEditor": "code --wait"}).get_external_editor_command() == (
            "code --wait"
        )
        assert SettingsManager.in_memory().get_external_editor_command() == "vim"

        monkeypatch.delenv("VISUAL")
        monkeypatch.setenv("EDITOR", "emacs")
        assert SettingsManager.in_memory().get_external_editor_command() == "emacs"

    def test_falls_back_to_platform_default(self, monkeypatch):
        monkeypatch.delenv("VISUAL", raising=False)
        monkeypatch.delenv("EDITOR", raising=False)
        assert SettingsManager.in_memory().get_external_editor_command() == "nano"


class TestOutputPad:
    def test_defaults_to_1_and_persists_binary_values(self, dirs):
        agent_dir, project_dir = dirs
        manager = SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_output_pad() == 1

        manager.set_output_pad(0)
        manager.flush()

        assert manager.get_output_pad() == 0
        saved_settings = read_json(agent_dir / "settings.json")
        assert saved_settings["outputPad"] == 0

    def test_treats_unsupported_output_pad_values_as_default_padding(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"outputPad": 2})

        manager = SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_output_pad() == 1


class TestShellCommandPrefix:
    def test_loads_shell_command_prefix_from_settings(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"shellCommandPrefix": "shopt -s expand_aliases"})

        manager = SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_shell_command_prefix() == "shopt -s expand_aliases"

    def test_returns_none_when_shell_command_prefix_is_not_set(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"theme": "dark"})

        manager = SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_shell_command_prefix() is None

    def test_preserves_shell_command_prefix_when_saving_unrelated_settings(self, dirs):
        agent_dir, project_dir = dirs
        settings_path = agent_dir / "settings.json"
        write_json(settings_path, {"shellCommandPrefix": "shopt -s expand_aliases"})

        manager = SettingsManager.create(str(project_dir), str(agent_dir))
        manager.set_theme("light")
        manager.flush()

        saved_settings = read_json(settings_path)
        assert saved_settings["shellCommandPrefix"] == "shopt -s expand_aliases"
        assert saved_settings["theme"] == "light"


class TestGetSessionDir:
    def test_returns_none_when_not_set(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"theme": "dark"})
        manager = SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_session_dir() is None

    def test_returns_global_session_dir(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"sessionDir": "/tmp/sessions"})
        manager = SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_session_dir() == "/tmp/sessions"

    def test_returns_project_session_dir_overriding_global(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"sessionDir": "/global/sessions"})
        write_json(project_dir / ".pidrei" / "settings.json", {"sessionDir": "./sessions"})
        manager = SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_session_dir() == "./sessions"

    def test_expands_tilde_in_session_dir(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"sessionDir": "~/sessions"})
        manager = SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_session_dir() == os.path.join(HOME, "sessions")


class TestGetShellPath:
    def test_returns_none_when_not_set(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"theme": "dark"})
        manager = SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_shell_path() is None

    def test_returns_an_absolute_shell_path_unchanged(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"shellPath": "/bin/zsh"})
        manager = SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_shell_path() == "/bin/zsh"

    def test_expands_tilde_in_shell_path(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"shellPath": "~/.local/bin/agent-shell-sandbox"})
        manager = SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_shell_path() == os.path.join(HOME, ".local/bin/agent-shell-sandbox")

    def test_expands_a_bare_tilde_in_shell_path(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"shellPath": "~"})
        manager = SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_shell_path() == HOME
