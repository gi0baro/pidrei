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
def dirs(tmp_dir):
    agent_dir = tmp_dir / "agent"
    project_dir = tmp_dir / "project"
    agent_dir.mkdir(parents=True)
    (project_dir / ".pidrei").mkdir(parents=True)
    return agent_dir, project_dir


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


class TestPreservesExternallyAddedSettings:
    @pytest.mark.tonio
    async def test_preserves_enabled_models_when_changing_thinking_level(self, dirs):
        agent_dir, project_dir = dirs
        settings_path = agent_dir / "settings.json"
        write_json(settings_path, {"theme": "dark", "defaultModel": "claude-sonnet"})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        current_settings = read_json(settings_path)
        current_settings["enabledModels"] = ["claude-opus-4-5", "gpt-5.2-codex"]
        settings_path.write_text(json.dumps(current_settings, indent=2), encoding="utf-8")

        manager.set_default_thinking_level("high")
        await manager.flush()

        saved_settings = read_json(settings_path)
        assert saved_settings["enabledModels"] == ["claude-opus-4-5", "gpt-5.2-codex"]
        assert saved_settings["defaultThinkingLevel"] == "high"
        assert saved_settings["theme"] == "dark"
        assert saved_settings["defaultModel"] == "claude-sonnet"

    @pytest.mark.tonio
    async def test_preserves_custom_settings_when_changing_theme(self, dirs):
        agent_dir, project_dir = dirs
        settings_path = agent_dir / "settings.json"
        write_json(settings_path, {"defaultModel": "claude-sonnet"})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        current_settings = read_json(settings_path)
        current_settings["shellPath"] = "/bin/zsh"
        current_settings["extensions"] = ["/path/to/extension.py"]
        settings_path.write_text(json.dumps(current_settings, indent=2), encoding="utf-8")

        manager.set_theme("light")
        await manager.flush()

        saved_settings = read_json(settings_path)
        assert saved_settings["shellPath"] == "/bin/zsh"
        assert saved_settings["extensions"] == ["/path/to/extension.py"]
        assert saved_settings["theme"] == "light"

    @pytest.mark.tonio
    async def test_lets_in_memory_changes_override_file_changes_for_same_key(self, dirs):
        agent_dir, project_dir = dirs
        settings_path = agent_dir / "settings.json"
        write_json(settings_path, {"theme": "dark"})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        current_settings = read_json(settings_path)
        current_settings["defaultThinkingLevel"] = "low"
        settings_path.write_text(json.dumps(current_settings, indent=2), encoding="utf-8")

        manager.set_default_thinking_level("high")
        await manager.flush()

        saved_settings = read_json(settings_path)
        assert saved_settings["defaultThinkingLevel"] == "high"


class TestPackagesMigration:
    @pytest.mark.tonio
    async def test_keeps_local_only_extensions_in_extensions_array(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"extensions": ["/local/ext.py", "./relative/ext.py"]})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_packages() == []
        assert manager.get_extension_paths() == ["/local/ext.py", "./relative/ext.py"]

    @pytest.mark.tonio
    async def test_handles_packages_with_filtering_objects(self, dirs):
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

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        packages = manager.get_packages()
        assert len(packages) == 2
        assert packages[0] == "npm:simple-pkg"
        assert packages[1] == {
            "source": "npm:shitty-extensions",
            "extensions": ["extensions/oracle.py"],
            "skills": [],
        }


class TestReload:
    @pytest.mark.tonio
    async def test_reloads_global_settings_from_disk(self, dirs):
        agent_dir, project_dir = dirs
        settings_path = agent_dir / "settings.json"
        write_json(settings_path, {"theme": "dark", "extensions": ["/before.py"]})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        write_json(settings_path, {"theme": "light", "extensions": ["/after.py"], "defaultModel": "claude-sonnet"})

        await manager.reload()

        assert manager.get_theme() == "light"
        assert manager.get_extension_paths() == ["/after.py"]
        assert manager.get_default_model() == "claude-sonnet"

    @pytest.mark.tonio
    async def test_keeps_previous_settings_and_reports_the_file_path_when_the_file_is_invalid(self, dirs):
        agent_dir, project_dir = dirs
        settings_path = agent_dir / "settings.json"
        write_json(settings_path, {"theme": "dark"})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        settings_path.write_text("{ invalid json", encoding="utf-8")
        await manager.reload()

        assert manager.get_theme() == "dark"
        errors = manager.drain_errors()
        assert [(error.scope, error.path) for error in errors] == [("global", str(settings_path))]


class TestThemeSetting:
    @pytest.mark.tonio
    async def test_stores_slash_separated_automatic_theme_settings_separately_from_fixed_theme_names(self, dirs):
        agent_dir, project_dir = dirs
        settings_path = agent_dir / "settings.json"
        write_json(settings_path, {"theme": "light/dark"})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_theme() is None
        assert manager.get_theme_setting() == "light/dark"

        manager.set_theme("solarized-light/tokyo-night")
        await manager.flush()

        saved_settings = read_json(settings_path)
        assert saved_settings["theme"] == "solarized-light/tokyo-night"


class TestErrorTracking:
    @pytest.mark.tonio
    async def test_collects_and_clears_load_errors_via_drain_errors(self, dirs):
        agent_dir, project_dir = dirs
        global_settings_path = agent_dir / "settings.json"
        project_settings_path = project_dir / ".pidrei" / "settings.json"
        global_settings_path.write_text("{ invalid global json", encoding="utf-8")
        project_settings_path.write_text("{ invalid project json", encoding="utf-8")

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))
        errors = manager.drain_errors()

        assert [(error.scope, error.path) for error in errors] == [
            ("global", str(global_settings_path)),
            ("project", str(project_settings_path)),
        ]
        assert manager.drain_errors() == []


class TestProjectTrust:
    @pytest.mark.tonio
    async def test_skips_project_settings_when_project_is_not_trusted(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"theme": "global"})
        write_json(project_dir / ".pidrei" / "settings.json", {"theme": "project"})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir), project_trusted=False)

        assert manager.is_project_trusted() is False
        assert manager.get_theme() == "global"
        assert manager.get_project_settings() == {}

    @pytest.mark.tonio
    async def test_reloads_project_settings_after_trust_changes_to_true(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"theme": "global"})
        write_json(project_dir / ".pidrei" / "settings.json", {"theme": "project"})
        manager = await SettingsManager.create(str(project_dir), str(agent_dir), project_trusted=False)

        manager.set_project_trusted(True)

        assert manager.is_project_trusted() is True
        assert manager.get_theme() == "project"

    @pytest.mark.tonio
    async def test_fails_project_settings_writes_when_project_is_not_trusted(self, dirs):
        agent_dir, project_dir = dirs
        project_settings_path = project_dir / ".pidrei" / "settings.json"
        write_json(project_settings_path, {"packages": ["npm:existing"]})
        manager = await SettingsManager.create(str(project_dir), str(agent_dir), project_trusted=False)

        with pytest.raises(Exception, match="Project is not trusted; refusing to write project settings"):
            manager.set_project_packages(["npm:new"])
        await manager.flush()

        assert manager.get_project_settings() == {}
        assert read_json(project_settings_path) == {"packages": ["npm:existing"]}

    @pytest.mark.tonio
    async def test_reads_default_project_trust_from_global_settings_only(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"defaultProjectTrust": "always"})
        write_json(project_dir / ".pidrei" / "settings.json", {"defaultProjectTrust": "never"})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_default_project_trust() == "always"

    @pytest.mark.tonio
    async def test_defaults_invalid_project_trust_settings_to_ask(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"defaultProjectTrust": "sometimes"})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_default_project_trust() == "ask"


class TestProjectSettingsDirectoryCreation:
    @pytest.mark.tonio
    async def test_does_not_create_config_folder_when_only_reading_project_settings(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"theme": "dark"})

        (project_dir / ".pidrei").rmdir()

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        assert not (project_dir / ".pidrei").exists()
        assert manager.get_theme() == "dark"

    @pytest.mark.tonio
    async def test_creates_config_folder_when_writing_project_settings(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"theme": "dark"})

        (project_dir / ".pidrei").rmdir()

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        assert not (project_dir / ".pidrei").exists()

        manager.set_project_packages([{"source": "npm:test-pkg"}])
        await manager.flush()

        assert (project_dir / ".pidrei").exists()
        assert (project_dir / ".pidrei" / "settings.json").exists()


class TestHttpIdleTimeoutMs:
    @pytest.mark.tonio
    async def test_defaults_to_5_minutes(self, dirs):
        agent_dir, project_dir = dirs
        manager = await SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_http_idle_timeout_ms() == DEFAULT_HTTP_IDLE_TIMEOUT_MS

    @pytest.mark.tonio
    async def test_uses_merged_global_and_project_settings(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"httpIdleTimeoutMs": 300000})
        write_json(project_dir / ".pidrei" / "settings.json", {"httpIdleTimeoutMs": 0})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_http_idle_timeout_ms() == 0

    @pytest.mark.tonio
    async def test_rejects_invalid_timeout_values(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"httpIdleTimeoutMs": -1})
        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

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


class TestDefaultTools:
    @pytest.mark.tonio
    async def test_loads_global_defaults_and_lets_project_settings_replace_them(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"defaultTools": ["read", "bash"]})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_default_tools() == ["read", "bash"]

        write_json(project_dir / ".pidrei" / "settings.json", {"defaultTools": ["grep"]})

        reloaded = await SettingsManager.create(str(project_dir), str(agent_dir))
        assert reloaded.get_default_tools() == ["grep"]

    def test_preserves_an_empty_tool_list(self):
        assert SettingsManager.in_memory({"defaultTools": []}).get_default_tools() == []
        assert SettingsManager.in_memory().get_default_tools() is None


class TestFullscreenScrollbar:
    @pytest.mark.tonio
    async def test_validates_and_persists_fullscreen_settings(self, dirs):
        agent_dir, project_dir = dirs
        manager = await SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_fullscreen_exit_output() == "transcript"
        assert manager.get_fullscreen_scrollbar() == "auto"

        manager.set_fullscreen_exit_output("resume-hint")
        manager.set_fullscreen_scrollbar("hidden")
        await manager.flush()
        saved_settings = read_json(agent_dir / "settings.json")
        assert saved_settings["fullscreenExitOutput"] == "resume-hint"
        assert saved_settings["fullscreenScrollbar"] == "hidden"

        write_json(agent_dir / "settings.json", {"fullscreenExitOutput": "nothing", "fullscreenScrollbar": "sometimes"})
        reloaded = await SettingsManager.create(str(project_dir), str(agent_dir))
        assert reloaded.get_fullscreen_exit_output() == "transcript"
        assert reloaded.get_fullscreen_scrollbar() == "auto"


class TestTuiMode:
    @pytest.mark.tonio
    async def test_defaults_to_regular_and_persists_fullscreen_mode(self, dirs):
        agent_dir, project_dir = dirs
        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_tui_mode() == "regular"

        manager.set_tui_mode("fullscreen")
        await manager.flush()

        assert manager.get_tui_mode() == "fullscreen"
        assert read_json(agent_dir / "settings.json")["tuiMode"] == "fullscreen"

    @pytest.mark.tonio
    async def test_falls_back_to_regular_for_unsupported_values(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"tuiMode": "other"})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_tui_mode() == "regular"


class TestOutputPad:
    @pytest.mark.tonio
    async def test_defaults_to_1_and_persists_binary_values(self, dirs):
        agent_dir, project_dir = dirs
        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_output_pad() == 1

        manager.set_output_pad(0)
        await manager.flush()

        assert manager.get_output_pad() == 0
        saved_settings = read_json(agent_dir / "settings.json")
        assert saved_settings["outputPad"] == 0

    @pytest.mark.tonio
    async def test_treats_unsupported_output_pad_values_as_default_padding(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"outputPad": 2})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_output_pad() == 1


class TestShellCommandPrefix:
    @pytest.mark.tonio
    async def test_loads_shell_command_prefix_from_settings(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"shellCommandPrefix": "shopt -s expand_aliases"})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_shell_command_prefix() == "shopt -s expand_aliases"

    @pytest.mark.tonio
    async def test_returns_none_when_shell_command_prefix_is_not_set(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"theme": "dark"})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_shell_command_prefix() is None

    @pytest.mark.tonio
    async def test_preserves_shell_command_prefix_when_saving_unrelated_settings(self, dirs):
        agent_dir, project_dir = dirs
        settings_path = agent_dir / "settings.json"
        write_json(settings_path, {"shellCommandPrefix": "shopt -s expand_aliases"})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))
        manager.set_theme("light")
        await manager.flush()

        saved_settings = read_json(settings_path)
        assert saved_settings["shellCommandPrefix"] == "shopt -s expand_aliases"
        assert saved_settings["theme"] == "light"


class TestGetSessionDir:
    @pytest.mark.tonio
    async def test_returns_none_when_not_set(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"theme": "dark"})
        manager = await SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_session_dir() is None

    @pytest.mark.tonio
    async def test_returns_global_session_dir(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"sessionDir": "/tmp/sessions"})
        manager = await SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_session_dir() == "/tmp/sessions"

    @pytest.mark.tonio
    async def test_returns_project_session_dir_overriding_global(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"sessionDir": "/global/sessions"})
        write_json(project_dir / ".pidrei" / "settings.json", {"sessionDir": "./sessions"})
        manager = await SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_session_dir() == "./sessions"

    @pytest.mark.tonio
    async def test_expands_tilde_in_session_dir(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"sessionDir": "~/sessions"})
        manager = await SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_session_dir() == os.path.join(HOME, "sessions")


class TestGetShellPath:
    @pytest.mark.tonio
    async def test_returns_none_when_not_set(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"theme": "dark"})
        manager = await SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_shell_path() is None

    @pytest.mark.tonio
    async def test_returns_an_absolute_shell_path_unchanged(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"shellPath": "/bin/zsh"})
        manager = await SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_shell_path() == "/bin/zsh"

    @pytest.mark.tonio
    async def test_expands_tilde_in_shell_path(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"shellPath": "~/.local/bin/agent-shell-sandbox"})
        manager = await SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_shell_path() == os.path.join(HOME, ".local/bin/agent-shell-sandbox")

    @pytest.mark.tonio
    async def test_expands_a_bare_tilde_in_shell_path(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"shellPath": "~"})
        manager = await SettingsManager.create(str(project_dir), str(agent_dir))
        assert manager.get_shell_path() == HOME


class TestProviderAttributionMigration:
    """pidrei-only: `enableInstallTelemetry` -> `enableProviderAttribution`.

    pi has no tests for its own settings migrations, so this covers ours. The
    key it renames is the one thing standing between an existing config and a
    silently reset preference.
    """

    @pytest.mark.tonio
    async def test_carries_the_legacy_key_across(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"enableInstallTelemetry": False, "theme": "dark"})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_enable_provider_attribution() is False

    @pytest.mark.tonio
    async def test_drops_the_legacy_key_on_the_next_write(self, dirs):
        agent_dir, project_dir = dirs
        settings_path = agent_dir / "settings.json"
        write_json(settings_path, {"enableInstallTelemetry": False, "theme": "dark"})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))
        manager.set_theme("light")
        await manager.flush()

        saved = read_json(settings_path)
        assert "enableInstallTelemetry" not in saved
        assert saved["enableProviderAttribution"] is False

    @pytest.mark.tonio
    async def test_the_new_key_wins_when_both_are_present(self, dirs):
        agent_dir, project_dir = dirs
        write_json(
            agent_dir / "settings.json",
            {"enableInstallTelemetry": True, "enableProviderAttribution": False},
        )

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_enable_provider_attribution() is False

    @pytest.mark.tonio
    async def test_defaults_to_enabled_without_either_key(self, dirs):
        agent_dir, project_dir = dirs
        write_json(agent_dir / "settings.json", {"theme": "dark"})

        manager = await SettingsManager.create(str(project_dir), str(agent_dir))

        assert manager.get_enable_provider_attribution() is True
