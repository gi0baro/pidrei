"""Mirror of pi coding-agent test/first-time-setup.test.ts and
test/first-time-setup-fork.test.ts (the fork case patches PACKAGE_NAME like
pi's vi.mock of config.ts)."""

import re

import pytest

from pidrei.cli import startup_ui
from pidrei.config import ENV_AGENT_DIR
from pidrei.core.settings_manager import SettingsManager


class TestShouldRunFirstTimeSetup:
    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        monkeypatch.setenv("PIDREI_EXPERIMENTAL", "1")
        monkeypatch.delenv(ENV_AGENT_DIR, raising=False)

    def test_returns_true_when_experimental_default_agent_dir_and_no_settings_json(self, tmp_path):
        assert startup_ui.should_run_first_time_setup(str(tmp_path / "settings.json")) is True

    def test_returns_false_when_experimental_features_are_disabled(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PIDREI_EXPERIMENTAL", raising=False)

        assert startup_ui.should_run_first_time_setup(str(tmp_path / "settings.json")) is False

    def test_returns_false_when_a_custom_agent_dir_is_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_AGENT_DIR, str(tmp_path))

        assert startup_ui.should_run_first_time_setup(str(tmp_path / "settings.json")) is False

    def test_returns_false_when_settings_json_already_exists(self, tmp_path):
        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}")

        assert startup_ui.should_run_first_time_setup(str(settings_path)) is False


class TestShouldRunFirstTimeSetupInForkedDistributions:
    def test_returns_false_for_a_forked_package(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PIDREI_EXPERIMENTAL", "1")
        monkeypatch.delenv(ENV_AGENT_DIR, raising=False)
        monkeypatch.setattr(startup_ui, "PACKAGE_NAME", "@example/pidrei-coding-agent")

        assert startup_ui.should_run_first_time_setup(str(tmp_path / "settings.json")) is False


class TestAnalyticsSettings:
    def test_defaults_to_disabled_with_no_tracking_identifier(self):
        manager = SettingsManager.in_memory()

        assert manager.get_enable_analytics() is False
        assert manager.get_tracking_id() is None

    def test_generates_a_tracking_identifier_on_opt_in(self):
        manager = SettingsManager.in_memory()

        manager.set_enable_analytics(True)

        assert manager.get_enable_analytics() is True
        assert re.fullmatch(r"[0-9a-f-]{36}", manager.get_tracking_id())

    def test_does_not_generate_a_tracking_identifier_on_opt_out(self):
        manager = SettingsManager.in_memory()

        manager.set_enable_analytics(False)

        assert manager.get_enable_analytics() is False
        assert manager.get_tracking_id() is None

    def test_keeps_the_tracking_identifier_when_toggling_analytics(self):
        manager = SettingsManager.in_memory()

        manager.set_enable_analytics(True)
        tracking_id = manager.get_tracking_id()
        manager.set_enable_analytics(False)
        manager.set_enable_analytics(True)

        assert manager.get_tracking_id() == tracking_id
