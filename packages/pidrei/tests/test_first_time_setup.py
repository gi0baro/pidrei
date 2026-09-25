"""Mirror of pi coding-agent test/first-time-setup.test.ts and
test/first-time-setup-fork.test.ts (the fork case patches PACKAGE_NAME like
pi's vi.mock of config.ts).

DIVERGED: pi's analytics-settings cases are not mirrored — pidrei sends no
telemetry, so it has no analytics opt-in or tracking identifier.
"""

import pytest

from pidrei.cli import startup_ui
from pidrei.config import ENV_AGENT_DIR
from pidrei.modes.interactive.components.first_time_setup import FirstTimeSetupComponent
from pidrei.modes.interactive.theme import init_theme_sync


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


class TestFirstTimeSetupComponent:
    @pytest.fixture(autouse=True)
    def _theme(self):
        init_theme_sync("dark")

    @pytest.mark.tonio
    async def test_confirming_the_theme_finishes_setup_without_an_analytics_step(self):
        submitted: list[dict] = []

        async def on_theme_preview(_theme_name: str) -> None:
            return None

        component = FirstTimeSetupComponent(
            {
                "detectedTheme": "light",
                "onThemePreview": on_theme_preview,
                "onSubmit": submitted.append,
                "onCancel": lambda: None,
            }
        )

        await component.handle_input("\n")

        assert submitted == [{"theme": "light"}]
