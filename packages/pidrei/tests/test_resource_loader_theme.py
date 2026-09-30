"""Mirror of pi coding-agent test/resource-loader-theme.test.ts."""

import json
import os

import pytest

from pidrei.config import get_themes_dir
from pidrei.core.resource_loader import DefaultResourceLoader
from pidrei.core.settings_manager import SettingsManager
from pidrei_tui import reset_capabilities_cache, set_capability_overrides


@pytest.fixture
def theme_dirs(tmp_path, monkeypatch):
    agent_dir = tmp_path / "agent"
    cwd = tmp_path / "project"
    agent_dir.mkdir()
    cwd.mkdir()
    with open(os.path.join(get_themes_dir(), "dark.json"), encoding="utf-8") as f:
        theme_json = json.load(f)
    theme_json["name"] = "capability-test"
    theme_json["colors"]["userMessageBg"] = "#3c3544"
    theme_path = tmp_path / "capability-test.json"
    theme_path.write_text(json.dumps(theme_json), encoding="utf-8")
    yield agent_dir, cwd, theme_path
    set_capability_overrides({})
    reset_capabilities_cache()


async def _loaded_theme(agent_dir, cwd, theme_path, settings_manager):
    loader = await DefaultResourceLoader(
        cwd=str(cwd),
        agent_dir=str(agent_dir),
        settings_manager=settings_manager,
        additional_theme_paths=[str(theme_path)],
        no_extensions=True,
        no_skills=True,
        no_prompt_templates=True,
        no_context_files=True,
    )
    await loader.reload()
    return loader, next(theme for theme in loader.get_themes()["themes"] if theme.name == "capability-test")


# Regression test for #9973.
@pytest.mark.tonio
@pytest.mark.parametrize(
    ("environment_override", "setting", "expected"),
    [
        # 256-color environment, truecolor setting.
        ("0", True, "\x1b[48;2;60;53;68mx\x1b[49m"),
        # Truecolor environment, 256-color setting.
        ("1", False, "\x1b[48;5;59mx\x1b[49m"),
    ],
)
async def test_uses_the_setting_over_the_environment(theme_dirs, monkeypatch, environment_override, setting, expected):
    agent_dir, cwd, theme_path = theme_dirs
    monkeypatch.setenv("PIDREI_TRUE_COLOR", environment_override)
    set_capability_overrides({})
    reset_capabilities_cache()

    settings_manager = SettingsManager.in_memory({"terminal": {"trueColor": setting}})
    _loader, loaded_theme = await _loaded_theme(agent_dir, cwd, theme_path, settings_manager)
    assert loaded_theme.bg("userMessageBg", "x") == expected


@pytest.mark.tonio
async def test_returns_to_automatic_detection_after_an_explicit_setting_is_removed(theme_dirs, monkeypatch):
    agent_dir, cwd, theme_path = theme_dirs
    monkeypatch.setenv("PIDREI_TRUE_COLOR", "1")
    set_capability_overrides({})
    reset_capabilities_cache()

    settings_path = agent_dir / "settings.json"
    settings_path.write_text(json.dumps({"terminal": {"trueColor": False}}), encoding="utf-8")
    settings_manager = await SettingsManager(str(cwd), str(agent_dir))
    loader, _loaded = await _loaded_theme(agent_dir, cwd, theme_path, settings_manager)
    set_capability_overrides(settings_manager.get_terminal_capability_overrides())

    settings_path.write_text("{}", encoding="utf-8")
    await loader.reload()

    loaded_theme = next(theme for theme in loader.get_themes()["themes"] if theme.name == "capability-test")
    assert loaded_theme.bg("userMessageBg", "x") == "\x1b[48;2;60;53;68mx\x1b[49m"
