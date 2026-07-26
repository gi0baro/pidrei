"""Mirror of pi coding-agent test/theme-picker.test.ts."""

import json
import os

import pytest

from pidrei.config import ENV_AGENT_DIR, get_themes_dir
from pidrei.modes.interactive.theme import (
    get_available_themes,
    get_available_themes_with_paths,
    set_registered_themes,
)


@pytest.fixture
def agent_dir(tmp_path, monkeypatch, request):
    agent_dir = tmp_path / "agent"
    monkeypatch.setenv(ENV_AGENT_DIR, str(agent_dir))
    (agent_dir / "themes").mkdir(parents=True)
    set_registered_themes([])
    request.addfinalizer(lambda: set_registered_themes([]))
    return agent_dir


class TestThemePicker:
    def test_uses_custom_theme_content_names_instead_of_file_names(self, agent_dir):
        with open(os.path.join(get_themes_dir(), "dark.json"), encoding="utf-8") as f:
            dark_theme = json.load(f)
        custom_theme = {**dark_theme, "name": "bar"}

        theme_path = agent_dir / "themes" / "foo.json"
        theme_path.write_text(json.dumps(custom_theme, indent=2))

        assert "bar" in get_available_themes()
        assert "foo" not in get_available_themes()
        assert {"name": "bar", "path": str(theme_path)} in get_available_themes_with_paths()
        assert not any(theme["name"] == "foo" for theme in get_available_themes_with_paths())
