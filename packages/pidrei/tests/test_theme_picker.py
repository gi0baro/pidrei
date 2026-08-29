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
def agent_dir(tmp_path, request):
    """The env var is restored by a finalizer (predates tonio 0.9.14;
    `monkeypatch` works in tonio tests now)."""
    agent_dir = tmp_path / "agent"
    (agent_dir / "themes").mkdir(parents=True)

    previous = os.environ.get(ENV_AGENT_DIR)
    os.environ[ENV_AGENT_DIR] = str(agent_dir)

    def restore() -> None:
        if previous is None:
            os.environ.pop(ENV_AGENT_DIR, None)
        else:
            os.environ[ENV_AGENT_DIR] = previous
        set_registered_themes([])

    set_registered_themes([])
    request.addfinalizer(restore)
    return agent_dir


class TestThemePicker:
    @pytest.mark.tonio
    async def test_uses_custom_theme_content_names_instead_of_file_names(self, agent_dir):
        with open(os.path.join(get_themes_dir(), "dark.json"), encoding="utf-8") as f:
            dark_theme = json.load(f)
        custom_theme = {**dark_theme, "name": "bar"}

        theme_path = agent_dir / "themes" / "foo.json"
        theme_path.write_text(json.dumps(custom_theme, indent=2))

        names = await get_available_themes()
        with_paths = await get_available_themes_with_paths()

        assert "bar" in names
        assert "foo" not in names
        assert {"name": "bar", "path": str(theme_path)} in with_paths
        assert not any(theme["name"] == "foo" for theme in with_paths)
