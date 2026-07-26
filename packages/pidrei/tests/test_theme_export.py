"""Mirror of pi coding-agent test/theme-export.test.ts."""

import json
import os

import pytest

from pidrei.config import ENV_AGENT_DIR, get_themes_dir
from pidrei.modes.interactive.theme import get_theme_export_colors


@pytest.fixture
def agent_dir(tmp_path, monkeypatch):
    agent_dir = tmp_path / "agent"
    monkeypatch.setenv(ENV_AGENT_DIR, str(agent_dir))
    (agent_dir / "themes").mkdir(parents=True)
    return agent_dir


def _load_dark_theme() -> dict:
    with open(os.path.join(get_themes_dir(), "dark.json"), encoding="utf-8") as f:
        return json.load(f)


class TestGetThemeExportColors:
    def test_resolves_export_variable_references_using_the_same_syntax_as_colors(self, agent_dir):
        dark_theme = _load_dark_theme()

        custom_theme = {
            **dark_theme,
            "name": "custom-export-vars",
            "vars": {
                **dark_theme.get("vars", {}),
                "pageBgVar": "#112233",
                "pageBgAlias": "pageBgVar",
                "infoBgVar": "#445566",
                "cardBgVar": "#223344",
            },
            "export": {
                "pageBg": "pageBgAlias",
                "cardBg": "cardBgVar",
                "infoBg": "infoBgVar",
            },
        }

        (agent_dir / "themes" / "custom-export-vars.json").write_text(json.dumps(custom_theme, indent=2))

        assert get_theme_export_colors("custom-export-vars") == {
            "pageBg": "#112233",
            "cardBg": "#223344",
            "infoBg": "#445566",
        }

    def test_resolves_recursive_vars_and_converts_256_color_export_values_to_hex(self, agent_dir):
        dark_theme = _load_dark_theme()

        custom_theme = {
            **dark_theme,
            "name": "custom-export-recursive",
            "vars": {
                **dark_theme.get("vars", {}),
                "deepPageBg": "#abcdef",
                "pageBgAlias": "deepPageBg",
                "cardBgAnsi": 24,
            },
            "export": {
                "pageBg": "pageBgAlias",
                "cardBg": "cardBgAnsi",
                "infoBg": "",
            },
        }

        (agent_dir / "themes" / "custom-export-recursive.json").write_text(json.dumps(custom_theme, indent=2))

        assert get_theme_export_colors("custom-export-recursive") == {
            "pageBg": "#abcdef",
            "cardBg": "#005f87",
            "infoBg": None,
        }
