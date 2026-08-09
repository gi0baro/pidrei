"""Mirror of pi coding-agent test/scrollbar-theme.test.ts."""

import json
import os

import pytest

from pidrei.modes.interactive.theme.theme import load_theme_from_path


_DARK_THEME_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pidrei",
    "modes",
    "interactive",
    "theme",
    "dark.json",
)


def _load_dark_theme() -> dict:
    with open(_DARK_THEME_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _write_theme(tmp_dir, theme: dict) -> str:
    theme_path = tmp_dir / f"{theme['name']}.json"
    theme_path.write_text(json.dumps(theme), encoding="utf-8")
    return str(theme_path)


@pytest.mark.tonio
async def test_falls_back_to_selected_bg_when_scrollbar_thumb_is_omitted(tmp_dir):
    theme_json = _load_dark_theme()
    theme_json["name"] = "legacy-scrollbar-theme"
    del theme_json["colors"]["scrollbarThumb"]

    loaded_theme = await load_theme_from_path(_write_theme(tmp_dir, theme_json), "truecolor")

    assert loaded_theme.get_bg_ansi("scrollbarThumb") == loaded_theme.get_bg_ansi("selectedBg")


@pytest.mark.tonio
async def test_uses_an_explicitly_configured_scrollbar_thumb(tmp_dir):
    theme_json = _load_dark_theme()
    theme_json["name"] = "custom-scrollbar-theme"
    theme_json["colors"]["scrollbarThumb"] = "#123456"

    loaded_theme = await load_theme_from_path(_write_theme(tmp_dir, theme_json), "truecolor")

    assert loaded_theme.get_bg_ansi("scrollbarThumb") == "\x1b[48;2;18;52;86m"
