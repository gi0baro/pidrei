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


def _write_theme(tmp_path, theme: dict) -> str:
    theme_path = tmp_path / f"{theme['name']}.json"
    theme_path.write_text(json.dumps(theme), encoding="utf-8")
    return str(theme_path)


@pytest.mark.tonio
@pytest.mark.parametrize(("token", "fallback"), [("scrollbarTrack", "muted"), ("scrollbarThumb", "text")])
async def test_falls_back_when_a_scrollbar_color_is_omitted(tmp_path, token, fallback):
    theme_json = _load_dark_theme()
    theme_json["name"] = f"missing-{token}-theme"
    del theme_json["colors"][token]

    loaded_theme = await load_theme_from_path(_write_theme(tmp_path, theme_json), "truecolor")

    assert loaded_theme.get_fg_ansi(token) == loaded_theme.get_fg_ansi(fallback)


@pytest.mark.tonio
async def test_uses_explicitly_configured_scrollbar_colors(tmp_path):
    theme_json = _load_dark_theme()
    theme_json["name"] = "custom-scrollbar-theme"
    theme_json["colors"]["scrollbarTrack"] = "#654321"
    theme_json["colors"]["scrollbarThumb"] = "#123456"

    loaded_theme = await load_theme_from_path(_write_theme(tmp_path, theme_json), "truecolor")

    assert loaded_theme.get_fg_ansi("scrollbarTrack") == "\x1b[38;2;101;67;33m"
    assert loaded_theme.get_fg_ansi("scrollbarThumb") == "\x1b[38;2;18;52;86m"


@pytest.mark.tonio
async def test_falls_back_to_existing_selection_and_text_colors_for_search_highlights(tmp_path):
    theme_json = _load_dark_theme()
    theme_json["name"] = "legacy-search-theme"
    del theme_json["colors"]["searchMatchBg"]
    del theme_json["colors"]["searchMatchText"]

    loaded_theme = await load_theme_from_path(_write_theme(tmp_path, theme_json), "truecolor")

    assert loaded_theme.get_bg_ansi("searchMatchBg") == loaded_theme.get_bg_ansi("selectedBg")
    assert loaded_theme.get_fg_ansi("searchMatchText") == loaded_theme.get_fg_ansi("text")


@pytest.mark.tonio
async def test_uses_explicitly_configured_search_highlight_colors(tmp_path):
    theme_json = _load_dark_theme()
    theme_json["name"] = "custom-search-theme"
    theme_json["colors"]["searchMatchBg"] = "#112233"
    theme_json["colors"]["searchMatchText"] = "#223344"

    loaded_theme = await load_theme_from_path(_write_theme(tmp_path, theme_json), "truecolor")

    assert loaded_theme.get_bg_ansi("searchMatchBg") == "\x1b[48;2;17;34;51m"
    assert loaded_theme.get_fg_ansi("searchMatchText") == "\x1b[38;2;34;51;68m"
