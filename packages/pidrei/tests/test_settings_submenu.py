"""Per-model thinking-level submenu flow (pidrei-only).

pi ships `settings-submenu.ts` untested; the port cannot inherit that, because
every callback on the path — `on_select`, `on_cancel`, `on_complete`, and the
`done` handed down by `SettingsList` — became a coroutine here, and a single
un-awaited one silently drops a step instead of failing. This drives the whole
two-step selector the way the settings list does: pick a model, pick a level,
loop back, and clear the override.
"""

import pytest

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive.components.settings_selector import SettingsSelectorComponent
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei.utils.ansi import strip_ansi
from pidrei_ai.providers.faux import FauxModelDefinition, faux_provider
from pidrei_tui import set_keybindings

from .test_settings_selector import BASE_CONFIG


DOWN = "\x1b[B"
ENTER = "\r"
ESCAPE = "\x1b"


@pytest.fixture(autouse=True)
def _theme():
    init_theme_sync("dark")
    set_keybindings(KeybindingsManager(None, None))


def _open_model_thinking_submenu(model_thinking_levels: dict):
    models = faux_provider(
        models=[
            FauxModelDefinition(id="faux-1", name="One", reasoning=True),
            FauxModelDefinition(id="faux-2", name="Two", reasoning=True),
        ]
    ).models
    changes: list = []
    removals: list = []

    async def on_change(provider: str, model_id: str, level: str) -> None:
        changes.append((provider, model_id, level))

    async def on_remove(provider: str, model_id: str) -> None:
        removals.append((provider, model_id))

    async def on_cancel() -> None:
        pass

    selector = SettingsSelectorComponent(
        {
            **BASE_CONFIG,
            "availableDefaultModels": models,
            "currentModel": models[0],
            "defaultModel": "faux/faux-1",
            "modelThinkingLevels": model_thinking_levels,
            "thinkingLevel": "medium",
        },
        {
            "onModelThinkingLevelChange": on_change,
            "onModelThinkingLevelRemove": on_remove,
            "onWarningsChange": lambda warnings: None,
            "onCancel": on_cancel,
        },
    )
    return selector.get_settings_list(), changes, removals


async def _select_item(settings_list, label: str) -> None:
    for character in label:
        await settings_list.handle_input(character)
    await settings_list.handle_input(ENTER)


def _render(settings_list) -> str:
    return strip_ansi("\n".join(settings_list.render(100)))


@pytest.mark.tonio
async def test_records_a_per_model_override_and_loops_back_to_the_model_step():
    settings_list, changes, removals = _open_model_thinking_submenu({})

    await _select_item(settings_list, "Default thinking level per model")
    assert "Step 1/2 · Select a model to configure" in _render(settings_list)

    # Second model, second level: faux-2 → minimal.
    await settings_list.handle_input(DOWN)
    await settings_list.handle_input(ENTER)
    assert "Thinking Level for faux-2 [faux]" in _render(settings_list)

    await settings_list.handle_input(DOWN)
    await settings_list.handle_input(ENTER)

    assert changes == [("faux", "faux-2", "minimal")]
    assert removals == []

    # `loop` returns to the model step, now showing the override it just wrote.
    rendered = _render(settings_list)
    assert "Step 1/2 · Select a model to configure" in rendered
    assert "faux-2 [faux]  minimal" in rendered

    # Leaving the submenu writes the summary back onto the settings item.
    await settings_list.handle_input(ESCAPE)
    assert "Default thinking level per model  1 configured" in _render(settings_list)


@pytest.mark.tonio
async def test_offers_clearing_an_existing_override():
    settings_list, changes, removals = _open_model_thinking_submenu({"faux/faux-2": "minimal"})

    await _select_item(settings_list, "Default thinking level per model")
    await settings_list.handle_input(DOWN)
    await settings_list.handle_input(ENTER)

    rendered = _render(settings_list)
    assert "(clear override)" in rendered
    assert "Revert to global default (medium)" in rendered

    # The clear entry sits after the five supported levels; the list preselects
    # the current override, so walk down from there.
    for _ in range(4):
        await settings_list.handle_input(DOWN)
    await settings_list.handle_input(ENTER)

    assert removals == [("faux", "faux-2")]
    assert changes == []

    await settings_list.handle_input(ESCAPE)
    assert "Default thinking level per model  none" in _render(settings_list)
