"""Mirror of pi coding-agent test/scoped-models-selector.test.ts."""

import pytest

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive.components.scoped_models_selector import ScopedModelsSelectorComponent
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import set_keybindings

from .harness import create_harness


ENTER = "\r"
CTRL_A = "\x01"
CTRL_X = "\x18"
DOWN = "\x1b[B"


@pytest.fixture(autouse=True)
def _setup():
    init_theme_sync("dark")
    set_keybindings(KeybindingsManager())


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


def render(selector: ScopedModelsSelectorComponent) -> str:
    return strip_ansi("\n".join(selector.render(120)))


def get_marker_states(selector: ScopedModelsSelectorComponent, models: list[dict]) -> list[bool]:
    lines = render(selector).split("\n")
    states = []
    for model in models:
        line = next((candidate for candidate in lines if f"{model['id']} [" in candidate), None)
        if line is None:
            raise AssertionError(f"Expected rendered row for {model['id']}")
        states.append(line[2:].startswith("✓ "))
    return states


async def create_selector(harnesses: list, models: list[dict]) -> ScopedModelsSelectorComponent:
    harness = await create_harness(models=[{"id": model["id"], "name": model["name"]} for model in models])
    harnesses.append(harness)
    provider = harness.models[0].provider

    def on_change(enabled_model_ids) -> None:
        for model in models:
            model["enabled"] = enabled_model_ids is None or f"{provider}/{model['id']}" in enabled_model_ids

    return ScopedModelsSelectorComponent(
        {
            "allModels": list(harness.models),
            "enabledModelIds": [f"{provider}/{model['id']}" for model in models if model["enabled"]],
        },
        {"onChange": on_change, "onPersist": lambda *args: None, "onCancel": lambda: None},
    )


@pytest.mark.tonio
async def test_marks_every_model_after_enabling_all(harnesses):
    models = [
        {"id": "model-a", "name": "Model A", "enabled": True},
        {"id": "model-b", "name": "Model B", "enabled": False},
        {"id": "model-c", "name": "Model C", "enabled": False},
    ]
    selector = await create_selector(harnesses, models)

    await selector.handle_input(CTRL_A)

    assert [model["enabled"] for model in models] == [True, True, True]
    assert get_marker_states(selector, models) == [True, True, True]
    assert "all enabled" in render(selector)


@pytest.mark.tonio
async def test_disables_only_the_selected_model_after_enabling_all(harnesses):
    models = [
        {"id": "model-a", "name": "Model A", "enabled": True},
        {"id": "model-b", "name": "Model B", "enabled": False},
        {"id": "model-c", "name": "Model C", "enabled": False},
    ]
    selector = await create_selector(harnesses, models)

    await selector.handle_input(CTRL_A)
    await selector.handle_input(ENTER)

    assert [model["enabled"] for model in models] == [False, True, True]
    assert get_marker_states(selector, models) == [False, True, True]


@pytest.mark.tonio
async def test_enables_only_the_selected_model_after_clearing_all(harnesses):
    models = [
        {"id": "model-a", "name": "Model A", "enabled": True},
        {"id": "model-b", "name": "Model B", "enabled": True},
        {"id": "model-c", "name": "Model C", "enabled": True},
    ]
    selector = await create_selector(harnesses, models)

    await selector.handle_input(CTRL_X)
    assert [model["enabled"] for model in models] == [False, False, False]
    assert get_marker_states(selector, models) == [False, False, False]

    await selector.handle_input(ENTER)
    assert [model["enabled"] for model in models] == [True, False, False]
    assert get_marker_states(selector, models) == [True, False, False]


@pytest.mark.tonio
async def test_restores_the_all_enabled_state_after_re_enabling_the_last_disabled_model(harnesses):
    models = [
        {"id": "model-a", "name": "Model A", "enabled": True},
        {"id": "model-b", "name": "Model B", "enabled": False},
        {"id": "model-c", "name": "Model C", "enabled": False},
    ]
    selector = await create_selector(harnesses, models)

    await selector.handle_input(CTRL_A)  # enable all -> None
    await selector.handle_input(ENTER)  # disable model-a; enabled models re-sort first: [b, c, a]
    await selector.handle_input(DOWN)
    await selector.handle_input(DOWN)  # move selection back to model-a
    await selector.handle_input(ENTER)  # re-enable model-a

    assert [model["enabled"] for model in models] == [True, True, True]
    assert get_marker_states(selector, models) == [True, True, True]
    assert "all enabled" in render(selector)
