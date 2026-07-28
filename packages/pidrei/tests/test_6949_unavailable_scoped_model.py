"""Mirror of pi coding-agent test/suite/regressions/6949-unavailable-scoped-model.test.ts.

pi's harness only supplies model objects and cleanup here; the models are
built directly instead. The interactive-mode cases call the unbound
`InteractiveMode._show_models_selector` on a stub context, like pi calling
the prototype method on a mock — the selector factory runs synchronously via
the stubbed `_show_selector`.
"""

from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive.components.scoped_models_selector import ScopedModelsSelectorComponent
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei.utils.ansi import strip_ansi
from pidrei_ai.types import Model, ModelCost
from pidrei_tui import set_keybindings


ENTER = "\r"
CTRL_S = "\x13"
ALT_DOWN = "\x1b[1;3B"


@pytest.fixture(autouse=True)
def _setup():
    init_theme_sync("dark")
    # Keybindings are a global singleton; reset per test.
    set_keybindings(KeybindingsManager())


def make_model(model_id: str, name: str) -> Model:
    return Model(
        id=model_id,
        name=name,
        api="anthropic-messages",
        provider="faux",
        base_url="http://127.0.0.1:9",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=100000,
        max_tokens=4096,
    )


def create_interactive_context(*, all_models, enabled_model_ids, scoped_models=None):
    """The attribute surface `_show_models_selector` touches, as a stub."""
    holder = {"selector": None}
    get_available_calls = {"count": 0}
    set_scoped_models_calls = []

    async def refresh():
        return None

    async def get_available():
        get_available_calls["count"] += 1
        return list(all_models)

    def set_scoped_models(scoped):
        set_scoped_models_calls.append(scoped)

    def show_selector(factory):
        holder["selector"] = factory(lambda: None)["component"]

    context = SimpleNamespace(
        session=SimpleNamespace(
            model_runtime=SimpleNamespace(refresh=refresh, get_available=get_available),
            scoped_models=list(scoped_models or []),
            set_scoped_models=set_scoped_models,
        ),
        settings_manager=SimpleNamespace(
            get_enabled_models=lambda: list(enabled_model_ids),
            set_enabled_models=lambda patterns: None,
        ),
        show_status=lambda message: None,
        _show_selector=show_selector,
        _update_available_provider_count=lambda: None,
        ui=SimpleNamespace(request_render=lambda: None),
    )
    return context, get_available_calls, holder, set_scoped_models_calls


async def show_models_selector(context) -> None:
    await InteractiveMode._show_models_selector(context)


class TestUnavailableScopedModels:
    @pytest.mark.tonio
    async def test_shows_and_removes_an_enabled_model_without_a_catalog_entry(self):
        available = make_model("available", "Available")
        available_id = f"{available.provider}/{available.id}"
        unavailable_id = f"{available.provider}/unavailable"
        changes = []
        persisted = []
        selector = ScopedModelsSelectorComponent(
            {
                "allModels": [available],
                "enabledModelIds": [unavailable_id, available_id],
            },
            {
                "onChange": changes.append,
                "onPersist": persisted.append,
                "onCancel": lambda: None,
            },
        )

        assert f"{unavailable_id} [unavailable] ✗" in strip_ansi("\n".join(selector.render(100)))
        await selector.handle_input(ENTER)
        assert changes == [[available_id]]
        await selector.handle_input(CTRL_S)
        assert persisted == [[available_id]]

    @pytest.mark.tonio
    async def test_passes_unmatched_settings_patterns_to_the_selector_with_one_combined_resolution(self):
        unavailable_ids = ["faux/unavailable-one", "faux/unavailable-two"]
        context, get_available_calls, holder, _scoped_calls = create_interactive_context(
            all_models=[],
            enabled_model_ids=unavailable_ids,
        )

        await show_models_selector(context)

        selector = holder["selector"]
        assert selector is not None, "Expected scoped-model selector to open"
        rendered = strip_ansi("\n".join(selector.render(100)))
        for unavailable_id in unavailable_ids:
            assert f"{unavailable_id} [unavailable] ✗" in rendered
        assert get_available_calls["count"] == 2

    @pytest.mark.tonio
    async def test_opens_when_only_a_session_scoped_model_is_unavailable(self):
        model = make_model("unavailable", "Unavailable")
        full_id = f"{model.provider}/{model.id}"
        context, _calls, holder, _scoped_calls = create_interactive_context(
            all_models=[],
            enabled_model_ids=[],
            scoped_models=[SimpleNamespace(model=model)],
        )

        await show_models_selector(context)

        selector = holder["selector"]
        assert selector is not None, "Expected scoped-model selector to open"
        assert f"{full_id} [unavailable] ✗" in strip_ansi("\n".join(selector.render(100)))

    @pytest.mark.tonio
    async def test_does_not_clear_a_partial_scope_when_an_enabled_model_is_unavailable(self):
        models = [make_model("one", "One"), make_model("two", "Two"), make_model("three", "Three")]
        one, two = models[0], models[1]
        enabled_ids = [f"{model.provider}/{model.id}" for model in (one, two)]
        unavailable_id = f"{one.provider}/unavailable"
        context, _calls, holder, scoped_calls = create_interactive_context(
            all_models=models,
            enabled_model_ids=[*enabled_ids, unavailable_id],
            scoped_models=[SimpleNamespace(model=one), SimpleNamespace(model=two)],
        )

        await show_models_selector(context)
        selector = holder["selector"]
        assert selector is not None, "Expected scoped-model selector to open"
        await selector.handle_input(ALT_DOWN)

        # onChange resolves the session scope in a spawned task.
        await tonio.time.sleep(0.01)
        assert scoped_calls, "Expected the session scope to be updated"
        last = scoped_calls[-1]
        assert [scoped.model for scoped in last] == [two, one]
        assert all(scoped.thinking_level is None for scoped in last)
