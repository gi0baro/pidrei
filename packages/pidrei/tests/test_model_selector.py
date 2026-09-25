"""Mirror of pi coding-agent test/model-selector.test.ts.

pi vitest-mocks `modelRuntime.refresh`; here the instance attribute is
replaced with a stub returning the same result.
"""

from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive.components.model_selector import ModelSelectorComponent
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei.utils.ansi import strip_ansi
from pidrei_ai.registry import ModelsRefreshResult
from pidrei_tui import set_keybindings

from .harness import create_harness
from .render_request_helpers import RenderRequests


@pytest.fixture(autouse=True)
def _setup():
    init_theme_sync("dark")
    # Keybindings are a global singleton; reset per test.
    set_keybindings(KeybindingsManager())


def fake_tui():
    return SimpleNamespace(request_render=lambda: None, post_ui=lambda fn: fn())


def render(selector: ModelSelectorComponent) -> str:
    return strip_ansi("\n".join(selector.render(120)))


@pytest.mark.tonio
async def test_keeps_the_current_model_marked_while_browsing():
    harness = await create_harness(
        models=[
            {"id": "current-model", "name": "Current Model", "reasoning": True},
            {"id": "browsed-model", "name": "Browsed Model", "reasoning": True},
        ]
    )
    refresh_released = tonio.Event()

    # pi's background catalog refresh is still pending when its synchronous
    # test body finishes; here it completes on another thread and would reset
    # the selection to the current model mid-test, so it is held pending.
    async def pending_refresh(_options=None, *, _requested_only=False):
        await refresh_released.wait()
        return ModelsRefreshResult(aborted=False, errors={})

    harness.session.model_runtime.refresh = pending_refresh
    try:
        current_model = harness.get_model("current-model")
        selector = ModelSelectorComponent(
            fake_tui(),
            current_model,
            harness.session.model_runtime,
            [],
            lambda *args: None,
            lambda *args: None,
        )

        def get_model_row(model_id: str) -> str | None:
            row = next((line for line in render(selector).split("\n") if f"{model_id} [" in line), None)
            return row.rstrip() if row is not None else None

        assert get_model_row("current-model") == f"→ ✓ current-model [{current_model.provider}]"
        await selector.handle_input("\x1b[B")
        assert get_model_row("current-model") == f"  ✓ current-model [{current_model.provider}]"
        assert get_model_row("browsed-model") == f"→   browsed-model [{current_model.provider}]"
        selector.dispose()
    finally:
        refresh_released.set()
        harness.cleanup()


@pytest.mark.tonio
async def test_uses_the_configured_save_binding():
    set_keybindings(KeybindingsManager({"app.models.save": "ctrl+r"}))
    harness = await create_harness()
    try:
        current_model = harness.get_model()
        save_default_calls = []
        selector = ModelSelectorComponent(
            fake_tui(),
            current_model,
            harness.session.model_runtime,
            [],
            lambda *args: None,
            lambda *args: None,
            None,
            save_default_calls.append,
        )

        assert "Ctrl+R to set as default" in render(selector)
        await selector.handle_input("\x13")
        assert save_default_calls == []
        await selector.handle_input("\x12")
        assert save_default_calls == [current_model]
    finally:
        harness.cleanup()


@pytest.mark.tonio
async def test_lists_every_catalog_that_failed_to_refresh():
    harness = await create_harness()
    try:

        async def failing_refresh(_options=None, *, _requested_only=False):
            return ModelsRefreshResult(
                aborted=False,
                errors={"openai": Exception("unavailable"), "anthropic": Exception("unavailable")},
            )

        harness.session.model_runtime.refresh = failing_refresh

        tui = RenderRequests()
        selector = ModelSelectorComponent(
            tui,
            harness.get_model(),
            harness.session.model_runtime,
            [],
            lambda *args: None,
            lambda *args: None,
        )

        await tui.until(lambda: "Could not refresh" in render(selector))

        assert "Could not refresh 2 model catalogs (openai, anthropic); showing cached models." in render(selector)
    finally:
        harness.cleanup()
