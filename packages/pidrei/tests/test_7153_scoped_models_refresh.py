"""Mirror of pi coding-agent test/suite/regressions/7153-scoped-models-refresh.test.ts."""

from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei.utils.ansi import strip_ansi
from pidrei_ai.registry import ModelsRefreshResult
from pidrei_tui import set_keybindings

from .harness import create_harness


ESC = "\x1b"


@pytest.fixture(autouse=True)
def _setup():
    init_theme_sync("dark")
    # Keybindings are a global singleton; reset per test.
    set_keybindings(KeybindingsManager())


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


class OpenedSelector:
    def __init__(self, harness, initial_models):
        self.snapshot = list(initial_models)
        self.refresh_cancel = None
        self.selector = None
        self.dispose = None
        self.done_calls = 0
        self._finish = tonio.Event()
        self._result: ModelsRefreshResult | None = None

        runtime = harness.session.model_runtime
        runtime.get_available_snapshot = lambda: list(self.snapshot)

        async def refresh(options=None):
            if options is None or options.cancel is None:
                # Leftover harness `_request_refresh` drain — not the selector's.
                return ModelsRefreshResult(aborted=False, errors={})
            self.refresh_cancel = options.cancel
            await self._finish.wait()
            return self._result if self._result is not None else ModelsRefreshResult(aborted=True, errors={})

        runtime.refresh = refresh

        def show_selector(factory):
            def close() -> None:
                if self.dispose is not None:
                    self.dispose()
                self.done_calls += 1

            created = factory(close)
            self.selector = created["component"]
            self.dispose = created.get("dispose")

        self.context = SimpleNamespace(
            session=harness.session,
            settings_manager=harness.settings_manager,
            _show_selector=show_selector,
            _update_available_provider_count=lambda: None,
            show_status=lambda message: None,
            ui=SimpleNamespace(request_render=lambda force=False: None),
        )

        InteractiveMode._show_models_selector(self.context)
        assert self.selector is not None, "Expected scoped-model selector to open"

    def complete(self, models, result: ModelsRefreshResult) -> None:
        self.snapshot = list(models)
        self._result = result
        self._finish.set()

    def release(self) -> None:
        self._finish.set()


def render(selector) -> str:
    return strip_ansi("\n".join(selector.render(100)))


@pytest.mark.tonio
async def test_renders_cached_models_immediately_and_updates_after_background_refresh(harnesses):
    harness = await create_harness(
        models=[{"id": "cached", "name": "Cached"}, {"id": "refreshed", "name": "Refreshed"}]
    )
    harnesses.append(harness)
    all_models = [harness.get_model("cached"), harness.get_model("refreshed")]
    opened = OpenedSelector(harness, [all_models[0]])

    try:
        initial = render(opened.selector)
        assert "cached" in initial
        assert "Refreshing model catalogs…" in initial
        assert "refreshed" not in initial

        opened.complete(all_models, ModelsRefreshResult(aborted=False, errors={}))
        for _ in range(200):
            rendered = render(opened.selector)
            if "refreshed" in rendered and "Model catalogs refreshed." in rendered:
                break
            await tonio.time.sleep(0.005)
        rendered = render(opened.selector)
        assert "refreshed" in rendered
        assert "Model catalogs refreshed." in rendered
    finally:
        opened.release()


@pytest.mark.tonio
async def test_cancels_the_background_refresh_when_the_selector_closes(harnesses):
    harness = await create_harness(models=[{"id": "cached", "name": "Cached"}])
    harnesses.append(harness)
    opened = OpenedSelector(harness, [harness.get_model("cached")])

    try:
        await tonio.time.sleep(0.005)
        assert opened.refresh_cancel is not None
        await opened.selector.handle_input(ESC)
        assert opened.refresh_cancel.cancelled is True
        assert opened.done_calls == 1
    finally:
        opened.release()
