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
from .render_request_helpers import RenderRequests


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
        # The selector's refresh runs detached: `refresh_started` marks the
        # stub reached (its cancel token captured), and `renders` is re-checked
        # on every `request_render` (the refresh's last step).
        self.refresh_started = tonio.Event()
        self.renders = RenderRequests()

        runtime = harness.session.model_runtime
        runtime.get_available_snapshot = lambda: list(self.snapshot)

        async def refresh(options=None, *, _requested_only=False):
            if options is None or options.cancel is None:
                # Leftover harness `_request_refresh` drain — not the selector's.
                # (It calls with `_requested_only=True`; a stub without the
                # keyword dies as an UNHANDLED detached task, seen on macOS CI.)
                return ModelsRefreshResult(aborted=False, errors={})
            self.refresh_cancel = options.cancel
            self.refresh_started.set()
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
            ui=self.renders,
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
        await opened.renders.until(lambda: "Model catalogs refreshed." in render(opened.selector))
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
        await opened.refresh_started.wait(5)
        assert opened.refresh_cancel is not None
        await opened.selector.handle_input(ESC)
        # 7d8c11d3: the shared coordinator aborts the runtime refresh only
        # after the last waiter detaches (pi switched this to vi.waitFor).
        await opened.refresh_cancel.wait(5)
        assert opened.refresh_cancel.cancelled is True
        assert opened.done_calls == 1
    finally:
        opened.release()
