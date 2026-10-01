"""Mirror of pi coding-agent test/theme-controller.test.ts.

pi's constructor calls `initTheme` inline; pidrei defers file-reading theme
initialization to `prime()` (see the controller docstring), so each test
awaits `prime()` where pi asserts right after construction.

pi mocks `queryTerminalColors` and resolves it (or calls `onLateReply`); here
the controller is the fake UI's `on_terminal_colors` listener, and the fake
answers a query the way the TUI's terminal-event loop does: the listener,
then the query's applied event.
"""

import threading

import pytest
import tonio.colored as tonio

from pidrei.core.settings_manager import SettingsManager
from pidrei.modes.interactive.theme import init_theme, set_terminal_color_scheme, set_terminal_colors, theme
from pidrei.modes.interactive.theme.theme_controller import InteractiveThemeController


DARK = {"foreground": {"r": 248, "g": 248, "b": 242}, "background": {"r": 40, "g": 42, "b": 54}}
LIGHT = {"foreground": {"r": 30, "g": 30, "b": 30}, "background": {"r": 250, "g": 250, "b": 250}}


class _FakeUi:
    def __init__(self):
        self.query_timeouts: list[float] = []
        self.notification_calls: list[bool] = []
        self.scheme_unsubscribe_calls = 0
        self.colors_unsubscribe_calls = 0
        self.render_requests = 0
        self._scheme_listener = None
        self._colors_listener = None
        self._pending_queries: list[tonio.Event] = []
        self.state_lock = threading.RLock()

    def invalidate(self) -> None:
        pass

    def request_render(self) -> None:
        self.render_requests += 1

    async def set_terminal_color_scheme_notifications(self, enabled: bool) -> None:
        self.notification_calls.append(enabled)

    def on_terminal_color_scheme_change(self, listener):
        self._scheme_listener = listener

        def unsubscribe() -> None:
            self.scheme_unsubscribe_calls += 1

        return unsubscribe

    def on_terminal_colors(self, listener):
        self._colors_listener = listener

        def unsubscribe() -> None:
            self.colors_unsubscribe_calls += 1

        return unsubscribe

    def query_terminal_colors(self, *, timeout_ms):
        self.query_timeouts.append(timeout_ms)
        applied = tonio.Event()
        self._pending_queries.append(applied)
        return applied

    async def answer(self, colors: dict) -> None:
        """The oldest query's first report: completed, or timed out with
        what arrived."""
        applied = self._pending_queries.pop(0)
        await self._colors_listener(colors)
        applied.set()

    async def late_reply(self, colors: dict) -> None:
        await self._colors_listener(colors)

    async def emit_terminal_color_scheme(self, terminal_theme: str) -> None:
        await self._scheme_listener(terminal_theme)


class _SpyManager:
    """Records set_theme/flush calls on a wrapped SettingsManager."""

    def __init__(self, manager: SettingsManager):
        self._manager = manager
        self.set_theme_calls: list[str] = []
        self.flush_calls = 0

    def __getattr__(self, name):
        return getattr(self._manager, name)

    def set_theme(self, name: str) -> None:
        self.set_theme_calls.append(name)
        self._manager.set_theme(name)

    def flush(self) -> None:
        self.flush_calls += 1
        self._manager.flush()


@pytest.fixture(autouse=True)
async def _reset_theme():
    await init_theme("dark")
    yield
    set_terminal_colors({})
    set_terminal_color_scheme(None)
    await init_theme("dark")


def _create_controller(ui, get_settings_manager, initial_theme_setting=None):
    return InteractiveThemeController(
        ui,
        {
            "getSettingsManager": get_settings_manager,
            "showError": lambda _message: None,
            "onChanged": lambda: None,
            "initialThemeSetting": initial_theme_setting,
        },
    )


@pytest.mark.tonio
async def test_uses_the_initial_theme_without_persisting_it():
    ui = _FakeUi()
    manager = _SpyManager(SettingsManager.in_memory({"theme": "dark"}))
    controller = _create_controller(ui, lambda: manager, "light")
    await controller.prime()

    assert theme.name == "light"
    assert controller.get_theme_selection() == "light"
    await controller.apply_from_settings()

    assert len(ui.query_timeouts) == 1
    assert manager.set_theme_calls == []
    assert manager.flush_calls == 0


@pytest.mark.tonio
async def test_applies_the_theme_immediately_and_lets_startup_wait_for_the_colors():
    ui = _FakeUi()
    controller = _create_controller(ui, lambda: SettingsManager.in_memory())
    await controller.prime()
    await controller.apply_from_settings()

    # Grayscale until the terminal answers.
    assert theme.name == "system"
    assert theme.get_fg_ansi("error") == "\x1b[39m"

    await ui.answer(DARK)
    await controller.wait_for_terminal_colors()
    assert theme.get_fg_ansi("error").startswith("\x1b[38;")


@pytest.mark.tonio
async def test_falls_back_to_palette_indices_then_applies_colors_that_arrive_after_the_timeout():
    ui = _FakeUi()
    controller = _create_controller(ui, lambda: SettingsManager.in_memory())
    await controller.prime()
    await controller.apply_from_settings()
    await ui.answer({})
    assert theme.get_fg_ansi("error") == "\x1b[38;5;1m"

    await ui.late_reply(DARK)
    assert theme.colors["error"].kind == "rgb"


@pytest.mark.tonio
async def test_re_queries_the_colors_on_appearance_changes_and_lets_them_decide():
    ui = _FakeUi()
    controller = _create_controller(ui, lambda: SettingsManager.in_memory(), "light/dark")
    await controller.prime()
    await controller.apply_from_settings()
    assert True in ui.notification_calls
    await ui.answer(LIGHT)
    assert theme.name == "light"

    # The report says light, but the terminal renders dark.
    await ui.emit_terminal_color_scheme("light")
    await ui.answer(DARK)
    assert theme.name == "dark"


@pytest.mark.tonio
async def test_uses_the_reported_scheme_for_the_system_theme_when_the_terminal_reports_no_colors(monkeypatch):
    monkeypatch.setenv("COLORFGBG", "")
    ui = _FakeUi()
    controller = _create_controller(ui, lambda: SettingsManager.in_memory())
    await controller.prime()
    await controller.apply_from_settings()
    await ui.answer({})
    assert theme.appearance == "dark"

    await ui.emit_terminal_color_scheme("light")
    assert theme.appearance == "light"
    assert controller.get_terminal_theme() == "light"


@pytest.mark.tonio
async def test_re_renders_only_when_the_reported_colors_change():
    ui = _FakeUi()
    controller = _create_controller(ui, lambda: SettingsManager.in_memory({"theme": "dark"}))
    await controller.prime()

    async def query(colors: dict) -> None:
        await controller.apply_from_settings()
        await ui.answer(colors)

    await query(DARK)
    # A timeout keeps the known colors; erasing them would count as a change and re-render.
    await query({})
    await query({key: dict(value) for key, value in DARK.items()})
    assert ui.render_requests == 1


@pytest.mark.tonio
async def test_disables_terminal_appearance_updates_when_disposed():
    ui = _FakeUi()
    controller = _create_controller(ui, lambda: SettingsManager.in_memory({"theme": "light/dark"}))
    await controller.prime()
    await controller.apply_from_settings()

    await controller.dispose()

    assert ui.notification_calls[-1] is False
    assert ui.scheme_unsubscribe_calls == 1
    assert ui.colors_unsubscribe_calls == 1


@pytest.mark.tonio
async def test_lets_an_explicit_selection_replace_the_initial_theme():
    ui = _FakeUi()
    first_manager = SettingsManager.in_memory({"theme": "dark"})
    second_manager = SettingsManager.in_memory({"theme": "light"})
    managers = {"current": first_manager}
    controller = _create_controller(ui, lambda: managers["current"], "light")
    await controller.prime()
    await controller.apply_from_settings()

    assert await controller.set_theme_name("dark") == {"success": True}
    managers["current"] = second_manager
    await controller.apply_from_settings()

    assert controller.get_theme_selection() == "dark"
    assert theme.name == "dark"


@pytest.mark.tonio
async def test_reloads_theme_settings_when_no_initial_theme_was_supplied():
    ui = _FakeUi()
    first_manager = SettingsManager.in_memory({"theme": "dark"})
    second_manager = SettingsManager.in_memory({"theme": "light"})
    managers = {"current": first_manager}
    controller = _create_controller(ui, lambda: managers["current"])
    await controller.prime()
    await controller.apply_from_settings()

    first_manager.apply_overrides({"theme": "light"})
    await controller.apply_from_settings()
    assert theme.name == "light"

    second_manager.apply_overrides({"theme": "dark"})
    managers["current"] = second_manager
    await controller.apply_from_settings()
    assert theme.name == "dark"
