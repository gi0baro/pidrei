"""Mirror of pi coding-agent test/theme-controller.test.ts.

pi's constructor calls `initTheme` inline; pidrei defers file-reading theme
initialization to `prime()` (see the controller docstring), so each test
awaits `prime()` where pi asserts right after construction. Env stubbing uses
in-test try/finally (predates tonio 0.9.14; `monkeypatch` works now).
"""

import contextlib
import os

import pytest

from pidrei.core.settings_manager import SettingsManager
from pidrei.modes.interactive.theme import init_theme_sync, theme
from pidrei.modes.interactive.theme.theme_controller import InteractiveThemeController


class _FakeUi:
    def __init__(self, color_scheme: str | None = None):
        self._color_scheme = color_scheme
        self.background_color_queries = 0
        self.color_scheme_queries = 0
        self.notification_calls: list[bool] = []
        self._listener = None

    def invalidate(self) -> None:
        pass

    def request_render(self) -> None:
        pass

    async def set_terminal_color_scheme_notifications(self, enabled: bool) -> None:
        self.notification_calls.append(enabled)

    def on_terminal_color_scheme_change(self, listener):
        self._listener = listener
        return lambda: None

    async def query_terminal_background_color(self, timeout_ms=None):
        self.background_color_queries += 1

    async def query_terminal_color_scheme(self, timeout_ms=None):
        self.color_scheme_queries += 1
        return self._color_scheme

    async def emit_terminal_color_scheme(self, terminal_theme: str) -> None:
        if self._listener is not None:
            await self._listener(terminal_theme)


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


@contextlib.contextmanager
def _env(name: str, value: str):
    original = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


@pytest.fixture(autouse=True)
def _reset_theme(request):
    init_theme_sync("dark")
    request.addfinalizer(lambda: init_theme_sync("dark"))


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

    assert ui.background_color_queries == 0
    assert manager.set_theme_calls == []
    assert manager.flush_calls == 0


@pytest.mark.tonio
async def test_resolves_a_theme_pair_and_follows_terminal_appearance_changes():
    with _env("COLORFGBG", "15;0"):
        ui = _FakeUi(color_scheme="light")
        manager = SettingsManager.in_memory({"theme": "dark/light"})
        controller = _create_controller(ui, lambda: manager, "light/dark")
        await controller.prime()

        assert theme.name == "dark"
        await controller.apply_from_settings()
        assert theme.name == "light"
        assert True in ui.notification_calls

        await ui.emit_terminal_color_scheme("dark")
        assert theme.name == "dark"


@pytest.mark.tonio
async def test_detects_the_current_terminal_appearance_when_selecting_a_theme_pair():
    with _env("COLORFGBG", ""):
        ui = _FakeUi(color_scheme="light")
        manager = SettingsManager.in_memory({"theme": "dark"})
        controller = _create_controller(ui, lambda: manager)
        await controller.prime()

        assert theme.name == "dark"
        await controller.set_theme_setting("light/dark")
        assert theme.name == "light"
        assert ui.color_scheme_queries == 1


@pytest.mark.tonio
async def test_lets_an_explicit_selection_replace_the_initial_theme():
    ui = _FakeUi()
    first_manager = SettingsManager.in_memory({"theme": "dark"})
    second_manager = SettingsManager.in_memory({"theme": "light"})
    managers = {"current": first_manager}
    controller = _create_controller(ui, lambda: managers["current"], "light")
    await controller.prime()
    await controller.apply_from_settings()

    result = await controller.set_theme_name("dark")
    assert result["success"] is True
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
