"""pidrei-specific: `pidrei config` closes only once the terminal is restored.

pi's `finish` stops the UI in place, then resolves. Here the stop is the
closing key's completion (UI_ISLAND_DESIGN §4.4): `select_config` must not
return while the terminal is still being stopped.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.cli import config_selector as config_selector_module


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tui" / "tests"))
from virtual_terminal import VirtualTerminal


class _GatedTerminal(VirtualTerminal):
    """Its stop parks until `unblock`."""

    def __init__(self) -> None:
        super().__init__(40, 8)
        self.started = tonio.Event()
        self.stopping = tonio.Event()
        self.unblock = tonio.Event()

    async def start(self, on_input, on_resize, on_reply=None, on_error=None) -> None:
        await super().start(on_input, on_resize, on_reply, on_error)
        self.started.set()

    async def stop(self) -> None:
        self.stopping.set()
        await self.unblock.wait(5)
        await super().stop()


class _ResourceList:
    def __init__(self, finish) -> None:
        self._finish = finish

    def render(self, width):
        return ["resources"]

    def invalidate(self):
        pass

    def handle_input(self, data):
        if data == "q":
            self._finish()


class _ConfigSelector:
    """Stands in for `ConfigSelectorComponent`: "q" closes the selector."""

    def __init__(self, _paths, _settings, _cwd, _agent_dir, finish, _exit_now, *_rest, **_kwargs) -> None:
        self._resource_list = _ResourceList(finish)

    def render(self, width):
        return ["config"]

    def invalidate(self):
        pass

    def get_resource_list(self):
        return self._resource_list


async def _no_theme(*_args) -> None:
    return None


@pytest.mark.tonio
async def test_select_config_returns_only_once_the_terminal_is_stopped(monkeypatch):
    terminal = _GatedTerminal()
    monkeypatch.setattr(config_selector_module, "ProcessTerminal", lambda: terminal)
    monkeypatch.setattr(config_selector_module, "ConfigSelectorComponent", _ConfigSelector)
    monkeypatch.setattr(config_selector_module, "init_theme", _no_theme)
    monkeypatch.setattr(config_selector_module, "stop_theme_watcher", lambda: None)
    settings_manager = SimpleNamespace(
        get_theme=lambda: "dark",
        get_show_hardware_cursor=lambda: False,
        get_clear_on_shrink=lambda: False,
    )
    returned = tonio.Event()

    async def run() -> None:
        await config_selector_module.select_config(
            resolved_paths={}, settings_manager=settings_manager, cwd="/", agent_dir="/tmp"
        )
        returned.set()

    async with tonio.scope() as scope:
        scope.spawn(run())
        await terminal.started.wait(5)
        assert terminal.started.is_set()
        scope.spawn(terminal.send_input("q"))
        await terminal.stopping.wait(5)
        assert terminal.stopping.is_set()
        # Bounded: `select_config` must still be waiting on the stop.
        await returned.wait(0.1)
        assert not returned.is_set(), "select_config returned before the terminal was restored"
        terminal.unblock.set()
        await returned.wait(5)
        assert returned.is_set()
