"""Mirror of pi's config-selector-keybindings.test.ts.

pi's `vi.mock` of the TUI, the config selector component and the theme
becomes `monkeypatch` of the names `pidrei.cli.config_selector` imports. The
terminal attach and the capability probe are stubbed too: the test has no tty.
The global keybindings have no conftest guard, so the test restores them.
"""

import json

import pytest

from pidrei.cli import config_selector
from pidrei_tui import get_keybindings, set_keybindings


class _FakeTerminal:
    rows = 24


class _FakeTui:
    def __init__(self, *_args) -> None:
        self.terminal = _FakeTerminal()

    def set_clear_on_shrink(self, _value) -> None:
        pass

    def add_child(self, _component) -> None:
        pass

    def set_focus(self, _component) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def request_render(self) -> None:
        pass

    def finish_before_next_input(self, _handle) -> None:
        pass


class _FakeConfigSelector:
    def __init__(self, *args, **_kwargs) -> None:
        on_close = args[4]
        on_close()

    def get_resource_list(self) -> dict:
        return {}


class _FakeSettingsManager:
    def get_theme(self):
        return None

    def get_show_hardware_cursor(self) -> bool:
        return False

    def get_clear_on_shrink(self) -> bool:
        return False


async def _noop(*_args, **_kwargs) -> None:
    return None


@pytest.mark.tonio
async def test_uses_the_users_keybindings_json(tmp_path, monkeypatch):
    agent_dir = str(tmp_path)
    (tmp_path / "keybindings.json").write_text(
        json.dumps({"tui.select.up": ["up", "ctrl+p"], "tui.select.down": ["down", "ctrl+n"]}), encoding="utf-8"
    )
    monkeypatch.setattr(config_selector, "ProcessTerminal", lambda: object())
    monkeypatch.setattr(config_selector, "TuiMainScreen", _FakeTui)
    monkeypatch.setattr(config_selector, "ConfigSelectorComponent", _FakeConfigSelector)
    monkeypatch.setattr(config_selector, "build_canonical_path_map_blocking", lambda _paths: {})
    monkeypatch.setattr(config_selector, "init_theme", _noop)
    monkeypatch.setattr(config_selector, "stop_theme_watcher", lambda: None)
    monkeypatch.setattr(config_selector, "prime_capabilities", _noop)
    monkeypatch.setattr(config_selector, "attach_terminal", _noop)
    monkeypatch.setattr(config_selector, "detach_terminal", lambda: None)
    previous = get_keybindings()

    try:
        await config_selector.select_config(
            resolved_paths={},
            settings_manager=_FakeSettingsManager(),
            cwd=agent_dir,
            agent_dir=agent_dir,
            write_scope="global",
            project_mode_available=False,
        )

        assert "ctrl+n" in get_keybindings().get_keys("tui.select.down")
        assert "ctrl+p" in get_keybindings().get_keys("tui.select.up")
    finally:
        set_keybindings(previous)
