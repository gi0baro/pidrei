"""Mirror of pi coding-agent test/thinking-selector.test.ts, plus wiring tests
for `InteractiveMode._show_thinking_selector` (pidrei-only).

The wiring tests drive the real factory the way
`test_6949_unavailable_scoped_model` does — the unbound method on a stub
context, selector captured by a stubbed `_show_selector` — so every callback
the factory wires is exercised through `handle_input`. A selection's work is
spawned and waited for, as the input consumer waits for it before the next key.
"""

from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive.components.thinking_selector import ThinkingSelectorComponent
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import set_keybindings


ENTER = "\r"
ESC = "\x1b"
CTRL_S = "\x13"

LEVELS = ["off", "low", "medium", "high"]


@pytest.fixture(autouse=True)
def _setup():
    init_theme_sync("dark")
    # Keybindings are a global singleton; reset per test.
    set_keybindings(KeybindingsManager())


def create_interactive_context():
    """The attribute surface `_show_thinking_selector` touches, as a stub."""
    holder = {"selector": None}
    done_calls = {"count": 0}
    render_calls = {"count": 0}
    select_calls = []
    pending_input_work = []

    def show_selector(factory):
        def done():
            done_calls["count"] += 1

        holder["selector"] = factory(done)["component"]

    async def select_thinking_level(level: str, persist: bool, *, on_owner: bool = False) -> None:
        select_calls.append((level, persist))

    def request_render():
        render_calls["count"] += 1

    context = SimpleNamespace(
        session=SimpleNamespace(
            thinking_level="medium",
            get_available_thinking_levels=lambda: list(LEVELS),
        ),
        settings_manager=SimpleNamespace(get_default_thinking_level=lambda: "medium"),
        _select_thinking_level=select_thinking_level,
        _show_selector=show_selector,
        _finish_before_next_input=lambda work: pending_input_work.append(tonio.spawn(work)),
        ui=SimpleNamespace(request_render=request_render),
    )
    return context, holder, done_calls, render_calls, select_calls, pending_input_work


async def press(selector, data: str, pending_input_work: list) -> None:
    selector.handle_input(data)
    while pending_input_work:
        await pending_input_work.pop(0)


@pytest.mark.tonio
async def test_keeps_the_current_thinking_level_marked_while_browsing():
    selector = ThinkingSelectorComponent("medium", ["medium", "high"], lambda *args: None, lambda *args: None)

    def get_level_row(level: str) -> str | None:
        return next(
            (line for line in (strip_ansi(row) for row in selector.get_select_list().render(80)) if level in line),
            None,
        )

    assert selector.get_select_list().get_selected_item()["label"] == "✓ medium"
    assert get_level_row("medium").startswith("→ ✓ medium")
    selector.handle_input("\x1b[B")
    assert get_level_row("medium").startswith("  ✓ medium")
    assert get_level_row("high").startswith("→   high")


@pytest.mark.tonio
async def test_uses_the_configured_save_binding():
    set_keybindings(KeybindingsManager({"app.thinking.save": "ctrl+r"}))
    save_default_calls = []

    def save_default(level: str) -> None:
        save_default_calls.append(level)

    selector = ThinkingSelectorComponent(
        "medium",
        ["medium", "high"],
        lambda *args: None,
        lambda *args: None,
        save_default,
    )

    assert "Ctrl+R to set as default" in strip_ansi("\n".join(selector.render(80)))
    selector.handle_input("\x13")
    assert save_default_calls == []
    selector.handle_input("\x12")
    assert save_default_calls == ["medium"]


class TestThinkingSelectorWiring:
    @pytest.mark.tonio
    async def test_escape_cancels_without_selecting(self):
        context, holder, done_calls, render_calls, select_calls, pending_input_work = create_interactive_context()
        InteractiveMode._show_thinking_selector(context)
        selector = holder["selector"]
        assert selector is not None, "Expected thinking selector to open"

        await press(selector, ESC, pending_input_work)

        assert done_calls["count"] == 1
        assert render_calls["count"] == 1
        assert select_calls == []

    @pytest.mark.tonio
    async def test_enter_selects_the_level_session_only(self):
        context, holder, done_calls, _render_calls, select_calls, pending_input_work = create_interactive_context()
        InteractiveMode._show_thinking_selector(context)

        await press(holder["selector"], ENTER, pending_input_work)

        assert select_calls == [("medium", False)]
        assert done_calls["count"] == 1

    @pytest.mark.tonio
    async def test_ctrl_s_selects_the_level_as_default(self):
        context, holder, done_calls, _render_calls, select_calls, pending_input_work = create_interactive_context()
        InteractiveMode._show_thinking_selector(context)

        await press(holder["selector"], CTRL_S, pending_input_work)

        assert select_calls == [("medium", True)]
        assert done_calls["count"] == 1
