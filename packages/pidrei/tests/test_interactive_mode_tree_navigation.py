"""Mirror of pi coding-agent test/interactive-mode-tree-navigation.test.ts.

pi calls `showTreeSelector` on a fake `this` and awaits the tree list's
`onSelect`. pidrei's `on_select` is sync (the list calls it from key
handling) and spawns the selection flow, so the fake records when the flow
reaches its end — the busy error, or the "Navigated" status — on an Event.
Driving `get_tree_list().on_select` also pins that `_show_tree_selector`
hands its callbacks to the component in pi's constructor order.
"""

from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.core.keybindings import KeybindingsManager
from pidrei.core.session_manager import SessionManager
from pidrei.core.settings_manager import SettingsManager
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei_tui import Container, set_keybindings

from .coding_session_helpers import assistant_msg, user_msg


BUSY_MESSAGE = "Wait for the current compaction or tree navigation to finish before navigating the session tree."


@pytest.fixture(autouse=True)
def _setup():
    init_theme_sync("dark")
    set_keybindings(KeybindingsManager())


class _Recorder:
    """A call recorder whose calls may run an async side effect."""

    def __init__(self, effect=None):
        self.calls: list[tuple] = []
        self.effect = effect

    async def __call__(self, *args):
        self.calls.append(args)
        if self.effect is not None:
            return await self.effect(*args)
        return None


async def _create_tree_ui():
    session_manager = SessionManager.in_memory()
    target_id = await session_manager.append_message(user_msg("first"))
    await session_manager.append_message(assistant_msg("reply"))
    finished = tonio.Event()
    selectors: list = []
    status_calls: list[str] = []
    error_calls: list[str] = []
    indicator_calls: list = []
    clear_indicator_calls: list = []
    restore_calls: list = []

    def on_escape():
        pass

    session = SimpleNamespace(is_streaming=False, is_compacting=False)

    async def abort():
        session.is_streaming = False

    async def navigate_tree(*_args):
        if session.is_compacting:
            raise RuntimeError(BUSY_MESSAGE)
        return SimpleNamespace(cancelled=False, aborted=False, editor_text=None)

    session.abort = _Recorder(abort)
    session.abort_branch_summary = lambda: None
    session.navigate_tree = _Recorder(navigate_tree)

    async def no_summary(*_args):
        return "No summary"

    def show_status(message: str) -> None:
        status_calls.append(message)
        if message == "Navigated to selected point":
            finished.set()

    def show_error(message: str) -> None:
        error_calls.append(message)
        finished.set()

    async def flush_compaction_queue(*_args):
        return None

    ui = SimpleNamespace(
        session_manager=session_manager,
        settings_manager=SettingsManager.in_memory(),
        session=session,
        _default_editor=SimpleNamespace(on_escape=on_escape),
        editor=SimpleNamespace(get_text=lambda: ""),
        _set_editor_text=lambda _text: None,
        _chat_container=Container(),
        ui=SimpleNamespace(terminal=SimpleNamespace(rows=24), request_render=lambda force=False: None),
        _show_selector=lambda create: selectors.append(create(lambda: None)["component"]),
        _show_extension_selector=_Recorder(no_summary),
        _show_status_indicator=indicator_calls.append,
        _clear_status_indicator=clear_indicator_calls.append,
        _restore_queued_messages_to_editor=lambda: restore_calls.append(True),
        _render_initial_messages=lambda: None,
        show_status=show_status,
        show_error=show_error,
        _flush_compaction_queue=flush_compaction_queue,
    )
    ui._show_tree_selector = lambda initial_selected_id=None: InteractiveMode._show_tree_selector(
        ui, initial_selected_id
    )
    InteractiveMode._show_tree_selector(ui)

    async def select() -> None:
        assert len(selectors) == 1
        selectors[0].get_tree_list().on_select(target_id)
        await finished.wait(5)
        assert finished.is_set(), "the tree selection flow never finished"

    return SimpleNamespace(
        ui=ui,
        session=session,
        on_escape=on_escape,
        target_id=target_id,
        select=select,
        error_calls=error_calls,
        indicator_calls=indicator_calls,
        clear_indicator_calls=clear_indicator_calls,
        restore_calls=restore_calls,
    )


# Regression for #9178 / PR #9179: rejection must not replace the active operation's UI.
@pytest.mark.tonio
@pytest.mark.parametrize("choice", ["Summarize", "No summary"])
async def test_preserves_operation_ui_when_choosing_while_busy(choice):
    tree = await _create_tree_ui()
    original_leaf_id = tree.ui.session_manager.get_leaf_id()

    async def open_dialog(*_args):
        # Compaction or another navigation can start while the dialog is open.
        tree.session.is_compacting = True
        return choice

    tree.ui._show_extension_selector.effect = open_dialog

    await tree.select()

    assert tree.error_calls == [BUSY_MESSAGE]
    assert tree.indicator_calls == []
    assert tree.clear_indicator_calls == []
    assert tree.ui._default_editor.on_escape is tree.on_escape
    assert tree.session.navigate_tree.calls == []
    assert tree.session.abort.calls == []
    assert tree.ui.session_manager.get_leaf_id() == original_leaf_id


@pytest.mark.tonio
async def test_allows_navigation_when_compaction_finishes_while_the_dialog_is_open():
    tree = await _create_tree_ui()
    tree.session.is_compacting = True

    async def open_dialog(*_args):
        tree.session.is_compacting = False
        return "No summary"

    tree.ui._show_extension_selector.effect = open_dialog

    await tree.select()

    assert tree.session.navigate_tree.calls == [
        (tree.target_id, {"summarize": False, "custom_instructions": None}),
    ]
    assert tree.error_calls == []


@pytest.mark.tonio
async def test_still_aborts_an_active_response_before_navigating():
    tree = await _create_tree_ui()
    tree.session.is_streaming = True

    async def navigate(*_args):
        assert tree.session.is_streaming is False
        assert tree.restore_calls == [True]
        return SimpleNamespace(cancelled=False, aborted=False, editor_text=None)

    tree.session.navigate_tree.effect = navigate

    await tree.select()

    assert len(tree.session.abort.calls) == 1
    assert len(tree.session.navigate_tree.calls) == 1
    assert tree.error_calls == []


@pytest.mark.tonio
async def test_rechecks_availability_after_the_response_abort_settles():
    tree = await _create_tree_ui()
    tree.session.is_streaming = True

    async def summarize(*_args):
        return "Summarize"

    async def abort():
        tree.session.is_streaming = False
        tree.session.is_compacting = True

    tree.ui._show_extension_selector.effect = summarize
    tree.session.abort.effect = abort

    await tree.select()

    assert len(tree.session.abort.calls) == 1
    assert tree.error_calls == [BUSY_MESSAGE]
    assert tree.indicator_calls == []
    assert tree.clear_indicator_calls == []
    assert tree.ui._default_editor.on_escape is tree.on_escape
    assert tree.session.navigate_tree.calls == []
