"""Mirror of pi coding-agent test/session-selector-selection.test.ts.

pi's `flushPromises()` after construction is a wait on the selector's state
(`StateUpdates.until`); pi's synchronous `setSessions` is awaited here (recipe
`session-list-apply-order`).
"""

from datetime import UTC, datetime

import pytest

from pidrei.core.keybindings import KeybindingsManager
from pidrei.core.session_manager import SessionInfo
from pidrei.modes.interactive.components import SessionSelectorComponent
from pidrei.modes.interactive.theme import init_theme
from pidrei_tui import set_keybindings

from .session_selector_helpers import InputCompletions, StateUpdates, lists_sessions


def make_session(id: str, text: str, modified: int) -> SessionInfo:
    return SessionInfo(
        path=f"/tmp/{id}.jsonl",
        id=id,
        cwd="",
        created=datetime.fromtimestamp(0, tz=UTC),
        modified=datetime.fromtimestamp(modified / 1000, tz=UTC),
        message_count=1,
        first_message=text,
        all_messages_text=text,
    )


async def create_selector(sessions: list[SessionInfo], selected: list[str]):
    async def load_current(on_progress=None, cancel=None):
        return sessions

    async def load_all(on_progress=None, cancel=None):
        return []

    state_updates = StateUpdates()
    selector = SessionSelectorComponent(
        load_current,
        load_all,
        selected.append,
        lambda: None,
        lambda: None,
        lambda: None,
        {"keybindings": KeybindingsManager.in_memory()},
        state_lock=state_updates,
        finish_before_next_input=InputCompletions(),
    )
    await state_updates.until(lists_sessions(selector, sessions))
    return selector.get_session_list()


@pytest.fixture(autouse=True)
async def _setup():
    await init_theme("dark")
    set_keybindings(KeybindingsManager.in_memory())


# Regression test for #12245: the /login fix applied to /resume.
@pytest.mark.tonio
async def test_selects_the_best_match_after_typing_a_query_even_after_scrolling():
    sessions = [
        make_session("login", "fix login", 6000),
        make_session("logout", "fix logout", 5000),
        make_session("refactor", "refactor", 4000),
        make_session("docs", "docs", 3000),
        make_session("tests", "tests", 2000),
    ]
    selected: list[str] = []
    session_list = await create_selector(sessions, selected)

    for _ in range(4):
        session_list.handle_input("\x1b[B")
    assert session_list.get_selected_session_path() == "/tmp/tests.jsonl"
    for char in "fix":
        session_list.handle_input(char)
    session_list.handle_input("\r")

    assert selected == ["/tmp/login.jsonl"]


@pytest.mark.tonio
async def test_keeps_the_selection_when_a_cursor_key_leaves_the_query_unchanged():
    sessions = [make_session("login", "fix login", 6000), make_session("logout", "fix logout", 5000)]
    session_list = await create_selector(sessions, [])

    for char in "fix":
        session_list.handle_input(char)
    session_list.handle_input("\x1b[B")
    session_list.handle_input("\x1b[D")  # Left

    assert session_list.get_selected_session_path() == "/tmp/logout.jsonl"


@pytest.mark.tonio
async def test_keeps_the_top_row_selected_when_sessions_arrive_after_typing_a_query():
    sessions = [make_session("login", "fix login", 6000), make_session("docs", "docs", 3000)]
    session_list = await create_selector(sessions, [])

    session_list.handle_input("\x1b[B")
    for char in "fix":
        session_list.handle_input(char)
    await session_list.set_sessions([make_session("newer", "fix newer", 7000), *sessions], False)

    assert session_list.get_selected_session_path() == "/tmp/newer.jsonl"
