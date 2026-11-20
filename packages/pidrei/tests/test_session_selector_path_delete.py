"""Mirror of pi coding-agent test/session-selector-path-delete.test.ts.

pi's 10 ms sleep loop waiting for the deleted file to go is a wait on the
selector's state: the reload has finished and every list update applied.
`test_keeps_the_next_session_selected_when_the_cached_list_applies_after_a_partial_reload_list`
is pidrei-only (recipe `session-list-apply-order`).
"""

import os
import threading
from datetime import UTC, datetime

import pytest
import tonio.colored as tonio
from tonio.colored import fs

from pidrei.core.keybindings import KeybindingsManager
from pidrei.core.session_manager import SessionInfo
from pidrei.modes.interactive.components import SessionSelectorComponent, session_selector as session_selector_module
from pidrei.modes.interactive.theme import init_theme
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import set_keybindings

from .session_selector_helpers import InputCompletions, StateUpdates, every_list_applied, lists_sessions


def make_session(*, id, **overrides):
    epoch = datetime.fromtimestamp(0, tz=UTC)
    return SessionInfo(
        path=overrides.get("path", f"/tmp/{id}.jsonl"),
        id=id,
        cwd=overrides.get("cwd", ""),
        name=overrides.get("name"),
        parent_session_path=overrides.get("parent_session_path"),
        created=overrides.get("created", epoch),
        modified=overrides.get("modified", epoch),
        message_count=overrides.get("message_count", 1),
        first_message=overrides.get("first_message", "hello"),
        all_messages_text=overrides.get("all_messages_text", "hello"),
    )


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _loader(sessions):
    async def load(on_progress=None, cancel=None):
        return sessions

    return load


def create_symlinked_session_paths(base_dir) -> dict:
    real_dir = base_dir / "real"
    alias_a_dir = base_dir / "alias-a"
    alias_b_dir = base_dir / "alias-b"
    real_dir.mkdir(parents=True)
    alias_a_dir.mkdir(parents=True)
    alias_b_dir.mkdir(parents=True)

    shared_dir = real_dir / "sessions"
    shared_dir.mkdir(parents=True)
    alias_a_sessions = alias_a_dir / "sessions"
    alias_b_sessions = alias_b_dir / "sessions"
    os.symlink(shared_dir, alias_a_sessions)
    os.symlink(shared_dir, alias_b_sessions)

    (shared_dir / "parent.jsonl").write_text("parent\n")
    (shared_dir / "child.jsonl").write_text("child\n")

    return {
        "parentAliasA": str(alias_a_sessions / "parent.jsonl"),
        "parentAliasB": str(alias_b_sessions / "parent.jsonl"),
        "childAliasB": str(alias_b_sessions / "child.jsonl"),
    }


CTRL_D = "\x04"
CTRL_BACKSPACE = "\x1b[127;5u"


@pytest.fixture(autouse=True)
async def _setup():
    # session selector uses the global theme instance; keybindings are a
    # global singleton
    await init_theme("dark")
    set_keybindings(KeybindingsManager.in_memory())


def _make_selector(current_loader, all_loader, current_session_file_path=None, completions=None):
    """pi's `await flushPromises()` after an async step becomes `await
    state_updates.until(<the state that step produces>)`: see `StateUpdates`."""
    keybindings = KeybindingsManager.in_memory()
    state_updates = StateUpdates()
    selector = SessionSelectorComponent(
        current_loader,
        all_loader,
        lambda path: None,
        lambda: None,
        lambda: None,
        lambda: None,
        {"keybindings": keybindings},
        current_session_file_path,
        state_lock=state_updates,
        finish_before_next_input=completions if completions is not None else InputCompletions(),
    )
    return selector, state_updates


class TestSessionSelectorPathDeleteInteractions:
    @pytest.mark.tonio
    async def test_does_not_treat_ctrl_backspace_as_delete_when_search_query_is_non_empty(self):
        sessions = [make_session(id="a"), make_session(id="b")]

        selector, state_updates = _make_selector(_loader(sessions), _loader([]))
        await state_updates.until(lists_sessions(selector, sessions))

        session_list = selector.get_session_list()
        confirmation_changes = []
        session_list.on_delete_confirmation_change = lambda path: confirmation_changes.append(path)

        session_list.handle_input("a")
        session_list.handle_input(CTRL_BACKSPACE)

        assert confirmation_changes == []

    @pytest.mark.tonio
    async def test_enters_confirmation_mode_on_ctrl_d_even_with_a_non_empty_search_query(self):
        sessions = [make_session(id="a"), make_session(id="b")]

        selector, state_updates = _make_selector(_loader(sessions), _loader([]))
        await state_updates.until(lists_sessions(selector, sessions))

        session_list = selector.get_session_list()
        confirmation_changes = []
        session_list.on_delete_confirmation_change = lambda path: confirmation_changes.append(path)

        session_list.handle_input("a")
        session_list.handle_input(CTRL_D)

        assert confirmation_changes == [sessions[0].path]

    @pytest.mark.tonio
    async def test_enters_confirmation_mode_on_ctrl_backspace_when_search_query_is_empty(self):
        sessions = [make_session(id="a"), make_session(id="b")]

        selector, state_updates = _make_selector(_loader(sessions), _loader([]))
        await state_updates.until(lists_sessions(selector, sessions))

        session_list = selector.get_session_list()
        confirmation_changes = []
        session_list.on_delete_confirmation_change = lambda path: confirmation_changes.append(path)

        deleted_path = None
        deleted = tonio.Event()

        async def on_delete_session(session_path):
            nonlocal deleted_path
            deleted_path = session_path
            deleted.set()

        session_list.on_delete_session = on_delete_session

        session_list.handle_input(CTRL_BACKSPACE)
        assert confirmation_changes == [sessions[0].path]

        session_list.handle_input("\r")
        # The delete handler runs on a detached task.
        await deleted.wait(5)
        assert confirmation_changes == [sessions[0].path, None]
        assert deleted_path == sessions[0].path

    @pytest.mark.tonio
    async def test_does_not_switch_scope_back_to_all_when_all_load_resolves_after_toggling_back_to_current(self):
        current_sessions = [make_session(id="current")]
        all_ready = tonio.Event()
        all_load_calls = 0

        async def all_loader(on_progress=None, cancel=None):
            nonlocal all_load_calls
            all_load_calls += 1
            await all_ready.wait(None)
            return [make_session(id="all")]

        completions = InputCompletions()
        selector, state_updates = _make_selector(_loader(current_sessions), all_loader, completions=completions)
        await state_updates.until(lists_sessions(selector, current_sessions))

        session_list = selector.get_session_list()
        await completions.press(session_list, "\t")  # current -> all (starts async load)
        await completions.press(session_list, "\t")  # all -> current

        all_ready.set()
        # The "all" load has finished and been applied.
        await state_updates.until(lambda: selector._all_load is None)

        assert all_load_calls == 1
        output = "\n".join(selector.render(120))
        assert "Resume Session (Current Folder)" in output
        assert "Resume Session (All)" not in output

    @pytest.mark.tonio
    async def test_does_not_start_redundant_all_loads_when_toggling_scopes_while_all_is_already_loading(self):
        current_sessions = [make_session(id="current")]
        all_sessions = [make_session(id="all")]
        all_ready = tonio.Event()
        all_load_calls = 0

        async def all_loader(on_progress=None, cancel=None):
            nonlocal all_load_calls
            all_load_calls += 1
            if on_progress is not None:
                on_progress(1, 2, all_sessions)
            await all_ready.wait(None)
            return all_sessions

        completions = InputCompletions()
        selector, state_updates = _make_selector(_loader(current_sessions), all_loader, completions=completions)
        await state_updates.until(lists_sessions(selector, current_sessions))

        session_list = selector.get_session_list()
        await completions.press(session_list, "\t")  # current -> all (starts async load)
        await completions.press(session_list, "\t")  # all -> current
        await completions.press(session_list, "\t")  # current -> all again while load pending
        # The loader's progress (partial "all" sessions) is shown, before or after
        # the third toggle depending on when the detached load reached it.
        await state_updates.until(lists_sessions(selector, all_sessions))

        assert all_load_calls == 1
        assert selector.get_session_list().get_selected_session_path() == all_sessions[0].path
        assert "Loading" in "\n".join(selector.render(120))

        all_ready.set()
        await state_updates.until(lambda: selector._all_load is None)

    @pytest.mark.tonio
    async def test_threads_sessions_when_parent_and_child_paths_use_different_symlink_aliases(self, tmp_path):
        paths = create_symlinked_session_paths(tmp_path)

        sessions = [
            make_session(
                id="parent",
                path=paths["parentAliasB"],
                name="Parent",
                modified=_dt("2026-01-01T00:00:00.000Z"),
            ),
            make_session(
                id="child",
                path=paths["childAliasB"],
                parent_session_path=paths["parentAliasA"],
                name="Child",
                modified=_dt("2025-12-31T00:00:00.000Z"),
            ),
        ]

        selector, state_updates = _make_selector(_loader(sessions), _loader([]))
        await state_updates.until(lists_sessions(selector, sessions))

        output = strip_ansi("\n".join(selector.render(120)))
        assert "Parent" in output
        assert "└─ Child" in output

    @pytest.mark.tonio
    async def test_sorts_threaded_sessions_by_latest_activity_in_their_subtree(self):
        parent_one = make_session(id="parent-one", name="Parent one", modified=_dt("2026-01-02T00:00:00.000Z"))
        parent_two = make_session(id="parent-two", name="Parent two", modified=_dt("2026-01-01T00:00:00.000Z"))
        child_two = make_session(
            id="child-two",
            name="Child two",
            parent_session_path=parent_two.path,
            modified=_dt("2026-01-03T00:00:00.000Z"),
        )

        sessions = [parent_one, parent_two, child_two]
        selector, state_updates = _make_selector(_loader(sessions), _loader([]))
        await state_updates.until(lists_sessions(selector, sessions))

        output = strip_ansi("\n".join(selector.render(120)))
        parent_two_index = output.find("Parent two")
        child_two_index = output.find("└─ Child two")
        parent_one_index = output.find("Parent one")

        assert parent_two_index >= 0
        assert child_two_index > parent_two_index
        assert parent_one_index > child_two_index

    @pytest.mark.tonio
    async def test_treats_the_current_session_as_active_across_symlink_aliases(self, tmp_path):
        paths = create_symlinked_session_paths(tmp_path)

        sessions = [make_session(id="parent", path=paths["parentAliasB"], name="Parent")]
        selector, state_updates = _make_selector(_loader(sessions), _loader([]), paths["parentAliasA"])
        await state_updates.until(lists_sessions(selector, sessions))

        session_list = selector.get_session_list()
        confirmation_changes = []
        error_message = None
        session_list.on_delete_confirmation_change = lambda path: confirmation_changes.append(path)

        def on_error(message):
            nonlocal error_message
            error_message = message

        session_list.on_error = on_error

        session_list.handle_input(CTRL_D)

        assert confirmation_changes == []
        assert error_message == "Cannot delete the currently active session"

    @pytest.mark.tonio
    async def test_keeps_the_next_session_selected_when_a_progressive_reload_after_delete_starts_with_another_session(
        self, tmp_path
    ):
        sessions = _session_files(tmp_path)

        async def current_loader(on_progress=None, cancel=None):
            remaining = [session for session in sessions if await fs.Path(session.path).exists()]
            # The first partial list holds only the file that finished loading first, here the oldest
            if on_progress is not None:
                on_progress(1, len(remaining), [remaining[-1]])
                on_progress(len(remaining), len(remaining), remaining)
            return remaining

        selector, state_updates = _make_selector(current_loader, _loader([]))
        await state_updates.until(lambda: selector._current_load is None and lists_sessions(selector, sessions)())
        await every_list_applied(selector)

        session_list = selector.get_session_list()
        session_list.handle_input("\x1b[B")
        session_list.handle_input("\x1b[A")
        assert session_list.get_selected_session_path() == sessions[0].path

        session_list.handle_input(CTRL_D)
        session_list.handle_input("\r")
        # The delete and the reload run on detached coroutines.
        await state_updates.until(lambda: selector._current_load is None and lists_sessions(selector, sessions[1:])())
        await every_list_applied(selector)

        assert not await fs.Path(sessions[0].path).exists()
        assert session_list.get_selected_session_path() == sessions[1].path

    @pytest.mark.tonio
    async def test_keeps_the_next_session_selected_when_the_cached_list_applies_after_a_partial_reload_list(
        self, tmp_path, monkeypatch
    ):
        # pidrei-only. After a delete, the cached list (complete) is shown and
        # the reload starts in the same hold; the reload's first partial list
        # is registered while the cached list's canonical-path map is still
        # being built. The lists still apply in call order: the cached one
        # moves the selection to the next session, and the reload keeps it.
        sessions = _session_files(tmp_path)
        armed = threading.Event()
        cached_map_entered = threading.Event()
        release_cached_map = threading.Event()
        partial_map_built = threading.Event()
        build_map = session_selector_module.build_canonical_path_map_blocking

        def gated_build_map_blocking(listed, *paths):
            # The first build once armed is the cached list's: the reload's
            # lists are registered only after it has entered.
            if armed.is_set() and not cached_map_entered.is_set():
                cached_map_entered.set()
                release_cached_map.wait(5)
            canonical = build_map(listed, *paths)
            if len(listed) == 1:
                partial_map_built.set()
            return canonical

        monkeypatch.setattr(session_selector_module, "build_canonical_path_map_blocking", gated_build_map_blocking)

        async def current_loader(on_progress=None, cancel=None):
            remaining = [session for session in sessions if await fs.Path(session.path).exists()]
            if armed.is_set() and on_progress is not None:
                await tonio.spawn_blocking(cached_map_entered.wait, 5)
                on_progress(1, len(remaining), [remaining[-1]])
                # The partial list's map is built while the cached list's is not.
                await tonio.spawn_blocking(partial_map_built.wait, 5)
                release_cached_map.set()
                await state_updates.until(lists_sessions(selector, [remaining[-1]]))
                on_progress(len(remaining), len(remaining), remaining)
            return remaining

        selector, state_updates = _make_selector(current_loader, _loader([]))
        await state_updates.until(lambda: selector._current_load is None and lists_sessions(selector, sessions)())
        await every_list_applied(selector)

        session_list = selector.get_session_list()
        session_list.handle_input("\x1b[B")
        session_list.handle_input("\x1b[B")
        assert session_list.get_selected_session_path() == sessions[2].path

        armed.set()
        session_list.handle_input(CTRL_D)
        session_list.handle_input("\r")
        remaining = [sessions[0], sessions[1], sessions[3], sessions[4]]
        await state_updates.until(lambda: selector._current_load is None and lists_sessions(selector, remaining)())
        await every_list_applied(selector)

        assert cached_map_entered.is_set()
        assert partial_map_built.is_set()
        assert session_list.get_selected_session_path() == sessions[3].path


def _session_files(directory) -> list:
    """Five sessions on real files, newest first."""
    sessions = []
    for i in range(1, 6):
        path = directory / f"s{i}.jsonl"
        path.write_text("session\n")
        modified = datetime.fromtimestamp((10_000 - i * 1000) / 1000, tz=UTC)
        sessions.append(make_session(id=f"s{i}", path=str(path), modified=modified))
    return sessions
