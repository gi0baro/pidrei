"""Mirror of pi coding-agent test/session-selector-path-delete.test.ts."""

import os
from datetime import UTC, datetime

import pytest
import tonio.colored as tonio

from pidrei.core.keybindings import KeybindingsManager
from pidrei.core.session_manager import SessionInfo
from pidrei.modes.interactive.components import SessionSelectorComponent
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import set_keybindings


async def flush_promises() -> None:
    await tonio.time.sleep(0.01)


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
    async def load(on_progress=None):
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
def _setup():
    # session selector uses the global theme instance; keybindings are a
    # global singleton
    init_theme_sync("dark")
    set_keybindings(KeybindingsManager())


def _make_selector(current_loader, all_loader, current_session_file_path=None):
    keybindings = KeybindingsManager()
    return SessionSelectorComponent(
        current_loader,
        all_loader,
        lambda path: None,
        lambda: None,
        lambda: None,
        lambda: None,
        {"keybindings": keybindings},
        current_session_file_path,
    )


class TestSessionSelectorPathDeleteInteractions:
    @pytest.mark.tonio
    async def test_does_not_treat_ctrl_backspace_as_delete_when_search_query_is_non_empty(self):
        sessions = [make_session(id="a"), make_session(id="b")]

        selector = _make_selector(_loader(sessions), _loader([]))
        await flush_promises()

        session_list = selector.get_session_list()
        confirmation_changes = []
        session_list.on_delete_confirmation_change = lambda path: confirmation_changes.append(path)

        await session_list.handle_input("a")
        await session_list.handle_input(CTRL_BACKSPACE)

        assert confirmation_changes == []

    @pytest.mark.tonio
    async def test_enters_confirmation_mode_on_ctrl_d_even_with_a_non_empty_search_query(self):
        sessions = [make_session(id="a"), make_session(id="b")]

        selector = _make_selector(_loader(sessions), _loader([]))
        await flush_promises()

        session_list = selector.get_session_list()
        confirmation_changes = []
        session_list.on_delete_confirmation_change = lambda path: confirmation_changes.append(path)

        await session_list.handle_input("a")
        await session_list.handle_input(CTRL_D)

        assert confirmation_changes == [sessions[0].path]

    @pytest.mark.tonio
    async def test_enters_confirmation_mode_on_ctrl_backspace_when_search_query_is_empty(self):
        sessions = [make_session(id="a"), make_session(id="b")]

        selector = _make_selector(_loader(sessions), _loader([]))
        await flush_promises()

        session_list = selector.get_session_list()
        confirmation_changes = []
        session_list.on_delete_confirmation_change = lambda path: confirmation_changes.append(path)

        deleted_path = None

        async def on_delete_session(session_path):
            nonlocal deleted_path
            deleted_path = session_path

        session_list.on_delete_session = on_delete_session

        await session_list.handle_input(CTRL_BACKSPACE)
        assert confirmation_changes == [sessions[0].path]

        await session_list.handle_input("\r")
        await flush_promises()
        assert confirmation_changes == [sessions[0].path, None]
        assert deleted_path == sessions[0].path

    @pytest.mark.tonio
    async def test_does_not_switch_scope_back_to_all_when_all_load_resolves_after_toggling_back_to_current(self):
        current_sessions = [make_session(id="current")]
        all_ready = tonio.Event()
        all_load_calls = 0

        async def all_loader(on_progress=None):
            nonlocal all_load_calls
            all_load_calls += 1
            await all_ready.wait(None)
            return [make_session(id="all")]

        selector = _make_selector(_loader(current_sessions), all_loader)
        await flush_promises()

        session_list = selector.get_session_list()
        await session_list.handle_input("\t")  # current -> all (starts async load)
        await session_list.handle_input("\t")  # all -> current

        all_ready.set()
        await flush_promises()

        assert all_load_calls == 1
        output = "\n".join(selector.render(120))
        assert "Resume Session (Current Folder)" in output
        assert "Resume Session (All)" not in output

    @pytest.mark.tonio
    async def test_does_not_start_redundant_all_loads_when_toggling_scopes_while_all_is_already_loading(self):
        current_sessions = [make_session(id="current")]
        all_ready = tonio.Event()
        all_load_calls = 0

        async def all_loader(on_progress=None):
            nonlocal all_load_calls
            all_load_calls += 1
            await all_ready.wait(None)
            return [make_session(id="all")]

        selector = _make_selector(_loader(current_sessions), all_loader)
        await flush_promises()

        session_list = selector.get_session_list()
        await session_list.handle_input("\t")  # current -> all (starts async load)
        await session_list.handle_input("\t")  # all -> current
        await session_list.handle_input("\t")  # current -> all again while load pending
        await flush_promises()

        assert all_load_calls == 1

        all_ready.set()
        await flush_promises()

    @pytest.mark.tonio
    async def test_threads_sessions_when_parent_and_child_paths_use_different_symlink_aliases(self, tmp_dir):
        paths = create_symlinked_session_paths(tmp_dir)

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

        selector = _make_selector(_loader(sessions), _loader([]))
        await flush_promises()

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

        selector = _make_selector(_loader([parent_one, parent_two, child_two]), _loader([]))
        await flush_promises()

        output = strip_ansi("\n".join(selector.render(120)))
        parent_two_index = output.find("Parent two")
        child_two_index = output.find("└─ Child two")
        parent_one_index = output.find("Parent one")

        assert parent_two_index >= 0
        assert child_two_index > parent_two_index
        assert parent_one_index > child_two_index

    @pytest.mark.tonio
    async def test_treats_the_current_session_as_active_across_symlink_aliases(self, tmp_dir):
        paths = create_symlinked_session_paths(tmp_dir)

        sessions = [make_session(id="parent", path=paths["parentAliasB"], name="Parent")]
        selector = _make_selector(_loader(sessions), _loader([]), paths["parentAliasA"])
        await flush_promises()

        session_list = selector.get_session_list()
        confirmation_changes = []
        error_message = None
        session_list.on_delete_confirmation_change = lambda path: confirmation_changes.append(path)

        def on_error(message):
            nonlocal error_message
            error_message = message

        session_list.on_error = on_error

        await session_list.handle_input(CTRL_D)

        assert confirmation_changes == []
        assert error_message == "Cannot delete the currently active session"
