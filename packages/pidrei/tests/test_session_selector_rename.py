"""Mirror of pi coding-agent test/session-selector-rename.test.ts."""

from datetime import UTC, datetime

import pytest
import tonio.colored as tonio

from pidrei.core.keybindings import KeybindingsManager
from pidrei.core.session_manager import SessionInfo
from pidrei.modes.interactive.components import SessionSelectorComponent
from pidrei.modes.interactive.theme import init_theme
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
        created=overrides.get("created", epoch),
        modified=overrides.get("modified", epoch),
        message_count=overrides.get("message_count", 1),
        first_message=overrides.get("first_message", "hello"),
        all_messages_text=overrides.get("all_messages_text", "hello"),
    )


# Kitty keyboard protocol encoding for Ctrl+R
CTRL_R = "\x1b[114;5u"


def _make_loader(sessions):
    async def loader(on_progress=None):
        return sessions

    return loader


@pytest.fixture(autouse=True)
def _setup():
    init_theme("dark")
    # Ensure test isolation: keybindings are a global singleton
    set_keybindings(KeybindingsManager())


class TestSessionSelectorRename:
    @pytest.mark.tonio
    async def test_shows_rename_hint_in_interactive_resume_picker_configuration(self):
        sessions = [make_session(id="a")]
        keybindings = KeybindingsManager()
        selector = SessionSelectorComponent(
            _make_loader(sessions),
            _make_loader([]),
            lambda path: None,
            lambda: None,
            lambda: None,
            lambda: None,
            {"showRenameHint": True, "keybindings": keybindings},
        )
        await flush_promises()

        output = "\n".join(selector.render(120))
        assert "ctrl+r" in output
        assert "rename" in output

    @pytest.mark.tonio
    async def test_does_not_show_rename_hint_in_resume_picker_configuration(self):
        sessions = [make_session(id="a")]
        keybindings = KeybindingsManager()
        selector = SessionSelectorComponent(
            _make_loader(sessions),
            _make_loader([]),
            lambda path: None,
            lambda: None,
            lambda: None,
            lambda: None,
            {"showRenameHint": False, "keybindings": keybindings},
        )
        await flush_promises()

        output = "\n".join(selector.render(120))
        assert "ctrl+r" not in output
        assert "rename" not in output

    @pytest.mark.tonio
    async def test_enters_rename_mode_on_ctrl_r_and_submits_with_enter(self):
        sessions = [make_session(id="a", name="Old")]
        rename_calls = []

        async def rename_session(session_path, name):
            rename_calls.append((session_path, name))

        keybindings = KeybindingsManager()
        selector = SessionSelectorComponent(
            _make_loader(sessions),
            _make_loader([]),
            lambda path: None,
            lambda: None,
            lambda: None,
            lambda: None,
            {"renameSession": rename_session, "showRenameHint": True, "keybindings": keybindings},
        )
        await flush_promises()

        selector.get_session_list().handle_input(CTRL_R)
        await flush_promises()

        # Rename mode layout
        output = "\n".join(selector.render(120))
        assert "Rename Session" in output
        assert "Resume Session" not in output

        # Type and submit
        selector.handle_input("X")
        selector.handle_input("\r")
        await flush_promises()

        assert rename_calls == [(sessions[0].path, "XOld")]
