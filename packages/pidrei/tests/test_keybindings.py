"""Mirror of pi coding-agent test/keybindings.test.ts.

pi's first case asserts native win32 also gets Windows defaults; pidrei is
POSIX-only and `use_windows_keybindings` has no win32 arm, so that case is
dropped and the `tui.editor.undo` expectation loses its native-Windows
branch. The rest is pi's, including that `WT_SESSION` alone is not WSL.
"""

from pidrei.core.keybindings import KEYBINDINGS, use_windows_keybindings


def test_uses_windows_keybindings_in_wsl_without_relying_on_windows_terminal_detection():
    assert use_windows_keybindings("linux", {"WSL_DISTRO_NAME": "Ubuntu"}) is True
    assert use_windows_keybindings("linux", {"WSL_INTEROP": "/run/WSL/123_interop"}) is True


def test_does_not_use_windows_keybindings_from_wt_session_alone():
    assert use_windows_keybindings("linux", {"WT_SESSION": "session"}) is False


def test_keeps_non_windows_defaults_on_other_platforms():
    assert use_windows_keybindings("linux", {}) is False
    assert use_windows_keybindings("darwin", {}) is False


def test_applies_the_detected_defaults_consistently():
    windows_keybindings = use_windows_keybindings()

    assert KEYBINDINGS["app.clipboard.pasteImage"]["defaultKeys"] == ("alt+v" if windows_keybindings else "ctrl+v")
    assert KEYBINDINGS["tui.altScreen.search"]["defaultKeys"] == ("ctrl+f" if windows_keybindings else "ctrl+shift+f")
    assert KEYBINDINGS["app.message.followUp"]["defaultKeys"] == ("ctrl+q" if windows_keybindings else "alt+enter")
    assert KEYBINDINGS["app.model.cycleBackward"]["defaultKeys"] == ("alt+p" if windows_keybindings else "shift+ctrl+p")
    assert KEYBINDINGS["tui.editor.undo"]["defaultKeys"] == ("alt+z" if windows_keybindings else "ctrl+-")
    assert KEYBINDINGS["tui.altScreen.previousPrompt"]["defaultKeys"] == (
        "ctrl+up" if windows_keybindings else ["ctrl+shift+up", "ctrl+up"]
    )
    assert KEYBINDINGS["tui.altScreen.nextPrompt"]["defaultKeys"] == (
        "ctrl+down" if windows_keybindings else ["ctrl+shift+down", "ctrl+down"]
    )
    assert KEYBINDINGS["app.message.dequeue"]["defaultKeys"] == ("alt+q" if windows_keybindings else "alt+up")
