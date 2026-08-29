"""Mirror of pi tui test/editor.test.ts."""

import re

import pytest
import tonio.colored as tonio

from pidrei_tui.autocomplete import CombinedAutocompleteProvider
from pidrei_tui.components import editor as editor_module
from pidrei_tui.components.editor import Editor, word_wrap_line
from pidrei_tui.tui_main_screen import TuiMainScreen
from pidrei_tui.utils import visible_width

from .themes import default_editor_theme
from .virtual_terminal import VirtualTerminal, poll_until


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(line: str) -> str:
    return _ANSI_RE.sub("", line)


def create_test_tui(cols=80, rows=24):
    """Create a TUI with a virtual terminal for testing."""
    return TuiMainScreen(VirtualTerminal(cols, rows))


def apply_completion(lines, cursor_line, cursor_col, item, prefix):
    """Standard apply_completion that replaces prefix with item value."""
    line = lines[cursor_line] if cursor_line < len(lines) else ""
    before = line[: cursor_col - len(prefix)]
    after = line[cursor_col:]
    new_lines = [*lines]
    new_lines[cursor_line] = before + item["value"] + after
    return {
        "lines": new_lines,
        "cursorLine": cursor_line,
        "cursorCol": cursor_col - len(prefix) + len(item["value"]),
    }


class MockProvider:
    """Duck-typed AutocompleteProvider built from plain functions."""

    def __init__(self, get_suggestions, apply=None, trigger_characters=None):
        self.get_suggestions = get_suggestions
        self.apply_completion = apply if apply is not None else apply_completion
        if trigger_characters is not None:
            self.trigger_characters = trigger_characters


async def flush_autocomplete():
    await tonio.sleep(0.02)


SLOW_DEBOUNCE_MS = 300


def slow_debounce(request) -> None:
    """Widen the autocomplete debounce for the test's lifetime.

    The debounce tests assert that no query ran *between* keystrokes; with the
    real 20 ms window that depends on how fast the runner gets from the last
    `handle_input` to the assertion (a loaded macOS CI runner did not make it).
    """
    original = editor_module.ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS
    editor_module.ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS = SLOW_DEBOUNCE_MS
    request.addfinalizer(lambda: setattr(editor_module, "ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS", original))


async def wait_slow_debounce() -> None:
    await tonio.sleep(SLOW_DEBOUNCE_MS / 1000 + 0.05)


# Prompt history navigation


@pytest.mark.tonio
async def test_does_nothing_on_up_arrow_when_history_is_empty():
    editor = Editor(create_test_tui(), default_editor_theme)

    await editor.handle_input("\x1b[A")  # Up arrow

    assert editor.get_text() == ""


@pytest.mark.tonio
async def test_shows_most_recent_history_entry_on_up_arrow_when_editor_is_empty():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.add_to_history("first prompt")
    editor.add_to_history("second prompt")

    await editor.handle_input("\x1b[A")  # Up arrow

    assert editor.get_text() == "second prompt"


@pytest.mark.tonio
async def test_cycles_through_history_entries_on_repeated_up_arrow():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.add_to_history("first")
    editor.add_to_history("second")
    editor.add_to_history("third")

    await editor.handle_input("\x1b[A")  # Up - shows "third"
    assert editor.get_text() == "third"

    await editor.handle_input("\x1b[A")  # Up - shows "second"
    assert editor.get_text() == "second"

    await editor.handle_input("\x1b[A")  # Up - shows "first"
    assert editor.get_text() == "first"

    await editor.handle_input("\x1b[A")  # Up - stays at "first" (oldest)
    assert editor.get_text() == "first"


@pytest.mark.tonio
async def test_jumps_to_start_before_entering_history_from_a_non_empty_draft():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.add_to_history("prompt")
    editor.set_text("draft")
    await editor.handle_input("\x1b[D")
    await editor.handle_input("\x1b[D")

    await editor.handle_input("\x1b[A")  # Up - jumps to start before history browsing
    assert editor.get_text() == "draft"
    assert editor.get_cursor() == {"line": 0, "col": 0}

    await editor.handle_input("\x1b[A")  # Up at start - shows "prompt"
    assert editor.get_text() == "prompt"

    await editor.handle_input("\x1b[B")  # Down - restores draft
    assert editor.get_text() == "draft"
    assert editor.get_cursor() == {"line": 0, "col": 0}


@pytest.mark.tonio
async def test_navigates_forward_through_history_with_down_arrow():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.add_to_history("first")
    editor.add_to_history("second")
    editor.add_to_history("third")
    editor.set_text("draft")

    # Go to oldest
    await editor.handle_input("\x1b[A")  # start of draft
    await editor.handle_input("\x1b[A")  # third
    await editor.handle_input("\x1b[A")  # second
    await editor.handle_input("\x1b[A")  # first

    # Navigate back
    await editor.handle_input("\x1b[B")  # second
    assert editor.get_text() == "second"

    await editor.handle_input("\x1b[B")  # third
    assert editor.get_text() == "third"

    await editor.handle_input("\x1b[B")  # draft
    assert editor.get_text() == "draft"


@pytest.mark.tonio
async def test_exits_history_mode_when_typing_a_character():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.add_to_history("old prompt")

    await editor.handle_input("\x1b[A")  # Up - shows "old prompt"
    await editor.handle_input("x")  # Type a character - exits history mode

    assert editor.get_text() == "xold prompt"


@pytest.mark.tonio
async def test_exits_history_mode_on_set_text():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.add_to_history("first")
    editor.add_to_history("second")

    await editor.handle_input("\x1b[A")  # Up - shows "second"
    editor.set_text("")  # External clear

    # Up should start fresh from most recent
    await editor.handle_input("\x1b[A")
    assert editor.get_text() == "second"


@pytest.mark.tonio
async def test_does_not_add_empty_strings_to_history():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.add_to_history("")
    editor.add_to_history("   ")
    editor.add_to_history("valid")

    await editor.handle_input("\x1b[A")
    assert editor.get_text() == "valid"

    # Should not have more entries
    await editor.handle_input("\x1b[A")
    assert editor.get_text() == "valid"


@pytest.mark.tonio
async def test_does_not_add_consecutive_duplicates_to_history():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.add_to_history("same")
    editor.add_to_history("same")
    editor.add_to_history("same")

    await editor.handle_input("\x1b[A")  # "same"
    assert editor.get_text() == "same"

    await editor.handle_input("\x1b[A")  # stays at "same" (only one entry)
    assert editor.get_text() == "same"


@pytest.mark.tonio
async def test_allows_non_consecutive_duplicates_in_history():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.add_to_history("first")
    editor.add_to_history("second")
    editor.add_to_history("first")  # Not consecutive, should be added

    await editor.handle_input("\x1b[A")  # "first"
    assert editor.get_text() == "first"

    await editor.handle_input("\x1b[A")  # "second"
    assert editor.get_text() == "second"

    await editor.handle_input("\x1b[A")  # "first" (older one)
    assert editor.get_text() == "first"


@pytest.mark.tonio
async def test_uses_cursor_movement_instead_of_history_when_editor_has_content():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.add_to_history("history item")
    editor.set_text("line1\nline2")

    # Cursor is at end of line2, Up should move to line1
    await editor.handle_input("\x1b[A")  # Up - cursor movement

    # Insert character to verify cursor position
    await editor.handle_input("X")

    # X should be inserted in line1, not replace with history
    assert editor.get_text() == "line1X\nline2"


@pytest.mark.tonio
async def test_limits_history_to_100_entries():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Add 105 entries
    for i in range(105):
        editor.add_to_history(f"prompt {i}")

    # Navigate to oldest
    for _ in range(100):
        await editor.handle_input("\x1b[A")

    # Should be at entry 5 (oldest kept), not entry 0
    assert editor.get_text() == "prompt 5"

    # One more Up should not change anything
    await editor.handle_input("\x1b[A")
    assert editor.get_text() == "prompt 5"


@pytest.mark.tonio
async def test_places_cursor_at_start_after_browsing_history_upward():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.add_to_history("older entry")
    editor.add_to_history("line1\nline2\nline3")

    await editor.handle_input("\x1b[A")  # Up - shows multi-line entry at start
    assert editor.get_text() == "line1\nline2\nline3"
    assert editor.get_cursor() == {"line": 0, "col": 0}

    await editor.handle_input("\x1b[A")  # Up again - immediately navigates to older entry
    assert editor.get_text() == "older entry"
    assert editor.get_cursor() == {"line": 0, "col": 0}


@pytest.mark.tonio
async def test_places_cursor_at_end_after_browsing_history_downward():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.add_to_history("older entry")
    editor.add_to_history("line1\nline2\nline3")
    editor.add_to_history("newer entry")

    await editor.handle_input("\x1b[A")  # newer entry
    await editor.handle_input("\x1b[A")  # multi-line entry
    await editor.handle_input("\x1b[A")  # older entry

    await editor.handle_input("\x1b[B")  # Down - shows multi-line entry at end
    assert editor.get_text() == "line1\nline2\nline3"
    assert editor.get_cursor() == {"line": 2, "col": 5}

    await editor.handle_input("\x1b[B")  # Down again - immediately navigates to newer entry
    assert editor.get_text() == "newer entry"


@pytest.mark.tonio
async def test_allows_opposite_direction_cursor_movement_within_multi_line_history_entry():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.add_to_history("line1\nline2\nline3")

    await editor.handle_input("\x1b[A")  # Up - shows entry at start
    assert editor.get_cursor() == {"line": 0, "col": 0}

    await editor.handle_input("\x1b[B")  # Down - cursor moves to line2
    assert editor.get_text() == "line1\nline2\nline3"
    assert editor.get_cursor() == {"line": 1, "col": 0}

    await editor.handle_input("\x1b[A")  # Up - cursor moves back to line1
    assert editor.get_text() == "line1\nline2\nline3"
    assert editor.get_cursor() == {"line": 0, "col": 0}


# public state accessors


@pytest.mark.tonio
async def test_returns_cursor_position():
    editor = Editor(create_test_tui(), default_editor_theme)

    assert editor.get_cursor() == {"line": 0, "col": 0}

    await editor.handle_input("a")
    await editor.handle_input("b")
    await editor.handle_input("c")

    assert editor.get_cursor() == {"line": 0, "col": 3}

    await editor.handle_input("\x1b[D")  # Left
    assert editor.get_cursor() == {"line": 0, "col": 2}


def test_returns_lines_as_a_defensive_copy():
    editor = Editor(create_test_tui(), default_editor_theme)
    editor.set_text("a\nb")

    lines = editor.get_lines()
    assert lines == ["a", "b"]

    lines[0] = "mutated"
    assert editor.get_lines() == ["a", "b"]


# Backslash+Enter newline workaround


@pytest.mark.tonio
async def test_inserts_backslash_immediately_no_buffering():
    editor = Editor(create_test_tui(), default_editor_theme)

    await editor.handle_input("\\")

    # Backslash should be visible immediately, not buffered
    assert editor.get_text() == "\\"


@pytest.mark.tonio
async def test_converts_standalone_backslash_to_newline_on_enter():
    editor = Editor(create_test_tui(), default_editor_theme)

    await editor.handle_input("\\")
    await editor.handle_input("\r")

    assert editor.get_text() == "\n"


@pytest.mark.tonio
async def test_inserts_backslash_normally_when_followed_by_other_characters():
    editor = Editor(create_test_tui(), default_editor_theme)

    await editor.handle_input("\\")
    await editor.handle_input("x")

    assert editor.get_text() == "\\x"


@pytest.mark.tonio
async def test_does_not_trigger_newline_when_backslash_is_not_immediately_before_cursor():
    editor = Editor(create_test_tui(), default_editor_theme)
    submitted = []

    editor.on_submit = lambda text: submitted.append(text)

    await editor.handle_input("\\")
    await editor.handle_input("x")
    await editor.handle_input("\r")

    # Should submit, not insert newline (backslash not at cursor)
    assert len(submitted) == 1


@pytest.mark.tonio
async def test_only_removes_one_backslash_when_multiple_are_present():
    editor = Editor(create_test_tui(), default_editor_theme)

    await editor.handle_input("\\")
    await editor.handle_input("\\")
    await editor.handle_input("\\")
    assert editor.get_text() == "\\\\\\"

    await editor.handle_input("\r")
    # Only the last backslash is removed, newline inserted
    assert editor.get_text() == "\\\\\n"


# Kitty CSI-u handling


@pytest.mark.tonio
async def test_ignores_printable_csi_u_sequences_with_unsupported_modifiers():
    editor = Editor(create_test_tui(), default_editor_theme)

    await editor.handle_input("\x1b[99;9u")

    assert editor.get_text() == ""


@pytest.mark.tonio
async def test_inserts_shifted_csi_u_letters_as_text():
    editor = Editor(create_test_tui(), default_editor_theme)

    await editor.handle_input("\x1b[69;2u")

    assert editor.get_text() == "E"


@pytest.mark.tonio
async def test_inserts_shifted_xterm_modify_other_keys_letters_as_text():
    editor = Editor(create_test_tui(), default_editor_theme)

    await editor.handle_input("\x1b[27;2;69~")

    assert editor.get_text() == "E"


# Unicode text editing behavior


@pytest.mark.tonio
async def test_inserts_mixed_ascii_umlauts_and_emojis_as_literal_text():
    editor = Editor(create_test_tui(), default_editor_theme)

    await editor.handle_input("H")
    await editor.handle_input("e")
    await editor.handle_input("l")
    await editor.handle_input("l")
    await editor.handle_input("o")
    await editor.handle_input(" ")
    await editor.handle_input("ä")
    await editor.handle_input("ö")
    await editor.handle_input("ü")
    await editor.handle_input(" ")
    await editor.handle_input("😀")

    assert editor.get_text() == "Hello äöü 😀"


@pytest.mark.tonio
async def test_deletes_single_code_unit_unicode_characters_umlauts_with_backspace():
    editor = Editor(create_test_tui(), default_editor_theme)

    await editor.handle_input("ä")
    await editor.handle_input("ö")
    await editor.handle_input("ü")

    # Delete the last character (ü)
    await editor.handle_input("\x7f")  # Backspace

    assert editor.get_text() == "äö"


@pytest.mark.tonio
async def test_deletes_multi_code_unit_emojis_with_single_backspace():
    editor = Editor(create_test_tui(), default_editor_theme)

    await editor.handle_input("😀")
    await editor.handle_input("👍")

    # Delete the last emoji (👍) - single backspace deletes whole grapheme cluster
    await editor.handle_input("\x7f")  # Backspace

    assert editor.get_text() == "😀"


@pytest.mark.tonio
async def test_inserts_characters_at_the_correct_position_after_cursor_movement_over_umlauts():
    editor = Editor(create_test_tui(), default_editor_theme)

    await editor.handle_input("ä")
    await editor.handle_input("ö")
    await editor.handle_input("ü")

    # Move cursor left twice
    await editor.handle_input("\x1b[D")  # Left arrow
    await editor.handle_input("\x1b[D")  # Left arrow

    # Insert 'x' in the middle
    await editor.handle_input("x")

    assert editor.get_text() == "äxöü"


@pytest.mark.tonio
async def test_moves_cursor_across_multi_code_unit_emojis_with_single_arrow_key():
    editor = Editor(create_test_tui(), default_editor_theme)

    await editor.handle_input("😀")
    await editor.handle_input("👍")
    await editor.handle_input("🎉")

    # Move cursor left over last emoji (🎉) - single arrow moves over whole grapheme
    await editor.handle_input("\x1b[D")  # Left arrow

    # Move cursor left over second emoji (👍)
    await editor.handle_input("\x1b[D")

    # Insert 'x' between first and second emoji
    await editor.handle_input("x")

    assert editor.get_text() == "😀x👍🎉"


@pytest.mark.tonio
async def test_preserves_umlauts_across_line_breaks():
    editor = Editor(create_test_tui(), default_editor_theme)

    await editor.handle_input("ä")
    await editor.handle_input("ö")
    await editor.handle_input("ü")
    await editor.handle_input("\n")  # new line
    await editor.handle_input("Ä")
    await editor.handle_input("Ö")
    await editor.handle_input("Ü")

    assert editor.get_text() == "äöü\nÄÖÜ"


def test_replaces_the_entire_document_with_unicode_text_via_set_text_paste_simulation():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Simulate bracketed paste / programmatic replacement
    editor.set_text("Hällö Wörld! 😀 äöüÄÖÜß")

    assert editor.get_text() == "Hällö Wörld! 😀 äöüÄÖÜß"


@pytest.mark.tonio
async def test_moves_cursor_to_document_start_on_ctrl_a_and_inserts_at_the_beginning():
    editor = Editor(create_test_tui(), default_editor_theme)

    await editor.handle_input("a")
    await editor.handle_input("b")
    await editor.handle_input("\x01")  # Ctrl+A (move to start)
    await editor.handle_input("x")  # Insert at start

    assert editor.get_text() == "xab"


@pytest.mark.tonio
async def test_deletes_words_correctly_with_ctrl_w_and_alt_backspace():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Basic word deletion
    editor.set_text("foo bar baz")
    await editor.handle_input("\x17")  # Ctrl+W
    assert editor.get_text() == "foo bar "

    # Trailing whitespace
    editor.set_text("foo bar   ")
    await editor.handle_input("\x17")
    assert editor.get_text() == "foo "

    # Punctuation run
    editor.set_text("foo bar...")
    await editor.handle_input("\x17")
    assert editor.get_text() == "foo bar"

    # ASCII punctuation inside Intl word-like segments preserves old boundaries
    editor.set_text("foo.bar")
    await editor.handle_input("\x17")
    assert editor.get_text() == "foo."

    editor.set_text("foo:bar")
    await editor.handle_input("\x17")
    assert editor.get_text() == "foo:"

    # Delete across multiple lines
    editor.set_text("line one\nline two")
    await editor.handle_input("\x17")
    assert editor.get_text() == "line one\nline "

    # Delete empty line (merge)
    editor.set_text("line one\n")
    await editor.handle_input("\x17")
    assert editor.get_text() == "line one"

    # Grapheme safety (emoji as a word)
    editor.set_text("foo 😀😀 bar")
    await editor.handle_input("\x17")
    assert editor.get_text() == "foo 😀😀 "
    await editor.handle_input("\x17")
    assert editor.get_text() == "foo "

    # Alt+Backspace
    editor.set_text("foo bar")
    await editor.handle_input("\x1b\x7f")  # Alt+Backspace (legacy)
    assert editor.get_text() == "foo "


@pytest.mark.tonio
async def test_navigates_words_correctly_with_ctrl_left_right():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("foo bar... baz")
    # Cursor at end

    # Move left over baz
    await editor.handle_input("\x1b[1;5D")  # Ctrl+Left
    assert editor.get_cursor() == {"line": 0, "col": 11}  # after '...'

    # Move left over punctuation
    await editor.handle_input("\x1b[1;5D")  # Ctrl+Left
    assert editor.get_cursor() == {"line": 0, "col": 7}  # after 'bar'

    # Move left over bar
    await editor.handle_input("\x1b[1;5D")  # Ctrl+Left
    assert editor.get_cursor() == {"line": 0, "col": 4}  # after 'foo '

    # Move right over bar
    await editor.handle_input("\x1b[1;5C")  # Ctrl+Right
    assert editor.get_cursor() == {"line": 0, "col": 7}  # at end of 'bar'

    # Move right over punctuation run
    await editor.handle_input("\x1b[1;5C")  # Ctrl+Right
    assert editor.get_cursor() == {"line": 0, "col": 10}  # after '...'

    # Move right skips space and lands after baz
    await editor.handle_input("\x1b[1;5C")  # Ctrl+Right
    assert editor.get_cursor() == {"line": 0, "col": 14}  # end of line

    # Test forward from start with leading whitespace
    editor.set_text("   foo bar")
    await editor.handle_input("\x01")  # Ctrl+A to go to start
    await editor.handle_input("\x1b[1;5C")  # Ctrl+Right
    assert editor.get_cursor() == {"line": 0, "col": 6}  # after 'foo'

    # ASCII punctuation inside Intl word-like segments preserves old boundaries
    editor.set_text("foo.bar baz")
    await editor.handle_input("\x1b[1;5D")  # Ctrl+Left over baz
    assert editor.get_cursor() == {"line": 0, "col": 8}
    await editor.handle_input("\x1b[1;5D")  # Ctrl+Left over bar
    assert editor.get_cursor() == {"line": 0, "col": 4}
    await editor.handle_input("\x1b[1;5D")  # Ctrl+Left over .
    assert editor.get_cursor() == {"line": 0, "col": 3}

    await editor.handle_input("\x01")  # Ctrl+A
    await editor.handle_input("\x1b[1;5C")  # Ctrl+Right over foo
    assert editor.get_cursor() == {"line": 0, "col": 3}
    await editor.handle_input("\x1b[1;5C")  # Ctrl+Right over .
    assert editor.get_cursor() == {"line": 0, "col": 4}
    await editor.handle_input("\x1b[1;5C")  # Ctrl+Right over bar
    assert editor.get_cursor() == {"line": 0, "col": 7}


@pytest.mark.tonio
async def test_stops_at_fullwidth_chinese_punctuation_issue_4972():
    editor = Editor(create_test_tui(), default_editor_theme)

    # 你好，世界 = 你好(0-2) ，(2-3) 世界(3-5)
    editor.set_text("你好，世界")
    # Cursor at end (col 5)

    # Move left over 世界
    await editor.handle_input("\x1b[1;5D")  # Ctrl+Left
    assert editor.get_cursor() == {"line": 0, "col": 3}  # after ，

    # Move left over ，
    await editor.handle_input("\x1b[1;5D")  # Ctrl+Left
    assert editor.get_cursor() == {"line": 0, "col": 2}  # after 你好

    # Move left over 你好
    await editor.handle_input("\x1b[1;5D")  # Ctrl+Left
    assert editor.get_cursor() == {"line": 0, "col": 0}  # start

    # Move right over 你好
    await editor.handle_input("\x1b[1;5C")  # Ctrl+Right
    assert editor.get_cursor() == {"line": 0, "col": 2}  # after 你好

    # Move right over ，
    await editor.handle_input("\x1b[1;5C")  # Ctrl+Right
    assert editor.get_cursor() == {"line": 0, "col": 3}  # after ，

    # Move right over 世界
    await editor.handle_input("\x1b[1;5C")  # Ctrl+Right
    assert editor.get_cursor() == {"line": 0, "col": 5}  # end


@pytest.mark.tonio
async def test_handles_mixed_cjk_and_ascii_word_movement():
    editor = Editor(create_test_tui(), default_editor_theme)

    # "hello你好，world世界" = hello(0-5) 你好(5-7) ，(7-8) world(8-13) 世界(13-15)
    editor.set_text("hello你好，world世界")
    # Cursor at end (col 15)

    # Move left over 世界
    await editor.handle_input("\x1b[1;5D")  # Ctrl+Left
    assert editor.get_cursor() == {"line": 0, "col": 13}  # after 'world'

    # Move left over world
    await editor.handle_input("\x1b[1;5D")  # Ctrl+Left
    assert editor.get_cursor() == {"line": 0, "col": 8}  # after ，

    # Move left over ，
    await editor.handle_input("\x1b[1;5D")  # Ctrl+Left
    assert editor.get_cursor() == {"line": 0, "col": 7}  # after 你好

    # Move left over 你好
    await editor.handle_input("\x1b[1;5D")  # Ctrl+Left
    assert editor.get_cursor() == {"line": 0, "col": 5}  # after 'hello'

    # Move left over hello
    await editor.handle_input("\x1b[1;5D")  # Ctrl+Left
    assert editor.get_cursor() == {"line": 0, "col": 0}  # start

    # Forward from start
    await editor.handle_input("\x1b[1;5C")  # Ctrl+Right
    assert editor.get_cursor() == {"line": 0, "col": 5}  # after 'hello'

    await editor.handle_input("\x1b[1;5C")  # Ctrl+Right
    assert editor.get_cursor() == {"line": 0, "col": 7}  # after 你好

    await editor.handle_input("\x1b[1;5C")  # Ctrl+Right
    assert editor.get_cursor() == {"line": 0, "col": 8}  # after ，

    await editor.handle_input("\x1b[1;5C")  # Ctrl+Right
    assert editor.get_cursor() == {"line": 0, "col": 13}  # after 'world'

    await editor.handle_input("\x1b[1;5C")  # Ctrl+Right
    assert editor.get_cursor() == {"line": 0, "col": 15}  # end


# Scroll indicators


@pytest.mark.tonio
async def test_keeps_truncated_scroll_indicators_within_width_and_preserves_their_color_issue_6962():
    width = 10
    border_color = lambda text: f"\x1b[35m{text}\x1b[39m"
    editor = Editor(create_test_tui(width), {**default_editor_theme, "borderColor": border_color})
    editor.set_text("\n".join(f"line {index}" for index in range(20)))

    # Render once to initialize wrapping, then move the cursor so content remains above and below the viewport.
    editor.render(width)
    for _ in range(10):
        await editor.handle_input("\x1b[A")

    lines = editor.render(width)
    top_border = lines[0]
    bottom_border = lines[-1]

    assert re.match(r"^─── ↑", strip_ansi(top_border))
    assert re.match(r"^─── ↓", strip_ansi(bottom_border))
    assert top_border == border_color(strip_ansi(top_border))
    assert bottom_border == border_color(strip_ansi(bottom_border))
    for line in lines:
        assert visible_width(line) == width, f"line exceeds width {width}: {line!r}"


# Grapheme-aware text wrapping


def test_wraps_lines_correctly_when_text_contains_wide_emojis():
    editor = Editor(create_test_tui(), default_editor_theme)
    width = 20

    # ✅ is 2 columns wide, so "Hello ✅ World" is 14 columns
    editor.set_text("Hello ✅ World")
    lines = editor.render(width)

    # All content lines (between borders) should fit within width
    for i in range(1, len(lines) - 1):
        line_width = visible_width(lines[i])
        assert line_width == width, f"Line {i} has width {line_width}, expected {width}"


def test_wraps_long_text_with_emojis_at_correct_positions():
    editor = Editor(create_test_tui(), default_editor_theme)
    width = 10

    # Each ✅ is 2 columns. "✅✅✅✅✅" = 10 columns, fits exactly
    # "✅✅✅✅✅✅" = 12 columns, needs wrap
    editor.set_text("✅✅✅✅✅✅")
    lines = editor.render(width)

    # Should have 2 content lines (plus 2 border lines)
    # First line: 5 emojis (10 cols), second line: 1 emoji (2 cols) + padding
    for i in range(1, len(lines) - 1):
        line_width = visible_width(lines[i])
        assert line_width == width, f"Line {i} has width {line_width}, expected {width}"


def test_renders_isolated_thai_and_lao_am_clusters_without_width_drift():
    for text in ["ำabc", "ຳabc"]:
        editor = Editor(create_test_tui(), default_editor_theme)
        width = 8
        editor.set_text(text)

        for line in editor.render(width):
            assert visible_width(line) == width, f"line width drift for {text!r}: {line}"


def test_wraps_cjk_characters_correctly_each_is_2_columns_wide():
    editor = Editor(create_test_tui(), default_editor_theme)
    width = 10 + 1  # +1 col reserved for cursor

    # Each CJK char is 2 columns. "日本語テスト" = 6 chars = 12 columns
    editor.set_text("日本語テスト")
    lines = editor.render(width)

    for i in range(1, len(lines) - 1):
        line_width = visible_width(lines[i])
        assert line_width == width, f"Line {i} has width {line_width}, expected {width}"

    # Verify content split correctly
    content_lines = [strip_ansi(line).strip() for line in lines[1:-1]]
    assert len(content_lines) == 2
    assert content_lines[0] == "日本語テス"  # 5 chars = 10 columns
    assert content_lines[1] == "ト"  # 1 char = 2 columns (+ padding)


def test_handles_mixed_ascii_and_wide_characters_in_wrapping():
    editor = Editor(create_test_tui(), default_editor_theme)
    width = 15 + 1  # +1 col reserved for cursor

    # "Test ✅ OK 日本" = 4 + 1 + 2 + 1 + 2 + 1 + 4 = 15 columns (fits in width-1=15)
    editor.set_text("Test ✅ OK 日本")
    lines = editor.render(width)

    # Should fit in one content line
    content_lines = lines[1:-1]
    assert len(content_lines) == 1

    assert visible_width(content_lines[0]) == width


def test_renders_cursor_correctly_on_wide_characters():
    editor = Editor(create_test_tui(), default_editor_theme)
    width = 20

    editor.set_text("A✅B")
    # Cursor should be at end (after B)
    lines = editor.render(width)

    # The cursor (reverse video space) should be visible
    content_line = lines[1]
    assert "\x1b[7m" in content_line, "Should have reverse video cursor"

    # Line should still be correct width
    assert visible_width(content_line) == width


def test_does_not_exceed_terminal_width_with_emoji_at_wrap_boundary():
    editor = Editor(create_test_tui(), default_editor_theme)
    width = 11

    # "0123456789✅" = 10 ASCII + 2-wide emoji = 12 columns
    # Should wrap before the emoji since it would exceed width
    editor.set_text("0123456789✅")
    lines = editor.render(width)

    for i in range(1, len(lines) - 1):
        line_width = visible_width(lines[i])
        assert line_width <= width, f"Line {i} has width {line_width}, exceeds max {width}"


@pytest.mark.tonio
async def test_shows_cursor_at_end_of_line_before_wrap_wraps_on_next_char():
    width = 10
    for padding_x in [0, 1]:
        editor = Editor(create_test_tui(width + padding_x), default_editor_theme, {"paddingX": padding_x})

        # Type 9 chars → fills layoutWidth exactly, cursor at end on same line
        for ch in "aaaaaaaaa":
            await editor.handle_input(ch)
        lines = editor.render(width + padding_x)
        content_lines = lines[1:-1]
        assert len(content_lines) == 1, "Should be 1 content line before wrap"
        assert content_lines[0].endswith("\x1b[7m \x1b[0m"), "Cursor should be at end of line"

        # Type 1 more → text wraps to second line
        await editor.handle_input("a")
        lines = editor.render(width + padding_x)
        content_lines = lines[1:-1]
        assert len(content_lines) == 2, "Should wrap to 2 content lines"


# Word wrapping


def test_wraps_at_word_boundaries_instead_of_mid_word():
    editor = Editor(create_test_tui(), default_editor_theme)
    width = 40

    editor.set_text("Hello world this is a test of word wrapping functionality")
    lines = editor.render(width)

    # Get content lines (between borders)
    content_lines = [strip_ansi(line).strip() for line in lines[1:-1]]

    # Should NOT break mid-word
    # Line 1 should end with a complete word
    assert not content_lines[0].endswith("-"), "Line should not end with hyphen (mid-word break)"

    # Each content line should be complete words
    for line in content_lines:
        # Words at end of line should be complete (no partial words)
        last_char = line.rstrip()[-1:]
        assert last_char == "" or re.match(r"[\w.,!?;:]", last_char), f'Line ends unexpectedly with: "{last_char}"'


def test_does_not_start_lines_with_leading_whitespace_after_word_wrap():
    editor = Editor(create_test_tui(), default_editor_theme)
    width = 20

    editor.set_text("Word1 Word2 Word3 Word4 Word5 Word6")
    lines = editor.render(width)

    # Get content lines (between borders)
    content_lines = lines[1:-1]

    # No line should start with whitespace (except for padding at the end)
    for i, raw_line in enumerate(content_lines):
        line = strip_ansi(raw_line)
        trimmed_start = line.lstrip()
        # The line should either be all padding or start with a word character
        if trimmed_start:
            assert not re.match(r"^\s+\S", line.rstrip()), f"Line {i} starts with unexpected whitespace before content"


def test_breaks_long_words_urls_at_character_level():
    editor = Editor(create_test_tui(), default_editor_theme)
    width = 30

    editor.set_text("Check https://example.com/very/long/path/that/exceeds/width here")
    lines = editor.render(width)

    # All lines should fit within width
    for i in range(1, len(lines) - 1):
        line_width = visible_width(lines[i])
        assert line_width == width, f"Line {i} has width {line_width}, expected {width}"


def test_preserves_multiple_spaces_within_words_on_same_line():
    editor = Editor(create_test_tui(), default_editor_theme)
    width = 50

    editor.set_text("Word1   Word2    Word3")
    lines = editor.render(width)

    content_line = strip_ansi(lines[1]).strip()
    # Multiple spaces should be preserved
    assert "Word1   Word2" in content_line, "Multiple spaces should be preserved"


def test_handles_empty_string():
    editor = Editor(create_test_tui(), default_editor_theme)
    width = 40

    editor.set_text("")
    lines = editor.render(width)

    # Should have border + empty content + border
    assert len(lines) == 3


def test_handles_single_word_that_fits_exactly():
    editor = Editor(create_test_tui(), default_editor_theme)
    width = 10 + 1  # +1 col reserved for cursor

    editor.set_text("1234567890")
    lines = editor.render(width)

    # Should have exactly 3 lines (top border, content, bottom border)
    assert len(lines) == 3
    content_line = strip_ansi(lines[1])
    assert "1234567890" in content_line, "Content should contain the word"


def test_wraps_word_to_next_line_when_it_ends_exactly_at_terminal_width():
    # "hello " (6) + "world" (5) = 11, but "world" is non-whitespace ending at width.
    # Thus, wrap it to next line. The trailing space stays with "hello" on line 1
    chunks = word_wrap_line("hello world test", 11)

    assert len(chunks) == 2
    assert chunks[0]["text"] == "hello "
    assert chunks[1]["text"] == "world test"


def test_keeps_whitespace_at_terminal_width_boundary_on_same_line():
    # "hello world " is exactly 12 chars (including trailing space)
    # The space at position 12 should stay on the first line
    chunks = word_wrap_line("hello world test", 12)

    assert len(chunks) == 2
    assert chunks[0]["text"] == "hello world "
    assert chunks[1]["text"] == "test"


def test_handles_unbreakable_word_filling_width_exactly_followed_by_space():
    chunks = word_wrap_line("aaaaaaaaaaaa aaaa", 12)

    assert len(chunks) == 2
    assert chunks[0]["text"] == "aaaaaaaaaaaa"
    assert chunks[1]["text"] == " aaaa"


def test_wraps_word_to_next_line_when_it_fits_width_but_not_remaining_space():
    chunks = word_wrap_line("      aaaaaaaaaaaa", 12)

    assert len(chunks) == 2
    assert chunks[0]["text"] == "      "
    assert chunks[1]["text"] == "aaaaaaaaaaaa"


def test_keeps_word_with_multi_space_and_following_word_together_when_they_fit():
    chunks = word_wrap_line("Lorem ipsum dolor sit amet,    consectetur", 30)

    assert len(chunks) == 2
    assert chunks[0]["text"] == "Lorem ipsum dolor sit "
    assert chunks[1]["text"] == "amet,    consectetur"


def test_keeps_word_with_multi_space_and_following_word_when_they_fill_width_exactly():
    chunks = word_wrap_line("Lorem ipsum dolor sit amet,              consectetur", 30)

    assert len(chunks) == 2
    assert chunks[0]["text"] == "Lorem ipsum dolor sit "
    assert chunks[1]["text"] == "amet,              consectetur"


def test_splits_when_word_plus_multi_space_plus_word_exceeds_width():
    chunks = word_wrap_line("Lorem ipsum dolor sit amet,               consectetur", 30)

    assert len(chunks) == 3
    assert chunks[0]["text"] == "Lorem ipsum dolor sit "
    assert chunks[1]["text"] == "amet,               "
    assert chunks[2]["text"] == "consectetur"


def test_breaks_long_whitespace_at_line_boundary():
    chunks = word_wrap_line("Lorem ipsum dolor sit amet,                         consectetur", 30)

    assert len(chunks) == 3
    assert chunks[0]["text"] == "Lorem ipsum dolor sit "
    assert chunks[1]["text"] == "amet,                         "
    assert chunks[2]["text"] == "consectetur"


def test_breaks_long_whitespace_at_line_boundary_2():
    chunks = word_wrap_line("Lorem ipsum dolor sit amet,                          consectetur", 30)

    assert len(chunks) == 3
    assert chunks[0]["text"] == "Lorem ipsum dolor sit "
    assert chunks[1]["text"] == "amet,                         "
    assert chunks[2]["text"] == " consectetur"


def test_breaks_whitespace_spanning_full_lines():
    chunks = word_wrap_line("Lorem ipsum dolor sit amet,                                     consectetur", 30)

    assert len(chunks) == 3
    assert chunks[0]["text"] == "Lorem ipsum dolor sit "
    assert chunks[1]["text"] == "amet,                         "
    assert chunks[2]["text"] == "            consectetur"


def test_force_breaks_when_wide_char_after_word_boundary_wrap_still_overflows():
    # " " (1) + "a"*186 (186) + "你" (2) = 189 visible width
    # maxWidth = 187: backtracking to the space would leave 186 + 2 = 188 > 187,
    # so the algorithm must force-break before the wide char instead.
    line = f" {'a' * 186}你"
    chunks = word_wrap_line(line, 187)

    for chunk in chunks:
        assert visible_width(chunk["text"]) <= 187, (
            f'chunk "{chunk["text"][:20]}..." has visible width {visible_width(chunk["text"])}, expected <= 187'
        )
    # Verify no content is lost
    reconstructed = "".join(line[c["startIndex"] : c["endIndex"]] for c in chunks)
    assert reconstructed == line


def test_splits_oversized_atomic_segment_across_multiple_chunks():
    # Simulate a paste marker wider than maxWidth by passing pre-segmented data
    marker = "[paste #1 +20 lines]"  # 21 chars
    line = f"A{marker}B"
    segments = [
        {"segment": "A", "index": 0},
        {"segment": marker, "index": 1},
        {"segment": "B", "index": 1 + len(marker)},
    ]

    chunks = word_wrap_line(line, 10, segments)

    # Every chunk must fit within maxWidth
    for chunk in chunks:
        assert visible_width(chunk["text"]) <= 10, (
            f'chunk "{chunk["text"]}" has visible width {visible_width(chunk["text"])}, expected <= 10'
        )

    # Verify no content is lost
    reconstructed = "".join(line[c["startIndex"] : c["endIndex"]] for c in chunks)
    assert reconstructed == line


def test_splits_oversized_atomic_segment_at_start_of_line():
    marker = "[paste #1 +20 lines]"  # 21 chars
    line = f"{marker}B"
    segments = [
        {"segment": marker, "index": 0},
        {"segment": "B", "index": len(marker)},
    ]

    chunks = word_wrap_line(line, 10, segments)

    for chunk in chunks:
        assert visible_width(chunk["text"]) <= 10
    # "B" ends up on the last line (either alone or with the marker tail)
    assert "B" in chunks[-1]["text"]

    reconstructed = "".join(line[c["startIndex"] : c["endIndex"]] for c in chunks)
    assert reconstructed == line


def test_splits_oversized_atomic_segment_at_end_of_line():
    marker = "[paste #1 +20 lines]"  # 21 chars
    line = f"A{marker}"
    segments = [
        {"segment": "A", "index": 0},
        {"segment": marker, "index": 1},
    ]

    chunks = word_wrap_line(line, 10, segments)

    for chunk in chunks:
        assert visible_width(chunk["text"]) <= 10
    assert chunks[0]["text"] == "A"

    reconstructed = "".join(line[c["startIndex"] : c["endIndex"]] for c in chunks)
    assert reconstructed == line


def test_splits_consecutive_oversized_atomic_segments():
    m1 = "[paste #1 +20 lines]"  # 21 chars
    m2 = "[paste #2 +30 lines]"  # 21 chars
    line = f"{m1}{m2}"
    segments = [
        {"segment": m1, "index": 0},
        {"segment": m2, "index": len(m1)},
    ]

    chunks = word_wrap_line(line, 10, segments)

    for chunk in chunks:
        assert visible_width(chunk["text"]) <= 10, (
            f'chunk "{chunk["text"]}" has visible width {visible_width(chunk["text"])}, expected <= 10'
        )

    reconstructed = "".join(line[c["startIndex"] : c["endIndex"]] for c in chunks)
    assert reconstructed == line


def test_wraps_normally_after_oversized_atomic_segment():
    marker = "[paste #1 +20 lines]"  # 21 chars
    line = f"{marker} hello world"
    segments = [
        {"segment": marker, "index": 0},
        {"segment": " ", "index": len(marker)},
        {"segment": "h", "index": len(marker) + 1},
        {"segment": "e", "index": len(marker) + 2},
        {"segment": "l", "index": len(marker) + 3},
        {"segment": "l", "index": len(marker) + 4},
        {"segment": "o", "index": len(marker) + 5},
        {"segment": " ", "index": len(marker) + 6},
        {"segment": "w", "index": len(marker) + 7},
        {"segment": "o", "index": len(marker) + 8},
        {"segment": "r", "index": len(marker) + 9},
        {"segment": "l", "index": len(marker) + 10},
        {"segment": "d", "index": len(marker) + 11},
    ]

    chunks = word_wrap_line(line, 10, segments)

    # All chunks must fit
    for chunk in chunks:
        assert visible_width(chunk["text"]) <= 10, (
            f'chunk "{chunk["text"]}" has visible width {visible_width(chunk["text"])}, expected <= 10'
        )

    # Last chunk should contain "world" (normal wrapping resumes)
    assert chunks[-1]["text"] == "world"

    reconstructed = "".join(line[c["startIndex"] : c["endIndex"]] for c in chunks)
    assert reconstructed == line


# Kill ring


@pytest.mark.tonio
async def test_ctrl_w_saves_deleted_text_to_kill_ring_and_ctrl_y_yanks_it():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("foo bar baz")
    await editor.handle_input("\x17")  # Ctrl+W - deletes "baz"
    assert editor.get_text() == "foo bar "

    # Move to beginning and yank
    await editor.handle_input("\x01")  # Ctrl+A
    await editor.handle_input("\x19")  # Ctrl+Y
    assert editor.get_text() == "bazfoo bar "


@pytest.mark.tonio
async def test_ctrl_u_saves_deleted_text_to_kill_ring():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("hello world")
    # Move cursor to middle
    await editor.handle_input("\x01")  # Ctrl+A (start)
    await editor.handle_input("\x1b[C")  # Right 6 times
    await editor.handle_input("\x1b[C")
    await editor.handle_input("\x1b[C")
    await editor.handle_input("\x1b[C")
    await editor.handle_input("\x1b[C")
    await editor.handle_input("\x1b[C")  # After "hello "

    await editor.handle_input("\x15")  # Ctrl+U - deletes "hello "
    assert editor.get_text() == "world"

    await editor.handle_input("\x19")  # Ctrl+Y
    assert editor.get_text() == "hello world"


@pytest.mark.tonio
async def test_ctrl_k_saves_deleted_text_to_kill_ring():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("hello world")
    await editor.handle_input("\x01")  # Ctrl+A (start)
    await editor.handle_input("\x0b")  # Ctrl+K - deletes "hello world"

    assert editor.get_text() == ""

    await editor.handle_input("\x19")  # Ctrl+Y
    assert editor.get_text() == "hello world"


@pytest.mark.tonio
async def test_ctrl_y_does_nothing_when_kill_ring_is_empty():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("test")
    await editor.handle_input("\x19")  # Ctrl+Y
    assert editor.get_text() == "test"


@pytest.mark.tonio
async def test_alt_y_cycles_through_kill_ring_after_ctrl_y():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Create kill ring with multiple entries
    editor.set_text("first")
    await editor.handle_input("\x17")  # Ctrl+W - deletes "first"
    editor.set_text("second")
    await editor.handle_input("\x17")  # Ctrl+W - deletes "second"
    editor.set_text("third")
    await editor.handle_input("\x17")  # Ctrl+W - deletes "third"

    # Kill ring now has: [first, second, third]
    assert editor.get_text() == ""

    await editor.handle_input("\x19")  # Ctrl+Y - yanks "third" (most recent)
    assert editor.get_text() == "third"

    await editor.handle_input("\x1by")  # Alt+Y - cycles to "second"
    assert editor.get_text() == "second"

    await editor.handle_input("\x1by")  # Alt+Y - cycles to "first"
    assert editor.get_text() == "first"

    await editor.handle_input("\x1by")  # Alt+Y - cycles back to "third"
    assert editor.get_text() == "third"


@pytest.mark.tonio
async def test_alt_y_does_nothing_if_not_preceded_by_yank():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("test")
    await editor.handle_input("\x17")  # Ctrl+W - deletes "test"
    editor.set_text("other")

    # Type something to break the yank chain
    await editor.handle_input("x")
    assert editor.get_text() == "otherx"

    # Alt+Y should do nothing
    await editor.handle_input("\x1by")  # Alt+Y
    assert editor.get_text() == "otherx"


@pytest.mark.tonio
async def test_alt_y_does_nothing_if_kill_ring_has_at_most_1_entry():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("only")
    await editor.handle_input("\x17")  # Ctrl+W - deletes "only"

    await editor.handle_input("\x19")  # Ctrl+Y - yanks "only"
    assert editor.get_text() == "only"

    await editor.handle_input("\x1by")  # Alt+Y - should do nothing (only 1 entry)
    assert editor.get_text() == "only"


@pytest.mark.tonio
async def test_consecutive_ctrl_w_accumulates_into_one_kill_ring_entry():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("one two three")
    await editor.handle_input("\x17")  # Ctrl+W - deletes "three"
    await editor.handle_input("\x17")  # Ctrl+W - deletes "two " (prepended)
    await editor.handle_input("\x17")  # Ctrl+W - deletes "one " (prepended)

    assert editor.get_text() == ""

    # Should be one combined entry
    await editor.handle_input("\x19")  # Ctrl+Y
    assert editor.get_text() == "one two three"


@pytest.mark.tonio
async def test_ctrl_u_accumulates_multiline_deletes_including_newlines():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Start with multiline text, cursor at end
    editor.set_text("line1\nline2\nline3")
    # Cursor is at end of line3 (line 2, col 5)

    # Delete "line3"
    await editor.handle_input("\x15")  # Ctrl+U
    assert editor.get_text() == "line1\nline2\n"

    # Delete newline (at start of empty line 2, merges with line1)
    await editor.handle_input("\x15")  # Ctrl+U
    assert editor.get_text() == "line1\nline2"

    # Delete "line2"
    await editor.handle_input("\x15")  # Ctrl+U
    assert editor.get_text() == "line1\n"

    # Delete newline
    await editor.handle_input("\x15")  # Ctrl+U
    assert editor.get_text() == "line1"

    # Delete "line1"
    await editor.handle_input("\x15")  # Ctrl+U
    assert editor.get_text() == ""

    # All deletions accumulated into one entry: "line1\nline2\nline3"
    await editor.handle_input("\x19")  # Ctrl+Y
    assert editor.get_text() == "line1\nline2\nline3"


@pytest.mark.tonio
async def test_backward_deletions_prepend_forward_deletions_append_during_accumulation():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("prefix|suffix")
    # Position cursor at |
    await editor.handle_input("\x01")  # Ctrl+A
    for _ in range(6):
        await editor.handle_input("\x1b[C")  # Move right 6 times

    await editor.handle_input("\x0b")  # Ctrl+K - deletes "suffix" (forward)
    await editor.handle_input("\x0b")  # Ctrl+K - deletes "|" (forward, appended)
    assert editor.get_text() == "prefix"

    await editor.handle_input("\x19")  # Ctrl+Y
    assert editor.get_text() == "prefix|suffix"


@pytest.mark.tonio
async def test_non_delete_actions_break_kill_accumulation():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Delete "baz", then type "x" to break accumulation, then delete "x"
    editor.set_text("foo bar baz")
    await editor.handle_input("\x17")  # Ctrl+W - deletes "baz"
    assert editor.get_text() == "foo bar "

    await editor.handle_input("x")  # Typing breaks accumulation
    assert editor.get_text() == "foo bar x"

    await editor.handle_input("\x17")  # Ctrl+W - deletes "x" (separate entry, not accumulated)
    assert editor.get_text() == "foo bar "

    # Yank most recent - should be "x", not "xbaz"
    await editor.handle_input("\x19")  # Ctrl+Y
    assert editor.get_text() == "foo bar x"

    # Cycle to previous - should be "baz" (separate entry)
    await editor.handle_input("\x1by")  # Alt+Y
    assert editor.get_text() == "foo bar baz"


@pytest.mark.tonio
async def test_non_yank_actions_break_alt_y_chain():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("first")
    await editor.handle_input("\x17")  # Ctrl+W
    editor.set_text("second")
    await editor.handle_input("\x17")  # Ctrl+W
    editor.set_text("")

    await editor.handle_input("\x19")  # Ctrl+Y - yanks "second"
    assert editor.get_text() == "second"

    await editor.handle_input("x")  # Type breaks yank chain
    assert editor.get_text() == "secondx"

    await editor.handle_input("\x1by")  # Alt+Y - should do nothing
    assert editor.get_text() == "secondx"


@pytest.mark.tonio
async def test_kill_ring_rotation_persists_after_cycling():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("first")
    await editor.handle_input("\x17")  # deletes "first"
    editor.set_text("second")
    await editor.handle_input("\x17")  # deletes "second"
    editor.set_text("third")
    await editor.handle_input("\x17")  # deletes "third"
    editor.set_text("")

    # Ring: [first, second, third]

    await editor.handle_input("\x19")  # Ctrl+Y - yanks "third"
    await editor.handle_input("\x1by")  # Alt+Y - cycles to "second", ring rotates

    # Now ring is: [third, first, second]
    assert editor.get_text() == "second"

    # Do something else
    await editor.handle_input("x")
    editor.set_text("")

    # New yank should get "second" (now at end after rotation)
    await editor.handle_input("\x19")  # Ctrl+Y
    assert editor.get_text() == "second"


@pytest.mark.tonio
async def test_consecutive_deletions_across_lines_coalesce_into_one_entry():
    editor = Editor(create_test_tui(), default_editor_theme)

    # "1\n2\n3" with cursor at end, delete everything with Ctrl+W
    editor.set_text("1\n2\n3")
    await editor.handle_input("\x17")  # Ctrl+W - deletes "3"
    assert editor.get_text() == "1\n2\n"

    await editor.handle_input("\x17")  # Ctrl+W - deletes newline (merge with prev line)
    assert editor.get_text() == "1\n2"

    await editor.handle_input("\x17")  # Ctrl+W - deletes "2"
    assert editor.get_text() == "1\n"

    await editor.handle_input("\x17")  # Ctrl+W - deletes newline
    assert editor.get_text() == "1"

    await editor.handle_input("\x17")  # Ctrl+W - deletes "1"
    assert editor.get_text() == ""

    # All deletions should have accumulated into one entry
    await editor.handle_input("\x19")  # Ctrl+Y
    assert editor.get_text() == "1\n2\n3"


@pytest.mark.tonio
async def test_ctrl_k_at_line_end_deletes_newline_and_coalesces():
    editor = Editor(create_test_tui(), default_editor_theme)

    # "ab" on line 1, "cd" on line 2, cursor at end of line 1
    editor.set_text("")
    await editor.handle_input("a")
    await editor.handle_input("b")
    await editor.handle_input("\n")
    await editor.handle_input("c")
    await editor.handle_input("d")
    # Move to end of first line
    await editor.handle_input("\x1b[A")  # Up arrow
    await editor.handle_input("\x05")  # Ctrl+E - end of line

    # Now at end of "ab", Ctrl+K should delete newline (merge with "cd")
    await editor.handle_input("\x0b")  # Ctrl+K - deletes newline
    assert editor.get_text() == "abcd"

    # Continue deleting
    await editor.handle_input("\x0b")  # Ctrl+K - deletes "cd"
    assert editor.get_text() == "ab"

    # Both deletions should accumulate
    await editor.handle_input("\x19")  # Ctrl+Y
    assert editor.get_text() == "ab\ncd"


@pytest.mark.tonio
async def test_handles_yank_in_middle_of_text():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("word")
    await editor.handle_input("\x17")  # Ctrl+W - deletes "word"
    editor.set_text("hello world")

    # Move to middle (after "hello ")
    await editor.handle_input("\x01")  # Ctrl+A
    for _ in range(6):
        await editor.handle_input("\x1b[C")

    await editor.handle_input("\x19")  # Ctrl+Y
    assert editor.get_text() == "hello wordworld"


@pytest.mark.tonio
async def test_handles_yank_pop_in_middle_of_text():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Create two kill ring entries
    editor.set_text("FIRST")
    await editor.handle_input("\x17")  # Ctrl+W - deletes "FIRST"
    editor.set_text("SECOND")
    await editor.handle_input("\x17")  # Ctrl+W - deletes "SECOND"

    # Ring: ["FIRST", "SECOND"]

    # Set up "hello world" and position cursor after "hello "
    editor.set_text("hello world")
    await editor.handle_input("\x01")  # Ctrl+A - go to start of line
    for _ in range(6):
        await editor.handle_input("\x1b[C")  # Move right 6

    # Yank "SECOND" in the middle
    await editor.handle_input("\x19")  # Ctrl+Y
    assert editor.get_text() == "hello SECONDworld"

    # Yank-pop replaces "SECOND" with "FIRST"
    await editor.handle_input("\x1by")  # Alt+Y
    assert editor.get_text() == "hello FIRSTworld"


@pytest.mark.tonio
async def test_multiline_yank_and_yank_pop_in_middle_of_text():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Create single-line entry
    editor.set_text("SINGLE")
    await editor.handle_input("\x17")  # Ctrl+W - deletes "SINGLE"

    # Create multiline entry via consecutive Ctrl+U
    editor.set_text("A\nB")
    await editor.handle_input("\x15")  # Ctrl+U - deletes "B"
    await editor.handle_input("\x15")  # Ctrl+U - deletes newline
    await editor.handle_input("\x15")  # Ctrl+U - deletes "A"
    # Ring: ["SINGLE", "A\nB"]

    # Insert in middle of "hello world"
    editor.set_text("hello world")
    await editor.handle_input("\x01")  # Ctrl+A
    for _ in range(6):
        await editor.handle_input("\x1b[C")

    # Yank multiline "A\nB"
    await editor.handle_input("\x19")  # Ctrl+Y
    assert editor.get_text() == "hello A\nBworld"

    # Yank-pop replaces with "SINGLE"
    await editor.handle_input("\x1by")  # Alt+Y
    assert editor.get_text() == "hello SINGLEworld"


@pytest.mark.tonio
async def test_alt_d_deletes_word_forward_and_saves_to_kill_ring():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("hello world test")
    await editor.handle_input("\x01")  # Ctrl+A - go to start

    await editor.handle_input("\x1bd")  # Alt+D - deletes "hello"
    assert editor.get_text() == " world test"

    await editor.handle_input("\x1bd")  # Alt+D - deletes " world" (skips whitespace, then word)
    assert editor.get_text() == " test"

    # Yank should get accumulated text
    await editor.handle_input("\x19")  # Ctrl+Y
    assert editor.get_text() == "hello world test"


@pytest.mark.tonio
async def test_alt_d_at_end_of_line_deletes_newline():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("line1\nline2")
    # Move to start of document, then to end of first line
    await editor.handle_input("\x1b[A")  # Up arrow - go to first line
    await editor.handle_input("\x05")  # Ctrl+E - end of line

    await editor.handle_input("\x1bd")  # Alt+D - deletes newline (merges lines)
    assert editor.get_text() == "line1line2"

    await editor.handle_input("\x19")  # Ctrl+Y
    assert editor.get_text() == "line1\nline2"


# Undo


@pytest.mark.tonio
async def test_does_nothing_when_undo_stack_is_empty():
    editor = Editor(create_test_tui(), default_editor_theme)

    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == ""


@pytest.mark.tonio
async def test_coalesces_consecutive_word_characters_into_one_undo_unit():
    editor = Editor(create_test_tui(), default_editor_theme)

    for ch in "hello world":
        await editor.handle_input(ch)
    assert editor.get_text() == "hello world"

    # Undo removes " world" (space captured state before it, so we restore to "hello")
    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "hello"

    # Undo removes "hello"
    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == ""


@pytest.mark.tonio
async def test_undoes_spaces_one_at_a_time():
    editor = Editor(create_test_tui(), default_editor_theme)

    for ch in "hello  ":
        await editor.handle_input(ch)
    assert editor.get_text() == "hello  "

    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo) - removes second " "
    assert editor.get_text() == "hello "

    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo) - removes first " "
    assert editor.get_text() == "hello"

    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo) - removes "hello"
    assert editor.get_text() == ""


@pytest.mark.tonio
async def test_undoes_newlines_and_signals_next_word_to_capture_state():
    editor = Editor(create_test_tui(), default_editor_theme)

    for ch in "hello":
        await editor.handle_input(ch)
    await editor.handle_input("\n")
    for ch in "world":
        await editor.handle_input(ch)
    assert editor.get_text() == "hello\nworld"

    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "hello\n"

    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "hello"

    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == ""


@pytest.mark.tonio
async def test_undoes_backspace():
    editor = Editor(create_test_tui(), default_editor_theme)

    for ch in "hello":
        await editor.handle_input(ch)
    await editor.handle_input("\x7f")  # Backspace
    assert editor.get_text() == "hell"

    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "hello"


@pytest.mark.tonio
async def test_undoes_forward_delete():
    editor = Editor(create_test_tui(), default_editor_theme)

    for ch in "hello":
        await editor.handle_input(ch)
    await editor.handle_input("\x01")  # Ctrl+A - go to start
    await editor.handle_input("\x1b[C")  # Right arrow
    await editor.handle_input("\x1b[3~")  # Delete key
    assert editor.get_text() == "hllo"

    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "hello"


@pytest.mark.tonio
async def test_undoes_ctrl_w_delete_word_backward():
    editor = Editor(create_test_tui(), default_editor_theme)

    for ch in "hello world":
        await editor.handle_input(ch)
    assert editor.get_text() == "hello world"

    await editor.handle_input("\x17")  # Ctrl+W
    assert editor.get_text() == "hello "

    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "hello world"


@pytest.mark.tonio
async def test_undoes_ctrl_k_delete_to_line_end():
    editor = Editor(create_test_tui(), default_editor_theme)

    for ch in "hello world":
        await editor.handle_input(ch)
    await editor.handle_input("\x01")  # Ctrl+A - go to start
    for _ in range(6):
        await editor.handle_input("\x1b[C")  # Move right 6 times

    await editor.handle_input("\x0b")  # Ctrl+K
    assert editor.get_text() == "hello "

    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "hello world"

    await editor.handle_input("|")
    assert editor.get_text() == "hello |world"


@pytest.mark.tonio
async def test_undoes_ctrl_u_delete_to_line_start():
    editor = Editor(create_test_tui(), default_editor_theme)

    for ch in "hello world":
        await editor.handle_input(ch)
    await editor.handle_input("\x01")  # Ctrl+A - go to start
    for _ in range(6):
        await editor.handle_input("\x1b[C")  # Move right 6 times

    await editor.handle_input("\x15")  # Ctrl+U
    assert editor.get_text() == "world"

    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "hello world"


@pytest.mark.tonio
async def test_undoes_yank():
    editor = Editor(create_test_tui(), default_editor_theme)

    for ch in "hello ":
        await editor.handle_input(ch)
    await editor.handle_input("\x17")  # Ctrl+W - delete "hello "
    await editor.handle_input("\x19")  # Ctrl+Y - yank
    assert editor.get_text() == "hello "

    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == ""


@pytest.mark.tonio
async def test_undoes_single_line_paste_atomically():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("hello world")
    await editor.handle_input("\x01")  # Ctrl+A - go to start
    for _ in range(5):
        await editor.handle_input("\x1b[C")  # Move right 5 (after "hello", before space)

    # Simulate bracketed paste of "beep boop"
    await editor.handle_input("\x1b[200~beep boop\x1b[201~")
    assert editor.get_text() == "hellobeep boop world"

    # Single undo should restore entire pre-paste state
    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "hello world"

    await editor.handle_input("|")
    assert editor.get_text() == "hello| world"


@pytest.mark.tonio
async def test_does_not_trigger_autocomplete_during_single_line_paste():
    editor = Editor(create_test_tui(), default_editor_theme)
    suggestion_calls = []

    async def get_suggestions(lines, cursor_line, cursor_col, options):
        suggestion_calls.append(1)

    editor.set_autocomplete_provider(MockProvider(get_suggestions))
    await editor.handle_input("\x1b[200~look at @node_modules/react/index.js please\x1b[201~")

    assert editor.get_text() == "look at @node_modules/react/index.js please"
    assert len(suggestion_calls) == 0
    assert editor.is_showing_autocomplete() is False


@pytest.mark.tonio
async def test_decodes_csi_u_ctrl_letter_sequences_inside_bracketed_paste_tmux_popup():
    editor = Editor(create_test_tui(), default_editor_theme)

    # tmux popups with extended-keys-format=csi-u re-encode \n in pastes as
    # \x1b[106;5u (Ctrl+J). Without decoding, the per-char filter strips ESC
    # and leaks "[106;5u" between lines. See issue #3599.
    await editor.handle_input("\x1b[200~line1\x1b[106;5uline2\x1b[106;5uline3\x1b[201~")
    assert editor.get_text() == "line1\nline2\nline3"


@pytest.mark.tonio
async def test_undoes_multi_line_paste_atomically():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("hello world")
    await editor.handle_input("\x01")  # Ctrl+A - go to start
    for _ in range(5):
        await editor.handle_input("\x1b[C")  # Move right 5 (after "hello", before space)

    # Simulate bracketed paste of multi-line text
    await editor.handle_input("\x1b[200~line1\nline2\nline3\x1b[201~")
    assert editor.get_text() == "helloline1\nline2\nline3 world"

    # Single undo should restore entire pre-paste state
    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "hello world"

    await editor.handle_input("|")
    assert editor.get_text() == "hello| world"


@pytest.mark.tonio
async def test_undoes_insert_text_at_cursor_atomically():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("hello world")
    await editor.handle_input("\x01")  # Ctrl+A - go to start
    for _ in range(5):
        await editor.handle_input("\x1b[C")  # Move right 5 (after "hello", before space)

    # Programmatic insertion (e.g., clipboard image path)
    editor.insert_text_at_cursor("/tmp/image.png")
    assert editor.get_text() == "hello/tmp/image.png world"

    # Single undo should restore entire pre-insert state
    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "hello world"

    await editor.handle_input("|")
    assert editor.get_text() == "hello| world"


@pytest.mark.tonio
async def test_insert_text_at_cursor_handles_multiline_text():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("hello world")
    await editor.handle_input("\x01")  # Ctrl+A - go to start
    for _ in range(5):
        await editor.handle_input("\x1b[C")  # Move right 5 (after "hello", before space)

    # Insert multiline text
    editor.insert_text_at_cursor("line1\nline2\nline3")
    assert editor.get_text() == "helloline1\nline2\nline3 world"

    # Cursor should be at end of inserted text (after "line3", before " world")
    cursor = editor.get_cursor()
    assert cursor["line"] == 2
    assert cursor["col"] == 5  # len("line3")

    # Single undo should restore entire pre-insert state
    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "hello world"


@pytest.mark.tonio
async def test_insert_text_at_cursor_normalizes_crlf_and_cr_line_endings():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("")

    # Insert text with CRLF
    editor.insert_text_at_cursor("a\r\nb\r\nc")
    assert editor.get_text() == "a\nb\nc"

    await editor.handle_input("\x1b[45;5u")  # Undo
    assert editor.get_text() == ""

    # Insert text with CR only
    editor.insert_text_at_cursor("x\ry\rz")
    assert editor.get_text() == "x\ny\nz"


@pytest.mark.tonio
async def test_undoes_set_text_to_empty_string():
    editor = Editor(create_test_tui(), default_editor_theme)

    for ch in "hello world":
        await editor.handle_input(ch)
    assert editor.get_text() == "hello world"

    editor.set_text("")
    assert editor.get_text() == ""

    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "hello world"


@pytest.mark.tonio
async def test_clears_undo_stack_on_submit():
    editor = Editor(create_test_tui(), default_editor_theme)
    submitted = []
    editor.on_submit = lambda text: submitted.append(text)

    for ch in "hello":
        await editor.handle_input(ch)
    await editor.handle_input("\r")  # Enter - submit

    assert submitted == ["hello"]
    assert editor.get_text() == ""

    # Undo should do nothing - stack was cleared
    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == ""


@pytest.mark.tonio
async def test_exits_history_browsing_mode_on_undo():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Add "hello" to history
    editor.add_to_history("hello")
    assert editor.get_text() == ""

    # Type "world"
    for ch in "world":
        await editor.handle_input(ch)
    assert editor.get_text() == "world"

    # Ctrl+W - delete word
    await editor.handle_input("\x17")  # Ctrl+W
    assert editor.get_text() == ""

    # Press Up - enter history browsing, shows "hello"
    await editor.handle_input("\x1b[A")  # Up arrow
    assert editor.get_text() == "hello"

    # Undo should restore to "" (state before entering history browsing)
    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == ""

    # Undo again should restore to "world" (state before Ctrl+W)
    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "world"


@pytest.mark.tonio
async def test_undo_restores_to_pre_history_state_even_after_multiple_history_navigations():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Add history entries
    editor.add_to_history("first")
    editor.add_to_history("second")
    editor.add_to_history("third")

    # Type something
    for ch in "current":
        await editor.handle_input(ch)
    assert editor.get_text() == "current"

    # Clear editor
    await editor.handle_input("\x17")  # Ctrl+W
    assert editor.get_text() == ""

    # Navigate through history multiple times
    await editor.handle_input("\x1b[A")  # Up - "third"
    assert editor.get_text() == "third"
    await editor.handle_input("\x1b[A")  # Up - "second"
    assert editor.get_text() == "second"
    await editor.handle_input("\x1b[A")  # Up - "first"
    assert editor.get_text() == "first"

    # Undo should go back to "" (state before we started browsing), not intermediate states
    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == ""

    # Another undo goes back to "current"
    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "current"


@pytest.mark.tonio
async def test_cursor_movement_starts_new_undo_unit():
    editor = Editor(create_test_tui(), default_editor_theme)

    for ch in "hello world":
        await editor.handle_input(ch)
    assert editor.get_text() == "hello world"

    # Move cursor left 5 (to after "hello ")
    for _ in range(5):
        await editor.handle_input("\x1b[D")

    # Type "lol" in the middle
    await editor.handle_input("l")
    await editor.handle_input("o")
    await editor.handle_input("l")
    assert editor.get_text() == "hello lolworld"

    # Undo should restore to "hello world" (before inserting "lol")
    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "hello world"

    await editor.handle_input("|")
    assert editor.get_text() == "hello |world"


@pytest.mark.tonio
async def test_no_op_delete_operations_do_not_push_undo_snapshots():
    editor = Editor(create_test_tui(), default_editor_theme)

    for ch in "hello":
        await editor.handle_input(ch)
    assert editor.get_text() == "hello"

    # Delete word on empty - multiple times (should be no-ops)
    await editor.handle_input("\x17")  # Ctrl+W - deletes "hello"
    assert editor.get_text() == ""
    await editor.handle_input("\x17")  # Ctrl+W - no-op (nothing to delete)
    await editor.handle_input("\x17")  # Ctrl+W - no-op

    # Single undo should restore "hello"
    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "hello"


@pytest.mark.tonio
async def test_undoes_autocomplete():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Create a mock autocomplete provider
    async def get_suggestions(lines, cursor_line, cursor_col, options):
        text = lines[0] if lines else ""
        prefix = text[:cursor_col]
        if prefix == "di":
            return {"items": [{"value": "dist/", "label": "dist/"}], "prefix": "di"}
        return None

    editor.set_autocomplete_provider(MockProvider(get_suggestions))

    # Type "di"
    await editor.handle_input("d")
    await editor.handle_input("i")
    assert editor.get_text() == "di"

    # Press Tab to trigger autocomplete
    await editor.handle_input("\t")
    await flush_autocomplete()
    assert editor.get_text() == "dist/"
    assert editor.is_showing_autocomplete() is False

    # Undo should restore to "di"
    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "di"


# Autocomplete


@pytest.mark.tonio
async def test_auto_applies_single_force_file_suggestion_without_showing_menu():
    editor = Editor(create_test_tui(), default_editor_theme)

    async def get_suggestions(lines, cursor_line, cursor_col, options):
        if not options.get("force"):
            return None
        text = lines[0] if lines else ""
        prefix = text[:cursor_col]
        if prefix == "Work":
            return {"items": [{"value": "Workspace/", "label": "Workspace/"}], "prefix": "Work"}
        return None

    editor.set_autocomplete_provider(MockProvider(get_suggestions))

    # Type "Work"
    for ch in "Work":
        await editor.handle_input(ch)
    assert editor.get_text() == "Work"

    # Press Tab - should auto-apply without showing menu
    await editor.handle_input("\t")
    await flush_autocomplete()
    assert editor.get_text() == "Workspace/"
    assert editor.is_showing_autocomplete() is False

    # Undo should restore to "Work"
    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "Work"


@pytest.mark.tonio
async def test_shows_menu_when_force_file_has_multiple_suggestions():
    editor = Editor(create_test_tui(), default_editor_theme)

    async def get_suggestions(lines, cursor_line, cursor_col, options):
        if not options.get("force"):
            return None
        text = lines[0] if lines else ""
        prefix = text[:cursor_col]
        if prefix == "src":
            return {
                "items": [
                    {"value": "src/", "label": "src/"},
                    {"value": "src.txt", "label": "src.txt"},
                ],
                "prefix": "src",
            }
        return None

    editor.set_autocomplete_provider(MockProvider(get_suggestions))

    # Type "src"
    for ch in "src":
        await editor.handle_input(ch)
    assert editor.get_text() == "src"

    # Press Tab - should show menu because there are multiple suggestions
    await editor.handle_input("\t")
    await flush_autocomplete()
    assert editor.get_text() == "src"
    assert editor.is_showing_autocomplete() is True

    # Press Tab again to accept first suggestion
    await editor.handle_input("\t")
    assert editor.get_text() == "src/"
    assert editor.is_showing_autocomplete() is False


@pytest.mark.tonio
async def test_keeps_suggestions_open_when_typing_in_force_mode_tab_triggered():
    editor = Editor(create_test_tui(), default_editor_theme)

    all_files = [
        {"value": "readme.md", "label": "readme.md"},
        {"value": "package.json", "label": "package.json"},
        {"value": "src/", "label": "src/"},
        {"value": "dist/", "label": "dist/"},
    ]

    async def get_suggestions(lines, cursor_line, cursor_col, options):
        text = lines[0] if lines else ""
        prefix = text[:cursor_col]
        should_match = options.get("force") or "/" in prefix or prefix.startswith(".")
        if not should_match:
            return None
        filtered = [f for f in all_files if f["value"].lower().startswith(prefix.lower())]
        if filtered:
            return {"items": filtered, "prefix": prefix}
        return None

    editor.set_autocomplete_provider(MockProvider(get_suggestions))

    # Press Tab on empty prompt - should show all files (force mode)
    await editor.handle_input("\t")
    await flush_autocomplete()
    assert editor.is_showing_autocomplete() is True

    # Type "r" - should narrow to "readme.md" (force mode keeps suggestions open)
    await editor.handle_input("r")
    await flush_autocomplete()
    assert editor.get_text() == "r"
    assert editor.is_showing_autocomplete() is True

    # Type "e" - should still show "readme.md"
    await editor.handle_input("e")
    await flush_autocomplete()
    assert editor.get_text() == "re"
    assert editor.is_showing_autocomplete() is True

    # Accept with Tab
    await editor.handle_input("\t")
    assert editor.get_text() == "readme.md"
    assert editor.is_showing_autocomplete() is False


@pytest.mark.tonio
async def test_debounces_at_autocomplete_while_typing(request):
    slow_debounce(request)
    editor = Editor(create_test_tui(), default_editor_theme)
    suggestion_calls = []

    async def get_suggestions(lines, cursor_line, cursor_col, options):
        suggestion_calls.append(1)
        text = (lines[0] if lines else "")[:cursor_col]
        return {"items": [{"value": "@main.ts", "label": "main.ts"}], "prefix": text}

    editor.set_autocomplete_provider(MockProvider(get_suggestions))

    await editor.handle_input("@")
    await editor.handle_input("m")
    await editor.handle_input("a")
    await editor.handle_input("i")

    assert len(suggestion_calls) == 0
    assert editor.is_showing_autocomplete() is False

    await wait_slow_debounce()
    await flush_autocomplete()

    assert len(suggestion_calls) == 1
    assert editor.is_showing_autocomplete() is True


@pytest.mark.tonio
async def test_re_queries_the_autocomplete_picker_when_the_cursor_moves_back_into_the_command_name():
    # Regression for earendil-works/pi#5496: arrowing left out of a slash
    # command's argument region must re-query the picker, not leave the
    # stale argument list showing. Before the fix, moveCursor() never
    # called updateAutocomplete(), so `/cmd ` (argument menu) + Left kept
    # displaying the arguments against a `/cmd` prefix — and a Tab there
    # would concatenate the stale suggestion onto the partial command name.
    editor = Editor(create_test_tui(), default_editor_theme)

    async def get_suggestions(lines, cursor_line, cursor_col, options):
        before = (lines[0] if lines else "")[:cursor_col]
        if not before.startswith("/"):
            return None
        # Past the command name (a space before the cursor): offer arguments.
        if " " in before:
            return {
                "items": [
                    {"value": "repo", "label": "repo"},
                    {"value": "message", "label": "message"},
                    {"value": "help", "label": "help"},
                ],
                "prefix": before[before.index(" ") + 1 :],
            }
        # Inside the command name: offer the command name only.
        return {"items": [{"value": "cmd", "label": "cmd"}], "prefix": before}

    editor.set_autocomplete_provider(MockProvider(get_suggestions))

    # Type `/cmd ` so the picker ends up showing the argument list.
    for ch in "/cmd ":
        await editor.handle_input(ch)
        await flush_autocomplete()
    assert editor.get_text() == "/cmd "
    assert editor.is_showing_autocomplete() is True
    at_arg = "\n".join(strip_ansi(line) for line in editor.render(80))
    assert "repo" in at_arg, "argument menu should be visible at `/cmd `"

    # Arrow Left back into the command name (`/cmd`).
    await editor.handle_input("\x1b[D")
    await flush_autocomplete()

    # The picker must have re-queried: the stale argument items are gone
    # (replaced by the command-name suggestion, or the picker closed).
    after_move = "\n".join(strip_ansi(line) for line in editor.render(80))
    assert "repo" not in after_move, "stale argument menu must not survive the cursor move"
    assert "message" not in after_move, "stale argument menu must not survive the cursor move"


@pytest.mark.tonio
async def test_debounces_hash_autocomplete_while_typing(request):
    slow_debounce(request)
    editor = Editor(create_test_tui(), default_editor_theme)
    suggestion_calls = []

    async def get_suggestions(lines, cursor_line, cursor_col, options):
        suggestion_calls.append(1)
        text = (lines[0] if lines else "")[:cursor_col]
        return {"items": [{"value": "#2983", "label": "#2983"}], "prefix": text}

    editor.set_autocomplete_provider(MockProvider(get_suggestions))

    await editor.handle_input("#")
    await editor.handle_input("2")
    await editor.handle_input("9")
    await editor.handle_input("8")

    assert len(suggestion_calls) == 0
    assert editor.is_showing_autocomplete() is False

    await wait_slow_debounce()
    await flush_autocomplete()

    assert len(suggestion_calls) == 1
    assert editor.is_showing_autocomplete() is True


@pytest.mark.tonio
async def test_debounces_custom_trigger_characters_autocomplete_while_typing(request):
    slow_debounce(request)
    editor = Editor(create_test_tui(), default_editor_theme)
    suggestion_calls = []

    async def get_suggestions(lines, cursor_line, cursor_col, options):
        suggestion_calls.append(1)
        prefix = (lines[0] if lines else "")[:cursor_col]
        return {"items": [{"value": "$skill-name", "label": "skill-name"}], "prefix": prefix}

    editor.set_autocomplete_provider(MockProvider(get_suggestions, trigger_characters=["$"]))

    await editor.handle_input("$")
    await editor.handle_input("s")
    await editor.handle_input("k")

    assert len(suggestion_calls) == 0
    await wait_slow_debounce()
    await flush_autocomplete()

    assert len(suggestion_calls) == 1
    assert editor.is_showing_autocomplete() is True


@pytest.mark.tonio
async def test_resets_custom_trigger_characters_when_provider_changes():
    editor = Editor(create_test_tui(), default_editor_theme)
    suggestion_calls = []

    async def first_get_suggestions(lines, cursor_line, cursor_col, options):
        return {"items": [{"value": "$skill-name", "label": "skill-name"}], "prefix": "$"}

    async def second_get_suggestions(lines, cursor_line, cursor_col, options):
        suggestion_calls.append(1)
        return {"items": [{"value": "$skill-name", "label": "skill-name"}], "prefix": "$"}

    editor.set_autocomplete_provider(MockProvider(first_get_suggestions, trigger_characters=["$"]))
    editor.set_autocomplete_provider(MockProvider(second_get_suggestions))

    await editor.handle_input("$")
    await editor.handle_input("s")
    await tonio.sleep(0.05)
    await flush_autocomplete()

    assert len(suggestion_calls) == 0
    assert editor.is_showing_autocomplete() is False


@pytest.mark.tonio
async def test_aborts_active_at_autocomplete_when_typing_continues():
    editor = Editor(create_test_tui(), default_editor_theme)
    aborts = []
    signals = []

    async def get_suggestions(lines, cursor_line, cursor_col, options):
        signals.append(options["signal"])
        await options["signal"].wait(0.5)
        if options["signal"].cancelled:
            aborts.append(options["signal"])
            return None
        return {"items": [{"value": "@main.ts", "label": "main.ts"}], "prefix": "@main"}

    editor.set_autocomplete_provider(MockProvider(get_suggestions))

    await editor.handle_input("@")
    await editor.handle_input("m")
    await editor.handle_input("a")
    await editor.handle_input("i")
    # Under load the 20ms debounce can fire between keystrokes, so how many
    # requests started (and were aborted by the next keystroke) is not fixed.
    # Each keystroke cancels the in-flight token synchronously, so the one
    # uncancelled signal is the request that survived typing: wait for it,
    # keep typing, and assert that exact request gets aborted.
    active = await poll_until(lambda: next((signal for signal in signals if not signal.cancelled), None))
    assert active is not None
    await editor.handle_input("n")

    assert await poll_until(lambda: active in aborts)


@pytest.mark.tonio
async def test_hides_autocomplete_when_backspacing_slash_command_to_empty():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Mock provider with slash commands
    async def get_suggestions(lines, cursor_line, cursor_col, options):
        text = lines[0] if lines else ""
        prefix = text[:cursor_col]
        # Only return slash command suggestions when line starts with /
        if prefix.startswith("/"):
            commands = [
                {"value": "/model", "label": "model", "description": "Change model"},
                {"value": "/help", "label": "help", "description": "Show help"},
            ]
            query = prefix[1:]  # Remove leading /
            filtered = [c for c in commands if c["value"].startswith(query)]
            if filtered:
                return {"items": filtered, "prefix": prefix}
        return None

    editor.set_autocomplete_provider(MockProvider(get_suggestions))

    # Type "/" - should show slash command suggestions
    await editor.handle_input("/")
    await flush_autocomplete()
    assert editor.get_text() == "/"
    assert editor.is_showing_autocomplete() is True

    # Backspace to delete "/" - should hide autocomplete completely
    await editor.handle_input("\x7f")  # Backspace
    await flush_autocomplete()
    assert editor.get_text() == ""
    assert editor.is_showing_autocomplete() is False


@pytest.mark.tonio
async def test_applies_exact_typed_slash_argument_value_on_enter_even_when_first_item_is_highlighted():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Mock provider for /argtest command with argument completions
    async def get_suggestions(lines, cursor_line, cursor_col, options):
        text = lines[0] if lines else ""
        before_cursor = text[:cursor_col]

        # Check if we're in argument completion context: "/argtest <prefix>"
        argtest_match = re.match(r"^/argtest\s+(\S+)$", before_cursor)
        if argtest_match:
            argument_text = argtest_match.group(1)
            all_arguments = [
                {"value": "one", "label": "one"},
                {"value": "two", "label": "two"},
                {"value": "three", "label": "three"},
            ]
            # Return all arguments that start with the typed prefix
            filtered = [arg for arg in all_arguments if arg["value"].startswith(argument_text)]
            if filtered:
                return {"items": filtered, "prefix": argument_text}
        return None

    editor.set_autocomplete_provider(MockProvider(get_suggestions))

    # Type "/argtest two"
    for ch in "/argtest two":
        await editor.handle_input(ch)

    assert editor.get_text() == "/argtest two"
    await flush_autocomplete()
    assert editor.is_showing_autocomplete() is True

    # Press Enter - should apply the exact typed value "two", not the first item
    await editor.handle_input("\r")

    # The exact typed value "two" should be retained
    assert editor.get_text() == "/argtest two"


@pytest.mark.tonio
async def test_selects_first_prefix_match_on_enter_when_typed_arg_is_not_exact_match():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Mock provider for /argtest command with argument completions
    async def get_suggestions(lines, cursor_line, cursor_col, options):
        text = lines[0] if lines else ""
        before_cursor = text[:cursor_col]

        # Check if we're in argument completion context
        argtest_match = re.match(r"^/argtest\s+(\S+)$", before_cursor)
        if argtest_match:
            argument_text = argtest_match.group(1)
            all_arguments = [
                {"value": "two", "label": "two"},
                {"value": "three", "label": "three"},
                {"value": "twelve", "label": "twelve"},
            ]
            # Return all items that start with the typed prefix
            filtered = [arg for arg in all_arguments if arg["value"].startswith(argument_text)]
            if filtered:
                return {"items": filtered, "prefix": argument_text}
        return None

    editor.set_autocomplete_provider(MockProvider(get_suggestions))

    # Type "/argtest t" - filtered to [two, three, twelve], prefix "t" matches "two" first
    for ch in "/argtest t":
        await editor.handle_input(ch)

    await flush_autocomplete()
    assert editor.is_showing_autocomplete() is True

    # Press Enter - "t" prefix matches "two" (first in list), so "two" is applied
    await editor.handle_input("\r")
    assert editor.get_text() == "/argtest two"


@pytest.mark.tonio
async def test_highlights_unique_prefix_match_as_user_types_before_full_exact_match():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Mock provider that returns all items unfiltered (like real extensions do)
    async def get_suggestions(lines, cursor_line, cursor_col, options):
        text = lines[0] if lines else ""
        before_cursor = text[:cursor_col]

        argtest_match = re.match(r"^/argtest\s+(\S+)$", before_cursor)
        if argtest_match:
            argument_text = argtest_match.group(1)
            # Return all items - provider does not filter
            all_arguments = [
                {"value": "one", "label": "one"},
                {"value": "two", "label": "two"},
                {"value": "three", "label": "three"},
            ]
            return {"items": all_arguments, "prefix": argument_text}
        return None

    editor.set_autocomplete_provider(MockProvider(get_suggestions))

    # Type "/argtest tw" - "tw" is a prefix of only "two"
    for ch in "/argtest tw":
        await editor.handle_input(ch)

    assert editor.get_text() == "/argtest tw"
    await flush_autocomplete()
    assert editor.is_showing_autocomplete() is True

    # Press Enter - "tw" uniquely matches "two", so "two" should be applied
    await editor.handle_input("\r")
    assert editor.get_text() == "/argtest two"


@pytest.mark.tonio
async def test_selects_first_prefix_match_when_multiple_items_match():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Mock provider that returns all items unfiltered
    async def get_suggestions(lines, cursor_line, cursor_col, options):
        text = lines[0] if lines else ""
        before_cursor = text[:cursor_col]

        argtest_match = re.match(r"^/argtest\s+(\S+)$", before_cursor)
        if argtest_match:
            argument_text = argtest_match.group(1)
            all_arguments = [
                {"value": "one", "label": "one"},
                {"value": "two", "label": "two"},
                {"value": "three", "label": "three"},
            ]
            return {"items": all_arguments, "prefix": argument_text}
        return None

    editor.set_autocomplete_provider(MockProvider(get_suggestions))

    # Type "/argtest t" - "t" is a prefix of both "two" and "three"
    for ch in "/argtest t":
        await editor.handle_input(ch)

    await flush_autocomplete()
    assert editor.is_showing_autocomplete() is True

    # Press Enter - "t" matches "two" first, so "two" is selected
    await editor.handle_input("\r")
    assert editor.get_text() == "/argtest two"


@pytest.mark.tonio
async def test_works_for_built_in_style_command_argument_completion_path_model_like():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Mock provider for /model command with model completions
    async def get_suggestions(lines, cursor_line, cursor_col, options):
        text = lines[0] if lines else ""
        before_cursor = text[:cursor_col]

        # Check if we're in /model argument completion context
        model_match = re.match(r"^/model\s+(\S+)$", before_cursor)
        if model_match:
            model_text = model_match.group(1)
            all_models = [
                {"value": "gpt-4o", "label": "gpt-4o"},
                {"value": "gpt-4o-mini", "label": "gpt-4o-mini"},
                {"value": "claude-sonnet", "label": "claude-sonnet"},
            ]
            # Return all models that start with the typed prefix
            filtered = [m for m in all_models if m["value"].startswith(model_text)]
            if filtered:
                return {"items": filtered, "prefix": model_text}
        return None

    editor.set_autocomplete_provider(MockProvider(get_suggestions))

    # Type "/model gpt-4o-mini" - exact match for second item in list
    for ch in "/model gpt-4o-mini":
        await editor.handle_input(ch)

    assert editor.get_text() == "/model gpt-4o-mini"
    await flush_autocomplete()
    assert editor.is_showing_autocomplete() is True

    # Press Enter - should retain exact typed value, not apply first highlighted item
    await editor.handle_input("\r")

    # The exact typed value should be retained
    assert editor.get_text() == "/model gpt-4o-mini"


@pytest.mark.tonio
async def test_awaits_async_slash_command_argument_completions(tmp_path):
    editor = Editor(create_test_tui(), default_editor_theme)

    async def get_argument_completions(prefix):
        return [{"value": "skill-a", "label": "skill-a"}] if prefix.startswith("s") else None

    provider = CombinedAutocompleteProvider(
        [{"name": "load-skills", "description": "Load skills", "getArgumentCompletions": get_argument_completions}],
        str(tmp_path),
    )
    editor.set_autocomplete_provider(provider)
    editor.set_text("/load-skills ")

    await editor.handle_input("s")
    await flush_autocomplete()
    assert editor.is_showing_autocomplete() is True

    await editor.handle_input("\t")
    assert editor.get_text() == "/load-skills skill-a"
    assert editor.is_showing_autocomplete() is False


@pytest.mark.tonio
async def test_ignores_invalid_slash_command_argument_completion_results(tmp_path):
    editor = Editor(create_test_tui(), default_editor_theme)

    async def get_argument_completions(prefix):
        return "not-an-array"

    provider = CombinedAutocompleteProvider(
        [
            {
                "name": "load-skills",
                "description": "Load skills",
                "getArgumentCompletions": get_argument_completions,
            }
        ],
        str(tmp_path),
    )
    editor.set_autocomplete_provider(provider)
    editor.set_text("/load-skills ")

    await editor.handle_input("s")
    await flush_autocomplete()
    assert editor.is_showing_autocomplete() is False
    assert editor.get_text() == "/load-skills s"


@pytest.mark.tonio
async def test_does_not_show_argument_completions_when_command_has_no_argument_completer(tmp_path):
    editor = Editor(create_test_tui(), default_editor_theme)

    async def get_model_completions(prefix):
        return [{"value": "claude-opus", "label": "claude-opus"}]

    provider = CombinedAutocompleteProvider(
        [
            {"name": "help", "description": "Show help"},
            {
                "name": "model",
                "description": "Switch model",
                "getArgumentCompletions": get_model_completions,
            },
        ],
        str(tmp_path),
    )
    editor.set_autocomplete_provider(provider)

    await editor.handle_input("/")
    await editor.handle_input("h")
    await editor.handle_input("e")
    await flush_autocomplete()
    assert editor.is_showing_autocomplete() is True

    await editor.handle_input("\t")
    assert editor.get_text() == "/help "
    assert editor.is_showing_autocomplete() is False


# Character jump (Ctrl+])


@pytest.mark.tonio
async def test_jumps_forward_to_first_occurrence_of_character_on_same_line():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("hello world")
    await editor.handle_input("\x01")  # Ctrl+A - go to start
    assert editor.get_cursor() == {"line": 0, "col": 0}

    await editor.handle_input("\x1d")  # Ctrl+] (legacy sequence for ctrl+])
    await editor.handle_input("o")  # Jump to first 'o'

    assert editor.get_cursor() == {"line": 0, "col": 4}  # 'o' in "hello"


@pytest.mark.tonio
async def test_jumps_forward_to_next_occurrence_after_cursor():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("hello world")
    await editor.handle_input("\x01")  # Ctrl+A - go to start
    # Move cursor to the 'o' in "hello" (col 4)
    for _ in range(4):
        await editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": 4}

    await editor.handle_input("\x1d")  # Ctrl+]
    await editor.handle_input("o")  # Jump to next 'o' (in "world")

    assert editor.get_cursor() == {"line": 0, "col": 7}  # 'o' in "world"


@pytest.mark.tonio
async def test_jumps_forward_across_multiple_lines():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("abc\ndef\nghi")
    # Cursor is at end (line 2, col 3). Move to line 0 via up arrows, then Ctrl+A
    await editor.handle_input("\x1b[A")  # Up
    await editor.handle_input("\x1b[A")  # Up - now on line 0
    await editor.handle_input("\x01")  # Ctrl+A - go to start of line
    assert editor.get_cursor() == {"line": 0, "col": 0}

    await editor.handle_input("\x1d")  # Ctrl+]
    await editor.handle_input("g")  # Jump to 'g' on line 3

    assert editor.get_cursor() == {"line": 2, "col": 0}


@pytest.mark.tonio
async def test_jumps_backward_to_first_occurrence_before_cursor_on_same_line():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("hello world")
    # Cursor at end (col 11)
    assert editor.get_cursor() == {"line": 0, "col": 11}

    await editor.handle_input("\x1b\x1d")  # Ctrl+Alt+] (ESC followed by Ctrl+])
    await editor.handle_input("o")  # Jump to last 'o' before cursor

    assert editor.get_cursor() == {"line": 0, "col": 7}  # 'o' in "world"


@pytest.mark.tonio
async def test_jumps_backward_across_multiple_lines():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("abc\ndef\nghi")
    # Cursor at end of line 3
    assert editor.get_cursor() == {"line": 2, "col": 3}

    await editor.handle_input("\x1b\x1d")  # Ctrl+Alt+]
    await editor.handle_input("a")  # Jump to 'a' on line 1

    assert editor.get_cursor() == {"line": 0, "col": 0}


@pytest.mark.tonio
async def test_does_nothing_when_character_is_not_found_forward():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("hello world")
    await editor.handle_input("\x01")  # Ctrl+A - go to start
    assert editor.get_cursor() == {"line": 0, "col": 0}

    await editor.handle_input("\x1d")  # Ctrl+]
    await editor.handle_input("z")  # 'z' doesn't exist

    assert editor.get_cursor() == {"line": 0, "col": 0}  # Cursor unchanged


@pytest.mark.tonio
async def test_does_nothing_when_character_is_not_found_backward():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("hello world")
    # Cursor at end
    assert editor.get_cursor() == {"line": 0, "col": 11}

    await editor.handle_input("\x1b\x1d")  # Ctrl+Alt+]
    await editor.handle_input("z")  # 'z' doesn't exist

    assert editor.get_cursor() == {"line": 0, "col": 11}  # Cursor unchanged


@pytest.mark.tonio
async def test_jump_is_case_sensitive():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("Hello World")
    await editor.handle_input("\x01")  # Ctrl+A - go to start
    assert editor.get_cursor() == {"line": 0, "col": 0}

    # Search for lowercase 'h' - should not find it (only 'H' exists)
    await editor.handle_input("\x1d")  # Ctrl+]
    await editor.handle_input("h")

    assert editor.get_cursor() == {"line": 0, "col": 0}  # Cursor unchanged

    # Search for uppercase 'W' - should find it
    await editor.handle_input("\x1d")  # Ctrl+]
    await editor.handle_input("W")

    assert editor.get_cursor() == {"line": 0, "col": 6}  # 'W' in "World"


@pytest.mark.tonio
async def test_cancels_jump_mode_when_ctrl_bracket_is_pressed_again():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("hello world")
    await editor.handle_input("\x01")  # Ctrl+A - go to start
    assert editor.get_cursor() == {"line": 0, "col": 0}

    await editor.handle_input("\x1d")  # Ctrl+] - enter jump mode
    await editor.handle_input("\x1d")  # Ctrl+] again - cancel

    # Type 'o' normally - should insert, not jump
    await editor.handle_input("o")
    assert editor.get_text() == "ohello world"


@pytest.mark.tonio
async def test_cancels_jump_mode_on_escape_and_processes_the_escape():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("hello world")
    await editor.handle_input("\x01")  # Ctrl+A - go to start
    assert editor.get_cursor() == {"line": 0, "col": 0}

    await editor.handle_input("\x1d")  # Ctrl+] - enter jump mode
    await editor.handle_input("\x1b")  # Escape - cancel jump mode

    # Cursor should be unchanged (Escape itself doesn't move cursor in editor)
    assert editor.get_cursor() == {"line": 0, "col": 0}

    # Type 'o' normally - should insert, not jump
    await editor.handle_input("o")
    assert editor.get_text() == "ohello world"


@pytest.mark.tonio
async def test_cancels_backward_jump_mode_when_ctrl_alt_bracket_is_pressed_again():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("hello world")
    # Cursor at end
    assert editor.get_cursor() == {"line": 0, "col": 11}

    await editor.handle_input("\x1b\x1d")  # Ctrl+Alt+] - enter backward jump mode
    await editor.handle_input("\x1b\x1d")  # Ctrl+Alt+] again - cancel

    # Type 'o' normally - should insert, not jump
    await editor.handle_input("o")
    assert editor.get_text() == "hello worldo"


@pytest.mark.tonio
async def test_jump_searches_for_special_characters():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("foo(bar) = baz;")
    await editor.handle_input("\x01")  # Ctrl+A - go to start
    assert editor.get_cursor() == {"line": 0, "col": 0}

    # Jump to '('
    await editor.handle_input("\x1d")  # Ctrl+]
    await editor.handle_input("(")

    assert editor.get_cursor() == {"line": 0, "col": 3}

    # Jump to '='
    await editor.handle_input("\x1d")  # Ctrl+]
    await editor.handle_input("=")

    assert editor.get_cursor() == {"line": 0, "col": 9}


@pytest.mark.tonio
async def test_jump_handles_empty_text_gracefully():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("")
    assert editor.get_cursor() == {"line": 0, "col": 0}

    await editor.handle_input("\x1d")  # Ctrl+]
    await editor.handle_input("x")

    assert editor.get_cursor() == {"line": 0, "col": 0}  # Cursor unchanged


@pytest.mark.tonio
async def test_resets_last_action_when_jumping():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("hello world")
    await editor.handle_input("\x01")  # Ctrl+A - go to start

    # Type to set last action to "type-word"
    await editor.handle_input("x")
    assert editor.get_text() == "xhello world"

    # Jump forward
    await editor.handle_input("\x1d")  # Ctrl+]
    await editor.handle_input("o")

    # Type more - should start a new undo unit (last action was reset)
    await editor.handle_input("Y")
    assert editor.get_text() == "xhellYo world"

    # Undo should only undo "Y", not "x" as well
    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "xhello world"


# Sticky column


async def _position_cursor(editor, line, col):
    """Position cursor at a specific line and column."""
    # Go to line 0 first
    for _ in range(20):
        await editor.handle_input("\x1b[A")
    # Go to target line
    for _ in range(line):
        await editor.handle_input("\x1b[B")
    # Go to target col
    await editor.handle_input("\x01")  # Ctrl+A
    for _ in range(col):
        await editor.handle_input("\x1b[C")


@pytest.mark.tonio
async def test_preserves_target_column_when_moving_up_through_a_shorter_line():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Line 0: "2222222222x222" (x at col 10)
    # Line 1: "" (empty)
    # Line 2: "1111111111_111111111111" (_ at col 10)
    editor.set_text("2222222222x222\n\n1111111111_111111111111")

    # Position cursor on _ (line 2, col 10)
    assert editor.get_cursor() == {"line": 2, "col": 23}  # At end
    await editor.handle_input("\x01")  # Ctrl+A - go to start of line
    for _ in range(10):
        await editor.handle_input("\x1b[C")  # Move right to col 10
    assert editor.get_cursor() == {"line": 2, "col": 10}

    # Press Up - should move to empty line (col clamped to 0)
    await editor.handle_input("\x1b[A")  # Up arrow
    assert editor.get_cursor() == {"line": 1, "col": 0}

    # Press Up again - should move to line 0 at col 10 (on 'x')
    await editor.handle_input("\x1b[A")  # Up arrow
    assert editor.get_cursor() == {"line": 0, "col": 10}


@pytest.mark.tonio
async def test_preserves_target_column_when_moving_down_through_a_shorter_line():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("1111111111_111\n\n2222222222x222222222222")

    # Position cursor on _ (line 0, col 10)
    await editor.handle_input("\x1b[A")  # Up to line 1
    await editor.handle_input("\x1b[A")  # Up to line 0
    await editor.handle_input("\x01")  # Ctrl+A
    for _ in range(10):
        await editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": 10}

    # Press Down - should move to empty line (col clamped to 0)
    await editor.handle_input("\x1b[B")  # Down arrow
    assert editor.get_cursor() == {"line": 1, "col": 0}

    # Press Down again - should move to line 2 at col 10 (on 'x')
    await editor.handle_input("\x1b[B")  # Down arrow
    assert editor.get_cursor() == {"line": 2, "col": 10}


@pytest.mark.tonio
async def test_resets_sticky_column_on_horizontal_movement_left_arrow():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("1234567890\n\n1234567890")

    # Start at line 2, col 5
    await editor.handle_input("\x01")  # Ctrl+A
    for _ in range(5):
        await editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 2, "col": 5}

    # Move up through empty line
    await editor.handle_input("\x1b[A")  # Up - line 1, col 0
    await editor.handle_input("\x1b[A")  # Up - line 0, col 5 (sticky)
    assert editor.get_cursor() == {"line": 0, "col": 5}

    # Move left - resets sticky column
    await editor.handle_input("\x1b[D")  # Left
    assert editor.get_cursor() == {"line": 0, "col": 4}

    # Move down twice
    await editor.handle_input("\x1b[B")  # Down - line 1, col 0
    await editor.handle_input("\x1b[B")  # Down - line 2, col 4 (new sticky from col 4)
    assert editor.get_cursor() == {"line": 2, "col": 4}


@pytest.mark.tonio
async def test_resets_sticky_column_on_horizontal_movement_right_arrow():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("1234567890\n\n1234567890")

    # Start at line 0, col 5
    await editor.handle_input("\x1b[A")  # Up to line 1
    await editor.handle_input("\x1b[A")  # Up to line 0
    await editor.handle_input("\x01")  # Ctrl+A
    for _ in range(5):
        await editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": 5}

    # Move down through empty line
    await editor.handle_input("\x1b[B")  # Down - line 1, col 0
    await editor.handle_input("\x1b[B")  # Down - line 2, col 5 (sticky)
    assert editor.get_cursor() == {"line": 2, "col": 5}

    # Move right - resets sticky column
    await editor.handle_input("\x1b[C")  # Right
    assert editor.get_cursor() == {"line": 2, "col": 6}

    # Move up twice
    await editor.handle_input("\x1b[A")  # Up - line 1, col 0
    await editor.handle_input("\x1b[A")  # Up - line 0, col 6 (new sticky from col 6)
    assert editor.get_cursor() == {"line": 0, "col": 6}


@pytest.mark.tonio
async def test_resets_sticky_column_on_typing():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("1234567890\n\n1234567890")

    # Start at line 2, col 8
    await editor.handle_input("\x01")  # Ctrl+A
    for _ in range(8):
        await editor.handle_input("\x1b[C")

    # Move up through empty line
    await editor.handle_input("\x1b[A")  # Up
    await editor.handle_input("\x1b[A")  # Up - line 0, col 8
    assert editor.get_cursor() == {"line": 0, "col": 8}

    # Type a character - resets sticky column
    await editor.handle_input("X")
    assert editor.get_cursor() == {"line": 0, "col": 9}

    # Move down twice
    await editor.handle_input("\x1b[B")  # Down - line 1, col 0
    await editor.handle_input("\x1b[B")  # Down - line 2, col 9 (new sticky from col 9)
    assert editor.get_cursor() == {"line": 2, "col": 9}


@pytest.mark.tonio
async def test_resets_sticky_column_on_backspace():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("1234567890\n\n1234567890")

    # Start at line 2, col 8
    await editor.handle_input("\x01")  # Ctrl+A
    for _ in range(8):
        await editor.handle_input("\x1b[C")

    # Move up through empty line
    await editor.handle_input("\x1b[A")  # Up
    await editor.handle_input("\x1b[A")  # Up - line 0, col 8
    assert editor.get_cursor() == {"line": 0, "col": 8}

    # Backspace - resets sticky column
    await editor.handle_input("\x7f")  # Backspace
    assert editor.get_cursor() == {"line": 0, "col": 7}

    # Move down twice
    await editor.handle_input("\x1b[B")  # Down - line 1, col 0
    await editor.handle_input("\x1b[B")  # Down - line 2, col 7 (new sticky from col 7)
    assert editor.get_cursor() == {"line": 2, "col": 7}


@pytest.mark.tonio
async def test_resets_sticky_column_on_ctrl_a_move_to_line_start():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("1234567890\n\n1234567890")

    # Start at line 2, col 8
    await editor.handle_input("\x01")  # Ctrl+A
    for _ in range(8):
        await editor.handle_input("\x1b[C")

    # Move up - establishes sticky col 8
    await editor.handle_input("\x1b[A")  # Up - line 1, col 0

    # Ctrl+A - resets sticky column to 0
    await editor.handle_input("\x01")  # Ctrl+A
    assert editor.get_cursor() == {"line": 1, "col": 0}

    # Move up
    await editor.handle_input("\x1b[A")  # Up - line 0, col 0 (new sticky from col 0)
    assert editor.get_cursor() == {"line": 0, "col": 0}


@pytest.mark.tonio
async def test_resets_sticky_column_on_ctrl_e_move_to_line_end():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("12345\n\n1234567890")

    # Start at line 2, col 3
    await editor.handle_input("\x01")  # Ctrl+A
    for _ in range(3):
        await editor.handle_input("\x1b[C")

    # Move up through empty line - establishes sticky col 3
    await editor.handle_input("\x1b[A")  # Up - line 1, col 0
    await editor.handle_input("\x1b[A")  # Up - line 0, col 3
    assert editor.get_cursor() == {"line": 0, "col": 3}

    # Ctrl+E - resets sticky column to end
    await editor.handle_input("\x05")  # Ctrl+E
    assert editor.get_cursor() == {"line": 0, "col": 5}

    # Move down twice
    await editor.handle_input("\x1b[B")  # Down - line 1, col 0
    await editor.handle_input("\x1b[B")  # Down - line 2, col 5 (new sticky from col 5)
    assert editor.get_cursor() == {"line": 2, "col": 5}


@pytest.mark.tonio
async def test_resets_sticky_column_on_word_movement_ctrl_left():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("hello world\n\nhello world")

    # Start at end of line 2 (col 11)
    assert editor.get_cursor() == {"line": 2, "col": 11}

    # Move up through empty line - establishes sticky col 11
    await editor.handle_input("\x1b[A")  # Up - line 1, col 0
    await editor.handle_input("\x1b[A")  # Up - line 0, col 11
    assert editor.get_cursor() == {"line": 0, "col": 11}

    # Ctrl+Left - word movement resets sticky column
    await editor.handle_input("\x1b[1;5D")  # Ctrl+Left
    assert editor.get_cursor() == {"line": 0, "col": 6}  # Before "world"

    # Move down twice
    await editor.handle_input("\x1b[B")  # Down - line 1, col 0
    await editor.handle_input("\x1b[B")  # Down - line 2, col 6 (new sticky from col 6)
    assert editor.get_cursor() == {"line": 2, "col": 6}


@pytest.mark.tonio
async def test_resets_sticky_column_on_word_movement_ctrl_right():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("hello world\n\nhello world")

    # Start at line 0, col 0
    await editor.handle_input("\x1b[A")  # Up
    await editor.handle_input("\x1b[A")  # Up
    await editor.handle_input("\x01")  # Ctrl+A
    assert editor.get_cursor() == {"line": 0, "col": 0}

    # Move down through empty line - establishes sticky col 0
    await editor.handle_input("\x1b[B")  # Down - line 1, col 0
    await editor.handle_input("\x1b[B")  # Down - line 2, col 0
    assert editor.get_cursor() == {"line": 2, "col": 0}

    # Ctrl+Right - word movement resets sticky column
    await editor.handle_input("\x1b[1;5C")  # Ctrl+Right
    assert editor.get_cursor() == {"line": 2, "col": 5}  # After "hello"

    # Move up twice
    await editor.handle_input("\x1b[A")  # Up - line 1, col 0
    await editor.handle_input("\x1b[A")  # Up - line 0, col 5 (new sticky from col 5)
    assert editor.get_cursor() == {"line": 0, "col": 5}


@pytest.mark.tonio
async def test_resets_sticky_column_on_undo():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("1234567890\n\n1234567890")

    # Go to line 0, col 8
    await editor.handle_input("\x1b[A")  # Up to line 1
    await editor.handle_input("\x1b[A")  # Up to line 0
    await editor.handle_input("\x01")  # Ctrl+A
    for _ in range(8):
        await editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": 8}

    # Move down through empty line - establishes sticky col 8
    await editor.handle_input("\x1b[B")  # Down - line 1, col 0
    await editor.handle_input("\x1b[B")  # Down - line 2, col 8 (sticky)
    assert editor.get_cursor() == {"line": 2, "col": 8}

    # Type something to create undo state - this clears sticky and sets col to 9
    await editor.handle_input("X")
    assert editor.get_text() == "1234567890\n\n12345678X90"
    assert editor.get_cursor() == {"line": 2, "col": 9}

    # Move up - establishes new sticky col 9
    await editor.handle_input("\x1b[A")  # Up - line 1, col 0
    await editor.handle_input("\x1b[A")  # Up - line 0, col 9
    assert editor.get_cursor() == {"line": 0, "col": 9}

    # Undo - resets sticky column and restores cursor to line 2, col 8
    await editor.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
    assert editor.get_text() == "1234567890\n\n1234567890"
    assert editor.get_cursor() == {"line": 2, "col": 8}

    # Move up - should capture new sticky from restored col 8, not old col 9
    await editor.handle_input("\x1b[A")  # Up - line 1, col 0
    await editor.handle_input("\x1b[A")  # Up - line 0, col 8 (new sticky from restored position)
    assert editor.get_cursor() == {"line": 0, "col": 8}


@pytest.mark.tonio
async def test_handles_multiple_consecutive_up_down_movements():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("1234567890\nab\ncd\nef\n1234567890")

    # Start at line 4, col 7
    await editor.handle_input("\x01")  # Ctrl+A
    for _ in range(7):
        await editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 4, "col": 7}

    # Move up multiple times through short lines
    await editor.handle_input("\x1b[A")  # Up - line 3, col 2 (clamped)
    await editor.handle_input("\x1b[A")  # Up - line 2, col 2 (clamped)
    await editor.handle_input("\x1b[A")  # Up - line 1, col 2 (clamped)
    await editor.handle_input("\x1b[A")  # Up - line 0, col 7 (restored)
    assert editor.get_cursor() == {"line": 0, "col": 7}

    # Move down multiple times - sticky should still be 7
    await editor.handle_input("\x1b[B")  # Down - line 1, col 2
    await editor.handle_input("\x1b[B")  # Down - line 2, col 2
    await editor.handle_input("\x1b[B")  # Down - line 3, col 2
    await editor.handle_input("\x1b[B")  # Down - line 4, col 7 (restored)
    assert editor.get_cursor() == {"line": 4, "col": 7}


@pytest.mark.tonio
async def test_moves_correctly_through_wrapped_visual_lines_without_getting_stuck():
    tui = create_test_tui(15, 24)  # Narrow terminal
    editor = Editor(tui, default_editor_theme)

    # Line 0: short
    # Line 1: 30 chars = wraps to 3 visual lines at width 10 (after padding)
    editor.set_text("short\n123456789012345678901234567890")
    editor.render(15)  # This gives 14 layout width

    # Position at end of line 1 (col 30)
    assert editor.get_cursor() == {"line": 1, "col": 30}

    # Move up repeatedly - should traverse all visual lines of the wrapped text
    # and eventually reach line 0
    await editor.handle_input("\x1b[A")  # Up - to previous visual line within line 1
    assert editor.get_cursor()["line"] == 1

    await editor.handle_input("\x1b[A")  # Up - another visual line
    assert editor.get_cursor()["line"] == 1

    await editor.handle_input("\x1b[A")  # Up - should reach line 0
    assert editor.get_cursor()["line"] == 0


@pytest.mark.tonio
async def test_handles_set_text_resetting_sticky_column():
    editor = Editor(create_test_tui(), default_editor_theme)

    editor.set_text("1234567890\n\n1234567890")

    # Establish sticky column
    await editor.handle_input("\x01")  # Ctrl+A
    for _ in range(8):
        await editor.handle_input("\x1b[C")
    await editor.handle_input("\x1b[A")  # Up

    # set_text should reset sticky column
    editor.set_text("abcdefghij\n\nabcdefghij")
    assert editor.get_cursor() == {"line": 2, "col": 10}  # At end

    # Move up - should capture new sticky from current position (10)
    await editor.handle_input("\x1b[A")  # Up - line 1, col 0
    await editor.handle_input("\x1b[A")  # Up - line 0, col 10
    assert editor.get_cursor() == {"line": 0, "col": 10}


@pytest.mark.tonio
async def test_sets_preferred_visual_col_when_pressing_right_at_end_of_prompt_last_line():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Line 0: 20 chars with 'x' at col 10
    # Line 1: empty
    # Line 2: 10 chars ending with '_'
    editor.set_text("111111111x1111111111\n\n333333333_")

    # Go to line 0, press Ctrl+E (end of line) - col 20
    await editor.handle_input("\x1b[A")  # Up to line 1
    await editor.handle_input("\x1b[A")  # Up to line 0
    await editor.handle_input("\x05")  # Ctrl+E - move to end of line
    assert editor.get_cursor() == {"line": 0, "col": 20}

    # Move down to line 2 - cursor clamped to col 10 (end of line)
    await editor.handle_input("\x1b[B")  # Down to line 1, col 0
    await editor.handle_input("\x1b[B")  # Down to line 2, col 10 (clamped)
    assert editor.get_cursor() == {"line": 2, "col": 10}

    # Press Right at end of prompt - nothing visible happens, but sets preferred visual col to 10
    await editor.handle_input("\x1b[C")  # Right - can't move, but sets preferred visual col
    assert editor.get_cursor() == {"line": 2, "col": 10}  # Still at same position

    # Move up twice to line 0 - should use preferred visual col (10) to land on 'x'
    await editor.handle_input("\x1b[A")  # Up to line 1, col 0
    await editor.handle_input("\x1b[A")  # Up to line 0, col 10 (on 'x')
    assert editor.get_cursor() == {"line": 0, "col": 10}


@pytest.mark.tonio
async def test_handles_editor_resizes_when_preferred_visual_col_is_on_the_same_line():
    # Create editor with wider terminal
    tui = create_test_tui(80, 24)
    editor = Editor(tui, default_editor_theme)

    editor.set_text("12345678901234567890\n\n12345678901234567890")

    # Start at line 2, col 15
    await editor.handle_input("\x01")  # Ctrl+A
    for _ in range(15):
        await editor.handle_input("\x1b[C")

    # Move up through empty line - establishes sticky col 15
    await editor.handle_input("\x1b[A")  # Up
    await editor.handle_input("\x1b[A")  # Up - line 0, col 15
    assert editor.get_cursor() == {"line": 0, "col": 15}

    # Render with narrower width to simulate resize
    editor.render(12)  # Width 12

    # Move down - sticky should be clamped to new width
    await editor.handle_input("\x1b[B")  # Down - line 1
    await editor.handle_input("\x1b[B")  # Down - line 2, col should be clamped
    assert editor.get_cursor()["col"] == 4


@pytest.mark.tonio
async def test_handles_editor_resizes_when_preferred_visual_col_is_on_a_different_line():
    tui = create_test_tui(80, 24)
    editor = Editor(tui, default_editor_theme)

    # Create a line that wraps into multiple visual lines at width 10
    # "12345678901234567890" = 20 chars, wraps to 2 visual lines at width 10
    editor.set_text("short\n12345678901234567890")

    # Go to line 1, col 15
    await editor.handle_input("\x01")  # Ctrl+A
    for _ in range(15):
        await editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 1, "col": 15}

    # Move up to establish sticky col 15
    await editor.handle_input("\x1b[A")  # Up to line 0
    # Line 0 has only 5 chars, so cursor at col 5
    assert editor.get_cursor() == {"line": 0, "col": 5}

    # Narrow the editor
    editor.render(10)

    # Move down - preferred visual col was 15, but width is 10
    # Should land on line 1, clamped to width (visual col 9, which is logical col 9)
    await editor.handle_input("\x1b[B")  # Down to line 1
    assert editor.get_cursor() == {"line": 1, "col": 8}

    # Move up
    await editor.handle_input("\x1b[A")  # Up - should go to line 0
    assert editor.get_cursor() == {"line": 0, "col": 5}  # Line 0 only has 5 chars

    # Restore the original width
    editor.render(80)

    # Move down - preferred visual col was kept at 15
    await editor.handle_input("\x1b[B")  # Down to line 1
    assert editor.get_cursor() == {"line": 1, "col": 15}


@pytest.mark.tonio
async def test_rewrapped_lines_target_fits_current_visual_column():
    tui = create_test_tui(80, 24)
    editor = Editor(tui, default_editor_theme)
    editor.set_text("abcdefghijklmnopqr\n123456789012345678")

    await _position_cursor(editor, 0, 18)
    assert editor.get_cursor() == {"line": 0, "col": 18}

    # Narrow to width 10 (layoutWidth = 9).
    # Line 0 last segment has visual col max 9, line 1 first segment max 8
    editor.render(10)

    # Move down: cursor clamps to 8
    await editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 1, "col": 8}

    # Widen back. Move up, the current visual col wins
    editor.render(80)
    await editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 8}

    # Preferred was cleared by the rewrapped branch
    await editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 1, "col": 8}


@pytest.mark.tonio
async def test_rewrapped_lines_target_shorter_than_current_visual_column():
    tui = create_test_tui(80, 24)
    editor = Editor(tui, default_editor_theme)
    editor.set_text("abcdefghijklmnopqr\n123456789012345678\nab")

    await _position_cursor(editor, 0, 18)
    assert editor.get_cursor() == {"line": 0, "col": 18}

    # Narrow to width 10 (layoutWidth = 9). Moving down clamps to col 8
    editor.render(10)
    await editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 1, "col": 8}

    # Widen the editor
    editor.render(80)

    # Move down to short line "ab".
    # preferred visual col is replaced with current visual col (8), cursor clamps to 2
    await editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 2, "col": 2}

    # Moving up restores to preferred col 8
    await editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 1, "col": 8}


# Paste marker atomic behavior


async def _paste_with_marker(editor):
    """Simulate a large paste that creates a marker."""
    big_content = ("line\n" * 20).rstrip()  # 20 lines
    await editor.handle_input(f"\x1b[200~{big_content}\x1b[201~")
    # The editor replaces large pastes with a marker like "[paste #1 +20 lines]"
    return editor.get_text()


def _big_paste(tag):
    """12-line paste content with a distinguishing tag."""
    return "\n".join(f"{tag}{i}" for i in range(12))


@pytest.mark.tonio
async def test_creates_a_paste_marker_for_large_pastes():
    editor = Editor(create_test_tui(), default_editor_theme)
    text = await _paste_with_marker(editor)
    assert re.search(r"\[paste #\d+ \+\d+ lines\]", text)


@pytest.mark.tonio
async def test_treats_paste_marker_as_single_unit_for_right_arrow():
    editor = Editor(create_test_tui(), default_editor_theme)
    await editor.handle_input("A")
    await _paste_with_marker(editor)
    await editor.handle_input("B")
    # Text: "A[paste #1 +20 lines]B", cursor at end

    # Go to start
    await editor.handle_input("\x01")  # Ctrl+A
    assert editor.get_cursor() == {"line": 0, "col": 0}

    # Right arrow: should move past "A"
    await editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": 1}

    # Right arrow: should skip the entire marker
    await editor.handle_input("\x1b[C")
    marker = re.search(r"\[paste #\d+ \+\d+ lines\]", editor.get_text()).group(0)
    assert editor.get_cursor() == {"line": 0, "col": 1 + len(marker)}

    # Right arrow: should move past "B"
    await editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": 1 + len(marker) + 1}


@pytest.mark.tonio
async def test_treats_paste_marker_as_single_unit_for_left_arrow():
    editor = Editor(create_test_tui(), default_editor_theme)
    await editor.handle_input("A")
    await _paste_with_marker(editor)
    await editor.handle_input("B")
    # Cursor at end

    # Left arrow: past "B"
    await editor.handle_input("\x1b[D")
    text = editor.get_text()
    marker = re.search(r"\[paste #\d+ \+\d+ lines\]", text).group(0)
    assert editor.get_cursor() == {"line": 0, "col": 1 + len(marker)}

    # Left arrow: skip the entire marker
    await editor.handle_input("\x1b[D")
    assert editor.get_cursor() == {"line": 0, "col": 1}

    # Left arrow: past "A"
    await editor.handle_input("\x1b[D")
    assert editor.get_cursor() == {"line": 0, "col": 0}


@pytest.mark.tonio
async def test_treats_paste_marker_as_single_unit_for_backspace():
    editor = Editor(create_test_tui(), default_editor_theme)
    await editor.handle_input("A")
    await _paste_with_marker(editor)
    await editor.handle_input("B")

    text = editor.get_text()
    marker = re.search(r"\[paste #\d+ \+\d+ lines\]", text).group(0)

    # Position cursor right after the marker (before "B")
    await editor.handle_input("\x01")  # Ctrl+A
    # Move past "A" and the marker
    await editor.handle_input("\x1b[C")  # past "A"
    await editor.handle_input("\x1b[C")  # past marker
    assert editor.get_cursor() == {"line": 0, "col": 1 + len(marker)}

    # Backspace: should delete the entire marker at once
    await editor.handle_input("\x7f")
    assert editor.get_text() == "AB"
    assert editor.get_cursor() == {"line": 0, "col": 1}


@pytest.mark.tonio
async def test_treats_paste_marker_as_single_unit_for_forward_delete():
    editor = Editor(create_test_tui(), default_editor_theme)
    await editor.handle_input("A")
    await _paste_with_marker(editor)
    await editor.handle_input("B")

    # Position cursor on "A" (col 0) then move right once to be just before marker
    await editor.handle_input("\x01")  # Ctrl+A
    await editor.handle_input("\x1b[C")  # past "A", now at col 1 (start of marker)

    # Forward delete: should delete the entire marker at once
    await editor.handle_input("\x1b[3~")  # Delete key
    assert editor.get_text() == "AB"
    assert editor.get_cursor() == {"line": 0, "col": 1}


@pytest.mark.tonio
async def test_treats_paste_marker_as_single_unit_for_word_movement():
    editor = Editor(create_test_tui(), default_editor_theme)
    await editor.handle_input("X")
    await editor.handle_input(" ")
    await _paste_with_marker(editor)
    await editor.handle_input(" ")
    await editor.handle_input("Y")
    # Text: "X [paste #1 +20 lines] Y"

    text = editor.get_text()
    marker = re.search(r"\[paste #\d+ \+\d+ lines\]", text).group(0)

    # Go to start
    await editor.handle_input("\x01")  # Ctrl+A

    # Ctrl+Right: skip "X"
    await editor.handle_input("\x1b[1;5C")
    assert editor.get_cursor() == {"line": 0, "col": 1}

    # Ctrl+Right: skip whitespace + marker (marker treated as single non-ws, non-punct unit)
    await editor.handle_input("\x1b[1;5C")
    assert editor.get_cursor() == {"line": 0, "col": 2 + len(marker)}


@pytest.mark.tonio
async def test_undo_restores_marker_after_backspace_deletion():
    editor = Editor(create_test_tui(), default_editor_theme)
    await editor.handle_input("A")
    await _paste_with_marker(editor)
    await editor.handle_input("B")

    text_before = editor.get_text()

    # Position after marker
    await editor.handle_input("\x01")
    await editor.handle_input("\x1b[C")  # past A
    await editor.handle_input("\x1b[C")  # past marker

    # Delete marker
    await editor.handle_input("\x7f")
    assert editor.get_text() == "AB"

    # Undo
    await editor.handle_input("\x1b[45;5u")
    assert editor.get_text() == text_before


@pytest.mark.tonio
async def test_undo_after_paste_marker_deletion_restores_the_paste_registry():
    editor = Editor(create_test_tui(), default_editor_theme)
    submitted = []
    editor.on_submit = lambda t: submitted.append(t)

    paste = _big_paste("alpha")
    await editor.handle_input(f"\x1b[200~{paste}\x1b[201~")
    await editor.handle_input("\x7f")  # delete the marker
    await editor.handle_input("\x1b[45;5u")  # undo: restores marker text and registry
    await editor.handle_input("\r")
    assert submitted == [paste]


@pytest.mark.tonio
async def test_undo_after_deleting_the_first_of_two_paste_markers_restores_both_registry_entries():
    editor = Editor(create_test_tui(), default_editor_theme)
    submitted = []
    editor.on_submit = lambda t: submitted.append(t)

    paste_a = _big_paste("alpha")
    paste_b = _big_paste("beta")
    await editor.handle_input(f"\x1b[200~{paste_a}\x1b[201~")  # #1 = A
    await editor.handle_input(f"\x1b[200~{paste_b}\x1b[201~")  # #2 = B, cursor at end
    await editor.handle_input("\x01")  # Ctrl+A
    await editor.handle_input("\x1b[C")  # right over marker #1
    await editor.handle_input("\x7f")  # delete marker #1, renumbers #2 -> #1
    await editor.handle_input("\x1b[45;5u")  # undo
    await editor.handle_input("\r")
    assert submitted == [paste_a + paste_b]


@pytest.mark.tonio
async def test_renumbers_the_paste_registry_in_ascending_id_order_when_markers_are_out_of_order_in_text():
    editor = Editor(create_test_tui(), default_editor_theme)
    submitted = []
    editor.on_submit = lambda t: submitted.append(t)

    paste_a = _big_paste("alpha")
    paste_b = _big_paste("beta")
    paste_c = _big_paste("gamma")
    await editor.handle_input(f"\x1b[200~{paste_a}\x1b[201~")  # #1 = A
    await editor.handle_input("\x01")  # Ctrl+A
    await editor.handle_input(f"\x1b[200~{paste_b}\x1b[201~")  # #2 = B, text: [#2][#1]
    await editor.handle_input("\x01")  # Ctrl+A
    await editor.handle_input(f"\x1b[200~{paste_c}\x1b[201~")  # #3 = C, text: [#3][#2][#1]
    await editor.handle_input("\x05")  # Ctrl+E
    await editor.handle_input("\x7f")  # delete marker #1, renumber #3 -> #2 and #2 -> #1
    await editor.handle_input("\r")
    assert submitted == [paste_c + paste_b]


@pytest.mark.tonio
async def test_undo_after_set_text_restores_paste_markers_and_registry():
    editor = Editor(create_test_tui(), default_editor_theme)
    submitted = []
    editor.on_submit = lambda t: submitted.append(t)

    paste = _big_paste("alpha")
    await editor.handle_input(f"\x1b[200~{paste}\x1b[201~")
    editor.set_text("replacement")
    await editor.handle_input("\x1b[45;5u")  # undo
    await editor.handle_input("\r")
    assert submitted == [paste]


@pytest.mark.tonio
async def test_handles_multiple_paste_markers_in_same_line():
    editor = Editor(create_test_tui(), default_editor_theme)
    await _paste_with_marker(editor)
    await editor.handle_input(" ")
    await _paste_with_marker(editor)

    text = editor.get_text()
    markers = [m.group(0) for m in re.finditer(r"\[paste #\d+ \+\d+ lines\]", text)]
    assert len(markers) == 2

    # Go to start
    await editor.handle_input("\x01")

    # Right arrow: should skip first marker atomically
    await editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": len(markers[0])}

    # Right arrow: past space
    await editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": len(markers[0]) + 1}

    # Right arrow: should skip second marker atomically
    await editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": len(markers[0]) + 1 + len(markers[1])}


@pytest.mark.tonio
async def test_does_not_treat_manually_typed_marker_like_text_as_atomic_no_valid_paste_id():
    editor = Editor(create_test_tui(), default_editor_theme)
    # Type text that matches the pattern but was typed manually (no paste entry)
    fake_marker = "[paste #99 +5 lines]"
    for ch in fake_marker:
        await editor.handle_input(ch)

    assert editor.get_text() == fake_marker

    # No paste with ID 99 exists, so the marker is NOT treated atomically.
    # Right arrow should move one grapheme at a time.
    await editor.handle_input("\x01")  # Ctrl+A
    await editor.handle_input("\x1b[C")  # Right
    assert editor.get_cursor() == {"line": 0, "col": 1}  # Just past "["


@pytest.mark.tonio
async def test_does_not_crash_when_paste_marker_is_wider_than_terminal_width():
    # Reproduce: terminal width 8, paste marker "[paste #1 +47 lines]" (21 chars)
    tui = create_test_tui()
    editor = Editor(tui, default_editor_theme)
    big_content = ("line\n" * 47).rstrip()
    await editor.handle_input(f"\x1b[200~{big_content}\x1b[201~")

    text = editor.get_text()
    marker = re.search(r"\[paste #\d+ \+\d+ lines\]", text)
    assert marker, "paste marker should be created"
    assert visible_width(marker.group(0)) > 8, "marker should be wider than render width"

    # Render at very narrow width - should not throw
    lines = editor.render(8)
    # Every rendered line must fit within the width (marker is split)
    for line in lines:
        assert visible_width(line) <= 8, f"line exceeds width 8: visible={visible_width(line)} text={line!r}"


@pytest.mark.tonio
async def test_does_not_crash_when_text_plus_paste_marker_exceeds_terminal_width_with_cursor_on_marker():
    # Reproduce: terminal width 54, text "b"*35 + "[paste #1 +27 lines]" + "bbbb"
    # Cursor lands on the paste marker after word-wrap, causing the rendered line
    # to be 55 visible chars (1 over the width).
    tui = create_test_tui()
    editor = Editor(tui, default_editor_theme)

    # Type 35 'b' characters
    for _ in range(35):
        await editor.handle_input("b")

    # Paste 27 lines
    big_content = ("line\n" * 27).rstrip()
    await editor.handle_input(f"\x1b[200~{big_content}\x1b[201~")

    # Type a few more characters
    for _ in range(4):
        await editor.handle_input("b")

    # Move cursor left to land on the paste marker
    await editor.handle_input("\x1b[D")  # past last 'b'
    await editor.handle_input("\x1b[D")  # past last 'b'
    await editor.handle_input("\x1b[D")  # past last 'b'
    await editor.handle_input("\x1b[D")  # past last 'b'
    await editor.handle_input("\x1b[D")  # now on the paste marker

    # Render at width 54 - should not throw
    render_width = 54
    lines = editor.render(render_width)
    for line in lines:
        assert visible_width(line) <= render_width, (
            f"line exceeds width {render_width}: visible={visible_width(line)} text={line!r}"
        )


@pytest.mark.tonio
async def test_word_wrap_line_re_checks_overflow_after_backtracking_to_wrap_opportunity():
    # Reproduce crash #2: " " + "b"*35 + atomic_marker(20 chars) + "bbbb"
    # layoutWidth=53. After wrapping at the space, the remaining 35 b's + marker = 55
    # must trigger a second force-break instead of silently overflowing.
    tui = create_test_tui()
    editor = Editor(tui, default_editor_theme)

    # Type a space, then 35 b's
    await editor.handle_input(" ")
    for _ in range(35):
        await editor.handle_input("b")

    # Paste 27 lines to create marker
    big_content = ("line\n" * 27).rstrip()
    await editor.handle_input(f"\x1b[200~{big_content}\x1b[201~")

    # Type trailing chars
    for _ in range(4):
        await editor.handle_input("b")

    # Render at width 54 (contentWidth=54, layoutWidth=53 with paddingX=0)
    render_width = 54
    lines = editor.render(render_width)
    for line in lines:
        assert visible_width(line) <= render_width, (
            f"line exceeds width {render_width}: visible={visible_width(line)} text={line!r}"
        )


@pytest.mark.tonio
async def test_expands_large_pasted_content_literally_in_get_expanded_text():
    editor = Editor(create_test_tui(), default_editor_theme)
    pasted_text = (
        "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\nline 9\nline 10\ntokens $1 $2 $& $$ $` $' end"
    )

    await editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")

    assert re.search(r"\[paste #\d+ \+\d+ lines\]", editor.get_text())
    assert editor.get_expanded_text() == pasted_text


@pytest.mark.tonio
async def test_snaps_to_the_paste_marker_start_when_navigating_down_into_it():
    editor = Editor(create_test_tui(), default_editor_theme)

    # Line 0: long enough text to establish a sticky column
    editor.set_text("12345678901234567890\n\nhello ")

    # Create a large paste to get a marker
    big_content = "x" * 2000
    await editor.handle_input(f"\x1b[200~{big_content}\x1b[201~")
    editor.render(80)

    text = editor.get_text()
    assert re.search(r"\[paste #\d+ \d+ chars\]", text)
    # Line 0: "12345678901234567890"
    # Line 1: "" (empty)
    # Line 2: "hello [paste #1 2000 chars]"
    #         marker starts at col 6

    # Navigate to line 0, col 10
    await editor.handle_input("\x1b[A")  # Up to line 1
    await editor.handle_input("\x1b[A")  # Up to line 0
    await editor.handle_input("\x01")  # Ctrl+A (start of line)
    for _ in range(10):
        await editor.handle_input("\x1b[C")  # Right 10
    assert editor.get_cursor() == {"line": 0, "col": 10}

    # Down to empty line
    await editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 1, "col": 0}

    # Down to paste marker line - sticky col 10 falls inside marker (starts at col 6).
    # Cursor should snap to start of marker (col 6), not end (col 6 + len(marker)).
    await editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 2, "col": 6}


@pytest.mark.tonio
async def test_preserves_sticky_column_when_navigating_through_paste_marker_line():
    tui = create_test_tui(30, 24)
    editor = Editor(tui, default_editor_theme)

    # Build:
    # Line 0: "1234567890123456" (16 chars)
    # Line 1: "" (empty)
    # Line 2: "[paste #1 2000 chars]" (22 chars, paste marker)
    # Line 3: "" (empty)
    # Line 4: "abcdefghijklmnop" (16 chars)
    for ch in "1234567890123456":
        await editor.handle_input(ch)
    await editor.handle_input("\n")
    await editor.handle_input("\n")
    await editor.handle_input(f"\x1b[200~{'x' * 2000}\x1b[201~")
    await editor.handle_input("\n")
    await editor.handle_input("\n")
    for ch in "abcdefghijklmnop":
        await editor.handle_input(ch)
    editor.render(30)

    # Navigate to line 0, col 10
    for _ in range(4):
        await editor.handle_input("\x1b[A")  # Up to line 0
    await editor.handle_input("\x01")  # Ctrl+A
    for _ in range(10):
        await editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": 10}

    # Down to empty line - sticky col 10 established
    await editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 1, "col": 0}

    # Down to paste marker - cursor snapped to col 0 (start of marker)
    await editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 2, "col": 0}

    # Down to empty line
    await editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 3, "col": 0}

    # Down to last line - should restore sticky col 10
    await editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 4, "col": 10}


@pytest.mark.tonio
async def test_does_not_get_stuck_moving_down_from_a_multi_visual_line_paste_marker():
    tui = create_test_tui(20, 24)
    editor = Editor(tui, default_editor_theme)

    # Build:
    # Logical line 0: "abcdefgh" + marker(21 chars) + "ijklmnopqr"
    # Logical line 1: "123456789012345678"
    #
    # Marker "[paste #1 +100 lines]" (21 chars) is wider than the
    # terminal (20). Word-wrap splits at the space before "lines",
    # producing:
    #   VL1: abcdefgh              (startCol 0,  len 8)
    #   VL2: [paste #1 +100        (startCol 8,  len 15) <- marker head
    #   VL3: lines]ijklmnopqr      (startCol 23, len 16) <- marker tail + content
    #   VL4: 123456789012345678    (line 1)
    #
    # On VL3 the marker tail "lines]" occupies visual cols 0-5.
    # Content ("i") starts at visual col 6 = logical col 29.
    for ch in "abcdefgh":
        await editor.handle_input(ch)
    big_content = ("line\n" * 100).rstrip()
    await editor.handle_input(f"\x1b[200~{big_content}\x1b[201~")
    for ch in "ijklmnopqr":
        await editor.handle_input(ch)
    await editor.handle_input("\n")
    for ch in "123456789012345678":
        await editor.handle_input(ch)
    editor.render(20)

    text = editor.get_text()
    marker_match = re.search(r"\[paste #\d+ \+\d+ lines]", text)
    assert marker_match, "paste marker should be created"
    marker_len = len(marker_match.group(0))  # 21
    assert marker_len > 20, "marker should be wider than terminal"
    marker_start = 8
    marker_end = marker_start + marker_len  # 29

    # Navigate to line 0, col 6 (on "g"). Preferred col 6 is past the
    # marker tail on VL3, so the cursor should land on content ("i" at
    # col 29) without snapping back.
    await editor.handle_input("\x1b[A")  # Up to line 0
    await editor.handle_input("\x01")  # Ctrl+A (start of line)
    for _ in range(6):
        await editor.handle_input("\x1b[C")  # Right to col 6
    assert editor.get_cursor() == {"line": 0, "col": 6}

    # Down: cursor lands on paste marker start
    await editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 0, "col": marker_start}

    # Down again: preferred col 6 lands at VL3 col 29 ("i"), which is
    # past the marker. Cursor stays on line 0.
    await editor.handle_input("\x1b[B")
    assert editor.get_cursor()["line"] == 0
    assert editor.get_cursor()["col"] == marker_end  # col 29 = "i"

    # Up: back to paste marker
    await editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": marker_start}

    # Up again: back to col 6 ("g")
    await editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 6}


@pytest.mark.tonio
async def test_skips_marker_continuation_vls_when_preferred_col_falls_in_marker_tail():
    tui = create_test_tui(20, 24)
    editor = Editor(tui, default_editor_theme)

    # Same layout. Start at col 3 ("d"). Preferred col 3 maps to VL3
    # visual col 3 which is inside the "lines]" marker tail.
    # _move_to_visual_line detects the continuation VL and skips to VL4
    # (line 1).
    #   VL1: abcdefgh              (startCol 0,  len 8)
    #   VL2: [paste #1 +100        (startCol 8,  len 15) <- marker head
    #   VL3: lines]ijklmnopqr      (startCol 23, len 16) <- marker tail + content
    #   VL4: 123456789012345678    (line 1)
    for ch in "abcdefgh":
        await editor.handle_input(ch)
    big_content = ("line\n" * 100).rstrip()
    await editor.handle_input(f"\x1b[200~{big_content}\x1b[201~")
    for ch in "ijklmnopqr":
        await editor.handle_input(ch)
    await editor.handle_input("\n")
    for ch in "123456789012345678":
        await editor.handle_input(ch)
    editor.render(20)

    # Navigate to line 0, col 3 (on "d")
    await editor.handle_input("\x1b[A")  # Up to line 0
    await editor.handle_input("\x01")  # Ctrl+A
    for _ in range(3):
        await editor.handle_input("\x1b[C")
    assert editor.get_cursor() == {"line": 0, "col": 3}

    # Down: marker
    await editor.handle_input("\x1b[B")
    assert editor.get_cursor()["col"] == 8

    # Down: skips VL3 (col 3 in marker tail) and lands on line 1
    await editor.handle_input("\x1b[B")
    assert editor.get_cursor() == {"line": 1, "col": 3}

    # Round-trip back
    await editor.handle_input("\x1b[A")
    assert editor.get_cursor()["col"] == 8  # marker
    await editor.handle_input("\x1b[A")
    assert editor.get_cursor() == {"line": 0, "col": 3}


@pytest.mark.tonio
async def test_submits_large_pasted_content_literally():
    editor = Editor(create_test_tui(), default_editor_theme)
    pasted_text = (
        "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\nline 9\nline 10\ntokens $1 $2 $& $$ $` $' end"
    )
    submitted = []
    editor.on_submit = lambda text: submitted.append(text)

    await editor.handle_input(f"\x1b[200~{pasted_text}\x1b[201~")
    await editor.handle_input("\r")

    assert submitted == [pasted_text]
