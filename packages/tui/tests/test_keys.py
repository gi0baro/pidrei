"""Mirror of pi tui test/keys.test.ts."""

import contextlib

import pytest

from pidrei_tui.keys import (
    Key,
    decode_kitty_printable,
    decode_printable_key,
    matches_key,
    parse_key,
    set_kitty_protocol_active,
)

from .tui_helpers import env_var


@pytest.fixture(autouse=True)
def _reset_kitty_flag():
    yield
    set_kitty_protocol_active(False)


@contextlib.contextmanager
def kitty_active():
    set_kitty_protocol_active(True)
    try:
        yield
    finally:
        set_kitty_protocol_active(False)


# matchesKey — Kitty protocol with alternate keys (non-Latin layouts)
#
# Kitty protocol flag 4 (Report alternate keys) sends:
# CSI codepoint:shifted:base ; modifier:event u
# Where base is the key in standard PC-101 layout


def test_matches_ctrl_c_when_pressing_cyrillic_ctrl_s_with_base_layout_key():
    with kitty_active():
        # Cyrillic 'с' = codepoint 1089, Latin 'c' = codepoint 99
        # Format: CSI 1089::99;5u (codepoint::base;modifier with ctrl=4, +1=5)
        cyrillic_ctrl_c = "\x1b[1089::99;5u"
        assert matches_key(cyrillic_ctrl_c, "ctrl+c") is True


def test_matches_ctrl_d_when_pressing_cyrillic_with_base_layout_key():
    with kitty_active():
        # Cyrillic 'в' = codepoint 1074, Latin 'd' = codepoint 100
        cyrillic_ctrl_d = "\x1b[1074::100;5u"
        assert matches_key(cyrillic_ctrl_d, "ctrl+d") is True


def test_matches_ctrl_z_when_pressing_cyrillic_with_base_layout_key():
    with kitty_active():
        # Cyrillic 'я' = codepoint 1103, Latin 'z' = codepoint 122
        cyrillic_ctrl_z = "\x1b[1103::122;5u"
        assert matches_key(cyrillic_ctrl_z, "ctrl+z") is True


def test_matches_ctrl_shift_p_with_base_layout_key():
    with kitty_active():
        # Cyrillic 'з' = codepoint 1079, Latin 'p' = codepoint 112
        # ctrl=4, shift=1, +1 = 6
        cyrillic_ctrl_shift_p = "\x1b[1079::112;6u"
        assert matches_key(cyrillic_ctrl_shift_p, "ctrl+shift+p") is True


def test_still_matches_direct_codepoint_when_no_base_layout_key():
    with kitty_active():
        # Latin ctrl+c without base layout key (terminal doesn't support flag 4)
        latin_ctrl_c = "\x1b[99;5u"
        assert matches_key(latin_ctrl_c, "ctrl+c") is True


def test_matches_super_modified_kitty_bindings_including_combined_modifiers():
    with kitty_active():
        assert matches_key("\x1b[107;9u", "super+k") is True
        assert matches_key("\x1b[13;9u", "super+enter") is True
        assert matches_key("\x1b[107;13u", Key.ctrl_super("k")) is True
        assert matches_key("\x1b[107;13u", "ctrl+super+k") is True
        assert matches_key("\x1b[107;14u", "ctrl+shift+super+k") is True
        assert matches_key("\x1b[107;13u", "super+k") is False
        assert parse_key("\x1b[107;9u") == "super+k"
        assert parse_key("\x1b[13;9u") == "super+enter"
        assert parse_key("\x1b[107;13u") == "ctrl+super+k"
        assert parse_key("\x1b[107;14u") == "shift+ctrl+super+k"


def test_matches_digit_bindings_via_kitty_csi_u():
    with kitty_active():
        assert matches_key("\x1b[49u", "1") is True
        assert matches_key("\x1b[49;5u", "ctrl+1") is True
        assert matches_key("\x1b[49;5u", "ctrl+2") is False
        assert parse_key("\x1b[49u") == "1"
        assert parse_key("\x1b[49;5u") == "ctrl+1"


def test_normalizes_kitty_keypad_functional_keys():
    with kitty_active():
        assert matches_key("\x1b[57400u", "1") is True
        assert matches_key("\x1b[57410u", "/") is True
        assert matches_key("\x1b[57417u", "left") is True
        assert matches_key("\x1b[57426u", "delete") is True
        assert parse_key("\x1b[57399u") == "0"
        assert parse_key("\x1b[57409u") == "."
        assert parse_key("\x1b[57413u") == "+"
        assert parse_key("\x1b[57416u") == ","
        assert parse_key("\x1b[57417u") == "left"
        assert parse_key("\x1b[57418u") == "right"
        assert parse_key("\x1b[57419u") == "up"
        assert parse_key("\x1b[57420u") == "down"
        assert parse_key("\x1b[57421u") == "pageUp"
        assert parse_key("\x1b[57422u") == "pageDown"
        assert parse_key("\x1b[57423u") == "home"
        assert parse_key("\x1b[57424u") == "end"
        assert parse_key("\x1b[57425u") == "insert"
        assert parse_key("\x1b[57426u") == "delete"


def test_handles_shifted_key_in_format():
    with kitty_active():
        # Format with shifted key: CSI codepoint:shifted:base;modifier u
        # Latin 'c' with shifted 'C' (67) and base 'c' (99)
        shifted_key = "\x1b[99:67:99;2u"  # shift modifier = 1, +1 = 2
        assert matches_key(shifted_key, "shift+c") is True


def test_handles_event_type_in_format():
    with kitty_active():
        # Format with event type: CSI codepoint::base;modifier:event u
        # Cyrillic ctrl+c release event (event type 3)
        release_event = "\x1b[1089::99;5:3u"
        assert matches_key(release_event, "ctrl+c") is True


def test_handles_full_format_with_shifted_key_base_key_and_event_type():
    with kitty_active():
        # Full format: CSI codepoint:shifted:base;modifier:event u
        # Cyrillic 'С' (shifted) with base 'c', Ctrl+Shift pressed, repeat event
        # Cyrillic 'с' = 1089, Cyrillic 'С' = 1057, Latin 'c' = 99
        # ctrl=4, shift=1, +1 = 6, repeat event = 2
        full_format = "\x1b[1089:1057:99;6:2u"
        assert matches_key(full_format, "ctrl+shift+c") is True


def test_prefers_codepoint_for_latin_letters_even_when_base_layout_differs():
    with kitty_active():
        # Dvorak Ctrl+K reports codepoint 'k' (107) and base layout 'v' (118)
        dvorak_ctrl_k = "\x1b[107::118;5u"
        assert matches_key(dvorak_ctrl_k, "ctrl+k") is True
        assert matches_key(dvorak_ctrl_k, "ctrl+v") is False


def test_prefers_codepoint_for_symbol_keys_even_when_base_layout_differs():
    with kitty_active():
        # Dvorak Ctrl+/ reports codepoint '/' (47) and base layout '[' (91)
        dvorak_ctrl_slash = "\x1b[47::91;5u"
        assert matches_key(dvorak_ctrl_slash, "ctrl+/") is True
        assert matches_key(dvorak_ctrl_slash, "ctrl+[") is False


def test_does_not_match_wrong_key_even_with_base_layout():
    with kitty_active():
        # Cyrillic ctrl+с with base 'c' should NOT match ctrl+d
        cyrillic_ctrl_c = "\x1b[1089::99;5u"
        assert matches_key(cyrillic_ctrl_c, "ctrl+d") is False


def test_does_not_match_wrong_modifiers_even_with_base_layout():
    with kitty_active():
        # Cyrillic ctrl+с should NOT match ctrl+shift+c
        cyrillic_ctrl_c = "\x1b[1089::99;5u"
        assert matches_key(cyrillic_ctrl_c, "ctrl+shift+c") is False


# matchesKey — modifyOtherKeys matching


def test_matches_xterm_modify_other_keys_ctrl_c():
    set_kitty_protocol_active(False)
    assert matches_key("\x1b[27;5;99~", "ctrl+c") is True
    assert parse_key("\x1b[27;5;99~") == "ctrl+c"


def test_matches_xterm_modify_other_keys_ctrl_d():
    set_kitty_protocol_active(False)
    assert matches_key("\x1b[27;5;100~", "ctrl+d") is True
    assert parse_key("\x1b[27;5;100~") == "ctrl+d"


def test_matches_xterm_modify_other_keys_ctrl_z():
    set_kitty_protocol_active(False)
    assert matches_key("\x1b[27;5;122~", "ctrl+z") is True
    assert parse_key("\x1b[27;5;122~") == "ctrl+z"


def test_matches_xterm_modify_other_keys_enter_variants():
    set_kitty_protocol_active(False)
    assert matches_key("\x1b[27;5;13~", "ctrl+enter") is True
    assert matches_key("\x1b[27;2;13~", "shift+enter") is True
    assert matches_key("\x1b[27;3;13~", "alt+enter") is True
    assert parse_key("\x1b[27;5;13~") == "ctrl+enter"
    assert parse_key("\x1b[27;2;13~") == "shift+enter"
    assert parse_key("\x1b[27;3;13~") == "alt+enter"


def test_matches_xterm_modify_other_keys_tab_variants():
    set_kitty_protocol_active(False)
    assert matches_key("\x1b[27;2;9~", "shift+tab") is True
    assert matches_key("\x1b[27;5;9~", "ctrl+tab") is True
    assert matches_key("\x1b[27;3;9~", "alt+tab") is True
    assert parse_key("\x1b[27;2;9~") == "shift+tab"
    assert parse_key("\x1b[27;5;9~") == "ctrl+tab"
    assert parse_key("\x1b[27;3;9~") == "alt+tab"


def test_matches_xterm_modify_other_keys_backspace_variants():
    set_kitty_protocol_active(False)
    assert matches_key("\x1b[27;1;127~", "backspace") is True
    assert matches_key("\x1b[27;5;127~", "ctrl+backspace") is True
    assert matches_key("\x1b[27;3;127~", "alt+backspace") is True
    assert parse_key("\x1b[27;1;127~") == "backspace"
    assert parse_key("\x1b[27;5;127~") == "ctrl+backspace"
    assert parse_key("\x1b[27;3;127~") == "alt+backspace"


def test_matches_xterm_modify_other_keys_escape():
    set_kitty_protocol_active(False)
    assert matches_key("\x1b[27;1;27~", "escape") is True
    assert parse_key("\x1b[27;1;27~") == "escape"


def test_matches_xterm_modify_other_keys_space_variants():
    set_kitty_protocol_active(False)
    assert matches_key("\x1b[27;1;32~", "space") is True
    assert matches_key("\x1b[27;5;32~", "ctrl+space") is True
    assert parse_key("\x1b[27;1;32~") == "space"
    assert parse_key("\x1b[27;5;32~") == "ctrl+space"


def test_matches_xterm_modify_other_keys_symbol_combos():
    set_kitty_protocol_active(False)
    assert matches_key("\x1b[27;5;47~", "ctrl+/") is True
    assert parse_key("\x1b[27;5;47~") == "ctrl+/"


def test_matches_xterm_modify_other_keys_digit_combos():
    set_kitty_protocol_active(False)
    assert matches_key("\x1b[27;5;49~", "ctrl+1") is True
    assert matches_key("\x1b[27;2;49~", "shift+1") is True
    assert parse_key("\x1b[27;5;49~") == "ctrl+1"
    assert parse_key("\x1b[27;2;49~") == "shift+1"


def test_matches_xterm_modify_other_keys_shifted_uppercase_letters():
    set_kitty_protocol_active(False)
    assert matches_key("\x1b[27;2;69~", "shift+e") is True
    assert matches_key("\x1b[27;6;69~", "ctrl+shift+e") is True
    assert parse_key("\x1b[27;2;69~") == "shift+e"
    assert parse_key("\x1b[27;6;69~") == "shift+ctrl+e"


def test_matches_ctrl_alt_letter_via_csi_u_when_kitty_inactive():
    set_kitty_protocol_active(False)
    assert matches_key("\x1b[104;7u", "ctrl+alt+h") is True
    assert parse_key("\x1b[104;7u") == "ctrl+alt+h"


def test_matches_ctrl_alt_letter_via_xterm_modify_other_keys():
    set_kitty_protocol_active(False)
    assert matches_key("\x1b[27;7;104~", "ctrl+alt+h") is True
    assert parse_key("\x1b[27;7;104~") == "ctrl+alt+h"


# matchesKey — Legacy key matching


def test_matches_legacy_ctrl_c():
    set_kitty_protocol_active(False)
    # Ctrl+c sends ASCII 3 (ETX)
    assert matches_key("\x03", "ctrl+c") is True


def test_matches_legacy_ctrl_d():
    set_kitty_protocol_active(False)
    # Ctrl+d sends ASCII 4 (EOT)
    assert matches_key("\x04", "ctrl+d") is True


def test_matches_escape_key():
    assert matches_key("\x1b", "escape") is True


def test_matches_legacy_linefeed_as_enter():
    set_kitty_protocol_active(False)
    assert matches_key("\n", "enter") is True
    assert parse_key("\n") == "enter"


def test_treats_linefeed_as_shift_enter_when_kitty_active():
    with kitty_active():
        assert matches_key("\n", "shift+enter") is True
        assert matches_key("\n", "enter") is False
        assert parse_key("\n") == "shift+enter"


def test_parses_ctrl_space():
    set_kitty_protocol_active(False)
    assert matches_key("\x00", "ctrl+space") is True
    assert parse_key("\x00") == "ctrl+space"


def test_matches_legacy_ctrl_symbol():
    set_kitty_protocol_active(False)
    # Ctrl+\ sends ASCII 28 (File Separator) in legacy terminals
    assert matches_key("\x1c", "ctrl+\\") is True
    assert parse_key("\x1c") == "ctrl+\\"
    # Ctrl+] sends ASCII 29 (Group Separator) in legacy terminals
    assert matches_key("\x1d", "ctrl+]") is True
    assert parse_key("\x1d") == "ctrl+]"
    # Ctrl+_ sends ASCII 31 (Unit Separator) in legacy terminals
    # Ctrl+- is on the same physical key on US keyboards
    assert matches_key("\x1f", "ctrl+_") is True
    assert matches_key("\x1f", "ctrl+-") is True
    assert parse_key("\x1f") == "ctrl+-"


def test_matches_legacy_ctrl_alt_symbol():
    set_kitty_protocol_active(False)
    # Ctrl+Alt+[ sends ESC followed by ESC (Ctrl+[ = ESC)
    assert matches_key("\x1b\x1b", "ctrl+alt+[") is True
    assert parse_key("\x1b\x1b") == "ctrl+alt+["
    # Ctrl+Alt+\ sends ESC followed by ASCII 28
    assert matches_key("\x1b\x1c", "ctrl+alt+\\") is True
    assert parse_key("\x1b\x1c") == "ctrl+alt+\\"
    # Ctrl+Alt+] sends ESC followed by ASCII 29
    assert matches_key("\x1b\x1d", "ctrl+alt+]") is True
    assert parse_key("\x1b\x1d") == "ctrl+alt+]"
    # Ctrl+_ sends ASCII 31 (Unit Separator) in legacy terminals
    # Ctrl+- is on the same physical key on US keyboards
    assert matches_key("\x1b\x1f", "ctrl+alt+_") is True
    assert matches_key("\x1b\x1f", "ctrl+alt+-") is True
    assert parse_key("\x1b\x1f") == "ctrl+alt+-"


def test_treats_raw_0x08_as_plain_backspace_outside_windows_terminal():
    set_kitty_protocol_active(False)
    with env_var("WT_SESSION", None):
        assert matches_key("\x7f", "backspace") is True
        assert matches_key("\x7f", "ctrl+backspace") is False
        assert parse_key("\x7f") == "backspace"
        assert matches_key("\x08", "backspace") is True
        assert matches_key("\x08", "ctrl+backspace") is False
        assert parse_key("\x08") == "backspace"
        assert matches_key("\x08", "ctrl+h") is True


def test_treats_raw_0x08_as_ctrl_backspace_in_local_windows_terminal():
    set_kitty_protocol_active(False)
    with (
        env_var("WT_SESSION", "test-session"),
        env_var("SSH_CONNECTION", None),
        env_var("SSH_CLIENT", None),
        env_var("SSH_TTY", None),
    ):
        assert matches_key("\x08", "ctrl+backspace") is True
        assert matches_key("\x08", "backspace") is False
        assert parse_key("\x08") == "ctrl+backspace"
        assert matches_key("\x08", "ctrl+h") is True


def test_treats_raw_0x08_as_plain_backspace_in_windows_terminal_over_ssh():
    set_kitty_protocol_active(False)
    with (
        env_var("WT_SESSION", "test-session"),
        env_var("SSH_CONNECTION", "1 2 3 4"),
        env_var("SSH_CLIENT", "1 2 3"),
        env_var("SSH_TTY", "/dev/pts/1"),
    ):
        assert matches_key("\x08", "ctrl+backspace") is False
        assert matches_key("\x08", "backspace") is True
        assert parse_key("\x08") == "backspace"
        assert matches_key("\x08", "ctrl+h") is True


def test_parses_legacy_alt_prefixed_sequences_when_kitty_inactive():
    set_kitty_protocol_active(False)
    assert matches_key("\x1b ", "alt+space") is True
    assert parse_key("\x1b ") == "alt+space"
    assert matches_key("\x1b\x08", "alt+backspace") is True
    assert parse_key("\x1b\x08") == "alt+backspace"
    assert matches_key("\x1b\x03", "ctrl+alt+c") is True
    assert parse_key("\x1b\x03") == "ctrl+alt+c"
    assert matches_key("\x1bB", "alt+left") is True
    assert parse_key("\x1bB") == "alt+left"
    assert matches_key("\x1bF", "alt+right") is True
    assert parse_key("\x1bF") == "alt+right"
    assert matches_key("\x1ba", "alt+a") is True
    assert parse_key("\x1ba") == "alt+a"
    assert matches_key("\x1b1", "alt+1") is True
    assert parse_key("\x1b1") == "alt+1"
    assert matches_key("\x1b,", "alt+,") is True
    assert parse_key("\x1b,") == "alt+,"
    assert matches_key("\x1b.", "alt+.") is True
    assert parse_key("\x1b.") == "alt+."
    assert matches_key("\x1by", "alt+y") is True
    assert parse_key("\x1by") == "alt+y"
    assert matches_key("\x1bz", "alt+z") is True
    assert parse_key("\x1bz") == "alt+z"

    with kitty_active():
        assert matches_key("\x1b ", "alt+space") is False
        assert parse_key("\x1b ") is None
        assert matches_key("\x1b\x08", "alt+backspace") is True
        assert parse_key("\x1b\x08") == "alt+backspace"
        assert matches_key("\x1b\x03", "ctrl+alt+c") is False
        assert parse_key("\x1b\x03") is None
        assert matches_key("\x1bB", "alt+left") is False
        assert parse_key("\x1bB") is None
        assert matches_key("\x1bF", "alt+right") is False
        assert parse_key("\x1bF") is None
        assert matches_key("\x1ba", "alt+a") is False
        assert parse_key("\x1ba") is None
        assert matches_key("\x1b1", "alt+1") is False
        assert parse_key("\x1b1") is None
        assert matches_key("\x1b,", "alt+,") is False
        assert parse_key("\x1b,") is None
        assert matches_key("\x1b.", "alt+.") is False
        assert parse_key("\x1b.") is None
        assert matches_key("\x1by", "alt+y") is False
        assert parse_key("\x1by") is None


def test_matches_arrow_keys():
    assert matches_key("\x1b[A", "up") is True
    assert matches_key("\x1b[B", "down") is True
    assert matches_key("\x1b[C", "right") is True
    assert matches_key("\x1b[D", "left") is True


def test_matches_ss3_arrows_and_home_end():
    assert matches_key("\x1bOA", "up") is True
    assert matches_key("\x1bOB", "down") is True
    assert matches_key("\x1bOC", "right") is True
    assert matches_key("\x1bOD", "left") is True
    assert matches_key("\x1bOH", "home") is True
    assert matches_key("\x1bOF", "end") is True


def test_matches_xterm_ctrl_modified_viewport_navigation():
    assert matches_key("\x1b[1;5H", "ctrl+home") is True
    assert matches_key("\x1b[1;5F", "ctrl+end") is True
    assert matches_key("\x1b[5;5~", "ctrl+pageUp") is True
    assert matches_key("\x1b[6;5~", "ctrl+pageDown") is True
    assert parse_key("\x1b[1;5H") == "ctrl+home"
    assert parse_key("\x1b[1;5F") == "ctrl+end"
    assert parse_key("\x1b[5;5~") == "ctrl+pageUp"
    assert parse_key("\x1b[6;5~") == "ctrl+pageDown"


def test_matches_legacy_function_keys_and_clear():
    assert matches_key("\x1bOP", "f1") is True
    assert matches_key("\x1b[24~", "f12") is True
    assert matches_key("\x1b[E", "clear") is True


def test_matches_alt_arrows():
    assert matches_key("\x1bp", "alt+up") is True
    assert matches_key("\x1bp", "up") is False


def test_matches_rxvt_modifier_sequences():
    assert matches_key("\x1b[a", "shift+up") is True
    assert matches_key("\x1bOa", "ctrl+up") is True
    assert matches_key("\x1b[2$", "shift+insert") is True
    assert matches_key("\x1b[2^", "ctrl+insert") is True
    assert matches_key("\x1b[7$", "shift+home") is True


# decodeKittyPrintable


def test_decodes_kitty_keypad_functional_keys_to_printable_characters():
    assert decode_kitty_printable("\x1b[57399u") == "0"
    assert decode_kitty_printable("\x1b[57400u") == "1"
    assert decode_kitty_printable("\x1b[57409u") == "."
    assert decode_kitty_printable("\x1b[57410u") == "/"
    assert decode_kitty_printable("\x1b[57411u") == "*"
    assert decode_kitty_printable("\x1b[57412u") == "-"
    assert decode_kitty_printable("\x1b[57413u") == "+"
    assert decode_kitty_printable("\x1b[57415u") == "="
    assert decode_kitty_printable("\x1b[57416u") == ","
    assert decode_kitty_printable("\x1b[57417u") is None


# decodePrintableKey


def test_decodes_printable_xterm_modify_other_keys_sequences():
    assert decode_printable_key("\x1b[27;2;69~") == "E"
    assert decode_printable_key("\x1b[27;2;196~") == "Ä"
    assert decode_printable_key("\x1b[27;2;32~") == " "
    assert decode_printable_key("\x1b[27;2;13~") is None
    assert decode_printable_key("\x1b[27;6;69~") is None


# parseKey — Kitty protocol with alternate keys


def test_parse_returns_latin_key_name_when_base_layout_key_is_present():
    with kitty_active():
        # Cyrillic ctrl+с with base layout 'c'
        cyrillic_ctrl_c = "\x1b[1089::99;5u"
        assert parse_key(cyrillic_ctrl_c) == "ctrl+c"


def test_parse_prefers_codepoint_for_latin_letters_when_base_layout_differs():
    with kitty_active():
        # Dvorak Ctrl+K reports codepoint 'k' (107) and base layout 'v' (118)
        dvorak_ctrl_k = "\x1b[107::118;5u"
        assert parse_key(dvorak_ctrl_k) == "ctrl+k"


def test_parse_prefers_codepoint_for_symbol_keys_when_base_layout_differs():
    with kitty_active():
        # Dvorak Ctrl+/ reports codepoint '/' (47) and base layout '[' (91)
        dvorak_ctrl_slash = "\x1b[47::91;5u"
        assert parse_key(dvorak_ctrl_slash) == "ctrl+/"


def test_parse_returns_key_name_from_codepoint_when_no_base_layout():
    with kitty_active():
        latin_ctrl_c = "\x1b[99;5u"
        assert parse_key(latin_ctrl_c) == "ctrl+c"


def test_parse_shifted_uppercase_csi_u_letters_as_shift_letter():
    with kitty_active():
        assert matches_key("\x1b[69;2u", "shift+e") is True
        assert parse_key("\x1b[69;2u") == "shift+e"


def test_parse_ignores_kitty_csi_u_with_unsupported_modifiers():
    with kitty_active():
        assert parse_key("\x1b[99;17u") is None


# parseKey — Legacy key parsing


def test_parses_legacy_ctrl_letter():
    set_kitty_protocol_active(False)
    assert parse_key("\x03") == "ctrl+c"
    assert parse_key("\x04") == "ctrl+d"


def test_parses_special_keys():
    assert parse_key("\x1b") == "escape"
    assert parse_key("\t") == "tab"
    assert parse_key("\r") == "enter"
    assert parse_key("\n") == "enter"
    assert parse_key("\x00") == "ctrl+space"
    assert parse_key(" ") == "space"
    assert parse_key("1") == "1"
    assert matches_key("1", "1") is True


def test_parses_arrow_keys():
    assert parse_key("\x1b[A") == "up"
    assert parse_key("\x1b[B") == "down"
    assert parse_key("\x1b[C") == "right"
    assert parse_key("\x1b[D") == "left"


def test_parses_ss3_arrows_and_home_end():
    assert parse_key("\x1bOA") == "up"
    assert parse_key("\x1bOB") == "down"
    assert parse_key("\x1bOC") == "right"
    assert parse_key("\x1bOD") == "left"
    assert parse_key("\x1bOH") == "home"
    assert parse_key("\x1bOF") == "end"


def test_parses_legacy_function_and_modifier_sequences():
    assert parse_key("\x1bOP") == "f1"
    assert parse_key("\x1b[24~") == "f12"
    assert parse_key("\x1b[E") == "clear"
    assert parse_key("\x1b[2^") == "ctrl+insert"
    assert parse_key("\x1bp") == "alt+up"


def test_parses_double_bracket_page_up():
    assert parse_key("\x1b[[5~") == "pageUp"
