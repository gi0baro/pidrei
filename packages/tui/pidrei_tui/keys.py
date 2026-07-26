"""Mirror of pi tui src/keys.ts.

Keyboard input handling for terminal applications.

Supports both legacy terminal sequences and Kitty keyboard protocol.
See: https://sw.kovidgoyal.net/kitty/keyboard-protocol/
Reference: https://github.com/sst/opentui/blob/7da92b4088aebfe27b9f691c04163a48821e49fd/packages/core/src/lib/parse.keypress.ts

Symbol keys are also supported, however some ctrl+symbol combos
overlap with ASCII codes, e.g. ctrl+[ = ESC.
See: https://sw.kovidgoyal.net/kitty/keyboard-protocol/#legacy-ctrl-mapping-of-ascii-keys
Those can still be used for ctrl+shift combos.

API:

- ``matches_key(data, key_id)`` - Check if input matches a key identifier
- ``parse_key(data)`` - Parse input and return the key identifier
- ``Key`` - Helper object for creating key identifiers
- ``set_kitty_protocol_active(active)`` - Set global Kitty protocol state
- ``is_kitty_protocol_active()`` - Query global Kitty protocol state

Port notes: pi's ``KeyId`` template-literal union is erasable typing — key
identifiers are plain ``str`` here. ``Key.return`` (alias of ``enter``) is
attached via setattr because ``return`` is a Python keyword. JS
``String.fromCharCode`` wraps the negative sentinel codepoints (arrows,
functional keys) to uint16 garbage that never matches a printable key;
``chr()`` would raise instead, so ``_safe_chr`` returns "" for them.
"""

import os
import re


# =============================================================================
# Global Kitty Protocol State
# =============================================================================

_kitty_protocol_active = False


def set_kitty_protocol_active(active: bool) -> None:
    """Set the global Kitty keyboard protocol state.

    Called by ProcessTerminal after detecting protocol support.
    """
    global _kitty_protocol_active
    _kitty_protocol_active = active


def is_kitty_protocol_active() -> bool:
    """Query whether Kitty keyboard protocol is currently active."""
    return _kitty_protocol_active


# =============================================================================
# Key Identifiers
# =============================================================================


class Key:
    """Helper object for creating key identifiers.

    Usage:

    - ``Key.escape``, ``Key.enter``, ``Key.tab``, etc. for special keys
    - ``Key.backtick``, ``Key.comma``, ``Key.period``, etc. for symbol keys
    - ``Key.ctrl("c")``, ``Key.alt("x")``, ``Key.super("k")`` for single modifiers
    - ``Key.ctrl_shift("p")``, ``Key.ctrl_alt("x")``, ``Key.ctrl_super("k")``
      for combined modifiers
    """

    # Special keys
    escape = "escape"
    esc = "esc"
    enter = "enter"
    tab = "tab"
    space = "space"
    backspace = "backspace"
    delete = "delete"
    insert = "insert"
    clear = "clear"
    home = "home"
    end = "end"
    pageUp = "pageUp"  # noqa: N815
    pageDown = "pageDown"  # noqa: N815
    up = "up"
    down = "down"
    left = "left"
    right = "right"
    f1 = "f1"
    f2 = "f2"
    f3 = "f3"
    f4 = "f4"
    f5 = "f5"
    f6 = "f6"
    f7 = "f7"
    f8 = "f8"
    f9 = "f9"
    f10 = "f10"
    f11 = "f11"
    f12 = "f12"

    # Symbol keys
    backtick = "`"
    hyphen = "-"
    equals = "="
    leftbracket = "["
    rightbracket = "]"
    backslash = "\\"
    semicolon = ";"
    quote = "'"
    comma = ","
    period = "."
    slash = "/"
    exclamation = "!"
    at = "@"
    hash = "#"
    dollar = "$"
    percent = "%"
    caret = "^"
    ampersand = "&"
    asterisk = "*"
    leftparen = "("
    rightparen = ")"
    underscore = "_"
    plus = "+"
    pipe = "|"
    tilde = "~"
    leftbrace = "{"
    rightbrace = "}"
    colon = ":"
    lessthan = "<"
    greaterthan = ">"
    question = "?"

    # Single modifiers
    @staticmethod
    def ctrl(key: str) -> str:
        return f"ctrl+{key}"

    @staticmethod
    def shift(key: str) -> str:
        return f"shift+{key}"

    @staticmethod
    def alt(key: str) -> str:
        return f"alt+{key}"

    @staticmethod
    def super(key: str) -> str:
        return f"super+{key}"

    # Combined modifiers
    @staticmethod
    def ctrl_shift(key: str) -> str:
        return f"ctrl+shift+{key}"

    @staticmethod
    def shift_ctrl(key: str) -> str:
        return f"shift+ctrl+{key}"

    @staticmethod
    def ctrl_alt(key: str) -> str:
        return f"ctrl+alt+{key}"

    @staticmethod
    def alt_ctrl(key: str) -> str:
        return f"alt+ctrl+{key}"

    @staticmethod
    def shift_alt(key: str) -> str:
        return f"shift+alt+{key}"

    @staticmethod
    def alt_shift(key: str) -> str:
        return f"alt+shift+{key}"

    @staticmethod
    def ctrl_super(key: str) -> str:
        return f"ctrl+super+{key}"

    @staticmethod
    def super_ctrl(key: str) -> str:
        return f"super+ctrl+{key}"

    @staticmethod
    def shift_super(key: str) -> str:
        return f"shift+super+{key}"

    @staticmethod
    def super_shift(key: str) -> str:
        return f"super+shift+{key}"

    @staticmethod
    def alt_super(key: str) -> str:
        return f"alt+super+{key}"

    @staticmethod
    def super_alt(key: str) -> str:
        return f"super+alt+{key}"

    # Triple modifiers
    @staticmethod
    def ctrl_shift_alt(key: str) -> str:
        return f"ctrl+shift+alt+{key}"

    @staticmethod
    def ctrl_shift_super(key: str) -> str:
        return f"ctrl+shift+super+{key}"


setattr(Key, "return", "return")


# =============================================================================
# Constants
# =============================================================================

SYMBOL_KEYS = frozenset("`-=[]\\;',./!@#$%^&*()_+|~{}:<>?")

MODIFIERS = {
    "shift": 1,
    "alt": 2,
    "ctrl": 4,
    "super": 8,
}

LOCK_MASK = 64 + 128  # Caps Lock + Num Lock

CODEPOINTS = {
    "escape": 27,
    "tab": 9,
    "enter": 13,
    "space": 32,
    "backspace": 127,
    "kpEnter": 57414,  # Numpad Enter (Kitty protocol)
}

ARROW_CODEPOINTS = {
    "up": -1,
    "down": -2,
    "right": -3,
    "left": -4,
}

FUNCTIONAL_CODEPOINTS = {
    "delete": -10,
    "insert": -11,
    "pageUp": -12,
    "pageDown": -13,
    "home": -14,
    "end": -15,
}

KITTY_FUNCTIONAL_KEY_EQUIVALENTS = {
    57399: 48,  # KP_0 -> 0
    57400: 49,  # KP_1 -> 1
    57401: 50,  # KP_2 -> 2
    57402: 51,  # KP_3 -> 3
    57403: 52,  # KP_4 -> 4
    57404: 53,  # KP_5 -> 5
    57405: 54,  # KP_6 -> 6
    57406: 55,  # KP_7 -> 7
    57407: 56,  # KP_8 -> 8
    57408: 57,  # KP_9 -> 9
    57409: 46,  # KP_DECIMAL -> .
    57410: 47,  # KP_DIVIDE -> /
    57411: 42,  # KP_MULTIPLY -> *
    57412: 45,  # KP_SUBTRACT -> -
    57413: 43,  # KP_ADD -> +
    57415: 61,  # KP_EQUAL -> =
    57416: 44,  # KP_SEPARATOR -> ,
    57417: ARROW_CODEPOINTS["left"],
    57418: ARROW_CODEPOINTS["right"],
    57419: ARROW_CODEPOINTS["up"],
    57420: ARROW_CODEPOINTS["down"],
    57421: FUNCTIONAL_CODEPOINTS["pageUp"],
    57422: FUNCTIONAL_CODEPOINTS["pageDown"],
    57423: FUNCTIONAL_CODEPOINTS["home"],
    57424: FUNCTIONAL_CODEPOINTS["end"],
    57425: FUNCTIONAL_CODEPOINTS["insert"],
    57426: FUNCTIONAL_CODEPOINTS["delete"],
}


def _safe_chr(codepoint: int) -> str:
    """chr() that returns "" for the negative/overflow sentinels (see module docstring)."""
    if 0 <= codepoint <= 0x10FFFF:
        return chr(codepoint)
    return ""


def _normalize_kitty_functional_codepoint(codepoint: int) -> int:
    return KITTY_FUNCTIONAL_KEY_EQUIVALENTS.get(codepoint, codepoint)


def _normalize_shifted_letter_identity_codepoint(codepoint: int, modifier: int) -> int:
    effective_modifier = modifier & ~LOCK_MASK
    if (effective_modifier & MODIFIERS["shift"]) != 0 and 65 <= codepoint <= 90:
        return codepoint + 32
    return codepoint


LEGACY_KEY_SEQUENCES = {
    "up": ("\x1b[A", "\x1bOA"),
    "down": ("\x1b[B", "\x1bOB"),
    "right": ("\x1b[C", "\x1bOC"),
    "left": ("\x1b[D", "\x1bOD"),
    "home": ("\x1b[H", "\x1bOH", "\x1b[1~", "\x1b[7~"),
    "end": ("\x1b[F", "\x1bOF", "\x1b[4~", "\x1b[8~"),
    "insert": ("\x1b[2~",),
    "delete": ("\x1b[3~",),
    "pageUp": ("\x1b[5~", "\x1b[[5~"),
    "pageDown": ("\x1b[6~", "\x1b[[6~"),
    "clear": ("\x1b[E", "\x1bOE"),
    "f1": ("\x1bOP", "\x1b[11~", "\x1b[[A"),
    "f2": ("\x1bOQ", "\x1b[12~", "\x1b[[B"),
    "f3": ("\x1bOR", "\x1b[13~", "\x1b[[C"),
    "f4": ("\x1bOS", "\x1b[14~", "\x1b[[D"),
    "f5": ("\x1b[15~", "\x1b[[E"),
    "f6": ("\x1b[17~",),
    "f7": ("\x1b[18~",),
    "f8": ("\x1b[19~",),
    "f9": ("\x1b[20~",),
    "f10": ("\x1b[21~",),
    "f11": ("\x1b[23~",),
    "f12": ("\x1b[24~",),
}

LEGACY_SHIFT_SEQUENCES = {
    "up": ("\x1b[a",),
    "down": ("\x1b[b",),
    "right": ("\x1b[c",),
    "left": ("\x1b[d",),
    "clear": ("\x1b[e",),
    "insert": ("\x1b[2$",),
    "delete": ("\x1b[3$",),
    "pageUp": ("\x1b[5$",),
    "pageDown": ("\x1b[6$",),
    "home": ("\x1b[7$",),
    "end": ("\x1b[8$",),
}

LEGACY_CTRL_SEQUENCES = {
    "up": ("\x1bOa",),
    "down": ("\x1bOb",),
    "right": ("\x1bOc",),
    "left": ("\x1bOd",),
    "clear": ("\x1bOe",),
    "insert": ("\x1b[2^",),
    "delete": ("\x1b[3^",),
    "pageUp": ("\x1b[5^",),
    "pageDown": ("\x1b[6^",),
    "home": ("\x1b[7^",),
    "end": ("\x1b[8^",),
}

LEGACY_SEQUENCE_KEY_IDS = {
    "\x1bOA": "up",
    "\x1bOB": "down",
    "\x1bOC": "right",
    "\x1bOD": "left",
    "\x1bOH": "home",
    "\x1bOF": "end",
    "\x1b[E": "clear",
    "\x1bOE": "clear",
    "\x1bOe": "ctrl+clear",
    "\x1b[e": "shift+clear",
    "\x1b[2~": "insert",
    "\x1b[2$": "shift+insert",
    "\x1b[2^": "ctrl+insert",
    "\x1b[3$": "shift+delete",
    "\x1b[3^": "ctrl+delete",
    "\x1b[[5~": "pageUp",
    "\x1b[[6~": "pageDown",
    "\x1b[a": "shift+up",
    "\x1b[b": "shift+down",
    "\x1b[c": "shift+right",
    "\x1b[d": "shift+left",
    "\x1bOa": "ctrl+up",
    "\x1bOb": "ctrl+down",
    "\x1bOc": "ctrl+right",
    "\x1bOd": "ctrl+left",
    "\x1b[5$": "shift+pageUp",
    "\x1b[6$": "shift+pageDown",
    "\x1b[7$": "shift+home",
    "\x1b[8$": "shift+end",
    "\x1b[5^": "ctrl+pageUp",
    "\x1b[6^": "ctrl+pageDown",
    "\x1b[7^": "ctrl+home",
    "\x1b[8^": "ctrl+end",
    "\x1bOP": "f1",
    "\x1bOQ": "f2",
    "\x1bOR": "f3",
    "\x1bOS": "f4",
    "\x1b[11~": "f1",
    "\x1b[12~": "f2",
    "\x1b[13~": "f3",
    "\x1b[14~": "f4",
    "\x1b[[A": "f1",
    "\x1b[[B": "f2",
    "\x1b[[C": "f3",
    "\x1b[[D": "f4",
    "\x1b[[E": "f5",
    "\x1b[15~": "f5",
    "\x1b[17~": "f6",
    "\x1b[18~": "f7",
    "\x1b[19~": "f8",
    "\x1b[20~": "f9",
    "\x1b[21~": "f10",
    "\x1b[23~": "f11",
    "\x1b[24~": "f12",
    "\x1bb": "alt+left",
    "\x1bf": "alt+right",
    "\x1bp": "alt+up",
    "\x1bn": "alt+down",
}


def _matches_legacy_sequence(data: str, sequences: tuple[str, ...]) -> bool:
    return data in sequences


def _matches_legacy_modifier_sequence(data: str, key: str, modifier: int) -> bool:
    if modifier == MODIFIERS["shift"]:
        return _matches_legacy_sequence(data, LEGACY_SHIFT_SEQUENCES[key])
    if modifier == MODIFIERS["ctrl"]:
        return _matches_legacy_sequence(data, LEGACY_CTRL_SEQUENCES[key])
    return False


# =============================================================================
# Kitty Protocol Parsing
# =============================================================================

# Event types from Kitty keyboard protocol (flag 2):
# 1 = key press, 2 = key repeat, 3 = key release ("press" | "repeat" | "release")
KeyEventType = str

# Store the last parsed event type for is_key_release() to query
_last_event_type: KeyEventType = "press"

_RELEASE_MARKERS = (":3u", ":3~", ":3A", ":3B", ":3C", ":3D", ":3H", ":3F")
_REPEAT_MARKERS = (":2u", ":2~", ":2A", ":2B", ":2C", ":2D", ":2H", ":2F")


def is_key_release(data: str) -> bool:
    """Check if the input is a key release event.

    Only meaningful when Kitty keyboard protocol with flag 2 is active.
    """
    # Don't treat bracketed paste content as key release, even if it contains
    # patterns like ":3F" (e.g., bluetooth MAC addresses like "90:62:3F:A5").
    # terminal.py re-wraps paste content with bracketed paste markers before
    # passing to TUI, so pasted data will always contain \x1b[200~.
    if "\x1b[200~" in data:
        return False

    # Quick check: release events with flag 2 contain ":3"
    # Format: \x1b[<codepoint>;<modifier>:3u
    return any(marker in data for marker in _RELEASE_MARKERS)


def is_key_repeat(data: str) -> bool:
    """Check if the input is a key repeat event.

    Only meaningful when Kitty keyboard protocol with flag 2 is active.
    """
    # Don't treat bracketed paste content as key repeat, even if it contains
    # patterns like ":2F". See is_key_release() for details.
    if "\x1b[200~" in data:
        return False

    return any(marker in data for marker in _REPEAT_MARKERS)


def _parse_event_type(event_type_str: str | None) -> KeyEventType:
    if not event_type_str:
        return "press"
    event_type = int(event_type_str)
    if event_type == 2:
        return "repeat"
    if event_type == 3:
        return "release"
    return "press"


_CSI_U_RE = re.compile(r"^\x1b\[(\d+)(?::(\d*))?(?::(\d+))?(?:;(\d+))?(?::(\d+))?u$")
_ARROW_RE = re.compile(r"^\x1b\[1;(\d+)(?::(\d+))?([ABCD])$")
_FUNC_RE = re.compile(r"^\x1b\[(\d+)(?:;(\d+))?(?::(\d+))?~$")
_HOME_END_RE = re.compile(r"^\x1b\[1;(\d+)(?::(\d+))?([HF])$")
_MODIFY_OTHER_KEYS_RE = re.compile(r"^\x1b\[27;(\d+);(\d+)~$")

_ARROW_CODES = {"A": -1, "B": -2, "C": -3, "D": -4}
_FUNC_CODES = {
    2: FUNCTIONAL_CODEPOINTS["insert"],
    3: FUNCTIONAL_CODEPOINTS["delete"],
    5: FUNCTIONAL_CODEPOINTS["pageUp"],
    6: FUNCTIONAL_CODEPOINTS["pageDown"],
    7: FUNCTIONAL_CODEPOINTS["home"],
    8: FUNCTIONAL_CODEPOINTS["end"],
}


def _parse_kitty_sequence(data: str) -> dict | None:
    """Parse a Kitty sequence into a record.

    Records: {"codepoint": int, "shiftedKey": int | None, "baseLayoutKey":
    int | None, "modifier": int, "eventType": KeyEventType}.

    CSI u format with alternate keys (flag 4)::

        \\x1b[<codepoint>u
        \\x1b[<codepoint>;<mod>u
        \\x1b[<codepoint>;<mod>:<event>u
        \\x1b[<codepoint>:<shifted>;<mod>u
        \\x1b[<codepoint>:<shifted>:<base>;<mod>u
        \\x1b[<codepoint>::<base>;<mod>u (no shifted key, only base)

    With flag 2, event type is appended after modifier colon: 1=press,
    2=repeat, 3=release. With flag 4, alternate keys are appended after
    codepoint with colons.
    """
    global _last_event_type

    csi_u_match = _CSI_U_RE.match(data)
    if csi_u_match:
        codepoint = int(csi_u_match.group(1))
        shifted_key = int(csi_u_match.group(2)) if csi_u_match.group(2) else None
        base_layout_key = int(csi_u_match.group(3)) if csi_u_match.group(3) else None
        mod_value = int(csi_u_match.group(4)) if csi_u_match.group(4) else 1
        event_type = _parse_event_type(csi_u_match.group(5))
        _last_event_type = event_type
        return {
            "codepoint": codepoint,
            "shiftedKey": shifted_key,
            "baseLayoutKey": base_layout_key,
            "modifier": mod_value - 1,
            "eventType": event_type,
        }

    # Arrow keys with modifier: \x1b[1;<mod>A/B/C/D or \x1b[1;<mod>:<event>A/B/C/D
    arrow_match = _ARROW_RE.match(data)
    if arrow_match:
        mod_value = int(arrow_match.group(1))
        event_type = _parse_event_type(arrow_match.group(2))
        _last_event_type = event_type
        return {
            "codepoint": _ARROW_CODES[arrow_match.group(3)],
            "shiftedKey": None,
            "baseLayoutKey": None,
            "modifier": mod_value - 1,
            "eventType": event_type,
        }

    # Functional keys: \x1b[<num>~ or \x1b[<num>;<mod>~ or \x1b[<num>;<mod>:<event>~
    func_match = _FUNC_RE.match(data)
    if func_match:
        key_num = int(func_match.group(1))
        mod_value = int(func_match.group(2)) if func_match.group(2) else 1
        event_type = _parse_event_type(func_match.group(3))
        codepoint = _FUNC_CODES.get(key_num)
        if codepoint is not None:
            _last_event_type = event_type
            return {
                "codepoint": codepoint,
                "shiftedKey": None,
                "baseLayoutKey": None,
                "modifier": mod_value - 1,
                "eventType": event_type,
            }

    # Home/End with modifier: \x1b[1;<mod>H/F or \x1b[1;<mod>:<event>H/F
    home_end_match = _HOME_END_RE.match(data)
    if home_end_match:
        mod_value = int(home_end_match.group(1))
        event_type = _parse_event_type(home_end_match.group(2))
        codepoint = FUNCTIONAL_CODEPOINTS["home"] if home_end_match.group(3) == "H" else FUNCTIONAL_CODEPOINTS["end"]
        _last_event_type = event_type
        return {
            "codepoint": codepoint,
            "shiftedKey": None,
            "baseLayoutKey": None,
            "modifier": mod_value - 1,
            "eventType": event_type,
        }

    return None


def _matches_kitty_sequence(data: str, expected_codepoint: int, expected_modifier: int) -> bool:
    parsed = _parse_kitty_sequence(data)
    if not parsed:
        return False
    actual_mod = parsed["modifier"] & ~LOCK_MASK
    expected_mod = expected_modifier & ~LOCK_MASK

    # Check if modifiers match
    if actual_mod != expected_mod:
        return False

    normalized_codepoint = _normalize_shifted_letter_identity_codepoint(
        _normalize_kitty_functional_codepoint(parsed["codepoint"]),
        parsed["modifier"],
    )
    normalized_expected_codepoint = _normalize_shifted_letter_identity_codepoint(
        _normalize_kitty_functional_codepoint(expected_codepoint),
        expected_modifier,
    )

    # Primary match: codepoint matches directly after normalizing functional keys
    if normalized_codepoint == normalized_expected_codepoint:
        return True

    # Alternate match: use base layout key for non-Latin keyboard layouts.
    # This allows Ctrl+С (Cyrillic) to match Ctrl+c (Latin) when terminal reports
    # the base layout key (the key in standard PC-101 layout).
    #
    # Only fall back to base layout key when the codepoint is NOT already a
    # recognized Latin letter (a-z) or symbol (e.g., /, -, [, ;, etc.).
    # When the codepoint is a recognized key, it is authoritative regardless
    # of physical key position. This prevents remapped layouts (Dvorak, Colemak,
    # xremap, etc.) from causing false matches: both letters and symbols move
    # to different physical positions, so Ctrl+K could falsely match Ctrl+V
    # (letter remapping) and Ctrl+/ could falsely match Ctrl+[ (symbol remapping)
    # if the base layout key were always considered.
    if parsed["baseLayoutKey"] is not None and parsed["baseLayoutKey"] == expected_codepoint:
        cp = normalized_codepoint
        is_latin_letter = 97 <= cp <= 122  # a-z
        is_known_symbol = _safe_chr(cp) in SYMBOL_KEYS
        if not is_latin_letter and not is_known_symbol:
            return True

    return False


def _parse_modify_other_keys_sequence(data: str) -> dict | None:
    """Records: {"codepoint": int, "modifier": int}."""
    match = _MODIFY_OTHER_KEYS_RE.match(data)
    if not match:
        return None
    mod_value = int(match.group(1))
    codepoint = int(match.group(2))
    return {"codepoint": codepoint, "modifier": mod_value - 1}


def _matches_modify_other_keys(data: str, expected_keycode: int, expected_modifier: int) -> bool:
    """Match xterm modifyOtherKeys format: CSI 27 ; modifiers ; keycode ~

    This is used by terminals when Kitty protocol is not enabled.
    Modifier values are 1-indexed: 2=shift, 3=alt, 5=ctrl, etc.
    """
    parsed = _parse_modify_other_keys_sequence(data)
    if not parsed:
        return False
    return parsed["codepoint"] == expected_keycode and parsed["modifier"] == expected_modifier


def _is_windows_terminal_session() -> bool:
    return bool(
        os.environ.get("WT_SESSION")
        and not os.environ.get("SSH_CONNECTION")
        and not os.environ.get("SSH_CLIENT")
        and not os.environ.get("SSH_TTY")
    )


def _matches_raw_backspace(data: str, expected_modifier: int) -> bool:
    """Raw 0x08 (BS) is ambiguous in legacy terminals.

    - Windows Terminal uses it for Ctrl+Backspace.
    - Some legacy terminals and tmux setups send it for plain Backspace.

    Prefer explicit Kitty / CSI-u / modifyOtherKeys sequences whenever they are
    available. Fall back to a Windows Terminal heuristic only for raw BS bytes.
    """
    if data == "\x7f":
        return expected_modifier == 0
    if data != "\x08":
        return False
    return expected_modifier == MODIFIERS["ctrl"] if _is_windows_terminal_session() else expected_modifier == 0


# =============================================================================
# Generic Key Matching
# =============================================================================


def _raw_ctrl_char(key: str) -> str | None:
    """Get the control character for a key.

    Uses the universal formula: code & 0x1f (mask to lower 5 bits)

    Works for:

    - Letters a-z → 1-26
    - Symbols [\\]_ → 27, 28, 29, 31
    - Also maps - to same as _ (same physical key on US keyboards)
    """
    char = key.lower()
    code = ord(char[0])
    if (97 <= code <= 122) or char in ("[", "\\", "]", "_"):
        return chr(code & 0x1F)
    # Handle - as _ (same physical key on US keyboards)
    if char == "-":
        return chr(31)  # Same as Ctrl+_
    return None


def _is_digit_key(key: str) -> bool:
    return "0" <= key <= "9"


def _matches_printable_modify_other_keys(data: str, expected_keycode: int, expected_modifier: int) -> bool:
    if expected_modifier == 0:
        return False
    parsed = _parse_modify_other_keys_sequence(data)
    if not parsed or parsed["modifier"] != expected_modifier:
        return False
    return _normalize_shifted_letter_identity_codepoint(
        parsed["codepoint"], parsed["modifier"]
    ) == _normalize_shifted_letter_identity_codepoint(expected_keycode, expected_modifier)


def _format_key_name_with_modifiers(key_name: str, modifier: int) -> str | None:
    mods: list[str] = []
    effective_mod = modifier & ~LOCK_MASK
    supported_modifier_mask = MODIFIERS["shift"] | MODIFIERS["ctrl"] | MODIFIERS["alt"] | MODIFIERS["super"]
    if (effective_mod & ~supported_modifier_mask) != 0:
        return None
    if effective_mod & MODIFIERS["shift"]:
        mods.append("shift")
    if effective_mod & MODIFIERS["ctrl"]:
        mods.append("ctrl")
    if effective_mod & MODIFIERS["alt"]:
        mods.append("alt")
    if effective_mod & MODIFIERS["super"]:
        mods.append("super")
    return f"{'+'.join(mods)}+{key_name}" if mods else key_name


def _parse_key_id(key_id: str) -> dict | None:
    parts = key_id.lower().split("+")
    key = parts[-1] if parts else ""
    if not key:
        return None
    return {
        "key": key,
        "ctrl": "ctrl" in parts,
        "shift": "shift" in parts,
        "alt": "alt" in parts,
        "super": "super" in parts,
    }


def matches_key(data: str, key_id: str) -> bool:  # noqa: C901
    """Match input data against a key identifier string.

    Supported key identifiers:

    - Single keys: "escape", "tab", "enter", "backspace", "delete", "home", "end", "space"
    - Arrow keys: "up", "down", "left", "right"
    - Ctrl combinations: "ctrl+c", "ctrl+z", etc.
    - Shift combinations: "shift+tab", "shift+enter"
    - Alt combinations: "alt+enter", "alt+backspace"
    - Super combinations: "super+k", "super+enter"
    - Combined modifiers: "shift+ctrl+p", "ctrl+alt+x", "ctrl+super+k"

    Use the Key helper: Key.ctrl("c"), Key.escape, Key.ctrl_shift("p"), Key.super("k")
    """
    parsed = _parse_key_id(key_id)
    if not parsed:
        return False

    key = parsed["key"]
    modifier = 0
    if parsed["shift"]:
        modifier |= MODIFIERS["shift"]
    if parsed["alt"]:
        modifier |= MODIFIERS["alt"]
    if parsed["ctrl"]:
        modifier |= MODIFIERS["ctrl"]
    if parsed["super"]:
        modifier |= MODIFIERS["super"]

    if key in ("escape", "esc"):
        if modifier != 0:
            return False
        return (
            data == "\x1b"
            or _matches_kitty_sequence(data, CODEPOINTS["escape"], 0)
            or _matches_modify_other_keys(data, CODEPOINTS["escape"], 0)
        )

    if key == "space":
        if not _kitty_protocol_active:
            if modifier == MODIFIERS["ctrl"] and data == "\x00":
                return True
            if modifier == MODIFIERS["alt"] and data == "\x1b ":
                return True
        if modifier == 0:
            return (
                data == " "
                or _matches_kitty_sequence(data, CODEPOINTS["space"], 0)
                or _matches_modify_other_keys(data, CODEPOINTS["space"], 0)
            )
        return _matches_kitty_sequence(data, CODEPOINTS["space"], modifier) or _matches_modify_other_keys(
            data, CODEPOINTS["space"], modifier
        )

    if key == "tab":
        if modifier == MODIFIERS["shift"]:
            return (
                data == "\x1b[Z"
                or _matches_kitty_sequence(data, CODEPOINTS["tab"], MODIFIERS["shift"])
                or _matches_modify_other_keys(data, CODEPOINTS["tab"], MODIFIERS["shift"])
            )
        if modifier == 0:
            return data == "\t" or _matches_kitty_sequence(data, CODEPOINTS["tab"], 0)
        return _matches_kitty_sequence(data, CODEPOINTS["tab"], modifier) or _matches_modify_other_keys(
            data, CODEPOINTS["tab"], modifier
        )

    if key in ("enter", "return"):
        if modifier == MODIFIERS["shift"]:
            # CSI u sequences (standard Kitty protocol)
            if _matches_kitty_sequence(data, CODEPOINTS["enter"], MODIFIERS["shift"]) or _matches_kitty_sequence(
                data, CODEPOINTS["kpEnter"], MODIFIERS["shift"]
            ):
                return True
            # xterm modifyOtherKeys format (fallback when Kitty protocol not enabled)
            if _matches_modify_other_keys(data, CODEPOINTS["enter"], MODIFIERS["shift"]):
                return True
            # When Kitty protocol is active, legacy sequences are custom terminal mappings
            # \x1b\r = Kitty's "map shift+enter send_text all \e\r"
            # \n = Ghostty's "keybind = shift+enter=text:\n"
            if _kitty_protocol_active:
                return data in ("\x1b\r", "\n")
            return False
        if modifier == MODIFIERS["alt"]:
            # CSI u sequences (standard Kitty protocol)
            if _matches_kitty_sequence(data, CODEPOINTS["enter"], MODIFIERS["alt"]) or _matches_kitty_sequence(
                data, CODEPOINTS["kpEnter"], MODIFIERS["alt"]
            ):
                return True
            # xterm modifyOtherKeys format (fallback when Kitty protocol not enabled)
            if _matches_modify_other_keys(data, CODEPOINTS["enter"], MODIFIERS["alt"]):
                return True
            # \x1b\r is alt+enter only in legacy mode (no Kitty protocol)
            # When Kitty protocol is active, alt+enter comes as CSI u sequence
            if not _kitty_protocol_active:
                return data == "\x1b\r"
            return False
        if modifier == 0:
            return (
                data == "\r"
                or (not _kitty_protocol_active and data == "\n")
                or data == "\x1bOM"  # SS3 M (numpad enter in some terminals)
                or _matches_kitty_sequence(data, CODEPOINTS["enter"], 0)
                or _matches_kitty_sequence(data, CODEPOINTS["kpEnter"], 0)
            )
        return (
            _matches_kitty_sequence(data, CODEPOINTS["enter"], modifier)
            or _matches_kitty_sequence(data, CODEPOINTS["kpEnter"], modifier)
            or _matches_modify_other_keys(data, CODEPOINTS["enter"], modifier)
        )

    if key == "backspace":
        if modifier == MODIFIERS["alt"]:
            if data in ("\x1b\x7f", "\x1b\x08"):
                return True
            return _matches_kitty_sequence(
                data, CODEPOINTS["backspace"], MODIFIERS["alt"]
            ) or _matches_modify_other_keys(data, CODEPOINTS["backspace"], MODIFIERS["alt"])
        if modifier == MODIFIERS["ctrl"]:
            # Legacy raw 0x08 is ambiguous: it can be Ctrl+Backspace on Windows
            # Terminal or plain Backspace on other terminals, while also
            # overlapping with Ctrl+H.
            if _matches_raw_backspace(data, MODIFIERS["ctrl"]):
                return True
            return _matches_kitty_sequence(
                data, CODEPOINTS["backspace"], MODIFIERS["ctrl"]
            ) or _matches_modify_other_keys(data, CODEPOINTS["backspace"], MODIFIERS["ctrl"])
        if modifier == 0:
            return (
                _matches_raw_backspace(data, 0)
                or _matches_kitty_sequence(data, CODEPOINTS["backspace"], 0)
                or _matches_modify_other_keys(data, CODEPOINTS["backspace"], 0)
            )
        return _matches_kitty_sequence(data, CODEPOINTS["backspace"], modifier) or _matches_modify_other_keys(
            data, CODEPOINTS["backspace"], modifier
        )

    if key == "insert":
        if modifier == 0:
            return _matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES["insert"]) or _matches_kitty_sequence(
                data, FUNCTIONAL_CODEPOINTS["insert"], 0
            )
        if _matches_legacy_modifier_sequence(data, "insert", modifier):
            return True
        return _matches_kitty_sequence(data, FUNCTIONAL_CODEPOINTS["insert"], modifier)

    if key == "delete":
        if modifier == 0:
            return _matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES["delete"]) or _matches_kitty_sequence(
                data, FUNCTIONAL_CODEPOINTS["delete"], 0
            )
        if _matches_legacy_modifier_sequence(data, "delete", modifier):
            return True
        return _matches_kitty_sequence(data, FUNCTIONAL_CODEPOINTS["delete"], modifier)

    if key == "clear":
        if modifier == 0:
            return _matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES["clear"])
        return _matches_legacy_modifier_sequence(data, "clear", modifier)

    if key == "home":
        if modifier == 0:
            return _matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES["home"]) or _matches_kitty_sequence(
                data, FUNCTIONAL_CODEPOINTS["home"], 0
            )
        if _matches_legacy_modifier_sequence(data, "home", modifier):
            return True
        return _matches_kitty_sequence(data, FUNCTIONAL_CODEPOINTS["home"], modifier)

    if key == "end":
        if modifier == 0:
            return _matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES["end"]) or _matches_kitty_sequence(
                data, FUNCTIONAL_CODEPOINTS["end"], 0
            )
        if _matches_legacy_modifier_sequence(data, "end", modifier):
            return True
        return _matches_kitty_sequence(data, FUNCTIONAL_CODEPOINTS["end"], modifier)

    if key == "pageup":
        if modifier == 0:
            return _matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES["pageUp"]) or _matches_kitty_sequence(
                data, FUNCTIONAL_CODEPOINTS["pageUp"], 0
            )
        if _matches_legacy_modifier_sequence(data, "pageUp", modifier):
            return True
        return _matches_kitty_sequence(data, FUNCTIONAL_CODEPOINTS["pageUp"], modifier)

    if key == "pagedown":
        if modifier == 0:
            return _matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES["pageDown"]) or _matches_kitty_sequence(
                data, FUNCTIONAL_CODEPOINTS["pageDown"], 0
            )
        if _matches_legacy_modifier_sequence(data, "pageDown", modifier):
            return True
        return _matches_kitty_sequence(data, FUNCTIONAL_CODEPOINTS["pageDown"], modifier)

    if key == "up":
        if modifier == MODIFIERS["alt"]:
            return data == "\x1bp" or _matches_kitty_sequence(data, ARROW_CODEPOINTS["up"], MODIFIERS["alt"])
        if modifier == 0:
            return _matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES["up"]) or _matches_kitty_sequence(
                data, ARROW_CODEPOINTS["up"], 0
            )
        if _matches_legacy_modifier_sequence(data, "up", modifier):
            return True
        return _matches_kitty_sequence(data, ARROW_CODEPOINTS["up"], modifier)

    if key == "down":
        if modifier == MODIFIERS["alt"]:
            return data == "\x1bn" or _matches_kitty_sequence(data, ARROW_CODEPOINTS["down"], MODIFIERS["alt"])
        if modifier == 0:
            return _matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES["down"]) or _matches_kitty_sequence(
                data, ARROW_CODEPOINTS["down"], 0
            )
        if _matches_legacy_modifier_sequence(data, "down", modifier):
            return True
        return _matches_kitty_sequence(data, ARROW_CODEPOINTS["down"], modifier)

    if key == "left":
        if modifier == MODIFIERS["alt"]:
            return (
                data == "\x1b[1;3D"
                or (not _kitty_protocol_active and data == "\x1bB")
                or data == "\x1bb"
                or _matches_kitty_sequence(data, ARROW_CODEPOINTS["left"], MODIFIERS["alt"])
            )
        if modifier == MODIFIERS["ctrl"]:
            return (
                data == "\x1b[1;5D"
                or _matches_legacy_modifier_sequence(data, "left", MODIFIERS["ctrl"])
                or _matches_kitty_sequence(data, ARROW_CODEPOINTS["left"], MODIFIERS["ctrl"])
            )
        if modifier == 0:
            return _matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES["left"]) or _matches_kitty_sequence(
                data, ARROW_CODEPOINTS["left"], 0
            )
        if _matches_legacy_modifier_sequence(data, "left", modifier):
            return True
        return _matches_kitty_sequence(data, ARROW_CODEPOINTS["left"], modifier)

    if key == "right":
        if modifier == MODIFIERS["alt"]:
            return (
                data == "\x1b[1;3C"
                or (not _kitty_protocol_active and data == "\x1bF")
                or data == "\x1bf"
                or _matches_kitty_sequence(data, ARROW_CODEPOINTS["right"], MODIFIERS["alt"])
            )
        if modifier == MODIFIERS["ctrl"]:
            return (
                data == "\x1b[1;5C"
                or _matches_legacy_modifier_sequence(data, "right", MODIFIERS["ctrl"])
                or _matches_kitty_sequence(data, ARROW_CODEPOINTS["right"], MODIFIERS["ctrl"])
            )
        if modifier == 0:
            return _matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES["right"]) or _matches_kitty_sequence(
                data, ARROW_CODEPOINTS["right"], 0
            )
        if _matches_legacy_modifier_sequence(data, "right", modifier):
            return True
        return _matches_kitty_sequence(data, ARROW_CODEPOINTS["right"], modifier)

    if key in ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12"):
        if modifier != 0:
            return False
        return _matches_legacy_sequence(data, LEGACY_KEY_SEQUENCES[key])

    # Handle single letter/digit keys and symbols
    if len(key) == 1 and (("a" <= key <= "z") or _is_digit_key(key) or key in SYMBOL_KEYS):
        codepoint = ord(key)
        raw_ctrl = _raw_ctrl_char(key)
        is_letter = "a" <= key <= "z"
        is_digit = _is_digit_key(key)

        # Legacy: ctrl+alt+key is ESC followed by the control character.
        # If that legacy form does not match, continue so CSI-u and
        # modifyOtherKeys sequences from tmux can still be recognized.
        if (
            modifier == MODIFIERS["ctrl"] + MODIFIERS["alt"]
            and not _kitty_protocol_active
            and raw_ctrl
            and data == f"\x1b{raw_ctrl}"
        ):
            return True

        # Legacy: alt+printable key is ESC followed by the key
        if (
            modifier == MODIFIERS["alt"]
            and not _kitty_protocol_active
            and (is_letter or is_digit or key in SYMBOL_KEYS)
            and data == f"\x1b{key}"
        ):
            return True

        if modifier == MODIFIERS["ctrl"]:
            # Legacy: ctrl+key sends the control character
            if raw_ctrl and data == raw_ctrl:
                return True
            return _matches_kitty_sequence(data, codepoint, MODIFIERS["ctrl"]) or _matches_printable_modify_other_keys(
                data, codepoint, MODIFIERS["ctrl"]
            )

        if modifier == MODIFIERS["shift"] + MODIFIERS["ctrl"]:
            return _matches_kitty_sequence(
                data, codepoint, MODIFIERS["shift"] + MODIFIERS["ctrl"]
            ) or _matches_printable_modify_other_keys(data, codepoint, MODIFIERS["shift"] + MODIFIERS["ctrl"])

        if modifier == MODIFIERS["shift"]:
            # Legacy: shift+letter produces uppercase
            if is_letter and data == key.upper():
                return True
            return _matches_kitty_sequence(data, codepoint, MODIFIERS["shift"]) or _matches_printable_modify_other_keys(
                data, codepoint, MODIFIERS["shift"]
            )

        if modifier != 0:
            return _matches_kitty_sequence(data, codepoint, modifier) or _matches_printable_modify_other_keys(
                data, codepoint, modifier
            )

        # Check both raw char and Kitty sequence (needed for release events)
        return data == key or _matches_kitty_sequence(data, codepoint, 0)

    return False


def _format_parsed_key(codepoint: int, modifier: int, base_layout_key: int | None = None) -> str | None:
    normalized_codepoint = _normalize_kitty_functional_codepoint(codepoint)
    identity_codepoint = _normalize_shifted_letter_identity_codepoint(normalized_codepoint, modifier)

    # Use base layout key only when codepoint is not a recognized Latin
    # letter (a-z), digit (0-9), or symbol (/, -, [, ;, etc.). For those,
    # the codepoint is authoritative regardless of physical key position.
    # This prevents remapped layouts (Dvorak, Colemak, xremap, etc.) from
    # reporting the wrong key name based on the QWERTY physical position.
    is_latin_letter = 97 <= identity_codepoint <= 122  # a-z
    is_digit = 48 <= identity_codepoint <= 57  # 0-9
    is_known_symbol = _safe_chr(identity_codepoint) in SYMBOL_KEYS
    if is_latin_letter or is_digit or is_known_symbol:
        effective_codepoint = identity_codepoint
    else:
        effective_codepoint = base_layout_key if base_layout_key is not None else identity_codepoint

    key_name: str | None = None
    if effective_codepoint == CODEPOINTS["escape"]:
        key_name = "escape"
    elif effective_codepoint == CODEPOINTS["tab"]:
        key_name = "tab"
    elif effective_codepoint in (CODEPOINTS["enter"], CODEPOINTS["kpEnter"]):
        key_name = "enter"
    elif effective_codepoint == CODEPOINTS["space"]:
        key_name = "space"
    elif effective_codepoint == CODEPOINTS["backspace"]:
        key_name = "backspace"
    elif effective_codepoint == FUNCTIONAL_CODEPOINTS["delete"]:
        key_name = "delete"
    elif effective_codepoint == FUNCTIONAL_CODEPOINTS["insert"]:
        key_name = "insert"
    elif effective_codepoint == FUNCTIONAL_CODEPOINTS["home"]:
        key_name = "home"
    elif effective_codepoint == FUNCTIONAL_CODEPOINTS["end"]:
        key_name = "end"
    elif effective_codepoint == FUNCTIONAL_CODEPOINTS["pageUp"]:
        key_name = "pageUp"
    elif effective_codepoint == FUNCTIONAL_CODEPOINTS["pageDown"]:
        key_name = "pageDown"
    elif effective_codepoint == ARROW_CODEPOINTS["up"]:
        key_name = "up"
    elif effective_codepoint == ARROW_CODEPOINTS["down"]:
        key_name = "down"
    elif effective_codepoint == ARROW_CODEPOINTS["left"]:
        key_name = "left"
    elif effective_codepoint == ARROW_CODEPOINTS["right"]:
        key_name = "right"
    elif (
        48 <= effective_codepoint <= 57
        or 97 <= effective_codepoint <= 122
        or _safe_chr(effective_codepoint) in SYMBOL_KEYS
    ):
        key_name = chr(effective_codepoint)

    if not key_name:
        return None
    return _format_key_name_with_modifiers(key_name, modifier)


def parse_key(data: str) -> str | None:
    """Parse input data and return the key identifier if recognized.

    Returns a key identifier string (e.g., "ctrl+c") or None.
    """
    kitty = _parse_kitty_sequence(data)
    if kitty:
        return _format_parsed_key(kitty["codepoint"], kitty["modifier"], kitty["baseLayoutKey"])

    modify_other_keys = _parse_modify_other_keys_sequence(data)
    if modify_other_keys:
        return _format_parsed_key(modify_other_keys["codepoint"], modify_other_keys["modifier"])

    # Mode-aware legacy sequences
    # When Kitty protocol is active, ambiguous sequences are interpreted as custom terminal mappings:
    # - \x1b\r = shift+enter (Kitty mapping), not alt+enter
    # - \n = shift+enter (Ghostty mapping)
    if _kitty_protocol_active and data in ("\x1b\r", "\n"):
        return "shift+enter"

    legacy_sequence_key_id = LEGACY_SEQUENCE_KEY_IDS.get(data)
    if legacy_sequence_key_id:
        return legacy_sequence_key_id

    # Legacy sequences (used when Kitty protocol is not active, or for unambiguous sequences)
    if data == "\x1b":
        return "escape"
    if data == "\x1c":
        return "ctrl+\\"
    if data == "\x1d":
        return "ctrl+]"
    if data == "\x1f":
        return "ctrl+-"
    if data == "\x1b\x1b":
        return "ctrl+alt+["
    if data == "\x1b\x1c":
        return "ctrl+alt+\\"
    if data == "\x1b\x1d":
        return "ctrl+alt+]"
    if data == "\x1b\x1f":
        return "ctrl+alt+-"
    if data == "\t":
        return "tab"
    if data == "\r" or (not _kitty_protocol_active and data == "\n") or data == "\x1bOM":
        return "enter"
    if data == "\x00":
        return "ctrl+space"
    if data == " ":
        return "space"
    if data == "\x7f":
        return "backspace"
    if data == "\x08":
        return "ctrl+backspace" if _is_windows_terminal_session() else "backspace"
    if data == "\x1b[Z":
        return "shift+tab"
    if not _kitty_protocol_active and data == "\x1b\r":
        return "alt+enter"
    if not _kitty_protocol_active and data == "\x1b ":
        return "alt+space"
    if data in ("\x1b\x7f", "\x1b\x08"):
        return "alt+backspace"
    if not _kitty_protocol_active and data == "\x1bB":
        return "alt+left"
    if not _kitty_protocol_active and data == "\x1bF":
        return "alt+right"
    if not _kitty_protocol_active and len(data) == 2 and data[0] == "\x1b":
        code = ord(data[1])
        if 1 <= code <= 26:
            return f"ctrl+alt+{chr(code + 96)}"
        # Legacy alt+letter/digit/symbol (ESC followed by the key)
        key = chr(code)
        if (97 <= code <= 122) or (48 <= code <= 57) or key in SYMBOL_KEYS:
            return f"alt+{key}"
    if data == "\x1b[A":
        return "up"
    if data == "\x1b[B":
        return "down"
    if data == "\x1b[C":
        return "right"
    if data == "\x1b[D":
        return "left"
    if data in ("\x1b[H", "\x1bOH"):
        return "home"
    if data in ("\x1b[F", "\x1bOF"):
        return "end"
    if data == "\x1b[3~":
        return "delete"
    if data == "\x1b[5~":
        return "pageUp"
    if data == "\x1b[6~":
        return "pageDown"

    # Raw Ctrl+letter
    if len(data) == 1:
        code = ord(data)
        if 1 <= code <= 26:
            return f"ctrl+{chr(code + 96)}"
        if 32 <= code <= 126:
            return data

    return None


# =============================================================================
# Kitty CSI-u Printable Decoding
# =============================================================================

KITTY_PRINTABLE_ALLOWED_MODIFIERS = MODIFIERS["shift"] | LOCK_MASK


def decode_kitty_printable(data: str) -> str | None:
    """Decode a Kitty CSI-u sequence into a printable character, if applicable.

    When Kitty keyboard protocol flag 1 (disambiguate) is active, terminals send
    CSI-u sequences for all keys, including plain printable characters. This
    function extracts the printable character from such sequences.

    Only accepts plain or Shift-modified keys. Rejects Ctrl, Alt, and unsupported
    modifier combinations (those are handled by keybinding matching instead).
    Prefers the shifted keycode when Shift is held and a shifted key is reported.
    """
    match = _CSI_U_RE.match(data)
    if not match:
        return None

    # CSI-u groups: <codepoint>[:<shifted>[:<base>]];<mod>[:<event>]u
    codepoint = int(match.group(1))

    shifted_key = int(match.group(2)) if match.group(2) else None
    mod_value = int(match.group(4)) if match.group(4) else 1
    # Modifiers are 1-indexed in CSI-u; normalize to our bitmask.
    modifier = mod_value - 1

    # Only accept printable CSI-u input for plain or Shift-modified text keys.
    # Reject unsupported modifier bits (e.g. Super/Meta) to avoid inserting
    # characters from modifier-only terminal events.
    if (modifier & ~KITTY_PRINTABLE_ALLOWED_MODIFIERS) != 0:
        return None
    if modifier & (MODIFIERS["alt"] | MODIFIERS["ctrl"]):
        return None

    # Prefer the shifted keycode when Shift is held.
    effective_codepoint = codepoint
    if modifier & MODIFIERS["shift"] and shifted_key is not None:
        effective_codepoint = shifted_key
    effective_codepoint = _normalize_kitty_functional_codepoint(effective_codepoint)
    # Drop control characters or invalid codepoints.
    if effective_codepoint < 32:
        return None

    try:
        return chr(effective_codepoint)
    except ValueError:
        return None


def _decode_modify_other_keys_printable(data: str) -> str | None:
    parsed = _parse_modify_other_keys_sequence(data)
    if not parsed:
        return None
    modifier = parsed["modifier"] & ~LOCK_MASK
    if (modifier & ~MODIFIERS["shift"]) != 0:
        return None
    if parsed["codepoint"] < 32:
        return None

    try:
        return chr(parsed["codepoint"])
    except ValueError:
        return None


def decode_printable_key(data: str) -> str | None:
    return decode_kitty_printable(data) or _decode_modify_other_keys_printable(data)
