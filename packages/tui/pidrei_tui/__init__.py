"""Mirror of pi tui src/index.ts (re-exports grow as modules are ported)."""

from .keybindings import (
    TUI_KEYBINDINGS,
    KeybindingsManager,
    get_keybindings,
    set_keybindings,
)
from .keys import (
    Key,
    decode_kitty_printable,
    is_key_release,
    is_key_repeat,
    is_kitty_protocol_active,
    matches_key,
    parse_key,
    set_kitty_protocol_active,
)
from .stdin_buffer import StdinBuffer
from .terminal import (
    ProcessTerminal,
    Terminal,
    is_apple_terminal_session,
    normalize_apple_terminal_input,
    parse_keyboard_protocol_negotiation_sequence,
)


__all__ = [
    "TUI_KEYBINDINGS",
    "Key",
    "KeybindingsManager",
    "ProcessTerminal",
    "StdinBuffer",
    "Terminal",
    "decode_kitty_printable",
    "get_keybindings",
    "is_apple_terminal_session",
    "is_key_release",
    "is_key_repeat",
    "is_kitty_protocol_active",
    "matches_key",
    "normalize_apple_terminal_input",
    "parse_key",
    "parse_keyboard_protocol_negotiation_sequence",
    "set_keybindings",
    "set_kitty_protocol_active",
]
