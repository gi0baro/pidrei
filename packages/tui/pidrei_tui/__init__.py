"""Mirror of pi tui src/index.ts (re-exports grow as modules are ported)."""

from .keys import is_kitty_protocol_active, set_kitty_protocol_active
from .stdin_buffer import StdinBuffer
from .terminal import (
    ProcessTerminal,
    Terminal,
    is_apple_terminal_session,
    normalize_apple_terminal_input,
    parse_keyboard_protocol_negotiation_sequence,
)


__all__ = [
    "ProcessTerminal",
    "StdinBuffer",
    "Terminal",
    "is_apple_terminal_session",
    "is_kitty_protocol_active",
    "normalize_apple_terminal_input",
    "parse_keyboard_protocol_negotiation_sequence",
    "set_kitty_protocol_active",
]
