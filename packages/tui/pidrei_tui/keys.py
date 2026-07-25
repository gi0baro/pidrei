"""Mirror of pi tui src/keys.ts.

Only the global Kitty-protocol flag is ported so far; the key parsing API
(`parse_key`/`matches_key`/`Key`) lands with the input-stack slice.
"""

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
