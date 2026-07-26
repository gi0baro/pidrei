"""Mirror of pi coding-agent src/modes/interactive/components/keybinding-hints.ts.

Utilities for formatting keybinding hints in the UI.
"""

import sys

from pidrei_tui import get_keybindings

from ..theme import theme


_DARWIN = sys.platform == "darwin"


def _format_key_part(part: str, options: dict) -> str:
    display_part = "option" if _DARWIN and part.lower() == "alt" else part
    if options.get("capitalize"):
        return display_part[:1].upper() + display_part[1:]
    return display_part


def format_key_text(key: str, options: dict | None = None) -> str:
    options = options or {}
    return "/".join("+".join(_format_key_part(part, options) for part in k.split("+")) for k in key.split("/"))


def _format_keys(keys: list, options: dict | None = None) -> str:
    if not keys:
        return ""
    return format_key_text("/".join(keys), options)


def key_text(keybinding: str) -> str:
    return _format_keys(get_keybindings().get_keys(keybinding))


def key_display_text(keybinding: str) -> str:
    return _format_keys(get_keybindings().get_keys(keybinding), {"capitalize": True})


def key_hint(keybinding: str, description: str) -> str:
    return theme.fg("dim", key_text(keybinding)) + theme.fg("muted", f" {description}")


def raw_key_hint(key: str, description: str) -> str:
    return theme.fg("dim", format_key_text(key)) + theme.fg("muted", f" {description}")
