"""Minimal chalk stand-in for CLI output styling.

pi uses chalk, which enables ANSI styling based on stdout color support and
disables it when output is piped. This mirrors that single global detection:
styling is applied only when stdout is a TTY and NO_COLOR is unset
(FORCE_COLOR overrides both, like chalk).
"""

import os
import sys


def _color_enabled() -> bool:
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _style(text: str, open_code: str, close_code: str) -> str:
    if not _color_enabled():
        return text
    return f"\x1b[{open_code}m{text}\x1b[{close_code}m"


def bold(text: str) -> str:
    return _style(text, "1", "22")


def dim(text: str) -> str:
    return _style(text, "2", "22")


def red(text: str) -> str:
    return _style(text, "31", "39")


def yellow(text: str) -> str:
    return _style(text, "33", "39")


def cyan(text: str) -> str:
    return _style(text, "36", "39")
