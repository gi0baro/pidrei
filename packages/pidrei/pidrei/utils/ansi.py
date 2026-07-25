"""Mirror of pi coding-agent src/utils/ansi.ts (derived from ansi-regex/strip-ansi, MIT)."""

import re


# Valid string terminator sequences are BEL, ESC\\, and 0x9c
_ST = "(?:\u0007|\u001b\u005c|\u009c)"
# OSC sequences only: ESC ] ... ST (non-greedy until the first ST)
_OSC = f"(?:\u001b\\][\\s\\S]*?{_ST})"
# CSI and related: ESC/C1, optional intermediates, optional params (supports ; and :) then final byte
_CSI = "[\u001b\u009b][\\[\\]()#;?]*(?:\\d{1,4}(?:[;:]\\d{0,4})*)?[\\dA-PR-TZcf-nq-uy=><~]"

_ANSI_RE = re.compile(f"{_OSC}|{_CSI}")


def strip_ansi(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected a `string`, got `{type(value).__name__}`")

    # Fast path: ANSI codes require ESC (7-bit) or CSI (8-bit) introducer
    if "\u001b" not in value and "\u009b" not in value:
        return value

    return _ANSI_RE.sub("", value)
