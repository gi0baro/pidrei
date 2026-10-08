"""Mirror of pi coding-agent src/utils/ansi.ts (derived from ansi-regex/strip-ansi, MIT)."""

import re


# Regex-level escapes throughout (raw strings), as in pi's pattern strings: a literal backslash
# in the pattern would escape the character after it.

# Valid string terminator sequences are BEL, ESC\, and 0x9c
_ST = r"(?:\x07|\x1b\\|\x9c)"

# OSC sequences: ESC ] ... ST
_OSC_START = r"\x1b\]"

# CSI and related: ESC/C1, optional intermediates, optional params (supports ; and :), then final byte
_CSI_START = r"[\x1b\x9b][\[\]()#;?]*(?:\d{1,4}(?:[;:]\d{0,4})*)?"
_CSI_FINAL = r"[\dA-PR-TZcf-nq-uy=><~]"

# Complete sequences. OSC is non-greedy until the first ST.
_ANSI_RE = re.compile(rf"(?:{_OSC_START}[\s\S]*?{_ST})|{_CSI_START}{_CSI_FINAL}")

# Unfinished sequence at the end of the text: OSC without its ST (a trailing ESC may start ESC\),
# or CSI without its final byte. `\Z` is JavaScript's `$`; Python's also matches before a final "\n".
_UNFINISHED_ANSI_AT_END_RE = re.compile(rf"(?:{_OSC_START}(?:[^\x07\x9c\x1b]|\x1b(?!\\))*|{_CSI_START})\Z")

# Longest unfinished sequence held back while streaming. Longer ones are processed as-is.
_MAX_PENDING_ANSI_LENGTH = 256


def split_incomplete_ansi_suffix(value: str) -> tuple[str, str]:
    """Split streamed text into a part that is safe to pass to strip_ansi now and a trailing
    unfinished escape sequence that should be prepended to the next chunk.

    Returns `(complete, pending)`.
    """
    if "\u001b" not in value and "\u009b" not in value:
        return value, ""
    window_start = max(0, len(value) - _MAX_PENDING_ANSI_LENGTH)
    match = _UNFINISHED_ANSI_AT_END_RE.search(value[window_start:])
    if match is None:
        return value, ""
    split_at = window_start + match.start()
    return value[:split_at], value[split_at:]


def strip_ansi(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected a `string`, got `{type(value).__name__}`")

    # Fast path: ANSI codes require ESC (7-bit) or CSI (8-bit) introducer
    if "\u001b" not in value and "\u009b" not in value:
        return value

    return _ANSI_RE.sub("", value)
