"""Mirror of pi tui src/utils.ts.

Text measurement, ANSI-aware wrapping/truncation/slicing for the renderer.

Port notes (pi runs on JS Intl/Unicode engines Python lacks in the stdlib):

- Grapheme segmentation: ``Intl.Segmenter`` (grapheme granularity) → the
  pure-Python ``grapheme`` package (UAX #29 extended clusters incl. ZWJ
  sequences, flag pairs, Thai/Lao AM joining). pi's shared word segmenter is
  not ported yet — it lands with the editor slice.
- ``get-east-asian-width`` → ``unicodedata.east_asian_width`` (W/F → 2,
  everything else → 1; ambiguous stays narrow like pi's default).
- JS ``\\p{...}`` property regexes → ``unicodedata.category`` checks plus a
  Default_Ignorable_Code_Point range table (stdlib has no DI property).
- ``\\p{RGI_Emoji}`` (a sequence property) has no Python equivalent; the
  check is approximated: VS16 or ZWJ present, flag pairs, skin-tone and
  keycap and tag sequences → emoji (width 2). Single-codepoint emoji get
  width 2 through their East_Asian_Width=W anyway.
- ``cjkBreakRegex`` (Script_Extensions) → explicit codepoint ranges for
  Han/Hiragana/Katakana/Hangul/Bopomofo blocks and CJK punctuation.
- JS ``String.length`` is UTF-16 units; where the distinction matters
  (couldBeEmoji's ``length > 2``) the UTF-16 length is computed explicitly.
- The pooled ``AnsiCodeTracker`` used by ``extract_segments`` is per-thread
  (``threading.local``): tonio may run layout work on parallel workers.
- The width cache is a plain dict with FIFO eviction under a lock; reads are
  lock-free (a racing double-compute is benign).
"""

import re
import threading
import unicodedata

import grapheme as grapheme_lib


# =============================================================================
# Character classification
# =============================================================================

_PRINTABLE_ASCII_RE = re.compile(r"[\x20-\x7e]*\Z")

# Default_Ignorable_Code_Point ranges (UCD PropList).
_DEFAULT_IGNORABLE_RANGES = (
    (0x00AD, 0x00AD),
    (0x034F, 0x034F),
    (0x061C, 0x061C),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)

cjk_break_regex = re.compile(
    "["
    "ᄀ-ᇿ"  # Hangul Jamo
    "⺀-⿟"  # CJK radicals
    "　-〿"  # CJK symbols and punctuation
    "ぁ-ヿ"  # Hiragana, Katakana
    "㄀-ㄯ"  # Bopomofo
    "㄰-㆏"  # Hangul compatibility Jamo
    "ㆠ-ㆿ"  # Bopomofo extended
    "㇀-㇯"  # CJK strokes
    "ㇰ-ㇿ"  # Katakana phonetic extensions
    "㈀-㏿"  # Enclosed CJK, CJK compatibility
    "㐀-䶿"  # CJK extension A
    "一-鿿"  # CJK unified ideographs
    "ꥠ-꥿"  # Hangul Jamo extended A
    "가-퟿"  # Hangul syllables, Jamo extended B
    "豈-﫿"  # CJK compatibility ideographs
    "︰-﹏"  # CJK compatibility forms
    "ｦ-ﾟ"  # Halfwidth Katakana
    "ﾠ-ￜ"  # Halfwidth Hangul
    "\U00020000-\U0003ffff"  # CJK extensions B+
    "]"
)


def _is_default_ignorable(cp: int) -> bool:
    for start, end in _DEFAULT_IGNORABLE_RANGES:
        if start <= cp <= end:
            return True
        if cp < start:
            return False
    return False


def _utf16_length(s: str) -> int:
    return sum(2 if ord(char) > 0xFFFF else 1 for char in s)


def _could_be_emoji(segment: str) -> bool:
    """Check if a grapheme cluster (after segmentation) could possibly be an RGI emoji.

    This is a fast heuristic to avoid the expensive emoji-sequence test. The
    tested Unicode blocks are deliberately broad to account for future
    Unicode additions.
    """
    cp = ord(segment[0])
    return (
        (0x1F000 <= cp <= 0x1FBFF)  # Emoji and Pictograph
        or (0x2300 <= cp <= 0x23FF)  # Misc technical
        or (0x2600 <= cp <= 0x27BF)  # Misc symbols, dingbats
        or (0x2B50 <= cp <= 0x2B55)  # Specific stars/circles
        or "️" in segment  # Contains VS16 (emoji presentation selector)
        or _utf16_length(segment) > 2  # Multi-codepoint sequences (ZWJ, skin tones, etc.)
    )


def _is_emoji_sequence(segment: str) -> bool:
    """Approximation of JS ``/^\\p{RGI_Emoji}$/v`` (see module docstring)."""
    if "️" in segment or "‍" in segment:
        return True
    codepoints = [ord(char) for char in segment]
    if len(codepoints) < 2:
        return False
    if all(0x1F1E6 <= cp <= 0x1F1FF for cp in codepoints):
        return True  # flag pairs
    if any(0x1F3FB <= cp <= 0x1F3FF for cp in codepoints[1:]):
        return True  # skin-tone sequences
    if codepoints[-1] == 0x20E3:
        return True  # keycap sequences
    # Tag sequences (subdivision flags)
    return 0xE0020 <= codepoints[-1] <= 0xE007F


def _is_zero_width_cluster(segment: str) -> bool:
    # JS: /^(?:\p{Default_Ignorable_Code_Point}|\p{Control}|\p{Mark}|\p{Surrogate})+$/v
    if not segment:
        return False
    for char in segment:
        category = unicodedata.category(char)
        if category in ("Cc", "Cs") or category.startswith("M") or _is_default_ignorable(ord(char)):
            continue
        return False
    return True


def _strip_leading_non_printing(segment: str) -> str:
    # JS: /^[\p{Default_Ignorable_Code_Point}\p{Control}\p{Format}\p{Mark}\p{Surrogate}]+/v
    for index, char in enumerate(segment):
        category = unicodedata.category(char)
        if category in ("Cc", "Cf", "Cs") or category.startswith("M") or _is_default_ignorable(ord(char)):
            continue
        return segment[index:]
    return ""


def _east_asian_width(cp: int) -> int:
    return 2 if unicodedata.east_asian_width(chr(cp)) in ("W", "F") else 1


def _is_printable_ascii(s: str) -> bool:
    return _PRINTABLE_ASCII_RE.match(s) is not None


# =============================================================================
# Width measurement
# =============================================================================

_WIDTH_CACHE_SIZE = 512
_width_cache: dict[str, int] = {}
_width_cache_lock = threading.Lock()


def _grapheme_width(segment: str) -> int:
    """Calculate the terminal width of a single grapheme cluster.

    Based on code from the string-width library, but includes a
    possible-emoji check to avoid running the emoji-sequence test
    unnecessarily.
    """
    if segment == "\t":
        return 3

    # Zero-width clusters
    if _is_zero_width_cluster(segment):
        return 0

    # Emoji check with pre-filter
    if _could_be_emoji(segment) and _is_emoji_sequence(segment):
        return 2

    # Get base visible codepoint
    base = _strip_leading_non_printing(segment)
    if not base:
        return 0
    cp = ord(base[0])

    # Regional indicator symbols (U+1F1E6..U+1F1FF) are often rendered as
    # full-width emoji in terminals, even when isolated during streaming.
    # Keep width conservative (2) to avoid terminal auto-wrap drift artifacts.
    if 0x1F1E6 <= cp <= 0x1F1FF:
        return 2

    width = _east_asian_width(cp)

    # Trailing halfwidth/fullwidth forms and AM vowels that segment with a base.
    if len(segment) > 1:
        for char in segment[1:]:
            c = ord(char)
            if 0xFF00 <= c <= 0xFFEF:
                width += _east_asian_width(c)
            elif c in (0x0E33, 0x0EB3):
                width += 1

    return width


def visible_width(s: str) -> int:
    """Calculate the visible width of a string in terminal columns."""
    if len(s) == 0:
        return 0

    # Fast path: pure ASCII printable
    if _is_printable_ascii(s):
        return len(s)

    # Check cache
    cached = _width_cache.get(s)
    if cached is not None:
        return cached

    # Normalize: tabs to 3 spaces, strip ANSI escape codes
    clean = s
    if "\t" in clean:
        clean = clean.replace("\t", "   ")
    if "\x1b" in clean:
        # Strip supported ANSI/OSC/APC escape sequences in one pass.
        # This covers CSI styling/cursor codes, OSC hyperlinks and prompt markers,
        # and APC sequences like CURSOR_MARKER.
        stripped: list[str] = []
        i = 0
        while i < len(clean):
            ansi = extract_ansi_code(clean, i)
            if ansi:
                i += ansi["length"]
                continue
            stripped.append(clean[i])
            i += 1
        clean = "".join(stripped)

    # Calculate width
    width = 0
    for segment in grapheme_lib.graphemes(clean):
        width += _grapheme_width(segment)

    # Cache result
    with _width_cache_lock:
        if len(_width_cache) >= _WIDTH_CACHE_SIZE:
            try:
                del _width_cache[next(iter(_width_cache))]
            except StopIteration, KeyError, RuntimeError:
                pass
        _width_cache[s] = width

    return width


THAI_LAO_AM_RE = re.compile("[ำຳ]")


def normalize_terminal_output(s: str) -> str:
    """Normalize text for terminal output without changing logical editor content.

    Some terminals render precomposed Thai/Lao AM vowels inconsistently during
    differential repaint. Their compatibility decompositions have the same cell
    width but avoid stale-cell artifacts in terminal renderers. Visible tabs are
    expanded to the fixed width used by layout so terminal tab stops cannot wrap
    a logical line, while tabs inside terminal string sequences stay untouched.
    """
    normalized = s
    if THAI_LAO_AM_RE.search(normalized):
        normalized = THAI_LAO_AM_RE.sub(
            lambda match: "ํา" if match.group(0) == "ำ" else "ໍາ",
            normalized,
        )
    if "\t" not in normalized:
        return normalized

    result: list[str] = []
    i = 0
    while i < len(normalized):
        ansi = extract_ansi_code(normalized, i)
        if ansi:
            result.append(ansi["code"])
            i += ansi["length"]
            continue
        result.append("   " if normalized[i] == "\t" else normalized[i])
        i += 1
    return "".join(result)


# =============================================================================
# ANSI escape extraction and tracking
# =============================================================================

# CSI sequence: ESC [ ... m/G/K/H/J
_ANSI_CSI_RE = re.compile(r"\x1b\[[^mGKHJ]*[mGKHJ]")
# OSC sequence: ESC ] ... BEL or ESC ] ... ST (ESC \) — used for hyperlinks
# (OSC 8), window titles, etc. A lone ESC inside the body does not terminate.
_ANSI_OSC_RE = re.compile(r"\x1b\](?:[^\x07\x1b]|\x1b(?!\\))*(?:\x07|\x1b\\)")
# APC sequence: ESC _ ... BEL or ESC _ ... ST — cursor marker etc.
_ANSI_APC_RE = re.compile(r"\x1b_(?:[^\x07\x1b]|\x1b(?!\\))*(?:\x07|\x1b\\)")

_SGR_PARAMS_RE = re.compile(r"\x1b\[([\d;]*)m")


def extract_ansi_code(s: str, pos: int) -> dict | None:
    """Extract ANSI escape sequences from a string at the given position.

    Returns ``{"code": str, "length": int}`` or None.
    """
    if pos >= len(s) or s[pos] != "\x1b":
        return None

    nxt = s[pos + 1 : pos + 2]
    if nxt == "[":
        pattern = _ANSI_CSI_RE
    elif nxt == "]":
        pattern = _ANSI_OSC_RE
    elif nxt == "_":
        pattern = _ANSI_APC_RE
    else:
        return None

    match = pattern.match(s, pos)
    if not match:
        return None
    return {"code": match.group(0), "length": match.end() - pos}


# Sentinel: the code is not an OSC 8 hyperlink sequence at all
# (parse result None means "close hyperlink").
_NOT_OSC8 = object()


def _parse_osc8_hyperlink(ansi_code: str):
    if not ansi_code.startswith("\x1b]8;"):
        return _NOT_OSC8

    terminator = "\x07" if ansi_code.endswith("\x07") else "\x1b\\"
    body = ansi_code[4 : -1 if terminator == "\x07" else -2]
    separator_index = body.find(";")
    if separator_index == -1:
        return _NOT_OSC8

    params = body[:separator_index]
    url = body[separator_index + 1 :]
    if not url:
        return None
    return {"params": params, "url": url, "terminator": terminator}


def _format_osc8_hyperlink(hyperlink: dict) -> str:
    return f"\x1b]8;{hyperlink['params']};{hyperlink['url']}{hyperlink['terminator']}"


def _format_osc8_close(terminator: str) -> str:
    return f"\x1b]8;;{terminator}"


class AnsiCodeTracker:
    """Track active ANSI SGR codes to preserve styling across line breaks."""

    def __init__(self) -> None:
        # Track individual attributes separately so we can reset them specifically
        self._bold = False
        self._dim = False
        self._italic = False
        self._underline = False
        self._blink = False
        self._inverse = False
        self._hidden = False
        self._strikethrough = False
        self._fg_color: str | None = None  # Full code like "31" or "38;5;240"
        self._bg_color: str | None = None  # Full code like "41" or "48;5;240"
        self._active_hyperlink: dict | None = None

    def process(self, ansi_code: str) -> None:
        # OSC 8 hyperlink: \x1b]8;;<url>\x1b\\ (open) or \x1b]8;;\x1b\\ (close).
        # Preserve the original terminator because some terminals only make
        # BEL-terminated links clickable. OAuth login URLs use BEL, so reopening
        # wrapped lines with ST made only the first physical line clickable there.
        hyperlink = _parse_osc8_hyperlink(ansi_code)
        if hyperlink is not _NOT_OSC8:
            self._active_hyperlink = hyperlink
            return

        if not ansi_code.endswith("m"):
            return

        # Extract the parameters between \x1b[ and m
        match = _SGR_PARAMS_RE.match(ansi_code)
        if not match:
            return

        params = match.group(1)
        if params in ("", "0"):
            # Full reset
            self._reset()
            return

        # Parse parameters (can be semicolon-separated)
        parts = params.split(";")
        i = 0
        while i < len(parts):
            code = int(parts[i]) if parts[i] else 0

            # Handle 256-color and RGB codes which consume multiple parameters
            if code in (38, 48):
                # 38;5;N (256 color fg) or 38;2;R;G;B (RGB fg)
                # 48;5;N (256 color bg) or 48;2;R;G;B (RGB bg)
                if i + 2 < len(parts) and parts[i + 1] == "5":
                    color_code = f"{parts[i]};{parts[i + 1]};{parts[i + 2]}"
                    if code == 38:
                        self._fg_color = color_code
                    else:
                        self._bg_color = color_code
                    i += 3
                    continue
                if i + 4 < len(parts) and parts[i + 1] == "2":
                    color_code = f"{parts[i]};{parts[i + 1]};{parts[i + 2]};{parts[i + 3]};{parts[i + 4]}"
                    if code == 38:
                        self._fg_color = color_code
                    else:
                        self._bg_color = color_code
                    i += 5
                    continue

            # Standard SGR codes
            if code == 0:
                self._reset()
            elif code == 1:
                self._bold = True
            elif code == 2:
                self._dim = True
            elif code == 3:
                self._italic = True
            elif code == 4:
                self._underline = True
            elif code == 5:
                self._blink = True
            elif code == 7:
                self._inverse = True
            elif code == 8:
                self._hidden = True
            elif code == 9:
                self._strikethrough = True
            elif code == 21:
                self._bold = False  # Some terminals
            elif code == 22:
                self._bold = False
                self._dim = False
            elif code == 23:
                self._italic = False
            elif code == 24:
                self._underline = False
            elif code == 25:
                self._blink = False
            elif code == 27:
                self._inverse = False
            elif code == 28:
                self._hidden = False
            elif code == 29:
                self._strikethrough = False
            elif code == 39:
                self._fg_color = None  # Default fg
            elif code == 49:
                self._bg_color = None  # Default bg
            elif (30 <= code <= 37) or (90 <= code <= 97):
                self._fg_color = str(code)
            elif (40 <= code <= 47) or (100 <= code <= 107):
                self._bg_color = str(code)
            i += 1

    def _reset(self) -> None:
        self._bold = False
        self._dim = False
        self._italic = False
        self._underline = False
        self._blink = False
        self._inverse = False
        self._hidden = False
        self._strikethrough = False
        self._fg_color = None
        self._bg_color = None
        # SGR reset does not affect OSC 8 hyperlink state

    def clear(self) -> None:
        """Clear all state for reuse."""
        self._reset()
        self._active_hyperlink = None

    def get_active_codes(self) -> str:
        codes: list[str] = []
        if self._bold:
            codes.append("1")
        if self._dim:
            codes.append("2")
        if self._italic:
            codes.append("3")
        if self._underline:
            codes.append("4")
        if self._blink:
            codes.append("5")
        if self._inverse:
            codes.append("7")
        if self._hidden:
            codes.append("8")
        if self._strikethrough:
            codes.append("9")
        if self._fg_color:
            codes.append(self._fg_color)
        if self._bg_color:
            codes.append(self._bg_color)

        result = f"\x1b[{';'.join(codes)}m" if codes else ""
        if self._active_hyperlink:
            result += _format_osc8_hyperlink(self._active_hyperlink)
        return result

    def has_active_codes(self) -> bool:
        return (
            self._bold
            or self._dim
            or self._italic
            or self._underline
            or self._blink
            or self._inverse
            or self._hidden
            or self._strikethrough
            or self._fg_color is not None
            or self._bg_color is not None
            or self._active_hyperlink is not None
        )

    def get_line_end_reset(self) -> str:
        """Get reset codes for attributes that need to be turned off at line end.

        Underline must be closed to prevent bleeding into padding. Active OSC 8
        hyperlinks must be closed and re-opened on the next line. Returns empty
        string if no attributes need closing.
        """
        result = ""
        if self._underline:
            result += "\x1b[24m"  # Underline off only
        if self._active_hyperlink:
            result += _format_osc8_close(self._active_hyperlink["terminator"])  # Re-opened via get_active_codes()
        return result


def _update_tracker_from_text(text: str, tracker: AnsiCodeTracker) -> None:
    i = 0
    while i < len(text):
        ansi_result = extract_ansi_code(text, i)
        if ansi_result:
            tracker.process(ansi_result["code"])
            i += ansi_result["length"]
        else:
            i += 1


# =============================================================================
# Wrapping
# =============================================================================


def _split_into_tokens_with_ansi(text: str) -> list[str]:
    """Split text into words while keeping ANSI codes attached."""
    tokens: list[str] = []
    current = ""
    pending_ansi = ""  # ANSI codes waiting to be attached to next visible content
    current_kind: str | None = None  # "space" | "word"
    i = 0

    def flush_current() -> None:
        nonlocal current, current_kind
        if not current:
            return
        tokens.append(current)
        current = ""
        current_kind = None

    while i < len(text):
        ansi_result = extract_ansi_code(text, i)
        if ansi_result:
            # Hold ANSI codes separately - they'll be attached to the next visible char
            pending_ansi += ansi_result["code"]
            i += ansi_result["length"]
            continue

        end = i
        while end < len(text) and not extract_ansi_code(text, end):
            end += 1

        for segment in grapheme_lib.graphemes(text[i:end]):
            segment_is_space = segment == " "
            if not segment_is_space and cjk_break_regex.search(segment):
                flush_current()
                token = pending_ansi + segment
                pending_ansi = ""
                tokens.append(token)
                continue

            segment_kind = "space" if segment_is_space else "word"
            if current and current_kind != segment_kind:
                flush_current()

            # Attach any pending ANSI codes to this visible character
            if pending_ansi:
                current += pending_ansi
                pending_ansi = ""

            current_kind = segment_kind
            current += segment

        i = end

    # Handle any remaining pending ANSI codes (attach to last token)
    if pending_ansi:
        if current:
            current += pending_ansi
        elif tokens:
            tokens[-1] += pending_ansi
        else:
            current = pending_ansi

    if current:
        tokens.append(current)

    return tokens


_NEWLINE_SPLIT_RE = re.compile(r"\r\n|\r|\n")


def wrap_text_with_ansi(text: str, width: int) -> list[str]:
    """Wrap text with ANSI codes preserved.

    ONLY does word wrapping - NO padding, NO background colors.
    Returns lines where each line is <= width visible chars.
    Active ANSI codes are preserved across line breaks.
    """
    if not text:
        return [""]

    # Handle newlines by processing each line separately
    # Track ANSI state across lines so styles carry over after literal newlines
    input_lines = _NEWLINE_SPLIT_RE.split(text)
    result: list[str] = []
    tracker = AnsiCodeTracker()

    for input_line in input_lines:
        # Prepend active ANSI codes from previous lines (except for first line)
        prefix = tracker.get_active_codes() if result else ""
        wrapped_lines = _wrap_single_line(prefix + input_line, width)
        result.extend(wrapped_lines)
        # Update tracker with codes from this line for next iteration
        _update_tracker_from_text(input_line, tracker)

    return result if result else [""]


def _wrap_single_line(line: str, width: int) -> list[str]:
    if not line:
        return [""]

    visible_length = visible_width(line)
    if visible_length <= width:
        return [line]

    wrapped: list[str] = []
    tracker = AnsiCodeTracker()
    tokens = _split_into_tokens_with_ansi(line)

    current_line = ""
    current_visible_length = 0

    for token in tokens:
        token_visible_length = visible_width(token)
        is_whitespace = token.strip() == ""

        # Token itself is too long - break it character by character
        if token_visible_length > width and not is_whitespace:
            if current_line:
                # Add specific reset for underline only (preserves background)
                line_end_reset = tracker.get_line_end_reset()
                if line_end_reset:
                    current_line += line_end_reset
                wrapped.append(current_line)
                current_line = ""
                current_visible_length = 0

            # Break long token - _break_long_word handles its own resets
            broken = _break_long_word(token, width, tracker)
            wrapped.extend(broken[:-1])
            current_line = broken[-1]
            current_visible_length = visible_width(current_line)
            continue

        # Check if adding this token would exceed width
        total_needed = current_visible_length + token_visible_length

        if total_needed > width and current_visible_length > 0:
            # Trim trailing whitespace, then add underline reset (not full reset, to preserve background)
            line_to_wrap = current_line.rstrip()
            line_end_reset = tracker.get_line_end_reset()
            if line_end_reset:
                line_to_wrap += line_end_reset
            wrapped.append(line_to_wrap)
            if is_whitespace:
                # Don't start new line with whitespace
                current_line = tracker.get_active_codes()
                current_visible_length = 0
            else:
                current_line = tracker.get_active_codes() + token
                current_visible_length = token_visible_length
        else:
            # Add to current line
            current_line += token
            current_visible_length += token_visible_length

        _update_tracker_from_text(token, tracker)

    if current_line:
        # No reset at end of final line - let caller handle it
        wrapped.append(current_line)

    # Trailing whitespace can cause lines to exceed the requested width
    return [line.rstrip() for line in wrapped] if wrapped else [""]


PUNCTUATION_REGEX = re.compile(r"[(){}\[\]<>.,;:'\"!?+\-=*/\\|&%^$#@~`]")

_WHITESPACE_RE = re.compile(r"\s")


def is_whitespace_char(char: str) -> bool:
    """Check if a character is whitespace."""
    return _WHITESPACE_RE.search(char) is not None


def is_punctuation_char(char: str) -> bool:
    """Check if a character is punctuation."""
    return PUNCTUATION_REGEX.search(char) is not None


def _break_long_word(word: str, width: int, tracker: AnsiCodeTracker) -> list[str]:
    lines: list[str] = []
    current_line = tracker.get_active_codes()
    current_width = 0

    # First, separate ANSI codes from visible content
    # We need to handle ANSI codes specially since they're not graphemes
    i = 0
    segments: list[tuple[str, str]] = []  # (type, value) with type "ansi" | "grapheme"

    while i < len(word):
        ansi_result = extract_ansi_code(word, i)
        if ansi_result:
            segments.append(("ansi", ansi_result["code"]))
            i += ansi_result["length"]
        else:
            # Find the next ANSI code or end of string
            end = i
            while end < len(word) and not extract_ansi_code(word, end):
                end += 1
            # Segment this non-ANSI portion into graphemes
            for segment in grapheme_lib.graphemes(word[i:end]):
                segments.append(("grapheme", segment))
            i = end

    # Now process segments
    for segment_type, value in segments:
        if segment_type == "ansi":
            current_line += value
            tracker.process(value)
            continue

        # Skip empty graphemes to avoid issues with width calculation
        if not value:
            continue

        grapheme_width = visible_width(value)

        if current_width + grapheme_width > width:
            # Add specific reset for underline only (preserves background)
            line_end_reset = tracker.get_line_end_reset()
            if line_end_reset:
                current_line += line_end_reset
            lines.append(current_line)
            current_line = tracker.get_active_codes()
            current_width = 0

        current_line += value
        current_width += grapheme_width

    if current_line:
        # No reset at end of final segment - caller handles continuation
        lines.append(current_line)

    return lines if lines else [""]


def apply_background_to_line(line: str, width: int, bg_fn) -> str:
    """Apply background color to a line, padding to full width."""
    visible_len = visible_width(line)
    padding_needed = max(0, width - visible_len)
    padding = " " * padding_needed

    # Apply background to content + padding
    return bg_fn(line + padding)


# =============================================================================
# Truncation
# =============================================================================


def _truncate_fragment_to_width(text: str, max_width: int) -> tuple[str, int]:
    if max_width <= 0 or len(text) == 0:
        return "", 0

    if _is_printable_ascii(text):
        clipped = text[:max_width]
        return clipped, len(clipped)

    has_ansi = "\x1b" in text
    has_tabs = "\t" in text
    if not has_ansi and not has_tabs:
        result = ""
        width = 0
        for segment in grapheme_lib.graphemes(text):
            w = _grapheme_width(segment)
            if width + w > max_width:
                break
            result += segment
            width += w
        return result, width

    result = ""
    width = 0
    i = 0
    pending_ansi = ""

    while i < len(text):
        ansi = extract_ansi_code(text, i)
        if ansi:
            pending_ansi += ansi["code"]
            i += ansi["length"]
            continue

        if text[i] == "\t":
            if width + 3 > max_width:
                break
            if pending_ansi:
                result += pending_ansi
                pending_ansi = ""
            result += "\t"
            width += 3
            i += 1
            continue

        end = i
        while end < len(text) and text[end] != "\t":
            if extract_ansi_code(text, end):
                break
            end += 1

        for segment in grapheme_lib.graphemes(text[i:end]):
            w = _grapheme_width(segment)
            if width + w > max_width:
                return result, width
            if pending_ansi:
                result += pending_ansi
                pending_ansi = ""
            result += segment
            width += w
        i = end

    return result, width


def _finalize_truncated_result(
    prefix: str,
    prefix_width: int,
    ellipsis: str,
    ellipsis_width: int,
    max_width: int,
    pad: bool,
) -> str:
    reset = "\x1b[0m"
    visible = prefix_width + ellipsis_width

    if len(ellipsis) > 0:
        result = f"{prefix}{reset}{ellipsis}{reset}"
    else:
        result = f"{prefix}{reset}"

    return result + " " * max(0, max_width - visible) if pad else result


def truncate_to_width(text: str, max_width: int, ellipsis: str = "...", pad: bool = False) -> str:
    """Truncate text to fit within a maximum visible width, adding ellipsis if needed.

    Optionally pad with spaces to reach exactly max_width.
    Properly handles ANSI escape codes (they don't count toward width).
    """
    if max_width <= 0:
        return ""

    if len(text) == 0:
        return " " * max_width if pad else ""

    ellipsis_width = visible_width(ellipsis)
    if ellipsis_width >= max_width:
        text_width = visible_width(text)
        if text_width <= max_width:
            return text + " " * (max_width - text_width) if pad else text

        clipped_ellipsis, clipped_width = _truncate_fragment_to_width(ellipsis, max_width)
        if clipped_width == 0:
            return " " * max_width if pad else ""
        return _finalize_truncated_result("", 0, clipped_ellipsis, clipped_width, max_width, pad)

    if _is_printable_ascii(text):
        if len(text) <= max_width:
            return text + " " * (max_width - len(text)) if pad else text
        target_width = max_width - ellipsis_width
        return _finalize_truncated_result(text[:target_width], target_width, ellipsis, ellipsis_width, max_width, pad)

    target_width = max_width - ellipsis_width
    result = ""
    pending_ansi = ""
    visible_so_far = 0
    kept_width = 0
    keep_contiguous_prefix = True
    overflowed = False
    exhausted_input = False
    has_ansi = "\x1b" in text
    has_tabs = "\t" in text

    if not has_ansi and not has_tabs:
        for segment in grapheme_lib.graphemes(text):
            width = _grapheme_width(segment)
            if keep_contiguous_prefix and kept_width + width <= target_width:
                result += segment
                kept_width += width
            else:
                keep_contiguous_prefix = False
            visible_so_far += width
            if visible_so_far > max_width:
                overflowed = True
                break
        exhausted_input = not overflowed
    else:
        i = 0
        while i < len(text):
            ansi = extract_ansi_code(text, i)
            if ansi:
                pending_ansi += ansi["code"]
                i += ansi["length"]
                continue

            if text[i] == "\t":
                if keep_contiguous_prefix and kept_width + 3 <= target_width:
                    if pending_ansi:
                        result += pending_ansi
                        pending_ansi = ""
                    result += "\t"
                    kept_width += 3
                else:
                    keep_contiguous_prefix = False
                    pending_ansi = ""
                visible_so_far += 3
                if visible_so_far > max_width:
                    overflowed = True
                    break
                i += 1
                continue

            end = i
            while end < len(text) and text[end] != "\t":
                if extract_ansi_code(text, end):
                    break
                end += 1

            for segment in grapheme_lib.graphemes(text[i:end]):
                width = _grapheme_width(segment)
                if keep_contiguous_prefix and kept_width + width <= target_width:
                    if pending_ansi:
                        result += pending_ansi
                        pending_ansi = ""
                    result += segment
                    kept_width += width
                else:
                    keep_contiguous_prefix = False
                    pending_ansi = ""

                visible_so_far += width
                if visible_so_far > max_width:
                    overflowed = True
                    break
            if overflowed:
                break
            i = end
        exhausted_input = i >= len(text)

    if not overflowed and exhausted_input:
        return text + " " * max(0, max_width - visible_so_far) if pad else text

    return _finalize_truncated_result(result, kept_width, ellipsis, ellipsis_width, max_width, pad)


# =============================================================================
# Column slicing and overlay segment extraction
# =============================================================================


def slice_by_column(line: str, start_col: int, length: int, strict: bool = False) -> str:
    """Extract a range of visible columns from a line. Handles ANSI codes and wide chars.

    With ``strict``, wide chars at the boundary that would extend past the
    range are excluded.
    """
    return slice_with_width(line, start_col, length, strict)[0]


def slice_with_width(line: str, start_col: int, length: int, strict: bool = False) -> tuple[str, int]:
    """Like slice_by_column but also returns the actual visible width of the result."""
    if length <= 0:
        return "", 0
    end_col = start_col + length
    result = ""
    result_width = 0
    current_col = 0
    i = 0
    pending_ansi = ""

    while i < len(line):
        ansi = extract_ansi_code(line, i)
        if ansi:
            if start_col <= current_col < end_col:
                result += ansi["code"]
            elif current_col < start_col:
                pending_ansi += ansi["code"]
            i += ansi["length"]
            continue

        text_end = i
        while text_end < len(line) and not extract_ansi_code(line, text_end):
            text_end += 1

        for segment in grapheme_lib.graphemes(line[i:text_end]):
            w = _grapheme_width(segment)
            in_range = start_col <= current_col < end_col
            fits = not strict or current_col + w <= end_col
            if in_range and fits:
                if pending_ansi:
                    result += pending_ansi
                    pending_ansi = ""
                result += segment
                result_width += w
            current_col += w
            if current_col >= end_col:
                break
        i = text_end
        if current_col >= end_col:
            break
    return result, result_width


# Per-thread pooled tracker for extract_segments (pi shares one instance;
# tonio may call this from parallel layout workers).
_pooled_tracker_local = threading.local()


def _pooled_style_tracker() -> AnsiCodeTracker:
    tracker = getattr(_pooled_tracker_local, "tracker", None)
    if tracker is None:
        tracker = AnsiCodeTracker()
        _pooled_tracker_local.tracker = tracker
    return tracker


def extract_segments(
    line: str,
    before_end: int,
    after_start: int,
    after_len: int,
    strict_after: bool = False,
) -> dict:
    """Extract "before" and "after" segments from a line in a single pass.

    Used for overlay compositing where we need content before and after the
    overlay region. Preserves styling from before the overlay that should
    affect content after it. Returns a record:
    ``{"before", "beforeWidth", "after", "afterWidth"}``.
    """
    before = ""
    before_width = 0
    after = ""
    after_width = 0
    current_col = 0
    i = 0
    pending_ansi_before = ""
    after_started = False
    after_end = after_start + after_len

    # Track styling state so "after" inherits styling from before the overlay
    tracker = _pooled_style_tracker()
    tracker.clear()

    while i < len(line):
        ansi = extract_ansi_code(line, i)
        if ansi:
            # Track all SGR codes to know styling state at after_start
            tracker.process(ansi["code"])
            # Include ANSI codes in their respective segments
            if current_col < before_end:
                pending_ansi_before += ansi["code"]
            elif after_start <= current_col < after_end and after_started:
                # Only include after we've started "after" (styling already prepended)
                after += ansi["code"]
            i += ansi["length"]
            continue

        text_end = i
        while text_end < len(line) and not extract_ansi_code(line, text_end):
            text_end += 1

        for segment in grapheme_lib.graphemes(line[i:text_end]):
            w = _grapheme_width(segment)

            if current_col < before_end and current_col + w <= before_end:
                if pending_ansi_before:
                    before += pending_ansi_before
                    pending_ansi_before = ""
                before += segment
                before_width += w
            elif after_start <= current_col < after_end:
                fits = not strict_after or current_col + w <= after_end
                if fits:
                    # On first "after" grapheme, prepend inherited styling from before overlay
                    if not after_started:
                        after += tracker.get_active_codes()
                        after_started = True
                    after += segment
                    after_width += w

            current_col += w
            # Early exit: done with "before" only, or done with both segments
            if (current_col >= before_end) if after_len <= 0 else (current_col >= after_end):
                break
        i = text_end
        if (current_col >= before_end) if after_len <= 0 else (current_col >= after_end):
            break

    return {"before": before, "beforeWidth": before_width, "after": after, "afterWidth": after_width}
