"""Port of pi's JSON repair + streaming parse (packages/ai/src/utils/json-parse.ts).

`partial_json_parser` (PyPI) is the official Python port of the npm
`partial-json` package pi uses, so partial-parse semantics match by construction.
"""

import json
from typing import Any

from partial_json_parser import loads as _partial_parse


_VALID_JSON_ESCAPES = frozenset('"\\/bfnrtu')
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_CONTROL_ESCAPES = {"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _escape_control_character(char: str) -> str:
    return _CONTROL_ESCAPES.get(char, f"\\u{ord(char):04x}")


def repair_json(json_text: str) -> str:
    """Repair malformed JSON string literals.

    - escapes raw control characters inside strings
    - doubles backslashes before invalid escape characters
    """
    repaired: list[str] = []
    in_string = False
    index = 0
    length = len(json_text)

    while index < length:
        char = json_text[index]

        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            index += 1
            continue

        if char == '"':
            repaired.append(char)
            in_string = False
            index += 1
            continue

        if char == "\\":
            if index + 1 >= length:
                repaired.append("\\\\")
                index += 1
                continue
            next_char = json_text[index + 1]

            if next_char == "u":
                unicode_digits = json_text[index + 2 : index + 6]
                if len(unicode_digits) == 4 and all(digit in _HEX_DIGITS for digit in unicode_digits):
                    repaired.append(f"\\u{unicode_digits}")
                    index += 6
                    continue

            if next_char in _VALID_JSON_ESCAPES:
                repaired.append(f"\\{next_char}")
                index += 2
                continue

            repaired.append("\\\\")
            index += 1
            continue

        repaired.append(_escape_control_character(char) if ord(char) <= 0x1F else char)
        index += 1

    return "".join(repaired)


def _is_repairable(error: ValueError) -> bool:
    # The decoder reports the first problem left to right, and `repair_json`
    # only touches string contents ("Invalid control character", "Invalid
    # \\escape", "Invalid \\uXXXX escape"). Any other first error — a truncated
    # prefix's "Unterminated string"/"Expecting …", "Extra data" — sits before
    # every repairable spot and survives the repair unchanged, so the O(n)
    # Python pass is skipped: this runs once per streamed tool-argument delta.
    return isinstance(error, json.JSONDecodeError) and error.msg.startswith("Invalid")


def parse_json_with_repair(json_text: str) -> Any:
    try:
        return json.loads(json_text)
    except ValueError as error:
        if not _is_repairable(error):
            raise
        repaired = repair_json(json_text)
        if repaired != json_text:
            return json.loads(repaired)
        raise


def parse_streaming_json(partial_json: str | None) -> Any:
    """Parse potentially incomplete JSON during streaming.

    Always returns a value, even if the JSON is incomplete; falls back to an
    empty dict when nothing can be parsed.
    """
    if not partial_json or partial_json.strip() == "":
        return {}

    try:
        return parse_json_with_repair(partial_json)
    except ValueError:
        try:
            result = _partial_parse(partial_json)
            return result if result is not None else {}
        except Exception:
            try:
                result = _partial_parse(repair_json(partial_json))
                return result if result is not None else {}
            except Exception:
                return {}
