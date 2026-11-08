"""Port of pi's JSON repair and finished-arguments parse (packages/ai/src/utils/json-parse.ts).

pi's `parseStreamingJson` (a full re-parse of the accumulated text through the
`partial-json` package) is not ported: since 1.5.0 every adapter reads streamed
tool-call arguments through `pidrei_ai.utils.streaming_json`, so it has no
caller here.
"""

import json
from typing import Any


_VALID_JSON_ESCAPES = frozenset('"\\/bfnrt')
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
    # Python pass is skipped.
    return isinstance(error, json.JSONDecodeError) and error.msg.startswith("Invalid")


def _reject_constant(constant: str) -> Any:
    # `JSON.parse` has no NaN/Infinity/-Infinity; Python's decoder accepts them
    # unless told otherwise.
    raise ValueError(f"Unexpected token {constant}")


def _loads(json_text: str) -> Any:
    return json.loads(json_text, parse_constant=_reject_constant)


def parse_json_with_repair(json_text: str) -> Any:
    try:
        return _loads(json_text)
    except ValueError as error:
        if not _is_repairable(error):
            raise
        repaired = repair_json(json_text)
        if repaired != json_text:
            return _loads(repaired)
        raise


def parse_tool_call_arguments(json_text: str | None) -> tuple[dict[str, Any], str | None]:
    """Parse the model's complete arguments text of a finished tool call.

    Unlike the streamed partial arguments, this never keeps a partially parsed prefix:
    once the call is finished, text that is not a valid JSON object means the
    model sent malformed arguments. Returns `(arguments, arguments_error)`; on
    failure the arguments are `{}` and the error describes the problem,
    including the text the model sent, so tool validation can report it back
    to the model.
    """
    if not json_text or not json_text.strip():
        return {}, None
    try:
        parsed = parse_json_with_repair(json_text)
    except (ValueError, RecursionError) as error:
        return {}, f"{error}\n\nReceived arguments:\n{json_text}"
    if not isinstance(parsed, dict):
        return {}, f"Tool arguments must be a JSON object\n\nReceived arguments:\n{json_text}"
    return parsed, None
