"""Mirror of pi codemode src/source.ts.

Codemode source format: Python, optionally preceded by one options line.

```python
# @options: {"max_output_tokens": 2000, "timeout_ms": 30000}
text = await tools.read(path="pyproject.toml")
text.splitlines()[0]
```
"""

import json
from dataclasses import dataclass
from typing import Any


CODEMODE_OPTIONS_PREFIX = "# @options:"

_SUPPORTED_FIELDS = ("max_output_tokens", "timeout_ms")
_SUPPORTED_FIELDS_TEXT = "`max_output_tokens` and `timeout_ms`"
# pi bounds `timeout_ms` by the largest delay `setTimeout` supports; the bound
# is kept so the options a script may pass are the same.
_MAX_TIMEOUT_MS = 2_147_483_647
_MAX_SAFE_INTEGER = 2**53 - 1

# Lark grammar for providers with grammar-constrained tool input. It only fixes
# the shape of the options line; the options JSON and the code are checked by
# `parse_codemode_source`.
CODEMODE_SOURCE_GRAMMAR = r"""
start: options_source | plain_source
options_source: OPTIONS_LINE NEWLINE SOURCE
plain_source: SOURCE

OPTIONS_LINE: /[ \t]*# @options:[^\r\n]*/
NEWLINE: /\r?\n/
SOURCE: /[\s\S]+/
"""


@dataclass(frozen=True, slots=True)
class CodemodeSourceOptions:
    # Token budget for the script's output.
    max_output_tokens: int | None = None
    # Hard deadline for the whole script in milliseconds, including tool calls.
    timeout_ms: int | None = None


@dataclass(frozen=True, slots=True)
class ParsedCodemodeSource:
    # The script with the options line replaced by an empty line, so line
    # numbers are unchanged.
    code: str
    options: CodemodeSourceOptions


class CodemodeSourceError(Exception):
    pass


def _safe_integer(value: Any) -> int | None:
    """JavaScript's `Number.isSafeInteger(value) && value >= 0`. JSON has one
    number type, so an integral float counts, as it does in pi."""
    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        if not value.is_integer():
            return None
        value = int(value)
    if isinstance(value, int) and 0 <= value <= _MAX_SAFE_INTEGER:
        return value
    return None


def _parse_options(directive: str) -> CodemodeSourceOptions:
    if directive == "":
        raise CodemodeSourceError(f"@options must be a JSON object with supported fields {_SUPPORTED_FIELDS_TEXT}")
    try:
        value = json.loads(directive)
    except ValueError as error:
        raise CodemodeSourceError(
            f"@options must be valid JSON with supported fields {_SUPPORTED_FIELDS_TEXT}: {error}"
        ) from None
    if not isinstance(value, dict):
        raise CodemodeSourceError(f"@options must be a JSON object with supported fields {_SUPPORTED_FIELDS_TEXT}")
    for key in value:
        if key not in _SUPPORTED_FIELDS:
            raise CodemodeSourceError(f"@options only supports {_SUPPORTED_FIELDS_TEXT}; got `{key}`")
    max_output_tokens = None
    timeout_ms = None
    if "max_output_tokens" in value:
        max_output_tokens = _safe_integer(value["max_output_tokens"])
        if max_output_tokens is None:
            raise CodemodeSourceError("@options field `max_output_tokens` must be a non-negative safe integer")
    if "timeout_ms" in value:
        timeout_ms = _safe_integer(value["timeout_ms"])
        if timeout_ms is None or timeout_ms == 0 or timeout_ms > _MAX_TIMEOUT_MS:
            raise CodemodeSourceError(f"@options field `timeout_ms` must be a positive integer up to {_MAX_TIMEOUT_MS}")
    return CodemodeSourceOptions(max_output_tokens=max_output_tokens, timeout_ms=timeout_ms)


def parse_codemode_source(source: str) -> ParsedCodemodeSource:
    """Split an optional first-line `# @options: {...}` from the script. Raises
    `CodemodeSourceError` for empty input and invalid options."""
    if source.strip() == "":
        raise CodemodeSourceError(
            "Expected Python source text (non-empty). Provide Python only, optionally with a first line "
            '`# @options: {"max_output_tokens": 1000}`.'
        )
    newline = source.find("\n")
    first_line = (source if newline == -1 else source[:newline]).removesuffix("\r")
    trimmed = first_line.lstrip()
    if not trimmed.startswith(CODEMODE_OPTIONS_PREFIX):
        return ParsedCodemodeSource(code=source, options=CodemodeSourceOptions())
    code = "" if newline == -1 else source[newline:]
    if code.strip() == "":
        raise CodemodeSourceError("The @options line must be followed by Python source on subsequent lines")
    return ParsedCodemodeSource(code=code, options=_parse_options(trimmed[len(CODEMODE_OPTIONS_PREFIX) :].strip()))
