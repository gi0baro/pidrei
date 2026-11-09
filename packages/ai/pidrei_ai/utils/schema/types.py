"""Port of pi-schema's public types (packages/schema/src/types.ts): the errors a
failed value produces and the result of `parse()`.

pi's `JsonSchema`, `CompileOptions`, `FormatCheck` and `Check` are TypeScript
shapes: a schema here is a `dict` or a `bool`, the options are `compile()`'s
keyword arguments, formats are a `jsonschema.FormatChecker` and a check is
`Validator.check`.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ValidationError:
    """One failure of a value: where it is (a JSON Pointer, "" for the value
    itself), the failing keyword and a message."""

    path: str
    keyword: str
    message: str


@dataclass(frozen=True, slots=True)
class ParseResult:
    """The result of `parse()`: the coerced value when it is valid (`ok`),
    else the errors of the coerced value."""

    ok: bool
    value: Any = None
    errors: tuple[ValidationError, ...] = ()
