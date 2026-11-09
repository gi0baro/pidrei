"""The helpers of pi-schema's checks.ts that the explainer and the coercer call
(packages/schema/src/checks.ts).

checks.ts is mostly closure compilers, the engine of pi's checks; here the
engine is `jsonschema` (engine.py), so only these helpers port: the type checks
by name, `enum`, `const`, `contains` and `uniqueItems`, `isMultiple` and JSON
equality. `codePoints` has no port: a Python `str` is code points, and the
strings of tool arguments come from JSON, whose surrogate pairs the parser
joins.

JavaScript's number and JSON formatting is part of the text the model sees
(`expected >= 1`, `expected one of [1,"a"]`), and JavaScript has one number
type where Python has `int` and `float`: `number_string` and `json_stringify`
are `String(number)` and `JSON.stringify(value)`, and `same_value` is `===`.
"""

import json
import math
import re
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from typing import Any


type Check = Callable[[Any], bool]


def is_object(v: Any) -> bool:
    """Whether `v` is a JSON object."""
    return isinstance(v, dict)


def is_number(v: Any) -> bool:
    """JavaScript's `typeof v === "number"`: an `int` or `float`, never a `bool`."""
    return isinstance(v, int | float) and not isinstance(v, bool)


def is_integer(v: Any) -> bool:
    """`Number.isInteger(v)`: a finite whole number (`1.0` included)."""
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    return isinstance(v, float) and v.is_integer()


#: The check of a `type` name. pi also has `bytes` (a Uint8Array), which no
#: JSON value is; an unknown name, as there, checks nothing true.
TYPE_CHECKS: dict[str, Check] = {
    "string": lambda v: isinstance(v, str),
    "number": is_number,
    "integer": is_integer,
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "array": lambda v: isinstance(v, list),
    "object": is_object,
}


def same_value(a: Any, b: Any) -> bool:
    """JavaScript's `a === b` for JSON values: numbers by value (`1 === 1.0`),
    never a boolean equal to a number, objects and arrays by identity."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if is_number(a) and is_number(b):
        return a == b
    if isinstance(a, dict | list) or isinstance(b, dict | list):
        return a is b
    return type(a) is type(b) and a == b


def deep_equal(a: Any, b: Any) -> bool:
    """JSON equality: same primitives (`1` is not `true`), arrays with equal
    items in order, objects with the same keys and equal values."""
    if a is b:
        return True
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(map(deep_equal, a, b, strict=True))
    if isinstance(a, dict):
        return isinstance(b, dict) and len(a) == len(b) and all(k in b and deep_equal(x, b[k]) for k, x in a.items())
    if isinstance(b, dict | list):
        return False
    return same_value(a, b)


def enum_check(values: list) -> Check:
    """`enum`."""
    return lambda v: any(deep_equal(v, member) for member in values)


def const_check(value: Any) -> Check:
    """`const`."""
    return lambda v: deep_equal(v, value)


def contains_check(check: Check, minimum: float, maximum: float) -> Check:
    """`contains` with its bounds (`v` is an array)."""

    def contains(v: list) -> bool:
        n = 0
        for item in v:
            if check(item):
                n += 1
                if n > maximum:
                    return False
        return n >= minimum

    return contains


def unique_check(v: list) -> bool:
    """`uniqueItems` (`v` is an array)."""
    return not any(deep_equal(v[i], v[j]) for i in range(1, len(v)) for j in range(i))


def is_multiple(v: float, d: float) -> bool:
    """Whether `v` is a multiple of `d`, exactly for decimals: 0.0075 / 0.0001
    is 74.99999999999999 in floating point."""
    try:
        q = v / d
    except ZeroDivisionError:
        return False
    except OverflowError:
        q = math.inf
    if math.isfinite(q) and q.is_integer():
        return True
    if not math.isfinite(q):
        try:
            return math.fmod(v, d) == 0
        except OverflowError, ValueError:
            return False
    # The shortest decimals of both (JavaScript's toString, Python's repr), exactly.
    return (Fraction(repr(abs(v))) / Fraction(repr(abs(d)))).denominator == 1


def number_string(x: float) -> str:
    """JavaScript's `String(x)` for a number: `1`, not `1.0`; `1e+21`, `1e-7`."""
    if isinstance(x, int):
        return str(x) if abs(x) < 10**21 else number_string(float(x))
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    _, digit_tuple, exponent = Decimal(repr(abs(x))).normalize().as_tuple()
    digits = "".join(map(str, digit_tuple))
    k = len(digits)
    n = k + int(exponent)
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return f"{sign}{digits[:n]}.{digits[n:]}"
    if -6 < n <= 0:
        return f"{sign}0.{'0' * -n}{digits}"
    e = n - 1
    mantissa = digits if k == 1 else f"{digits[0]}.{digits[1:]}"
    return f"{sign}{mantissa}e{'+' if e >= 0 else '-'}{abs(e)}"


_LONE_SURROGATE = re.compile(r"[\ud800-\udfff]")


def _quote(text: str) -> str:
    # JSON.stringify writes a lone surrogate as an escape; Python keeps it raw.
    return _LONE_SURROGATE.sub(lambda m: f"\\u{ord(m[0]):04x}", json.dumps(text, ensure_ascii=False))


def json_stringify(value: Any) -> str:
    """JavaScript's `JSON.stringify(value)` for a JSON value, without spaces."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int | float):
        return number_string(value) if math.isfinite(value) else "null"
    if isinstance(value, str):
        return _quote(value)
    if isinstance(value, list):
        return f"[{','.join(map(json_stringify, value))}]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{_quote(str(k))}:{json_stringify(v)}" for k, v in value.items()) + "}"
    return "null"


def js_string(value: Any) -> str:
    """JavaScript's `String(value)` for a JSON value (what `Array.join` writes)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return number_string(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ",".join("" if item is None else js_string(item) for item in value)
    return "[object Object]"
