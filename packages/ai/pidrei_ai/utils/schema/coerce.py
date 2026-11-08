"""Port of pi-schema's coercion (packages/schema/src/coerce.ts): turns a loosely
typed value (a model's tool call arguments) into the value a schema most
likely meant, before it is checked.

Three passes, in this order:

1. Optional nulls: a `null` in an optional property whose schema rejects null
   is dropped, as if omitted.
2. Conversion rules ported from TypeBox 1.3.27's Value.Convert (MIT), keyed on
   JSON Schema structure (`$ref`, `const`, `enum`, `anyOf`/`oneOf`, `allOf`,
   `type`, `items`/`prefixItems`, `properties`/`patternProperties`): `null`
   becomes "null" for a string, `"3.5"` becomes 3 for an integer, a single
   value becomes a one-item array, and a union takes the first branch whose
   conversion passes.
3. Primitive coercion by `type` (also type arrays), as in Ajv's
   `coerceTypes`: what pass 2 left, for example a type array
   `["integer", "null"]`, or `""` for `null`.

A value that cannot be coerced is left as it is; the check then reports it.

Every pass is copy-on-write: an object or array is copied only when one of
its members changes, so an unchanged value comes back as the same object and
the input is never modified. All passes walk the same keywords for members:
`properties`, then `patternProperties`, then `additionalProperties` for keys
neither names; tuple items by `prefixItems` (or a draft-7 `items` array), the
rest by `items`. Draft 7's `additionalItems` is not coerced.

Where Python differs from JavaScript:

- pi keeps the re-entry stack, the cut count and the memo of pass 2 on the
  coercer for the length of one call; here each call converts with its own
  `_Conversion`, so one coercer serves concurrent tool calls.
- JavaScript has one number type. A number this coercion produces is an
  `int` when it is whole and a safe integer (`"42"` and `42.0` for an
  integer give `42`), else a `float`; an integer schema truncates a float to
  an `int`.
- `String(number)` and `Number(string)` follow JavaScript (`"0x10"` is 16,
  `"1e3"` is 1000, `1e21` is `"1e+21"`). TypeBox's `undefined` and `bigint`
  conversions have no JSON value and no branch here.
"""

import math
import re
from collections.abc import Callable
from typing import Any

from pidrei_ai.utils.schema.checks import is_integer, is_number, number_string, same_value
from pidrei_ai.utils.schema.engine import Engine, Scope


_ABSENT = object()
_NO_RESULT = object()

_MAX_SAFE_INTEGER = 2**53 - 1


class Coercer:
    """Coerces values for one compiled schema. Passes 1 and 3 keep no state;
    pass 2 runs in a `_Conversion` per call."""

    __slots__ = ("_engine",)

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def coerce(self, value: Any) -> Any:
        """`value` coerced: a copy along the changed paths only, or `value` itself when nothing changes."""
        schema, scope = self._engine.schema, self._engine.root_scope
        without_nulls = self._drop_optional_nulls(value, schema, scope)
        return self._coerce_types(self.convert_only(without_nulls), schema, scope)

    def convert_only(self, value: Any) -> Any:
        """Pass 2 alone, as TypeBox's `Value.Convert`."""
        return _Conversion(self._engine).convert(self._engine.schema, self._engine.root_scope, value)

    def _passes(self, node: Any, b: Scope, value: Any) -> bool | None:
        return _passes(self._engine, node, b, value)

    def _enter(self, node: Any, b: Scope) -> Scope:
        return self._engine.enter(node, b)

    # -----------------------------------------------------------------------------------------------------------------
    # Pass 1: optional nulls
    # -----------------------------------------------------------------------------------------------------------------

    def _drop_optional_nulls(self, value: Any, schema: Any, b: Scope) -> Any:
        if not isinstance(schema, dict):
            return value
        if isinstance(value, list):

            def drop_item(item: Any, index: int) -> Any:
                item_schema = _item_schema_at(schema, index)
                if item_schema is _ABSENT:
                    return item
                return self._drop_optional_nulls(item, item_schema, self._enter(item_schema, b))

            return _map_items(value, drop_item)
        if not isinstance(value, dict):
            return value
        properties, patterns, additional = _members(schema)
        if not patterns and additional is _ABSENT and not isinstance(schema.get("properties"), dict):
            return value
        required = (
            {key for key in schema["required"] if isinstance(key, str)} if _is_list(schema, "required") else set()
        )
        copy = _CopyOnWrite(value)
        for key, prop in properties.items():
            if key not in value:
                continue
            is_ref = isinstance(prop, dict) and isinstance(prop.get("$ref"), str)
            if (
                value[key] is None
                and key not in required
                and not is_ref
                and self._passes(prop, self._enter(prop, b), None) is False
            ):
                copy.delete(key)
                continue
            copy.set(key, self._drop_optional_nulls(value[key], prop, self._enter(prop, b)))
        # Values of pattern and additional properties are walked for their own optional nulls; the keys themselves
        # are not optional properties, so they stay.
        if patterns or additional is not _ABSENT:
            for key in list(value):
                matched = key in properties
                for pattern, item in patterns:
                    if re.search(pattern, key) is None:
                        continue
                    matched = True
                    if copy.has(key):
                        copy.set(key, self._drop_optional_nulls(copy.get(key), item, self._enter(item, b)))
                if not matched and additional is not _ABSENT:
                    copy.set(key, self._drop_optional_nulls(value[key], additional, self._enter(additional, b)))
        return copy.result()

    # -----------------------------------------------------------------------------------------------------------------
    # Pass 3: primitive coercion by type
    # -----------------------------------------------------------------------------------------------------------------

    def _coerce_types(self, value: Any, schema: Any, b: Scope) -> Any:
        if not isinstance(schema, dict):
            return value
        next_value = value
        if _is_list(schema, "allOf"):
            for member in schema["allOf"]:
                next_value = self._coerce_types(next_value, member, self._enter(member, b))
        if _is_list(schema, "anyOf"):
            next_value = self._coerce_union(next_value, schema["anyOf"], b)
        if _is_list(schema, "oneOf"):
            next_value = self._coerce_union(next_value, schema["oneOf"], b)
        types = _types_of(schema)
        matches_union_member = len(types) > 1 and any(_matches_type(next_value, t) for t in types)
        if types and not matches_union_member:
            for t in types:
                candidate = _coerce_primitive(next_value, t)
                if candidate is not next_value:
                    next_value = candidate
                    break
        if "object" in types and isinstance(next_value, dict):
            next_value = self._coerce_properties(next_value, schema, b)
        if "array" in types and isinstance(next_value, list):
            next_value = self._coerce_items(next_value, schema, b)
        return next_value

    def _coerce_union(self, value: Any, branches: list, b: Scope) -> Any:
        for branch in branches:
            if self._passes(branch, self._enter(branch, b), value) is True:
                return value
        for branch in branches:
            scope = self._enter(branch, b)
            coerced = self._coerce_types(value, branch, scope)
            if self._passes(branch, scope, coerced) is True:
                return coerced
        return value

    def _coerce_properties(self, value: dict, schema: dict, b: Scope) -> dict:
        properties, patterns, additional = _members(schema)
        copy = _CopyOnWrite(value)
        for key, prop in properties.items():
            if key in value:
                copy.set(key, self._coerce_types(value[key], prop, self._enter(prop, b)))
        for key in list(value):
            matched = key in properties
            for pattern, item in patterns:
                if re.search(pattern, key) is None:
                    continue
                matched = True
                copy.set(key, self._coerce_types(copy.get(key), item, self._enter(item, b)))
            if not matched and additional is not _ABSENT:
                copy.set(key, self._coerce_types(value[key], additional, self._enter(additional, b)))
        return copy.result()

    def _coerce_items(self, value: list, schema: dict, b: Scope) -> list:
        def coerce_item(item: Any, index: int) -> Any:
            item_schema = _item_schema_at(schema, index)
            if not isinstance(item_schema, dict):
                return item
            return self._coerce_types(item, item_schema, self._enter(item_schema, b))

        return _map_items(value, coerce_item)


class _Conversion:
    """Pass 2 of one call: TypeBox's conversion rules."""

    __slots__ = ("_cuts", "_engine", "_entered", "_memo")

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        # The (schema node, value) pairs being converted through a re-entry point, as a stack. Conversion can enter
        # a node again with the same value only through a `$ref` or by wrapping a single value into an array: a
        # recursive schema that meets a value no branch accepts would convert it without end. A pair already on the
        # stack (primitives by value, objects by identity) leaves the value unchanged. Only those two points push,
        # so schemas without references or wraps pay nothing.
        self._entered: list[tuple[dict, tuple, Any]] = []
        # How often a re-entry was cut short; results computed across one are not memoized.
        self._cuts = 0
        # Results of `$ref` targets by (node, value): a recursive union whose branches all convert the same children
        # (and all fail) would convert them once per branch, exponentially in the depth. Each entry keeps its value,
        # so an object's id cannot be reused while the conversion runs.
        self._memo: dict[int, dict[tuple, tuple[Any, Any]]] = {}

    def _enter(self, node: Any, b: Scope) -> Scope:
        return self._engine.enter(node, b)

    def _passes(self, node: Any, b: Scope, value: Any) -> bool | None:
        return _passes(self._engine, node, b, value)

    def _reenter(self, node: dict, b: Scope, value: Any, *, memoize: bool) -> Any:
        """`node` converted for `value` at a re-entry point: unchanged when that pair is already being converted."""
        key = _identity_key(value)
        for entered, entered_key, _ in self._entered:
            if entered is node and entered_key == key:
                self._cuts += 1
                return value
        values = self._memo.get(id(node)) if memoize else None
        if values is not None and key in values:
            return values[key][1]
        cuts = self._cuts
        self._entered.append((node, key, value))
        try:
            result = self.convert(node, b, value)
        finally:
            self._entered.pop()
        if memoize and self._cuts == cuts:
            self._memo.setdefault(id(node), {})[key] = (value, result)
        return result

    def convert(self, node: Any, b: Scope, value: Any) -> Any:
        """`value` converted by `node`, whose scope is `b`."""
        if not isinstance(node, dict):
            return value
        if isinstance(node.get("$ref"), str):
            target = self._engine.resolve(node["$ref"], b)
            if target is None or not isinstance(target[0], dict):
                return value
            return self._reenter(target[0], target[1], value, memoize=True)
        return self._convert_node(node, b, value)

    def _convert_node(self, node: dict, b: Scope, value: Any) -> Any:
        """One schema node's conversion: its first keyword among `const`, `enum`, the unions, `allOf` and `type`."""
        if "const" in node:
            return _convert_literal(node["const"], value)
        if _is_list(node, "enum"):
            return self._convert_enum(node, node["enum"], b, value)
        # A union or intersection node may also declare members (plain JSON Schema; TypeBox's never do): they convert
        # after the branches.
        if _is_list(node, "anyOf"):
            return self._convert_union(node, node["anyOf"], b, value)
        if _is_list(node, "oneOf"):
            return self._convert_union(node, node["oneOf"], b, value)
        if _is_list(node, "allOf"):
            next_value = value
            for member in node["allOf"]:
                next_value = self.convert(member, self._enter(member, b), next_value)
            return self._convert_members(node, b, next_value)
        if "type" not in node:
            if "properties" in node or "patternProperties" in node:
                return self._convert_object(node, b, value)
            if "items" in node or "prefixItems" in node:
                return self._convert_array(node, b, value)
            return value
        match node["type"]:
            case "string":
                return _or_value(_try_string(value), value)
            case "number":
                return _or_value(_try_number(value), value)
            case "integer":
                result = _try_number(value)
                return value if result is _NO_RESULT else _truncate(result)
            case "boolean":
                return _or_value(_try_boolean(value), value)
            case "null":
                return _or_value(_try_null(value), value)
            case "array":
                return self._convert_array(node, b, value)
            case "object":
                return self._convert_object(node, b, value)
            case _:
                return value

    def _convert_enum(self, node: dict, values: list, b: Scope, value: Any) -> Any:
        """An enum is a union of its values: a member stays, else the first value the input converts to."""
        if any(same_value(member, value) for member in values):
            return value
        for member in values:
            candidate = _convert_literal(member, value)
            if candidate is not value and self._passes(node, b, candidate) is True:
                return candidate
        return value

    def _convert_union(self, node: dict, branches: list, b: Scope, value: Any) -> Any:
        """A value one branch accepts stays; else the first branch's conversion that the whole union accepts.
        Branches are converted in order and only until one is accepted, so a later recursive branch is not entered
        needlessly. A node that also declares members converts them into each candidate before it is checked, so a
        branch is not discarded for a sibling property that is converted after it."""
        if any(self._passes(branch, self._enter(branch, b), value) is True for branch in branches):
            return self._convert_members(node, b, value)
        for branch in branches:
            candidate = self._convert_members(node, b, self.convert(branch, self._enter(branch, b), value))
            if self._passes(node, b, candidate) is True:
                return candidate
        return self._convert_members(node, b, value)

    def _convert_members(self, node: dict, b: Scope, value: Any) -> Any:
        """The object members of a union or intersection node, when it declares any."""
        declares = "properties" in node or "patternProperties" in node or "additionalProperties" in node
        return self._convert_object(node, b, value) if declares else value

    def _convert_array(self, node: dict, b: Scope, value: Any) -> Any:
        items = node.get("items", _ABSENT)
        prefix = items if isinstance(items, list) else node["prefixItems"] if _is_list(node, "prefixItems") else None
        if prefix is not None:
            # A tuple converts its items and never wraps a single value.
            if not isinstance(value, list):
                return value
            rest = _ABSENT if isinstance(items, list) else items

            def convert_item(item: Any, index: int) -> Any:
                schema = prefix[index] if index < len(prefix) else rest
                return item if schema is _ABSENT else self.convert(schema, self._enter(schema, b), item)

            return _map_items(value, convert_item)
        if isinstance(value, list):
            if items is _ABSENT:
                return value
            scope = self._enter(items, b)
            return _map_items(value, lambda item, _index: self.convert(items, scope, item))
        # The wrapped single value is converted by the item schema with the same value: a re-entry point.
        if isinstance(items, dict):
            return [self._reenter(items, self._enter(items, b), value, memoize=False)]
        return [value]

    def _convert_object(self, node: dict, b: Scope, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        properties, patterns, additional = _members(node)
        copy = _CopyOnWrite(value)
        for key, original in value.items():
            next_value = original
            matched = key in properties
            if matched:
                next_value = self.convert(properties[key], self._enter(properties[key], b), next_value)
            for pattern, schema in patterns:
                if re.search(pattern, key) is None:
                    continue
                matched = True
                next_value = self.convert(schema, self._enter(schema, b), next_value)
            if not matched and additional is not _ABSENT:
                next_value = self.convert(additional, self._enter(additional, b), next_value)
            copy.set(key, next_value)
        return copy.result()


def _passes(engine: Engine, node: Any, b: Scope, value: Any) -> bool | None:
    """Whether `value` passes the schema `node`; None when the engine cannot check it."""
    try:
        return engine.check(node, b, value)
    except Exception:
        return None


def _is_list(node: dict, key: str) -> bool:
    return isinstance(node.get(key), list)


def _members(node: dict) -> tuple[dict, list[tuple[str, Any]], Any]:
    """`properties`, `patternProperties` and an object `additionalProperties` (else absent)."""
    properties = node["properties"] if isinstance(node.get("properties"), dict) else {}
    patterns = list(node["patternProperties"].items()) if isinstance(node.get("patternProperties"), dict) else []
    additional = node["additionalProperties"] if isinstance(node.get("additionalProperties"), dict) else _ABSENT
    return properties, patterns, additional


class _CopyOnWrite:
    """An object whose members may change: copied on the first change only."""

    __slots__ = ("_out", "_value")

    def __init__(self, value: dict) -> None:
        self._value = value
        self._out: dict | None = None

    def _current(self) -> dict:
        return self._value if self._out is None else self._out

    def has(self, key: str) -> bool:
        return key in self._current()

    def get(self, key: str) -> Any:
        return self._current()[key]

    def set(self, key: str, next_value: Any) -> None:
        current = self._current()
        if key in current and next_value is current[key]:
            return
        if self._out is None:
            self._out = dict(self._value)
        self._out[key] = next_value

    def delete(self, key: str) -> None:
        if self._out is None:
            self._out = dict(self._value)
        self._out.pop(key, None)

    def result(self) -> dict:
        return self._current()


def _item_schema_at(schema: dict, index: int) -> Any:
    """The schema of an array's item at `index`: a tuple's (`prefixItems`, or a draft-7 `items` array) for its
    prefix, `items` after `prefixItems`, or `items` for every item. Absent when nothing applies."""
    items = schema.get("items", _ABSENT)
    if isinstance(items, list):
        return items[index] if index < len(items) else _ABSENT
    prefix = schema.get("prefixItems")
    if isinstance(prefix, list) and index < len(prefix):
        return prefix[index]
    return items


def _map_items(items: list, convert: Callable[[Any, int], Any]) -> list:
    """`items` mapped by `convert`; the same list when every item maps to itself."""
    out: list | None = None
    for index, item in enumerate(items):
        next_item = convert(item, index)
        if next_item is item:
            continue
        if out is None:
            out = list(items)
        out[index] = next_item
    return items if out is None else out


def _identity_key(value: Any) -> tuple:
    """JavaScript's `Object.is` as a key: primitives by value (`1` and `1.0` alike, `true` apart from `1`),
    objects and arrays by identity."""
    if isinstance(value, bool):
        return ("boolean", value)
    if is_number(value):
        return ("number", value)
    if isinstance(value, dict | list):
        return ("object", id(value))
    return (type(value).__name__, value)


# ---------------------------------------------------------------------------------------------------------------------
# TypeBox's conversions of one value (value/convert/try in TypeBox 1.3.27): a result, or _NO_RESULT when there is none
# ---------------------------------------------------------------------------------------------------------------------


def _or_value(result: Any, value: Any) -> Any:
    return value if result is _NO_RESULT else result


def _number(x: float) -> int | float:
    """A number this coercion produced: an `int` when it is a whole safe integer."""
    if isinstance(x, float) and x.is_integer() and abs(x) <= _MAX_SAFE_INTEGER:
        return int(x)
    return x


def _truncate(x: float) -> int | float:
    """`Math.trunc`, as an `int` when the result is a safe integer."""
    if isinstance(x, int):
        return x
    whole = math.trunc(x)
    if abs(whole) <= _MAX_SAFE_INTEGER:
        return whole
    return x if x.is_integer() else float(whole)


def _convert_literal(literal: Any, value: Any) -> Any:
    """A literal (`const`, an enum value) taken from a value of another type that converts to exactly it."""
    if same_value(literal, value):
        return value
    if isinstance(literal, str):
        result = _try_string(value)
    elif isinstance(literal, bool):
        result = _try_boolean(value)
    elif is_number(literal):
        result = _try_number(value)
    else:
        result = _NO_RESULT
    return result if result is not _NO_RESULT and same_value(result, literal) else value


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _try_string(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if _is_finite_number(value):
        return number_string(value)
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return _NO_RESULT


def _try_number(value: Any) -> Any:
    if isinstance(value, bool):
        return 1 if value else 0
    if _is_finite_number(value):
        return value
    if value is None:
        return 0
    if not isinstance(value, str):
        return _NO_RESULT
    coerced = string_to_number(value)
    if math.isfinite(coerced):
        return _number(coerced)
    lowercase = value.lower()
    if lowercase == "false":
        return 0
    if lowercase == "true":
        return 1
    big = _big_int_of_string(value)
    return big if big is not None and abs(big) <= _MAX_SAFE_INTEGER else _NO_RESULT


_BIGINT_LITERAL = re.compile(r"-?(?:0|[1-9][0-9]*)n")
_DECIMAL_LITERAL = re.compile(r"-?(?:0|[1-9][0-9]*)\.[0-9]+")
_INTEGER_LITERAL = re.compile(r"-?(?:0|[1-9][0-9]*)")


def _big_int_of_string(value: str) -> int | None:
    """TypeBox's TryBigInt of a string: `12n`, `12.5` (truncated) or `12`."""
    if _BIGINT_LITERAL.fullmatch(value):
        return int(value[:-1])
    if _DECIMAL_LITERAL.fullmatch(value):
        return int(value.split(".")[0])
    if _INTEGER_LITERAL.fullmatch(value):
        return int(value)
    return None


def _try_boolean(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if _is_finite_number(value):
        return False if value == 0 else True if value == 1 else _NO_RESULT
    if value is None:
        return False
    if not isinstance(value, str):
        return _NO_RESULT
    lowercase = value.lower()
    if lowercase == "false" or value == "0":
        return False
    if lowercase == "true" or value == "1":
        return True
    return _NO_RESULT


def _try_null(value: Any) -> Any:
    if isinstance(value, bool):
        return None if value is False else _NO_RESULT
    if _is_finite_number(value):
        return None if value == 0 else _NO_RESULT
    if value is None:
        return None
    if not isinstance(value, str):
        return _NO_RESULT
    return None if value.lower() in ("undefined", "null") or value in ("", "0") else _NO_RESULT


# JavaScript's StrWhiteSpaceChar: WhiteSpace (Zs included) and LineTerminator.
_JS_WHITESPACE = "\t\n\v\f\r \xa0                　﻿"
_JS_DECIMAL = re.compile(r"[+-]?(?:Infinity|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)")
_JS_NON_DECIMAL = re.compile(r"0(?:[xX][0-9a-fA-F]+|[oO][0-7]+|[bB][01]+)")


def string_to_number(text: str) -> float:
    """JavaScript's `Number(text)`: NaN when `text` is not a number."""
    s = text.strip(_JS_WHITESPACE)
    if not s:
        return 0.0
    if _JS_NON_DECIMAL.fullmatch(s):
        try:
            return float(int(s, 0))
        except OverflowError:
            return math.inf
    if not _JS_DECIMAL.fullmatch(s):
        return math.nan
    if s.endswith("Infinity"):
        return -math.inf if s.startswith("-") else math.inf
    return float(s)


# ---------------------------------------------------------------------------------------------------------------------
# Primitive coercion by type (pass 3)
# ---------------------------------------------------------------------------------------------------------------------


def _types_of(schema: dict) -> list[str]:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return [schema_type]
    if isinstance(schema_type, list):
        return [t for t in schema_type if isinstance(t, str)]
    return []


def _matches_type(value: Any, type_name: str) -> bool:
    match type_name:
        case "number":
            return is_number(value)
        case "integer":
            return is_integer(value)
        case "boolean":
            return isinstance(value, bool)
        case "string":
            return isinstance(value, str)
        case "null":
            return value is None
        case "array":
            return isinstance(value, list)
        case "object":
            return isinstance(value, dict)
        case _:
            return False


def _coerce_primitive(value: Any, type_name: str) -> Any:
    match type_name:
        case "number" | "integer":
            if value is None:
                return 0
            if isinstance(value, str) and value.strip(_JS_WHITESPACE):
                parsed = string_to_number(value)
                if math.isfinite(parsed) and (type_name == "number" or parsed.is_integer()):
                    return _number(parsed)
            if isinstance(value, bool):
                return 1 if value else 0
            return value
        case "boolean":
            if value is None:
                return False
            if value == "true" or (is_number(value) and value == 1):
                return True
            if value == "false" or (is_number(value) and value == 0):
                return False
            return value
        case "string":
            if value is None:
                return ""
            if isinstance(value, bool):
                return "true" if value else "false"
            if is_number(value):
                return number_string(value)
            return value
        case "null":
            if value == "" or value is False or (is_number(value) and value == 0):
                return None
            return value
        case _:
            return value
