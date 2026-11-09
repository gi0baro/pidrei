"""Port of pi-schema's explainer (packages/schema/src/explain.ts): the errors of
a failed value, a walk over the schema that reports each failing keyword at
the value it failed on. It runs only for values that failed the check, and
asks the engine (engine.py) whether a value passes a subschema.

pi keeps the `$ref` memo on the explainer for the length of one `errors()`
call; here each call walks with its own `_Walk`, so one explainer serves
concurrent calls. pi's draft 7 branches are not ported: tool arguments are
explained under draft 2020-12, as pi's validation compiles them. pi's `bytes`
type (a Uint8Array) has no JSON value and no branch here.
"""

import math
import re
from typing import Any

from pidrei_ai.utils.schema.checks import (
    TYPE_CHECKS,
    const_check,
    contains_check,
    enum_check,
    is_integer,
    is_multiple,
    is_number,
    js_string,
    json_stringify,
    number_string,
    same_value,
    unique_check,
)
from pidrei_ai.utils.schema.engine import Engine, Scope
from pidrei_ai.utils.schema.types import ValidationError


_ABSENT = object()

# Enum errors name the allowed values; a long list (an enum of thousands) is cut, so the message stays readable.
_LISTED_VALUES = 20


def list_values(values: list) -> str:
    """`values` as JSON, at most 20 of them, then how many more there are."""
    if len(values) <= _LISTED_VALUES:
        return json_stringify(values)
    listed = json_stringify(values[:_LISTED_VALUES])
    return f"{listed[:-1]}, ... and {len(values) - _LISTED_VALUES} more]"


def type_name(v: Any) -> str:
    """The JSON type of `v` as the messages name it: `integer` for a whole number."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    if is_integer(v):
        return "integer"
    if is_number(v):
        return "number"
    if isinstance(v, str):
        return "string"
    return type(v).__name__


def _contains_message(minimum: float, maximum: float) -> str:
    if maximum == math.inf:
        return f"expected at least {number_string(minimum)} matching {'item' if minimum == 1 else 'items'}"
    if minimum == 0:
        return f"expected at most {number_string(maximum)} matching {'item' if maximum == 1 else 'items'}"
    return f"expected between {number_string(minimum)} and {number_string(maximum)} matching items"


def pointer(path: str, key: str | int) -> str:
    """`path` with `key` appended, escaped as JSON Pointer does."""
    return f"{path}/{str(key).replace('~', '~0').replace('/', '~1')}"


def _identity_key(value: Any) -> tuple:
    """A key that tells primitives apart as JavaScript's Set does (`1` and `1.0`
    alike, `true` apart from `1`)."""
    if isinstance(value, bool):
        return ("boolean", value)
    if is_number(value):
        return ("number", value)
    if isinstance(value, dict | list):
        return ("object", id(value))
    return (type(value).__name__, value)


class Explainer:
    """Explains failures of one compiled schema."""

    __slots__ = ("_engine",)

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def errors(self, value: Any) -> list[ValidationError]:
        """The errors of `value`, at least one (`value` failed the check)."""
        out: list[ValidationError] = []
        _Walk(self._engine).walk(self._engine.schema, self._engine.root_scope, value, "", out)
        if not out:
            out.append(ValidationError("", "schema", "invalid"))
        return out


class _Walk:
    """One `errors()` call."""

    __slots__ = ("_engine", "_memo")

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        # The errors of `$ref` targets for object values: a recursive union explains each branch to pick the one to
        # blame, and every branch would explain the same children again, exponentially in the depth. Keyed by the
        # target and the value's identity; the value is kept so its id cannot be reused while the walk runs.
        self._memo: dict[tuple[int, int], tuple[Any, str, list[ValidationError]]] = {}

    def _ok(self, node: Any, b: Scope, v: Any) -> bool:
        return self._engine.check(node, b, v)

    def _ok_in(self, child: Any, b: Scope, v: Any) -> bool:
        return self._engine.check(child, self._engine.enter(child, b), v)

    def _walk_in(self, child: Any, b: Scope, v: Any, path: str, out: list[ValidationError]) -> None:
        self.walk(child, self._engine.enter(child, b), v, path, out)

    def walk(self, node: Any, b: Scope, v: Any, path: str, out: list[ValidationError]) -> None:
        """Explains `v` against `node`, whose scope is `b`."""
        if node is True:
            return
        if node is False:
            out.append(ValidationError(path, "false", "no value is allowed here"))
            return
        if not isinstance(node, dict):
            return
        if self._ok(node, b, v):
            return
        start = len(out)

        def push(keyword: str, message: str, at: str = path) -> None:
            out.append(ValidationError(at, keyword, message))

        if "type" in node:
            types = node["type"] if isinstance(node["type"], list) else [node["type"]]
            if not any(isinstance(t, str) and t in TYPE_CHECKS and TYPE_CHECKS[t](v) for t in types):
                push("type", f"expected {' or '.join(js_string(t) for t in types)}, got {type_name(v)}")
                return
        if isinstance(node.get("enum"), list) and not enum_check(node["enum"])(v):
            push("enum", f"expected one of {list_values(node['enum'])}")
        if "const" in node and not const_check(node["const"])(v):
            push("const", f"expected {json_stringify(node['const'])}")
        if isinstance(v, str):
            self._walk_string(node, v, push)
        if is_number(v):
            self._walk_number(node, v, push)
        if isinstance(v, dict):
            self._walk_object(node, b, v, path, out, push)
        if isinstance(v, list):
            self._walk_array(node, b, v, path, out, push)
        if isinstance(node.get("allOf"), list):
            for s in node["allOf"]:
                self._walk_in(s, b, v, path, out)
        if isinstance(node.get("anyOf"), list):
            self._walk_union(node["anyOf"], b, v, path, out, exactly_one=False)
        if isinstance(node.get("oneOf"), list):
            self._walk_union(node["oneOf"], b, v, path, out, exactly_one=True)
        if "not" in node and self._ok_in(node["not"], b, v):
            push("not", "must not match the schema in not")
        if "if" in node:
            branch = node.get("then", _ABSENT) if self._ok_in(node["if"], b, v) else node.get("else", _ABSENT)
            if branch is not _ABSENT:
                self._walk_in(branch, b, v, path, out)
        if isinstance(node.get("$ref"), str):
            self._walk_ref(node["$ref"], b, v, path, out)
        if len(out) == start and ("unevaluatedProperties" in node or "unevaluatedItems" in node):
            self._walk_unevaluated(node, b, v, path, out)
        if len(out) == start and isinstance(node.get("$dynamicRef"), str):
            push("$dynamicRef", "does not match the dynamically referenced schema")

    def _walk_ref(self, ref: str, b: Scope, v: Any, path: str, out: list[ValidationError]) -> None:
        target = self._engine.resolve(ref, b)
        if target is None:
            out.append(ValidationError(path, "$ref", f"unresolved reference {ref}"))
            return
        node, base = target
        if not isinstance(node, dict) or not isinstance(v, dict | list):
            self.walk(node, base, v, path, out)
            return
        key = (id(node), id(v))
        known = self._memo.get(key)
        if known is not None and known[0] is v and known[1] == path:
            out.extend(known[2])
            return
        errors: list[ValidationError] = []
        self.walk(node, base, v, path, errors)
        self._memo[key] = (v, path, errors)
        out.extend(errors)

    def _walk_string(self, node: dict, v: str, push) -> None:
        n = len(v)
        min_length = node.get("minLength")
        if is_number(min_length) and n < min_length:
            push("minLength", f"expected at least {number_string(min_length)} characters")
        max_length = node.get("maxLength")
        if is_number(max_length) and n > max_length:
            push("maxLength", f"expected at most {number_string(max_length)} characters")
        pattern = node.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, v) is None:
            push("pattern", f"expected to match {pattern}")
        format_name = node.get("format")
        if isinstance(format_name, str) and self._engine.conforms(v, format_name) is False:
            push("format", f"expected format {format_name}")

    def _walk_number(self, node: dict, v: float, push) -> None:
        if is_number(minimum := node.get("minimum")) and v < minimum:
            push("minimum", f"expected >= {number_string(minimum)}")
        if is_number(maximum := node.get("maximum")) and v > maximum:
            push("maximum", f"expected <= {number_string(maximum)}")
        if is_number(exclusive := node.get("exclusiveMinimum")) and v <= exclusive:
            push("exclusiveMinimum", f"expected > {number_string(exclusive)}")
        if is_number(exclusive := node.get("exclusiveMaximum")) and v >= exclusive:
            push("exclusiveMaximum", f"expected < {number_string(exclusive)}")
        if is_number(multiple := node.get("multipleOf")) and not is_multiple(v, multiple):
            push("multipleOf", f"expected a multiple of {number_string(multiple)}")

    def _walk_object(self, node: dict, b: Scope, v: dict, path: str, out: list[ValidationError], push) -> None:
        properties = node["properties"] if isinstance(node.get("properties"), dict) else {}
        required = node["required"] if isinstance(node.get("required"), list) else []
        for k in required:
            if isinstance(k, str) and k not in v:
                push("required", "missing", pointer(path, k))
        for k, s in properties.items():
            if k in v:
                self._walk_in(s, b, v[k], pointer(path, k), out)
        patterns = list(node["patternProperties"].items()) if isinstance(node.get("patternProperties"), dict) else []
        for k, x in v.items():
            matched = k in properties
            for p, s in patterns:
                if re.search(p, k) is not None:
                    matched = True
                    self._walk_in(s, b, x, pointer(path, k), out)
            if not matched and "additionalProperties" in node:
                additional = node["additionalProperties"]
                if additional is False:
                    push("additionalProperties", "unknown field", pointer(path, k))
                else:
                    self._walk_in(additional, b, x, pointer(path, k), out)
            if "propertyNames" in node and not self._ok_in(node["propertyNames"], b, k):
                push("propertyNames", "invalid property name", pointer(path, k))
        count = len(v)
        if is_number(minimum := node.get("minProperties")) and count < minimum:
            push("minProperties", f"expected at least {number_string(minimum)} properties")
        if is_number(maximum := node.get("maxProperties")) and count > maximum:
            push("maxProperties", f"expected at most {number_string(maximum)} properties")
        dependencies = {
            **(node["dependentRequired"] if isinstance(node.get("dependentRequired"), dict) else {}),
            **(node["dependencies"] if isinstance(node.get("dependencies"), dict) else {}),
        }
        for k, d in dependencies.items():
            if k not in v:
                continue
            if not isinstance(d, list):
                self._walk_in(d, b, v, path, out)
                continue
            for r in d:
                if isinstance(r, str) and r not in v:
                    push("dependentRequired", f"required when {k} is present", pointer(path, r))
        if isinstance(node.get("dependentSchemas"), dict):
            for k, s in node["dependentSchemas"].items():
                if k in v:
                    self._walk_in(s, b, v, path, out)

    def _walk_array(self, node: dict, b: Scope, v: list, path: str, out: list[ValidationError], push) -> None:
        items = node.get("items", _ABSENT)
        if isinstance(items, list):
            prefix, rest = items, node.get("additionalItems", _ABSENT)
        else:
            prefix = node["prefixItems"] if isinstance(node.get("prefixItems"), list) else []
            rest = items
        for i, item in enumerate(v):
            s = prefix[i] if i < len(prefix) else rest
            if s is not _ABSENT:
                self._walk_in(s, b, item, f"{path}/{i}", out)
        if is_number(minimum := node.get("minItems")) and len(v) < minimum:
            push("minItems", f"expected at least {number_string(minimum)} items")
        if is_number(maximum := node.get("maxItems")) and len(v) > maximum:
            push("maxItems", f"expected at most {number_string(maximum)} items")
        if node.get("uniqueItems") is True and not unique_check(v):
            push("uniqueItems", "expected unique items")
        if "contains" in node:
            minimum = node["minContains"] if is_number(node.get("minContains")) else 1
            maximum = node["maxContains"] if is_number(node.get("maxContains")) else math.inf
            contains = node["contains"]
            scope = self._engine.enter(contains, b)
            if not contains_check(lambda x: self._ok(contains, scope, x), minimum, maximum)(v):
                push("contains", _contains_message(minimum, maximum))

    def _walk_union(
        self, branches: list, b: Scope, v: Any, path: str, out: list[ValidationError], *, exactly_one: bool
    ) -> None:
        matches = sum(1 for s in branches if self._ok_in(s, b, v))
        if matches == 1 or (not exactly_one and matches > 0):
            return
        keyword = "oneOf" if exactly_one else "anyOf"
        if matches > 1:
            out.append(ValidationError(path, keyword, "matches more than one alternative"))
            return
        # A tagged union reports the branch its tag selects; otherwise the branch whose first error is deepest.
        tag = self._tag_of(branches, b)
        if tag is not None and isinstance(v, dict):
            key, values = tag
            index = next((i for i, x in enumerate(values) if key in v and same_value(x, v[key])), -1)
            if index < 0:
                out.append(ValidationError(pointer(path, key), keyword, f"expected one of {list_values(values)}"))
            else:
                self._walk_in(branches[index], b, v, path, out)
            return
        # An object is blamed on the branch that declares most of its properties (the one it was most likely meant
        # for), then on the branch whose first error is deepest.
        best: tuple[list[ValidationError], int] | None = None
        for s in branches:
            errors: list[ValidationError] = []
            self._walk_in(s, b, v, path, errors)
            if not errors:
                continue
            declared = self._declared_keys(s, b, v) if isinstance(v, dict) else 0
            if (
                best is None
                or declared > best[1]
                or (declared == best[1] and len(errors[0].path) > len(best[0][0].path))
            ):
                best = (errors, declared)
        out.extend(best[0] if best is not None else [ValidationError(path, keyword, "matches no alternative")])

    def _branch(self, s: Any, b: Scope) -> Any:
        """A union branch, through its `$ref`."""
        if isinstance(s, dict) and isinstance(s.get("$ref"), str):
            target = self._engine.resolve(s["$ref"], self._engine.enter(s, b))
            return False if target is None else target[0]
        return s

    def _declared_keys(self, s: Any, b: Scope, v: dict) -> int:
        """How many of the keys of `v` the branch `s` (through a `$ref`) declares in `properties`."""
        node = self._branch(s, b)
        properties = node["properties"] if isinstance(node, dict) and isinstance(node.get("properties"), dict) else {}
        return sum(1 for k in v if k in properties)

    def _tag_of(self, branches: list, b: Scope) -> tuple[str, list] | None:
        resolved = [self._branch(s, b) for s in branches]
        first = resolved[0] if resolved else None
        if not isinstance(first, dict) or not isinstance(first.get("properties"), dict):
            return None
        for key in first["properties"]:
            values = []
            for s in resolved:
                tag = (
                    s["properties"].get(key) if isinstance(s, dict) and isinstance(s.get("properties"), dict) else None
                )
                values.append(tag["const"] if isinstance(tag, dict) and "const" in tag else _ABSENT)
            primitive = all(x is not _ABSENT and not isinstance(x, dict | list) for x in values)
            if primitive and len({_identity_key(x) for x in values}) == len(values):
                return key, values
        return None

    def _walk_unevaluated(self, node: dict, b: Scope, v: Any, path: str, out: list[ValidationError]) -> None:
        rest = {k: x for k, x in node.items() if k not in ("unevaluatedProperties", "unevaluatedItems")}
        a = self._engine.annotations(rest, b, v)
        if isinstance(v, dict) and "unevaluatedProperties" in node and not a.all_props:
            unevaluated = node["unevaluatedProperties"]
            for k, x in v.items():
                if k in a.props:
                    continue
                if unevaluated is False:
                    out.append(ValidationError(pointer(path, k), "unevaluatedProperties", "unknown field"))
                else:
                    self._walk_in(unevaluated, b, x, pointer(path, k), out)
        if isinstance(v, list) and "unevaluatedItems" in node and not a.all_items:
            unevaluated = node["unevaluatedItems"]
            for i in range(a.items, len(v)):
                x = v[i]
                if i in a.indexes:
                    continue
                if unevaluated is False:
                    out.append(ValidationError(f"{path}/{i}", "unevaluatedItems", "unexpected item"))
                else:
                    self._walk_in(unevaluated, b, x, f"{path}/{i}", out)
