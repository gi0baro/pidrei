"""The engine under pi-schema's rules: `jsonschema`, where pi has its own
compiler and resolver (packages/schema/src/compiler.ts, resolver.ts).

The explainer and the coercer ask the engine four things, as pi's ask its
compiler: whether a value passes a subschema (`check`), what a `$ref` names
(`resolve`), whether a string has a format (`conforms`) and which keys or
items a schema evaluates (for `unevaluatedProperties`/`unevaluatedItems`). A
scope is the `referencing` resolver of one schema node, pi's `base`: `enter`
moves into a node's `$id`, and a resolved reference comes with its target's
own scope.

The validator is `jsonschema`'s draft 2020-12, with the keywords whose answer
differs from pi's compiler answered as pi does:

- `type`: a type name the draft does not define (TypeBox's `function`,
  `bigint`, `undefined`) matches no value, instead of raising;
- `$ref`: a reference that names nothing (a missing definition, a document
  that is not registered) fails the value at its place, instead of raising,
  when the schema is compiled with `unresolved_refs="reject"`; nothing is
  ever retrieved over the network;
- `multipleOf`: exact for decimals, as pi's `isMultiple`;
- draft 7's array `items` (a tuple), `additionalItems` and `dependencies`,
  which pi's compiler also reads under draft 2020-12;
- number-valued keywords whose value is not a number (draft 4's boolean
  `exclusiveMinimum`) and applicators whose value has the wrong shape are
  ignored, as pi's compiler ignores them.

`compile()` raises for a schema whose subschema is neither an object nor a
boolean, and, unless references are rejected, for a `$ref` that names
nothing, as pi's compile throws; only subschemas the root reaches count.

Known differences from pi's engine: `pattern` and `patternProperties` run on
Python's `re`, not ECMAScript regular expressions (`\\p{L}` is an error
there, raised when a value reaches it); formats follow the Python checkers
(formats.py); a nested `$schema` switches the subschema to that draft, as
`jsonschema` does; `$dynamicRef` resolves as `jsonschema` resolves it.
"""

from collections.abc import Callable, Iterator
from typing import Any

from jsonschema import Draft7Validator, Draft202012Validator, FormatChecker, validators
from jsonschema._utils import find_evaluated_item_indexes_by_schema, find_evaluated_property_keys_by_schema
from jsonschema.exceptions import ValidationError as EngineError
from referencing import Registry
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012

from pidrei_ai.utils.schema.checks import TYPE_CHECKS, is_multiple, is_number, json_stringify


#: The scope of a schema node: the `referencing` resolver its references
#: resolve with (`referencing` exports no name for the class).
Scope = type(Registry().resolver())

type Keyword = Callable[[Any, Any, Any, dict], Iterator[EngineError]]

_DRAFT = Draft202012Validator.VALIDATORS
_LEGACY = Draft7Validator.VALIDATORS


def _type(validator, types, instance, schema) -> Iterator[EngineError]:
    names = types if isinstance(types, list) else [types]
    if not any(isinstance(name, str) and name in TYPE_CHECKS and TYPE_CHECKS[name](instance) for name in names):
        yield EngineError(f"{instance!r} is not of type {types!r}")


def _rejecting(ref: Keyword) -> Keyword:
    def check(validator, value, instance, schema) -> Iterator[EngineError]:
        errors = ref(validator, value, instance, schema)
        # The reference is looked up when the first error is asked for; a
        # reference further down is answered by its own keyword.
        try:
            first = next(errors, None)
        except Unresolvable:
            yield EngineError(f"unresolved reference {value}")
            return
        if first is not None:
            yield first
            yield from errors

    return check


def _multiple_of(validator, d, instance, schema) -> Iterator[EngineError]:
    if is_number(d) and is_number(instance) and not is_multiple(instance, d):
        yield EngineError(f"{instance!r} is not a multiple of {d!r}")


def _branch_passes(validator, instance, branches: list, index: int) -> bool:
    return next(validator.descend(instance, branches[index], schema_path=index), None) is None


# `jsonschema`'s unions collect every error of every branch for their error
# context, so a failing branch is explored to its end; pi's stop at the answer.
def _any_of(validator, branches, instance, schema) -> Iterator[EngineError]:
    if not any(_branch_passes(validator, instance, branches, index) for index in range(len(branches))):
        yield EngineError(f"{instance!r} is not valid under any of the given schemas")


def _one_of(validator, branches, instance, schema) -> Iterator[EngineError]:
    matches = 0
    for index in range(len(branches)):
        if _branch_passes(validator, instance, branches, index):
            matches += 1
            if matches > 1:
                break
    if matches != 1:
        yield EngineError(f"{instance!r} is not valid under exactly one of the given schemas")


def _items(validator, items, instance, schema) -> Iterator[EngineError]:
    keyword = _LEGACY["items"] if isinstance(items, list) else _DRAFT["items"]
    yield from keyword(validator, items, instance, schema)


def _when(accepts: Callable[[Any], bool], keyword: Keyword) -> Keyword:
    def check(validator, value, instance, schema) -> Iterator[EngineError]:
        if accepts(value):
            yield from keyword(validator, value, instance, schema)

    return check


def _is_list(value: Any) -> bool:
    return isinstance(value, list)


def _is_dict(value: Any) -> bool:
    return isinstance(value, dict)


_NUMERIC = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "minProperties",
    "maxProperties",
)
_LISTS = ("required", "enum", "allOf", "prefixItems")
_MAPS = ("properties", "patternProperties", "dependentRequired", "dependentSchemas")

# Keywords that check a value through subschemas.
_APPLICATORS = frozenset(
    {
        "$dynamicRef",
        "$ref",
        "additionalItems",
        "additionalProperties",
        "allOf",
        "anyOf",
        "contains",
        "dependencies",
        "dependentSchemas",
        "if",
        "items",
        "not",
        "oneOf",
        "patternProperties",
        "prefixItems",
        "properties",
        "propertyNames",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)


def _assertions_first(schema: dict) -> list[tuple[str, Any]]:
    # A check stops at the first failing keyword. Run the ones that look only at
    # the value (`type`, `required`, ...) before those that descend, as pi's
    # compiled checks do: a recursive union whose branches each require a key
    # would otherwise descend into every child of every branch, exponentially
    # in the depth, before finding the key missing. The order changes no answer.
    return sorted(schema.items(), key=lambda keyword: keyword[0] in _APPLICATORS)


_Validator = validators.create(
    meta_schema=Draft202012Validator.META_SCHEMA,
    validators={
        **_DRAFT,
        "type": _type,
        "$ref": _rejecting(_DRAFT["$ref"]),
        "multipleOf": _multiple_of,
        "anyOf": _when(_is_list, _any_of),
        "oneOf": _when(_is_list, _one_of),
        "items": _items,
        "additionalItems": _LEGACY["additionalItems"],
        "dependencies": _when(_is_dict, _LEGACY["dependencies"]),
        **{name: _when(is_number, _DRAFT[name]) for name in _NUMERIC},
        **{name: _when(_is_list, _DRAFT[name]) for name in _LISTS},
        **{name: _when(_is_dict, _DRAFT[name]) for name in _MAPS},
    },
    type_checker=Draft202012Validator.TYPE_CHECKER,
    format_checker=None,
    id_of=Draft202012Validator.ID_OF,
    applicable_validators=_assertions_first,
)

# Keywords whose value is one subschema, a list of them, or a map to them.
_ONE = (
    "additionalProperties",
    "propertyNames",
    "contains",
    "not",
    "if",
    "then",
    "else",
    "unevaluatedProperties",
    "unevaluatedItems",
    "additionalItems",
)
_MANY = ("allOf", "anyOf", "oneOf", "prefixItems")
_BY_KEY = ("properties", "patternProperties", "dependentSchemas")


def _subschemas(node: dict) -> Iterator[Any]:
    for key in _ONE:
        if key in node:
            yield node[key]
    if "items" in node:
        items = node["items"]
        if isinstance(items, list):
            yield from items
        else:
            yield items
    for key in _MANY:
        if isinstance(node.get(key), list):
            yield from node[key]
    for key in _BY_KEY:
        if isinstance(node.get(key), dict):
            yield from node[key].values()
    if isinstance(node.get("dependencies"), dict):
        yield from (d for d in node["dependencies"].values() if not isinstance(d, list))


def _malformed(node: Any) -> ValueError:
    return ValueError(f"A schema must be an object or a boolean, got {json_stringify(node)}")


class Engine:
    """One compiled schema on `jsonschema`. Immutable once built, so one engine
    serves concurrent tool calls; every walk keeps its own state."""

    __slots__ = ("_formats", "_root", "root_scope", "schema")

    def __init__(self, schema: dict | bool, formats: FormatChecker | None, *, reject_unresolved: bool) -> None:
        if not isinstance(schema, dict | bool):
            raise _malformed(schema)
        self.schema = schema
        # An empty registry: a reference to another document names nothing
        # unless that document is the root or inside it.
        self.root_scope: Scope = Registry().resolver_with_root(DRAFT202012.create_resource(schema))
        self._formats = formats
        self._root = _Validator(schema, format_checker=formats, registry=Registry(), _resolver=self.root_scope)
        self._verify(reject_unresolved)

    def _verify(self, reject_unresolved: bool) -> None:
        pending: list[tuple[Any, Scope]] = [(self.schema, self.root_scope)]
        seen: set[int] = set()
        while pending:
            node, scope = pending.pop()
            if not isinstance(node, dict) or id(node) in seen:
                continue
            seen.add(id(node))
            for child in _subschemas(node):
                if not isinstance(child, dict | bool):
                    raise _malformed(child)
                pending.append((child, self.enter(child, scope)))
            ref = node.get("$ref")
            if not isinstance(ref, str):
                continue
            target = self.resolve(ref, scope)
            if target is None:
                if not reject_unresolved:
                    raise ValueError(f"Unresolved $ref {ref}")
                continue
            if not isinstance(target[0], dict | bool):
                raise _malformed(target[0])
            pending.append(target)

    def check(self, node: Any, scope: Scope, value: Any) -> bool:
        """Whether `value` passes `node`, whose scope is `scope`."""
        if node is self.schema and scope is self.root_scope:
            return self._root.is_valid(value)
        try:
            return next(self._root.descend(value, node, resolver=scope), None) is None
        except Unresolvable:
            return False

    def enter(self, node: Any, scope: Scope) -> Scope:
        """The scope of `node`, a subschema of the node whose scope is `scope`."""
        if isinstance(node, dict) and isinstance(node.get("$id"), str):
            return scope.in_subresource(DRAFT202012.create_resource(node))
        return scope

    def resolve(self, ref: str, scope: Scope) -> tuple[Any, Scope] | None:
        """The schema `ref` names from `scope`, with its scope; None when it names nothing."""
        try:
            resolved = scope.lookup(ref)
        except Unresolvable:
            return None
        return resolved.contents, resolved.resolver

    def conforms(self, value: str, format_name: str) -> bool | None:
        """Whether `value` has the format; None when the format is not checked."""
        if self._formats is None or format_name not in self._formats.checkers:
            return None
        return self._formats.conforms(value, format_name)

    def evaluated_keys(self, node: dict, scope: Scope, value: dict) -> set[str]:
        """The keys of `value` that `node` evaluates (`unevaluatedProperties`)."""
        validator = self._root.evolve(schema=node, _resolver=scope)
        return set(find_evaluated_property_keys_by_schema(validator, value, node))

    def evaluated_items(self, node: dict, scope: Scope, value: list) -> set[int]:
        """The indexes of `value` that `node` evaluates (`unevaluatedItems`)."""
        validator = self._root.evolve(schema=node, _resolver=scope)
        return set(find_evaluated_item_indexes_by_schema(validator, value, node))
