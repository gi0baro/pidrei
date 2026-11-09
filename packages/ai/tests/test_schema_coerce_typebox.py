"""Mirror of pi-schema's coerce-typebox.test.ts: TypeBox 1.3.27's Value.Convert
tests (MIT, (c) Haydn Paterson), on the conversion pass of the coercion
(`Coercer.convert_only`).

pi builds the schemas with TypeBox and reads only their JSON structure; here
`Type` emits that JSON. The cases pi skips (no JSON form) are not mirrored,
nor are those whose value is `undefined` or a BigInt, which no JSON argument
is. Values are compared through `json_stringify`, so `true` is never taken for
`1`.
"""

import pytest

from pidrei_ai.utils.schema import FORMATS, compile
from pidrei_ai.utils.schema.checks import json_stringify
from pidrei_ai.utils.schema.coerce import Coercer
from pidrei_ai.utils.schema.engine import Engine


_NUMBER_KEY = "^-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?$"


class _Optional:
    def __init__(self, schema: dict) -> None:
        self.schema = schema


class Type:
    """The JSON Schema TypeBox's builders emit (`Type.number()` is `Type.number()`)."""

    @staticmethod
    def any() -> dict:
        return {}

    unknown = any

    @staticmethod
    def never() -> dict:
        return {"not": {}}

    @staticmethod
    def number() -> dict:
        return {"type": "number"}

    @staticmethod
    def integer() -> dict:
        return {"type": "integer"}

    @staticmethod
    def string(**options) -> dict:
        return {"type": "string", **options}

    @staticmethod
    def boolean(**options) -> dict:
        return {"type": "boolean", **options}

    @staticmethod
    def null() -> dict:
        return {"type": "null"}

    @staticmethod
    def array(items: dict) -> dict:
        return {"type": "array", "items": items}

    @staticmethod
    def optional(schema: dict) -> _Optional:
        return _Optional(schema)

    @staticmethod
    def object(properties: dict, **options) -> dict:
        props = {k: v.schema if isinstance(v, _Optional) else v for k, v in properties.items()}
        required = [k for k, v in properties.items() if not isinstance(v, _Optional)]
        return {"type": "object", "properties": props, **({"required": required} if required else {}), **options}

    @staticmethod
    def union(schemas: list) -> dict:
        return {"anyOf": schemas}

    @staticmethod
    def intersect(schemas: list) -> dict:
        return {"allOf": schemas}

    @staticmethod
    def literal(value) -> dict:
        return {"const": value}

    @staticmethod
    def enum(values: list) -> dict:
        return {"enum": values}

    @staticmethod
    def record(key: dict, value: dict, **options) -> dict:
        pattern = key.get("pattern", _NUMBER_KEY if key.get("type") == "number" else "^.*$")
        return {"type": "object", "patternProperties": {pattern: value}, **options}

    @staticmethod
    def tuple(items: list) -> dict:
        return {"type": "array", "prefixItems": items, "items": False, "minItems": len(items), "maxItems": len(items)}

    @staticmethod
    def template_literal(pattern: str) -> dict:
        return {"type": "string", "pattern": pattern}


def convert(schema: dict, value):
    """Pass 2 alone, as TypeBox's Value.Convert(schema, value)."""
    return Coercer(Engine(schema, FORMATS, reject_unresolved=False)).convert_only(value)


def check(schema: dict, value) -> bool:
    return compile(schema, formats=FORMATS).check(value)


SAMPLE = [1, 3.14, "1", "3.14", True, False, "true", "false", "hello"]
X_Y = [Type.object({"x": Type.number()}), Type.object({"y": Type.number()})]

CONVERTS: list[tuple[str, dict, object, object]] = [
    ("Any 1", Type.any(), None, None),
    ("Any 3", Type.any(), "hello", "hello"),
    ("Any 4", Type.any(), 42, 42),
    ("Any 5", Type.any(), True, True),
    ("Any 6", Type.any(), {"x": 1}, {"x": 1}),
    ("Any 7", Type.any(), [1, 2, 3], [1, 2, 3]),
    ("Array 1", Type.array(Type.number()), SAMPLE, [1, 3.14, 1, 3.14, 1, 0, 1, 0, "hello"]),
    ("Array 2", Type.array(Type.boolean()), SAMPLE, [True, 3.14, True, "3.14", True, False, True, False, "hello"]),
    (
        "Array 3",
        Type.array(Type.string()),
        SAMPLE,
        ["1", "3.14", "1", "3.14", "true", "false", "true", "false", "hello"],
    ),
    ("Array 4", Type.array(Type.number()), "1", [1]),
    ("Array 5", Type.array(Type.number()), [], []),
    ("Array 6", Type.array(Type.number()), 42, [42]),
    ("Array 7", Type.array(Type.boolean()), True, [True]),
    ("Array 8", Type.array(Type.null()), None, [None]),
    ("Array 11", Type.array(Type.object({})), {}, [{}]),
    ("Array 12", Type.array(Type.object({"x": Type.number()})), {"x": 1}, [{"x": 1}]),
    ("Array 13", Type.array(Type.number()), "42", [42]),
    ("Array 14", Type.array(Type.boolean()), "true", [True]),
    ("Array 15", Type.array(Type.boolean()), 0, [False]),
    ("Array 16", Type.array(Type.number()), "hello", ["hello"]),
    ("Array 17", Type.array(Type.number()), [1], [1]),
    ("Array 19", Type.array(Type.number()), {"0": 1, "length": 1}, [{"0": 1, "length": 1}]),
    ("Boolean 1", Type.boolean(), "1", True),
    ("Boolean 2", Type.boolean(), "3.14", "3.14"),
    ("Boolean 3", Type.boolean(), "true", True),
    ("Boolean 4", Type.boolean(), "false", False),
    ("Boolean 5", Type.boolean(), 1, True),
    ("Boolean 6", Type.boolean(), 3.14, 3.14),
    ("Boolean 7", Type.boolean(), 1.1, 1.1),
    ("Boolean 8", Type.boolean(), True, True),
    ("Boolean 9", Type.boolean(), False, False),
    ("Boolean cast hello", Type.boolean(), "hello", "hello"),
    ("Boolean cast true", Type.boolean(), "true", True),
    ("Boolean cast TRUE", Type.boolean(), "TRUE", True),
    ("Boolean cast false", Type.boolean(), "false", False),
    ("Boolean cast 0", Type.boolean(), "0", False),
    ("Boolean cast 1", Type.boolean(), "1", True),
    ("Boolean cast 0 default true", Type.boolean(default=True), "0", False),
    ("Boolean cast 1 default false", Type.boolean(default=False), "1", True),
    ("Boolean cast 2 default true", Type.boolean(default=True), "2", "2"),
    ("Boolean cast number 0", Type.boolean(), 0, False),
    ("Boolean cast number 1", Type.boolean(), 1, True),
    ("Boolean cast number 2", Type.boolean(), 2, 2),
    ("Boolean cast number 0 default true", Type.boolean(default=True), 0, False),
    ("Boolean cast number 1 default false", Type.boolean(default=False), 1, True),
    ("Boolean cast number 2 default true", Type.boolean(default=True), 2, 2),
    ("Boolean cast object", Type.boolean(), {}, {}),
    ("Boolean cast array", Type.boolean(), [], []),
    ("Boolean coverage null", Type.boolean(), None, False),
    ("Enum R0", Type.object({"x": Type.enum([1, 2, "hello"])}), {"x": "1"}, {"x": 1}),
    ("Enum R1", Type.object({"x": Type.enum([1, 2, "hello"])}), {"x": "2"}, {"x": 2}),
    ("Enum R2", Type.object({"x": Type.enum([1, 2, "hello"])}), {"x": "hello"}, {"x": "hello"}),
    ("Enum R3", Type.object({"x": Type.enum([1, 2, "hello"])}), {"x": "world"}, {"x": "world"}),
    ("Integer 1", Type.integer(), "3.14", 3),
    ("Integer 2", Type.integer(), "42", 42),
    ("Integer 3", Type.integer(), True, 1),
    ("Integer 4", Type.integer(), False, 0),
    ("Integer 5", Type.integer(), 3.14, 3),
    ("Integer cast hello", Type.integer(), "hello", "hello"),
    ("Integer cast -0", Type.integer(), "-0", 0),
    ("Integer cast -100", Type.integer(), "-100", -100),
    ("Integer cast 42", Type.integer(), 42, 42),
    ("Integer cast object", Type.integer(), {}, {}),
    ("Integer cast array", Type.integer(), [], []),
    ("Integer 1147 a", Type.integer(), 1738213389080, 1738213389080),
    ("Integer 1147 b", Type.integer(), 1738213389080.5555, 1738213389080),
    ("Integer 1147 c", Type.integer(), "1738213389080", 1738213389080),
    ("Integer 1147 d", Type.integer(), "1738213389080.555", 1738213389080),
    ("Intersect 1", Type.intersect(X_Y), {"x": "1", "y": "2"}, {"x": 1, "y": 2}),
    ("Intersect 2", Type.intersect([Type.number(), *X_Y]), {"x": "1", "y": "2"}, {"x": 1, "y": 2}),
    ("Intersect 3", Type.intersect([*X_Y, Type.number()]), {"x": "3", "y": "4"}, {"x": 3, "y": 4}),
    # Differs from TypeBox: allOf converts with each member in turn; TypeBox first merges an intersection into one
    # object and drops the number.
    ("Intersect 4", Type.intersect([Type.number(), *X_Y]), "123", 123),
    ("Literal 1", Type.literal("1"), 1, "1"),
    ("Literal 2", Type.literal("1"), 2, 2),
    ("Literal 3", Type.literal("true"), True, "true"),
    ("LiteralNumber 1", Type.literal(3.14), "3.14", 3.14),
    ("LiteralNumber 2", Type.literal(3.14), "3.15", "3.15"),
    ("LiteralNumber 3", Type.literal(1), True, 1),
    ("LiteralNumber 4", Type.literal(0), False, 0),
    ("LiteralNumber 5", Type.literal(2), True, True),
    ("LiteralBoolean 1", Type.literal(True), 3.14, 3.14),
    ("LiteralBoolean 2", Type.literal(True), 1, True),
    ("LiteralBoolean 3", Type.literal(True), "true", True),
    ("LiteralBoolean 4", Type.literal(False), "false", False),
    ("LiteralBoolean 5", Type.literal(True), "1", True),
    ("LiteralBoolean 6", Type.literal(False), "0", False),
    ("LiteralBoolean 7", Type.literal(False), "2", "2"),
    ("Never 1", Type.never(), True, True),
    ("Never 2", Type.never(), 42, 42),
    ("Never 3", Type.never(), "true", "true"),
    ("Null 1", Type.null(), "null", None),
    ("Null 2", Type.null(), "NULL", None),
    ("Null 3", Type.null(), "nil", "nil"),
    ("Null 4", Type.null(), "undefined", None),
    ("Null 5", Type.null(), "hello", "hello"),
    ("Null 6", Type.null(), None, None),
    ("Null 7", Type.null(), 0, None),
    ("Null 8", Type.null(), False, None),
    ("Null 9", Type.null(), "", None),
    ("Null coverage true", Type.null(), True, True),
    ("Null coverage 1", Type.null(), 1, 1),
    ("Number 1", Type.number(), "3.14", 3.14),
    ("Number 2", Type.number(), "42", 42),
    ("Number 3", Type.number(), True, 1),
    ("Number 4", Type.number(), False, 0),
    ("Number 5", Type.number(), 3.14, 3.14),
    ("Number cast hello", Type.number(), "hello", "hello"),
    ("Number cast -0", Type.number(), "-0", 0),
    ("Number cast -100", Type.number(), "-100", -100),
    ("Number cast 42", Type.number(), 42, 42),
    ("Number cast object", Type.number(), {}, {}),
    ("Number cast array", Type.number(), [], []),
    ("Number coverage null", Type.number(), None, 0),
    ("Number coverage false", Type.number(), "false", 0),
    ("Number coverage true", Type.number(), "true", 1),
    ("Number coverage 123", Type.number(), "123", 123),
    ("Number coverage 123.456", Type.number(), "123.456", 123.456),
    ("Number coverage 123n", Type.number(), "123n", 123),
    # The trailing 0 makes it too large for a safe integer.
    ("Number coverage too large", Type.number(), "90071992547409910n", "90071992547409910n"),
    (
        "Object 1",
        Type.object({"x": Type.number(), "y": Type.boolean(), "z": Type.boolean()}),
        {"x": "42", "y": "true", "z": "hello"},
        {"x": 42, "y": True, "z": "hello"},
    ),
    (
        "Object 2",
        Type.object({"x": Type.number(), "y": Type.boolean()}),
        {"x": "42", "y": "true", "z": "hello"},
        {"x": 42, "y": True, "z": "hello"},
    ),
    ("Object 3", Type.object({"x": Type.number()}), {}, {}),
    (
        "Object 4",
        Type.object({"x": Type.number()}, additionalProperties=Type.null()),
        {"x": "1", "y": 0},
        {"x": 1, "y": None},
    ),
    ("Object optional 1", Type.object({"x": Type.optional(Type.number())}), {"x": 1}, {"x": 1}),
    ("Object optional 2", Type.object({"x": Type.optional(Type.number())}), {"x": "1"}, {"x": 1}),
    (
        "Record 1",
        Type.record(Type.string(), Type.number()),
        {"x": "42", "y": "24", "z": "hello"},
        {"x": 42, "y": 24, "z": "hello"},
    ),
    ("Record 930 a", Type.union([Type.null(), Type.record(Type.number(), Type.any())]), {}, {}),
    ("Record 930 b", Type.union([Type.record(Type.number(), Type.any()), Type.null()]), {}, {}),
    ("Record 930 c", Type.union([Type.null(), Type.record(Type.number(), Type.any())]), None, None),
    ("Record 930 d", Type.union([Type.record(Type.number(), Type.any()), Type.null()]), None, None),
    ("Record 930 e", Type.union([Type.null(), Type.record(Type.number(), Type.any())]), "NULL", None),
    ("Record 930 f", Type.union([Type.record(Type.number(), Type.any()), Type.null()]), "NULL", None),
    (
        "Record additional",
        Type.record(Type.number(), Type.number(), additionalProperties=Type.null()),
        {"0": "1", "x": 0},
        {"0": 1, "x": None},
    ),
    ("String 1", Type.string(), 3.14, "3.14"),
    ("String 2", Type.string(), 3, "3"),
    ("String 3", Type.string(), True, "true"),
    ("String 4", Type.string(), False, "false"),
    ("String cast hello", Type.string(), "hello", "hello"),
    ("String cast 42", Type.string(), 42, "42"),
    ("String cast object", Type.string(), {}, {}),
    ("String cast array", Type.string(), [], []),
    ("String coverage null", Type.string(), None, "null"),
    # Differs from TypeBox: a template literal is a string with a pattern in JSON Schema; TypeBox converts a finite one
    # as a union of literals.
    ("TemplateLiteral R0", Type.template_literal("^(?:1|2|3)$"), 1, "1"),
    ("TemplateLiteral R1", Type.template_literal("^(?:1|2|3)$"), 2, "2"),
    ("TemplateLiteral R2", Type.template_literal("^(?:1|2|3)$"), 3, "3"),
    ("TemplateLiteral R3", Type.template_literal("^(?:1|2|3)$"), 4, "4"),
    (
        "TemplateLiteral 1510",
        Type.object({"test": Type.template_literal(f"{_NUMBER_KEY[:-1]}(?:a|b|c)$")}),
        {"test": "123b"},
        {"test": "123b"},
    ),
    ("Tuple 1", Type.tuple([Type.number(), Type.number()]), ["1", "true"], [1, 1]),
    ("Tuple 2", Type.tuple([Type.number(), Type.number()]), ["1"], [1]),
    ("Tuple 3", Type.tuple([Type.number(), Type.number()]), ["1", "2", "3"], [1, 2, "3"]),
    ("Tuple 4", Type.tuple([Type.number(), Type.number()]), "not an array", "not an array"),
    ("Tuple 5", Type.tuple([Type.number(), Type.number()]), 42, 42),
    ("Tuple 6", Type.tuple([Type.number(), Type.number()]), None, None),
    ("Tuple 7", Type.tuple([Type.number(), Type.number()]), {"0": "1", "1": "2"}, {"0": "1", "1": "2"}),
    ("Union R0", Type.object({"x": Type.union([Type.number(), Type.null()])}), {"x": "42"}, {"x": 42}),
    ("Union R1", Type.object({"x": Type.union([Type.number(), Type.null()])}), {"x": "null"}, {"x": None}),
    ("Union R2", Type.object({"x": Type.union([Type.number(), Type.null()])}), {"x": "hello"}, {"x": "hello"}),
    ("Union 2", Type.object({"x": Type.union([Type.boolean(), Type.number()])}), {"x": "1"}, {"x": True}),
    (
        "Union 787 R0",
        Type.intersect(
            [
                Type.union([Type.object({"a": Type.number()}), Type.object({"b": Type.number()})]),
                Type.object({"c": Type.number()}),
            ]
        ),
        {"a": "1", "c": "2"},
        {"a": 1, "c": 2},
    ),
    (
        "Union 787 R1",
        Type.intersect(
            [
                Type.union([Type.object({"a": Type.number()}), Type.object({"b": Type.number()})]),
                Type.object({"c": Type.number()}),
            ]
        ),
        {"b": "1", "c": "2"},
        {"b": 1, "c": 2},
    ),
    (
        # Matching on the first branch.
        "Union 787 R2",
        Type.intersect(
            [
                Type.union([Type.object({"a": Type.number()}), Type.object({"b": Type.number()})]),
                Type.object({"c": Type.number()}),
            ]
        ),
        {"a": "1", "b": "2", "c": "3"},
        {"a": 1, "b": "2", "c": 3},
    ),
    (
        "Union guards against Array conversion in Object",
        Type.union(
            [
                Type.object({"type": Type.literal("A"), "values": Type.union([Type.string(), Type.number()])}),
                Type.object({"type": Type.literal("B"), "values": Type.string()}),
            ]
        ),
        [{"type": "A", "values": 1}],
        [{"type": "A", "values": 1}],
    ),
    (
        "Union guards against Array conversion in Record",
        Type.union(
            [
                Type.record(Type.string(pattern="^values$"), Type.union([Type.string(), Type.number()])),
                Type.record(Type.string(pattern="^type$"), Type.union([Type.string(), Type.number()])),
            ]
        ),
        [{"type": "A", "values": 1}],
        [{"type": "A", "values": 1}],
    ),
    ("Unknown 1", Type.unknown(), None, None),
    ("Unknown 3", Type.unknown(), "hello", "hello"),
    ("Unknown 4", Type.unknown(), 42, 42),
    ("Unknown 5", Type.unknown(), True, True),
    ("Unknown 6", Type.unknown(), {"x": 1}, {"x": 1}),
    ("Unknown 7", Type.unknown(), [1, 2, 3], [1, 2, 3]),
]


@pytest.mark.parametrize(("schema", "value", "expected"), [c[1:] for c in CONVERTS], ids=[c[0] for c in CONVERTS])
def test_value_convert(schema, value, expected):
    result = convert(schema, value)
    assert json_stringify(result) == json_stringify(expected)
    assert result == expected


ARRAY_NUMBER_OR_STRING = Type.array(Type.union([Type.number(), Type.string()]))

CHECKS: list[tuple[str, dict, object, bool]] = [
    ("Array 18", Type.array(Type.array(Type.number())), [1, 2, 3], True),
    ("Array 20", Type.array(Type.array(Type.number())), 42, True),
    ("Array union 1", ARRAY_NUMBER_OR_STRING, 1, True),
    ("Array union 2", ARRAY_NUMBER_OR_STRING, "hello", True),
    ("Array union 3", ARRAY_NUMBER_OR_STRING, [1, "hello", True, "3.14"], True),
    ("Array union 4", Type.union([Type.array(Type.number()), Type.string()]), "1", True),
    ("Array union 5", Type.array(Type.union([Type.array(Type.number()), Type.string()])), 42, True),
    ("Array intersect 1", Type.array(Type.intersect(X_Y)), {"x": 1, "y": 2}, True),
    ("Array intersect 2", Type.array(Type.intersect(X_Y)), [{"x": "1", "y": "2"}, {"x": "3", "y": "4"}], True),
    (
        "Array intersect 3",
        Type.intersect([Type.array(Type.number()), Type.array(Type.number())]),
        42,
        True,
    ),
    (
        "Array intersect 4",
        Type.intersect([Type.array(Type.number()), Type.object({"x": Type.number()})]),
        42,
        False,
    ),
]


@pytest.mark.parametrize(("schema", "value", "valid"), [c[1:] for c in CHECKS], ids=[c[0] for c in CHECKS])
def test_value_convert_then_check(schema, value, valid):
    assert check(schema, convert(schema, value)) is valid
