"""Mirror of pi's validation-probes.test.ts: the reviewers' probes of the schema
switch as regression tests, through `validate_tool_arguments`. Each case's
outcome is "ok <coerced arguments>", "REJECT <first error line>" or
"THROW <message>".

pi builds part of the schemas with TypeBox; here they are written as the JSON
Schema TypeBox emits (`Type.Object` lists its non-optional properties in
`required`, `Type.Cyclic` gives each definition its name as `$id`,
`Type.Record(Type.String(), V)` is `patternProperties: {"^.*$": V}`), since
only that JSON reaches validation. The arguments and the outcomes are written
with `json_stringify`, JavaScript's `JSON.stringify`, so the keys and the
expected strings are pi's.

Where the outcome differs from pi's: patterns run on Python's `re`, which
rejects `\\p{L}` and words its errors its own way (`pattern-unicode-prop`,
`mcp-invalid-regex`). JavaScript-only cases are kept with their JSON
arguments: `prototype-key` and `proto-key-value` are ordinary keys here.
"""

import json

import pytest

from pidrei_ai.types import Tool, ToolCall
from pidrei_ai.utils.schema.checks import json_stringify
from pidrei_ai.utils.validation import validate_tool_arguments


def outcome(name: str, schema: dict, args) -> str:
    tool = Tool(name=name, description="", parameters=schema)
    call = ToolCall(id="1", name=name, arguments=args)
    try:
        return f"ok {json_stringify(validate_tool_arguments(tool, call))}"
    except Exception as error:
        message = str(error)
        if message.startswith("Validation failed"):
            return f"REJECT {message.split(chr(10))[1].strip()}"
        return f"THROW {message.split(chr(10))[0]}"


def obj(properties: dict, *, optional: tuple[str, ...] = (), **extra) -> dict:
    """`Type.Object(properties)`: every property not in `optional` is required."""
    required = [key for key in properties if key not in optional]
    return {"type": "object", "properties": properties, **({"required": required} if required else {}), **extra}


def o(properties: dict, **extra) -> dict:
    return {"type": "object", "properties": properties, **extra}


NODE = {
    "$defs": {
        "Node": {
            "$id": "Node",
            **obj({"name": {"type": "string"}, "children": {"type": "array", "items": {"$ref": "Node"}}}),
        }
    },
    "$ref": "Node",
}

CASES: list[tuple[str, dict, list]] = [
    (
        "cyclic",
        obj({"tree": NODE}),
        [{"tree": {"name": "a", "children": [{"name": "b", "children": []}]}}, {"tree": {"name": 1, "children": []}}],
    ),
    ("ref-with-$id", obj({"a": {"$ref": "A"}}), [{"a": {"x": "1"}}]),
    (
        "function",
        obj({"f": {"type": "function", "parameters": [], "returnType": {"type": "void"}}}, optional=("f",)),
        [{}, {"f": "x"}],
    ),
    ("codec", obj({"c": {"type": "string"}}), [{"c": "12"}, {"c": 12}]),
    ("undefined-optional", obj({"u": {"type": "undefined"}}, optional=("u",)), [{}, {"u": None}]),
    ("bigint", obj({"n": {"type": "bigint"}}), [{"n": 1}, {"n": "12"}]),
    ("refine", obj({"n": {"type": "number"}}), [{"n": 5}, {"n": -5}]),
    (
        "template-literal",
        obj({"t": {"type": "string", "pattern": "^id--?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?$"}}),
        [{"t": "id-12"}, {"t": "id-x"}, {"t": 3}],
    ),
    (
        "record-string",
        obj({"r": {"type": "object", "patternProperties": {"^.*$": {"type": "number"}}}}),
        [{"r": {"a": "1"}}, {"r": {"a": 1, "b": "x"}}],
    ),
    (
        "record-literal-keys",
        obj({"r": obj({"a": {"type": "number"}, "b": {"type": "number"}})}),
        [{"r": {"a": 1, "b": 2}}, {"r": {"a": 1}}, {"r": {"a": 1, "b": 2, "c": 3}}],
    ),
    (
        "intersect",
        {"allOf": [obj({"a": {"type": "number"}}), obj({"b": {"type": "boolean"}})]},
        [{"a": "1", "b": "true"}, {"a": 1, "b": True, "c": 1}],
    ),
    (
        "union-objects",
        obj(
            {
                "v": {
                    "anyOf": [
                        obj({"k": {"const": "x"}, "n": {"type": "number"}}),
                        obj({"k": {"const": "y"}, "s": {"type": "string"}}),
                    ]
                }
            }
        ),
        [{"v": {"k": "x", "n": "2"}}, {"v": {"k": "y", "s": 2}}],
    ),
    ("minLength-astral", obj({"s": {"type": "string", "minLength": 2}}), [{"s": "😀"}]),
    ("maxLength-astral", obj({"s": {"type": "string", "maxLength": 1}}), [{"s": "😀"}]),
    ("pattern-astral", obj({"s": {"type": "string", "pattern": "^.$"}}), [{"s": "😀"}]),
    ("pattern-class-dash", obj({"s": {"type": "string", "pattern": "^[\\w-]+$"}}), [{"s": "a-b"}]),
    ("pattern-unicode-prop", obj({"s": {"type": "string", "pattern": "^\\p{L}+$"}}), [{"s": "äb"}, {"s": "p{L}"}]),
    ("multipleOf-float", obj({"n": {"type": "number", "multipleOf": 0.01}}), [{"n": 0.07}, {"n": 1.1}]),
    (
        "uniqueItems-keyorder",
        obj({"a": {"type": "array", "items": {}, "uniqueItems": True}}),
        [{"a": [{"x": 1, "y": 2}, {"y": 2, "x": 1}]}, {"a": [1, 1.0]}],
    ),
    ("const-object", obj({"c": {"const": "a"}}), [{"c": "a"}]),
    (
        "formats",
        obj(
            {
                "e": {"type": "string", "format": "email"},
                "u": {"type": "string", "format": "uri"},
                "d": {"type": "string", "format": "date-time"},
                "x": {"type": "string", "format": "made-up"},
                "h": {"type": "string", "format": "hostname"},
                "r": {"type": "string", "format": "regex"},
                "i": {"type": "string", "format": "uuid"},
                "ip": {"type": "string", "format": "ipv4"},
            },
            optional=("e", "u", "d", "x", "h", "r", "i", "ip"),
        ),
        [
            {"e": "a@b.co"},
            {"e": "not-an-email"},
            {"u": "relative/path"},
            {"u": "https://x.y/z"},
            {"d": "2024-01-01T00:00:00Z"},
            {"d": "2024-01-01"},
            {"x": "anything"},
            {"h": "under_score.example"},
            {"r": "["},
            {"i": "00000000-0000-0000-0000-000000000000"},
            {"i": "not"},
            {"ip": "999.1.1.1"},
        ],
    ),
    ("integer-big", obj({"n": {"type": "integer"}}), [{"n": 1e21}, {"n": 2**53 + 2}]),
    ("additional-false", obj({"a": {"type": "string"}}, additionalProperties=False), [{"a": "x", "b": 1}]),
    (
        "prototype-key",
        obj({"constructor": {"type": "string"}, "toString": {"type": "number"}}, optional=("constructor", "toString")),
        [{}, {"constructor": 1}],
    ),
    (
        "proto-key-value",
        obj({}, additionalProperties={"type": "array", "items": {"type": "string"}}),
        [json.loads('{"__proto__": "x", "a": "y"}')],
    ),
    # Plain JSON (MCP-shaped)
    (
        "mcp-draft7-ref-definitions",
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"t": {"$ref": "#/definitions/T"}},
            "required": ["t"],
            "definitions": {"T": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        },
        [{"t": {"path": "a"}}, {"t": {"path": 42}}],
    ),
    (
        "mcp-ref-unresolved",
        {"type": "object", "properties": {"t": {"$ref": "#/definitions/Missing"}}},
        [{"t": 1}, {}],
    ),
    (
        "mcp-ref-external",
        {"type": "object", "properties": {"t": {"$ref": "https://example.com/schemas/t.json"}}},
        [{"t": 1}],
    ),
    (
        "mcp-nullable-openapi",
        {"type": "object", "properties": {"s": {"type": "string", "nullable": True}}, "required": ["s"]},
        [{"s": None}, {"s": "x"}],
    ),
    (
        "mcp-draft7-tuple",
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "p": {"type": "array", "items": [{"type": "number"}, {"type": "string"}], "additionalItems": False}
            },
        },
        [{"p": [1, "a"]}, {"p": [1, "a", 3]}, {"p": ["1", 2]}],
    ),
    (
        "mcp-draft7-ref-sibling",
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"s": {"$ref": "#/definitions/S", "maxLength": 1}},
            "definitions": {"S": {"type": "string"}},
        },
        [{"s": "abc"}],
    ),
    (
        "mcp-draft4-exclusive-bool",
        {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "type": "object",
            "properties": {"n": {"type": "number", "minimum": 0, "exclusiveMinimum": True}},
        },
        [{"n": 0}, {"n": 1}],
    ),
    (
        "mcp-id-draft4",
        {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "id": "http://x/y",
            "type": "object",
            "properties": {"a": {"type": "string"}},
        },
        [{"a": "x"}],
    ),
    (
        "mcp-bad-pattern",
        {"type": "object", "properties": {"s": {"type": "string", "pattern": "(?<!x)y"}}},
        [{"s": "y"}],
    ),
    ("mcp-invalid-regex", {"type": "object", "properties": {"s": {"type": "string", "pattern": "["}}}, [{"s": "y"}]),
    (
        "mcp-type-integer-float-0",
        {"type": "object", "properties": {"n": {"type": "integer"}}},
        [{"n": 1.0}, {"n": 1.5}],
    ),
    (
        "mcp-format-unknown",
        {"type": "object", "properties": {"s": {"type": "string", "format": "binary"}}},
        [{"s": "xx"}],
    ),
    (
        "mcp-required-null-string",
        {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        [{"q": None}],
    ),
    (
        "mcp-array-json-string",
        {"type": "object", "properties": {"p": {"type": "array", "items": {"type": "string"}}}, "required": ["p"]},
        [{"p": '["a","b"]'}],
    ),
    (
        "mcp-integer-float",
        {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
        [{"n": 3.7}, {"n": "3.7"}],
    ),
    ("mcp-empty-schema", {}, [{"a": 1}]),
    ("mcp-type-object-no-props", {"type": "object"}, [{"a": 1}]),
    ("mcp-boolean-true-prop", {"type": "object", "properties": {"a": True, "b": False}}, [{"a": 1}, {"b": 1}]),
    (
        "mcp-huge-enum",
        {"type": "object", "properties": {"e": {"type": "string", "enum": [f"v{i}" for i in range(5000)]}}},
        [{"e": "v4999"}, {"e": "v5000"}],
    ),
]

KEYWORD_CASES: list[tuple[str, dict, list]] = [
    (
        "oneOf-overlap",
        o({"v": {"oneOf": [{"type": "string"}, {"type": "string", "maxLength": 5}]}}),
        [{"v": "abc"}, {"v": "abcdefgh"}],
    ),
    (
        "oneOf-number-integer",
        o({"v": {"oneOf": [{"type": "number"}, {"type": "integer"}]}}),
        [{"v": 2}, {"v": 2.5}],
    ),
    (
        "oneOf-objects-open",
        o(
            {
                "v": {
                    "oneOf": [
                        {"type": "object", "properties": {"a": {"type": "string"}}},
                        {"type": "object", "properties": {"b": {"type": "string"}}},
                    ]
                }
            }
        ),
        [{"v": {"a": "x"}}, {"v": {}}],
    ),
    (
        "if-then-else",
        o(
            {"kind": {"type": "string"}, "n": {"type": "number"}},
            **{"if": {"properties": {"kind": {"const": "a"}}}, "then": {"required": ["n"]}},
        ),
        [{"kind": "a"}, {"kind": "b"}],
    ),
    (
        "dependentRequired",
        o({"a": {"type": "string"}, "b": {"type": "string"}}, dependentRequired={"a": ["b"]}),
        [{"a": "x"}],
    ),
    (
        "dependencies-d7",
        o({"a": {"type": "string"}, "b": {"type": "string"}}, dependencies={"a": ["b"]}),
        [{"a": "x"}],
    ),
    ("propertyNames", o({}, propertyNames={"pattern": "^[a-z]+$"}), [{"abc": 1}, {"ABC": 1}]),
    ("contains", o({"l": {"type": "array", "contains": {"type": "number"}}}), [{"l": ["a"]}, {"l": ["a", 1]}]),
    ("not", o({"s": {"not": {"type": "null"}}}), [{"s": "x"}, {"s": None}]),
    ("minProperties", o({"m": {"type": "object", "minProperties": 1}}), [{"m": {}}]),
    (
        "unevaluatedProperties",
        o({"a": {"type": "string"}}, unevaluatedProperties=False),
        [{"a": "x"}, {"a": "x", "b": 1}],
    ),
    (
        "allOf-additional-false",
        {"allOf": [o({"a": {"type": "string"}}, additionalProperties=False), o({"b": {"type": "string"}})]},
        [{"a": "x"}, {"a": "x", "b": "y"}],
    ),
    ("const-object", o({"c": {"const": {"x": 1, "y": 2}}}), [{"c": {"y": 2, "x": 1}}]),
    ("enum-object", o({"e": {"enum": [{"x": 1, "y": 2}]}}), [{"e": {"y": 2, "x": 1}}]),
    ("type-array-multi", o({"v": {"type": ["string", "number"]}}), [{"v": True}, {"v": None}]),
    ("nullable-type-array", o({"v": {"type": ["string", "null"]}}, required=["v"]), [{"v": None}, {"v": 5}]),
    ("exclusiveMinimum-number", o({"n": {"type": "number", "exclusiveMinimum": 0}}), [{"n": 0}, {"n": "0"}]),
    ("maxItems", o({"l": {"type": "array", "maxItems": 1}}), [{"l": [1, 2]}, {"l": "x"}]),
    (
        "discriminator-openapi",
        o(
            {
                "p": {
                    "oneOf": [
                        o({"kind": {"type": "string", "enum": ["a"]}, "x": {"type": "number"}}, required=["kind"]),
                        o({"kind": {"type": "string", "enum": ["b"]}}, required=["kind"]),
                    ],
                    "discriminator": {"propertyName": "kind"},
                }
            }
        ),
        [{"p": {"kind": "a", "x": "1"}}, {"p": {"kind": "b"}}],
    ),
    (
        "ref-root-defs",
        {"$ref": "#/$defs/Root", "$defs": {"Root": o({"a": {"type": "string"}}, required=["a"])}},
        [{"a": "x"}, {"a": 1}, {}],
    ),
    (
        "ref-recursive",
        {
            "$defs": {"N": o({"c": {"type": "array", "items": {"$ref": "#/$defs/N"}}, "v": {"type": "integer"}})},
            **o({"root": {"$ref": "#/$defs/N"}}),
        },
        [{"root": {"v": "2", "c": [{"v": "3", "c": []}]}}],
    ),
    (
        "ref-percent",
        {**o({"a": {"$ref": "#/$defs/a%20b"}}), "$defs": {"a b": {"type": "string"}}},
        [{"a": "x"}, {"a": 1}],
    ),
    ("ref-tilde", {**o({"a": {"$ref": "#/$defs/a~1b"}}), "$defs": {"a/b": {"type": "string"}}}, [{"a": "x"}]),
    (
        "defs-with-$id",
        {**o({"a": {"$ref": "item.json"}}), "$defs": {"item": {"$id": "item.json", "type": "string"}}},
        [{"a": "x"}, {"a": 2}],
    ),
    (
        "anyOf-null-default",
        o({"v": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}}),
        [{"v": None}, {}],
    ),
    (
        "format-date-time-lower",
        o({"d": {"type": "string", "format": "date-time"}}),
        [{"d": "2024-01-01t00:00:00z"}, {"d": "2024-01-01 00:00:00Z"}],
    ),
    ("format-url", o({"u": {"type": "string", "format": "url"}}), [{"u": "not a url"}]),
    ("integer-as-float-1e0", o({"n": {"type": "integer"}}), [{"n": 1e3}]),
    ("required-missing-props", {"type": "object", "required": ["a"]}, [{}, {"a": None}]),
    (
        "string-to-number-trim",
        o({"n": {"type": "number"}}),
        [{"n": " 12 "}, {"n": "0x10"}, {"n": "1e3"}, {"n": ""}],
    ),
]

TREE = {
    "type": "object",
    "properties": {"t": {"$ref": "#/$defs/Tree"}},
    "$defs": {"Tree": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"$ref": "#/$defs/Tree"}}]}},
}
JSON_VALUE = {
    "type": "object",
    "properties": {"data": {"$ref": "#/$defs/Json"}},
    "required": ["data"],
    "$defs": {
        "Json": {
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "boolean"},
                {"type": "array", "items": {"$ref": "#/$defs/Json"}},
                {"type": "object", "additionalProperties": {"$ref": "#/$defs/Json"}},
            ]
        }
    },
}
FILTER = {
    "type": "object",
    "properties": {"where": {"$ref": "#/$defs/F"}},
    "$defs": {
        "F": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {"field": {"type": "string"}, "eq": {"type": "string"}},
                    "required": ["field", "eq"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {"and": {"type": "array", "items": {"$ref": "#/$defs/F"}}},
                    "required": ["and"],
                    "additionalProperties": False,
                },
            ]
        }
    },
}
RECURSION_CASES: list[tuple[str, dict, dict]] = [
    ("tree valid", TREE, {"t": ["a", ["b"]]}),
    ("tree number", TREE, {"t": 5}),
    ("tree [5]", TREE, {"t": [5]}),
    ("json valid", JSON_VALUE, {"data": {"a": [1, "x", {"b": True}]}}),
    ("json null", JSON_VALUE, {"data": None}),
    ("json nested null", JSON_VALUE, {"data": {"a": [1, None]}}),
    ("filter valid", FILTER, {"where": {"and": [{"field": "a", "eq": "b"}]}}),
    ("filter bad leaf", FILTER, {"where": {"and": [{"field": "a", "eq": 1, "x": 2}]}}),
    (
        "filter deep bad",
        FILTER,
        {"where": {"and": [{"and": [{"and": [{"and": [{"and": [{"field": 1}]}]}]}]}]}},
    ),
]

EXPECTED: dict[str, str] = {
    'cyclic {"tree":{"name":"a","children":[{"name":"b","children":[]}]}}': (
        'ok {"tree":{"name":"a","children":[{"name":"b","children":[]}]}}'
    ),
    'cyclic {"tree":{"name":1,"children":[]}}': 'ok {"tree":{"name":"1","children":[]}}',
    'ref-with-$id {"a":{"x":"1"}}': "REJECT - a: unresolved reference A",
    "function {}": "ok {}",
    'function {"f":"x"}': "REJECT - f: expected function, got string",
    'codec {"c":"12"}': 'ok {"c":"12"}',
    'codec {"c":12}': 'ok {"c":"12"}',
    "undefined-optional {}": "ok {}",
    'undefined-optional {"u":null}': "ok {}",
    'bigint {"n":1}': "REJECT - n: expected bigint, got integer",
    'bigint {"n":"12"}': "REJECT - n: expected bigint, got string",
    'refine {"n":5}': 'ok {"n":5}',
    'refine {"n":-5}': 'ok {"n":-5}',
    'template-literal {"t":"id-12"}': 'ok {"t":"id-12"}',
    'template-literal {"t":"id-x"}': "REJECT - t: expected to match ^id--?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?$",
    'template-literal {"t":3}': "REJECT - t: expected to match ^id--?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?$",
    'record-string {"r":{"a":"1"}}': 'ok {"r":{"a":1}}',
    'record-string {"r":{"a":1,"b":"x"}}': "REJECT - r.b: expected number, got string",
    'record-literal-keys {"r":{"a":1,"b":2}}': 'ok {"r":{"a":1,"b":2}}',
    'record-literal-keys {"r":{"a":1}}': "REJECT - r.b: missing",
    'record-literal-keys {"r":{"a":1,"b":2,"c":3}}': 'ok {"r":{"a":1,"b":2,"c":3}}',
    'intersect {"a":"1","b":"true"}': 'ok {"a":1,"b":true}',
    'intersect {"a":1,"b":true,"c":1}': 'ok {"a":1,"b":true,"c":1}',
    'union-objects {"v":{"k":"x","n":"2"}}': 'ok {"v":{"k":"x","n":2}}',
    'union-objects {"v":{"k":"y","s":2}}': 'ok {"v":{"k":"y","s":"2"}}',
    'minLength-astral {"s":"😀"}': "REJECT - s: expected at least 2 characters",
    'maxLength-astral {"s":"😀"}': 'ok {"s":"😀"}',
    'pattern-astral {"s":"😀"}': 'ok {"s":"😀"}',
    'pattern-class-dash {"s":"a-b"}': 'ok {"s":"a-b"}',
    # pi: ok, and "REJECT - s: expected to match ^\\p{L}+$": ECMAScript has \p{L}, Python's re does not.
    'pattern-unicode-prop {"s":"äb"}': "THROW bad escape \\p at position 1",
    'pattern-unicode-prop {"s":"p{L}"}': "THROW bad escape \\p at position 1",
    'multipleOf-float {"n":0.07}': 'ok {"n":0.07}',
    'multipleOf-float {"n":1.1}': 'ok {"n":1.1}',
    'uniqueItems-keyorder {"a":[{"x":1,"y":2},{"y":2,"x":1}]}': "REJECT - a: expected unique items",
    'uniqueItems-keyorder {"a":[1,1]}': "REJECT - a: expected unique items",
    'const-object {"c":"a"}': 'ok {"c":"a"}',
    'formats {"e":"a@b.co"}': 'ok {"e":"a@b.co"}',
    'formats {"e":"not-an-email"}': "REJECT - e: expected format email",
    'formats {"u":"relative/path"}': "REJECT - u: expected format uri",
    'formats {"u":"https://x.y/z"}': 'ok {"u":"https://x.y/z"}',
    'formats {"d":"2024-01-01T00:00:00Z"}': 'ok {"d":"2024-01-01T00:00:00Z"}',
    'formats {"d":"2024-01-01"}': "REJECT - d: expected format date-time",
    'formats {"x":"anything"}': 'ok {"x":"anything"}',
    'formats {"h":"under_score.example"}': "REJECT - h: expected format hostname",
    'formats {"r":"["}': "REJECT - r: expected format regex",
    'formats {"i":"00000000-0000-0000-0000-000000000000"}': 'ok {"i":"00000000-0000-0000-0000-000000000000"}',
    'formats {"i":"not"}': "REJECT - i: expected format uuid",
    'formats {"ip":"999.1.1.1"}': "REJECT - ip: expected format ipv4",
    'integer-big {"n":1e+21}': 'ok {"n":1e+21}',
    'integer-big {"n":9007199254740994}': 'ok {"n":9007199254740994}',
    'additional-false {"a":"x","b":1}': "REJECT - b: unknown field",
    "prototype-key {}": "ok {}",
    'prototype-key {"constructor":1}': 'ok {"constructor":"1"}',
    'proto-key-value {"__proto__":"x","a":"y"}': 'ok {"__proto__":["x"],"a":["y"]}',
    'mcp-draft7-ref-definitions {"t":{"path":"a"}}': 'ok {"t":{"path":"a"}}',
    'mcp-draft7-ref-definitions {"t":{"path":42}}': 'ok {"t":{"path":"42"}}',
    'mcp-ref-unresolved {"t":1}': "REJECT - t: unresolved reference #/definitions/Missing",
    "mcp-ref-unresolved {}": "ok {}",
    'mcp-ref-external {"t":1}': "REJECT - t: unresolved reference https://example.com/schemas/t.json",
    'mcp-nullable-openapi {"s":null}': 'ok {"s":"null"}',
    'mcp-nullable-openapi {"s":"x"}': 'ok {"s":"x"}',
    'mcp-draft7-tuple {"p":[1,"a"]}': 'ok {"p":[1,"a"]}',
    'mcp-draft7-tuple {"p":[1,"a",3]}': "REJECT - p.2: no value is allowed here",
    'mcp-draft7-tuple {"p":["1",2]}': 'ok {"p":[1,"2"]}',
    'mcp-draft7-ref-sibling {"s":"abc"}': "REJECT - s: expected at most 1 characters",
    'mcp-draft4-exclusive-bool {"n":0}': 'ok {"n":0}',
    'mcp-draft4-exclusive-bool {"n":1}': 'ok {"n":1}',
    'mcp-id-draft4 {"a":"x"}': 'ok {"a":"x"}',
    'mcp-bad-pattern {"s":"y"}': 'ok {"s":"y"}',
    # pi: "THROW Invalid regular expression: /[/: Unterminated character class", ECMAScript's wording.
    'mcp-invalid-regex {"s":"y"}': "THROW unterminated character set at position 0",
    'mcp-type-integer-float-0 {"n":1}': 'ok {"n":1}',
    'mcp-type-integer-float-0 {"n":1.5}': 'ok {"n":1}',
    'mcp-format-unknown {"s":"xx"}': 'ok {"s":"xx"}',
    'mcp-required-null-string {"q":null}': 'ok {"q":"null"}',
    'mcp-array-json-string {"p":"[\\"a\\",\\"b\\"]"}': 'ok {"p":["[\\"a\\",\\"b\\"]"]}',
    'mcp-integer-float {"n":3.7}': 'ok {"n":3}',
    'mcp-integer-float {"n":"3.7"}': 'ok {"n":3}',
    'mcp-empty-schema {"a":1}': 'ok {"a":1}',
    'mcp-type-object-no-props {"a":1}': 'ok {"a":1}',
    'mcp-boolean-true-prop {"a":1}': 'ok {"a":1}',
    'mcp-boolean-true-prop {"b":1}': "REJECT - b: no value is allowed here",
    'mcp-huge-enum {"e":"v4999"}': 'ok {"e":"v4999"}',
    'mcp-huge-enum {"e":"v5000"}': (
        'REJECT - e: expected one of ["v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13",'
        '"v14","v15","v16","v17","v18","v19", ... and 4980 more]'
    ),
    'oneOf-overlap {"v":"abc"}': "REJECT - v: matches more than one alternative",
    'oneOf-overlap {"v":"abcdefgh"}': 'ok {"v":"abcdefgh"}',
    'oneOf-number-integer {"v":2}': "REJECT - v: matches more than one alternative",
    'oneOf-number-integer {"v":2.5}': 'ok {"v":2.5}',
    'oneOf-objects-open {"v":{"a":"x"}}': "REJECT - v: matches more than one alternative",
    'oneOf-objects-open {"v":{}}': "REJECT - v: matches more than one alternative",
    'if-then-else {"kind":"a"}': "REJECT - n: missing",
    'if-then-else {"kind":"b"}': 'ok {"kind":"b"}',
    'dependentRequired {"a":"x"}': "REJECT - b: required when a is present",
    'dependencies-d7 {"a":"x"}': "REJECT - b: required when a is present",
    'propertyNames {"abc":1}': 'ok {"abc":1}',
    'propertyNames {"ABC":1}': "REJECT - ABC: invalid property name",
    'contains {"l":["a"]}': "REJECT - l: expected at least 1 matching item",
    'contains {"l":["a",1]}': 'ok {"l":["a",1]}',
    'not {"s":"x"}': 'ok {"s":"x"}',
    'not {"s":null}': "ok {}",
    'minProperties {"m":{}}': "REJECT - m: expected at least 1 properties",
    'unevaluatedProperties {"a":"x"}': 'ok {"a":"x"}',
    'unevaluatedProperties {"a":"x","b":1}': "REJECT - b: unknown field",
    'allOf-additional-false {"a":"x"}': 'ok {"a":"x"}',
    'allOf-additional-false {"a":"x","b":"y"}': "REJECT - b: unknown field",
    'const-object {"c":{"y":2,"x":1}}': 'ok {"c":{"y":2,"x":1}}',
    'enum-object {"e":{"y":2,"x":1}}': 'ok {"e":{"y":2,"x":1}}',
    'type-array-multi {"v":true}': 'ok {"v":"true"}',
    'type-array-multi {"v":null}': "ok {}",
    'nullable-type-array {"v":null}': 'ok {"v":null}',
    'nullable-type-array {"v":5}': 'ok {"v":"5"}',
    'exclusiveMinimum-number {"n":0}': "REJECT - n: expected > 0",
    'exclusiveMinimum-number {"n":"0"}': "REJECT - n: expected > 0",
    'maxItems {"l":[1,2]}': "REJECT - l: expected at most 1 items",
    'maxItems {"l":"x"}': 'ok {"l":["x"]}',
    'discriminator-openapi {"p":{"kind":"a","x":"1"}}': 'ok {"p":{"kind":"a","x":1}}',
    'discriminator-openapi {"p":{"kind":"b"}}': 'ok {"p":{"kind":"b"}}',
    'ref-root-defs {"a":"x"}': 'ok {"a":"x"}',
    'ref-root-defs {"a":1}': 'ok {"a":"1"}',
    "ref-root-defs {}": "REJECT - a: missing",
    'ref-recursive {"root":{"v":"2","c":[{"v":"3","c":[]}]}}': 'ok {"root":{"v":2,"c":[{"v":3,"c":[]}]}}',
    'ref-percent {"a":"x"}': 'ok {"a":"x"}',
    'ref-percent {"a":1}': 'ok {"a":"1"}',
    'ref-tilde {"a":"x"}': 'ok {"a":"x"}',
    'defs-with-$id {"a":"x"}': 'ok {"a":"x"}',
    'defs-with-$id {"a":2}': 'ok {"a":"2"}',
    'anyOf-null-default {"v":null}': 'ok {"v":null}',
    "anyOf-null-default {}": "ok {}",
    'format-date-time-lower {"d":"2024-01-01t00:00:00z"}': 'ok {"d":"2024-01-01t00:00:00z"}',
    'format-date-time-lower {"d":"2024-01-01 00:00:00Z"}': "REJECT - d: expected format date-time",
    'format-url {"u":"not a url"}': "REJECT - u: expected format url",
    'integer-as-float-1e0 {"n":1000}': 'ok {"n":1000}',
    "required-missing-props {}": "REJECT - a: missing",
    'required-missing-props {"a":null}': 'ok {"a":null}',
    'string-to-number-trim {"n":" 12 "}': 'ok {"n":12}',
    'string-to-number-trim {"n":"0x10"}': 'ok {"n":16}',
    'string-to-number-trim {"n":"1e3"}': 'ok {"n":1000}',
    'string-to-number-trim {"n":""}': 'ok {"n":0}',
    "tree valid": 'ok {"t":["a",["b"]]}',
    "tree number": 'ok {"t":"5"}',
    "tree [5]": 'ok {"t":["5"]}',
    "json valid": 'ok {"data":{"a":[1,"x",{"b":true}]}}',
    "json null": 'ok {"data":"null"}',
    "json nested null": 'ok {"data":{"a":[1,"null"]}}',
    "filter valid": 'ok {"where":{"and":[{"field":"a","eq":"b"}]}}',
    "filter bad leaf": "REJECT - where.and.0.eq: expected string, got integer",
    "filter deep bad": "REJECT - where.and.0.and.0.and.0.and.0.and.0.eq: missing",
}

ALL = [
    *((f"{name} {json_stringify(arg)}", name, schema, arg) for name, schema, args in CASES for arg in args),
    *((f"{name} {json_stringify(arg)}", name, schema, arg) for name, schema, args in KEYWORD_CASES for arg in args),
    *((name, name, schema, arg) for name, schema, arg in RECURSION_CASES),
]


@pytest.mark.parametrize(("key", "name", "schema", "arg"), ALL, ids=[case[0] for case in ALL])
def test_reviewer_probes(key, name, schema, arg):
    assert outcome(name, schema, arg) == EXPECTED[key]


def test_every_probe_has_an_expected_outcome():
    assert sorted(EXPECTED) == sorted(case[0] for case in ALL)


SOL_CASES: list[tuple[str, dict, dict, str]] = [
    # TypeBox-only: Type.BigInt and Type.Undefined have no JSON form; TypeBox converted to them.
    (
        "BigInt from a JSON string",
        obj({"n": {"type": "bigint"}}),
        {"n": "42"},
        "REJECT - n: expected bigint, got string",
    ),
    (
        "Undefined from JSON null",
        obj({"n": {"type": "undefined"}}),
        {"n": None},
        "REJECT - n: expected undefined, got null",
    ),
    # TypeBox-only: its additionalProperties also converted declared properties; pi-schema converts only undeclared ones.
    (
        "additionalProperties on declared properties",
        obj({"amount": {"type": "number"}, "label": {"type": "string"}}, additionalProperties={"type": "integer"}),
        {"amount": 3.5, "label": "7"},
        'ok {"amount":3.5,"label":"7"}',
    ),
    # TypeBox-only: property names are names, not regular expressions.
    (
        "property names are not patterns",
        obj({"a.b": {"type": "integer"}, "axb": {"type": "string"}}),
        {"a.b": "4", "axb": "5.8"},
        'ok {"a.b":4,"axb":"5.8"}',
    ),
    # TypeBox-only: Refine runs only in TypeBox's own validator.
    ("Refine is not run", obj({"mode": {"type": "string"}}), {"mode": "unsafe"}, 'ok {"mode":"unsafe"}'),
    (
        "scalar-or-array that refers to itself",
        {
            "type": "object",
            "properties": {
                "value": {"anyOf": [{"type": "number"}, {"type": "array", "items": {"$ref": "#/properties/value"}}]}
            },
            "required": ["value"],
        },
        {"value": "42"},
        'ok {"value":42}',
    ),
    (
        "StringEnum with an empty member from null",
        obj({"mode": {"type": "string", "enum": ["", "x"]}}),
        {"mode": None},
        'ok {"mode":""}',
    ),
    (
        "Cyclic with named local definitions",
        obj(
            {
                "node": {
                    "$defs": {
                        "Node": {
                            "$id": "Node",
                            **obj({"n": {"type": "number"}, "next": {"$ref": "Node"}}, optional=("next",)),
                        }
                    },
                    "$ref": "Node",
                }
            }
        ),
        {"node": {"n": 1}},
        'ok {"node":{"n":1}}',
    ),
    (
        "unpaired high surrogate, minLength 2",
        obj({"s": {"type": "string", "minLength": 2}}),
        {"s": "\ud800A"},
        'ok {"s":"\\ud800A"}',
    ),
    (
        "unpaired high surrogate, maxLength 1",
        obj({"s": {"type": "string", "maxLength": 1}}),
        {"s": "\ud800A"},
        "REJECT - s: expected at most 1 characters",
    ),
    ("integer from a string", obj({"n": {"type": "integer"}}), {"n": "42"}, 'ok {"n":42}'),
    ("recursive union with a value no branch converts", TREE, {"t": {}}, "REJECT - t: expected string, got object"),
]


@pytest.mark.parametrize(("name", "schema", "arg", "result"), SOL_CASES, ids=[case[0] for case in SOL_CASES])
def test_reviewer_probes_sol(name, schema, arg, result):
    assert outcome(name, schema, arg) == result


def test_accepts_the_last_member_of_a_10000_value_enum():
    values = [f"member{i}" for i in range(10_000)]
    schema = {"type": "object", "properties": {"choice": {"type": "string", "enum": values}}, "required": ["choice"]}
    assert outcome("enum", schema, {"choice": "member9999"}) == 'ok {"choice":"member9999"}'
    assert outcome("enum", schema, {"choice": "x"}) == (
        f"REJECT - choice: expected one of {json_stringify(values[:20])[:-1]}, ... and 9980 more]"
    )


def test_keeps_the_validator_of_a_schema_object_also_when_the_schema_is_changed_later():
    parameters = obj({"n": {"type": "number", "minimum": 0}})
    assert outcome("late", parameters, {"n": "4"}) == 'ok {"n":4}'
    parameters["properties"]["n"]["minimum"] = 10
    assert outcome("late", parameters, {"n": "4"}) == 'ok {"n":4}'


ITEM = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}}


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"entry": ITEM}},
        {"type": "object", "additionalProperties": ITEM},
        {"type": "object", "patternProperties": {"^entry$": ITEM}, "additionalProperties": False},
    ],
    ids=["declared", "additional", "pattern"],
)
def test_drops_an_optional_null_inside_property_values(schema):
    assert outcome("nested", schema, {"entry": {"n": None}}) == 'ok {"entry":{}}'


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
def test_converts_a_branch_together_with_the_nodes_sibling_properties(keyword):
    schema = {
        "type": "object",
        keyword: [
            {"type": "object", "properties": {"a": {"type": "array", "items": {"type": "integer"}}}, "required": ["a"]},
            {"type": "object", "properties": {"b": {"type": "boolean"}}, "required": ["b"]},
        ],
        "properties": {"c": {"type": "integer"}},
        "required": ["c"],
    }
    assert outcome(keyword, schema, {"a": "3", "c": "2"}) == 'ok {"a":[3],"c":2}'


def test_rejects_a_deep_value_every_branch_of_a_recursive_object_union_rejects_without_exponential_work():
    s = {
        "anyOf": [
            {"type": "object", "additionalProperties": {"$ref": "#/$defs/S"}, "required": ["a"]},
            {"type": "object", "additionalProperties": {"$ref": "#/$defs/S"}, "required": ["b"]},
        ]
    }
    schema = {"type": "object", "properties": {"s": {"$ref": "#/$defs/S"}}, "$defs": {"S": s}}
    value: dict = {"x": 1}
    for _ in range(40):
        value = {"c": value}
    assert outcome("deep", schema, {"s": value}) == "REJECT - s.a: missing"
