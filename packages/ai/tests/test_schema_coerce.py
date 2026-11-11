"""Mirror of pi-schema's coerce.test.ts: `compile(schema, coerce=True)`, its
`coerce()` and `parse()`, the three coercion passes, their order where they
conflict, and the guarantees around them.

pi builds the schemas with pi-schema's builders (`obj`, `opt`, `arr`, ...) and
TypeBox; here the builders are local helpers that emit the same JSON, and the
TypeBox schemas are written as the JSON they emit. Not mirrored: the static
type assertions (`expectTypeOf`), frozen schemas, the unavailable Function
constructor (JavaScript-only) and TypeBox's undefined conversion (see "bigint
and undefined").
"""

from pidrei_ai.utils.schema import FORMATS, CoercingValidator, ParseResult, ValidationError, compile


class Optional:
    def __init__(self, schema: dict) -> None:
        self.schema = schema


def opt(schema: dict) -> Optional:
    return Optional(schema)


def obj(properties: dict, *, open: bool = False) -> dict:
    """pi-schema's `obj()`: required unless `opt()`, closed unless `open`."""
    props = {name: p.schema if isinstance(p, Optional) else p for name, p in properties.items()}
    required = [name for name, p in properties.items() if not isinstance(p, Optional)]
    return {
        "type": "object",
        "properties": props,
        **({"required": required} if required else {}),
        **({} if open else {"additionalProperties": False}),
    }


def num(**options) -> dict:
    return {"type": "number", **options}


def int_(**options) -> dict:
    return {"type": "integer", **options}


def str_() -> dict:
    return {"type": "string"}


def arr(items: dict) -> dict:
    return {"type": "array", "items": items}


def nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


def map_(value: dict) -> dict:
    return {"type": "object", "additionalProperties": value}


def tuple_(items: list) -> dict:
    return {"type": "array", "prefixItems": items, "items": False, "minItems": len(items)}


def coerce(schema, value):
    return compile(schema, formats=FORMATS, coerce=True).coerce(value)


def coerce_value(schema, value):
    """The coerced `value` property of `{ value }` under an object schema with one required property `value`."""
    root = {"type": "object", "properties": {"value": schema}, "required": ["value"]}
    return coerce(root, {"value": value})["value"]


def ok(value) -> ParseResult:
    return ParseResult(ok=True, value=value)


# compile options


def test_adds_coerce_and_parse_only_with_coerce_true():
    plain = compile(obj({"n": num()}))
    assert not hasattr(plain, "coerce")
    assert not hasattr(plain, "parse")
    coercing = compile(obj({"n": num()}), coerce=True)
    assert isinstance(coercing, CoercingValidator)
    assert coercing.check({"n": 1}) is True


# coerce


def test_never_modifies_the_value():
    value = {"count": "42", "tags": "a"}
    assert coerce(obj({"count": num(), "tags": arr(str_())}), value) == {"count": 42, "tags": ["a"]}
    assert value == {"count": "42", "tags": "a"}


def test_returns_the_value_itself_when_nothing_changes():
    value = {"a": {"b": [1, 2]}, "c": "x"}
    assert coerce(obj({"a": obj({"b": arr(num())}), "c": str_()}), value) is value


def test_copies_only_the_changed_paths():
    unchanged = {"b": [1, 2]}
    value = {"a": unchanged, "c": {"d": "1"}}
    coerced = coerce(obj({"a": obj({"b": arr(num())}), "c": obj({"d": num()})}), value)
    assert coerced == {"a": {"b": [1, 2]}, "c": {"d": 1}}
    assert coerced is not value
    assert coerced["a"] is unchanged
    assert value["c"] == {"d": "1"}


def test_leaves_values_it_cannot_coerce_for_the_check_to_report():
    validator = compile(obj({"count": num()}), coerce=True)
    coerced = validator.coerce({"count": "many"})
    assert coerced == {"count": "many"}
    assert validator.check(coerced) is False


# parse


def test_parse_returns_the_coerced_value_when_it_is_valid():
    validator = compile(obj({"count": num(), "tags": arr(str_())}), coerce=True)
    assert validator.parse({"count": "42", "tags": "a"}) == ok({"count": 42, "tags": ["a"]})


def test_parse_returns_the_errors_of_the_coerced_value_when_it_is_invalid():
    validator = compile(obj({"count": num(), "name": str_()}), coerce=True)
    assert validator.parse({"count": "many", "name": 7}) == ParseResult(
        ok=False, errors=(ValidationError("/count", "type", "expected number, got string"),)
    )


# validate


def test_validate_checks_without_coercing_with_or_without_the_coerce_option():
    schema = obj({"count": num(), "tags": arr(str_())})
    value = {"count": 42, "tags": ["a"]}
    for validator in (compile(schema), compile(schema, coerce=True)):
        result = validator.validate(value)
        assert result == ok(value)
        assert result.value is value
        # parse() would turn "42" into 42 and "a" into ["a"]; validate() reports both.
        assert validator.validate({"count": "42", "tags": "a"}) == ParseResult(
            ok=False,
            errors=(
                ValidationError("/count", "type", "expected number, got string"),
                ValidationError("/tags", "type", "expected array, got string"),
            ),
        )


# bigint and undefined, as TypeBox's Value.Convert
#
# PiDrei: the bigint is an `int`. TypeBox's TryUndefined is not mirrored: its result is a key present and set to
# undefined, which a Python dict cannot hold.


def test_converts_to_a_bigint_like_typeboxs_try_big_int():
    def big_int(value):
        coerced = coerce_value({"type": "bigint"}, value)
        # `True == 1` and `3.0 == 3` in Python: the type is part of the answer.
        assert type(coerced) is int, coerced
        return coerced

    schema = {"type": "object", "properties": {"n": {"type": "bigint", "minimum": 0}}, "required": ["n"]}
    assert compile(schema, coerce=True).parse({"n": "42"}) == ok({"n": 42})
    assert big_int("12n") == 12
    assert big_int("12.9") == 12
    assert big_int(3.7) == 3
    assert big_int(True) == 1
    assert coerce_value({"type": "bigint"}, "x") == "x"


# pass 1: optional nulls


def test_drops_null_in_optional_non_nullable_properties_also_nested():
    schema = obj(
        {
            "path": str_(),
            "offset": opt(num()),
            "nullable": opt(nullable(str_())),
            "metadata": obj({"enabled": opt({"type": "boolean"})}),
        },
        open=True,
    )
    assert coerce(schema, {"path": "file.txt", "offset": None, "nullable": None, "metadata": {"enabled": None}}) == {
        "path": "file.txt",
        "nullable": None,
        "metadata": {},
    }


def test_keeps_a_null_whose_property_is_a_ref():
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/value"}},
        "$defs": {"value": {"anyOf": [{"type": "number"}, {"type": "null"}]}},
    }
    assert coerce(schema, {"value": None}) == {"value": None}


def test_walks_array_items():
    schema = {"type": "array", "items": obj({"a": str_(), "b": opt(int_())})}
    assert coerce(schema, [{"a": "x", "b": None}]) == [{"a": "x"}]


# pass 2: TypeBox's conversion rules on JSON Schema


def test_converts_primitives_by_type():
    assert coerce_value({"type": "number"}, "42") == 42
    assert coerce_value({"type": "number"}, "") == 0
    assert coerce_value({"type": "number"}, "true") == 1
    assert coerce_value({"type": "integer"}, "3.5") == 3
    assert coerce_value({"type": "integer"}, 1.5) == 1
    assert coerce_value({"type": "boolean"}, "1") is True
    assert coerce_value({"type": "boolean"}, "FALSE") is False
    assert coerce_value({"type": "null"}, "null") is None
    assert coerce_value({"type": "string"}, 42) == "42"


def test_wraps_a_single_value_into_an_array_and_converts_the_items():
    assert coerce_value({"type": "array", "items": {"type": "number"}}, "1") == [1]
    assert coerce_value({"type": "array", "items": {"type": "string"}}, None) == ["null"]


def test_converts_tuple_items_in_place_and_never_wraps_for_a_tuple():
    schema = {"type": "array", "prefixItems": [{"type": "number"}, {"type": "boolean"}], "items": False}
    assert coerce_value(schema, ["1", "true"]) == [1, True]
    assert coerce_value(schema, "1") == "1"


def test_converts_const_and_enum_values_only_when_the_result_is_a_member():
    assert coerce_value({"const": 1}, "1") == 1
    assert coerce_value({"const": 1}, "2") == "2"
    assert coerce_value({"enum": [1, 2, "hello"]}, "2") == 2
    assert coerce_value({"enum": [1, 2, "hello"]}, "world") == "world"


def test_keeps_a_value_one_union_branch_accepts_else_takes_the_first_branch_whose_conversion_the_union_accepts():
    union = {"anyOf": [{"type": "number"}, {"type": "null"}]}
    assert coerce_value(union, None) is None
    assert coerce_value(union, "42") == 42
    assert coerce_value(union, "null") is None
    assert coerce_value({"oneOf": [{"type": "boolean"}, {"type": "number"}]}, "1") is True


def test_converts_all_of_with_each_member_in_turn():
    schema = {"allOf": [obj({"a": num()}, open=True), obj({"b": {"type": "boolean"}}, open=True)]}
    assert coerce(schema, {"a": "1", "b": "true"}) == {"a": 1, "b": True}


def test_follows_ref():
    schema = {
        "type": "object",
        "properties": {"item": {"$ref": "#/$defs/item"}},
        "$defs": {"item": {"type": "object", "properties": {"id": {"type": "integer"}}}},
    }
    assert coerce(schema, {"item": {"id": "7"}}) == {"item": {"id": 7}}


def test_converts_pattern_properties_and_additional_properties():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}},
        "patternProperties": {"^n_": {"type": "number"}},
        "additionalProperties": {"type": "boolean"},
    }
    assert coerce(schema, {"a": "1", "n_x": "2", "other": "true"}) == {"a": 1, "n_x": 2, "other": True}


# pass 3: primitive coercion by type


def test_coerces_like_ajvs_coerce_types():
    cases = [
        ({"type": "number"}, True, 1),
        ({"type": "number"}, None, 0),
        ({"type": "integer"}, "42", 42),
        ({"type": "boolean"}, "true", True),
        ({"type": "boolean"}, 0, False),
        ({"type": "string"}, True, "true"),
        ({"type": "null"}, "", None),
        ({"type": "null"}, 0, None),
        ({"type": "null"}, False, None),
    ]
    for schema, value, expected in cases:
        assert coerce_value(schema, value) == expected


def test_leaves_a_value_that_matches_one_of_a_type_arrays_types_else_takes_the_first_coercible_type():
    assert coerce_value({"type": ["number", "string"]}, "1") == "1"
    assert coerce_value({"type": ["boolean", "number"]}, "1") == 1
    assert coerce_value({"type": ["integer", "null"]}, "5") == 5


def test_accepts_null_for_nullable_array_schemas_with_items():
    assert coerce_value({"type": ["array", "null"], "items": {"type": "string"}}, None) is None


def test_coerces_nullable_unions_when_the_value_matches_no_branch():
    assert coerce_value({"anyOf": [{"type": "number"}, {"type": "null"}]}, "42") == 42
    assert coerce_value({"oneOf": [{"type": "number"}, {"type": "null"}]}, None) is None


def test_leaves_values_no_rule_coerces():
    validator = compile(
        {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}, coerce=True
    )
    for value in ("abc", "4x", {}, ["1", "2"]):
        assert validator.check(validator.coerce({"value": value})) is False


# where the passes conflict, the conversion rules win


def test_turns_null_into_null_for_a_required_string():
    assert coerce_value({"type": "string"}, None) == "null"


def test_turns_empty_string_into_0_for_a_number():
    assert coerce_value({"type": "number"}, "") == 0


def test_turns_1_and_0_into_booleans():
    assert coerce_value({"type": "boolean"}, "0") is False


def test_truncates_42_1_for_an_integer():
    assert coerce_value({"type": "integer"}, "42.1") == 42


def test_drops_an_optional_null_before_converting_it():
    assert coerce(obj({"name": opt(str_())}), {"name": None}) == {}


# recursive schemas

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


def test_converts_through_a_recursive_union_by_its_first_accepting_branch():
    validator = compile(TREE, coerce=True)
    assert validator.parse({"t": ["a", ["b"]]}) == ok({"t": ["a", ["b"]]})
    assert validator.parse({"t": 5}) == ok({"t": "5"})
    assert validator.parse({"t": [5]}) == ok({"t": ["5"]})
    assert compile(JSON_VALUE, coerce=True).parse({"data": {"a": [1, None]}}) == ok({"data": {"a": [1, "null"]}})


def test_leaves_a_value_a_recursive_schema_cannot_convert_as_it_is():
    validator = compile(TREE, coerce=True)
    assert validator.parse({"t": {}}).ok is False
    assert validator.coerce({"t": {"a": 1}}) == {"t": {"a": 1}}


def test_converts_a_scalar_or_array_schema_that_refers_to_itself():
    schema = {
        "type": "object",
        "properties": {
            "value": {"anyOf": [{"type": "number"}, {"type": "array", "items": {"$ref": "#/properties/value"}}]}
        },
        "required": ["value"],
    }
    validator = compile(schema, coerce=True)
    assert validator.parse({"value": "42"}) == ok({"value": 42})
    assert validator.parse({"value": ["1", ["2"]]}) == ok({"value": [1, [2]]})
    assert validator.parse({"value": {}}).ok is False


def test_stops_at_a_schema_that_cycles_through_shared_objects_instead_of_ref():
    node: dict = {"anyOf": [{"type": "number"}]}
    node["anyOf"].append({"type": "array", "items": node})
    validator = compile(node, coerce=True)
    assert validator.parse("42") == ok(42)
    assert validator.parse(["1", ["2"]]) == ok([1, [2]])
    assert validator.parse({}).ok is False


def test_resets_the_reentry_state_between_calls_and_leaves_the_input_unchanged():
    validator = compile({"anyOf": [{"type": "number"}, {"type": "array", "items": {"$ref": "#"}}]}, coerce=True)
    for _ in range(20):
        assert validator.parse({}).ok is False
        assert validator.parse("42") == ok(42)
        value = [1, "2", ["3"]]
        assert validator.parse(value) == ok([1, 2, [3]])
        assert value == [1, "2", ["3"]]


def test_converts_and_explains_a_deep_value_every_branch_of_a_recursive_union_rejects_in_linear_time():
    # Each object branch converts and explains every child: without memoizing per call this is 2^depth work.
    s = {
        "anyOf": [
            {"type": "object", "additionalProperties": {"$ref": "#/$defs/S"}, "required": ["a"]},
            {"type": "object", "additionalProperties": {"$ref": "#/$defs/S"}, "required": ["b"]},
        ]
    }
    validator = compile({"type": "object", "properties": {"s": {"$ref": "#/$defs/S"}}, "$defs": {"S": s}}, coerce=True)
    value: dict = {"x": 1}
    for _ in range(40):
        value = {"c": value}
    missing = tuple(ValidationError(f"/s{'/c' * i}/a", "required", "missing") for i in range(41))
    leaf = ValidationError(f"/s{'/c' * 40}/x", "type", "expected object, got integer")
    assert validator.parse({"s": value}) == ParseResult(ok=False, errors=(*missing, leaf))


def test_converts_a_typebox_cyclic_union():
    # Type.Object({ t: Type.Cyclic({ Tree: Type.Union([Type.String(), Type.Array(Type.Ref("Tree"))]) }, "Tree") })
    tree = {
        "$defs": {"Tree": {"$id": "Tree", "anyOf": [{"type": "string"}, {"type": "array", "items": {"$ref": "Tree"}}]}},
        "$ref": "Tree",
    }
    validator = compile({"type": "object", "properties": {"t": tree}, "required": ["t"]}, coerce=True)
    assert validator.parse({"t": [5]}) == ok({"t": ["5"]})
    assert validator.parse({"t": {"a": 1}}).ok is False


# member keywords across the passes


def test_keeps_a_pattern_matched_property_from_additional_properties_in_every_pass():
    # Type.Record(Type.String(), Type.Number(), { additionalProperties: Type.String() })
    record = {
        "type": "object",
        "patternProperties": {"^.*$": {"type": "number"}},
        "additionalProperties": {"type": "string"},
    }
    assert compile(record, coerce=True).parse({"n1": "1"}) == ok({"n1": 1})
    plain = {
        "type": "object",
        "patternProperties": {"^n_": {"type": ["integer", "null"]}},
        "additionalProperties": {"type": "string"},
    }
    assert coerce(plain, {"n_a": "5", "other": 7}) == {"n_a": 5, "other": "7"}


def test_drops_optional_nulls_inside_declared_pattern_and_additional_property_values_alike():
    item = obj({"n": opt(int_(minimum=1))})
    schemas = [
        obj({"entry": item}),
        map_(item),
        {"type": "object", "patternProperties": {"^entry$": item}, "additionalProperties": False},
    ]
    for schema in schemas:
        value = {"entry": {"n": None}}
        assert compile(schema, coerce=True).parse(value) == ok({"entry": {}})
        assert value == {"entry": {"n": None}}


def test_keeps_a_null_under_pattern_and_additional_properties():
    schema = {"type": "object", "additionalProperties": {"type": ["integer", "null"]}}
    assert compile(schema, coerce=True).parse({"n": None}) == ok({"n": None})


def test_drops_optional_nulls_in_tuple_items_as_in_array_items():
    item = obj({"n": opt(int_())})
    assert compile(arr(item), coerce=True).coerce([{"n": None}]) == [{}]
    assert compile(tuple_([item]), coerce=True).coerce([{"n": None}]) == [{}]


def test_coerces_tuple_items_by_type_in_pass_3():
    schema = {"type": "array", "prefixItems": [{"type": ["integer", "null"]}, {"type": ["boolean", "null"]}]}
    assert coerce(schema, ["5", "true"]) == [5, True]


def test_converts_the_properties_a_union_node_declares_besides_its_branches():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "anyOf": [{"required": ["a"]}, {"required": ["b"]}],
    }
    assert compile(schema, coerce=True).parse({"a": "3.5"}) == ok({"a": 3})


def test_checks_union_candidates_after_the_nodes_own_properties_convert():
    for keyword in ("anyOf", "oneOf"):
        schema = {
            "type": "object",
            keyword: [
                {
                    "type": "object",
                    "properties": {"a": {"type": "array", "items": {"type": "integer"}}},
                    "required": ["a"],
                },
                {"type": "object", "properties": {"b": {"type": "boolean"}}, "required": ["b"]},
            ],
            "properties": {"c": {"type": "integer"}},
            "required": ["c"],
        }
        value = {"a": "3", "c": "2"}
        assert compile(schema, coerce=True).parse(value) == ok({"a": [3], "c": 2})
        assert value == {"a": "3", "c": "2"}
