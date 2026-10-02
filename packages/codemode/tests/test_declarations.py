"""Mirror of pi codemode test/declarations.test.ts.

The cases keep pi's structure; every expected string is new, since the
declarations are Python (see the module docstring of `declarations.py`).
"""

from pidrei_codemode import BUILTIN_STUBS, CodemodeGlobal, CodemodeTool, RenderedType
from pidrei_codemode.declarations import (
    mcp_structured_content_schema,
    render_stubs,
    render_tools,
    schema_to_type,
)


async def _execute(*_args):
    return None


def _tool(name: str, **fields) -> CodemodeTool:
    return CodemodeTool(name=name, execute=_execute, **fields)


def _global(name: str, **fields) -> CodemodeGlobal:
    return CodemodeGlobal(name=name, execute=_execute, **fields)


def _mcp_result_schema(structured_content=None):
    properties = {"content": {"type": "array", "items": {"type": "object"}}}
    if structured_content is not None:
        properties["structuredContent"] = structured_content
    properties |= {"isError": {"type": "boolean"}, "_meta": {"type": "object"}}
    return {"type": "object", "properties": properties, "required": ["content"]}


def _annotation(schema, **options) -> str:
    return schema_to_type(schema, **options).annotation


# -- schema_to_type ----------------------------------------------------------


def test_renders_primitives_literals_and_unions():
    assert _annotation({"type": "string"}) == "str"
    assert _annotation({"type": "integer"}) == "int"
    assert _annotation({"type": "number"}) == "float"
    assert _annotation({"type": ["string", "null"]}) == "str | None"
    assert _annotation({"const": "a"}) == "Literal['a']"
    assert _annotation({"enum": ["a", 1, None]}) == "Literal['a', 1, None]"
    assert _annotation({"anyOf": [{"type": "string"}, {"type": "number"}]}) == "str | float"
    assert _annotation({"anyOf": [{"type": "string"}, {}]}) == "Any"
    # Python has no intersection: one renderable part is kept, more become Any.
    assert _annotation({"allOf": [{"type": "string"}, {}]}) == "str"
    assert _annotation({"allOf": [{"anyOf": [{"type": "string"}, {"type": "number"}]}, {"const": 1}]}) == "Any"
    assert _annotation({"$ref": "#/defs/x"}) == "Any"
    assert _annotation(True) == "Any"
    assert _annotation(False) == "Never"


def test_renders_objects_as_typed_dicts_in_schema_order():
    assert schema_to_type(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}, "city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        name="Place",
    ) == RenderedType("Place", ("class Place(TypedDict):\n    name: NotRequired[str]\n    city: str",))
    # Keys that are not identifiers use the functional form.
    assert schema_to_type(
        {"type": "object", "properties": {"city": {"type": "string"}, "max-lines": {"type": "number"}}},
        name="Query",
    ).definitions == ("Query = TypedDict('Query', {'city': NotRequired[str], 'max-lines': NotRequired[float]})",)
    assert _annotation({"type": "object", "additionalProperties": {"type": "number"}}) == "dict[str, float]"
    assert _annotation({"type": "object"}) == "dict[str, Any]"
    assert _annotation({"type": "object", "properties": {}, "additionalProperties": False}) == "dict[str, Any]"
    # Declared properties plus extra keys cannot be one TypedDict.
    assert (
        _annotation({"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": True})
        == "dict[str, Any]"
    )


def test_puts_property_descriptions_on_comment_lines():
    assert schema_to_type(
        {
            "type": "object",
            "properties": {
                "weather": {
                    "type": "array",
                    "description": "look up weather for a given list of locations",
                    "items": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            },
            "required": ["weather"],
        }
    ) == RenderedType(
        "Type",
        (
            "class TypeWeatherItem(TypedDict):\n    location: str",
            "class Type(TypedDict):\n    # look up weather for a given list of locations\n    weather: list[TypeWeatherItem]",
        ),
    )
    assert schema_to_type(
        {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "description": "Outer\nsecond line",
                    "properties": {"inner": {"type": "string", "description": "Inner"}},
                },
            },
        }
    ).definitions == (
        "class TypeOuter(TypedDict):\n    # Inner\n    inner: NotRequired[str]",
        "class Type(TypedDict):\n    # Outer\n    # second line\n    outer: NotRequired[TypeOuter]",
    )


def test_resolves_local_references_and_stops_at_recursive_ones():
    schema = {
        "type": "object",
        "properties": {
            "item": {"$ref": "#/$defs/Item"},
            "legacy": {"$ref": "#/definitions/Legacy"},
            "remote": {"$ref": "https://example.com/schema.json"},
        },
        "required": ["item"],
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "parent": {"$ref": "#/$defs/Item"}},
                "required": ["id"],
            },
        },
        "definitions": {"Legacy": {"enum": ["a", "b"]}},
    }
    assert schema_to_type(schema).definitions == (
        "class TypeItem(TypedDict):\n    id: str\n    parent: NotRequired[Any]",
        (
            "class Type(TypedDict):\n"
            "    item: TypeItem\n"
            "    legacy: NotRequired[Literal['a', 'b']]\n"
            "    remote: NotRequired[Any]"
        ),
    )


def test_renders_arrays_and_tuples():
    assert _annotation({"type": "array", "items": {"type": "string"}}) == "list[str]"
    # Tuples arrive as lists.
    assert (
        _annotation({"type": "array", "prefixItems": [{"type": "string"}, {"type": "number"}]}) == "list[str | float]"
    )
    assert _annotation({"type": "array"}) == "list[Any]"


def test_renders_types_over_the_budget_as_any():
    schema = {"type": "object", "properties": {f"field{i}": {"type": "string"} for i in range(50)}}
    assert schema_to_type(schema, max_chars=100) == RenderedType("Any", ())
    assert "    field49: NotRequired[str]" in schema_to_type(schema).definitions[0]


# -- tool declarations -------------------------------------------------------


def test_renders_signatures_with_normalized_identifiers():
    [dynamic, free, keyword] = render_tools(
        [
            _tool(
                "hidden-dynamic-tool",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
                output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
            ),
            _tool("free"),
            _tool("class", input_schema={"type": "object", "properties": {"max-results": {"type": "integer"}}}),
        ]
    )
    assert dynamic.method == "    async def hidden_dynamic_tool(self, *, city: str) -> HiddenDynamicToolResult: ..."
    assert dynamic.definitions == ("class HiddenDynamicToolResult(TypedDict):\n    ok: bool",)
    assert free.method == "    async def free(self, **args: Any) -> Any: ..."
    # A keyword tool name gets a trailing `_`; a parameter that is not an
    # identifier makes the whole declaration `**args`.
    assert keyword.method == "    async def class_(self, **args: Any) -> Any: ..."
    assert keyword.parameters is None


def test_puts_required_parameters_first_and_lays_out_described_ones_per_line():
    [read] = render_tools(
        [
            _tool(
                "read",
                input_schema={
                    "type": "object",
                    "properties": {
                        "offset": {"type": "number", "description": "Line number to start reading from"},
                        "path": {"type": "string", "description": "Path to the file"},
                        "limit": {"type": "number"},
                        "tag": {"type": ["string", "null"]},
                    },
                    "required": ["path"],
                },
                output_schema={"type": "string"},
            )
        ]
    )
    assert read.method == (
        "    async def read(\n"
        "        self,\n"
        "        *,\n"
        "        # Path to the file\n"
        "        path: str,\n"
        "        # Line number to start reading from\n"
        "        offset: float | None = None,\n"
        "        limit: float | None = None,\n"
        "        tag: str | None = None,\n"
        "    ) -> str: ..."
    )
    assert [(p.name, p.required, p.accepts_null) for p in read.parameters] == [
        ("path", True, False),
        ("offset", False, False),
        ("limit", False, False),
        ("tag", False, True),
    ]


def test_renders_mcp_call_tool_result_output_schemas_as_call_tool_result_of_t():
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}
    [search, plain] = render_tools(
        [
            _tool(
                "mcp__sample__search",
                input_schema=input_schema,
                output_schema=_mcp_result_schema(
                    {
                        "type": "object",
                        "properties": {
                            "results": {"type": "array", "items": {"$ref": "#/definitions/Result~1item~0v1"}}
                        },
                        "required": ["results"],
                        "additionalProperties": False,
                        "definitions": {
                            "Result/item~v1": {
                                "type": "object",
                                "properties": {"id": {"type": "string"}, "score": {"type": "number"}},
                                "required": ["id", "score"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ),
            ),
            _tool("plain", input_schema=input_schema, output_schema=_mcp_result_schema()),
        ]
    )
    assert search.method == "    async def mcp__sample__search(self) -> CallToolResult[McpSampleSearchResult]: ..."
    assert search.definitions == (
        "class McpSampleSearchResultResultsItem(TypedDict):\n    id: str\n    score: float",
        "class McpSampleSearchResult(TypedDict):\n    results: list[McpSampleSearchResultResultsItem]",
    )
    assert search.uses_mcp_types
    assert plain.method == "    async def plain(self) -> CallToolResult: ..."
    assert mcp_structured_content_schema({"type": "object", "properties": {"content": {"type": "array"}}}) is None


def test_renders_the_per_tool_sample():
    [foo] = render_tools([_tool("foo", description="bar", input_schema={"type": "string"})])
    assert foo.sample == (
        "bar\n\ncodemode tool declaration:\n```python\nclass Tools:\n    async def foo(self, **args: Any) -> Any: ...\n```"
    )


def test_keeps_generated_names_unique_across_tools_and_reserved_names():
    output = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    [first, second, call_tool] = render_tools(
        [
            _tool("a_b", output_schema=output),
            _tool("aB", output_schema=output),
            _tool("call_tool", output_schema=output),
        ]
    )
    assert first.output_type == "ABResult"
    assert second.output_type == "ABResult2"
    # `CallToolResult` is an MCP type.
    assert call_tool.output_type == "CallToolResult2"
    [shadowing] = render_tools([_tool("my-tool"), _tool("my_tool")])
    assert shadowing.tool.name == "my-tool"


# -- render_stubs ------------------------------------------------------------


def test_renders_tools_and_globals():
    tools = render_tools(
        [
            _tool(
                "read",
                description="Read a file.\nSecond line.",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                output_schema={"type": "string"},
            ),
            _tool("remote-api"),
        ]
    )
    stubs = render_stubs(tools, [_global("attach", signature="(value: str) -> None")])
    assert stubs == (
        "from typing import Any, Literal, Never, NotRequired, TypedDict\n\n"
        "class Tools:\n"
        "    async def read(self, *, path: str) -> str: ...\n"
        "    async def remote_api(self, **args: Any) -> Any: ...\n\n"
        "tools: Tools\n\n"
        f"{BUILTIN_STUBS}\n\n"
        "async def attach(value: str) -> None: ...\n"
    )


def test_renders_namespaced_globals_and_explicit_signatures():
    stubs = render_stubs(
        [],
        [
            _global("models.list", signature="(type: str) -> list[str]"),
            _global("models.get"),
            _global("plain", signature="() -> None"),
        ],
        preamble="class Model(TypedDict):\n    id: str",
    )
    assert stubs.endswith(
        f"{BUILTIN_STUBS}\n\n"
        "class Model(TypedDict):\n    id: str\n\n"
        "async def plain() -> None: ...\n\n"
        "class Models:\n"
        "    async def list(self, type: str) -> list[str]: ...\n"
        "    async def get(self, *args: Any, **kwargs: Any) -> Any: ...\n\n"
        "models: Models\n"
    )
    assert "class Tools:\n    pass" in stubs


def test_adds_the_mcp_types_only_when_a_tool_needs_them():
    assert "class CallToolResult" not in render_stubs(render_tools([_tool("x")]))
    assert "class CallToolResult" in render_stubs(render_tools([_tool("x", output_schema=_mcp_result_schema())]))
