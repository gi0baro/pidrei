"""Mirror of pi codemode src/declarations.ts, rendering Python instead of
TypeScript.

pi renders TypeScript declarations that only the model reads. Here the same
renderer produces two things from one pass over the tools: each tool's sample
(its description and declaration, shown to the model) and the stub file the
type checker checks scripts against, so the names the model sees are the names
the checker reports.

Mapping from JSON Schema: `str`/`int`/`float`/`bool`/`None` for the scalar
types, `Literal[...]` for `const` and `enum`, `A | B` for `anyOf`/`oneOf` and
type lists, `list[T]` for arrays (tuples too: values arrive as lists), a
generated `TypedDict` for an object with `properties`, `dict[str, T]` for an
object with only `additionalProperties`, `Any` for anything else (`allOf` with
more than one renderable part, recursive or remote references, past 32
reference expansions). Generated `TypedDict` names are the tool identifier in
PascalCase followed by the property path (`EditEditsItem`), `Result` for the
output (`BashResult`).
"""

import re
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .identifier import is_identifier, to_codemode_identifier
from .types import CodemodeGlobal, CodemodeJsonSchema, CodemodeTool


INDENT = "    "
# Largest rendered input, in characters, before the declaration takes
# `**args: Any` instead.
DEFAULT_INPUT_SCHEMA_MAX_CHARS = 16_000
# Local `$ref` expansions per rendered schema, so shared definitions cannot
# blow up the output.
_MAX_REF_EXPANSIONS = 32

# Python types for MCP results, from the MCP `CallToolResult` schema, so
# `CallToolResult[T]` declarations can refer to them.
MCP_PYTHON_PREAMBLE = """type Role = Literal['user', 'assistant']
type MetaObject = dict[str, Any]

class Annotations(TypedDict, total=False):
    audience: list[Role]
    priority: float
    lastModified: str

class Icon(TypedDict):
    src: str
    mimeType: NotRequired[str]
    sizes: NotRequired[list[str]]
    theme: NotRequired[Literal['light', 'dark']]

class TextResourceContents(TypedDict):
    uri: str
    mimeType: NotRequired[str]
    _meta: NotRequired[MetaObject]
    text: str

class BlobResourceContents(TypedDict):
    uri: str
    mimeType: NotRequired[str]
    _meta: NotRequired[MetaObject]
    blob: str

class TextContent(TypedDict):
    type: Literal['text']
    text: str
    annotations: NotRequired[Annotations]
    _meta: NotRequired[MetaObject]

class ImageContent(TypedDict):
    type: Literal['image']
    data: str
    mimeType: str
    annotations: NotRequired[Annotations]
    _meta: NotRequired[MetaObject]

class AudioContent(TypedDict):
    type: Literal['audio']
    data: str
    mimeType: str
    annotations: NotRequired[Annotations]
    _meta: NotRequired[MetaObject]

class ResourceLink(TypedDict):
    icons: NotRequired[list[Icon]]
    name: str
    title: NotRequired[str]
    uri: str
    description: NotRequired[str]
    mimeType: NotRequired[str]
    annotations: NotRequired[Annotations]
    size: NotRequired[float]
    _meta: NotRequired[MetaObject]
    type: Literal['resource_link']

class EmbeddedResource(TypedDict):
    type: Literal['resource']
    resource: TextResourceContents | BlobResourceContents
    annotations: NotRequired[Annotations]
    _meta: NotRequired[MetaObject]

type ContentBlock = TextContent | ImageContent | AudioContent | ResourceLink | EmbeddedResource

class CallToolResult[T = dict[str, Any]](TypedDict):
    _meta: NotRequired[MetaObject]
    content: list[ContentBlock]
    isError: NotRequired[bool]
    structuredContent: NotRequired[T]"""

# The globals every script has, declared for the type checker. `has_tool` and
# `all_settled` are defined by the prelude; the others are host functions.
BUILTIN_STUBS = """class ToolInfo(TypedDict):
    name: str
    description: str

class ImageUrl(TypedDict):
    image_url: str

class ImageBlock(TypedDict):
    type: Literal['image']
    data: str
    mimeType: str

class SettledFulfilled(TypedDict):
    status: Literal['fulfilled']
    value: Any

class SettledRejected(TypedDict):
    status: Literal['rejected']
    reason: str

ALL_TOOLS: list[ToolInfo]

def text(value: Any) -> None: ...
def image(value: str | ImageUrl | ImageBlock) -> None: ...
def exit() -> Never: ...
def store(key: str, value: Any) -> None: ...
def load(key: str) -> Any: ...
def has_tool(name: str) -> bool: ...
async def all_settled(*calls: Any) -> list[SettledFulfilled | SettledRejected]: ...
async def call_tool(name: str, **args: Any) -> Any: ..."""

_STUB_IMPORTS = "from typing import Any, Literal, Never, NotRequired, TypedDict"

_DEFINED_NAME = re.compile(r"^(?:class|type)\s+(\w+)|^(\w+)\s*(?::|=)", re.MULTILINE)


def _defined_names(source: str) -> set[str]:
    """Top-level names a stub source defines (`class X`, `type X = ...`, `X = ...`, `X: ...`)."""
    return {first or second for first, second in _DEFINED_NAME.findall(source)}


_RESERVED_NAMES = frozenset(
    {"Any", "Literal", "Never", "NotRequired", "TypedDict", "Tools", "tools"}
    | _defined_names(MCP_PYTHON_PREAMBLE)
    | _defined_names(BUILTIN_STUBS)
)


def _is_object(value: Any) -> bool:
    return isinstance(value, Mapping)


def mcp_structured_content_schema(schema: CodemodeJsonSchema | None) -> CodemodeJsonSchema | None:
    """The `structuredContent` schema of an MCP `CallToolResult` output schema
    (detected by a `content` array of objects, boolean `isError`, and object
    `_meta`), `True` when it declares none, or `None` when the schema is not a
    `CallToolResult`."""
    if not _is_object(schema) or not _is_object(schema.get("properties")):
        return None
    properties = schema["properties"]
    content = properties.get("content")
    is_error = properties.get("isError")
    meta = properties.get("_meta")
    if (
        not _is_object(content)
        or content.get("type") != "array"
        or not _is_object(content.get("items"))
        or content["items"].get("type") != "object"
    ):
        return None
    if (
        not _is_object(is_error)
        or is_error.get("type") != "boolean"
        or not _is_object(meta)
        or meta.get("type") != "object"
    ):
        return None
    structured = properties.get("structuredContent")
    return structured if _is_object(structured) or isinstance(structured, bool) else True


def _pascal(identifier: str) -> str:
    """`github__search_issues` -> `GithubSearchIssues`, `oldText` -> `OldText`."""
    name = "".join(part[:1].upper() + part[1:] for part in to_codemode_identifier(identifier).split("_") if part)
    return name if name and not name[0].isdigit() else f"T{name}"


def _literal(value: Any) -> str | None:
    """A `Literal[...]` member for a JSON value, or None when Literal cannot hold it."""
    if value is None or isinstance(value, (bool, int, str)):
        return repr(value)
    return None


def _union(types: Iterable[str]) -> str:
    unique = list(dict.fromkeys(types))
    if "Any" in unique:
        return "Any"
    return "Never" if not unique else " | ".join(unique)


def _description(schema: Any) -> str:
    return schema["description"].strip() if _is_object(schema) and isinstance(schema.get("description"), str) else ""


def accepts_null(schema: Any) -> bool:
    """Whether a value of the schema may be JSON `null` (an untyped schema may)."""
    if schema is True or not _is_object(schema):
        return schema is not False
    if "const" in schema:
        return schema["const"] is None
    if isinstance(schema.get("enum"), list):
        return None in schema["enum"]
    variants = schema.get("anyOf") if isinstance(schema.get("anyOf"), list) else schema.get("oneOf")
    if isinstance(variants, list):
        return any(accepts_null(variant) for variant in variants)
    kind = schema.get("type")
    if isinstance(kind, list):
        return "null" in kind
    return kind is None or kind == "null"


class _Names:
    """Generated type names for one render, unique across every tool in it."""

    def __init__(self, reserved: Iterable[str]) -> None:
        self._taken = set(_RESERVED_NAMES) | set(reserved)

    def take(self, base: str) -> str:
        name = base
        suffix = 2
        while name in self._taken:
            name = f"{base}{suffix}"
            suffix += 1
        self._taken.add(name)
        return name


class _SchemaRenderer:
    """Renders one schema, collecting the `TypedDict`s it generates (nested ones first)."""

    def __init__(self, names: _Names, root: CodemodeJsonSchema) -> None:
        self._names = names
        self._root = root
        self._resolving: set[str] = set()
        self._expansions = 0
        self.definitions: list[str] = []

    def _resolve_ref(self, ref: str) -> Any:
        if ref != "#" and not ref.startswith("#/"):
            return None
        current: Any = self._root
        for segment in (part for part in ref[2:].split("/") if part):
            key = urllib.parse.unquote(segment).replace("~1", "/").replace("~0", "~")
            if not _is_object(current) or key not in current:
                return None
            current = current[key]
        return current if isinstance(current, bool) or _is_object(current) else None

    def render(self, schema: Any, hint: str) -> str:
        if schema is True:
            return "Any"
        if schema is False:
            return "Never"
        if not _is_object(schema):
            return "Any"
        ref = schema.get("$ref")
        if isinstance(ref, str):
            if ref in self._resolving or self._expansions >= _MAX_REF_EXPANSIONS:
                return "Any"
            target = self._resolve_ref(ref)
            if target is None:
                return "Any"
            self._expansions += 1
            self._resolving.add(ref)
            try:
                return self.render(target, hint)
            finally:
                self._resolving.discard(ref)

        if "const" in schema:
            literal = _literal(schema["const"])
            return f"Literal[{literal}]" if literal is not None else "Any"
        if isinstance(schema.get("enum"), list):
            literals = [_literal(value) for value in schema["enum"]]
            if not literals or any(literal is None for literal in literals):
                return "Any"
            return f"Literal[{', '.join(dict.fromkeys(literals))}]"

        variants = schema.get("anyOf") if isinstance(schema.get("anyOf"), list) else schema.get("oneOf")
        if isinstance(variants, list):
            return _union(self.render(variant, hint) for variant in variants)
        if isinstance(schema.get("allOf"), list):
            parts = [part for part in (self.render(part, hint) for part in schema["allOf"]) if part != "Any"]
            return parts[0] if len(parts) == 1 else "Any"

        kind = schema.get("type")
        if isinstance(kind, list):
            return _union(self.render({**schema, "type": entry}, hint) for entry in kind)
        match kind:
            case "string":
                return "str"
            case "integer":
                return "int"
            case "number":
                return "float"
            case "boolean":
                return "bool"
            case "null":
                return "None"
            case "array":
                return self._array(schema, hint)
            case "object":
                return self._object(schema, hint)
            case None:
                if "properties" in schema or "additionalProperties" in schema or "required" in schema:
                    return self._object(schema, hint)
                if "items" in schema or "prefixItems" in schema:
                    return self._array(schema, hint)
                return "Any"
            case _:
                return "Any"

    def _array(self, schema: Mapping[str, Any], hint: str) -> str:
        items = schema.get("items")
        if items is not None and not isinstance(items, list):
            return f"list[{self.render(items, f'{hint}Item')}]"
        tuple_items = schema.get("prefixItems") if isinstance(schema.get("prefixItems"), list) else items
        if isinstance(tuple_items, list) and tuple_items:
            return f"list[{_union(self.render(item, f'{hint}Item') for item in tuple_items)}]"
        return "list[Any]"

    def _object(self, schema: Mapping[str, Any], hint: str) -> str:
        properties = schema.get("properties") if _is_object(schema.get("properties")) else {}
        additional = schema.get("additionalProperties")
        if not properties:
            if _is_object(additional):
                return f"dict[str, {self.render(additional, f'{hint}Value')}]"
            return "dict[str, Any]"
        if additional is not None and additional is not False:
            return "dict[str, Any]"
        required = set(schema.get("required") or ()) if isinstance(schema.get("required"), list) else set()
        members = [
            (key, self.render(value, f"{hint}{_pascal(key)}"), key in required, _description(value))
            for key, value in properties.items()
        ]
        name = self._names.take(hint)
        self.definitions.append(_typed_dict(name, members))
        return name


def _typed_dict(name: str, members: Sequence[tuple[str, str, bool, str]]) -> str:
    def annotation(kind: str, required: bool) -> str:
        return kind if required else f"NotRequired[{kind}]"

    if all(is_identifier(key) for key, _kind, _required, _description in members):
        lines = [f"class {name}(TypedDict):"]
        for key, kind, required, description in members:
            lines.extend(f"{INDENT}# {line.strip()}" for line in description.splitlines() if line.strip())
            lines.append(f"{INDENT}{key}: {annotation(kind, required)}")
        return "\n".join(lines)
    fields = ", ".join(f"{key!r}: {annotation(kind, required)}" for key, kind, required, _ in members)
    return f"{name} = TypedDict({name!r}, {{{fields}}})"


@dataclass(frozen=True, slots=True)
class RenderedType:
    # The annotation (`str`, `list[Item]`, a generated name).
    annotation: str
    # The `TypedDict`s it names, nested ones first.
    definitions: tuple[str, ...]


def schema_to_type(schema: CodemodeJsonSchema, *, name: str = "Type", max_chars: int | None = None) -> RenderedType:
    """Convert a JSON Schema to a Python annotation, generating `TypedDict`s
    named from `name`. Local references (`#/$defs/...`, `#/definitions/...`)
    resolve against `schema`; recursive and remote references render as `Any`.
    A rendering longer than `max_chars` (annotation and definitions together)
    renders as `Any`."""
    renderer = _SchemaRenderer(_Names(()), schema)
    annotation = renderer.render(schema, name)
    definitions = tuple(renderer.definitions)
    if max_chars is not None and len(annotation) + sum(len(definition) for definition in definitions) > max_chars:
        return RenderedType("Any", ())
    return RenderedType(annotation, definitions)


@dataclass(frozen=True, slots=True)
class CodemodeParameter:
    name: str
    required: bool
    # Whether the schema accepts `null`: a `None` passed for an optional
    # parameter that does not is dropped instead of sent.
    accepts_null: bool


def input_parameters(schema: CodemodeJsonSchema | None) -> tuple[CodemodeParameter, ...] | None:
    """A tool's keyword parameters, required ones first, each group in the
    schema's order; empty for an object that allows no properties; None when
    the declaration takes `**args: Any` (no object schema, an open one without
    properties, or a parameter name that is not an identifier)."""
    if not _is_object(schema):
        return None
    properties = schema.get("properties")
    if not _is_object(properties) or not properties:
        return () if _is_object(properties) and schema.get("additionalProperties") is False else None
    if not all(is_identifier(name) for name in properties):
        return None
    required = set(schema.get("required") or ()) if isinstance(schema.get("required"), list) else set()
    parameters = [CodemodeParameter(name, name in required, accepts_null(value)) for name, value in properties.items()]
    return tuple(sorted(parameters, key=lambda parameter: not parameter.required))


@dataclass(frozen=True, slots=True)
class RenderedTool:
    tool: CodemodeTool
    # `tools.<identifier>` in scripts.
    identifier: str
    # The `TypedDict`s the declaration needs, nested ones first.
    definitions: tuple[str, ...]
    # The `async def` member of `class Tools`, indented.
    method: str
    # The type a call returns, as in the declaration.
    output_type: str
    # Keyword parameters, None for `**args: Any`.
    parameters: tuple[CodemodeParameter, ...] | None
    # Whether the declaration needs `MCP_PYTHON_PREAMBLE`.
    uses_mcp_types: bool

    @property
    def sample(self) -> str:
        """The tool's description followed by its declaration: what tool
        listings, `describe_tool()` and `ALL_TOOLS` entries show."""
        code = "\n\n".join([*self.definitions, f"class Tools:\n{self.method}"])
        return f"{(self.tool.description or '').strip()}\n\ncodemode tool declaration:\n```python\n{code}\n```"


def _render_tool(tool: CodemodeTool, identifier: str, names: _Names, input_max_chars: int) -> RenderedTool:
    base = _pascal(identifier)
    definitions: list[str] = []

    parameters = input_parameters(tool.input_schema)
    lines: list[str] | None = None
    if parameters is not None:
        renderer = _SchemaRenderer(names, tool.input_schema)  # type: ignore[arg-type]
        properties = tool.input_schema["properties"]  # type: ignore[index]
        lines = []
        for parameter in parameters:
            schema = properties[parameter.name]
            kind = renderer.render(schema, f"{base}{_pascal(parameter.name)}")
            if parameter.required:
                declaration = f"{parameter.name}: {kind}"
            elif kind == "Any" or "None" in kind.split(" | "):
                declaration = f"{parameter.name}: {kind} = None"
            else:
                declaration = f"{parameter.name}: {kind} | None = None"
            lines.extend(f"# {line.strip()}" for line in _description(schema).splitlines() if line.strip())
            lines.append(declaration)
        if sum(len(line) for line in [*lines, *renderer.definitions]) > input_max_chars:
            parameters = None
            lines = None
        else:
            definitions.extend(renderer.definitions)

    uses_mcp_types = False
    structured = mcp_structured_content_schema(tool.output_schema)
    if structured is not None:
        uses_mcp_types = True
        renderer = _SchemaRenderer(names, structured)
        kind = renderer.render(structured, f"{base}Result")
        definitions.extend(renderer.definitions)
        output_type = "CallToolResult" if kind == "Any" else f"CallToolResult[{kind}]"
    elif tool.output_schema is None:
        output_type = "Any"
    else:
        renderer = _SchemaRenderer(names, tool.output_schema)
        output_type = renderer.render(tool.output_schema, f"{base}Result")
        definitions.extend(renderer.definitions)

    if lines is None:
        method = f"{INDENT}async def {identifier}(self, **args: Any) -> {output_type}: ..."
    elif not lines:
        method = f"{INDENT}async def {identifier}(self) -> {output_type}: ..."
    elif any(line.startswith("#") for line in lines):
        body = "\n".join(f"{INDENT * 2}{line}" if line.startswith("#") else f"{INDENT * 2}{line}," for line in lines)
        method = f"{INDENT}async def {identifier}(\n{INDENT * 2}self,\n{INDENT * 2}*,\n{body}\n{INDENT}) -> {output_type}: ..."
    else:
        method = f"{INDENT}async def {identifier}(self, *, {', '.join(lines)}) -> {output_type}: ..."

    return RenderedTool(
        tool=tool,
        identifier=identifier,
        definitions=tuple(definitions),
        method=method,
        output_type=output_type,
        parameters=parameters,
        uses_mcp_types=uses_mcp_types,
    )


def render_tools(
    tools: Iterable[CodemodeTool],
    *,
    reserved_names: Iterable[str] = (),
    input_max_chars: int = DEFAULT_INPUT_SCHEMA_MAX_CHARS,
) -> list[RenderedTool]:
    """Render every tool's declaration with names unique across all of them.
    The first tool wins when two names normalize to the same identifier; the
    others are left out, as they are unreachable from scripts.
    `reserved_names` are names the stub preamble defines."""
    names = _Names(reserved_names)
    rendered: list[RenderedTool] = []
    seen: set[str] = set()
    for tool in tools:
        identifier = to_codemode_identifier(tool.name)
        if identifier in seen:
            continue
        seen.add(identifier)
        rendered.append(_render_tool(tool, identifier, names, input_max_chars))
    return rendered


def reserved_type_names(global_names: Iterable[str], preamble: str = "") -> set[str]:
    """The names `render_tools` must not generate next to these globals and
    stub preamble: the names the preamble defines and the namespace classes
    (`models.classify` -> `Models`). Rendering with them keeps a tool's
    declaration the same as in the stubs."""
    return _defined_names(preamble) | {
        namespace_class_name(name.partition(".")[0]) for name in global_names if "." in name
    }


def _global_stub(name: str, signature: str, indent: str = "") -> str:
    if indent:
        rest = signature[1:].lstrip()
        signature = f"(self{'' if rest.startswith(')') else ', '}{rest}"
    return f"{indent}async def {name}{signature}: ..."


def render_stubs(
    tools: Sequence[RenderedTool],
    globals: Sequence[CodemodeGlobal] = (),
    *,
    preamble: str = "",
) -> str:
    """The stub file scripts are type-checked against: the MCP types when a
    tool needs them, every tool's `TypedDict`s and its member of `class Tools`,
    the built-in globals, `preamble` (types the globals' signatures name), and
    the globals, `ns.member` globals grouped into a `class` per namespace."""
    sections = [_STUB_IMPORTS]
    if any(tool.uses_mcp_types for tool in tools):
        sections.append(MCP_PYTHON_PREAMBLE)
    sections.extend(definition for tool in tools for definition in tool.definitions)
    methods = "\n".join(tool.method for tool in tools) or f"{INDENT}pass"
    sections.extend([f"class Tools:\n{methods}", "tools: Tools", BUILTIN_STUBS])
    if preamble.strip():
        sections.append(preamble.strip())
    namespaces: dict[str, list[str]] = {}
    for item in globals:
        namespace, dot, member = item.name.partition(".")
        if not dot:
            sections.append(_global_stub(item.name, item.signature))
        else:
            namespaces.setdefault(namespace, []).append(_global_stub(member, item.signature, INDENT))
    for namespace, members in namespaces.items():
        sections.append(f"class {namespace_class_name(namespace)}:\n" + "\n".join(members))
        sections.append(f"{namespace}: {namespace_class_name(namespace)}")
    return "\n\n".join(sections) + "\n"


def namespace_class_name(namespace: str) -> str:
    """The class of a namespace object (`models` -> `Models`), in the stubs and in
    the sandbox, so error messages and diagnostics name the same class."""
    return _pascal(namespace)
