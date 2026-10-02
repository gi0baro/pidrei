"""Mirror of pi coding-agent src/extensions/tool-search/tool.ts.

Tool discovery: a BM25 ranker over tool metadata, shared by `search_tools()` in
codemode scripts and the optional `tool_search` tool.

`tool_search` searches tools that are not declared to the model (`codemode`
and `deferred` exposure) and loads the matches, so they are declared for the
next model call. Loading goes through the active tool set, so it is recorded in
the transcript like any other tool change and survives `/tree`, resume, and
fork on that branch.
"""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent

from ...core.extensions.types import ToolDefinition, ToolExposure, ToolNamespace


TOOL_SEARCH_TOOL_NAME = "tool_search"
DEFAULT_TOOL_SEARCH_LIMIT = 8


@dataclass(frozen=True, slots=True)
class ToolSearchDocument:
    """A tool as the ranker sees it: its name and the text built by
    `create_tool_search_document`."""

    name: str
    text: str


@dataclass(frozen=True, slots=True)
class ToolSearchMatch:
    name: str
    score: float


class ToolRanker(Protocol):
    """Ranks tools for a query. BM25 today; a hybrid ranker with embeddings can
    replace it."""

    def rank(self, query: str, documents: list[ToolSearchDocument], limit: int) -> list[ToolSearchMatch]: ...


_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
)

_CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_ACRONYM_BOUNDARY = re.compile(r"([A-Z]+)([A-Z][a-z])")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_SIBILANT_PLURAL = re.compile(r"(ches|shes|sses|xes|zes)$")


def _stem(term: str) -> str:
    """Naive singular form, so `issues` matches `issue` and `searches` matches `search`."""
    if len(term) > 4 and term.endswith("ies"):
        return f"{term[:-3]}y"
    if len(term) > 4 and _SIBILANT_PLURAL.search(term):
        return term[:-2]
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def tokenize(text: str) -> list[str]:
    """Lowercase terms, split at camelCase boundaries and non-alphanumerics, without stop words."""
    split = _ACRONYM_BOUNDARY.sub(r"\1 \2", _CAMEL_BOUNDARY.sub(r"\1 \2", text)).lower()
    return [_stem(term) for term in _NON_ALPHANUMERIC.split(split) if term and term not in _STOP_WORDS]


def _schema_text(schema: Any, parts: list[str]) -> None:
    """Schema descriptions and property names, recursively."""
    if not isinstance(schema, dict):
        return
    if isinstance(schema.get("description"), str):
        parts.append(schema["description"])
    if isinstance(schema.get("properties"), dict):
        for name, property_schema in schema["properties"].items():
            parts.append(name)
            _schema_text(property_schema, parts)
    _schema_text(schema.get("items"), parts)
    for key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for variant in variants:
                _schema_text(variant, parts)


def create_tool_search_document(tool: Any, namespace: ToolNamespace | None = None) -> ToolSearchDocument:
    """Search text of a tool (anything with `name`, `description` and
    `parameters`): the name, the name with `_` as spaces, the description,
    schema descriptions and property names, and the namespace with its
    description and instructions."""
    parts = [tool.name, tool.name.replace("_", " "), tool.description]
    _schema_text(tool.parameters, parts)
    if namespace is not None:
        parts.extend([namespace.name, namespace.description or "", namespace.instructions or ""])
    return ToolSearchDocument(name=tool.name, text=" ".join(part for part in parts if part.strip()))


class Bm25Ranker:
    """Okapi BM25 with the usual parameters. Ties keep document order."""

    def __init__(self, *, k1: float = 1.2, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b

    def rank(self, query: str, documents: list[ToolSearchDocument], limit: int) -> list[ToolSearchMatch]:
        query_terms = list(dict.fromkeys(tokenize(query)))
        if not query_terms or not documents or limit <= 0:
            return []
        term_counts: list[dict[str, int]] = []
        for document in documents:
            counts: dict[str, int] = {}
            for term in tokenize(document.text):
                counts[term] = counts.get(term, 0) + 1
            term_counts.append(counts)
        lengths = [sum(counts.values()) for counts in term_counts]
        average_length = sum(lengths) / len(documents) or 1
        idf = {}
        for term in query_terms:
            frequency = sum(1 for counts in term_counts if term in counts)
            idf[term] = math.log(1 + (len(documents) - frequency + 0.5) / (frequency + 0.5))
        matches: list[ToolSearchMatch] = []
        for index, document in enumerate(documents):
            score = 0.0
            for term in query_terms:
                count = term_counts[index].get(term)
                if not count:
                    continue
                norm = self._k1 * (1 - self._b + (self._b * lengths[index]) / average_length)
                score += idf[term] * ((count * (self._k1 + 1)) / (count + norm))
            if score > 0:
                matches.append(ToolSearchMatch(name=document.name, score=score))
        # `sorted` is stable, so ties keep document order.
        return sorted(matches, key=lambda match: -match.score)[:limit]


def is_positive_integer(value: Any) -> bool:
    """Whether a `limit` is a whole number above zero, like JavaScript's
    `Number.isInteger(value) && value > 0` (so `3.0` is one)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    return isinstance(value, float) and value.is_integer() and value > 0


TOOL_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query for deferred tools."},
        "limit": {
            "type": "number",
            "description": f"Maximum number of tools to return. Defaults to {DEFAULT_TOOL_SEARCH_LIMIT}.",
        },
    },
    "required": ["query"],
}


def is_tool_search_tool(tool: Any) -> bool:
    """Whether the tool is this `tool_search`, not another extension's tool of
    the same name. Compares the parameter schema, which the definition passes
    through by reference."""
    return tool.name == TOOL_SEARCH_TOOL_NAME and tool.parameters is TOOL_SEARCH_SCHEMA


@dataclass(frozen=True, slots=True)
class ToolSearchResultTool:
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class ToolSearchToolDetails:
    # Tools loaded by this call.
    loaded: list[str] = field(default_factory=list)


class ToolSearchTools(Protocol):
    """The session's tools, as `tool_search` needs them. The extension API
    fits."""

    def get_all_tools(self) -> list[Any]: ...

    def update_active_tools(self, update: Callable[[list[str]], list[str] | None]) -> None: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolSearchToolOptions:
    # The session's tools. `tool_search` searches the tools that are not
    # declared to the model and activates the matches. Without it, the tool
    # finds nothing.
    tools: ToolSearchTools | None = None


def _is_searchable(exposure: ToolExposure) -> bool:
    """Whether `tool_search` can load a tool with this exposure."""
    return exposure in ("codemode", "deferred")


def _search_and_load(tools: ToolSearchTools, query: str, limit: int) -> list[ToolSearchResultTool]:
    """Rank the searchable tools that are not active yet and activate the
    matches, so the next model call declares them. Activation is recorded in
    the transcript like any tool change.

    pi reads the active tools and sets them in two calls, which nothing can
    interleave on its single thread; here tool calls and other writers run in
    parallel, so the search and the activation run in one
    `update_active_tools()` step and a concurrent change is never lost."""
    loaded: list[ToolSearchResultTool] = []

    def update(active: list[str]) -> list[str] | None:
        candidates = [
            tool for tool in tools.get_all_tools() if _is_searchable(tool.exposure) and tool.name not in active
        ]
        documents = [create_tool_search_document(tool, tool.namespace) for tool in candidates]
        matches = Bm25Ranker().rank(query, documents, limit)
        if not matches:
            return None
        descriptions: dict[str, str] = {}
        for tool in candidates:
            descriptions.setdefault(tool.name, tool.description)
        loaded.extend(ToolSearchResultTool(name=match.name, description=descriptions[match.name]) for match in matches)
        return [*active, *(match.name for match in matches)]

    tools.update_active_tools(update)
    return loaded


# The `tool_search` description. It does not list the searchable tools or their
# namespaces, so it stays the same while tools are registered, for example when
# MCP servers connect.
TOOL_SEARCH_DESCRIPTION = (
    "# Tool discovery\n\nSearches over deferred tool metadata with BM25 and exposes matching tools for the next "
    "model call.\n\nSome of the tools, such as tools of MCP servers, may not have been provided to you upfront, "
    f"and you should use this tool (`{TOOL_SEARCH_TOOL_NAME}`) to search for the required tools. For MCP tool "
    f"discovery, always use `{TOOL_SEARCH_TOOL_NAME}`."
)

_LINE_BREAK = re.compile(r"\r?\n")


def _format_loaded(tools: list[ToolSearchResultTool]) -> str:
    if not tools:
        return "No matching tools found."
    lines = "\n".join(f"- {tool.name}: {_LINE_BREAK.split(tool.description.strip())[0]}" for tool in tools)
    return f"Loaded {len(tools)} tool{'' if len(tools) == 1 else 's'}. They are available from your next call:\n{lines}"


def create_tool_search_tool_definition(options: ToolSearchToolOptions | None = None) -> ToolDefinition:
    options = options if options is not None else ToolSearchToolOptions()

    async def execute(_tool_call_id, params, *_rest):
        query = params["query"]
        if query.strip() == "":
            raise Exception("query must not be empty")
        limit = params.get("limit")
        limit = DEFAULT_TOOL_SEARCH_LIMIT if limit is None else limit
        if not is_positive_integer(limit):
            raise Exception("limit must be a positive integer")
        tools = _search_and_load(options.tools, query, int(limit)) if options.tools is not None else []
        return AgentToolResult(
            content=[TextContent(text=_format_loaded(tools))],
            details=ToolSearchToolDetails(loaded=[tool.name for tool in tools]),
        )

    return ToolDefinition(
        name=TOOL_SEARCH_TOOL_NAME,
        label=TOOL_SEARCH_TOOL_NAME,
        description=TOOL_SEARCH_DESCRIPTION,
        prompt_snippet="Search for tools that are not loaded yet and load the matches",
        parameters=TOOL_SEARCH_SCHEMA,
        # Searching is not something scripts need; it changes what the model sees.
        exposure="model-only",
        execute=execute,
    )
