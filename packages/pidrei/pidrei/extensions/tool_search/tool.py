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
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent

from ...core.extensions.types import ToolDefinition, ToolExposure, ToolNamespace, ToolReach


TOOL_SEARCH_TOOL_NAME = "tool_search"
DEFAULT_TOOL_SEARCH_LIMIT = 8


@dataclass(frozen=True, slots=True)
class ToolSearchDocument:
    """A tool as the ranker sees it: its name and the texts built by
    `create_tool_search_document`."""

    name: str
    # Description, schema descriptions, and property names.
    text: str
    # Name, description, and instructions of the tool's namespace.
    namespace_text: str = ""


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
_ASCII_WORD = re.compile(r"[a-z0-9]+")
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


# Scripts written without spaces between words: each character is its own
# token (pi's `Intl.Segmenter` gives dictionary words). Blocks are enough:
# only letters, marks and numbers are kept, so symbols in them never reach
# the check. Han, Hiragana, Katakana, Thai, Lao, Tibetan, Myanmar, Khmer.
_UNIGRAM_BLOCKS = (
    (0x0E00, 0x0EFF),  # Thai, Lao
    (0x0F00, 0x0FFF),  # Tibetan
    (0x1000, 0x109F),  # Myanmar
    (0x1780, 0x17FF),  # Khmer
    (0x2E80, 0x2FDF),  # CJK Radicals Supplement, Kangxi Radicals
    (0x3005, 0x3007),  # ideographic iteration and closing marks, ideographic zero
    (0x3021, 0x3029),  # Hangzhou numerals
    (0x3038, 0x303B),
    (0x3040, 0x30FF),  # Hiragana, Katakana
    (0x31F0, 0x31FF),  # Katakana Phonetic Extensions
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xA9E0, 0xA9FF),  # Myanmar Extended-B
    (0xAA60, 0xAA7F),  # Myanmar Extended-A
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF66, 0xFF9F),  # halfwidth Katakana
    (0x1AFF0, 0x1B16F),  # Kana Extended-B, Kana Supplement, Kana Extended-A, Small Kana Extension
    (0x20000, 0x3FFFF),  # CJK Extensions B onwards, CJK Compatibility Ideographs Supplement
)


def _is_unigram_script(char: str) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in _UNIGRAM_BLOCKS)


def _segment_words(text: str) -> list[str]:
    """Words of non-ASCII text: runs of letters, marks and numbers, with each
    character of a script written without spaces as its own word, its
    combining marks attached. A mark with no letter or number before it is
    dropped, as pi's segmenter leaves it outside any word."""
    words: list[str] = []
    word: list[str] = []
    # Whether `word` is one unigram-script character with its marks.
    unigram = False
    for char in text:
        kind = unicodedata.category(char)[0]
        if kind == "M":
            if word:
                word.append(char)
            continue
        if kind in "LN":
            if _is_unigram_script(char):
                if word:
                    words.append("".join(word))
                word = [char]
                unigram = True
                continue
            if unigram:
                words.append("".join(word))
                word = []
                unigram = False
            word.append(char)
            continue
        if word:
            words.append("".join(word))
            word = []
        unigram = False
    if word:
        words.append("".join(word))
    return words


def tokenize(text: str) -> list[str]:
    """Unicode words with identifier boundaries, without English stop words, in singular form."""
    split = _ACRONYM_BOUNDARY.sub(r"\1 \2", _CAMEL_BOUNDARY.sub(r"\1 \2", unicodedata.normalize("NFC", text))).lower()
    if split.isascii():
        return [_stem(term) for term in _NON_ALPHANUMERIC.split(split) if term and term not in _STOP_WORDS]
    return [
        _stem(word) if _ASCII_WORD.fullmatch(word) else word
        for word in _segment_words(split)
        if word not in _STOP_WORDS
    ]


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
    `parameters`): the description and the schema descriptions and property
    names, and separately the namespace name, description, and instructions.
    The ranker reads the name from `name`."""
    parts = [tool.description]
    _schema_text(tool.parameters, parts)
    namespace_text = (
        " ".join([namespace.name, namespace.description or "", namespace.instructions or ""])
        if namespace is not None
        else ""
    )
    return ToolSearchDocument(
        name=tool.name, text=" ".join(part for part in parts if part.strip()), namespace_text=namespace_text
    )


_INVALID_NAME_CHARACTER = re.compile(r"[^a-z0-9_$]")


def _normalize_name(name: str) -> str:
    """Lowercase with characters invalid in script identifiers as `_`, so `dev-radius` matches `dev_radius`."""
    return _INVALID_NAME_CHARACTER.sub("_", name.strip().lower())


def _is_exact_name_query(query: str, name: str) -> bool:
    """Whether the whole query is the tool's name: its full name or a suffix
    after a `__` separator, so `thread_read` and `app__thread_read` both name
    `mcp__app__thread_read`."""
    normalized = _normalize_name(name)
    return normalized == query or normalized.endswith(f"__{query}")


@dataclass(frozen=True, slots=True)
class _IndexedDocument:
    """A document split into the fields the ranker weighs differently."""

    name: str
    # Terms of the name, for example `mcp`, `app`, `thread`, `read` for `mcp__app__thread_read`.
    name_terms: frozenset[str]
    # Terms of the last `__` segment of the name, the part models say: `thread`, `read`.
    local_name_terms: tuple[str, ...]
    text_counts: dict[str, int]
    text_length: int
    namespace_terms: frozenset[str]


def _index_document(document: ToolSearchDocument) -> _IndexedDocument:
    text_counts: dict[str, int] = {}
    for term in tokenize(document.text):
        text_counts[term] = text_counts.get(term, 0) + 1
    return _IndexedDocument(
        name=document.name,
        name_terms=frozenset(tokenize(document.name)),
        local_name_terms=tuple(dict.fromkeys(tokenize(document.name.split("__")[-1]))),
        text_counts=text_counts,
        text_length=sum(text_counts.values()),
        namespace_terms=frozenset(tokenize(document.namespace_text)),
    )


# A query term in the name counts this many times its IDF. The text part of a
# term is at most `k1 + 1` (2.2) times its IDF and about 1 for a single
# mention, so a name match beats any number of mentions in the description.
_NAME_WEIGHT = 3

# A query term in the namespace name, description, or instructions counts this
# many times its IDF. They are shared by every tool of the server, so they help
# find the server but not the tool in it.
_NAMESPACE_WEIGHT = 0.5


class Bm25Ranker:
    """Okapi BM25 over the description, with the name and the namespace as separate fields:

    - a tool that matches more of the query's terms, in any field, ranks above
      one that matches fewer, whatever their scores: for `screenshot window`, a
      tool whose description has both words beats `window`, whose name has one;
    - a query term in the name scores more than any number of mentions in the
      description;
    - a tool whose whole local name is in the query (`app` in `peekaboo app`
      for `mcp__peekaboo__app`) gets a bonus worth one more match of every
      query term, so among tools matching as many terms it mostly wins over
      tools that only mention the words;
    - a query that is exactly a tool's name ranks that tool first, before
      coverage.

    The description alone is length-normalized, so a long name or long
    namespace instructions do not lower a tool's score. Ties keep document
    order.
    """

    def __init__(self, *, k1: float = 1.2, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b

    def rank(self, query: str, documents: list[ToolSearchDocument], limit: int) -> list[ToolSearchMatch]:
        query_terms = dict.fromkeys(tokenize(query))
        if not query_terms or not documents or limit <= 0:
            return []
        indexed = [_index_document(document) for document in documents]
        average_length = sum(document.text_length for document in indexed) / len(indexed) or 1
        idf = {}
        for term in query_terms:
            frequency = sum(
                1
                for document in indexed
                if term in document.name_terms or term in document.text_counts or term in document.namespace_terms
            )
            idf[term] = math.log(1 + (len(indexed) - frequency + 0.5) / (frequency + 0.5))
        full_name_bonus = sum(idf.values())
        exact_query = _normalize_name(query)
        # (exact, covered, score, name) per matching document.
        matches: list[tuple[bool, int, float, str]] = []
        for document in indexed:
            norm = self._k1 * (1 - self._b + (self._b * document.text_length) / average_length)
            score = 0.0
            covered = 0
            for term in query_terms:
                count = document.text_counts.get(term, 0)
                weight = (
                    (_NAME_WEIGHT if term in document.name_terms else 0)
                    + (_NAMESPACE_WEIGHT if term in document.namespace_terms else 0)
                    + (count * (self._k1 + 1)) / (count + norm)
                )
                if weight > 0:
                    covered += 1
                score += idf[term] * weight
            if score == 0:
                continue
            local = document.local_name_terms
            if local and all(term in query_terms for term in local):
                score += full_name_bonus
            matches.append((_is_exact_name_query(exact_query, document.name), covered, score, document.name))
        # `sorted` is stable, so ties keep document order.
        ranked = sorted(matches, key=lambda match: (not match[0], -match[1], -match[2]))
        return [ToolSearchMatch(name=name, score=score) for _exact, _covered, score, name in ranked[:limit]]


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


async def _reaches_all_tools(_tool_input: dict[str, Any]) -> ToolReach:
    # A query can match any tool that is not declared yet.
    return "all"


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
        reaches_tools=_reaches_all_tools,
        execute=execute,
    )
