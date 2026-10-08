"""Mirror of pi's tool-search.test.ts, plus pidrei-only cases for the
`tool_search` tool's search-and-load step.

The codemode description cases check the Python declarations
(`codemode tool declaration:` blocks) and pass `unlisted` where pi passes
`deferred`.
"""

import pytest

from pidrei.core.extensions.types import ToolNamespace
from pidrei.extensions.codemode.tool import create_codemode_description
from pidrei.extensions.tool_search.tool import (
    Bm25Ranker,
    ToolSearchToolDetails,
    ToolSearchToolOptions,
    create_tool_search_document,
    create_tool_search_tool_definition,
    tokenize,
)
from pidrei_agent.types import AgentTool, AgentToolResult


class StubTool(AgentTool):
    def __init__(self, name: str, description: str, properties: dict | None = None, **info) -> None:
        self.name = name
        self.label = name
        self.description = description
        self.parameters = {"type": "object", "properties": properties or {}}
        # `ToolInfo` fields, for the search-and-load cases.
        self.exposure = info.get("exposure", "direct")
        self.namespace = info.get("namespace")

    async def execute(self, *_args):
        return AgentToolResult(content=[], details=None)


def tool(name: str, description: str, properties: dict | None = None) -> StubTool:
    return StubTool(name, description, properties)


class TestTokenize:
    def test_splits_camel_case_and_snake_case_drops_stop_words_and_folds_plurals(self):
        assert tokenize("listIssues for the GitHub_repo") == ["list", "issue", "git", "hub", "repo"]
        assert tokenize("searches queries HTTPServer") == ["search", "query", "http", "server"]


RANKED_TOOLS = (
    tool(
        "mcp__github__list_issues",
        "List issues in a repository.",
        {"state": {"type": "string", "description": "open or closed"}},
    ),
    tool("mcp__github__create_pull_request", "Open a pull request."),
    tool("mcp__linear__search_issues", "Search Linear issues by text."),
    tool("mcp__docs__search", "Search the documentation."),
)
DOCUMENTS = [create_tool_search_document(entry) for entry in RANKED_TOOLS]


class TestBm25Ranker:
    def test_ranks_by_term_relevance_and_respects_the_limit(self):
        ranker = Bm25Ranker()
        assert [match.name for match in ranker.rank("issue", DOCUMENTS, 8)] == [
            "mcp__linear__search_issues",
            "mcp__github__list_issues",
        ]
        assert ranker.rank("pull requests", DOCUMENTS, 8)[0].name == "mcp__github__create_pull_request"
        assert len(ranker.rank("search", DOCUMENTS, 1)) == 1
        # Property names and descriptions are searchable.
        assert [match.name for match in ranker.rank("closed", DOCUMENTS, 8)] == ["mcp__github__list_issues"]

    def test_returns_nothing_for_unknown_or_empty_queries(self):
        ranker = Bm25Ranker()
        assert ranker.rank("kubernetes", DOCUMENTS, 8) == []
        assert ranker.rank("the", DOCUMENTS, 8) == []
        # Known v1 limit: no synonyms, so "tickets" does not find "issues".
        assert ranker.rank("tickets", DOCUMENTS, 8) == []

    def test_includes_the_namespace_in_the_search_text(self):
        document = create_tool_search_document(
            tool("mcp__x__run", "Run it."), ToolNamespace(name="mcp__x", description="Kubernetes cluster tools")
        )
        matches = Bm25Ranker().rank("kubernetes", [document], 8)
        assert [match.name for match in matches] == ["mcp__x__run"]
        assert isinstance(matches[0].score, float)


PLAIN = tool("read_notes", "Read notes.")
GITHUB = [tool(f"mcp__github__{suffix}", f"GitHub {suffix}.") for suffix in ("a", "b", "c")]
DOCS = [tool("mcp__docs__search", "Search docs."), tool("mcp__docs__long", "Long " * 200)]
ALL = [PLAIN, *GITHUB, *DOCS]
NAMESPACES = {
    **{entry.name: ToolNamespace(name="mcp__github", description="GitHub server") for entry in GITHUB},
    **{entry.name: ToolNamespace(name="mcp__docs") for entry in DOCS},
}


class TestCodemodeDescriptionCatalog:
    def test_lists_everything_without_a_budget(self):
        description = create_codemode_description(ALL, namespaces=NAMESPACES)
        assert "Nested tools:" in description
        assert "## mcp__github\nGitHub server" in description
        assert "## mcp__docs\n\n### `mcp__docs" in description
        # The search guidance is always there, so tools that appear later do not change it.
        assert "find unlisted tools, such as MCP tools" in description
        # pi #10555: the lookup helpers are async; without `await` scripts get a coroutine.
        assert "`await search_tools(query, limit=8, namespace=None)`" in description
        assert "`await describe_tool(name)`" in description
        assert "`await describe_namespace(name)`" in description

    def test_fills_the_budget_round_robin_cheapest_first_and_says_what_is_missing(self):
        # Each small section costs 35 to 39 tokens: one tool per group, then one more.
        description = create_codemode_description(ALL, namespaces=NAMESPACES, inline_budget=170)
        assert "### `read_notes`" in description
        assert "## mcp__docs (some tools not listed)" in description
        assert "### `mcp__docs__search`" in description
        assert "### `mcp__docs__long`" not in description
        assert "## mcp__github (some tools not listed)" in description
        assert "find unlisted tools, such as MCP tools" in description
        # Deterministic: the same input gives the same description.
        assert create_codemode_description(ALL, namespaces=NAMESPACES, inline_budget=170) == description

    def test_leaves_deferred_tools_and_their_namespaces_out_entirely(self):
        deferred = {entry.name for entry in GITHUB}
        description = create_codemode_description(ALL, namespaces=NAMESPACES, unlisted=deferred)
        assert "mcp__github" not in description
        # Deferred tools, such as those of a server that connects later, do not change the description.
        assert description == create_codemode_description([PLAIN, *DOCS], namespaces=NAMESPACES)

    def test_leaves_namespace_instructions_out(self):
        description = create_codemode_description(
            GITHUB,
            namespaces={
                entry.name: ToolNamespace(name="mcp__github", instructions="Long usage guide.") for entry in GITHUB
            },
        )
        assert "## mcp__github\n\n### `mcp__github__a`" in description
        assert "Long usage guide." not in description

    def test_lists_only_namespaces_with_a_zero_budget(self):
        description = create_codemode_description(ALL, namespaces=NAMESPACES, inline_budget=0)
        assert "## mcp__docs (tools not listed)" in description
        assert "codemode tool declaration:" not in description


class FakeTools:
    """The session's tools as `tool_search` needs them: `update_active_tools`
    hands the update the active names it is given and records what it
    returns."""

    def __init__(self, tools: list[StubTool], active: list[str]) -> None:
        self.tools = tools
        self.active = active
        self.updates: list[list[str] | None] = []

    def get_all_tools(self) -> list[StubTool]:
        return self.tools

    def update_active_tools(self, update) -> None:
        names = update(list(self.active))
        self.updates.append(names)
        if names is not None:
            self.active = names


DOCS_TOOLS = [
    StubTool("read", "Read files.", exposure="direct"),
    StubTool("docs_search", "Search the docs.\nSecond line.", exposure="deferred"),
    StubTool("docs_fetch", "Fetch a docs page.", exposure="codemode"),
    StubTool("docs_hidden", "Hidden docs tool.", exposure="hidden"),
    StubTool("docs_direct", "Direct docs tool.", exposure="direct"),
]


async def run_tool_search(tools, params):
    definition = create_tool_search_tool_definition(ToolSearchToolOptions(tools=tools))
    return await definition.execute("call-1", params, None, None, None)


class TestToolSearchTool:
    """pidrei-only: pi covers the tool through the MCP suite, which ports with
    MCP; these check the search-and-load step on its own."""

    @pytest.mark.tonio
    async def test_loads_matching_codemode_and_deferred_tools_that_are_not_active(self):
        tools = FakeTools(DOCS_TOOLS, ["read", "tool_search", "docs_fetch"])
        result = await run_tool_search(tools, {"query": "docs"})
        # docs_fetch is already active; hidden and direct tools are never loaded.
        assert tools.active == ["read", "tool_search", "docs_fetch", "docs_search"]
        assert result.content[0].text == (
            "Loaded 1 tool. They are available from your next call:\n- docs_search: Search the docs."
        )
        assert result.details == ToolSearchToolDetails(loaded=["docs_search"])

    @pytest.mark.tonio
    @pytest.mark.parametrize(("limit", "loaded"), [(1.0, ["docs_fetch"]), (10**400, ["docs_fetch", "docs_search"])])
    async def test_takes_any_whole_limit(self, limit, loaded):
        tools = FakeTools(DOCS_TOOLS, [])
        result = await run_tool_search(tools, {"query": "docs", "limit": limit})
        assert result.details == ToolSearchToolDetails(loaded=loaded)
        assert tools.active == loaded

    @pytest.mark.tonio
    async def test_sets_nothing_when_nothing_matches(self):
        tools = FakeTools(DOCS_TOOLS, ["read"])
        result = await run_tool_search(tools, {"query": "kubernetes"})
        assert result.content[0].text == "No matching tools found."
        assert result.details == ToolSearchToolDetails(loaded=[])
        assert tools.updates == [None]

    @pytest.mark.tonio
    async def test_finds_nothing_without_tools(self):
        definition = create_tool_search_tool_definition()
        result = await definition.execute("call-1", {"query": "docs"}, None, None, None)
        assert result.content[0].text == "No matching tools found."

    @pytest.mark.tonio
    @pytest.mark.parametrize(
        ("params", "message"),
        [
            ({"query": "  "}, "query must not be empty"),
            ({"query": "docs", "limit": 0}, "limit must be a positive integer"),
            ({"query": "docs", "limit": 1.5}, "limit must be a positive integer"),
        ],
    )
    async def test_validates_the_query_and_the_limit(self, params, message):
        tools = FakeTools(DOCS_TOOLS, [])
        with pytest.raises(Exception, match=message):
            await run_tool_search(tools, params)
        assert tools.updates == []
