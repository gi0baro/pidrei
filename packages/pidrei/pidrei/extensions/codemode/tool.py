"""Mirror of pi coding-agent src/extensions/codemode/tool.ts.

The `codemode` tool: the model writes Python that calls other tools. Scripts use
`tools`, `ALL_TOOLS`, `text()`, `image()`, `exit()`, `store()`/`load()`,
`print()`, and a final expression line, may start with a `# @options:` line, and
reach the model catalog, classifiers, and image models through `models.*`.
Results start with a "Script completed" or "Script failed" header.

Scripts can call the agent loop's nested tools: active `direct` tools and every
`codemode` or `deferred` tool. Nested calls run through the agent loop's tool
pipeline (`ctx.execute_tool`), so validation, `tool_call`/`tool_result` hooks,
and permission checks apply exactly as for direct calls. Only the script's
output reaches the model; nested results do not.

Nested results are handed to the script as follows:
- A tool that declares `output_schema` resolves to its `structured_content`,
  also for error results that carry one (MCP tools resolve to their
  `CallToolResult`, including `isError`).
- Any other tool resolves to its text content as one string.
- A failed, blocked, or invalid call raises with the tool's error text.

A script that fails returns a normal error result that keeps its partial
output, followed by "Script error:" and the error. `store(key, value)` and
`load(key)` keep JSON values across calls; successful scripts append their
writes to the session as `codemode-store` custom entries, so each branch sees
the values written on its own path.

pi's text describes JavaScript; this one is pidrei's own (Python), keeping pi's
wording wherever it does not name JavaScript. Scripts run in a Monty pool owned
by whoever creates the tool: the codemode extension opens one at
`session_start` and closes it at `session_shutdown`; `create_codemode_tool()`
takes the caller's.
"""

import math
import os
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pidrei_agent.types import AgentTool
from pidrei_ai.types import GrammarConstrainedSampling
from pidrei_codemode import (
    CODEMODE_SOURCE_GRAMMAR,
    MCP_PYTHON_PREAMBLE,
    CodemodePool,
    CodemodeTool,
    RenderedTool,
    mcp_structured_content_schema,
    render_tools,
    reserved_type_names,
)

from ...config import get_docs_path
from ...core.extensions.types import ToolDefinition, ToolLoadout, ToolLoadoutChanges, ToolNamespace
from ...core.settings_manager import CodemodeMode
from ...core.tools.tool_definition_wrapper import WrappedDefinitionTool, wrap_tool_definition
from .renderer import CODEMODE_RENDERERS


CODEMODE_TOOL_NAME = "codemode"

# Custom entry type holding one script's `store()` writes: `{"set": {...}, "delete": [...]}`.
CODEMODE_STORE_ENTRY_TYPE = "codemode-store"

TEXT_OUTPUT_SCHEMA: dict[str, Any] = {"type": "string"}

CODEMODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"code": {"type": "string", "description": "Raw Python source."}},
    "required": ["code"],
}


def is_codemode_tool(tool: Any) -> bool:
    """Whether a registered tool is this package's `codemode` tool rather than
    another extension's tool with the same name. Compares the parameter schema,
    which the definition passes through by reference."""
    return tool.name == CODEMODE_TOOL_NAME and tool.parameters is CODEMODE_SCHEMA


type CodemodeNestedCallStatus = Literal["running", "ok", "error", "cancelled"]


@dataclass(frozen=True, slots=True)
class CodemodeNestedCall:
    # Tool call id of the nested call, `<codemode call id>/<n>`.
    id: str
    name: str
    # Compact JSON of the arguments, truncated for display.
    args: str
    status: CodemodeNestedCallStatus
    duration_ms: float | None = None
    # Error text, truncated for display.
    error: str | None = None
    # Cost in USD of a `models.*` call that reported usage.
    cost: float | None = None


@dataclass(frozen=True, slots=True)
class CodemodeToolDetails:
    calls: list[CodemodeNestedCall] = field(default_factory=list)
    # Temp file with the full text output, when the output was truncated.
    full_output_path: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class CodemodeToolOptions:
    # The pool scripts run in; None (or a closed pool) fails scripts as a
    # sandbox error.
    get_pool: Callable[[], CodemodePool | None]
    # Namespace of a tool, for `search_tools()` ranking and its `namespace` filter.
    get_tool_namespace: Callable[[str], ToolNamespace | None] | None = None
    # Prompt guidelines of every tool, by tool name, shown with declarations by `describe_tool()` and `ALL_TOOLS`.
    get_tool_guidelines: Callable[[], Mapping[str, Sequence[str]]] | None = None
    # Expose the `models` namespace to scripts, backed by the session's model
    # registry (`ctx.model_registry`). Without it, `models` is not declared.
    models: bool = False
    # Persists `store()` writes as a session custom entry. Without it, writes
    # last only for the current script; `load()` still reads entries already
    # on the branch.
    append_entry: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None
    # How the tool presents the loadout while active (the `codemode.mode` setting). Default: "on".
    get_mode: Callable[[], CodemodeMode] | None = None
    # Token budget for tool declarations in the description. Default: `DEFAULT_CODEMODE_INLINE_BUDGET`.
    get_inline_budget: Callable[[], float | None] | None = None
    # Whether scripts are type-checked before they run (the `codemode.typeCheck` setting). Default: True.
    get_type_check: Callable[[], bool] | None = None


def type_check_enabled(options: CodemodeToolOptions) -> bool:
    return options.get_type_check() if options.get_type_check is not None else True


CODEMODE_PROMPT_SNIPPET = "Run Python that calls other tools"
CODEMODE_PROMPT_GUIDELINES = (
    (
        "Use codemode to batch independent tool calls (all_settled), chain them, or filter large output, instead "
        "of many separate calls."
    ),
)

# The reference for scripts: globals, tool results, `store()`, the `models` API, and limits.
CODEMODE_DOCS_PATH = os.path.join(get_docs_path(), "codemode.md")


def _description_intro(type_check: bool) -> str:
    lines = [
        (
            "Run Python that calls other tools. The input is raw Python (not JSON, no code fence), run in a "
            "sandboxed interpreter: top-level `await` works, and if the last line is an expression, its value is "
            "added to the output. No third-party packages, file system, network, environment, subprocesses, or "
            "sleep. The interpreter supports a subset of Python: no `match`, `yield`, `del`, class inheritance (so "
            "no custom exception classes), or `asyncio.create_task`."
        ),
        (
            "- `await tools.<name>(arg=value, ...)` takes keyword arguments only, returns a `str`, or a `dict` if "
            "the tool's declaration says so, and raises on failure (catch with `except Exception`). Calls still "
            "running when the script ends are cancelled."
        ),
    ]
    if type_check:
        lines.append(
            "- The script is type-checked against the declarations before it runs; a script that fails the check "
            "does not run."
        )
    lines.append('- Optional first line: `# @options: {"max_output_tokens": 10000, "timeout_ms": 60000}`')
    return "\n".join(lines)


def _describe_globals(models: bool) -> str:
    """One line per global. The details live in `CODEMODE_DOCS_PATH`."""
    lines = [
        "Globals:",
        (
            "- `text(value)`, `image(data_url_or_image_block)`, `print(...)`, and a final expression line add "
            "output; `exit()` ends the script. With several text items, each starts with a `==> text N/M <==` "
            "line, and `print` lines follow the other output in one `<console_output>` block. `image()` also "
            "saves the image to a temp file and the result names its path."
        ),
        (
            "- `store(key, value)` and `load(key)` keep JSON values across codemode calls; `store(key, None)` "
            "deletes a key."
        ),
        (
            "- `await all_settled(*calls)` waits for every call and returns `{'status': 'fulfilled', 'value': ...}` "
            "or `{'status': 'rejected', 'reason': ...}` for each; `asyncio.gather()` raises on the first failure."
        ),
        (
            "- `ALL_TOOLS`, `has_tool(name)`, `await search_tools(query, limit=8, namespace=None)`, "
            "`await describe_tool(name)`, `await describe_namespace(name)`: find unlisted tools, such as MCP tools. "
            "`await call_tool(name, **args)` calls a tool by name."
        ),
    ]
    if models:
        lines.append(f"- `models`: classifiers and image generation. Read {CODEMODE_DOCS_PATH} first.")
    return "\n".join(lines)


# --- stubs for the extension's globals -----------------------------------------

DISCOVERY_STUBS = """class NamespaceInfo(TypedDict):
    name: str
    description: NotRequired[str]
    instructions: NotRequired[str]
    tools: list[str]"""

DISCOVERY_SIGNATURES: Mapping[str, str] = {
    "search_tools": "(query: str, limit: int = 8, namespace: str | None = None) -> list[ToolInfo]",
    "describe_tool": "(name: str) -> str | None",
    "describe_namespace": "(name: str) -> NamespaceInfo | None",
}

MODELS_STUBS = """type ModelType = Literal['chat', 'image', 'classifier']

class ModelRef(TypedDict):
    provider: str
    id: str

class ModelInfo(TypedDict):
    type: NotRequired[ModelType]
    provider: str
    id: str
    name: str
    api: str
    input: list[Literal['text', 'image']]
    contextWindow: NotRequired[int]

class ModelUsageCost(TypedDict):
    total: float

class ModelUsage(TypedDict):
    input: int
    output: int
    totalTokens: int
    cost: ModelUsageCost

class ChoiceQuestion(TypedDict):
    type: Literal['choice']
    instructions: str
    criteria: dict[str, str]

class ScoreQuestion(TypedDict):
    type: Literal['score']
    instructions: str
    criteria: list[str]

BoolCriteria = TypedDict('BoolCriteria', {'true': str, 'false': str})

class BoolQuestion(TypedDict):
    type: Literal['bool']
    instructions: str
    criteria: BoolCriteria

class ClassifierContext(TypedDict):
    state: dict[str, Any]
    images: NotRequired[list[ImageBlock]]
    questions: dict[str, ChoiceQuestion | ScoreQuestion | BoolQuestion]

class ChoiceAnswer(TypedDict):
    type: Literal['choice']
    choice: str
    probabilities: dict[str, float]
    confidence: float

class ScoreAnswer(TypedDict):
    type: Literal['score']
    score: float
    confidence: float

class BoolAnswer(TypedDict):
    type: Literal['bool']
    probability: float

class ClassifierResult(TypedDict):
    provider: str
    model: str
    answers: dict[str, ChoiceAnswer | ScoreAnswer | BoolAnswer]
    usage: NotRequired[ModelUsage]
    stopReason: Literal['stop', 'error', 'aborted']
    errorMessage: NotRequired[str]

class TextBlock(TypedDict):
    type: Literal['text']
    text: str

class ImagesContext(TypedDict):
    input: list[TextBlock | ImageBlock]

class ImagesResult(TypedDict):
    provider: str
    model: str
    output: list[TextBlock | ImageBlock]
    usage: NotRequired[ModelUsage]
    stopReason: Literal['stop', 'error', 'aborted']
    errorMessage: NotRequired[str]"""

MODELS_SIGNATURES: Mapping[str, str] = {
    "models.get_models_of_type": "(type: ModelType, provider: str | None = None) -> list[ModelInfo]",
    "models.get_available_of_type": "(type: ModelType, provider: str | None = None) -> list[ModelInfo]",
    "models.get_model_of_type": "(type: ModelType, provider: str, id: str) -> ModelInfo | None",
    "models.classify": "(model: ModelRef, context: ClassifierContext) -> ClassifierResult",
    "models.generate_images": "(model: ModelRef, context: ImagesContext) -> ImagesResult",
}


def stubs_preamble(models: bool) -> str:
    """The types the extension's globals name, for the sandbox's stubs."""
    return f"{DISCOVERY_STUBS}\n\n{MODELS_STUBS}" if models else DISCOVERY_STUBS


def global_signatures(models: bool) -> dict[str, str]:
    return {**DISCOVERY_SIGNATURES, **(MODELS_SIGNATURES if models else {})}


# --- declarations ----------------------------------------------------------------

# Default for the inline budget, in estimated tokens.
DEFAULT_CODEMODE_INLINE_BUDGET = 3000
# Characters per token when estimating the cost of a tool section.
_CHARS_PER_TOKEN = 4


async def _declared_only(_args: dict[str, Any]) -> Any:
    raise RuntimeError("A codemode declaration is not callable")


def to_codemode_declaration(tool: AgentTool, guidelines: Sequence[str] = ()) -> CodemodeTool:
    """What a script sees of a tool: its description followed by its prompt
    guidelines, which the system prompt only has for declared tools. Tools
    without an output schema resolve to their text output. Its `execute` is
    not callable: the executor builds the callable tools."""
    bullets = [f"- {guideline.strip()}" for guideline in guidelines if guideline.strip()]
    return CodemodeTool(
        name=tool.name,
        execute=_declared_only,
        description=(f"{tool.description.strip()}\n\n" + "\n".join(bullets)) if bullets else tool.description,
        input_schema=tool.parameters,
        output_schema=tool.output_schema if tool.output_schema is not None else TEXT_OUTPUT_SCHEMA,
    )


def get_codemode_callable_tools(tools: Iterable[AgentTool]) -> list[AgentTool]:
    """Tools a script may call: every given tool except the codemode tool itself."""
    return [tool for tool in tools if tool.name != CODEMODE_TOOL_NAME]


def render_codemode_tools(tools: Iterable[CodemodeTool], *, models: bool) -> list[RenderedTool]:
    """Declarations of the callable tools, named as the sandbox's stubs name them
    (same tools, same order, same reserved names), so the description, the
    `ALL_TOOLS` entries and the type checker's diagnostics agree."""
    reserved = reserved_type_names(global_signatures(models), stubs_preamble(models))
    return render_tools(tools, reserved_names=reserved)


def _render_tool_section(rendered: RenderedTool) -> str:
    """`### \\`id\\` (\\`raw name\\`)` followed by the tool's description and declaration."""
    name = rendered.tool.name
    heading = (
        f"### `{rendered.identifier}`" if rendered.identifier == name else f"### `{rendered.identifier}` (`{name}`)"
    )
    return f"{heading}\n{rendered.sample.strip()}"


@dataclass(slots=True)
class _CatalogEntry:
    name: str
    section: str
    cost: int


@dataclass(slots=True)
class _CatalogGroup:
    namespace: ToolNamespace | None
    entries: list[_CatalogEntry]


def _select_catalog(groups: Sequence[_CatalogGroup], budget: float | None) -> set[str]:
    """Pick the tool sections that fit the budget, like OpenCode's catalog: in
    each round every group (tools without a namespace first, then namespaces by
    name) places its cheapest remaining tool; a group whose next tool does not
    fit drops out while the others continue. Every namespace is represented
    before any namespace is complete."""
    if budget is None:
        return {entry.name for group in groups for entry in group.entries}
    queues = [sorted(group.entries, key=lambda entry: entry.cost) for group in groups]
    shown: set[str] = set()
    remaining = budget
    active = [queue for queue in queues if queue]
    while active:
        still_active = []
        for queue in active:
            entry = queue[0]
            if entry.cost > remaining:
                continue
            remaining -= entry.cost
            shown.add(entry.name)
            queue.pop(0)
            if queue:
                still_active.append(queue)
        active = still_active
    return shown


def create_codemode_description(
    tools: Sequence[AgentTool],
    *,
    models: bool = False,
    type_check: bool = True,
    namespaces: Mapping[str, ToolNamespace] | None = None,
    unlisted: Iterable[str] = (),
    guidelines: Mapping[str, Sequence[str]] | None = None,
    inline_budget: float | None = None,
) -> str:
    """Model-facing description: the helper list, guidance for finding tools
    that are not listed, the shared MCP types when listed tools need them, the
    `models` API, and one section per listed tool, grouped by namespace.

    Every callable tool in `tools` is rendered, so its declaration names match
    the stubs; `unlisted` ones (`deferred` exposure, and `direct` ones in mode
    "on") are not listed and do not affect the description at all, so it stays
    the same while MCP servers connect or change their tools. `guidelines` are
    each tool's prompt guidelines, listed after its description. Tool sections
    are limited to `inline_budget`."""
    unlisted = set(unlisted)
    rendered = [
        item
        for item in render_codemode_tools(
            (
                to_codemode_declaration(tool, guidelines.get(tool.name, ()) if guidelines is not None else ())
                for tool in get_codemode_callable_tools(tools)
            ),
            models=models,
        )
        if item.tool.name not in unlisted
    ]
    groups: dict[str, _CatalogGroup] = {"": _CatalogGroup(namespace=None, entries=[])}
    for item in rendered:
        namespace = namespaces.get(item.tool.name) if namespaces is not None else None
        key = f"ns:{namespace.name}" if namespace is not None else ""
        group = groups.setdefault(key, _CatalogGroup(namespace=namespace, entries=[]))
        section = _render_tool_section(item)
        group.entries.append(
            _CatalogEntry(name=item.tool.name, section=section, cost=math.ceil(len(section) / _CHARS_PER_TOKEN))
        )
    ordered = sorted(
        groups.values(),
        key=lambda group: (group.namespace is not None, group.namespace.name if group.namespace else ""),
    )
    shown = _select_catalog(ordered, inline_budget)

    sections = [_description_intro(type_check), _describe_globals(models)]
    if any(item.tool.name in shown and item.uses_mcp_types for item in rendered):
        sections.append(f"Shared MCP Types:\n```python\n{MCP_PYTHON_PREAMBLE}\n```")
    if not rendered:
        return "\n\n".join(sections)

    tool_sections = ["Nested tools:"]
    for group in ordered:
        visible = [entry for entry in group.entries if entry.name in shown]
        if group.namespace is not None:
            # Only tools that did not fit the budget are counted as not listed here.
            if len(visible) == len(group.entries):
                listing = ""
            elif not visible:
                listing = " (tools not listed)"
            else:
                listing = " (some tools not listed)"
            description = (group.namespace.description or "").strip()
            tool_sections.append(f"## {group.namespace.name}{listing}" + (f"\n{description}" if description else ""))
        tool_sections.extend(entry.section for entry in visible)
    sections.append("\n\n".join(tool_sections))
    return "\n\n".join(sections)


def _describe_output(rendered: RenderedTool) -> str:
    """What a script call returns, in a few words: `a string`, the keys of a
    dict, or the rendered type for anything else."""
    if rendered.output_type == "str":
        return "a string"
    schema = rendered.tool.output_schema
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    if (
        isinstance(schema, Mapping)
        and schema.get("type") == "object"
        and isinstance(properties, Mapping)
        and mcp_structured_content_schema(schema) is None
    ):
        required = set(schema["required"]) if isinstance(schema.get("required"), list) else set()
        keys = [name if name in required else f"{name} (optional)" for name in properties]
        return f"a dict with keys {', '.join(keys)}"
    return f"`{' '.join(rendered.output_type.split())}`"


def _describe_script_call(tool: AgentTool, rendered: RenderedTool) -> str:
    """A declared tool's description followed by how scripts call it and what
    the call returns. The arguments are the tool's declared parameters, so they
    are not repeated."""
    return (
        f"{tool.description.strip()}\n\nCodemode: `await tools.{rendered.identifier}(...)` takes the parameters as "
        f"keyword arguments and returns {_describe_output(rendered)}."
    )


def _prepare_codemode_loadout(loadout: ToolLoadout, options: CodemodeToolOptions) -> ToolLoadoutChanges:
    """How the codemode tool presents tools that are both declared and callable
    from scripts:
    - "on": their descriptions say how scripts call them, and the codemode
      description lists only the callable tools without "direct" exposure.
    - "only": the codemode description lists every callable tool, and requests
      leave out the declarations of active "direct" tools.

    Listed tools carry their prompt guidelines, which the system prompt only
    has for declared tools.

    Listing by exposure, not by the active set, keeps the codemode description
    unchanged when `tool_search` loads a tool, so loads do not redeclare
    codemode."""
    mode = options.get_mode() if options.get_mode is not None else "on"

    def is_direct(tool: AgentTool) -> bool:
        return loadout.get_exposure(tool.name) == "direct"

    callable_tools = get_codemode_callable_tools(loadout.callable)
    descriptions: dict[str, str] = {}
    if mode == "on":
        rendered = {
            item.tool.name: item
            for item in render_codemode_tools(
                (to_codemode_declaration(tool) for tool in callable_tools), models=options.models
            )
        }
        for tool in loadout.declared:
            if tool.name in rendered:
                descriptions[tool.name] = _describe_script_call(tool, rendered[tool.name])
    listed = callable_tools if mode == "only" else [tool for tool in callable_tools if not is_direct(tool)]
    listed_names = {tool.name for tool in listed}
    namespaces = {
        tool.name: namespace for tool in listed if (namespace := loadout.get_namespace(tool.name)) is not None
    }
    guidelines = {tool.name: loadout.get_prompt_guidelines(tool.name) for tool in listed}
    budget = options.get_inline_budget() if options.get_inline_budget is not None else None
    descriptions[CODEMODE_TOOL_NAME] = create_codemode_description(
        callable_tools,
        models=options.models,
        type_check=type_check_enabled(options),
        namespaces=namespaces,
        guidelines=guidelines,
        unlisted={
            tool.name
            for tool in callable_tools
            if tool.name not in listed_names or loadout.get_exposure(tool.name) == "deferred"
        },
        inline_budget=budget if budget is not None else DEFAULT_CODEMODE_INLINE_BUDGET,
    )
    declared_names = {tool.name for tool in loadout.declared}
    return ToolLoadoutChanges(
        descriptions=descriptions,
        hidden_declarations=(
            tuple(tool.name for tool in callable_tools if is_direct(tool) and tool.name in declared_names)
            if mode == "only"
            else ()
        ),
    )


def create_codemode_tool_definition(options: CodemodeToolOptions) -> ToolDefinition:
    async def execute(tool_call_id, params, cancel, on_update, ctx):
        # Imported here: execute.py imports this module.
        from .execute import execute_codemode

        return await execute_codemode(tool_call_id, params, cancel, on_update, ctx, options)

    return ToolDefinition(
        name=CODEMODE_TOOL_NAME,
        label=CODEMODE_TOOL_NAME,
        # Replaced with the declarations of the callable tools when the tool is
        # activated. Settings cannot be read yet while extensions load.
        description=create_codemode_description([], models=options.models),
        prompt_snippet=CODEMODE_PROMPT_SNIPPET,
        prompt_guidelines=list(CODEMODE_PROMPT_GUIDELINES),
        parameters=CODEMODE_SCHEMA,
        # Scripts must not start other scripts.
        exposure="model-only",
        prepare_loadout=lambda loadout: _prepare_codemode_loadout(loadout, options),
        # Capable models write the script as raw text instead of a JSON-escaped string.
        constrained_sampling=GrammarConstrainedSampling(variants={"openai_lark": CODEMODE_SOURCE_GRAMMAR}),
        execute=execute,
        render_call=CODEMODE_RENDERERS.render_call,
        render_result=CODEMODE_RENDERERS.render_result,
    )


def create_codemode_tool(
    pool: CodemodePool, tools: Sequence[AgentTool] = (), *, models: bool = False, type_check: bool = True
) -> WrappedDefinitionTool:
    """The codemode tool as an AgentTool, running scripts in `pool`, which the
    caller owns and closes. The description lists the given tools; the script
    can call whatever tools the agent loop provides at execution time."""
    options = CodemodeToolOptions(get_pool=lambda: pool, models=models, get_type_check=lambda: type_check)
    tool = wrap_tool_definition(create_codemode_tool_definition(options))
    tool.description = create_codemode_description(tools, models=models, type_check=type_check)
    return tool
