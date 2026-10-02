"""Mirror of pi coding-agent src/extensions/codemode/execute.ts: runs one
codemode script in the sandbox.

pi loads this module lazily so the QuickJS runtime only loads when a script
runs. Here the sandbox runtime (Monty) is primed at import, at program start,
so nothing is deferred.

Nested calls run in parallel on the runtime's threads, so the call rows, the
model usage and the generated image count are guarded by one lock, and each
update publishes a copy taken under it.
"""

import json
import math
import secrets
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from tonio.colored.sync import Semaphore

from pidrei_agent.types import AgentTool, AgentToolCallOutcome, AgentToolResult
from pidrei_ai.auth.types import AuthOperationOptions
from pidrei_ai.types import (
    AnyModel,
    ClassifierBoolQuestion,
    ClassifierChoiceQuestion,
    ClassifierContext,
    ClassifierOptions,
    ClassifierScoreQuestion,
    ImageContent,
    ImagesContext,
    ImagesOptions,
    ModelType,
    TextContent,
    Usage,
)
from pidrei_ai.utils import clock
from pidrei_codemode import (
    CodemodeError,
    CodemodeGlobal,
    CodemodeResult,
    CodemodeSandbox,
    CodemodeTool,
    RenderedTool,
    parse_codemode_source,
    to_codemode_identifier,
)

from ...config import TEMP_DIR
from ...core.extensions.types import ToolNamespace
from ...core.message_wire import to_wire_value
from ...core.model_wire import model_to_dict
from ...core.usage_totals import combine_usage
from ..tool_search.tool import (
    DEFAULT_TOOL_SEARCH_LIMIT,
    Bm25Ranker,
    create_tool_search_document,
    is_positive_integer,
)
from .tool import (
    CODEMODE_DOCS_PATH,
    CODEMODE_STORE_ENTRY_TYPE,
    CodemodeNestedCall,
    CodemodeNestedCallStatus,
    CodemodeToolDetails,
    CodemodeToolOptions,
    get_codemode_callable_tools,
    global_signatures,
    render_codemode_tools,
    stubs_preamble,
    to_codemode_declaration,
    type_check_enabled,
)


_ARGS_PREVIEW_CHARS = 200
_ERROR_PREVIEW_CHARS = 500
# `models.classify()` and `models.generate_images()` calls one script may have
# in flight; `asyncio.gather` over many items queues the rest.
_MAX_CONCURRENT_MODEL_CALLS = 4
# Memory limit of a script's worker. Overruns end the script with a
# `MemoryError` it cannot catch (pi's QuickJS limit is catchable).
CODEMODE_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024
_MODEL_TYPES: tuple[ModelType, ...] = ("chat", "image", "classifier")

_NO_POOL_MESSAGE = "The codemode sandbox is not running (the session has not started or has shut down)"


def _truncate_text(text: str, max_chars: int) -> str:
    return f"{text[: max_chars - 3]}..." if len(text) > max_chars else text


def _compact_json(value: Any) -> str:
    """`JSON.stringify` output: compact, non-ASCII kept."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _preview_args(args: Any) -> str:
    try:
        return _truncate_text(_compact_json(args), _ARGS_PREVIEW_CHARS)
    except TypeError, ValueError:
        return ""


def _text_of(result: AgentToolResult) -> str:
    return "\n".join(block.text for block in result.content or [] if block.type == "text")


def _to_model_type(value: Any) -> ModelType:
    if isinstance(value, str) and value in _MODEL_TYPES:
        return value  # type: ignore[return-value]
    raise ValueError(f'Unknown model type {_compact_json(value)}. Use "chat", "image", or "classifier".')


def _to_provider(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("provider must be a string")
    return value


def _to_model_info(model: AnyModel) -> dict[str, Any]:
    """Catalog entry for scripts. `headers` is dropped because models.json
    headers can carry credentials."""
    info = model_to_dict(model)
    info.pop("headers", None)
    return info


def _with_article(word: str) -> str:
    """`an image`, `a classifier`."""
    return f"{'an' if word[:1] in 'aeiou' else 'a'} {word}"


def _describe_value(value: Any) -> str:
    """A script value in an error message: `None`, `a str`, `a list`, or a
    dict's keys (`a dict with keys 'prompt'`)."""
    if value is None:
        return "None"
    if isinstance(value, list):
        return "an empty list" if not value else "a list"
    if isinstance(value, dict):
        if not value:
            return "an empty dict"
        keys = [repr(key) for key in list(value)[:6]]
        return f"a dict with keys {', '.join(keys)}{', ...' if len(value) > 6 else ''}"
    if isinstance(value, bool):
        return "a bool"
    if isinstance(value, int):
        return "an int"
    if isinstance(value, float):
        return "a float"
    return "a str" if isinstance(value, str) else f"a {type(value).__name__}"


_CLASSIFIER_CONTEXT_SHAPE = (
    "{'state': {...}, 'questions': {<id>: {'type': 'choice', 'instructions': ..., 'criteria': {<label>: <meaning>}} "
    "| {'type': 'score', 'instructions': ..., 'criteria': [<lowest level>, ..., <highest level>]} "
    "| {'type': 'bool', 'instructions': ..., 'criteria': {'true': <meaning>, 'false': <meaning>}}}}"
)
_IMAGES_CONTEXT_SHAPE = (
    "{'input': [{'type': 'text', 'text': <prompt>}, ...optional {'type': 'image', 'data': <base64>, "
    "'mimeType': ...} references]}"
)


def _is_strings(values: list[Any]) -> bool:
    return bool(values) and all(isinstance(value, str) for value in values)


def _check_classifier_context(context: Any) -> ClassifierContext:
    """Check a script's classifier context, so mistakes fail with the expected
    shape instead of a provider error."""

    def fail(problem: str) -> ValueError:
        return ValueError(
            f"models.classify() {problem}. Expected context: {_CLASSIFIER_CONTEXT_SHAPE}. "
            f'See "Classify" in {CODEMODE_DOCS_PATH}.'
        )

    if not isinstance(context, dict):
        raise fail(f"expects a context dict as its second argument, got {_describe_value(context)}")
    state = context.get("state")
    if not isinstance(state, dict):
        raise fail(f"context['state'] must be a dict, got {_describe_value(state)}")
    questions = context.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise fail(f"context['questions'] must map question IDs to questions, got {_describe_value(questions)}")
    checked: dict[str, Any] = {}
    for question_id, question in questions.items():
        at = f"context['questions'][{question_id!r}]"
        if not isinstance(question, dict):
            raise fail(f"{at} must be a question dict, got {_describe_value(question)}")
        instructions = question.get("instructions")
        if not isinstance(instructions, str):
            raise fail(f"{at}['instructions'] must be a string")
        criteria = question.get("criteria")
        match question.get("type"):
            case "choice":
                if not isinstance(criteria, dict) or not _is_strings(list(criteria.values())):
                    raise fail(f'{at} is a "choice" question, so criteria must map each label to its meaning')
                checked[question_id] = ClassifierChoiceQuestion(instructions=instructions, criteria=dict(criteria))
            case "score":
                if not isinstance(criteria, list) or not _is_strings(criteria):
                    raise fail(f'{at} is a "score" question, so criteria must list the levels as strings, lowest first')
                checked[question_id] = ClassifierScoreQuestion(instructions=instructions, criteria=list(criteria))
            case "bool":
                if (
                    not isinstance(criteria, dict)
                    or not isinstance(criteria.get("true"), str)
                    or not isinstance(criteria.get("false"), str)
                ):
                    raise fail(f"{at} is a \"bool\" question, so criteria must be {{'true': str, 'false': str}}")
                checked[question_id] = ClassifierBoolQuestion(
                    instructions=instructions, criteria={"true": criteria["true"], "false": criteria["false"]}
                )
            case other:
                raise fail(f'{at}[\'type\'] must be "choice", "score", or "bool", got {other!r}')
    return ClassifierContext(state=state, questions=checked)


def _check_images_context(context: Any) -> ImagesContext:
    """Check a script's image context, so mistakes such as `{'prompt': ...}`
    fail with the expected shape."""

    def fail(problem: str) -> ValueError:
        return ValueError(
            f"models.generate_images() {problem}. Expected context: {_IMAGES_CONTEXT_SHAPE}. "
            f'See "Generate images" in {CODEMODE_DOCS_PATH}.'
        )

    if not isinstance(context, dict):
        raise fail(f"expects a context dict as its second argument, got {_describe_value(context)}")
    blocks = context.get("input")
    if not isinstance(blocks, list) or not blocks:
        raise fail(f"context['input'] must be a non-empty list of blocks, got {_describe_value(blocks)}")
    checked: list[TextContent | ImageContent] = []
    for index, block in enumerate(blocks):
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            checked.append(TextContent(text=block["text"]))
        elif (
            isinstance(block, dict)
            and block.get("type") == "image"
            and isinstance(block.get("data"), str)
            and isinstance(block.get("mimeType"), str)
        ):
            checked.append(ImageContent(data=block["data"], mime_type=block["mimeType"]))
        else:
            raise fail(f"context['input'][{index}] must be a text or image block, got {_describe_value(block)}")
    return ImagesContext(input=checked)


def _bind(name: str, args: Sequence[Any], kwargs: Mapping[str, Any], parameters: Sequence[str]) -> dict[str, Any]:
    """The arguments a global was called with, by parameter name (only those
    given), with Python's errors for arguments it does not take."""
    if len(args) > len(parameters):
        raise TypeError(f"{name}() takes at most {len(parameters)} arguments ({len(args)} given)")
    bound = dict(zip(parameters, args, strict=False))
    for key, value in kwargs.items():
        if key not in parameters:
            raise TypeError(f"{name}() got an unexpected keyword argument {key!r}")
        if key in bound:
            raise TypeError(f"{name}() got multiple values for argument {key!r}")
        bound[key] = value
    return {parameter: bound[parameter] for parameter in parameters if parameter in bound}


def _is_store_entry_data(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    deleted = data.get("delete")
    return (
        isinstance(data.get("set"), dict) and isinstance(deleted, list) and all(isinstance(key, str) for key in deleted)
    )


def read_codemode_store(branch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Values of `load()`: the `codemode-store` entries on the branch, applied from the root."""
    store: dict[str, Any] = {}
    for entry in branch:
        data = entry.get("data")
        if (
            entry.get("type") != "custom"
            or entry.get("customType") != CODEMODE_STORE_ENTRY_TYPE
            or not _is_store_entry_data(data)
        ):
            continue
        for key in data["delete"]:
            store.pop(key, None)
        store.update(data["set"])
    return store


# Default token budget for script output.
_DEFAULT_MAX_OUTPUT_TOKENS = 10_000
# Characters per token when estimating.
_CHARS_PER_TOKEN = 4


def _value_text(value: Any) -> str:
    """Like the script's `text()`: strings as is, other values as compact JSON."""
    return value if isinstance(value, str) else _compact_json(value)


def _format_call_summary(calls: Sequence[CodemodeNestedCall]) -> str:
    if not calls:
        return "No tool calls were made."
    made = ", ".join(f"{call.name} ({call.status})" for call in calls)
    return f"Tool calls made before the failure (they are not undone): {made}"


def _format_error(error: CodemodeError, calls: Sequence[CodemodeNestedCall]) -> str:
    match error.kind:
        case "script":
            head = error.stack if error.stack is not None else f"{error.name or 'Error'}: {error.message}"
        case "timeout":
            head = f"Script timed out: {error.message}"
        case "aborted":
            head = f"Script aborted: {error.message}"
        case _:
            head = f"Script sandbox failed: {error.message}"
    return f"{head}\n\n{_format_call_summary(calls)}"


async def _spill_output(text: str) -> tuple[str | None, str | None]:
    """Write the full text output to a temp file, like bash does for truncated
    output. Returns the path, or the error."""
    path = TEMP_DIR / f"pidrei-codemode-{secrets.token_hex(8)}.txt"
    try:
        await path.write_text(text, encoding="utf-8")
    except Exception as error:
        return None, str(error)
    return str(path), None


async def _truncate_output(
    items: list[TextContent | ImageContent], max_tokens: int
) -> tuple[list[TextContent | ImageContent], str | None]:
    """Apply the token budget: when the combined text exceeds it, the text items
    become one item that keeps the start and end of the text, and images follow
    it. The full text is written to a temp file, whose path is returned."""
    texts = [item.text for item in items if item.type == "text"]
    combined = "\n".join(texts)
    budget = max_tokens * _CHARS_PER_TOKEN
    if not texts or len(combined) <= budget:
        return items, None
    head_chars = budget // 2
    tail_chars = budget - head_chars
    removed = len(combined) - head_chars - tail_chars
    head = combined[:head_chars]
    tail = combined[-tail_chars:] if tail_chars > 0 else ""
    text = (
        f"Warning: truncated output (original token count: {math.ceil(len(combined) / _CHARS_PER_TOKEN)})\n"
        f"Total output lines: {len(combined.split(chr(10)))}\n\n"
        f"{head}…{math.ceil(removed / _CHARS_PER_TOKEN)} tokens truncated…{tail}"
    )
    path, error = await _spill_output(combined)
    text += (
        f"\n\n[Full output: {path} (read with offset/limit)]"
        if path is not None
        else f"\n\n[Could not save the full output: {error}]"
    )
    return [TextContent(text=text), *(item for item in items if item.type == "image")], path


def _to_script_value(tool: AgentTool, outcome: AgentToolCallOutcome) -> Any:
    """The value a script receives for a nested call: a tool that declares
    `output_schema` returns its `structured_content`, also for error results
    that carry one (such as MCP results with `isError`); any other tool returns
    its text content. Other failures raise with the tool's error text."""
    result = outcome.result
    if tool.output_schema is not None and result.structured_content is not None:
        return result.structured_content
    text = _text_of(result)
    if outcome.is_error:
        raise RuntimeError(text or f'Tool "{tool.name}" failed')
    return text


@dataclass(slots=True)
class _CallRow:
    """A nested call row while the script runs; frozen into a
    `CodemodeNestedCall` for each update."""

    id: str
    name: str
    args: str
    status: CodemodeNestedCallStatus = "running"
    duration_ms: float | None = None
    error: str | None = None
    cost: float | None = None

    def freeze(self) -> CodemodeNestedCall:
        return CodemodeNestedCall(
            id=self.id,
            name=self.name,
            args=self.args,
            status=self.status,
            duration_ms=self.duration_ms,
            error=self.error,
            cost=self.cost,
        )


class _ScriptState:
    """What one script's host side accumulates: the call rows, the usage of its
    `models.*` calls (nested tool calls report theirs through the session), and
    the images `models.generate_images()` returned, to notice a script that
    never shows them."""

    def __init__(self, on_update: Callable[[AgentToolResult], None] | None) -> None:
        self._on_update = on_update
        # Guards the fields below. Never held across an await.
        self._guard = threading.Lock()
        self._calls: list[_CallRow] = []
        self._model_usage: Usage | None = None
        self._generated_images = 0
        # Held from taking a snapshot to delivering it, so updates reach the UI
        # in the order they were taken: an older one never lands after a newer
        # one. Taken before `_guard`, never the other way round.
        self._publish_guard = threading.Lock()

    def _publish(self) -> None:
        if self._on_update is None:
            return
        with self._publish_guard:
            with self._guard:
                details = CodemodeToolDetails(calls=[row.freeze() for row in self._calls])
            self._on_update(AgentToolResult(content=[], details=details))

    def add_call(self, row: _CallRow) -> None:
        with self._guard:
            self._calls.append(row)
        self._publish()

    def update_call(self, row: _CallRow, **changes: Any) -> None:
        with self._guard:
            for key, value in changes.items():
                setattr(row, key, value)
        self._publish()

    def add_model_usage(self, usage: Usage) -> None:
        with self._guard:
            self._model_usage = usage if self._model_usage is None else combine_usage(self._model_usage, usage)

    def add_generated_images(self, count: int) -> None:
        with self._guard:
            self._generated_images += count

    def finish(self) -> tuple[CodemodeToolDetails, Usage | None, int]:
        """The final rows, model usage and generated image count. Calls still
        marked running were cut off by the script ending, a timeout, or an
        abort."""
        with self._guard:
            for row in self._calls:
                if row.status == "running":
                    row.status = "cancelled"
            details = CodemodeToolDetails(calls=[row.freeze() for row in self._calls])
            return details, self._model_usage, self._generated_images


def _create_sandbox_tool(
    rendered: RenderedTool, tool: AgentTool, tool_call_id: str, ctx: Any, state: _ScriptState
) -> CodemodeTool:
    async def execute(args: dict[str, Any]) -> Any:
        row = _CallRow(id=f"{tool_call_id}/?", name=tool.name, args=_preview_args(args))
        state.add_call(row)
        started_at = clock.monotonic()
        outcome = await ctx.execute_tool(tool.name, args)
        changes: dict[str, Any] = {
            "id": outcome.tool_call.id,
            "duration_ms": (clock.monotonic() - started_at) * 1000,
            "status": "ok",
        }
        if outcome.is_error:
            changes["status"] = "error"
            changes["error"] = _truncate_text(
                _text_of(outcome.result) or f'Tool "{tool.name}" failed', _ERROR_PREVIEW_CHARS
            )
        state.update_call(row, **changes)
        return _to_script_value(tool, outcome)

    # ALL_TOOLS entries carry the declaration.
    return CodemodeTool(
        name=tool.name,
        execute=execute,
        description=rendered.sample,
        input_schema=rendered.tool.input_schema,
        output_schema=rendered.tool.output_schema,
    )


async def execute_codemode(
    tool_call_id: str,
    params: Mapping[str, Any],
    cancel: Any,
    on_update: Callable[[AgentToolResult], None] | None,
    ctx: Any,
    options: CodemodeToolOptions,
) -> AgentToolResult[CodemodeToolDetails]:
    """Run one script. Without a session context (a plain Agent or a direct
    call) scripts cannot call tools, `store()` starts empty, and writes are
    dropped."""
    started_at = clock.monotonic()
    source = parse_codemode_source(params["code"])
    state = _ScriptState(on_update)
    models = options.models and ctx is not None

    callable_tools = get_codemode_callable_tools(ctx.tools) if ctx is not None else []
    by_name = {tool.name: tool for tool in callable_tools}
    rendered = render_codemode_tools((to_codemode_declaration(tool) for tool in callable_tools), models=models)
    samples = {item.tool.name: item.sample for item in rendered}
    reachable = [by_name[item.tool.name] for item in rendered]

    pool = options.get_pool()
    if pool is None:
        result = CodemodeResult(
            ok=False, output=(), calls=(), error=CodemodeError(kind="sandbox", message=_NO_POOL_MESSAGE)
        )
    else:
        globals = _create_discovery_globals(reachable, samples, options)
        if models:
            globals.extend(_create_model_globals(ctx.model_registry, tool_call_id, cancel, state))
        sandbox = CodemodeSandbox(
            pool,
            tools=[_create_sandbox_tool(item, by_name[item.tool.name], tool_call_id, ctx, state) for item in rendered],
            globals=globals,
            stubs_preamble=stubs_preamble(models),
            timeout_ms=source.options.timeout_ms if source.options.timeout_ms is not None else math.inf,
            memory_limit_bytes=CODEMODE_MEMORY_LIMIT_BYTES,
            type_check=type_check_enabled(options),
        )
        store = read_codemode_store(ctx.session_manager.get_branch()) if ctx is not None else {}
        # Each script has its own sandbox, done with once `execute` returns
        # (pi closes its worker here); the pool stays with its owner.
        result = await sandbox.execute(source.code, cancel=cancel, store=store)
    details, model_usage, generated_images = state.finish()

    items: list[TextContent | ImageContent] = [
        TextContent(text=item.text) if item.type == "text" else ImageContent(data=item.data, mime_type=item.mime_type)
        for item in result.output
    ]
    if result.ok:
        writes = result.store_writes
        if writes is not None and (writes.set or writes.delete) and options.append_entry is not None:
            await options.append_entry(
                CODEMODE_STORE_ENTRY_TYPE, {"set": dict(writes.set), "delete": list(writes.delete)}
            )
        # pi's extension: a returned value is appended like text().
        if result.value is not None:
            items.append(TextContent(text=_value_text(result.value)))
    else:
        items.append(TextContent(text=f"Script error:\n{_format_error(result.error, details.calls)}"))
    if generated_images > 0 and not any(item.type == "image" for item in items):
        items.append(
            TextContent(
                text=f"Note: models.generate_images() returned {generated_images} "
                f"image{'' if generated_images == 1 else 's'} that the "
                "script did not show. Show each image block of result['output'] with image(block)."
            )
        )

    max_tokens = source.options.max_output_tokens
    items, full_output_path = await _truncate_output(
        items, max_tokens if max_tokens is not None else _DEFAULT_MAX_OUTPUT_TOKENS
    )
    wall_time = f"{clock.monotonic() - started_at:.1f}"
    header = f"{'Script completed' if result.ok else 'Script failed'}\nWall time {wall_time} seconds\nOutput:\n"
    return AgentToolResult(
        content=[TextContent(text=header), *items],
        details=replace(details, full_output_path=full_output_path),
        usage=model_usage,
        is_error=None if result.ok else True,
    )


def _is_namespace_name(namespace: str, query: str) -> bool:
    """Whether `query` names the namespace: its name, its script identifier
    (`mcp__dev-radius` is `mcp__dev_radius`), or the part after its last `__`
    in either form (`dev-radius`, `dev_radius`)."""
    identifier = to_codemode_identifier(namespace)
    query_identifier = to_codemode_identifier(query)

    def suffix(name: str) -> str | None:
        return name[name.rindex("__") + 2 :] if "__" in name else None

    return (
        namespace == query
        or identifier == query_identifier
        or suffix(namespace) == query
        or suffix(identifier) == query_identifier
    )


def _create_discovery_globals(
    tools: Sequence[AgentTool], samples: Mapping[str, str], options: CodemodeToolOptions
) -> list[CodemodeGlobal]:
    """`search_tools()`, `describe_tool()`, and `describe_namespace()`: ranked
    search and lookup over the script's nested tools and their namespaces.
    Tools are named by their identifier, the only form scripts call them by."""
    ranker = Bm25Ranker()
    signatures = global_signatures(False)

    def namespace_of(name: str) -> ToolNamespace | None:
        return options.get_tool_namespace(name) if options.get_tool_namespace is not None else None

    def entry(name: str) -> dict[str, str]:
        return {"name": to_codemode_identifier(name), "description": samples.get(name, "")}

    async def search_tools(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        bound = _bind("search_tools", args, kwargs, ("query", "limit", "namespace"))
        query = bound.get("query")
        if not isinstance(query, str):
            raise TypeError("search_tools() expects a query string")
        limit = bound.get("limit")
        limit = DEFAULT_TOOL_SEARCH_LIMIT if limit is None else limit
        if not is_positive_integer(limit):
            raise ValueError("search_tools() limit must be a positive integer")
        namespace = bound.get("namespace")
        if namespace is not None and not isinstance(namespace, str):
            raise TypeError("search_tools() namespace must be a string")
        documents = []
        for tool in tools:
            tool_namespace = namespace_of(tool.name)
            if namespace and (tool_namespace is None or not _is_namespace_name(tool_namespace.name, namespace)):
                continue
            documents.append(create_tool_search_document(tool, tool_namespace))
        return [entry(match.name) for match in ranker.rank(query, documents, int(limit))]

    async def describe_tool(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        name = _bind("describe_tool", args, kwargs, ("name",)).get("name")
        if not isinstance(name, str):
            raise TypeError("describe_tool() expects a tool name")
        tool = next((tool for tool in tools if to_codemode_identifier(tool.name) == name), None)
        return samples.get(tool.name) if tool is not None else None

    async def describe_namespace(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        name = _bind("describe_namespace", args, kwargs, ("name",)).get("name")
        if not isinstance(name, str):
            raise TypeError("describe_namespace() expects a namespace name")
        found: ToolNamespace | None = None
        names: list[str] = []
        for tool in tools:
            tool_namespace = namespace_of(tool.name)
            if tool_namespace is None or not _is_namespace_name(tool_namespace.name, name):
                continue
            if found is None:
                found = tool_namespace
            names.append(to_codemode_identifier(tool.name))
        if found is None:
            return None
        info: dict[str, Any] = {"name": found.name}
        if found.description:
            info["description"] = found.description
        if found.instructions:
            info["instructions"] = found.instructions
        info["tools"] = names
        return info

    return [
        CodemodeGlobal(name="search_tools", execute=search_tools, signature=signatures["search_tools"]),
        CodemodeGlobal(name="describe_tool", execute=describe_tool, signature=signatures["describe_tool"]),
        CodemodeGlobal(
            name="describe_namespace", execute=describe_namespace, signature=signatures["describe_namespace"]
        ),
    ]


def _create_model_globals(registry: Any, tool_call_id: str, cancel: Any, state: _ScriptState) -> list[CodemodeGlobal]:
    """`models.*` for scripts: the model registry methods documented in
    docs/codemode.md. Classifier and image calls appear as nested call rows so
    the renderer shows them, and their usage is added to the result. Rows show
    only the model, never prompts or image data."""
    limiter = Semaphore(_MAX_CONCURRENT_MODEL_CALLS)
    counter_guard = threading.Lock()
    call_count = 0

    def next_call_number() -> int:
        nonlocal call_count
        with counter_guard:
            call_count += 1
            return call_count

    async def run_model_call(
        name: str,
        model_type: Literal["classifier", "image"],
        bound: dict[str, Any],
        check_context: Callable[[Any], Any],
        run: Callable[[AnyModel, Any], Any],
    ) -> Any:
        """Resolve the script's model by provider and id only, check the
        context, then run the call as a nested call row. A script-supplied
        baseUrl or headers must never receive the credentials."""
        model = bound.get("model")
        list_hint = (
            f"List the {model_type} models you can use with models.get_available_of_type({_compact_json(model_type)})."
        )
        if (
            not isinstance(model, dict)
            or not isinstance(model.get("provider"), str)
            or not isinstance(model.get("id"), str)
        ):
            none_hint = (
                " models.get_model_of_type() returns None for an unknown provider or id." if model is None else ""
            )
            raise TypeError(
                f"{name}() expects {_with_article(model_type)} model as its first argument, got "
                f"{_describe_value(model)}.{none_hint} {list_hint}"
            )
        provider, model_id = model["provider"], model["id"]
        ref = f"{provider}/{model_id}"
        resolved = registry.get_model_of_type(model_type, provider, model_id)
        if resolved is None:
            actual = next(
                (
                    other
                    for other in _MODEL_TYPES
                    if other != model_type and registry.get_model_of_type(other, provider, model_id) is not None
                ),
                None,
            )
            raise ValueError(
                f'"{ref}" is {_with_article(actual)} model, not {_with_article(model_type)} model. {list_hint}'
                if actual is not None
                else f'Unknown {model_type} model "{ref}". {list_hint}'
            )
        checked = check_context(bound.get("context"))

        row = _CallRow(
            id=f"{tool_call_id}/{name}/{next_call_number()}", name=name, args=f"{resolved.provider}/{resolved.id}"
        )
        state.add_call(row)
        started_at = clock.monotonic()
        async with limiter:
            result = await run(resolved, checked)
        changes: dict[str, Any] = {
            "duration_ms": (clock.monotonic() - started_at) * 1000,
            "status": "ok"
            if result.stop_reason == "stop"
            else "cancelled"
            if result.stop_reason == "aborted"
            else "error",
        }
        if result.error_message:
            changes["error"] = _truncate_text(result.error_message, _ERROR_PREVIEW_CHARS)
        if result.usage is not None:
            changes["cost"] = result.usage.cost.total
            state.add_model_usage(result.usage)
        state.update_call(row, **changes)
        return to_wire_value(result)

    async def get_models_of_type(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        bound = _bind("models.get_models_of_type", args, kwargs, ("type", "provider"))
        models = registry.get_models_of_type(_to_model_type(bound.get("type")), _to_provider(bound.get("provider")))
        return [_to_model_info(model) for model in models]

    async def get_available_of_type(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        bound = _bind("models.get_available_of_type", args, kwargs, ("type", "provider"))
        available = await registry.get_available_of_type(
            _to_model_type(bound.get("type")), _to_provider(bound.get("provider")), AuthOperationOptions(cancel=cancel)
        )
        return [_to_model_info(model) for model in available]

    async def get_model_of_type(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        bound = _bind("models.get_model_of_type", args, kwargs, ("type", "provider", "id"))
        provider, model_id = bound.get("provider"), bound.get("id")
        if not isinstance(provider, str) or not isinstance(model_id, str):
            given = ", ".join(_describe_value(value) for value in bound.values())
            raise TypeError(
                f"models.get_model_of_type(type, provider, id) expects three strings, got ({given}). The provider "
                'and the id are separate arguments, for example models.get_model_of_type("classifier", "typesafe", '
                '"jev-latest").'
            )
        model = registry.get_model_of_type(_to_model_type(bound.get("type")), provider, model_id)
        return None if model is None else _to_model_info(model)

    async def classify(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        bound = _bind("models.classify", args, kwargs, ("model", "context"))
        return await run_model_call(
            "models.classify",
            "classifier",
            bound,
            _check_classifier_context,
            lambda resolved, context: registry.classify(resolved, context, ClassifierOptions(cancel=cancel)),
        )

    async def generate_images(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        bound = _bind("models.generate_images", args, kwargs, ("model", "context"))

        async def run(resolved: AnyModel, context: ImagesContext) -> Any:
            result = await registry.generate_images(resolved, context, ImagesOptions(cancel=cancel))
            state.add_generated_images(sum(1 for block in result.output if block.type == "image"))
            return result

        return await run_model_call("models.generate_images", "image", bound, _check_images_context, run)

    implementations = {
        "models.get_models_of_type": get_models_of_type,
        "models.get_available_of_type": get_available_of_type,
        "models.get_model_of_type": get_model_of_type,
        "models.classify": classify,
        "models.generate_images": generate_images,
    }
    signatures = global_signatures(True)
    return [
        CodemodeGlobal(name=name, execute=execute, signature=signatures[name])
        for name, execute in implementations.items()
    ]
