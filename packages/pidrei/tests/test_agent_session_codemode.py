"""Mirror of pi's suite/agent-session-codemode.test.ts, with the scripts in
Python.

The codemode extension opens its Monty pool at `session_start` (which the
harness emits through `bind_extensions`) and closes it at `session_shutdown`,
which the `harnesses` fixture emits for every harness it created, so no pool
outlives its test.

Translations from pi's cases:
- A script's value is its last expression line, not a `return` (Monty's type
  checker rejects a top-level `return`).
- Failures carry a CPython traceback (`<python-input-1>", line 3`) where pi's
  carry a V8 stack (`codemode.js:3`).
- pi's memory limit is catchable inside the script; Monty's is not, so the case
  checks that the script fails with a `MemoryError` it never caught.
- Cases about invalid arguments to `models.*` run with type checking off: with
  it on, the checker rejects them before the script runs.
"""

import base64
import json
import re
import threading

import pytest
import tonio.colored as tonio
from tonio.colored import fs

from pidrei.core.extensions import ToolDefinition
from pidrei.core.extensions.runner import emit_session_shutdown_event
from pidrei.core.extensions.types import ToolNamespace
from pidrei.extensions.codemode import (
    CODEMODE_DOCS_PATH,
    CODEMODE_STORE_ENTRY_TYPE,
    create_codemode_extension,
    create_codemode_tool,
)
from pidrei.extensions.codemode.execute import read_codemode_store
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import (
    AssistantImages,
    ClassifierBoolAnswer,
    ClassifierResult,
    ImageContent,
    TextContent,
    Usage,
    UsageCost,
)
from pidrei_ai.utils.transcript import get_current_system_prompt, get_current_tools
from pidrei_codemode import CodemodePool

from .harness import create_harness


TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

EMPTY_SCHEMA = {"type": "object", "properties": {}}
ECHO_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string", "description": "Text to echo"}},
    "required": ["text"],
}
STATS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"files": {"type": "number"}, "names": {"type": "array", "items": {"type": "string"}}},
    "required": ["files", "names"],
}


TINY_PNG_LABEL = re.compile(r"^\[Image saved to (\S+\.png) \(image/png, \d+B\)\]$")


async def check_saved_images(text: str) -> str:
    """Replace the `[Image saved to ...]` labels in `text` with `<saved>` after
    checking that each file holds the tiny PNG, and remove the files."""
    lines = []
    for line in text.split("\n"):
        match = TINY_PNG_LABEL.match(line)
        if match is None:
            lines.append(line)
            continue
        path = fs.Path(match[1])
        try:
            assert base64.b64encode(await path.read_bytes()).decode() == TINY_PNG_BASE64
        finally:
            await path.unlink(missing_ok=True)
        lines.append("<saved>")
    return "\n".join(lines)


def usage(input_tokens: int, cost: float) -> Usage:
    return Usage(input=input_tokens, total_tokens=input_tokens, cost=UsageCost(input=cost, total=cost))


async def _echo(_id, params, *_rest):
    return AgentToolResult(content=[TextContent(text=f"echo: {params['text']}")], details={})


async def _stats(*_rest):
    return AgentToolResult(
        content=[TextContent(text="2 files")], details={}, structured_content={"files": 2, "names": ["a", "b"]}
    )


async def _screenshot(*_rest):
    return AgentToolResult(
        content=[TextContent(text="captured"), ImageContent(data=TINY_PNG_BASE64, mime_type="image/png")], details={}
    )


async def register_tools(pi) -> None:
    """Registered by an extension so they run with the session's tool
    context, next to the codemode extension."""
    for definition in (
        ToolDefinition(
            name="echo",
            label="Echo",
            description="Echo text back.\n\nSecond paragraph.",
            parameters=ECHO_SCHEMA,
            execute=_echo,
        ),
        ToolDefinition(
            name="stats",
            label="Stats",
            description="Return structured stats",
            parameters=EMPTY_SCHEMA,
            output_schema=STATS_OUTPUT_SCHEMA,
            execute=_stats,
        ),
        ToolDefinition(
            name="screenshot",
            label="Screenshot",
            description="Return a screenshot",
            parameters=EMPTY_SCHEMA,
            execute=_screenshot,
        ),
    ):
        pi.register_tool(definition)


@pytest.fixture
async def harnesses():
    """Harnesses a test created; each session is shut down afterwards, which
    closes its codemode pool."""
    created: list = []
    try:
        yield created
    finally:
        for harness in created:
            await emit_session_shutdown_event(
                harness.session.extension_runner, {"type": "session_shutdown", "reason": "quit"}
            )
            harness.cleanup()


@pytest.fixture
async def pool():
    """A pool for the bare codemode tool, which the caller owns."""
    pool = await CodemodePool()
    try:
        yield pool
    finally:
        await pool.close()


async def setup(harnesses, extension_factories=(), *, codemode=None, tools=None, settings=None):
    harness = await create_harness(
        settings=settings,
        initial_active_tool_names=tools if tools is not None else ["codemode"],
        extension_factories=[codemode or create_codemode_extension(), *extension_factories],
    )
    harnesses.append(harness)
    return harness


def codemode_result(harness):
    results = [
        message
        for message in harness.session.messages
        if message.role == "toolResult" and message.tool_name == "codemode"
    ]
    return results[-1]


def result_text(message) -> str:
    """The output after the script header, which is checked on the way."""
    header, *items = message.content
    assert header.type == "text"
    lines = header.text.split("\n")
    assert lines[0] in ("Script completed", "Script failed")
    assert lines[1].startswith("Wall time ") and lines[1].endswith(" seconds")
    assert lines[2:] == ["Output:", ""]
    return "\n".join(item.text if item.type == "text" else f"<{item.type}>" for item in items)


async def run(harness, code: str):
    harness.set_responses(
        [
            faux_assistant_message([faux_tool_call("codemode", {"code": code})], stop_reason="toolUse"),
            faux_assistant_message("done"),
        ]
    )
    await harness.session.prompt("go")
    return codemode_result(harness)


def call_rows(result) -> list:
    return result.details.calls


# --- AgentSession codemode tool ----------------------------------------------------


@pytest.mark.tonio
async def test_presents_callable_tools_per_codemode_mode(harnesses):
    harness = await setup(harnesses, [register_tools])

    def description(name: str) -> str:
        return next((tool.description for tool in harness.session.agent.state.tools if tool.name == name), "")

    request_tools: list[list[str]] = []
    request_prompts: list[str] = []

    async def record(context, *_rest):
        request_tools.append([tool.name for tool in get_current_tools(context.messages)])
        request_prompts.append(get_current_system_prompt(context.messages))
        return faux_assistant_message("ok")

    # on: declared tools say how scripts call them and are not listed again in codemode.
    harness.session.set_active_tools_by_name(["read", "echo", "codemode"])
    assert "Codemode: `await tools.echo(...)` takes the parameters as keyword arguments and returns" in description(
        "echo"
    )
    assert "codemode tool declaration:" not in description("echo")
    assert "### `echo`" not in description("codemode")
    harness.set_responses([record])
    await harness.session.prompt("on")
    assert {"read", "echo", "codemode"} <= set(request_tools[0])
    assert "\n- read: " in request_prompts[0]

    # only: codemode lists echo, which stays active but is left out of requests.
    harness.settings_manager.apply_overrides({"codemode": {"mode": "only"}})
    harness.session.set_active_tools_by_name(["read", "echo", "codemode"])
    assert "Codemode: `await tools.echo" not in description("echo")
    assert "### `echo`" in description("codemode")
    assert "### `stats`" not in description("codemode")
    harness.set_responses([record])
    await harness.session.prompt("only")
    assert "codemode" in request_tools[1]
    assert "echo" not in request_tools[1]
    assert "read" not in request_tools[1]
    # The prompt's tool list matches the declarations: hidden tools are not listed (#10192).
    assert "\n- read: " not in request_prompts[1]
    assert "\n- codemode: " in request_prompts[1]
    assert "\n- read: " not in harness.session.system_prompt
    # Hidden tools' guidelines move from the rules to their codemode sections (#10343).
    assert "Use read to examine files" not in request_prompts[1]
    assert "- Use read to examine files instead of cat or sed." in description("codemode")

    # Without codemode, tools keep their plain descriptions.
    harness.session.set_active_tools_by_name(["echo"])
    assert description("echo") == "Echo text back.\n\nSecond paragraph."


# #10343
@pytest.mark.tonio
async def test_shows_the_guidelines_of_tools_that_do_not_fit_the_inline_budget_through_describe_tool(harnesses):
    harness = await setup(harnesses, [register_tools])
    harness.settings_manager.apply_overrides({"codemode": {"mode": "only", "inlineBudget": 0}})
    harness.session.set_active_tools_by_name(["read", "codemode"])
    codemode = next(tool for tool in harness.session.agent.state.tools if tool.name == "codemode")
    assert "### `read`" not in codemode.description

    result = await run(harness, "text(await describe_tool('read'))")

    assert "- Use read to examine files instead of cat or sed." in result_text(result)


@pytest.mark.tonio
async def test_runs_nested_calls_in_parallel_and_returns_only_the_script_result(harnesses):
    harness = await setup(harnesses, [register_tools])

    result = await run(
        harness,
        """import asyncio
a, b, stats = await asyncio.gather(tools.echo(text='one'), tools.echo(text='two'), tools.stats())
print('files', stats['files'])
text(','.join(tool['name'] for tool in ALL_TOOLS))
{'a': a, 'b': b, 'names': stats['names']}""",
    )

    assert result.is_error is False
    assert result_text(result) == 'files 2\necho,stats,screenshot\n{"a":"echo: one","b":"echo: two","names":["a","b"]}'
    assert sorted((row.name, row.status) for row in call_rows(result)) == [
        ("echo", "ok"),
        ("echo", "ok"),
        ("stats", "ok"),
    ]
    assert all(row.id.startswith(f"{result.tool_call_id}/") for row in call_rows(result))
    # Nested calls never become transcript tool results; their events carry the parent id.
    assert len([message for message in harness.session.messages if message.role == "toolResult"]) == 1


@pytest.mark.tonio
async def test_routes_nested_calls_through_extension_hooks(harnesses):
    async def hooks(pi) -> None:
        async def on_tool_call(event, _ctx):
            if event["toolName"] == "echo" and event["input"]["text"] == "forbidden":
                return {"block": True, "reason": "echo of forbidden text is blocked"}
            return None

        async def on_tool_result(event, _ctx):
            return {"content": [TextContent(text="redacted")]} if event["toolName"] == "stats" else None

        pi.on("tool_call", on_tool_call)
        pi.on("tool_result", on_tool_result)

    harness = await setup(harnesses, [register_tools, hooks])

    result = await run(
        harness,
        """blocked = None
try:
    await tools.echo(text='forbidden')
except Exception as error:
    blocked = str(error)
stats = await tools.stats()
{'blocked': blocked, 'stats': stats}""",
    )

    # Replacing content without replacing structured content drops the structured result.
    assert json.loads(result_text(result)) == {"blocked": "echo of forbidden text is blocked", "stats": "redacted"}
    assert [row.status for row in call_rows(result)] == ["error", "ok"]


@pytest.mark.tonio
async def test_adds_the_usage_of_nested_results_to_the_codemode_result(harnesses):
    async def billed_tool(pi) -> None:
        async def billed(*_rest):
            return AgentToolResult(content=[TextContent(text="ran")], details={}, usage=usage(100, 0.25))

        pi.register_tool(
            ToolDefinition(
                name="billed", label="Billed", description="Run a model", parameters=EMPTY_SCHEMA, execute=billed
            )
        )

    harness = await setup(harnesses, [register_tools, billed_tool])

    result = await run(harness, "await tools.billed()\nawait tools.billed()\nawait tools.echo(text='x')")

    assert (result.usage.input, result.usage.total_tokens, result.usage.cost.total) == (200, 200, 0.5)
    # The usage is persisted with the result, so session totals count it.
    persisted = next(
        entry["message"]
        for entry in harness.session_manager.get_entries()
        if entry["type"] == "message" and entry["message"].role == "toolResult"
    )
    assert persisted.usage == result.usage
    assert harness.session.get_session_stats().cost == 0.5


@pytest.mark.tonio
async def test_keeps_structured_content_that_tool_result_handlers_replace_along_with_the_content(harnesses):
    async def hooks(pi) -> None:
        async def replace_stats(event, _ctx):
            if event["toolName"] == "stats":
                return {"content": [TextContent(text="0 files")], "structuredContent": {"files": 0, "names": []}}
            return None

        # A later handler that only touches details keeps what the first one set.
        async def audit_stats(event, _ctx):
            return {"details": {"audited": True}} if event["toolName"] == "stats" else None

        pi.on("tool_result", replace_stats)
        pi.on("tool_result", audit_stats)

    harness = await setup(harnesses, [register_tools, hooks])

    result = await run(harness, "await tools.stats()")

    assert json.loads(result_text(result)) == {"files": 0, "names": []}


# Saved images: https://github.com/earendil-works/pi/issues/10310
@pytest.mark.tonio
async def test_attaches_only_the_images_the_script_passes_to_image_in_output_order_each_after_its_saved_path(
    harnesses,
):
    harness = await setup(harnesses, [register_tools])

    result = await run(
        harness,
        f"""# Tools without an output schema return their text; images are not passed on.
shot = await tools.screenshot()
text(shot)
image('data:image/png;base64,{TINY_PNG_BASE64}')
image('data:image/png;base64,{TINY_PNG_BASE64}')
text('after')""",
    )

    # The same image shown twice is saved once, so both labels name one file.
    lines = result_text(result).split("\n")
    assert lines == ["captured", lines[1], "<image>", lines[1], "<image>", "after"]
    assert await check_saved_images(lines[1]) == "<saved>"
    assert result.content[3] == ImageContent(data=TINY_PNG_BASE64, mime_type="image/png")


@pytest.mark.tonio
async def test_reports_script_failures_as_results_that_keep_partial_output_and_the_calls_that_ran(harnesses):
    harness = await setup(harnesses, [register_tools])

    result = await run(harness, "text('partial')\nawait tools.echo(text='x')\nraise Exception('boom')")

    assert result.is_error is True
    assert result.content[0].text.startswith("Script failed\n")
    text = result_text(result)
    assert text.startswith("partial\nScript error:\nTraceback (most recent call last):\n")
    assert '", line 3, in <module>' in text
    assert "\nException: boom\n" in text
    assert text.endswith("Tool calls made before the failure (they are not undone): echo (ok)")
    assert [row.name for row in call_rows(result)] == ["echo"]


@pytest.mark.tonio
async def test_rejects_a_script_that_fails_the_type_check_before_any_tool_runs(harnesses):
    """pidrei-only: scripts are checked against the declarations first."""
    harness = await setup(harnesses, [register_tools])

    result = await run(harness, "await tools.echo(text='x')\nawait tools.echo(txt='y')")

    assert result.is_error is True
    text = result_text(result)
    assert text.startswith("Script error:\nThe script did not run: type checking failed.\n")
    assert "unknown-argument" in text
    assert text.endswith("No tool calls were made.")
    assert call_rows(result) == []


@pytest.mark.tonio
async def test_codemode_type_check_false_runs_scripts_unchecked(harnesses):
    """pidrei-only: the `codemode.typeCheck` setting turns the check off, and
    the description no longer promises it."""
    harness = await setup(harnesses, [register_tools], settings={"codemode": {"typeCheck": False}})
    harness.session.set_active_tools_by_name(["codemode"])
    codemode = next(tool for tool in harness.session.agent.state.tools if tool.name == "codemode")
    assert "type-checked" not in codemode.description

    # The checker would reject the unknown member; unchecked, it runs up to it.
    result = await run(harness, "text('ran')\nawait tools.nothing()")

    assert result.is_error is True
    assert result_text(result).startswith("ran\nScript error:\n")
    assert "tools.nothing does not exist." in result_text(result)


@pytest.mark.tonio
async def test_finds_tools_by_identifier_and_calls_them_by_computed_name(harnesses):
    """pidrei-only (pi has no session-level case): the discovery globals and
    `call_tool` over namespaced and deferred tools, which the description does
    not list."""
    github = ToolNamespace(name="github", description="GitHub issues and pull requests", instructions="Be nice.")

    async def namespaced_tools(pi) -> None:
        async def search(_id, params, *_rest):
            return AgentToolResult(content=[TextContent(text=f"found {params['query']}")], details={})

        pi.register_tool(
            ToolDefinition(
                name="github__search-issues",
                label="Search issues",
                description="Search issues and pull requests.",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                exposure="deferred",
                namespace=github,
                execute=search,
            )
        )

    harness = await setup(harnesses, [register_tools, namespaced_tools])
    harness.session.set_active_tools_by_name(["codemode"])
    codemode = next(tool for tool in harness.session.agent.state.tools if tool.name == "codemode")
    assert "github__search" not in codemode.description

    result = await run(
        harness,
        """{
    'found': [tool['name'] for tool in await search_tools('pull requests')],
    'in_namespace': [tool['name'] for tool in await search_tools('search', namespace='github')],
    'described': (await describe_tool('github__search_issues') or '').split('\\n')[0],
    'raw_name': await describe_tool('github__search-issues'),
    'namespace': await describe_namespace('github'),
    'has': [has_tool('github__search_issues'), has_tool('github__search-issues')],
    'called': await call_tool('github__search_issues', query='bugs'),
}""",
    )

    assert result.is_error is False
    assert json.loads(result_text(result)) == {
        "found": ["github__search_issues"],
        "in_namespace": ["github__search_issues"],
        "described": "Search issues and pull requests.",
        "raw_name": None,
        "namespace": {
            "name": "github",
            "description": "GitHub issues and pull requests",
            "instructions": "Be nice.",
            "tools": ["github__search_issues"],
        },
        "has": [True, False],
        "called": "found bugs",
    }
    # A call by computed name is a nested call like any other.
    assert [(row.name, row.status) for row in call_rows(result)] == [("github__search-issues", "ok")]


@pytest.mark.tonio
async def test_fails_scripts_as_a_sandbox_error_once_the_session_shut_down(harnesses):
    """pidrei-only: the extension's pool lives from `session_start` to
    `session_shutdown`."""
    harness = await setup(harnesses)
    await emit_session_shutdown_event(harness.session.extension_runner, {"type": "session_shutdown", "reason": "quit"})

    result = await run(harness, "1")

    assert result.is_error is True
    assert result_text(result).startswith("Script error:\nScript sandbox failed: ")


# --- codemode options and store --------------------------------------------------

INCREMENT = "next_value = (load('count') or 0) + 1\nstore('count', next_value)\nnext_value"


def store_entries(harness) -> list:
    return [
        entry["data"]
        for entry in harness.session_manager.get_branch()
        if entry["type"] == "custom" and entry["customType"] == CODEMODE_STORE_ENTRY_TYPE
    ]


@pytest.mark.tonio
async def test_applies_the_timeout_ms_option_and_rejects_invalid_options(harnesses):
    harness = await setup(harnesses)

    timed_out = await run(harness, '# @options: {"timeout_ms": 200}\nwhile True:\n    pass')
    assert timed_out.is_error is True
    assert "Script error:\nScript timed out" in result_text(timed_out)

    invalid = await run(harness, '# @options: {"yield": 1}\ntext(1)')
    assert invalid.is_error is True
    assert invalid.content == [
        TextContent(text="@options only supports `max_output_tokens` and `timeout_ms`; got `yield`")
    ]


@pytest.mark.tonio
async def test_limits_script_memory_so_runaway_allocations_end_the_script(harnesses):
    """pi's limit raises inside the script, which can catch it; Monty's ends
    the script with a `MemoryError` the script cannot catch."""
    harness = await setup(harnesses)

    result = await run(
        harness,
        """# @options: {"timeout_ms": 30000}
a = []
try:
    while True:
        a.append('x' * (1 << 20) + str(len(a)))
except MemoryError:
    text('caught')""",
    )

    assert result.is_error is True
    text = result_text(result)
    assert "MemoryError" in text
    assert "caught" not in text


@pytest.mark.tonio
async def test_truncates_output_to_the_token_budget_and_spills_the_full_text(harnesses):
    harness = await setup(harnesses)

    result = await run(
        harness,
        f"""# @options: {{"max_output_tokens": 10}}
for i in range(100):
    text(f'row {{i}}')
image('data:image/png;base64,{TINY_PNG_BASE64}')""",
    )

    path = result.details.full_output_path
    assert path is not None
    try:
        text = result_text(result)
        assert text.startswith("Warning: truncated output")
        assert "row 0\n" in text
        assert "tokens truncated" in text
        assert "row 99\n" in text
        assert "row 50\n" not in text
        assert f"[Full output: {path} (read with offset/limit)]" in text
        # Images follow the truncated text, each after the path it was saved to.
        assert result.content[-1] == ImageContent(data=TINY_PNG_BASE64, mime_type="image/png")
        assert await check_saved_images(text.split("\n")[-2]) == "<saved>"
        assert await fs.Path(path).read_text() == "\n".join(f"row {i}" for i in range(100))
    finally:
        await fs.Path(path).unlink(missing_ok=True)

    small = await run(harness, "{'ok': True}")
    assert small.details.full_output_path is None


@pytest.mark.tonio
async def test_resolves_bash_calls_to_structured_results_also_for_non_zero_exit_codes(harnesses):
    harness = await setup(harnesses, tools=["codemode", "bash"])

    result = await run(
        harness,
        """import json
r = await tools.bash(command='echo out; exit 3')
text(json.dumps([r['output'], r['exit_code'], type(r['wall_time_seconds']).__name__]))""",
    )

    assert result.is_error is False
    assert json.loads(result_text(result)) == ["out\n", 3, "float"]


# https://github.com/earendil-works/pi/issues/10251
@pytest.mark.tonio
async def test_resolves_read_calls_to_text_for_text_files_and_to_image_blocks_that_image_shows(harnesses):
    harness = await setup(harnesses, tools=["codemode", "read"])
    await (fs.Path(harness.temp_dir) / "notes.txt").write_text("hello")
    await (fs.Path(harness.temp_dir) / "pixel.png").write_bytes(base64.b64decode(TINY_PNG_BASE64))

    # `read` returns `str | <image block>`; the type check needs the narrowing pi's script does without.
    result = await run(
        harness,
        """text(await tools.read(path='notes.txt'))
shot = await tools.read(path='pixel.png')
assert not isinstance(shot, str)
text(shot['note'])
image(shot)""",
    )

    assert result.is_error is False
    assert await check_saved_images(result_text(result)) == "hello\nRead image file [image/png]\n<saved>\n<image>"
    assert result.content[-1] == ImageContent(data=TINY_PNG_BASE64, mime_type="image/png")


@pytest.mark.tonio
async def test_persists_store_writes_as_custom_entries_for_later_calls(harnesses):
    harness = await setup(harnesses)

    assert result_text(await run(harness, INCREMENT)) == "1"
    assert result_text(await run(harness, INCREMENT)) == "2"
    assert store_entries(harness) == [{"set": {"count": 1}, "delete": []}, {"set": {"count": 2}, "delete": []}]
    appended = [
        event
        for event in harness.events_of_type("entry_appended")
        if event.entry["type"] == "custom" and event.entry["customType"] == CODEMODE_STORE_ENTRY_TYPE
    ]
    assert len(appended) == 2

    assert result_text(await run(harness, "store('count', None)\nload('count') is None")) == "true"
    assert store_entries(harness)[-1] == {"set": {}, "delete": ["count"]}


@pytest.mark.tonio
async def test_appends_nothing_for_failed_scripts_or_scripts_without_writes(harnesses):
    harness = await setup(harnesses)

    assert (await run(harness, "store('count', 5)\nraise Exception('boom')")).is_error is True
    assert (await run(harness, "load('count') or 'missing'")).is_error is False
    assert store_entries(harness) == []


@pytest.mark.tonio
async def test_runs_without_a_session_starting_from_an_empty_store(pool):
    tool = create_codemode_tool(pool)

    result = await tool.execute("direct", {"code": INCREMENT})

    assert result.content[1] == TextContent(text="1")


@pytest.mark.tonio
async def test_loads_the_values_written_on_the_current_branch(harnesses):
    harness = await setup(harnesses)
    await run(harness, INCREMENT)
    first_prompt = next(entry for entry in harness.session_manager.get_branch() if entry["type"] == "message")
    assert result_text(await run(harness, INCREMENT)) == "2"

    # Branch from the first prompt: the store entries written after it are on another path.
    harness.session_manager.branch(first_prompt["id"])
    assert result_text(await run(harness, INCREMENT)) == "1"


def test_folds_store_entries_from_the_root_ignoring_malformed_data():
    def entry(data, custom_type=CODEMODE_STORE_ENTRY_TYPE) -> dict:
        return {"type": "custom", "customType": custom_type, "data": data, "id": "x", "parentId": None}

    assert read_codemode_store(
        [
            entry({"set": {"a": 1, "b": {"c": 2}}, "delete": []}),
            entry({"set": {"a": 3}, "delete": ["b"]}),
            entry({"set": {"z": 1}}),
            entry({"set": {"other": 1}, "delete": []}, "other-extension"),
        ]
    ) == {"a": 3}


# --- codemode models -------------------------------------------------------------

QUESTIONS = "{'approved': {'type': 'bool', 'instructions': 'Approval?', 'criteria': {'true': 'yes', 'false': 'no'}}}"


def _classifier_definition() -> dict:
    return {
        "type": "classifier",
        "id": "judge",
        "name": "Judge",
        "api": "test-classifier",
        "baseUrl": "https://classifier.test/v1",
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 1000,
        "headers": {"X-Secret": "hunter2"},
    }


def _image_definition() -> dict:
    return {
        "type": "image",
        "id": "painter",
        "name": "Painter",
        "api": "test-images",
        "baseUrl": "https://images.test/v1",
        "input": ["text", "image"],
        "output": ["text", "image"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    }


class ScorerProvider:
    """The `scorer` provider's classifier and image implementations, recording
    what reached them. With `saturate_at` set, classifications park until that
    many are in flight at once (bounded), so a test sees what the limit lets
    through."""

    def __init__(self) -> None:
        self.observed: list[dict] = []
        self.image_requests: list[dict] = []
        self.saturate_at: int | None = None
        self.saturated = tonio.Event()
        self._guard = threading.Lock()
        self.active = 0
        self.max_active = 0

    async def classify(self, model, context, options=None):
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.saturate_at is not None and self.active >= self.saturate_at:
                self.saturated.set()
        if self.saturate_at is not None:
            await self.saturated.wait(5)
        with self._guard:
            self.active -= 1
        text = context.state.get("text")
        self.observed.append({"base_url": model.base_url, "api_key": options.api_key, "text": text})
        base = {"api": model.api, "provider": model.provider, "model": model.id, "timestamp": 0}
        if text == "explode":
            return ClassifierResult(**base, answers={}, stop_reason="error", error_message="classifier exploded")
        return ClassifierResult(
            **base,
            answers={"approved": ClassifierBoolAnswer(probability=0.9 if text == "good" else 0.1)},
            usage=usage(300, 0.001),
            stop_reason="stop",
        )

    async def generate_images(self, model, context, options=None):
        self.image_requests.append({"base_url": model.base_url, "api_key": options.api_key, "input": context.input})
        prompt = next((block.text for block in context.input if block.type == "text"), None)
        base = {"api": model.api, "provider": model.provider, "model": model.id, "timestamp": 0}
        if prompt == "explode":
            return AssistantImages(**base, output=[], stop_reason="error", error_message="painter exploded")
        return AssistantImages(
            **base,
            output=[TextContent(text=f"painted {prompt}"), ImageContent(data=TINY_PNG_BASE64, mime_type="image/png")],
            usage=usage(100, 0.04),
            stop_reason="stop",
        )


async def setup_models(harnesses, *, type_check=None):
    harness = await setup(harnesses, codemode=create_codemode_extension(type_check=type_check))
    provider = ScorerProvider()
    harness.session.model_runtime.register_provider(
        "scorer",
        {
            "apiKey": "secret-key",
            "models": [_classifier_definition(), _image_definition()],
            "images": {"test-images": provider},
            "classifiers": {"test-classifier": provider},
        },
    )
    harness.session.set_active_tools_by_name(["codemode"])
    return harness, provider


@pytest.mark.tonio
async def test_declares_models_only_for_the_sessions_own_codemode_tool(harnesses, pool):
    harness, _provider = await setup_models(harnesses)
    codemode = next(tool for tool in harness.session.agent.state.tools if tool.name == "codemode")
    # The description names the models globals and points to the docs for the API.
    assert "`models`: classifiers and image generation" in codemode.description
    assert CODEMODE_DOCS_PATH in codemode.description

    overridden = await create_harness(tools=[create_codemode_tool(pool)])
    harnesses.append(overridden)
    plain = next(tool for tool in overridden.session.agent.state.tools if tool.name == "codemode")
    assert "`models`" not in plain.description


@pytest.mark.tonio
async def test_lists_models_and_classifies_with_catalog_auth_ignoring_script_supplied_fields(harnesses):
    harness, provider = await setup_models(harnesses, type_check=False)
    # The first four calls park until all four are in flight: the limit must let four through at once.
    provider.saturate_at = 4
    result = await run(
        harness,
        f"""import asyncio
[model] = await models.get_available_of_type('classifier', 'scorer')
listed = await models.get_models_of_type('classifier')
same = await models.get_model_of_type('classifier', 'scorer', 'judge')
texts = ['good', 'bad', 'good', 'bad', 'good', 'bad']
results = await asyncio.gather(
    *[models.classify({{**model, 'baseUrl': 'https://evil.test'}}, {{'state': {{'text': t}}, 'questions': {QUESTIONS}}}) for t in texts]
)
{{
    'id': model['id'],
    'headers': 'headers' in model,
    'listed': any(entry['provider'] == 'scorer' and entry['id'] == 'judge' for entry in listed),
    'same': same['id'],
    'missing': (await models.get_model_of_type('classifier', 'scorer', 'nope')) is None,
    'probabilities': [r['answers']['approved']['probability'] for r in results],
    'cost': results[0]['usage']['cost']['total'],
}}""",
    )

    assert result.is_error is False
    assert json.loads(result_text(result)) == {
        "id": "judge",
        "headers": False,
        "listed": True,
        "same": "judge",
        "missing": True,
        "probabilities": [0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
        "cost": 0.001,
    }
    assert len(provider.observed) == 6
    assert all(
        entry["base_url"] == "https://classifier.test/v1" and entry["api_key"] == "secret-key"
        for entry in provider.observed
    )
    # Six classifications with at most four in flight.
    assert provider.max_active == 4
    assert [(row.name, row.args, row.status, row.cost) for row in call_rows(result)] == [
        ("models.classify", "scorer/judge", "ok", 0.001)
    ] * 6
    # The classifications' usage becomes the codemode result's usage.
    assert result.usage.input == 1800
    assert result.usage.cost.total == pytest.approx(0.006)
    assert harness.session.get_session_stats().cost == pytest.approx(0.006)


@pytest.mark.tonio
async def test_generates_images_with_catalog_auth_and_attaches_them_through_image(harnesses):
    harness, provider = await setup_models(harnesses, type_check=False)

    result = await run(
        harness,
        f"""[model] = await models.get_available_of_type('image', 'scorer')
reference = {{'type': 'image', 'data': '{TINY_PNG_BASE64}', 'mimeType': 'image/png'}}
generated = await models.generate_images(
    {{**model, 'baseUrl': 'https://evil.test'}},
    {{'input': [{{'type': 'text', 'text': 'a fox'}}, reference]}},
)
for block in generated['output']:
    if block['type'] == 'image':
        image(block)
    else:
        text(block['text'])
failed = await models.generate_images(model, {{'input': [{{'type': 'text', 'text': 'explode'}}]}})
try:
    await models.generate_images({{'provider': 'scorer', 'id': 'judge'}}, {{'input': []}})
    wrong_type = 'ok'
except Exception as error:
    wrong_type = str(error)
{{
    'id': model['id'],
    'stopReason': generated['stopReason'],
    'failed': [failed['stopReason'], failed['errorMessage']],
    'wrongType': wrong_type,
}}""",
    )

    assert result.is_error is False
    first, saved, image_marker, *rest = (await check_saved_images(result_text(result))).split("\n")
    assert first == "painted a fox"
    assert saved == "<saved>"
    assert image_marker == "<image>"
    assert json.loads("\n".join(rest)) == {
        "id": "painter",
        "stopReason": "stop",
        "failed": ["error", "painter exploded"],
        "wrongType": '"scorer/judge" is a classifier model, not an image model. List the image models you can use '
        'with models.get_available_of_type("image").',
    }
    assert result.content[3] == ImageContent(data=TINY_PNG_BASE64, mime_type="image/png")
    assert [(request["base_url"], request["api_key"]) for request in provider.image_requests] == [
        ("https://images.test/v1", "secret-key"),
        ("https://images.test/v1", "secret-key"),
    ]
    assert provider.image_requests[0]["input"] == [
        TextContent(text="a fox"),
        ImageContent(data=TINY_PNG_BASE64, mime_type="image/png"),
    ]
    assert [(row.name, row.args, row.status, row.cost, row.error) for row in call_rows(result)] == [
        ("models.generate_images", "scorer/painter", "ok", 0.04, None),
        ("models.generate_images", "scorer/painter", "error", None, "painter exploded"),
    ]
    assert result.usage.cost.total == pytest.approx(0.04)
    assert harness.session.get_session_stats().cost == pytest.approx(0.04)


@pytest.mark.tonio
async def test_notes_generated_images_that_the_script_did_not_show(harnesses):
    harness, _provider = await setup_models(harnesses)

    result = await run(
        harness,
        """[model] = await models.get_available_of_type('image', 'scorer')
generated = await models.generate_images(model, {'input': [{'type': 'text', 'text': 'a fox'}]})
generated['stopReason']""",
    )

    assert result.is_error is False
    assert result_text(result) == (
        "stop\nNote: models.generate_images() returned 1 image that the script did not show. Show each image block "
        "of result['output'] with image(block)."
    )


@pytest.mark.tonio
async def test_reports_provider_errors_as_results_and_invalid_arguments_as_exceptions(harnesses):
    harness, _provider = await setup_models(harnesses, type_check=False)

    result = await run(
        harness,
        f"""model = await models.get_model_of_type('classifier', 'scorer', 'judge')
failed = await models.classify(model, {{'state': {{'text': 'explode'}}, 'questions': {QUESTIONS}}})

async def attempt(call):
    try:
        await call
        return 'ok'
    except Exception as error:
        return str(error)

{{
    'failed': [failed['stopReason'], failed['errorMessage']],
    'badType': await attempt(models.get_models_of_type('video')),
    'unknown': await attempt(models.classify({{'provider': 'scorer', 'id': 'nope'}}, {{}})),
    'noModel': await attempt(models.classify('judge', {{}})),
    'noneModel': await attempt(models.classify(None, {{}})),
    'noState': await attempt(models.classify(model, {{'questions': {QUESTIONS}}})),
    'badQuestion': await attempt(
        models.classify(model, {{'state': {{}}, 'questions': {{'kind': {{'type': 'choice', 'instructions': 'Kind?', 'criteria': ['a', 'b']}}}}}})
    ),
    'badImage': await attempt(models.generate_images({{'provider': 'scorer', 'id': 'painter'}}, {{'prompt': 'a fox'}})),
    'badSplit': await attempt(models.get_model_of_type('classifier', 'scorer/judge')),
}}""",
    )

    assert result.is_error is False
    value = json.loads(result_text(result))
    assert value["failed"] == ["error", "classifier exploded"]
    assert 'Unknown model type "video"' in value["badType"]
    assert value["unknown"] == (
        'Unknown classifier model "scorer/nope". List the classifier models you can use with '
        'models.get_available_of_type("classifier").'
    )
    assert "models.classify() expects a classifier model as its first argument, got a str." in value["noModel"]
    assert "models.get_model_of_type() returns None for an unknown provider or id." in value["noneModel"]
    assert "models.classify() context['state'] must be a dict, got None." in value["noState"]
    assert "codemode.md" in value["noState"]
    assert (
        "context['questions']['kind'] is a \"choice\" question, so criteria must map each label to its meaning."
        in value["badQuestion"]
    )
    assert (
        "models.generate_images() context['input'] must be a non-empty list of blocks, got None." in value["badImage"]
    )
    assert "The provider and the id are separate arguments" in value["badSplit"]
    assert [(row.name, row.status, row.error) for row in call_rows(result)] == [
        ("models.classify", "error", "classifier exploded")
    ]
    assert result.usage is None
