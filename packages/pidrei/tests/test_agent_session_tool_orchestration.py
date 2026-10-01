"""Mirror of pi's suite/agent-session-tool-orchestration.test.ts, plus the core
cases of suite/agent-session-codemode.test.ts.

pi's "registers codemode and tool_search inactive until they are named" case
needs the codemode and tool-search extensions, which port later with codemode.
The codemode suite drives the mechanisms under test (nested calls in parallel,
hooks on nested calls, nested usage, structured content through the hooks,
bash's structured result) from codemode scripts; here a Python tool built on
`ctx.execute_tool()` drives them, and the script-only cases (declarations per
`codemode.mode`, images, script errors, the store, models) port with codemode.
"""

import pytest
import tonio.colored as tonio

from pidrei.core.extensions import ToolDefinition
from pidrei.core.extensions.types import ToolLoadoutChanges
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import TextContent, Usage, UsageCost
from pidrei_ai.utils.transcript import get_current_tools

from .harness import create_harness


EMPTY_SCHEMA = {"type": "object", "properties": {}}
TEXT_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


def text_of(outcome) -> str:
    return outcome.result.content[0].text


async def orchestrator_extension(pi) -> None:
    """A tool that calls other tools, built only on the extension API: its own
    name, exposure, loadout hook, and ctx.execute_tool(). Codemode and tool
    search use the same mechanisms."""

    async def echo(_id, params, *_rest):
        return AgentToolResult(content=[TextContent(text=f"echo: {params['text']}")], details={})

    async def helper(*_rest):
        return AgentToolResult(content=[TextContent(text="helped")], details={})

    async def run_tools(_id, _params, _cancel, _on_update, ctx):
        helped = await ctx.execute_tool("helper", {})
        echoed = await ctx.execute_tool("echo", {"text": "hi"})
        itself = await ctx.execute_tool("run_tools", {})
        text = " | ".join(text_of(outcome) for outcome in (helped, echoed, itself))
        return AgentToolResult(content=[TextContent(text=text)], details={})

    def prepare_loadout(loadout):
        return ToolLoadoutChanges(
            descriptions={
                "run_tools": f"Runs tools: {', '.join(tool.name for tool in loadout.callable)}",
                "echo": "Echo text (also callable from run_tools).",
            },
            hidden_declarations=("echo",),
        )

    pi.register_tool(
        ToolDefinition(name="echo", label="echo", description="Echo text.", parameters=TEXT_SCHEMA, execute=echo)
    )
    pi.register_tool(
        ToolDefinition(
            name="helper",
            label="helper",
            description="Only reachable from other tools.",
            parameters=EMPTY_SCHEMA,
            exposure="codemode",
            execute=helper,
        )
    )
    pi.register_tool(
        ToolDefinition(
            name="run_tools",
            label="run_tools",
            description="Runs tools.",
            parameters=EMPTY_SCHEMA,
            exposure="model-only",
            prepare_loadout=prepare_loadout,
            execute=run_tools,
        )
    )


def tool_result(harness):
    results = [message for message in harness.session.messages if message.role == "toolResult"]
    assert len(results) == 1
    return results[0]


def persisted_tool_result(harness):
    return next(
        entry["message"]
        for entry in harness.session_manager.get_branch()
        if entry["type"] == "message" and entry["message"].role == "toolResult"
    )


@pytest.mark.tonio
async def test_supports_tools_that_call_other_tools_under_any_name_through_the_extension_api(harnesses):
    tool_calls: list[str] = []

    async def record_tool_calls(pi) -> None:
        async def on_tool_call(event, _ctx):
            tool_calls.append(f"{event['toolName']}:{event.get('parentToolCallId') or 'top'}")

        pi.on("tool_call", on_tool_call)

    harness = await create_harness(
        initial_active_tool_names=[], extension_factories=[orchestrator_extension, record_tool_calls]
    )
    harnesses.append(harness)

    assert harness.session.get_active_tool_names() == ["echo", "run_tools"]
    assert harness.session.get_callable_tool_names() == ["echo", "helper"]
    tools = {tool.name: tool for tool in harness.session.agent.state.tools}
    assert tools["run_tools"].description == "Runs tools: echo, helper"
    assert tools["echo"].description == "Echo text (also callable from run_tools)."

    request_tools: list[list[str]] = []

    async def first(context, *_rest):
        request_tools.append([tool.name for tool in get_current_tools(context.messages)])
        return faux_assistant_message([faux_tool_call("run_tools", {})], stop_reason="toolUse")

    harness.set_responses([first, faux_assistant_message("done")])
    await harness.session.prompt("go")

    # echo stays active, but its declaration is left out of requests.
    assert request_tools[0] == ["run_tools"]
    result = tool_result(harness)
    assert result.content == [TextContent(text="helped | echo: hi | Tool run_tools not found")]
    parent = result.tool_call_id
    assert tool_calls == ["run_tools:top", f"helper:{parent}", f"echo:{parent}"]
    assert [(call.id, call.name, call.status) for call in result.nested_calls.calls] == [
        (f"{parent}/1", "helper", "ok"),
        (f"{parent}/2", "echo", "ok"),
        (f"{parent}/3", "run_tools", "error"),
    ]
    # The record is persisted with the session.
    assert persisted_tool_result(harness).nested_calls == result.nested_calls


@pytest.mark.tonio
async def test_leaves_results_without_nested_calls_unchanged(harnesses):
    harness = await create_harness(initial_active_tool_names=[], extension_factories=[orchestrator_extension])
    harnesses.append(harness)
    harness.set_responses(
        [
            faux_assistant_message([faux_tool_call("echo", {"text": "x"})], stop_reason="toolUse"),
            faux_assistant_message("done"),
        ]
    )

    await harness.session.prompt("go")

    assert tool_result(harness).nested_calls is None


# --- core cases of suite/agent-session-codemode.test.ts -----------------------


def usage(input_tokens: int, cost: float) -> Usage:
    return Usage(input=input_tokens, total_tokens=input_tokens, cost=UsageCost(input=cost, total=cost))


def script_extension(script):
    """Registers `echo`, `stats` (with an output schema) and `billed` (reports
    usage) as callable tools, and `script`, a model-only tool that runs
    `script(ctx)` and returns its text."""

    async def factory(pi) -> None:
        async def echo(_id, params, *_rest):
            return AgentToolResult(content=[TextContent(text=f"echo: {params['text']}")], details={})

        async def stats(*_rest):
            return AgentToolResult(
                content=[TextContent(text="2 files")], details={}, structured_content={"files": 2, "names": ["a", "b"]}
            )

        async def billed(*_rest):
            return AgentToolResult(content=[TextContent(text="ran")], details={}, usage=usage(100, 0.25))

        async def run_script(_id, _params, _cancel, _on_update, ctx):
            return AgentToolResult(content=[TextContent(text=await script(ctx))], details={})

        stats_schema = {
            "type": "object",
            "properties": {"files": {"type": "number"}, "names": {"type": "array", "items": {"type": "string"}}},
        }
        for definition in (
            ToolDefinition(name="echo", label="Echo", description="Echo", parameters=TEXT_SCHEMA, execute=echo),
            ToolDefinition(
                name="stats",
                label="Stats",
                description="Return structured stats",
                parameters=EMPTY_SCHEMA,
                output_schema=stats_schema,
                execute=stats,
            ),
            ToolDefinition(
                name="billed", label="Billed", description="Run a model", parameters=EMPTY_SCHEMA, execute=billed
            ),
            ToolDefinition(
                name="script",
                label="Script",
                description="Run a script",
                parameters=EMPTY_SCHEMA,
                exposure="model-only",
                execute=run_script,
            ),
        ):
            pi.register_tool(definition)

    return factory


async def run_script(harnesses, script, *, extension_factories=None, tools=None):
    harness = await create_harness(
        initial_active_tool_names=tools if tools is not None else ["script", "echo", "stats", "billed"],
        extension_factories=[script_extension(script), *(extension_factories or [])],
    )
    harnesses.append(harness)
    harness.set_responses(
        [
            faux_assistant_message([faux_tool_call("script", {})], stop_reason="toolUse"),
            faux_assistant_message("done"),
        ]
    )
    await harness.session.prompt("go")
    return harness, tool_result(harness)


@pytest.mark.tonio
async def test_runs_nested_calls_in_parallel_and_records_only_the_calling_tools_result(harnesses):
    async def script(ctx):
        first, second, stats = await tonio.spawn(
            ctx.execute_tool("echo", {"text": "one"}),
            ctx.execute_tool("echo", {"text": "two"}),
            ctx.execute_tool("stats", {}),
        )
        return f"{text_of(first)} | {text_of(second)} | {stats.result.structured_content['names']}"

    harness, result = await run_script(harnesses, script)

    assert result.is_error is False
    assert result.content == [TextContent(text="echo: one | echo: two | ['a', 'b']")]
    calls = result.nested_calls.calls
    assert sorted((call.name, call.status) for call in calls) == [("echo", "ok"), ("echo", "ok"), ("stats", "ok")]
    assert all(call.id.startswith(f"{result.tool_call_id}/") for call in calls)
    # Nested calls never become transcript tool results; their events carry the parent id.
    nested_starts = [
        event
        for event in harness.events_of_type("tool_execution_start")
        if event.parent_tool_call_id == result.tool_call_id
    ]
    assert sorted(event.tool_name for event in nested_starts) == ["echo", "echo", "stats"]


@pytest.mark.tonio
async def test_routes_nested_calls_through_extension_hooks(harnesses):
    async def hooks(pi) -> None:
        async def on_tool_call(event, _ctx):
            if event["toolName"] == "echo" and event["input"]["text"] == "forbidden":
                return {"block": True, "reason": "echo of forbidden text is blocked"}
            return None

        async def on_tool_result(event, _ctx):
            if event["toolName"] == "stats":
                return {"content": [TextContent(text="redacted")]}
            return None

        pi.on("tool_call", on_tool_call)
        pi.on("tool_result", on_tool_result)

    outcomes: list = []

    async def script(ctx):
        outcomes.append(await ctx.execute_tool("echo", {"text": "forbidden"}))
        outcomes.append(await ctx.execute_tool("stats", {}))
        return "ran"

    _harness, result = await run_script(harnesses, script, extension_factories=[hooks])

    blocked, stats = outcomes
    assert (blocked.is_error, text_of(blocked)) == (True, "echo of forbidden text is blocked")
    # Replacing content without replacing structured content drops the structured result.
    assert (text_of(stats), stats.result.structured_content) == ("redacted", None)
    assert [call.status for call in result.nested_calls.calls] == ["error", "ok"]


@pytest.mark.tonio
async def test_adds_the_usage_of_nested_results_to_the_calling_tools_result(harnesses):
    async def script(ctx):
        await ctx.execute_tool("billed", {})
        await ctx.execute_tool("billed", {})
        await ctx.execute_tool("echo", {"text": "x"})
        return "ran"

    harness, result = await run_script(harnesses, script)

    assert (result.usage.input, result.usage.total_tokens, result.usage.cost.total) == (200, 200, 0.5)
    # The usage is persisted with the result, so session totals count it.
    assert persisted_tool_result(harness).usage == result.usage
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

    outcomes: list = []

    async def script(ctx):
        outcomes.append(await ctx.execute_tool("stats", {}))
        return "ran"

    await run_script(harnesses, script, extension_factories=[hooks])

    assert outcomes[0].result.structured_content == {"files": 0, "names": []}
    assert outcomes[0].result.details == {"audited": True}


@pytest.mark.tonio
async def test_resolves_bash_calls_to_structured_results_also_for_non_zero_exit_codes(harnesses):
    outcomes: list = []

    async def script(ctx):
        outcomes.append(await ctx.execute_tool("bash", {"command": "echo out; exit 3"}))
        return "ran"

    await run_script(harnesses, script, tools=["script", "bash"])

    [outcome] = outcomes
    assert outcome.is_error is True
    assert (outcome.result.structured_content["output"], outcome.result.structured_content["exit_code"]) == (
        "out\n",
        3,
    )
