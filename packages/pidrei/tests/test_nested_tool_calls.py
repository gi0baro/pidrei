"""Mirror of pi's nested-tool-calls.test.ts.

pi drives `NestedCallRecorder` (`start`/`finish`/`snapshot`) and a
`NestedToolCallRunner` whose host runs tools directly. pidrei has no shared
recorder (the `nested-calls-channel` recipe): the recorder cases feed the same
message sequences to `fold_nested_calls`, and the runner cases run through the
tool wrapper — the host binds the target to the scope it gets, like the
session does — with a root `NestedCallFeed` standing in for the model-issued
call, whose drain is pi's `takeRecord`.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.core.extensions.types import ToolDefinition
from pidrei.core.nested_tool_calls import (
    NESTED_CALL_LIMITS,
    ExecuteToolOptions,
    NestedCallFeed,
    NestedCallFinished,
    NestedCallScope,
    NestedCallStarted,
    NestedToolCallRunner,
    fold_nested_calls,
)
from pidrei.core.tools.tool_definition_wrapper import wrap_tool_definition
from pidrei_agent.agent_loop import RunToolCallOptions, run_tool_call
from pidrei_agent.types import AgentContext, AgentToolCallOutcome, AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message
from pidrei_ai.types import NestedToolCallRecord, NestedToolCalls, TextContent, Usage, UsageCost


EMPTY_SCHEMA = {"type": "object", "properties": {}}


def usage(input_tokens: int, cost: float) -> Usage:
    return Usage(input=input_tokens, total_tokens=input_tokens, cost=UsageCost(input=cost, total=cost))


def tool(name: str, execute, *, execution_mode: str | None = None) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        label=name,
        description=name,
        parameters=EMPTY_SCHEMA,
        execute=execute,
        execution_mode=execution_mode,
    )


@dataclass
class RunnerFixture:
    runner: NestedToolCallRunner
    events: list


def create_runner(definitions: list[ToolDefinition], *, sequential: bool = False) -> RunnerFixture:
    events: list = []
    runner_ref: list[NestedToolCallRunner] = []

    def ctx_factory(_tool_call_id, cancel, scope):
        def execute_tool(name, args, options=None):
            return runner_ref[0].execute(scope, name, args, options or ExecuteToolOptions(cancel=cancel))

        return SimpleNamespace(execute_tool=execute_tool)

    tools = [wrap_tool_definition(definition, ctx_factory) for definition in definitions]

    class Host:
        def get_tools(self):
            return tools

        def is_sequential(self):
            return sequential

        async def run_tool_call(self, tool_call, _parent_id, scope, cancel, on_update) -> AgentToolCallOutcome:
            bound = [entry.with_nested_scope(scope) if entry.name == tool_call.name else entry for entry in tools]
            return await run_tool_call(
                tool_call,
                RunToolCallOptions(
                    tools=bound,
                    assistant_message=faux_assistant_message(""),
                    context=AgentContext(messages=[]),
                    cancel=cancel,
                    on_update=on_update,
                ),
            )

        async def emit(self, event):
            events.append(event)

    runner_ref.append(NestedToolCallRunner(Host()))
    return RunnerFixture(runner=runner_ref[0], events=events)


def root(parent_id: str = "call") -> NestedCallScope:
    return NestedCallScope(parent_id=parent_id, feed=NestedCallFeed())


class TestNestedToolCallRunner:
    @pytest.mark.tonio
    async def test_assigns_ids_below_the_caller_emits_events_with_the_parent_id_and_records_the_calls(self):
        async def echo(_id, _params, _cancel, on_update, _ctx):
            on_update(AgentToolResult(content=[TextContent(text="partial")], details={}))
            return AgentToolResult(content=[TextContent(text="ok")], details={})

        fixture = create_runner([tool("echo", echo)])
        updates: list = []
        scope = root()

        first = await fixture.runner.execute(scope, "echo", {"a": 1}, ExecuteToolOptions(on_update=updates.append))
        missing = await fixture.runner.execute(scope, "missing", {})

        assert first.tool_call.id == "call/1"
        assert (missing.tool_call.id, missing.is_error) == ("call/2", True)
        assert len(updates) == 1
        assert [(event.type, event.tool_call_id, event.parent_tool_call_id) for event in fixture.events] == [
            ("tool_execution_start", "call/1", "call"),
            ("tool_execution_update", "call/1", "call"),
            ("tool_execution_end", "call/1", "call"),
            ("tool_execution_start", "call/2", "call"),
            ("tool_execution_end", "call/2", "call"),
        ]
        calls = scope.feed.drain().calls
        assert calls is not None
        assert all(isinstance(call.duration_ms, int) for call in calls.calls)
        assert [(call.id, call.name, call.arguments, call.status, call.error) for call in calls.calls] == [
            ("call/1", "echo", {"a": 1}, "ok", None),
            ("call/2", "missing", {}, "error", "Tool missing not found"),
        ]
        assert calls.complete is True

    @pytest.mark.tonio
    async def test_records_calls_of_nested_tools_on_the_model_issued_call(self):
        async def leaf(*_args):
            return AgentToolResult(content=[], details={})

        async def middle(_id, _params, _cancel, _on_update, ctx):
            await ctx.execute_tool("leaf", {})
            return AgentToolResult(content=[], details={})

        fixture = create_runner([tool("leaf", leaf), tool("middle", middle)])
        scope = root()

        await fixture.runner.execute(scope, "middle", {})

        assert [call.id for call in scope.feed.drain().calls.calls] == ["call/1", "call/1/1"]

    @pytest.mark.tonio
    async def test_sums_the_usage_of_nested_results_at_every_depth(self):
        async def leaf(*_args):
            return AgentToolResult(content=[], details={}, usage=usage(10, 0.01))

        async def plain(*_args):
            return AgentToolResult(content=[], details={})

        async def middle(_id, _params, _cancel, _on_update, ctx):
            await ctx.execute_tool("leaf", {})
            # Its own usage only: the leaf's usage is counted once, by the record.
            return AgentToolResult(content=[], details={}, usage=usage(5, 0.005))

        fixture = create_runner([tool("leaf", leaf), tool("plain", plain), tool("middle", middle)])
        call, free = root("call"), root("free")

        await fixture.runner.execute(call, "middle", {})
        await fixture.runner.execute(call, "leaf", {})
        await fixture.runner.execute(call, "plain", {})
        await fixture.runner.execute(free, "plain", {})

        summary = call.feed.drain()
        assert summary.usage.input == 25
        assert summary.usage.cost.total == pytest.approx(0.025, abs=1e-10)
        free_summary = free.feed.drain()
        assert (free_summary.calls.complete, free_summary.usage) == (True, None)

    @pytest.mark.tonio
    async def test_serializes_concurrent_calls_to_sequential_tools(self):
        active = 0
        max_active = {"sequential": 0, "parallel": 0}
        never = tonio.Event()
        all_parallel = tonio.Event()
        parallel_arrivals = 0

        def enter(name: str) -> None:
            nonlocal active
            active += 1
            max_active[name] = max(max_active[name], active)

        async def sequential(*_args):
            nonlocal active
            enter("sequential")
            # A window for another call to overlap, if the queue let it in.
            await never.wait(0.05)
            active -= 1
            return AgentToolResult(content=[], details={})

        async def parallel(*_args):
            nonlocal active, parallel_arrivals
            enter("parallel")
            parallel_arrivals += 1
            if parallel_arrivals == 3:
                all_parallel.set()
            await all_parallel.wait(5)
            active -= 1
            return AgentToolResult(content=[], details={})

        fixture = create_runner(
            [tool("sequential", sequential, execution_mode="sequential"), tool("parallel", parallel)]
        )
        scope = root()

        await tonio.spawn(*(fixture.runner.execute(scope, "sequential", {}) for _ in range(3)))
        await tonio.spawn(*(fixture.runner.execute(scope, "parallel", {}) for _ in range(3)))

        assert max_active == {"sequential": 1, "parallel": 3}


def started(call_id: str, arguments) -> NestedCallStarted:
    return NestedCallStarted(id=call_id, name="t", arguments=arguments)


def finished(call_id: str, *, is_error: bool = False, error_text: str = "") -> NestedCallFinished:
    return NestedCallFinished(id=call_id, is_error=is_error, error_text=error_text, duration_ms=0)


class TestNestedCallFold:
    def test_omits_oversized_arguments_and_drops_calls_beyond_the_limit(self):
        messages: list = []
        assert fold_nested_calls(messages).calls is None
        messages += [started("a", {"x": 1}), finished("a")]
        assert fold_nested_calls(messages).calls == NestedToolCalls(
            calls=[NestedToolCallRecord(id="a", name="t", arguments={"x": 1}, status="ok", duration_ms=0)],
            complete=True,
        )

        big = {"text": "x" * NESTED_CALL_LIMITS.max_argument_bytes_per_call}
        messages += [started("b", big), finished("b", is_error=True, error_text="e" * 1000)]
        snapshot = fold_nested_calls(messages).calls
        assert snapshot.complete is False
        record = snapshot.calls[1]
        assert (record.id, record.status, record.arguments) == ("b", "error", None)
        assert record.arguments_bytes > NESTED_CALL_LIMITS.max_argument_bytes_per_call
        assert len(record.error) == NESTED_CALL_LIMITS.max_error_chars

        for index in range(NESTED_CALL_LIMITS.max_calls):
            messages += [started(f"c{index}", {}), finished(f"c{index}")]
        assert len(fold_nested_calls(messages).calls.calls) == NESTED_CALL_LIMITS.max_calls

    def test_caps_the_total_argument_size_and_marks_unfinished_calls_incomplete(self):
        chunk = {"text": "x" * 7000}
        snapshot = fold_nested_calls([started(f"c{index}", chunk) for index in range(6)]).calls
        # 32 KiB fits four 7000-byte argument objects.
        assert len([entry for entry in snapshot.calls if entry.arguments is not None]) == 4
        assert all(entry.status == "unfinished" for entry in snapshot.calls)
        assert snapshot.complete is False
        assert len(snapshot.calls) == 6


class TestNestedCallFeed:
    """pidrei-only: the feed's close contract (the recipe's "producers tolerate
    a dropped channel")."""

    def test_drops_messages_sent_after_the_drain(self):
        feed = NestedCallFeed()
        feed.send(started("a", {}))
        summary = feed.drain()
        feed.send(finished("a"))

        assert [call.status for call in summary.calls.calls] == ["unfinished"]
        assert feed.drain().calls is None
