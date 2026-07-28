"""Mirror of pi agent/test/harness/tools.test.ts (bash tool portion)."""

import re
import tempfile
from dataclasses import dataclass

import pytest
import tonio.colored as tonio

from pidrei_agent.harness.env.local import LocalExecutionEnv
from pidrei_agent.harness.tools.bash import BashExecution, BashToolOptions, create_bash_tool
from pidrei_agent.harness.tools.tool_context import ExecutionToolContext
from pidrei_agent.harness.types import ShellExecOptions, ShellExecResult, get_or_throw, ok
from pidrei_agent.types import AgentToolResult
from pidrei_ai.utils.cancel import CancelToken


def create_temp_dir() -> str:
    return tempfile.mkdtemp(prefix="pidrei-agent-test-")


def text_output(result: AgentToolResult) -> str:
    return "\n".join(part.text for part in result.content if part.type == "text")


def create_context() -> ExecutionToolContext:
    return ExecutionToolContext(env=LocalExecutionEnv(cwd=create_temp_dir()))


class LateOutputExecutionEnv(LocalExecutionEnv):
    """Emits one chunk during exec and one after the execution settles."""

    def __init__(self, cwd: str):
        super().__init__(cwd)
        self.settled = tonio.Event()

    async def exec(self, _command: str, options: ShellExecOptions | None = None):
        options.on_stdout("before\n")

        async def late() -> None:
            await self.settled.wait(None)
            options.on_stdout("late\n")

        tonio.spawn.without_tracking(late())
        return ok(ShellExecResult(stdout="before\n", stderr="", exit_code=0))


@pytest.mark.tonio
async def test_executes_commands_and_combines_stdout_and_stderr():
    context = create_context()
    result = await create_bash_tool().execute("bash-1", {"command": "printf out; printf err >&2"}, None, None, context)

    assert "out" in text_output(result)
    assert "err" in text_output(result)


@pytest.mark.tonio
async def test_reports_nonzero_exits_and_timeouts():
    context = create_context()
    tool = create_bash_tool()

    with pytest.raises(Exception, match="(?s)failed.*Command exited with code 7"):
        await tool.execute("bash-2", {"command": "printf failed; exit 7"}, None, None, context)
    with pytest.raises(Exception, match=re.escape("Command timed out after 0.01 seconds")):
        await tool.execute("bash-3", {"command": "sleep 2", "timeout": 0.01}, None, None, context)


@pytest.mark.tonio
async def test_preserves_truncated_output_when_a_command_times_out():
    context = create_context()
    error: Exception | None = None
    try:
        await create_bash_tool().execute(
            "bash-timeout-output",
            {
                # The timeout must expire inside `sleep 2`, after the loop
                # finished printing: 0.05s was not enough headroom on a loaded
                # macOS CI runner, where the kill landed mid-loop and the
                # "full" output legitimately stopped short of line-3000.
                "command": "i=1; while [ $i -le 3000 ]; do echo line-$i; i=$((i + 1)); done; sleep 2",
                "timeout": 0.5,
            },
            None,
            None,
            context,
        )
    except Exception as cause:
        error = cause

    assert isinstance(error, Exception)
    message = str(error)
    assert "Command timed out after 0.5 seconds" in message
    match = re.search(r"Full output: ([^\]\n]+)", message)
    assert match is not None
    full_output = get_or_throw(await context.env.read_text_file(match.group(1)))
    assert "line-1\nline-2" in full_output
    assert "line-2999\nline-3000" in full_output


@pytest.mark.tonio
async def test_ignores_output_callbacks_after_execution_settles():
    env = LateOutputExecutionEnv(cwd=create_temp_dir())
    updates: list[str] = []
    result = await create_bash_tool().execute(
        "bash-late",
        {"command": "late"},
        None,
        lambda update: updates.append(text_output(update)),
        ExecutionToolContext(env=env),
    )
    env.settled.set()
    await tonio.sleep(0.02)

    assert text_output(result) == "before\n"
    assert not any("late" in update for update in updates)


@pytest.mark.tonio
async def test_reports_the_total_size_of_an_oversized_final_line():
    context = create_context()
    result = await create_bash_tool().execute("bash-long-line", {"command": "printf '%060000d' 0"}, None, None, context)

    assert re.search(r"Showing last 50\.0KB of line 1 \(line is 58\.6KB\)\. Full output:", text_output(result))


@dataclass
class WorkspaceContext(ExecutionToolContext):
    workspace: str = ""


@pytest.mark.tonio
async def test_prepares_command_cwd_and_explicit_environment_with_the_turn_context():
    env = LocalExecutionEnv(cwd=create_temp_dir(), shell_env={"PIDREI_BASH_PREPARE_INHERITED": "inherited"})
    get_or_throw(await env.create_dir("workspace"))
    context = WorkspaceContext(env=env, workspace=f"{env.cwd}/workspace")
    controller = CancelToken()
    received: dict = {}

    async def prepare(execution: BashExecution, turn_context, cancel) -> None:
        received["context"] = turn_context
        received["cancel"] = cancel
        execution.cwd = turn_context.workspace
        execution.env = {"PIDREI_BASH_PREPARE_EXPLICIT": "explicit"}
        execution.inherit_env = False
        execution.command += (
            '\nprintf \'%s:%s:%s:%s\' "$prefix" "${PIDREI_BASH_PREPARE_INHERITED-}"'
            ' "$PIDREI_BASH_PREPARE_EXPLICIT" "$PWD"'
        )

    tool = create_bash_tool(BashToolOptions(command_prefix="prefix=ready", prepare=prepare))

    result = await tool.execute("bash-prepare", {"command": ":"}, controller, None, context)

    assert received["context"] is context
    assert received["cancel"] is controller
    canonical_workspace = get_or_throw(await env.canonical_path(context.workspace))
    assert text_output(result) == f"ready::explicit:{canonical_workspace}"


@pytest.mark.tonio
async def test_supports_command_prefixes():
    context = create_context()
    result = await create_bash_tool(BashToolOptions(command_prefix="value=hello")).execute(
        "bash-4", {"command": "printf $value"}, None, None, context
    )

    assert text_output(result) == "hello"


@pytest.mark.tonio
async def test_coalesces_updates_and_persists_truncated_full_output():
    context = create_context()
    updates: list[AgentToolResult] = []
    result = await create_bash_tool().execute(
        "bash-5",
        {"command": "i=1; while [ $i -le 3000 ]; do echo line-$i; i=$((i + 1)); done"},
        None,
        updates.append,
        context,
    )

    assert len(updates) < 25
    assert result.details is not None
    truncation = result.details.truncation
    assert (truncation.truncated, truncation.truncated_by, truncation.total_lines, truncation.output_lines) == (
        True,
        "lines",
        3000,
        2000,
    )
    assert "line-3000" in text_output(result)
    assert result.details.full_output_path is not None
    final_update = updates[-1]
    assert "line-3000" in text_output(final_update)
    assert final_update.details is not None
    assert final_update.details.truncation is not None
    assert final_update.details.truncation.total_lines == 3000
    assert final_update.details.full_output_path == result.details.full_output_path
    full_output = get_or_throw(await context.env.read_text_file(result.details.full_output_path))
    assert "line-1\nline-2" in full_output
    assert "line-2999\nline-3000" in full_output
