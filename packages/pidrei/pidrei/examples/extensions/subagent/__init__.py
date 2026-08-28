"""Subagent tool - delegate tasks to specialized agents.

Runs each subagent as an in-process `AgentSession` with an isolated context
window. pi's example spawns a `pi` subprocess per invocation because on a
single-threaded runtime that is the only way subagents can make progress
without janking the parent; on tonio, in-process sessions run genuinely in
parallel on the runtime's threads, skip an interpreter start per task, and
stream typed events instead of a JSONL pipe.

Supports three modes:
  - Single: { agent: "name", task: "..." }
  - Parallel: { tasks: [{ agent: "name", task: "..." }, ...] }
  - Chain: { chain: [{ agent: "name", task: "... {previous} ..." }, ...] }

Deliberate divergences from pi's example (recipe `subagent-inprocess` in
UPSTREAM_DELTA_PORT.md):
  - No subprocess: `run_single_agent` builds a lean session (in-memory session
    manager, no extensions/themes/prompt templates, the agent's system prompt
    appended directly). Skipping extensions also means a subagent cannot
    recursively spawn subagents, which the child process could when this
    extension was installed globally.
  - Result dicts carry `status` ("running" | "done" | "failed") and
    `errorMessage` instead of the process-shaped `exitCode`/`stderr`.
  - Usage is accumulated from the typed frozen messages; an unknown
    `model:` in agent frontmatter fails the task instead of silently
    falling back to the default model.
  - `on_update` streams per-delta (the child process only reported at
    message boundaries).
The `messages` in results/details remain camelCase wire dicts: details are
persisted with the parent session and come back as plain JSON on resume, so
the wire shape is the one rendering format.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/subagent
"""

import json
import os

import tonio.colored as tonio
from tonio.colored import fs
from tonio.colored.sync import channel

from pidrei.config import CONFIG_DIR_NAME, get_agent_dir
from pidrei.core.extensions.types import ToolDefinition
from pidrei.core.json_wire import to_wire
from pidrei.core.model_resolver import resolve_cli_model
from pidrei.core.model_runtime import ModelRuntime
from pidrei.core.resource_loader import DefaultResourceLoader
from pidrei.core.sdk import CreateAgentSessionOptions, create_agent_session
from pidrei.core.session_manager import SessionManager
from pidrei.core.settings_manager import SettingsManager
from pidrei.core.tools.render_utils import shorten_path
from pidrei.modes.interactive.theme import get_markdown_theme
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent
from pidrei_tui import Container, Markdown, Spacer, Text

from .agents import discover_agents


MAX_PARALLEL_TASKS = 8
MAX_CONCURRENCY = 4
COLLAPSED_ITEM_COUNT = 10
PER_TASK_OUTPUT_CAP = 50 * 1024


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_tokens(count: int) -> str:
    if count < 1000:
        return str(count)
    if count < 10000:
        return f"{count / 1000:.1f}k"
    if count < 1000000:
        return f"{round(count / 1000)}k"
    return f"{count / 1000000:.1f}M"


def format_usage_stats(usage: dict, model: str | None = None) -> str:
    parts: list[str] = []
    if usage.get("turns"):
        parts.append(f"{usage['turns']} turn{'s' if usage['turns'] > 1 else ''}")
    if usage.get("input"):
        parts.append(f"↑{format_tokens(usage['input'])}")
    if usage.get("output"):
        parts.append(f"↓{format_tokens(usage['output'])}")
    if usage.get("cacheRead"):
        parts.append(f"R{format_tokens(usage['cacheRead'])}")
    if usage.get("cacheWrite"):
        parts.append(f"W{format_tokens(usage['cacheWrite'])}")
    if usage.get("cost"):
        parts.append(f"${usage['cost']:.4f}")
    if usage.get("contextTokens"):
        parts.append(f"ctx:{format_tokens(usage['contextTokens'])}")
    if model:
        parts.append(model)
    return " ".join(parts)


def format_tool_call(tool_name: str, args: dict, theme) -> str:
    """Compact one-liner per tool call, mimicking the built-in renderers."""
    args = args or {}

    if tool_name == "bash":
        command = args.get("command") or "..."
        preview = f"{command[:60]}..." if len(command) > 60 else command
        return theme.fg("muted", "$ ") + theme.fg("toolOutput", preview)
    if tool_name == "read":
        file_path = shorten_path(args.get("file_path") or args.get("path") or "...")
        offset = args.get("offset")
        limit = args.get("limit")
        text = theme.fg("accent", file_path)
        if offset is not None or limit is not None:
            start_line = offset if offset is not None else 1
            end_line = start_line + limit - 1 if limit is not None else ""
            text += theme.fg("warning", f":{start_line}{f'-{end_line}' if end_line else ''}")
        return theme.fg("muted", "read ") + text
    if tool_name == "write":
        file_path = shorten_path(args.get("file_path") or args.get("path") or "...")
        lines = len((args.get("content") or "").split("\n"))
        text = theme.fg("muted", "write ") + theme.fg("accent", file_path)
        if lines > 1:
            text += theme.fg("dim", f" ({lines} lines)")
        return text
    if tool_name == "edit":
        file_path = shorten_path(args.get("file_path") or args.get("path") or "...")
        return theme.fg("muted", "edit ") + theme.fg("accent", file_path)
    if tool_name == "ls":
        return theme.fg("muted", "ls ") + theme.fg("accent", shorten_path(args.get("path") or "."))
    if tool_name == "find":
        pattern = args.get("pattern") or "*"
        path = shorten_path(args.get("path") or ".")
        return theme.fg("muted", "find ") + theme.fg("accent", pattern) + theme.fg("dim", f" in {path}")
    if tool_name == "grep":
        pattern = args.get("pattern") or ""
        path = shorten_path(args.get("path") or ".")
        return theme.fg("muted", "grep ") + theme.fg("accent", f"/{pattern}/") + theme.fg("dim", f" in {path}")

    args_str = json.dumps(args)
    preview = f"{args_str[:50]}..." if len(args_str) > 50 else args_str
    return theme.fg("accent", tool_name) + theme.fg("dim", f" {preview}")


# ---------------------------------------------------------------------------
# Result helpers. Subagent messages are the same camelCase wire dicts pi's
# example collected from the child's `--mode json` stream: details are
# persisted with the parent session and deserialize as plain JSON, so the wire
# shape is the one format rendering can rely on.
# ---------------------------------------------------------------------------


def empty_usage() -> dict:
    return {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "cost": 0.0, "contextTokens": 0, "turns": 0}


def get_final_output(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            for part in msg.get("content") or []:
                if part.get("type") == "text":
                    return part["text"]
    return ""


def is_failed_result(result: dict) -> bool:
    # pi keys this on the child's exitCode; without a process, `status` is set
    # directly from the run outcome.
    return result["status"] == "failed"


def is_running_result(result: dict) -> bool:
    # pi's example encodes "still running" as exitCode -1.
    return result["status"] == "running"


def get_result_output(result: dict) -> str:
    if is_failed_result(result):
        return result.get("errorMessage") or get_final_output(result["messages"]) or "(no output)"
    return get_final_output(result["messages"]) or "(no output)"


def truncate_parallel_output(output: str) -> str:
    byte_length = len(output.encode("utf-8"))
    if byte_length <= PER_TASK_OUTPUT_CAP:
        return output

    truncated = output.encode("utf-8")[:PER_TASK_OUTPUT_CAP].decode("utf-8", "ignore")
    omitted = byte_length - len(truncated.encode("utf-8"))
    return f"{truncated}\n\n[Output truncated: {omitted} bytes omitted. Full output preserved in tool details.]"


def get_display_items(messages: list[dict]) -> list[dict]:
    items: list[dict] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for part in msg.get("content") or []:
            if part.get("type") == "text":
                items.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "toolCall":
                items.append({"type": "toolCall", "name": part["name"], "args": part.get("arguments") or {}})
    return items


async def map_with_concurrency_limit(items: list, concurrency: int, fn) -> list:
    """pi's mapWithConcurrencyLimit (a hand-rolled shared-index worker pool)
    as a tonio work queue: every item goes on a channel sized to hold them
    all, N consumers drain it, and the closed channel is the termination
    signal. Results keep item order."""
    if not items:
        return []
    limit = max(1, min(concurrency, len(items)))
    results: list = [None] * len(items)

    sender, receiver = channel.unbounded()
    for pair in enumerate(items):
        sender.send(pair)
    sender.close()

    async def consume(_consumer_index: int) -> None:
        while True:
            try:
                index, item = await receiver.receive()
            except BrokenPipeError:
                return
            results[index] = await fn(item, index)

    await tonio.map(consume, range(limit))
    return results


# ---------------------------------------------------------------------------
# In-process session plumbing. pi's example builds a CLI invocation here
# (`--mode json -p --no-session`, temp file for `--append-system-prompt`) and
# pumps the child's stdout; the in-process equivalent builds the same
# configuration as session options and subscribes to the session's typed
# events.
# ---------------------------------------------------------------------------


def _failed_result(agent_name: str, agent_source: str, task: str, step: int | None, error_message: str) -> dict:
    return {
        "agent": agent_name,
        "agentSource": agent_source,
        "task": task,
        "status": "failed",
        "messages": [],
        "errorMessage": error_message,
        "usage": empty_usage(),
        "step": step,
    }


async def run_single_agent(
    default_cwd: str,
    dispatch_defaults: dict,
    agents: list,
    agent_name: str,
    task: str,
    cwd: str | None,
    step: int | None,
    cancel,
    on_update,
    make_details,
) -> dict:
    agent = next((a for a in agents if a.name == agent_name), None)

    if agent is None:
        available = ", ".join(f'"{a.name}"' for a in agents) or "none"
        return _failed_result(
            agent_name, "unknown", task, step, f'Unknown agent: "{agent_name}". Available agents: {available}.'
        )

    run_cwd = cwd or default_cwd
    if not await fs.Path(run_cwd).exists():
        # The child process failed at spawn on a bad cwd; fail before building
        # a session for the same effect.
        return _failed_result(agent_name, agent.source, task, step, f"Working directory does not exist: {run_cwd}")

    inherits_dispatch_config = not agent.model
    model = agent.model or dispatch_defaults.get("model")

    current_result: dict = {
        "agent": agent_name,
        "agentSource": agent.source,
        "task": task,
        "status": "running",
        "messages": [],
        "usage": empty_usage(),
        "model": model,
        "step": step,
    }

    # Per-delta preview of the assistant text being streamed; cleared when the
    # message lands in `messages`. The child-process variant could only report
    # at message boundaries.
    streaming_preview = {"text": ""}

    def emit_update() -> None:
        if on_update is not None:
            text = streaming_preview["text"] or get_final_output(current_result["messages"]) or "(running...)"
            on_update(
                AgentToolResult(
                    content=[TextContent(text=text)],
                    details=make_details([current_result]),
                )
            )

    def process_message_end(message) -> None:
        # The session emits tool results as message_end events too (role
        # "toolResult"), so one branch covers what pi splits into message_end
        # and tool_result_end. Messages are stored as their wire dicts — the
        # shape details keep across parent-session persistence.
        current_result["messages"].append(to_wire(message))

        if getattr(message, "role", None) == "assistant":
            usage = current_result["usage"]
            usage["turns"] += 1
            msg_usage = message.usage
            usage["input"] += msg_usage.input
            usage["output"] += msg_usage.output
            usage["cacheRead"] += msg_usage.cache_read
            usage["cacheWrite"] += msg_usage.cache_write
            usage["cost"] += msg_usage.cost.total
            usage["contextTokens"] = msg_usage.total_tokens
            if not current_result.get("model") and message.model:
                current_result["model"] = message.model
            if message.stop_reason:
                current_result["stopReason"] = message.stop_reason
            if message.error_message:
                current_result["errorMessage"] = message.error_message
        streaming_preview["text"] = ""
        emit_update()

    def process_message_update(message) -> None:
        for block in reversed(message.content):
            if getattr(block, "type", None) == "text" and block.text:
                streaming_preview["text"] = block.text
                emit_update()
                return

    def on_event(event) -> None:
        event_type = getattr(event, "type", None)
        if event_type == "message_end" and event.message is not None:
            process_message_end(event.message)
        elif event_type == "message_update":
            process_message_update(event.message)

    # A lean stand-in for the child's `--mode json -p --no-session` CLI run:
    # nothing persisted, no extensions (a subagent cannot spawn subagents), no
    # themes or prompt templates, the agent's system prompt appended directly.
    agent_dir = get_agent_dir()
    settings_manager = await SettingsManager.create(run_cwd, agent_dir)
    model_runtime = await ModelRuntime.create()

    resolved_model = None
    thinking_level = None
    if model:
        resolved = resolve_cli_model(cli_model=model, model_runtime=model_runtime)
        resolved_model = resolved.model
        if agent.model and resolved_model is None:
            # The child ran anyway on the default model and buried the warning
            # in stderr; failing loudly beats burning tokens on the wrong model.
            return _failed_result(
                agent_name, agent.source, task, step, resolved.error or f'Unknown model: "{agent.model}".'
            )
        if not inherits_dispatch_config and resolved.thinking_level:
            thinking_level = resolved.thinking_level
    if inherits_dispatch_config and dispatch_defaults.get("thinkingLevel"):
        thinking_level = dispatch_defaults["thinkingLevel"]

    resource_loader = DefaultResourceLoader(
        cwd=run_cwd,
        agent_dir=agent_dir,
        settings_manager=settings_manager,
        no_extensions=True,
        no_themes=True,
        no_prompt_templates=True,
        append_system_prompt=[agent.system_prompt] if agent.system_prompt.strip() else None,
    )
    await resource_loader.reload()

    created = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=run_cwd,
            model_runtime=model_runtime,
            model=resolved_model,
            thinking_level=thinking_level,
            tools=list(agent.tools) if agent.tools else None,
            session_manager=SessionManager.in_memory(run_cwd),
            settings_manager=settings_manager,
            resource_loader=resource_loader,
        )
    )
    session = created.session
    if session.model is None:
        session.dispose()
        return _failed_result(
            agent_name, agent.source, task, step, created.model_fallback_message or "No models available."
        )

    aborted = {"flag": False}
    unsubscribe_cancel = None
    if cancel is not None:

        def on_abort(_reason) -> None:
            aborted["flag"] = True
            tonio.spawn.without_tracking(session.abort())

        unsubscribe_cancel = cancel.on_cancel(on_abort)

    unsubscribe = session.subscribe(on_event)
    try:
        await session.prompt(f"Task: {task}")
    finally:
        if unsubscribe_cancel is not None:
            unsubscribe_cancel()
        unsubscribe()
        session.dispose()

    if aborted["flag"]:
        raise Exception("Subagent was aborted")
    current_result["status"] = "failed" if current_result.get("stopReason") in ("error", "aborted") else "done"
    return current_result


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

_TASK_ITEM = {
    "type": "object",
    "properties": {
        "agent": {"type": "string", "description": "Name of the agent to invoke"},
        "task": {"type": "string", "description": "Task to delegate to the agent"},
        "cwd": {"type": "string", "description": "Working directory for the agent process"},
    },
    "required": ["agent", "task"],
}

_CHAIN_ITEM = {
    "type": "object",
    "properties": {
        "agent": {"type": "string", "description": "Name of the agent to invoke"},
        "task": {"type": "string", "description": "Task with optional {previous} placeholder for prior output"},
        "cwd": {"type": "string", "description": "Working directory for the agent process"},
    },
    "required": ["agent", "task"],
}

SUBAGENT_PARAMS = {
    "type": "object",
    "properties": {
        "agent": {"type": "string", "description": "Name of the agent to invoke (for single mode)"},
        "task": {"type": "string", "description": "Task to delegate (for single mode)"},
        "tasks": {
            "type": "array",
            "items": _TASK_ITEM,
            "description": "Array of {agent, task} for parallel execution",
        },
        "chain": {
            "type": "array",
            "items": _CHAIN_ITEM,
            "description": "Array of {agent, task} for sequential execution",
        },
        "concurrency": {
            "type": "integer",
            "description": (
                f"How many parallel tasks run at once (parallel mode only). "
                f"Default: {MAX_CONCURRENCY}, max: {MAX_PARALLEL_TASKS}. Raise it only when the "
                "provider account's rate limits allow that many concurrent streams."
            ),
            "default": MAX_CONCURRENCY,
        },
        "agentScope": {
            "type": "string",
            "enum": ["user", "project", "both"],
            "description": 'Which agent directories to use. Default: "user". Use "both" to include project-local agents.',
            "default": "user",
        },
        "confirmProjectAgents": {
            "type": "boolean",
            "description": "Prompt before running project-local agents. Default: true.",
            "default": True,
        },
        "cwd": {"type": "string", "description": "Working directory for the agent process (single mode)"},
    },
}


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


async def _execute(_tool_call_id, params, cancel=None, on_update=None, ctx=None):
    agent_scope = params.get("agentScope") or "user"
    model = getattr(ctx, "model", None)
    dispatch_defaults = {
        "model": f"{model.provider}/{model.id}" if model is not None else None,
        "thinkingLevel": getattr(ctx, "thinking_level", None),
    }
    discovery = await tonio.spawn_blocking(discover_agents, ctx.cwd, agent_scope)
    agents = discovery.agents
    confirm_project_agents = params.get("confirmProjectAgents", True)

    has_chain = bool(params.get("chain"))
    has_tasks = bool(params.get("tasks"))
    has_single = bool(params.get("agent") and params.get("task"))
    mode_count = int(has_chain) + int(has_tasks) + int(has_single)

    def make_details(mode: str):
        def build(results: list[dict]) -> dict:
            return {
                "mode": mode,
                "agentScope": agent_scope,
                "projectAgentsDir": discovery.project_agents_dir,
                "results": results,
            }

        return build

    if mode_count != 1:
        available = ", ".join(f"{a.name} ({a.source})" for a in agents) or "none"
        return AgentToolResult(
            content=[TextContent(text=f"Invalid parameters. Provide exactly one mode.\nAvailable agents: {available}")],
            details=make_details("single")([]),
        )

    if agent_scope in ("project", "both") and confirm_project_agents and ctx.has_ui and not ctx.is_project_trusted():
        requested_agent_names = set()
        for item in params.get("chain") or []:
            requested_agent_names.add(item["agent"])
        for item in params.get("tasks") or []:
            requested_agent_names.add(item["agent"])
        if params.get("agent"):
            requested_agent_names.add(params["agent"])

        project_agents_requested = [
            agent
            for name in requested_agent_names
            if (agent := next((a for a in agents if a.name == name), None)) is not None and agent.source == "project"
        ]

        if project_agents_requested:
            names = ", ".join(a.name for a in project_agents_requested)
            dir = discovery.project_agents_dir or "(unknown)"
            ok = await ctx.ui.confirm(
                "Run project-local agents?",
                f"Agents: {names}\nSource: {dir}\n\n"
                "Project agents are repo-controlled. Only continue for trusted repositories.",
            )
            if not ok:
                return AgentToolResult(
                    content=[TextContent(text="Canceled: project-local agents not approved.")],
                    details=make_details("chain" if has_chain else "parallel" if has_tasks else "single")([]),
                )

    if has_chain:
        chain = params["chain"]
        results: list[dict] = []
        previous_output = ""

        for i, chain_step in enumerate(chain):
            task_with_context = chain_step["task"].replace("{previous}", previous_output)

            # Update callback that includes all previous results
            chain_update = None
            if on_update is not None:

                def chain_update(partial, _completed=results):
                    # Combine completed results with the current streaming
                    # result
                    streaming = (partial.details or {}).get("results") or []
                    if streaming:
                        on_update(
                            AgentToolResult(
                                content=partial.content,
                                details=make_details("chain")([*_completed, streaming[0]]),
                            )
                        )

            result = await run_single_agent(
                ctx.cwd,
                dispatch_defaults,
                agents,
                chain_step["agent"],
                task_with_context,
                chain_step.get("cwd"),
                i + 1,
                cancel,
                chain_update,
                make_details("chain"),
            )
            results.append(result)

            if is_failed_result(result):
                # pi returns the partial results with isError: true;
                # pidrei tools signal failure by raising instead.
                error_msg = get_result_output(result)
                raise Exception(f"Chain stopped at step {i + 1} ({chain_step['agent']}): {error_msg}")
            previous_output = get_final_output(result["messages"])

        return AgentToolResult(
            content=[TextContent(text=get_final_output(results[-1]["messages"]) or "(no output)")],
            details=make_details("chain")(results),
        )

    if has_tasks:
        tasks = params["tasks"]
        if len(tasks) > MAX_PARALLEL_TASKS:
            return AgentToolResult(
                content=[TextContent(text=f"Too many parallel tasks ({len(tasks)}). Max is {MAX_PARALLEL_TASKS}.")],
                details=make_details("parallel")([]),
            )

        # pidrei-only: pi's fixed MAX_CONCURRENCY bounded child processes; in
        # process the cap protects the provider quota, so callers on
        # high-limit accounts may raise it (clamped to the task cap).
        requested_concurrency = params.get("concurrency")
        concurrency = requested_concurrency if isinstance(requested_concurrency, int) else MAX_CONCURRENCY
        concurrency = max(1, min(concurrency, MAX_PARALLEL_TASKS))

        # Track all results for streaming updates (pi: exitCode -1 = running)
        all_results: list[dict] = [
            {
                "agent": t["agent"],
                "agentSource": "unknown",
                "task": t["task"],
                "status": "running",
                "messages": [],
                "usage": empty_usage(),
            }
            for t in tasks
        ]

        def emit_parallel_update() -> None:
            if on_update is not None:
                running = sum(1 for r in all_results if is_running_result(r))
                done = len(all_results) - running
                on_update(
                    AgentToolResult(
                        content=[TextContent(text=f"Parallel: {done}/{len(all_results)} done, {running} running...")],
                        details=make_details("parallel")(list(all_results)),
                    )
                )

        async def run_one(t: dict, index: int) -> dict:
            def task_update(partial) -> None:
                streaming = (partial.details or {}).get("results") or []
                if streaming:
                    all_results[index] = streaming[0]
                    emit_parallel_update()

            result = await run_single_agent(
                ctx.cwd,
                dispatch_defaults,
                agents,
                t["agent"],
                t["task"],
                t.get("cwd"),
                None,
                cancel,
                task_update,
                make_details("parallel"),
            )
            all_results[index] = result
            emit_parallel_update()
            return result

        results = await map_with_concurrency_limit(tasks, concurrency, run_one)

        success_count = sum(1 for r in results if not is_failed_result(r))
        summaries = []
        for r in results:
            output = truncate_parallel_output(get_result_output(r))
            if is_failed_result(r):
                stop_reason = r.get("stopReason")
                status = f"failed ({stop_reason})" if stop_reason and stop_reason != "end" else "failed"
            else:
                status = "completed"
            summaries.append(f"### [{r['agent']}] {status}\n\n{output}")
        joined = "\n\n---\n\n".join(summaries)
        return AgentToolResult(
            content=[TextContent(text=f"Parallel: {success_count}/{len(results)} succeeded\n\n{joined}")],
            details=make_details("parallel")(results),
        )

    # Single mode
    result = await run_single_agent(
        ctx.cwd,
        dispatch_defaults,
        agents,
        params["agent"],
        params["task"],
        params.get("cwd"),
        None,
        cancel,
        on_update,
        make_details("single"),
    )
    if is_failed_result(result):
        # pi returns isError: true; pidrei tools raise instead.
        raise Exception(f"Agent {result.get('stopReason') or 'failed'}: {get_result_output(result)}")
    return AgentToolResult(
        content=[TextContent(text=get_final_output(result["messages"]) or "(no output)")],
        details=make_details("single")([result]),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_call(args, theme, _context):
    args = args or {}
    scope = args.get("agentScope") or "user"
    chain = args.get("chain") or []
    tasks = args.get("tasks") or []

    if chain:
        text = (
            theme.fg("toolTitle", theme.bold("subagent "))
            + theme.fg("accent", f"chain ({len(chain)} steps)")
            + theme.fg("muted", f" [{scope}]")
        )
        for i, chain_step in enumerate(chain[:3]):
            # Clean up {previous} placeholder for display
            clean_task = chain_step.get("task", "").replace("{previous}", "").strip()
            preview = f"{clean_task[:40]}..." if len(clean_task) > 40 else clean_task
            text += (
                "\n  "
                + theme.fg("muted", f"{i + 1}.")
                + " "
                + theme.fg("accent", chain_step.get("agent", ""))
                + theme.fg("dim", f" {preview}")
            )
        if len(chain) > 3:
            text += f"\n  {theme.fg('muted', f'... +{len(chain) - 3} more')}"
        return Text(text, 0, 0)

    if tasks:
        text = (
            theme.fg("toolTitle", theme.bold("subagent "))
            + theme.fg("accent", f"parallel ({len(tasks)} tasks)")
            + theme.fg("muted", f" [{scope}]")
        )
        for t in tasks[:3]:
            task = t.get("task", "")
            preview = f"{task[:40]}..." if len(task) > 40 else task
            text += f"\n  {theme.fg('accent', t.get('agent', ''))}{theme.fg('dim', f' {preview}')}"
        if len(tasks) > 3:
            text += f"\n  {theme.fg('muted', f'... +{len(tasks) - 3} more')}"
        return Text(text, 0, 0)

    agent_name = args.get("agent") or "..."
    task = args.get("task")
    preview = (f"{task[:60]}..." if len(task) > 60 else task) if task else "..."
    text = (
        theme.fg("toolTitle", theme.bold("subagent "))
        + theme.fg("accent", agent_name)
        + theme.fg("muted", f" [{scope}]")
    )
    text += f"\n  {theme.fg('dim', preview)}"
    return Text(text, 0, 0)


def _render_result(result, options, theme, _context):
    expanded = bool(options.get("expanded"))
    details = result["details"] if isinstance(result, dict) else result.details
    content = result["content"] if isinstance(result, dict) else result.content
    if not details or not details.get("results"):
        first = content[0] if content else None
        return Text(getattr(first, "text", None) or "(no output)", 0, 0)

    md_theme = get_markdown_theme()

    def render_display_items(items: list[dict], limit: int | None = None) -> str:
        to_show = items[-limit:] if limit else items
        skipped = len(items) - len(to_show)
        text = ""
        if skipped > 0:
            text += theme.fg("muted", f"... {skipped} earlier items\n")
        for item in to_show:
            if item["type"] == "text":
                preview = item["text"] if expanded else "\n".join(item["text"].split("\n")[:3])
                text += f"{theme.fg('toolOutput', preview)}\n"
            else:
                text += f"{theme.fg('muted', '→ ') + format_tool_call(item['name'], item['args'], theme)}\n"
        return text.rstrip()

    def add_tool_calls(container: Container, items: list[dict]) -> None:
        for item in items:
            if item["type"] == "toolCall":
                container.add_child(
                    Text(theme.fg("muted", "→ ") + format_tool_call(item["name"], item["args"], theme), 0, 0)
                )

    def aggregate_usage(results: list[dict]) -> dict:
        total = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "cost": 0.0, "turns": 0}
        for r in results:
            for key in total:
                total[key] += r["usage"][key]
        return total

    if details["mode"] == "single" and len(details["results"]) == 1:
        r = details["results"][0]
        is_error = is_failed_result(r)
        icon = theme.fg("error", "✗") if is_error else theme.fg("success", "✓")
        display_items = get_display_items(r["messages"])
        final_output = get_final_output(r["messages"])

        if expanded:
            container = Container()
            header = (
                f"{icon} {theme.fg('toolTitle', theme.bold(r['agent']))}{theme.fg('muted', f' ({r["agentSource"]})')}"
            )
            if is_error and r.get("stopReason"):
                header += f" {theme.fg('error', f'[{r["stopReason"]}]')}"
            container.add_child(Text(header, 0, 0))
            if is_error and r.get("errorMessage"):
                container.add_child(Text(theme.fg("error", f"Error: {r['errorMessage']}"), 0, 0))
            container.add_child(Spacer(1))
            container.add_child(Text(theme.fg("muted", "─── Task ───"), 0, 0))
            container.add_child(Text(theme.fg("dim", r["task"]), 0, 0))
            container.add_child(Spacer(1))
            container.add_child(Text(theme.fg("muted", "─── Output ───"), 0, 0))
            if not display_items and not final_output:
                container.add_child(Text(theme.fg("muted", "(no output)"), 0, 0))
            else:
                add_tool_calls(container, display_items)
                if final_output:
                    container.add_child(Spacer(1))
                    container.add_child(Markdown(final_output.strip(), 0, 0, md_theme))
            usage_str = format_usage_stats(r["usage"], r.get("model"))
            if usage_str:
                container.add_child(Spacer(1))
                container.add_child(Text(theme.fg("dim", usage_str), 0, 0))
            return container

        text = f"{icon} {theme.fg('toolTitle', theme.bold(r['agent']))}{theme.fg('muted', f' ({r["agentSource"]})')}"
        if is_error and r.get("stopReason"):
            text += f" {theme.fg('error', f'[{r["stopReason"]}]')}"
        if is_error and r.get("errorMessage"):
            text += f"\n{theme.fg('error', f'Error: {r["errorMessage"]}')}"
        elif not display_items:
            text += f"\n{theme.fg('muted', '(no output)')}"
        else:
            text += f"\n{render_display_items(display_items, COLLAPSED_ITEM_COUNT)}"
            if len(display_items) > COLLAPSED_ITEM_COUNT:
                text += f"\n{theme.fg('muted', '(Ctrl+O to expand)')}"
        usage_str = format_usage_stats(r["usage"], r.get("model"))
        if usage_str:
            text += f"\n{theme.fg('dim', usage_str)}"
        return Text(text, 0, 0)

    if details["mode"] == "chain":
        results = details["results"]
        success_count = sum(1 for r in results if not is_failed_result(r))
        icon = theme.fg("success", "✓") if success_count == len(results) else theme.fg("error", "✗")

        if expanded:
            container = Container()
            container.add_child(
                Text(
                    icon
                    + " "
                    + theme.fg("toolTitle", theme.bold("chain "))
                    + theme.fg("accent", f"{success_count}/{len(results)} steps"),
                    0,
                    0,
                )
            )

            for r in results:
                r_icon = theme.fg("error", "✗") if is_failed_result(r) else theme.fg("success", "✓")
                display_items = get_display_items(r["messages"])
                final_output = get_final_output(r["messages"])

                container.add_child(Spacer(1))
                container.add_child(
                    Text(
                        f"{theme.fg('muted', f'─── Step {r.get("step")}: ') + theme.fg('accent', r['agent'])} {r_icon}",
                        0,
                        0,
                    )
                )
                container.add_child(Text(theme.fg("muted", "Task: ") + theme.fg("dim", r["task"]), 0, 0))
                add_tool_calls(container, display_items)
                if final_output:
                    container.add_child(Spacer(1))
                    container.add_child(Markdown(final_output.strip(), 0, 0, md_theme))
                step_usage = format_usage_stats(r["usage"], r.get("model"))
                if step_usage:
                    container.add_child(Text(theme.fg("dim", step_usage), 0, 0))

            usage_str = format_usage_stats(aggregate_usage(results))
            if usage_str:
                container.add_child(Spacer(1))
                container.add_child(Text(theme.fg("dim", f"Total: {usage_str}"), 0, 0))
            return container

        # Collapsed view
        text = (
            icon
            + " "
            + theme.fg("toolTitle", theme.bold("chain "))
            + theme.fg("accent", f"{success_count}/{len(results)} steps")
        )
        for r in results:
            r_icon = theme.fg("error", "✗") if is_failed_result(r) else theme.fg("success", "✓")
            display_items = get_display_items(r["messages"])
            text += f"\n\n{theme.fg('muted', f'─── Step {r.get("step")}: ')}{theme.fg('accent', r['agent'])} {r_icon}"
            if not display_items:
                text += f"\n{theme.fg('muted', '(no output)')}"
            else:
                text += f"\n{render_display_items(display_items, 5)}"
        usage_str = format_usage_stats(aggregate_usage(results))
        if usage_str:
            text += f"\n\n{theme.fg('dim', f'Total: {usage_str}')}"
        text += f"\n{theme.fg('muted', '(Ctrl+O to expand)')}"
        return Text(text, 0, 0)

    if details["mode"] == "parallel":
        results = details["results"]
        running = sum(1 for r in results if is_running_result(r))
        success_count = sum(1 for r in results if not is_running_result(r) and not is_failed_result(r))
        fail_count = sum(1 for r in results if not is_running_result(r) and is_failed_result(r))
        is_running = running > 0
        icon = (
            theme.fg("warning", "⏳")
            if is_running
            else theme.fg("warning", "◐")
            if fail_count > 0
            else theme.fg("success", "✓")
        )
        status = (
            f"{success_count + fail_count}/{len(results)} done, {running} running"
            if is_running
            else f"{success_count}/{len(results)} tasks"
        )

        if expanded and not is_running:
            container = Container()
            container.add_child(
                Text(f"{icon} {theme.fg('toolTitle', theme.bold('parallel '))}{theme.fg('accent', status)}", 0, 0)
            )

            for r in results:
                r_icon = theme.fg("error", "✗") if is_failed_result(r) else theme.fg("success", "✓")
                display_items = get_display_items(r["messages"])
                final_output = get_final_output(r["messages"])

                container.add_child(Spacer(1))
                container.add_child(
                    Text(f"{theme.fg('muted', '─── ') + theme.fg('accent', r['agent'])} {r_icon}", 0, 0)
                )
                container.add_child(Text(theme.fg("muted", "Task: ") + theme.fg("dim", r["task"]), 0, 0))
                add_tool_calls(container, display_items)
                if final_output:
                    container.add_child(Spacer(1))
                    container.add_child(Markdown(final_output.strip(), 0, 0, md_theme))
                task_usage = format_usage_stats(r["usage"], r.get("model"))
                if task_usage:
                    container.add_child(Text(theme.fg("dim", task_usage), 0, 0))

            usage_str = format_usage_stats(aggregate_usage(results))
            if usage_str:
                container.add_child(Spacer(1))
                container.add_child(Text(theme.fg("dim", f"Total: {usage_str}"), 0, 0))
            return container

        # Collapsed view (or still running)
        text = f"{icon} {theme.fg('toolTitle', theme.bold('parallel '))}{theme.fg('accent', status)}"
        for r in results:
            r_icon = (
                theme.fg("warning", "⏳")
                if is_running_result(r)
                else theme.fg("error", "✗")
                if is_failed_result(r)
                else theme.fg("success", "✓")
            )
            display_items = get_display_items(r["messages"])
            text += f"\n\n{theme.fg('muted', '─── ')}{theme.fg('accent', r['agent'])} {r_icon}"
            if not display_items:
                text += f"\n{theme.fg('muted', '(running...)' if is_running_result(r) else '(no output)')}"
            else:
                text += f"\n{render_display_items(display_items, 5)}"
        if not is_running:
            usage_str = format_usage_stats(aggregate_usage(results))
            if usage_str:
                text += f"\n\n{theme.fg('dim', f'Total: {usage_str}')}"
        if not expanded:
            text += f"\n{theme.fg('muted', '(Ctrl+O to expand)')}"
        return Text(text, 0, 0)

    first = content[0] if content else None
    return Text(getattr(first, "text", None) or "(no output)", 0, 0)


def extension(pi):
    pi.register_tool(
        ToolDefinition(
            name="subagent",
            label="Subagent",
            description=" ".join(
                [
                    "Delegate tasks to specialized subagents with isolated context.",
                    "Modes: single (agent + task), parallel (tasks array), chain (sequential with {previous} placeholder).",
                    f'Default agent scope is "user" (from {os.path.join(get_agent_dir(), "agents")}).',
                    f'To enable project-local agents in {CONFIG_DIR_NAME}/agents, set agentScope: "both" (or "project").',
                ]
            ),
            parameters=SUBAGENT_PARAMS,
            execute=_execute,
            render_call=_render_call,
            render_result=_render_result,
        )
    )
