"""Configurable coding-agent Harness factory (port of pi `server/create-harness.ts`).

Builds an `AgentHarness` preloaded with the coding-agent policy: the default
read/bash/edit/write harness tools bound to an execution env, their system
prompt contributions, the PIDREI_* bash session environment, and a default
system prompt callback that reflects the harness's *current* tool
configuration on every run.

pi's `CodingAgentHarnessTool` interface (HarnessTool + `promptSnippet` /
`promptGuidelines`) is structural: any object with the harness tool surface
plus those attributes qualifies. `_bind_coding_agent_harness_tool` mirrors
pi's `{ ...tool, execute: bound }` spread with attribute delegation.
"""

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pidrei_agent.harness.agent_harness import AgentHarness, AgentHarnessOptions, SuspendedOperation
from pidrei_agent.harness.session.session import Session
from pidrei_agent.harness.tools.bash import BashToolOptions, create_bash_tool
from pidrei_agent.harness.tools.edit import create_edit_tool
from pidrei_agent.harness.tools.read import create_read_tool
from pidrei_agent.harness.tools.tool_context import ExecutionToolContext
from pidrei_agent.harness.tools.write import create_write_tool
from pidrei_agent.harness.types import ExecutionEnv

from ..core.experimental import get_experimental_tool_sampling
from ..core.system_prompt import BuildSystemPromptOptions, ContextFile, build_system_prompt
from ..core.tools.bash import BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION
from ..core.tools.edit import EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION
from ..core.tools.read import READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION
from ..core.tools.write import WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION


class _ContextBoundHarnessTool:
    """pi's `createCodingAgentHarnessTool`: the tool spread plus a context-bound execute."""

    def __init__(self, tool: Any, context: ExecutionToolContext, prompt_snippet: str, prompt_guidelines: list[str]):
        self._tool = tool
        self._context = context
        self.prompt_snippet = prompt_snippet
        self.prompt_guidelines = prompt_guidelines
        self.constrained_sampling = get_experimental_tool_sampling()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)

    async def execute(self, tool_call_id: str, params: Any, cancel: Any = None, on_update: Any = None) -> Any:
        return await self._tool.execute(tool_call_id, params, cancel, on_update, self._context)


@dataclass(slots=True, kw_only=True)
class CodingAgentSystemPromptOptions:
    """`BuildSystemPromptOptions` minus the fields the Harness factory owns."""

    custom_prompt: str | None = None
    append_system_prompt: str | None = None
    context_files: list[ContextFile] = field(default_factory=list)
    skills: list[Any] = field(default_factory=list)


@dataclass(slots=True, kw_only=True)
class CreateCodingAgentHarnessOptions:
    """`AgentHarnessOptions` sans `tool_context`, plus the coding-agent surface."""

    session: Session
    models: Any
    model: Any
    env: ExecutionEnv
    bash_command_prefix: str | None = None
    # Path to the JSONL session file exposed to default bash commands as PIDREI_SESSION_FILE.
    session_file: str | None = None
    tools: list[Any] | None = None  # CodingAgentHarnessTool
    system_prompt_options: CodingAgentSystemPromptOptions | None = None
    thinking_level: Any = None
    active_tool_names: list[str] | None = None
    system_prompt: Any = None  # str | () -> Awaitable[str]
    resources: Any = None
    stream_options: Any = None
    retry: Any = None
    compaction: Any = None
    steering_mode: Any = None
    follow_up_mode: Any = None
    tool_execution: Any = None
    drive: Any = None
    to_provider_messages: Any = None
    entry_projectors: Any = None


@dataclass(slots=True, kw_only=True)
class BuildCodingAgentHarnessSystemPromptOptions:
    cwd: str
    tools: list[Any]
    active_tool_names: list[str]
    system_prompt_options: CodingAgentSystemPromptOptions | None = None


def build_coding_agent_harness_system_prompt(options: BuildCodingAgentHarnessSystemPromptOptions) -> str:
    active_tools = [
        tool
        for name in options.active_tool_names
        if (tool := next((candidate for candidate in options.tools if candidate.name == name), None)) is not None
    ]
    tool_snippets: dict[str, str] = {}
    for tool in active_tools:
        prompt_snippet = getattr(tool, "prompt_snippet", None)
        if prompt_snippet is not None:
            prompt_snippet = re.sub(r"\s+", " ", re.sub(r"[\r\n]+", " ", prompt_snippet)).strip()
        if prompt_snippet:
            tool_snippets[tool.name] = prompt_snippet
    prompt_guidelines = [
        guideline for tool in active_tools for guideline in (getattr(tool, "prompt_guidelines", None) or [])
    ]
    prompt_options = options.system_prompt_options
    return build_system_prompt(
        BuildSystemPromptOptions(
            custom_prompt=prompt_options.custom_prompt if prompt_options is not None else None,
            append_system_prompt=prompt_options.append_system_prompt if prompt_options is not None else None,
            context_files=list(prompt_options.context_files) if prompt_options is not None else [],
            skills=list(prompt_options.skills) if prompt_options is not None else [],
            cwd=options.cwd,
            selected_tools=[tool.name for tool in active_tools],
            tool_snippets=tool_snippets,
            prompt_guidelines=prompt_guidelines,
        )
    )


async def create_coding_agent_harness(
    options: CreateCodingAgentHarnessOptions,
) -> tuple[AgentHarness, list[SuspendedOperation]]:
    harness: AgentHarness | None = None

    def get_harness() -> AgentHarness:
        if harness is None:
            raise Exception("Coding-agent Harness callback ran before Harness initialization")
        return harness

    tools = options.tools
    if tools is None:
        metadata = await options.session.get_metadata()
        tool_context = ExecutionToolContext(env=options.env)

        async def prepare(execution: Any, _context: Any, _cancel: Any) -> None:
            current_harness = get_harness()
            model = await current_harness.get_model()
            thinking_level = await current_harness.get_thinking_level()
            execution.env["PIDREI_SESSION_ID"] = metadata.id
            execution.env["PIDREI_SESSION_FILE"] = options.session_file if options.session_file is not None else ""
            execution.env["PIDREI_PROVIDER"] = model.provider
            execution.env["PIDREI_MODEL"] = model.id
            execution.env["PIDREI_REASONING_LEVEL"] = thinking_level

        tools = [
            _ContextBoundHarnessTool(
                create_read_tool(),
                tool_context,
                READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION["snippet"],
                list(READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION["guidelines"]),
            ),
            _ContextBoundHarnessTool(
                create_bash_tool(BashToolOptions(command_prefix=options.bash_command_prefix, prepare=prepare)),
                tool_context,
                BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION["snippet"],
                list(BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION["guidelines"]),
            ),
            _ContextBoundHarnessTool(
                create_edit_tool(),
                tool_context,
                EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION["snippet"],
                list(EDIT_TOOL_SYSTEM_PROMPT_CONTRIBUTION["guidelines"]),
            ),
            _ContextBoundHarnessTool(
                create_write_tool(),
                tool_context,
                WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION["snippet"],
                list(WRITE_TOOL_SYSTEM_PROMPT_CONTRIBUTION["guidelines"]),
            ),
        ]
    active_tool_names = (
        list(options.active_tool_names) if options.active_tool_names is not None else [tool.name for tool in tools]
    )

    system_prompt: str | Callable[[], Awaitable[str]] | None = options.system_prompt
    if system_prompt is None:

        async def default_system_prompt() -> str:
            current_harness = get_harness()
            current_tools = await current_harness.get_tools()
            current_active_tool_names = await current_harness.get_active_tools()
            return build_coding_agent_harness_system_prompt(
                BuildCodingAgentHarnessSystemPromptOptions(
                    cwd=options.env.cwd,
                    tools=current_tools,
                    active_tool_names=current_active_tool_names,
                    system_prompt_options=options.system_prompt_options,
                )
            )

        system_prompt = default_system_prompt

    created = await AgentHarness.create(
        AgentHarnessOptions(
            session=options.session,
            models=options.models,
            model=options.model,
            thinking_level=options.thinking_level,
            tools=tools,
            active_tool_names=active_tool_names,
            system_prompt=system_prompt,
            resources=options.resources,
            stream_options=options.stream_options,
            retry=options.retry,
            compaction=options.compaction,
            steering_mode=options.steering_mode,
            follow_up_mode=options.follow_up_mode,
            tool_execution=options.tool_execution,
            drive=options.drive,
            to_provider_messages=options.to_provider_messages,
            entry_projectors=options.entry_projectors,
        )
    )
    harness = created[0]
    return created
