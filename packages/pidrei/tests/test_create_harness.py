"""Mirror of pi coding-agent test/server/create-harness.test.ts."""

import os
from dataclasses import dataclass, field
from typing import Any

import pytest

from pidrei.core.skills import Skill
from pidrei.core.source_info import SourceInfo
from pidrei.core.system_prompt import ContextFile
from pidrei.server.create_harness import (
    BuildCodingAgentHarnessSystemPromptOptions,
    CodingAgentSystemPromptOptions,
    CreateCodingAgentHarnessOptions,
    build_coding_agent_harness_system_prompt,
    create_coding_agent_harness,
)
from pidrei_agent.harness.agent_harness import AgentHarness
from pidrei_agent.harness.env.local import LocalExecutionEnv
from pidrei_agent.harness.session.memory import InMemorySessionStorage
from pidrei_agent.harness.session.session import Session
from pidrei_agent.harness.session.types import SessionMetadata
from pidrei_agent.harness.types import AgentHarnessStreamOptions, ShellExecOptions
from pidrei_agent.types import AgentToolResult
from pidrei_ai.models_generated import MODELS as CATALOG
from pidrei_ai.registry import create_models
from pidrei_ai.types import TextContent
from pidrei_ai.utils.retry import RetryPolicy


MODELS = create_models()


def get_model(provider: str, model_id: str):
    """Upstream's `getModel` from pi-ai/compat: a generated-catalog read."""
    return next(model for model in CATALOG[provider] if model.id == model_id)


class CapturingExecutionEnv(LocalExecutionEnv):
    def __init__(self, cwd: str, shell_env: dict[str, str] | None = None):
        super().__init__(cwd, shell_env=shell_env)
        self.execution_overrides: dict[str, str] | None = None

    async def exec(self, command: str, options: ShellExecOptions | None = None):
        self.execution_overrides = dict(options.env) if options is not None else None
        return await super().exec(command, options)


async def resolve_system_prompt(system_prompt: Any) -> str:
    if isinstance(system_prompt, str):
        return system_prompt
    if system_prompt is None:
        raise Exception("Expected a system prompt callback")
    return await system_prompt()


@dataclass(slots=True)
class PromptTool:
    __test__ = False

    name: str
    label: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    prompt_snippet: str | None = None
    prompt_guidelines: list[str] | None = None

    async def execute(self, tool_call_id, params, cancel=None, on_update=None, context=None):
        return AgentToolResult(content=[TextContent(text="ok")], details=None)


def create_prompt_tool(name: str, prompt_snippet: str | None = None, prompt_guidelines: list[str] | None = None):
    return PromptTool(
        name=name,
        label=name,
        description=f"{name} description",
        prompt_snippet=prompt_snippet,
        prompt_guidelines=prompt_guidelines,
    )


DEFAULT_PROMPT_TOOLS = [
    create_prompt_tool("read", "Read file contents", ["Use read to examine files instead of cat or sed."]),
    create_prompt_tool(
        "bash",
        "Execute bash commands (ls, grep, find, etc.)",
        ["You can inspect PIDREI_* environment variables for current model and session details."],
    ),
    create_prompt_tool("edit", "Edit files", ["Edit carefully."]),
    create_prompt_tool("write", "Create or overwrite files", ["Use write only for new files or complete rewrites."]),
]


def create_session(id: str) -> Session:
    return Session(InMemorySessionStorage(SessionMetadata(id=id, created_at=1)))


@pytest.mark.tonio
async def test_adds_coding_agent_policy_to_explicit_harness_options():
    session = create_session("harness-session")
    env = LocalExecutionEnv("/workspace")
    stream_options = AgentHarnessStreamOptions(timeout_ms=123)
    harness, suspended = await create_coding_agent_harness(
        CreateCodingAgentHarnessOptions(
            session=session,
            models=MODELS,
            model=get_model("google", "gemini-2.5-flash"),
            thinking_level="high",
            env=env,
            stream_options=stream_options,
            retry=RetryPolicy(enabled=True, max_retries=2, base_delay_ms=10),
            steering_mode="all",
            follow_up_mode="all",
        )
    )
    try:
        assert suspended == []
        assert await harness.get_active_tools() == ["read", "bash", "edit", "write"]
        assert [tool.name for tool in await harness.get_tools()] == ["read", "bash", "edit", "write"]
        assert await harness.get_stream_options() == stream_options
        assert await harness.get_retry_policy() == RetryPolicy(enabled=True, max_retries=2, base_delay_ms=10)
        assert await harness.get_steering_mode() == "all"
        assert await harness.get_follow_up_mode() == "all"
    finally:
        await harness.close()
        await env.cleanup()


def test_preserves_coding_agent_prompt_snippets_and_guideline_order():
    prompt = build_coding_agent_harness_system_prompt(
        BuildCodingAgentHarnessSystemPromptOptions(
            cwd="/workspace",
            tools=DEFAULT_PROMPT_TOOLS,
            active_tool_names=["read", "bash", "edit", "write"],
        )
    )
    assert "- read: Read file contents" in prompt
    assert "- bash: Execute bash commands (ls, grep, find, etc.)" in prompt
    assert "Use read to examine files instead of cat or sed." in prompt
    assert "You can inspect PIDREI_* environment variables for current model and session details." in prompt
    assert prompt.index("Use read to examine files") < prompt.index("You can inspect PIDREI_* environment variables")


@pytest.mark.tonio
async def test_preserves_caller_supplied_tools_and_activation():
    session = create_session("custom-harness-session")
    env = LocalExecutionEnv("/workspace")
    custom_tool = create_prompt_tool("inspect")
    custom_tool.description = "Inspect the configured service"
    harness, _suspended = await create_coding_agent_harness(
        CreateCodingAgentHarnessOptions(
            session=session,
            models=MODELS,
            model=get_model("google", "gemini-2.5-flash"),
            env=env,
            tools=[custom_tool],
            active_tool_names=[],
            system_prompt="Server-owned prompt",
        )
    )
    try:
        assert [tool.name for tool in await harness.get_tools()] == ["inspect"]
        assert await harness.get_active_tools() == []
    finally:
        await harness.close()
        await env.cleanup()


@pytest.mark.tonio
async def test_sets_the_optional_session_file_in_the_default_bash_tool_environment():
    session = create_session("session-file-harness")
    env = CapturingExecutionEnv(
        os.getcwd(),
        shell_env={**os.environ, "PIDREI_SESSION_FILE": "/stale/parent.jsonl", "PIDREI_CODING_AGENT": "true"},
    )
    harness, _suspended = await create_coding_agent_harness(
        CreateCodingAgentHarnessOptions(
            session=session,
            models=MODELS,
            model=get_model("google", "gemini-2.5-flash"),
            thinking_level="high",
            env=env,
            session_file="/sessions/current.jsonl",
        )
    )
    try:
        bash = next((tool for tool in await harness.get_tools() if tool.name == "bash"), None)
        assert bash is not None, "Expected the default bash tool"

        result = await bash.execute(
            "bash-call",
            {
                "command": (
                    "printf '%s' \"$PIDREI_SESSION_ID|$PIDREI_SESSION_FILE|$PIDREI_PROVIDER"
                    '|$PIDREI_MODEL|$PIDREI_REASONING_LEVEL|$PIDREI_CODING_AGENT"'
                )
            },
        )

        assert env.execution_overrides == {
            "PIDREI_SESSION_ID": "session-file-harness",
            "PIDREI_SESSION_FILE": "/sessions/current.jsonl",
            "PIDREI_PROVIDER": "google",
            "PIDREI_MODEL": "gemini-2.5-flash",
            "PIDREI_REASONING_LEVEL": "high",
        }
        assert result.content == [
            TextContent(text="session-file-harness|/sessions/current.jsonl|google|gemini-2.5-flash|high|true")
        ]
    finally:
        await harness.close()
        await env.cleanup()


@pytest.mark.tonio
async def test_keeps_bash_pidrei_model_variables_synchronized_with_harness_state():
    session = create_session("dynamic-bash-session")
    env = CapturingExecutionEnv(
        os.getcwd(),
        shell_env={**os.environ, "PIDREI_SESSION_FILE": "/stale/parent.jsonl", "PIDREI_CODING_AGENT": "true"},
    )
    harness, _suspended = await create_coding_agent_harness(
        CreateCodingAgentHarnessOptions(
            session=session,
            models=MODELS,
            model=get_model("google", "gemini-2.5-flash"),
            thinking_level="high",
            env=env,
        )
    )
    try:
        await harness.set_model(get_model("anthropic", "claude-sonnet-4-5"))
        await harness.set_thinking_level("low")
        bash = next((tool for tool in await harness.get_tools() if tool.name == "bash"), None)
        assert bash is not None, "Expected the default bash tool"

        result = await bash.execute(
            "bash-call",
            {
                "command": (
                    'printf \'%s:%s\' "${PIDREI_SESSION_FILE+x}" "$PIDREI_SESSION_ID|$PIDREI_PROVIDER'
                    '|$PIDREI_MODEL|$PIDREI_REASONING_LEVEL|$PIDREI_CODING_AGENT"'
                )
            },
        )

        assert env.execution_overrides == {
            "PIDREI_SESSION_ID": "dynamic-bash-session",
            "PIDREI_SESSION_FILE": "",
            "PIDREI_PROVIDER": "anthropic",
            "PIDREI_MODEL": "claude-sonnet-4-5",
            "PIDREI_REASONING_LEVEL": "low",
        }
        assert "PIDREI_SESSION_FILE" in (env.execution_overrides or {})
        assert env.execution_overrides["PIDREI_SESSION_FILE"] == ""
        assert result.content == [TextContent(text="x:dynamic-bash-session|anthropic|claude-sonnet-4-5|low|true")]
    finally:
        await harness.close()
        await env.cleanup()


@pytest.mark.tonio
async def test_builds_each_default_system_prompt_from_current_harness_tool_metadata():
    original_create = AgentHarness.create
    configured: list[Any] = []

    async def capturing_create(options):
        configured.append(options.system_prompt)
        return await original_create(options)

    AgentHarness.create = staticmethod(capturing_create)
    session = create_session("dynamic-prompt-session")
    env = LocalExecutionEnv("/workspace")
    try:
        harness, _suspended = await create_coding_agent_harness(
            CreateCodingAgentHarnessOptions(
                session=session,
                models=MODELS,
                model=get_model("google", "gemini-2.5-flash"),
                env=env,
            )
        )
        AgentHarness.create = original_create
        configured_system_prompt = configured[-1]
        try:
            initial_prompt = await resolve_system_prompt(configured_system_prompt)
            assert "- read: Read file contents" in initial_prompt
            assert "- bash: Execute bash commands (ls, grep, find, etc.)" in initial_prompt
            assert "- edit: Make precise file edits with exact text replacement" in initial_prompt
            assert "- write: Create or overwrite files" in initial_prompt

            await harness.set_active_tools(["write"])
            write_prompt = await resolve_system_prompt(configured_system_prompt)
            assert "- write: Create or overwrite files" in write_prompt
            assert "- read:" not in write_prompt
            assert "- bash:" not in write_prompt

            read = next((tool for tool in await harness.get_tools() if tool.name == "read"), None)
            assert read is not None, "Expected the default read tool"
            await harness.set_tools([read])
            read_prompt = await resolve_system_prompt(configured_system_prompt)
            assert "- read: Read file contents" in read_prompt
            assert "- write:" not in read_prompt

            inspect_tool = create_prompt_tool("inspect")
            inspect_tool.description = "Inspect the configured service"
            inspect_tool.prompt_snippet = "  Inspect\nthe   configured service  "
            inspect_tool.prompt_guidelines = ["Use inspect for service diagnostics."]
            await harness.set_tools([inspect_tool])
            inspect_prompt = await resolve_system_prompt(configured_system_prompt)
            assert "- inspect: Inspect the configured service" in inspect_prompt
            assert "Use inspect for service diagnostics." in inspect_prompt
        finally:
            await harness.close()
            await env.cleanup()
    finally:
        AgentHarness.create = original_create


def test_omits_active_custom_tools_without_prompt_metadata_from_the_textual_tools_section():
    prompt = build_coding_agent_harness_system_prompt(
        BuildCodingAgentHarnessSystemPromptOptions(
            cwd="/workspace",
            tools=[create_prompt_tool("hidden")],
            active_tool_names=["hidden"],
        )
    )

    assert "Available tools:\n(none)" in prompt
    assert "- hidden:" not in prompt
    assert "hidden description" not in prompt


@pytest.mark.parametrize(
    ("name", "built_in_snippet", "built_in_guideline"),
    [
        (
            "bash",
            "Execute bash commands (ls, grep, find, etc.)",
            "You can inspect PIDREI_* environment variables for current model and session details.",
        ),
        ("read", "Read file contents", "Use read to examine files instead of cat or sed."),
        (
            "edit",
            "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
            "Use edit for precise changes (edits[].oldText must match exactly)",
        ),
        ("write", "Create or overwrite files", "Use write only for new files or complete rewrites."),
    ],
)
def test_does_not_infer_prompt_metadata_for_a_caller_supplied_replacement(name, built_in_snippet, built_in_guideline):
    prompt = build_coding_agent_harness_system_prompt(
        BuildCodingAgentHarnessSystemPromptOptions(
            cwd="/workspace",
            tools=[create_prompt_tool(name)],
            active_tool_names=[name],
        )
    )

    assert "Available tools:\n(none)" in prompt
    assert built_in_snippet not in prompt
    assert built_in_guideline not in prompt


def test_builds_the_default_prompt_from_active_tools_and_resolved_prompt_resources():
    prompt = build_coding_agent_harness_system_prompt(
        BuildCodingAgentHarnessSystemPromptOptions(
            cwd="/workspace",
            tools=DEFAULT_PROMPT_TOOLS,
            active_tool_names=["write", "read"],
            system_prompt_options=CodingAgentSystemPromptOptions(
                context_files=[ContextFile(path="/workspace/AGENTS.md", content="Follow project policy.")],
                skills=[
                    Skill(
                        name="review",
                        description="Review server changes",
                        file_path="/skills/review/SKILL.md",
                        base_dir="/skills/review",
                        source_info=SourceInfo(
                            path="/skills/review/SKILL.md",
                            source="test",
                            scope="temporary",
                            origin="top-level",
                        ),
                        disable_model_invocation=False,
                    )
                ],
            ),
        )
    )

    assert "- write: Create or overwrite files" in prompt
    assert "- read: Read file contents" in prompt
    assert "- bash:" not in prompt
    assert "You can inspect PIDREI_* environment variables" not in prompt
    assert '<project_instructions path="/workspace/AGENTS.md">' in prompt
    assert "<name>review</name>" in prompt
    assert prompt.index("Use write only for new files or complete rewrites.") < prompt.index(
        "Use read to examine files instead of cat or sed."
    )
