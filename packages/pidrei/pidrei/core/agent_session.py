"""Mirror of pi coding-agent src/core/agent-session.ts.

AgentSession - core abstraction for agent lifecycle and session management,
shared between all run modes (print, rpc; interactive is Phase 4). It
encapsulates agent state access, event subscription with automatic session
persistence, model and thinking-level management, compaction (manual and
auto), bash execution, and tree navigation. Modes add their own I/O layer.

pi's six AbortController scopes map to named CancelTokens: prompt (owned by
the Agent), compaction, auto-compaction, branch-summary, retry, and bash.

export_to_html landed with the Phase 4 export-html slice; export_to_jsonl
is here.
"""

import json
import os
import re
import threading
import time as time_module
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace as dataclass_replace
from datetime import UTC, datetime
from typing import Any, Literal

import tonio.colored as tonio
from tonio.colored import fs

from pidrei_agent.agent import Agent
from pidrei_agent.types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopTurnUpdate,
    AgentTool,
    BeforeToolCallContext,
    BeforeToolCallResult,
    MessageEndEvent as AgentMessageEndEvent,
    MessageStartEvent as AgentMessageStartEvent,
    PrepareNextTurnContext,
)
from pidrei_ai.registry import clamp_thinking_level, get_supported_thinking_levels, models_are_equal
from pidrei_ai.types import AssistantMessage, ImageContent, Model, TextContent, Usage, UserMessage
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.overflow import is_context_overflow
from pidrei_ai.utils.retry import RetryCallbacks, RetryPolicy, is_retryable_assistant_error
from pidrei_ai.utils.session_resources import cleanup_session_resources
from pidrei_ai.utils.text import content_text

from ..utils.frontmatter import strip_frontmatter
from ..utils.paths import resolve_path
from ..utils.sleep import sleep
from .auth_guidance import format_no_api_key_found_message, format_no_model_selected_message
from .bash_executor import BashResult, execute_bash_with_operations
from .compaction import (
    CompactionResult,
    CompactionSettings,
    calculate_context_tokens,
    collect_entries_for_branch_summary,
    compact as run_compact,
    estimate_context_tokens,
    estimate_tokens,
    generate_branch_summary,
    prepare_compaction,
    should_compact,
)
from .defaults import DEFAULT_THINKING_LEVEL
from .extensions import ExtensionRunner, wrap_registered_tools
from .extensions.runner import emit_session_shutdown_event
from .messages import BashExecutionMessage, CustomMessage
from .model_registry import ModelRegistry
from .prompt_templates import expand_prompt_template
from .session_manager import CURRENT_SESSION_VERSION, get_latest_compaction_entry
from .source_info import create_synthetic_source_info
from .system_prompt import BuildSystemPromptOptions, build_system_prompt
from .tools import create_all_tool_definitions, create_local_bash_operations
from .tools.tool_definition_wrapper import create_tool_definition_from_agent_tool


def _now_ms() -> int:

    return int(time_module.time() * 1000)


def _iso_now() -> str:

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _iso_to_epoch_ms(timestamp: Any) -> float:
    try:
        return datetime.fromisoformat(timestamp).timestamp() * 1000
    except TypeError, ValueError:
        return float("nan")


# ============================================================================
# Skill Block Parsing
# ============================================================================


@dataclass(slots=True)
class ParsedSkillBlock:
    """Parsed skill block from a user message."""

    name: str
    location: str
    content: str
    user_message: str | None


_SKILL_BLOCK_RE = re.compile(r'^<skill name="([^"]+)" location="([^"]+)">\n([\s\S]*?)\n</skill>(?:\n\n([\s\S]+))?$')


def parse_skill_block(text: str) -> ParsedSkillBlock | None:
    """Parse a skill block from message text. Returns None if absent."""
    match = _SKILL_BLOCK_RE.match(text)
    if not match:
        return None
    user_message = (match.group(4) or "").strip() or None
    return ParsedSkillBlock(
        name=match.group(1), location=match.group(2), content=match.group(3), user_message=user_message
    )


# ============================================================================
# Session-specific events extending the core AgentEvent
# ============================================================================


@dataclass(slots=True)
class SessionAgentEndEvent:
    """agent_end enriched with the session's retry decision."""

    messages: list[Any]
    will_retry: bool
    type: Literal["agent_end"] = "agent_end"


@dataclass(slots=True)
class AgentSettledEvent:
    type: Literal["agent_settled"] = "agent_settled"


@dataclass(slots=True)
class QueueUpdateEvent:
    steering: list[str]
    follow_up: list[str]
    type: Literal["queue_update"] = "queue_update"


@dataclass(slots=True)
class CompactionStartEvent:
    reason: str  # "manual" | "threshold" | "overflow"
    type: Literal["compaction_start"] = "compaction_start"


@dataclass(slots=True)
class EntryAppendedEvent:
    entry: dict[str, Any]
    type: Literal["entry_appended"] = "entry_appended"


@dataclass(slots=True)
class SessionInfoChangedEvent:
    name: str | None
    type: Literal["session_info_changed"] = "session_info_changed"


@dataclass(slots=True)
class ThinkingLevelChangedEvent:
    level: str
    type: Literal["thinking_level_changed"] = "thinking_level_changed"


@dataclass(slots=True)
class CompactionEndEvent:
    reason: str
    result: CompactionResult | None
    aborted: bool
    will_retry: bool
    error_message: str | None = None
    type: Literal["compaction_end"] = "compaction_end"


@dataclass(slots=True)
class AutoRetryStartEvent:
    attempt: int
    max_attempts: int
    delay_ms: float
    error_message: str
    type: Literal["auto_retry_start"] = "auto_retry_start"


@dataclass(slots=True)
class AutoRetryEndEvent:
    success: bool
    attempt: int
    final_error: str | None = None
    type: Literal["auto_retry_end"] = "auto_retry_end"


@dataclass(slots=True)
class SummarizationRetryScheduledEvent:
    attempt: int
    max_attempts: int
    delay_ms: float
    error_message: str
    type: Literal["summarization_retry_scheduled"] = "summarization_retry_scheduled"


@dataclass(slots=True)
class SummarizationRetryAttemptStartEvent:
    source: str  # "branchSummary" | "compaction"
    reason: str | None = None  # compaction only: "manual" | "threshold" | "overflow"
    type: Literal["summarization_retry_attempt_start"] = "summarization_retry_attempt_start"


@dataclass(slots=True)
class SummarizationRetryFinishedEvent:
    type: Literal["summarization_retry_finished"] = "summarization_retry_finished"


@dataclass(slots=True)
class BashExecutionUpdateEvent:
    delta: str
    id: str | None = None
    type: Literal["bash_execution_update"] = "bash_execution_update"


AgentSessionEventListener = Callable[[Any], None]


# ============================================================================
# Types
# ============================================================================


def _without_deleted_headers(headers: dict[str, Any] | None) -> dict[str, str] | None:
    if headers is None:
        return None
    return {key: value for key, value in headers.items() if value is not None}


@dataclass(slots=True)
class ScopedModel:
    model: Model
    thinking_level: str | None = None


@dataclass(slots=True, kw_only=True)
class AgentSessionConfig:
    agent: Agent
    session_manager: Any
    settings_manager: Any
    cwd: str
    # Resource loader for extensions, skills, prompts, themes, context files, system prompt
    resource_loader: Any
    # Canonical model/auth runtime used by coding-agent internals.
    model_runtime: Any
    # Models to cycle through (from --models flag)
    scoped_models: list[ScopedModel] | None = None
    # SDK custom tools registered outside extensions
    custom_tools: list[Any] | None = None
    # Initial active built-in tool names. Default: [read, bash, edit, write]
    initial_active_tool_names: list[str] | None = None
    # Optional allowlist of tool names.
    allowed_tool_names: list[str] | None = None
    # Optional denylist of tool names.
    excluded_tool_names: list[str] | None = None
    # Override base tools (useful for custom runtimes).
    base_tools_override: dict[str, AgentTool] | None = None
    # Mutable ref used by the Agent stream fn to access the current ExtensionRunner
    extension_runner_ref: Any = None
    # Session start event metadata emitted when extensions bind to this runtime.
    session_start_event: dict[str, Any] | None = None


@dataclass(slots=True, kw_only=True)
class ExtensionBindings:
    ui_context: Any = None
    mode: str | None = None
    command_context_actions: dict[str, Any] | None = None
    abort_handler: Callable[[], None] | None = None
    shutdown_handler: Callable[[], None] | None = None
    on_error: Callable[[Any], None] | None = None


@dataclass(slots=True, kw_only=True)
class PromptOptions:
    """Options for AgentSession.prompt()."""

    # Whether to expand file-based prompt templates (default: True)
    expand_prompt_templates: bool = True
    # Image attachments
    images: list[ImageContent] | None = None
    # When streaming, how to queue: "steer" (interrupt) or "followUp" (wait).
    streaming_behavior: str | None = None
    # Source of input for extension input event handlers.
    source: str = "interactive"
    # Internal hook used by RPC mode to observe prompt preflight acceptance.
    preflight_result: Callable[[bool], None] | None = None


@dataclass(slots=True)
class ModelCycleResult:
    model: Model
    thinking_level: str
    # Whether cycling through scoped models (--models flag) or all available
    is_scoped: bool


@dataclass(slots=True)
class SessionTokens:
    input: int
    output: int
    cache_read: int
    cache_write: int
    total: int


@dataclass(slots=True)
class ContextUsage:
    tokens: int | None
    context_window: int
    percent: float | None


@dataclass(slots=True)
class SessionStats:
    """Session statistics for /session command."""

    session_file: str | None
    session_id: str
    user_messages: int
    assistant_messages: int
    tool_calls: int
    tool_results: int
    total_messages: int
    tokens: SessionTokens
    cost: float
    context_usage: ContextUsage | None = None


@dataclass(slots=True)
class ToolInfo:
    name: str
    description: str
    parameters: dict[str, Any]
    prompt_guidelines: list[str] | None
    source_info: Any


@dataclass(slots=True)
class _ToolDefinitionEntry:
    definition: Any
    source_info: Any


@dataclass(slots=True)
class NavigateTreeResult:
    cancelled: bool
    editor_text: str | None = None
    aborted: bool | None = None
    summary_entry: dict[str, Any] | None = None


def _estimate_messages_tokens(messages: list[Any]) -> int:
    return sum(estimate_tokens(message) for message in messages)


def _compaction_settings_from(settings: dict[str, Any]) -> CompactionSettings:
    return CompactionSettings(
        enabled=settings["enabled"],
        reserve_tokens=settings["reserve_tokens"],
        keep_recent_tokens=settings["keep_recent_tokens"],
    )


def _retry_policy_from(settings: dict[str, Any]) -> RetryPolicy:
    return RetryPolicy(
        enabled=settings["enabled"],
        max_retries=settings["max_retries"],
        base_delay_ms=settings["base_delay_ms"],
    )


# ============================================================================
# Constants
# ============================================================================

# Standard thinking levels
_THINKING_LEVELS = ["off", "minimal", "low", "medium", "high"]


# ============================================================================
# AgentSession
# ============================================================================


class AgentSession:
    def __init__(self, config: AgentSessionConfig):
        self.agent = config.agent
        self.session_manager = config.session_manager
        self.settings_manager = config.settings_manager

        self._scoped_models: list[ScopedModel] = config.scoped_models or []

        # Event subscription state
        self._unsubscribe_agent: Callable[[], None] | None = None
        self._event_listeners: list[AgentSessionEventListener] = []
        self._is_agent_run_active = False
        self._idle_wait_event: tonio.Event | None = None
        self._state_guard = threading.RLock()

        # Pending steering/follow-up messages tracked for UI display.
        self._steering_messages: list[str] = []
        self._follow_up_messages: list[str] = []
        # Messages queued to be included with the next user prompt as context.
        self._pending_next_turn_messages: list[CustomMessage] = []

        # Compaction state
        self._compaction_cancel: CancelToken | None = None
        self._auto_compaction_cancel: CancelToken | None = None
        self._overflow_recovery_attempted = False

        # Branch summarization state
        self._branch_summary_cancel: CancelToken | None = None

        # Retry state
        self._retry_cancel: CancelToken | None = None
        self._retry_attempt = 0

        # Bash execution state
        self._bash_cancel: CancelToken | None = None
        self._pending_bash_messages: list[BashExecutionMessage] = []

        # Extension system
        self._extension_runner: ExtensionRunner | None = None
        self._turn_index = 0

        self._resource_loader = config.resource_loader
        self._custom_tools: list[Any] = config.custom_tools or []
        self._base_tool_definitions: dict[str, Any] = {}
        self._cwd = config.cwd
        self._extension_runner_ref = config.extension_runner_ref
        self._initial_active_tool_names = config.initial_active_tool_names
        self._allowed_tool_names = set(config.allowed_tool_names) if config.allowed_tool_names is not None else None
        self._excluded_tool_names = set(config.excluded_tool_names) if config.excluded_tool_names is not None else None
        self._base_tools_override = config.base_tools_override
        self._session_start_event = (
            config.session_start_event
            if config.session_start_event is not None
            else {"type": "session_start", "reason": "startup"}
        )
        self._extension_ui_context: Any = None
        self._extension_mode = "print"
        self._extension_command_context_actions: dict[str, Any] | None = None
        self._extension_abort_handler: Callable[[], None] | None = None
        self._extension_shutdown_handler: Callable[[], None] | None = None
        self._extension_error_listener: Callable[[Any], None] | None = None
        self._extension_error_unsubscriber: Callable[[], None] | None = None

        self._model_runtime = config.model_runtime

        # Tool registry for extension getTools/setTools
        self._tool_registry: dict[str, AgentTool] = {}
        self._tool_definitions: dict[str, _ToolDefinitionEntry] = {}
        self._tool_prompt_snippets: dict[str, str] = {}
        self._tool_prompt_guidelines: dict[str, list[str]] = {}

        # Base system prompt (without extension appends)
        self._base_system_prompt = ""
        self._base_system_prompt_options = BuildSystemPromptOptions(cwd=self._cwd)
        self._system_prompt_override: str | None = None

        # Track last assistant message for auto-compaction check
        self._last_assistant_message: AssistantMessage | None = None

        # Always subscribe to agent events for internal handling
        # (session persistence, extensions, auto-compaction, retry logic)
        self._unsubscribe_agent = self.agent.subscribe(self._handle_agent_event)
        self._install_agent_tool_hooks()
        self._install_agent_next_turn_refresh()

        self._build_runtime(
            active_tool_names=self._initial_active_tool_names,
            include_all_extension_tools=True,
        )

    @property
    def model_runtime(self) -> Any:
        return self._model_runtime

    async def _get_required_request_auth(self, model: Model) -> dict[str, Any]:
        try:
            result = await self._model_runtime.get_auth(model)
        except Exception as error:
            cause = error.__cause__
            if isinstance(cause, Exception) and str(cause) == "authHeader requires a resolved API key":
                raise Exception(format_no_api_key_found_message(model.provider))
            if str(error) == "authHeader requires a resolved API key":
                raise Exception(format_no_api_key_found_message(model.provider))
            raise
        if result is not None and (result.auth.api_key or result.auth.headers):
            return {
                "api_key": result.auth.api_key,
                "headers": _without_deleted_headers(result.auth.headers),
                "env": dict(result.env) if result.env is not None else None,
            }

        if self._model_runtime.is_using_oauth(model.provider):
            raise Exception(
                f'Authentication failed for "{model.provider}". '
                "Credentials may have expired or network is unavailable. "
                f"Run '/login {model.provider}' to re-authenticate."
            )
        raise Exception(format_no_api_key_found_message(model.provider))

    def _uses_default_stream_simple(self) -> bool:
        """pi checks `agent.streamFunction === streamSimple` (the pi-ai compat
        global). Sessions built by pidrei's sdk always install a runtime-bound
        stream closure — exactly like pi's sdk — so the check is False for
        every session this package creates; the compat global itself is not
        ported."""
        return False

    async def _get_summarization_request_auth(self, model: Model) -> dict[str, Any]:
        if self._uses_default_stream_simple():
            return await self._get_required_request_auth(model)

        try:
            result = await self._model_runtime.get_auth(model)
            if result is None:
                return {"api_key": None, "headers": None, "env": None}
            return {
                "api_key": result.auth.api_key,
                "headers": _without_deleted_headers(result.auth.headers),
                "env": dict(result.env) if result.env is not None else None,
            }
        except Exception:
            return {"api_key": None, "headers": None, "env": None}

    def _install_agent_tool_hooks(self) -> None:
        """Install tool hooks once on the Agent instance.

        The callbacks read `self._extension_runner` at execution time, so
        extension reload swaps in the new runner without reinstalling hooks."""

        async def before_tool_call(ctx: BeforeToolCallContext, _cancel=None):
            runner = self._extension_runner
            if not runner.has_handlers("tool_call"):
                return None

            try:
                result = await runner.emit_tool_call(
                    {
                        "type": "tool_call",
                        "toolName": ctx.tool_call.name,
                        "toolCallId": ctx.tool_call.id,
                        "input": ctx.args,
                    }
                )
            except Exception as err:
                if isinstance(err, Exception):
                    raise
                raise Exception(f"Extension failed, blocking execution: {err}")
            if result is None:
                return None

            return BeforeToolCallResult(block=result.get("block"), reason=result.get("reason"))

        async def after_tool_call(ctx: AfterToolCallContext, _cancel=None):
            runner = self._extension_runner
            if not runner.has_handlers("tool_result"):
                return None

            hook_result = await runner.emit_tool_result(
                {
                    "type": "tool_result",
                    "toolName": ctx.tool_call.name,
                    "toolCallId": ctx.tool_call.id,
                    "input": ctx.args,
                    "content": ctx.result.content,
                    "details": ctx.result.details,
                    "isError": ctx.is_error,
                    "usage": ctx.result.usage,
                }
            )

            if hook_result is None:
                return None

            return AfterToolCallResult(
                content=hook_result.get("content"),
                details=hook_result.get("details"),
                is_error=hook_result.get("isError") if hook_result.get("isError") is not None else ctx.is_error,
                usage=hook_result.get("usage"),
            )

        self.agent.before_tool_call = before_tool_call
        self.agent.after_tool_call = after_tool_call

    def _install_agent_next_turn_refresh(self) -> None:
        previous_with_context = self.agent.prepare_next_turn_with_context
        if previous_with_context is None and self.agent.prepare_next_turn is not None:
            previous_plain = self.agent.prepare_next_turn

            async def adapted(_turn: PrepareNextTurnContext, cancel=None):
                return await previous_plain(cancel)

            previous_with_context = adapted

        async def prepare_next_turn_with_context(turn: PrepareNextTurnContext, cancel=None):
            previous_snapshot = None
            if previous_with_context is not None:
                previous_snapshot = await previous_with_context(turn, cancel)
            previous_context = (
                previous_snapshot.context
                if previous_snapshot is not None and previous_snapshot.context
                else turn.context
            )

            system_prompt = (
                self._system_prompt_override if self._system_prompt_override is not None else self._base_system_prompt
            )
            return AgentLoopTurnUpdate(
                context=AgentContext(
                    system_prompt=system_prompt,
                    messages=previous_context.messages,
                    tools=list(self.agent.state.tools),
                ),
                model=self.agent.state.model,
                thinking_level=self.agent.state.thinking_level,
            )

        self.agent.prepare_next_turn_with_context = prepare_next_turn_with_context

    # =========================================================================
    # Event Subscription
    # =========================================================================

    def _emit(self, event: Any) -> None:
        for listener in list(self._event_listeners):
            listener(event)

    def _emit_queue_update(self) -> None:
        self._emit(QueueUpdateEvent(steering=list(self._steering_messages), follow_up=list(self._follow_up_messages)))

    def _get_idle_wait_event(self) -> tonio.Event:
        with self._state_guard:
            if self._idle_wait_event is None:
                self._idle_wait_event = tonio.Event()
            return self._idle_wait_event

    def _resolve_idle_wait_if_idle(self) -> None:
        with self._state_guard:
            if self._is_agent_run_active or self._idle_wait_event is None:
                return
            event = self._idle_wait_event
            self._idle_wait_event = None
        event.set()

    async def _emit_agent_settled(self) -> None:
        # `_is_agent_run_active` is cleared by the caller, before the pending
        # bash flush (see `_run_agent_prompt`).
        try:
            await self._extension_runner.emit({"type": "agent_settled"})
            self._emit(AgentSettledEvent())
        finally:
            self._resolve_idle_wait_if_idle()

    async def _handle_agent_event(self, event: AgentEvent, _cancel=None) -> None:
        """Internal handler for agent events - shared by subscribe and reconnect."""
        # When a user message starts, check if it's from either queue and remove it
        # BEFORE emitting so the UI sees the updated queue state.
        if event.type == "message_start" and getattr(event.message, "role", None) == "user":
            self._overflow_recovery_attempted = False
            message_text = content_text(event.message.content, "")
            if message_text:
                if message_text in self._steering_messages:
                    self._steering_messages.remove(message_text)
                    self._emit_queue_update()
                elif message_text in self._follow_up_messages:
                    self._follow_up_messages.remove(message_text)
                    self._emit_queue_update()

        # Emit to extensions first
        await self._emit_extension_event(event)

        # Notify all listeners
        if event.type == "agent_end":
            self._emit(
                SessionAgentEndEvent(messages=event.messages, will_retry=self._will_retry_after_agent_end(event))
            )
        else:
            self._emit(event)

        # Handle session persistence
        if event.type == "message_end":
            message = event.message
            role = getattr(message, "role", None)
            if role == "custom":
                # Persist as CustomMessageEntry
                await self.session_manager.append_custom_message_entry(
                    message.custom_type,
                    message.content,
                    message.display,
                    message.details,
                )
            elif role in ("user", "assistant", "toolResult"):
                # Regular LLM message - persist as SessionMessageEntry
                await self.session_manager.append_message(message)
            # Other message types (bashExecution, compactionSummary, branchSummary)
            # are persisted elsewhere.

            # Track assistant message for auto-compaction (checked on agent_end)
            if role == "assistant":
                self._last_assistant_message = message

                if message.stop_reason != "error":
                    self._overflow_recovery_attempted = False

                # Reset retry counter immediately on successful assistant response.
                # This prevents accumulation across multiple LLM calls within a turn.
                if message.stop_reason != "error" and self._retry_attempt > 0:
                    self._emit(AutoRetryEndEvent(success=True, attempt=self._retry_attempt))
                    self._retry_attempt = 0

    def _will_retry_after_agent_end(self, event: AgentEndEvent) -> bool:
        settings = self.settings_manager.get_retry_settings()
        if not settings["enabled"] or self._retry_attempt >= settings["max_retries"]:
            return False

        for message in reversed(event.messages):
            if getattr(message, "role", None) == "assistant":
                return self._is_retryable_error(message)
        return False

    def _find_last_assistant_message(self) -> AssistantMessage | None:
        """Find the last assistant message in agent state (including aborted ones)."""
        for message in reversed(self.agent.state.messages):
            if getattr(message, "role", None) == "assistant":
                return message
        return None

    async def _emit_extension_event(self, event: AgentEvent) -> None:
        """Emit extension events based on agent events."""
        runner = self._extension_runner
        if event.type == "agent_start":
            self._turn_index = 0
            await runner.emit({"type": "agent_start"})
        elif event.type == "agent_end":
            await runner.emit({"type": "agent_end", "messages": event.messages})
        elif event.type == "turn_start":
            await runner.emit({"type": "turn_start", "turnIndex": self._turn_index, "timestamp": _now_ms()})
        elif event.type == "turn_end":
            await runner.emit(
                {
                    "type": "turn_end",
                    "turnIndex": self._turn_index,
                    "message": event.message,
                    "toolResults": event.tool_results,
                }
            )
            self._turn_index += 1
        elif event.type == "message_start":
            await runner.emit({"type": "message_start", "message": event.message})
        elif event.type == "message_update":
            await runner.emit(
                {
                    "type": "message_update",
                    "message": event.message,
                    "assistantMessageEvent": event.assistant_message_event,
                }
            )
        elif event.type == "message_end":
            replacement = await runner.emit_message_end({"type": "message_end", "message": event.message})
            if replacement is not None:
                # Untyped extension handlers can return messages with null/missing
                # content; normalize so it never enters agent state or history.
                if (
                    getattr(replacement, "role", None) in ("user", "assistant", "toolResult", "custom")
                    and getattr(replacement, "content", None) is None
                ):
                    replacement = dataclass_replace(replacement, content=[])
                self._replace_message_in_state(event, replacement)
        elif event.type == "tool_execution_start":
            await runner.emit(
                {
                    "type": "tool_execution_start",
                    "toolCallId": event.tool_call_id,
                    "toolName": event.tool_name,
                    "args": event.args,
                }
            )
        elif event.type == "tool_execution_update":
            await runner.emit(
                {
                    "type": "tool_execution_update",
                    "toolCallId": event.tool_call_id,
                    "toolName": event.tool_name,
                    "args": event.args,
                    "partialResult": event.partial_result,
                }
            )
        elif event.type == "tool_execution_end":
            await runner.emit(
                {
                    "type": "tool_execution_end",
                    "toolCallId": event.tool_call_id,
                    "toolName": event.tool_name,
                    "result": event.result,
                    "isError": event.is_error,
                }
            )

    def _replace_message_in_state(self, event: AgentMessageEndEvent, replacement: Any) -> None:
        """pi mutates the finalized message object in place so agent state, later
        events, and persistence stay in sync. Frozen-ish dataclasses cannot be
        gutted the same way; instead swap the reference in agent state and on the
        event (persistence in _handle_agent_event reads event.message)."""
        target = event.message
        if target is replacement:
            return
        messages = self.agent.state.messages
        for index in range(len(messages) - 1, -1, -1):
            if messages[index] is target:
                messages[index] = replacement
                break
        event.message = replacement

    def subscribe(self, listener: AgentSessionEventListener) -> Callable[[], None]:
        """Subscribe to agent session events. Session persistence is handled
        internally. Returns an unsubscribe function for this listener."""
        self._event_listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._event_listeners:
                self._event_listeners.remove(listener)

        return unsubscribe

    def _disconnect_from_agent(self) -> None:
        if self._unsubscribe_agent is not None:
            self._unsubscribe_agent()
            self._unsubscribe_agent = None

    def _reconnect_to_agent(self) -> None:
        if self._unsubscribe_agent is not None:
            return  # Already connected
        self._unsubscribe_agent = self.agent.subscribe(self._handle_agent_event)

    def dispose(self) -> None:
        """Remove all listeners and disconnect from agent."""
        try:
            self.abort_retry()
            self.abort_compaction()
            self.abort_branch_summary()
            self.abort_bash()
            self.agent.abort()
        except Exception:
            # Dispose must succeed even if an abort hook throws.
            pass

        self._extension_runner.invalidate()
        self._disconnect_from_agent()
        self._event_listeners = []
        cleanup_session_resources(self.session_id)

    # =========================================================================
    # Read-only State Access
    # =========================================================================

    @property
    def state(self) -> Any:
        return self.agent.state

    @property
    def model(self) -> Model | None:
        return self.agent.state.model

    @property
    def thinking_level(self) -> str:
        return self.agent.state.thinking_level

    @property
    def is_streaming(self) -> bool:
        """Whether the session is currently processing an agent run or post-run continuation."""
        return self._is_agent_run_active

    @property
    def is_idle(self) -> bool:
        return not self._is_agent_run_active

    @property
    def system_prompt(self) -> str:
        """Current effective system prompt (includes per-turn extension modifications)."""
        return self.agent.state.system_prompt

    @property
    def retry_attempt(self) -> int:
        return self._retry_attempt

    def get_active_tool_names(self) -> list[str]:
        return [tool.name for tool in self.agent.state.tools]

    def get_all_tools(self) -> list[ToolInfo]:
        return [
            ToolInfo(
                name=entry.definition.name,
                description=entry.definition.description,
                parameters=entry.definition.parameters,
                prompt_guidelines=entry.definition.prompt_guidelines,
                source_info=entry.source_info,
            )
            for entry in self._tool_definitions.values()
        ]

    def get_tool_definition(self, name: str) -> Any:
        entry = self._tool_definitions.get(name)
        return entry.definition if entry is not None else None

    def set_active_tools_by_name(self, tool_names: list[str]) -> None:
        """Set active tools by name. Unknown tool names are ignored. Also rebuilds
        the system prompt. Changes take effect on the next agent turn."""
        tools: list[AgentTool] = []
        valid_tool_names: list[str] = []
        for name in tool_names:
            tool = self._tool_registry.get(name)
            if tool is not None:
                tools.append(tool)
                valid_tool_names.append(name)
        self.agent.state.tools = tools

        # Rebuild base system prompt with new tool set
        self._base_system_prompt = self._rebuild_system_prompt(valid_tool_names)
        self.agent.state.system_prompt = (
            self._system_prompt_override if self._system_prompt_override is not None else self._base_system_prompt
        )

    @property
    def is_compacting(self) -> bool:
        return (
            self._auto_compaction_cancel is not None
            or self._compaction_cancel is not None
            or self._branch_summary_cancel is not None
        )

    @property
    def messages(self) -> list[Any]:
        """All messages including custom types like BashExecutionMessage."""
        return self.agent.state.messages

    @property
    def steering_mode(self) -> str:
        return self.agent.steering_mode

    @property
    def follow_up_mode(self) -> str:
        return self.agent.follow_up_mode

    @property
    def session_file(self) -> str | None:
        return self.session_manager.get_session_file()

    @property
    def session_id(self) -> str:
        return self.session_manager.get_session_id()

    @property
    def session_name(self) -> str | None:
        return self.session_manager.get_session_name()

    @property
    def scoped_models(self) -> list[ScopedModel]:
        return self._scoped_models

    def set_scoped_models(self, scoped_models: list[ScopedModel]) -> None:
        self._scoped_models = scoped_models

    @property
    def prompt_templates(self) -> list[Any]:
        return self._resource_loader.get_prompts().prompts

    def _normalize_prompt_snippet(self, text: str | None) -> str | None:
        if not text:
            return None
        one_line = re.sub(r"\s+", " ", re.sub(r"[\r\n]+", " ", text)).strip()
        return one_line if one_line else None

    def _normalize_prompt_guidelines(self, guidelines: list[str] | None) -> list[str]:
        if not guidelines:
            return []
        unique: dict[str, None] = {}
        for guideline in guidelines:
            normalized = guideline.strip()
            if normalized:
                unique[normalized] = None
        return list(unique)

    def _rebuild_system_prompt(self, tool_names: list[str]) -> str:
        valid_tool_names = [name for name in tool_names if name in self._tool_registry]
        tool_snippets: dict[str, str] = {}
        prompt_guidelines: list[str] = []
        for name in valid_tool_names:
            snippet = self._tool_prompt_snippets.get(name)
            if snippet:
                tool_snippets[name] = snippet

            tool_guidelines = self._tool_prompt_guidelines.get(name)
            if tool_guidelines:
                prompt_guidelines.extend(tool_guidelines)

        loader_system_prompt = self._resource_loader.get_system_prompt()
        loader_append_system_prompt = self._resource_loader.get_append_system_prompt()
        append_system_prompt = "\n\n".join(loader_append_system_prompt) if loader_append_system_prompt else None
        loaded_skills = self._resource_loader.get_skills().skills
        loaded_context_files = self._resource_loader.get_agents_files()

        self._base_system_prompt_options = BuildSystemPromptOptions(
            cwd=self._cwd,
            skills=loaded_skills,
            context_files=loaded_context_files,
            custom_prompt=loader_system_prompt,
            append_system_prompt=append_system_prompt,
            selected_tools=valid_tool_names,
            tool_snippets=tool_snippets,
            prompt_guidelines=prompt_guidelines,
        )
        return build_system_prompt(self._base_system_prompt_options)

    # =========================================================================
    # Prompting
    # =========================================================================

    async def _run_agent_prompt(self, messages: Any) -> None:
        self._is_agent_run_active = True
        try:
            await self.agent.prompt(messages)
            while await self._handle_post_agent_run():
                await self.agent.continue_()
        finally:
            self._system_prompt_override = None
            # The flag flips before the flush, under the guard that
            # `record_bash_result` takes for its pending-vs-direct decision:
            # a bash result recorded in the settle window then persists
            # directly instead of landing in `_pending_bash_messages` after
            # this flush already ran — such a message was stranded until a
            # future run settled (surfaced on CI as a bashExecution entry
            # missing from the session file).
            with self._state_guard:
                self._is_agent_run_active = False
            await self._flush_pending_bash_messages()
            await self._emit_agent_settled()

    async def _handle_post_agent_run(self) -> bool:
        msg = self._last_assistant_message
        self._last_assistant_message = None
        if msg is None:
            return False

        if self._is_retryable_error(msg) and await self._prepare_retry(msg):
            return True

        if msg.stop_reason == "error" and self._retry_attempt > 0:
            self._emit(AutoRetryEndEvent(success=False, attempt=self._retry_attempt, final_error=msg.error_message))
            self._retry_attempt = 0

        if await self._check_compaction(msg):
            return True

        # The agent loop drains both queues before emitting agent_end. Any messages
        # here were queued by agent_end extension handlers and need a continuation.
        return self.agent.has_queued_messages()

    async def prompt(self, text: str, options: PromptOptions | None = None) -> None:
        """Send a prompt to the agent.

        - Handles extension commands immediately, even during streaming
        - Expands file-based prompt templates by default
        - During streaming, queues via steer()/follow_up() based on streaming_behavior
        - Validates model and API key before sending (when not streaming)
        """
        options = options if options is not None else PromptOptions()
        expand_prompt_templates = options.expand_prompt_templates
        preflight_result = options.preflight_result
        messages: list[Any] | None = None

        try:
            # Handle extension commands first (execute immediately, even during
            # streaming). Extension commands manage their own LLM interaction.
            if expand_prompt_templates and text.startswith("/"):
                handled = await self._try_execute_extension_command(text)
                if handled:
                    if preflight_result:
                        preflight_result(True)
                    return

            # Emit input event for extension interception (before skill/template expansion)
            current_text = text
            current_images = options.images
            if self._extension_runner.has_handlers("input"):
                input_result = await self._extension_runner.emit_input(
                    current_text,
                    current_images,
                    options.source,
                    options.streaming_behavior if self.is_streaming else None,
                )
                if input_result.action == "handled":
                    if preflight_result:
                        preflight_result(True)
                    return
                if input_result.action == "transform":
                    current_text = input_result.text
                    current_images = input_result.images if input_result.images is not None else current_images

            # Expand skill commands (/skill:name args) and prompt templates (/template args)
            expanded_text = current_text
            if expand_prompt_templates:
                expanded_text = await self._expand_skill_command(expanded_text)
                expanded_text = expand_prompt_template(expanded_text, list(self.prompt_templates))

            # If streaming, queue via steer() or follow_up() based on option
            if self.is_streaming:
                if not options.streaming_behavior:
                    raise Exception(
                        "Agent is already processing. Specify streamingBehavior ('steer' or 'followUp') "
                        "to queue the message."
                    )
                if options.streaming_behavior == "followUp":
                    await self._queue_follow_up(expanded_text, current_images)
                else:
                    await self._queue_steer(expanded_text, current_images)
                if preflight_result:
                    preflight_result(True)
                return

            # Flush any pending bash messages before the new prompt
            await self._flush_pending_bash_messages()

            # Validate model
            if self.model is None:
                raise Exception(format_no_model_selected_message())

            has_configured_auth = (
                self._model_runtime.has_configured_auth(self.model.provider)
                or (await self._model_runtime.check_auth(self.model.provider)) is not None
            )
            if not has_configured_auth:
                if self._model_runtime.is_using_oauth(self.model.provider):
                    raise Exception(
                        f'Authentication failed for "{self.model.provider}". '
                        "Credentials may have expired or network is unavailable. "
                        f"Run '/login {self.model.provider}' to re-authenticate."
                    )
                raise Exception(format_no_api_key_found_message(self.model.provider))

            # Check if we need to compact before sending (catches aborted responses).
            # The user's new prompt is sent below, so do not call agent.continue_() here.
            last_assistant = self._find_last_assistant_message()
            if last_assistant is not None:
                await self._check_compaction(last_assistant, False)

            # Build messages array (user message, then pending nextTurn messages)
            messages = []

            user_content: list[TextContent | ImageContent] = [TextContent(text=expanded_text)]
            if current_images:
                user_content.extend(current_images)
            messages.append(UserMessage(content=user_content, timestamp=_now_ms()))

            # Inject any pending "nextTurn" messages as context alongside the user message
            messages.extend(self._pending_next_turn_messages)
            self._pending_next_turn_messages = []

            # Emit before_agent_start extension event
            result = await self._extension_runner.emit_before_agent_start(
                expanded_text,
                current_images,
                self._base_system_prompt,
                self._base_system_prompt_options,
            )
            # Add all custom messages from extensions
            if result and result.get("messages"):
                for msg in result["messages"]:
                    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
                    messages.append(
                        CustomMessage(
                            custom_type=msg.get("customType") if isinstance(msg, dict) else msg.custom_type,
                            # Untyped extensions can pass null content; normalize at ingestion.
                            content=content if content is not None else [],
                            display=msg.get("display") if isinstance(msg, dict) else msg.display,
                            details=msg.get("details") if isinstance(msg, dict) else getattr(msg, "details", None),
                            timestamp=_now_ms(),
                        )
                    )
            # Apply extension-modified system prompt, or reset to base
            if result and result.get("systemPrompt") is not None:
                self._system_prompt_override = result["systemPrompt"]
                self.agent.state.system_prompt = result["systemPrompt"]
            else:
                # Ensure we're using the base prompt (in case previous turn had modifications)
                self._system_prompt_override = None
                self.agent.state.system_prompt = self._base_system_prompt
        except Exception:
            if preflight_result:
                preflight_result(False)
            raise

        if messages is None:
            return

        if preflight_result:
            preflight_result(True)
        await self._run_agent_prompt(messages)

    async def _try_execute_extension_command(self, text: str) -> bool:
        """Try to execute an extension command. Returns True if found and executed."""
        space_index = text.find(" ")
        command_name = text[1:] if space_index == -1 else text[1:space_index]
        args = "" if space_index == -1 else text[space_index + 1 :]

        command = self._extension_runner.get_command(command_name)
        if command is None:
            return False

        # Get command context from extension runner (includes session control methods)
        ctx = self._extension_runner.create_command_context()

        try:
            await command.handler(args, ctx)
            return True
        except Exception as err:
            # lazy: import cycle within core
            from .extensions.types import ExtensionError

            self._extension_runner.emit_error(
                ExtensionError(extension_path=f"command:{command_name}", event="command", error=str(err))
            )
            return True

    async def _expand_skill_command(self, text: str) -> str:
        """Expand skill commands (/skill:name args) to their full content."""
        if not text.startswith("/skill:"):
            return text

        space_index = text.find(" ")
        skill_name = text[7:] if space_index == -1 else text[7:space_index]
        args = "" if space_index == -1 else text[space_index + 1 :].strip()

        skill = next((s for s in self.resource_loader.get_skills().skills if s.name == skill_name), None)
        if skill is None:
            return text  # Unknown skill, pass through

        try:
            content = await fs.Path(skill.file_path).read_text(encoding="utf-8")
            body = strip_frontmatter(content).strip()
            skill_block = (
                f'<skill name="{skill.name}" location="{skill.file_path}">\n'
                f"References are relative to {skill.base_dir}.\n\n{body}\n</skill>"
            )
            return f"{skill_block}\n\n{args}" if args else skill_block
        except Exception as err:
            # lazy: import cycle within core
            from .extensions.types import ExtensionError

            self._extension_runner.emit_error(
                ExtensionError(extension_path=skill.file_path, event="skill_expansion", error=str(err))
            )
            return text  # Return original on error

    async def steer(self, text: str, images: list[ImageContent] | None = None) -> None:
        """Queue a steering message while the agent is running. Delivered after the
        current assistant turn finishes executing its tool calls, before the next
        LLM call. Errors on extension commands."""
        if text.startswith("/"):
            self._throw_if_extension_command(text)

        expanded_text = await self._expand_skill_command(text)
        expanded_text = expand_prompt_template(expanded_text, list(self.prompt_templates))

        await self._queue_steer(expanded_text, images)

    async def follow_up(self, text: str, images: list[ImageContent] | None = None) -> None:
        """Queue a follow-up message to be processed after the agent finishes.
        Delivered only when agent has no more tool calls or steering messages.
        Errors on extension commands."""
        if text.startswith("/"):
            self._throw_if_extension_command(text)

        expanded_text = await self._expand_skill_command(text)
        expanded_text = expand_prompt_template(expanded_text, list(self.prompt_templates))

        await self._queue_follow_up(expanded_text, images)

    async def _queue_steer(self, text: str, images: list[ImageContent] | None = None) -> None:
        self._steering_messages.append(text)
        self._emit_queue_update()
        content: list[TextContent | ImageContent] = [TextContent(text=text)]
        if images:
            content.extend(images)
        self.agent.steer(UserMessage(content=content, timestamp=_now_ms()))

    async def _queue_follow_up(self, text: str, images: list[ImageContent] | None = None) -> None:
        self._follow_up_messages.append(text)
        self._emit_queue_update()
        content: list[TextContent | ImageContent] = [TextContent(text=text)]
        if images:
            content.extend(images)
        self.agent.follow_up(UserMessage(content=content, timestamp=_now_ms()))

    def _throw_if_extension_command(self, text: str) -> None:
        space_index = text.find(" ")
        command_name = text[1:] if space_index == -1 else text[1:space_index]
        command = self._extension_runner.get_command(command_name)

        if command is not None:
            raise Exception(
                f'Extension command "/{command_name}" cannot be queued. '
                "Use prompt() or execute the command when not streaming."
            )

    async def send_custom_message(
        self,
        message: CustomMessage | dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> None:
        """Send a custom message to the session. Creates a CustomMessageEntry.

        - Streaming: queues message, processed when loop pulls from queue
        - Not streaming + trigger_turn: appends to state/session, starts new turn
        - Not streaming + no trigger: appends to state/session, no turn
        """
        options = options or {}
        if isinstance(message, dict):
            custom_type = message.get("customType", message.get("custom_type"))
            content = message.get("content")
            display = message.get("display")
            details = message.get("details")
        else:
            custom_type = message.custom_type
            content = message.content
            display = message.display
            details = message.details

        app_message = CustomMessage(
            custom_type=custom_type,
            # Untyped extensions can pass null/missing content; normalize at ingestion.
            content=content if content is not None else [],
            display=display,
            details=details,
            timestamp=_now_ms(),
        )
        deliver_as = options.get("deliver_as", options.get("deliverAs"))
        if deliver_as == "nextTurn":
            self._pending_next_turn_messages.append(app_message)
        elif self.is_streaming:
            if deliver_as == "followUp":
                self.agent.follow_up(app_message)
            else:
                self.agent.steer(app_message)
        elif options.get("trigger_turn", options.get("triggerTurn")):
            await self._run_agent_prompt(app_message)
        else:
            self.agent.state.messages.append(app_message)
            await self.session_manager.append_custom_message_entry(custom_type, content, display, details)
            self._emit(AgentMessageStartEvent(message=app_message))
            self._emit(AgentMessageEndEvent(message=app_message))

    async def send_user_message(
        self,
        content: str | list[TextContent | ImageContent],
        options: dict[str, Any] | None = None,
    ) -> None:
        """Send a user message to the agent. Always triggers a turn.
        When streaming, options["deliver_as"] specifies how to queue."""
        options = options or {}
        if isinstance(content, str):
            text = content
            images: list[ImageContent] | None = None
        else:
            text_parts: list[str] = []
            images = []
            for part in content:
                if getattr(part, "type", None) == "text":
                    text_parts.append(part.text)
                else:
                    images.append(part)
            text = "\n".join(text_parts)
            if not images:
                images = None

        # Use prompt() with expand_prompt_templates=False to skip command handling
        # and template expansion
        await self.prompt(
            text,
            PromptOptions(
                expand_prompt_templates=False,
                streaming_behavior=options.get("deliver_as", options.get("deliverAs")),
                images=images,
                source="extension",
            ),
        )

    def clear_queue(self) -> dict[str, list[str]]:
        """Clear all queued messages and return them."""
        steering = list(self._steering_messages)
        follow_up = list(self._follow_up_messages)
        self._steering_messages = []
        self._follow_up_messages = []
        self.agent.clear_all_queues()
        self._emit_queue_update()
        return {"steering": steering, "followUp": follow_up}

    @property
    def pending_message_count(self) -> int:
        """Number of pending messages (includes both steering and follow-up)."""
        return len(self._steering_messages) + len(self._follow_up_messages)

    def get_steering_messages(self) -> list[str]:
        return list(self._steering_messages)

    def get_follow_up_messages(self) -> list[str]:
        return list(self._follow_up_messages)

    @property
    def resource_loader(self) -> Any:
        return self._resource_loader

    async def abort(self) -> None:
        """Abort current operation and wait for agent to become idle."""
        self.abort_retry()
        self.agent.abort()
        await self.wait_for_idle()

    async def wait_for_idle(self) -> None:
        if self.is_idle:
            return
        await self._get_idle_wait_event().wait()

    # =========================================================================
    # Model Management
    # =========================================================================

    async def _emit_model_select(self, next_model: Model, previous_model: Model | None, source: str) -> None:
        if models_are_equal(previous_model, next_model):
            return
        await self._extension_runner.emit(
            {"type": "model_select", "model": next_model, "previousModel": previous_model, "source": source}
        )

    async def set_model(self, model: Model) -> None:
        """Set model directly. Validates auth, saves to session and settings."""
        if await self._model_runtime.check_auth(model.provider) is None:
            raise Exception(f"No API key for {model.provider}/{model.id}")

        previous_model = self.model
        thinking_level = self._get_thinking_level_for_model_switch()
        self.agent.state.model = model
        await self.session_manager.append_model_change(model.provider, model.id)
        self.settings_manager.set_default_model_and_provider(model.provider, model.id)

        # Re-clamp thinking level for new model's capabilities
        await self.set_thinking_level(thinking_level)

        await self._emit_model_select(model, previous_model, "set")

    async def cycle_model(self, direction: str = "forward") -> ModelCycleResult | None:
        """Cycle to next/previous model. Uses scoped models (--models flag) if
        available, otherwise all available models."""
        if self._scoped_models:
            return await self._cycle_scoped_model(direction)
        return await self._cycle_available_model(direction)

    async def _cycle_scoped_model(self, direction: str) -> ModelCycleResult | None:
        checks = []
        for scoped in self._scoped_models:
            auth = await self._model_runtime.check_auth(scoped.model.provider)
            checks.append((scoped, auth))
        scoped_models = [scoped for scoped, auth in checks if auth is not None]
        if len(scoped_models) <= 1:
            return None

        current_model = self.model
        current_index = next((i for i, sm in enumerate(scoped_models) if models_are_equal(sm.model, current_model)), -1)

        if current_index == -1:
            current_index = 0
        length = len(scoped_models)
        next_index = (current_index + 1) % length if direction == "forward" else (current_index - 1 + length) % length
        next_scoped = scoped_models[next_index]
        thinking_level = self._get_thinking_level_for_model_switch(next_scoped.thinking_level)

        # Apply model
        self.agent.state.model = next_scoped.model
        await self.session_manager.append_model_change(next_scoped.model.provider, next_scoped.model.id)
        self.settings_manager.set_default_model_and_provider(next_scoped.model.provider, next_scoped.model.id)

        # Apply thinking level: explicit scoped level overrides the session level,
        # None inherits the current session preference. set_thinking_level clamps.
        await self.set_thinking_level(thinking_level)

        await self._emit_model_select(next_scoped.model, current_model, "cycle")

        return ModelCycleResult(model=next_scoped.model, thinking_level=self.thinking_level, is_scoped=True)

    async def _cycle_available_model(self, direction: str) -> ModelCycleResult | None:
        available_models = await self._model_runtime.get_available()
        if len(available_models) <= 1:
            return None

        current_model = self.model
        current_index = next((i for i, m in enumerate(available_models) if models_are_equal(m, current_model)), -1)

        if current_index == -1:
            current_index = 0
        length = len(available_models)
        next_index = (current_index + 1) % length if direction == "forward" else (current_index - 1 + length) % length
        next_model = available_models[next_index]

        thinking_level = self._get_thinking_level_for_model_switch()
        self.agent.state.model = next_model
        await self.session_manager.append_model_change(next_model.provider, next_model.id)
        self.settings_manager.set_default_model_and_provider(next_model.provider, next_model.id)

        # Re-clamp thinking level for new model's capabilities
        await self.set_thinking_level(thinking_level)

        await self._emit_model_select(next_model, current_model, "cycle")

        return ModelCycleResult(model=next_model, thinking_level=self.thinking_level, is_scoped=False)

    # =========================================================================
    # Thinking Level Management
    # =========================================================================

    async def set_thinking_level(self, level: str) -> None:
        """Set thinking level, clamped to model capabilities. Saves to session
        and settings only if the level actually changes."""
        available_levels = self.get_available_thinking_levels()
        effective_level = level if level in available_levels else self._clamp_thinking_level(level, available_levels)

        # Only persist if actually changing
        previous_level = self.agent.state.thinking_level
        is_changing = effective_level != previous_level

        self.agent.state.thinking_level = effective_level

        if is_changing:
            await self.session_manager.append_thinking_level_change(effective_level)
            if self.supports_thinking() or effective_level != "off":
                self.settings_manager.set_default_thinking_level(effective_level)
            self._emit(ThinkingLevelChangedEvent(level=effective_level))
            tonio.spawn.without_tracking(
                self._extension_runner.emit(
                    {"type": "thinking_level_select", "level": effective_level, "previousLevel": previous_level}
                )
            )

    async def cycle_thinking_level(self) -> str | None:
        """Cycle to next thinking level. None if model doesn't support thinking."""
        if not self.supports_thinking():
            return None

        levels = self.get_available_thinking_levels()
        current_index = levels.index(self.thinking_level) if self.thinking_level in levels else -1
        next_level = levels[(current_index + 1) % len(levels)]

        await self.set_thinking_level(next_level)
        return next_level

    def get_available_thinking_levels(self) -> list[str]:
        """Available thinking levels for the current model."""
        if self.model is None:
            return list(_THINKING_LEVELS)
        return list(get_supported_thinking_levels(self.model))

    def supports_thinking(self) -> bool:
        return bool(self.model is not None and self.model.reasoning)

    def _get_thinking_level_for_model_switch(self, explicit_level: str | None = None) -> str:
        if explicit_level is not None:
            return explicit_level
        if not self.supports_thinking():
            default = self.settings_manager.get_default_thinking_level()
            return default if default is not None else DEFAULT_THINKING_LEVEL
        return self.thinking_level

    def _clamp_thinking_level(self, level: str, _available_levels: list[str]) -> str:
        return clamp_thinking_level(self.model, level) if self.model is not None else "off"

    # =========================================================================
    # Queue Mode Management
    # =========================================================================

    def sync_queue_modes_from_settings(self) -> None:
        self.agent.steering_mode = self.settings_manager.get_steering_mode()
        self.agent.follow_up_mode = self.settings_manager.get_follow_up_mode()

    def set_steering_mode(self, mode: str) -> None:
        self.agent.steering_mode = mode
        self.settings_manager.set_steering_mode(mode)

    def set_follow_up_mode(self, mode: str) -> None:
        self.agent.follow_up_mode = mode
        self.settings_manager.set_follow_up_mode(mode)

    # =========================================================================
    # Compaction
    # =========================================================================

    async def compact(self, custom_instructions: str | None = None) -> CompactionResult:
        """Manually compact the session context. Aborts current agent operation first."""
        self._disconnect_from_agent()
        await self.abort()
        self._compaction_cancel = CancelToken()
        self._emit(CompactionStartEvent(reason="manual"))

        try:
            if self.model is None:
                raise Exception(format_no_model_selected_message())

            auth = await self._get_summarization_request_auth(self.model)

            path_entries = self.session_manager.get_branch()
            settings = _compaction_settings_from(self.settings_manager.get_compaction_settings())

            preparation = prepare_compaction(path_entries, settings)
            if preparation is None:
                # Check why we can't compact
                last_entry = path_entries[-1] if path_entries else None
                if last_entry is not None and last_entry.get("type") == "compaction":
                    raise Exception("Already compacted")
                raise Exception("Nothing to compact (session too small)")

            extension_compaction: CompactionResult | None = None
            from_extension = False

            if self._extension_runner.has_handlers("session_before_compact"):
                result = await self._extension_runner.emit(
                    {
                        "type": "session_before_compact",
                        "preparation": preparation,
                        "branchEntries": path_entries,
                        "customInstructions": custom_instructions,
                        "reason": "manual",
                        "willRetry": False,
                        "signal": self._compaction_cancel,
                    }
                )

                if isinstance(result, dict) and result.get("cancel"):
                    raise Exception("Compaction cancelled")

                if isinstance(result, dict) and result.get("compaction") is not None:
                    extension_compaction = result["compaction"]
                    from_extension = True

            if extension_compaction is not None:
                # Extension provided compaction content
                summary = extension_compaction.summary
                first_kept_entry_id = extension_compaction.first_kept_entry_id
                tokens_before = extension_compaction.tokens_before
                usage = extension_compaction.usage
                details = extension_compaction.details
            else:
                # Generate compaction result
                result = await run_compact(
                    preparation,
                    self.model,
                    auth["api_key"],
                    auth["headers"],
                    custom_instructions,
                    self._compaction_cancel,
                    self.thinking_level,
                    self.agent.stream_function,
                    auth["env"],
                    _retry_policy_from(self.settings_manager.get_retry_settings()),
                    self._summarization_retry_callbacks({"source": "compaction", "reason": "manual"}),
                )
                summary = result.summary
                first_kept_entry_id = result.first_kept_entry_id
                tokens_before = result.tokens_before
                usage = result.usage
                details = result.details

            if self._compaction_cancel.cancelled:
                raise Exception("Compaction cancelled")

            await self.session_manager.append_compaction(
                summary, first_kept_entry_id, tokens_before, details, from_extension, usage
            )
            new_entries = self.session_manager.get_entries()
            session_context = self.session_manager.build_session_context()
            self.agent.state.messages = session_context.messages
            estimated_tokens_after = _estimate_messages_tokens(session_context.messages)

            # Get the saved compaction entry for the extension event
            saved_compaction_entry = next(
                (e for e in new_entries if e.get("type") == "compaction" and e.get("summary") == summary), None
            )

            if self._extension_runner is not None and saved_compaction_entry is not None:
                await self._extension_runner.emit(
                    {
                        "type": "session_compact",
                        "compactionEntry": saved_compaction_entry,
                        "fromExtension": from_extension,
                        "reason": "manual",
                        "willRetry": False,
                    }
                )

            compaction_result = CompactionResult(
                summary=summary,
                first_kept_entry_id=first_kept_entry_id,
                tokens_before=tokens_before,
                estimated_tokens_after=estimated_tokens_after,
                usage=usage,
                details=details,
            )
            self._emit(CompactionEndEvent(reason="manual", result=compaction_result, aborted=False, will_retry=False))
            return compaction_result
        except Exception as error:
            message = str(error)
            aborted = message == "Compaction cancelled" or type(error).__name__ == "AbortError"
            self._emit(
                CompactionEndEvent(
                    reason="manual",
                    result=None,
                    aborted=aborted,
                    will_retry=False,
                    error_message=None if aborted else f"Compaction failed: {message}",
                )
            )
            raise
        finally:
            self._compaction_cancel = None
            self._reconnect_to_agent()

    def abort_compaction(self) -> None:
        """Cancel in-progress compaction (manual or auto)."""
        if self._compaction_cancel is not None:
            self._compaction_cancel.cancel()
        if self._auto_compaction_cancel is not None:
            self._auto_compaction_cancel.cancel()

    def abort_branch_summary(self) -> None:
        """Cancel in-progress branch summarization."""
        if self._branch_summary_cancel is not None:
            self._branch_summary_cancel.cancel()

    async def _check_compaction(self, assistant_message: AssistantMessage, skip_aborted_check: bool = True) -> bool:
        """Check if compaction is needed and run it. Called after agent_end and
        before prompt submission.

        1. Overflow: LLM returned context overflow error - remove error message
           from agent state, compact, auto-retry.
        2. Threshold: context over threshold - compact, NO auto-retry."""
        settings = _compaction_settings_from(self.settings_manager.get_compaction_settings())
        if not settings.enabled:
            return False

        # Skip if message was aborted (user cancelled) - unless skip_aborted_check is False
        if skip_aborted_check and assistant_message.stop_reason == "aborted":
            return False

        context_window = self.model.context_window if self.model is not None else 0
        context_window = context_window or 0

        # Skip overflow check if the message came from a different model (e.g. the
        # user switched from a smaller-context to a larger-context model).
        same_model = (
            self.model is not None
            and assistant_message.provider == self.model.provider
            and assistant_message.model == self.model.id
        )

        # Skip compaction checks if this assistant message is older than the latest
        # compaction boundary. This prevents a stale pre-compaction usage/error
        # from retriggering compaction on the first prompt after compaction.
        compaction_entry = get_latest_compaction_entry(self.session_manager.get_branch())
        assistant_is_from_before_compaction = (
            compaction_entry is not None
            and assistant_message.timestamp <= _iso_to_epoch_ms(compaction_entry.get("timestamp"))
        )
        if assistant_is_from_before_compaction:
            return False

        # Case 1: Overflow. A successful response over the configured window should
        # compact but must not retry: the assistant answer already completed and
        # agent.continue_() cannot continue from an assistant message.
        if same_model and is_context_overflow(assistant_message, context_window):
            will_retry = assistant_message.stop_reason != "stop"

            if not will_retry:
                return await self._run_auto_compaction("overflow", False)

            if self._overflow_recovery_attempted:
                self._emit(
                    CompactionEndEvent(
                        reason="overflow",
                        result=None,
                        aborted=False,
                        will_retry=False,
                        error_message=(
                            "Context overflow recovery failed after one compact-and-retry attempt. "
                            "Try reducing context or switching to a larger-context model."
                        ),
                    )
                )
                return False

            self._overflow_recovery_attempted = True
            # Remove the error message from agent state (it IS saved to session for
            # history, but we don't want it in context for the retry)
            messages = self.agent.state.messages
            if messages and getattr(messages[-1], "role", None) == "assistant":
                self.agent.state.messages = messages[:-1]
            return await self._run_auto_compaction("overflow", will_retry)

        # Case 2: Threshold. For error messages or all-zero usage messages, estimate
        # from the last valid response so sessions hitting persistent API errors can
        # still compact and do not reset context accounting.
        direct_context_tokens = (
            calculate_context_tokens(assistant_message.usage) if assistant_message.usage is not None else 0
        )
        if assistant_message.stop_reason == "error" or direct_context_tokens == 0:
            messages = self.agent.state.messages
            estimate = estimate_context_tokens(messages)
            if estimate.last_usage_index is None:
                return False  # No usage data at all
            # Verify the usage source is post-compaction. Kept pre-compaction messages
            # have stale usage reflecting the old (larger) context and would falsely
            # trigger compaction right after one just finished.
            usage_msg = messages[estimate.last_usage_index]
            if (
                compaction_entry is not None
                and getattr(usage_msg, "role", None) == "assistant"
                and usage_msg.timestamp <= _iso_to_epoch_ms(compaction_entry.get("timestamp"))
            ):
                return False
            context_tokens = estimate.tokens
        else:
            context_tokens = direct_context_tokens
        if should_compact(context_tokens, context_window, settings):
            return await self._run_auto_compaction("threshold", False)
        return False

    async def _run_auto_compaction(self, reason: str, will_retry: bool) -> bool:
        """Internal: run auto-compaction with events."""
        settings = _compaction_settings_from(self.settings_manager.get_compaction_settings())
        started = False

        try:
            if self.model is None:
                return False

            if self._uses_default_stream_simple():
                auth = await self._get_required_request_auth(self.model)
            else:
                auth = await self._get_summarization_request_auth(self.model)

            path_entries = self.session_manager.get_branch()

            preparation = prepare_compaction(path_entries, settings)
            if preparation is None:
                return False

            self._emit(CompactionStartEvent(reason=reason))
            self._auto_compaction_cancel = CancelToken()
            started = True

            extension_compaction: CompactionResult | None = None
            from_extension = False

            if self._extension_runner.has_handlers("session_before_compact"):
                extension_result = await self._extension_runner.emit(
                    {
                        "type": "session_before_compact",
                        "preparation": preparation,
                        "branchEntries": path_entries,
                        "customInstructions": None,
                        "reason": reason,
                        "willRetry": will_retry,
                        "signal": self._auto_compaction_cancel,
                    }
                )

                if isinstance(extension_result, dict) and extension_result.get("cancel"):
                    self._emit(CompactionEndEvent(reason=reason, result=None, aborted=True, will_retry=False))
                    return False

                if isinstance(extension_result, dict) and extension_result.get("compaction") is not None:
                    extension_compaction = extension_result["compaction"]
                    from_extension = True

            if extension_compaction is not None:
                summary = extension_compaction.summary
                first_kept_entry_id = extension_compaction.first_kept_entry_id
                tokens_before = extension_compaction.tokens_before
                usage = extension_compaction.usage
                details = extension_compaction.details
            else:
                compact_result = await run_compact(
                    preparation,
                    self.model,
                    auth["api_key"],
                    auth["headers"],
                    None,
                    self._auto_compaction_cancel,
                    self.thinking_level,
                    self.agent.stream_function,
                    auth["env"],
                    _retry_policy_from(self.settings_manager.get_retry_settings()),
                    self._summarization_retry_callbacks({"source": "compaction", "reason": reason}),
                )
                summary = compact_result.summary
                first_kept_entry_id = compact_result.first_kept_entry_id
                tokens_before = compact_result.tokens_before
                usage = compact_result.usage
                details = compact_result.details

            if self._auto_compaction_cancel.cancelled:
                self._emit(CompactionEndEvent(reason=reason, result=None, aborted=True, will_retry=False))
                return False

            await self.session_manager.append_compaction(
                summary, first_kept_entry_id, tokens_before, details, from_extension, usage
            )
            new_entries = self.session_manager.get_entries()
            session_context = self.session_manager.build_session_context()
            self.agent.state.messages = session_context.messages
            estimated_tokens_after = _estimate_messages_tokens(session_context.messages)

            saved_compaction_entry = next(
                (e for e in new_entries if e.get("type") == "compaction" and e.get("summary") == summary), None
            )

            if self._extension_runner is not None and saved_compaction_entry is not None:
                await self._extension_runner.emit(
                    {
                        "type": "session_compact",
                        "compactionEntry": saved_compaction_entry,
                        "fromExtension": from_extension,
                        "reason": reason,
                        "willRetry": will_retry,
                    }
                )

            result = CompactionResult(
                summary=summary,
                first_kept_entry_id=first_kept_entry_id,
                tokens_before=tokens_before,
                estimated_tokens_after=estimated_tokens_after,
                usage=usage,
                details=details,
            )
            self._emit(CompactionEndEvent(reason=reason, result=result, aborted=False, will_retry=will_retry))

            if will_retry:
                messages = self.agent.state.messages
                last_msg = messages[-1] if messages else None
                if (
                    last_msg is not None
                    and getattr(last_msg, "role", None) == "assistant"
                    and last_msg.stop_reason == "error"
                ):
                    self.agent.state.messages = messages[:-1]
                return True

            # Auto-compaction can complete while follow-up/steering/custom messages
            # are waiting. Continue once so queued messages are delivered.
            return self.agent.has_queued_messages()
        except Exception as error:
            error_message = str(error) or "compaction failed"
            if started:
                self._emit(
                    CompactionEndEvent(
                        reason=reason,
                        result=None,
                        aborted=False,
                        will_retry=False,
                        error_message=(
                            f"Context overflow recovery failed: {error_message}"
                            if reason == "overflow"
                            else f"Auto-compaction failed: {error_message}"
                        ),
                    )
                )
            return False
        finally:
            self._auto_compaction_cancel = None

    def set_auto_compaction_enabled(self, enabled: bool) -> None:
        self.settings_manager.set_compaction_enabled(enabled)

    @property
    def auto_compaction_enabled(self) -> bool:
        return self.settings_manager.get_compaction_enabled()

    # =========================================================================
    # Extension bindings
    # =========================================================================

    async def bind_extensions(self, bindings: ExtensionBindings) -> None:
        if bindings.ui_context is not None:
            self._extension_ui_context = bindings.ui_context
        if bindings.mode is not None:
            self._extension_mode = bindings.mode
        if bindings.command_context_actions is not None:
            self._extension_command_context_actions = bindings.command_context_actions
        if bindings.abort_handler is not None:
            self._extension_abort_handler = bindings.abort_handler
        if bindings.shutdown_handler is not None:
            self._extension_shutdown_handler = bindings.shutdown_handler
        if bindings.on_error is not None:
            self._extension_error_listener = bindings.on_error

        self._apply_extension_bindings(self._extension_runner)
        await self._extension_runner.emit(self._session_start_event)
        await self._extend_resources_from_extensions(
            "reload" if self._session_start_event.get("reason") == "reload" else "startup"
        )

    async def _extend_resources_from_extensions(self, reason: str) -> None:
        if not self._extension_runner.has_handlers("resources_discover"):
            return

        discovered = await self._extension_runner.emit_resources_discover(self._cwd, reason)

        if not discovered.skill_paths and not discovered.prompt_paths and not discovered.theme_paths:
            return

        extension_paths = {
            "skill_paths": self._build_extension_resource_paths(discovered.skill_paths),
            "prompt_paths": self._build_extension_resource_paths(discovered.prompt_paths),
            "theme_paths": self._build_extension_resource_paths(discovered.theme_paths),
        }

        await self._resource_loader.extend_resources(extension_paths)
        self._base_system_prompt = self._rebuild_system_prompt(self.get_active_tool_names())
        self.agent.state.system_prompt = self._base_system_prompt

    def _build_extension_resource_paths(self, entries: list[dict[str, str]]) -> list[dict[str, Any]]:
        results = []
        for entry in entries:
            extension_path = entry["extension_path"]
            source = self._get_extension_source_label(extension_path)
            base_dir = None if extension_path.startswith("<") else os.path.dirname(extension_path)
            results.append(
                {
                    "path": entry["path"],
                    "metadata": {
                        "source": source,
                        "scope": "temporary",
                        "origin": "top-level",
                        "base_dir": base_dir,
                    },
                }
            )
        return results

    def _get_extension_source_label(self, extension_path: str) -> str:
        if extension_path.startswith("<"):
            return "extension:" + extension_path.replace("<", "").replace(">", "")
        base = os.path.basename(extension_path)
        name = re.sub(r"\.(py|ts|js)$", "", base)
        return f"extension:{name}"

    def _apply_extension_bindings(self, runner: ExtensionRunner) -> None:
        runner.set_ui_context(self._extension_ui_context, self._extension_mode)
        runner.bind_command_context(self._extension_command_context_actions)

        if self._extension_error_unsubscriber is not None:
            self._extension_error_unsubscriber()
        self._extension_error_unsubscriber = (
            runner.on_error(self._extension_error_listener) if self._extension_error_listener is not None else None
        )

    def _refresh_current_model_from_registry(self) -> None:
        current_model = self.model
        if current_model is None:
            return

        refreshed_model = self._model_runtime.get_model(current_model.provider, current_model.id)
        if refreshed_model is None or refreshed_model is current_model:
            return

        self.agent.state.model = refreshed_model

    def _bind_extension_core(self, runner: ExtensionRunner) -> None:
        def get_commands() -> list[Any]:
            # lazy: import cycle within core
            from .slash_commands import SlashCommandInfo

            extension_commands = [
                SlashCommandInfo(
                    name=command.invocation_name,
                    description=command.description,
                    source="extension",
                    source_info=command.source_info,
                )
                for command in runner.get_registered_commands()
            ]

            templates = [
                SlashCommandInfo(
                    name=template.name,
                    description=template.description,
                    source="prompt",
                    source_info=template.source_info,
                )
                for template in self.prompt_templates
            ]

            skills = [
                SlashCommandInfo(
                    name=f"skill:{skill.name}",
                    description=skill.description,
                    source="skill",
                    source_info=skill.source_info,
                )
                for skill in self._resource_loader.get_skills().skills
            ]

            return [*extension_commands, *templates, *skills]

        def send_message_action(message: Any, options: Any = None) -> None:
            async def run() -> None:
                try:
                    await self.send_custom_message(message, options)
                except Exception as err:
                    # lazy: import cycle within core
                    from .extensions.types import ExtensionError

                    runner.emit_error(ExtensionError(extension_path="<runtime>", event="send_message", error=str(err)))

            tonio.spawn.without_tracking(run())

        def send_user_message_action(content: Any, options: Any = None) -> None:
            async def run() -> None:
                try:
                    await self.send_user_message(content, options)
                except Exception as err:
                    # lazy: import cycle within core
                    from .extensions.types import ExtensionError

                    runner.emit_error(
                        ExtensionError(extension_path="<runtime>", event="send_user_message", error=str(err))
                    )

            tonio.spawn.without_tracking(run())

        async def append_entry_action(custom_type: str, data: Any = None) -> None:
            entry_id = await self.session_manager.append_custom_entry(custom_type, data)
            entry = self.session_manager.get_entry(entry_id)
            if entry is not None:
                self._emit(EntryAppendedEvent(entry=entry))

        async def set_model_action(model: Model) -> bool:
            if not self._model_runtime.has_configured_auth(model.provider):
                return False
            await self.set_model(model)
            return True

        def compact_action(options: Any = None) -> None:
            async def run() -> None:
                options_dict = options or {}
                try:
                    result = await self.compact(options_dict.get("custom_instructions"))
                    on_complete = options_dict.get("on_complete")
                    if on_complete is not None:
                        on_complete(result)
                except Exception as error:
                    on_error = options_dict.get("on_error")
                    if on_error is not None:
                        on_error(error)

            tonio.spawn.without_tracking(run())

        def abort_action() -> None:
            if self._extension_abort_handler is not None:
                self._extension_abort_handler()
                return
            tonio.spawn.without_tracking(self.abort())

        runner.bind_core(
            {
                "send_message": send_message_action,
                "send_user_message": send_user_message_action,
                "append_entry": append_entry_action,
                "set_session_name": lambda name: self.set_session_name(name),
                "get_session_name": lambda: self.session_manager.get_session_name(),
                "set_label": lambda entry_id, label: self.session_manager.append_label_change(entry_id, label),
                # `set_session_name`/`set_label`/`append_entry`/`set_thinking_level`
                # now return coroutines; `loader.py` awaits them.
                "get_active_tools": lambda: self.get_active_tool_names(),
                "get_all_tools": lambda: self.get_all_tools(),
                "set_active_tools": lambda tool_names: self.set_active_tools_by_name(tool_names),
                "refresh_tools": lambda: self._refresh_tool_registry(),
                "get_commands": get_commands,
                "set_model": set_model_action,
                "get_thinking_level": lambda: self.thinking_level,
                "set_thinking_level": lambda level: self.set_thinking_level(level),
            },
            {
                "get_model": lambda: self.model,
                "is_idle": lambda: self.is_idle,
                "is_project_trusted": lambda: self.settings_manager.is_project_trusted(),
                "get_signal": lambda: self.agent.signal,
                "abort": abort_action,
                "has_pending_messages": lambda: self.pending_message_count > 0,
                "shutdown": lambda: (
                    self._extension_shutdown_handler() if self._extension_shutdown_handler is not None else None
                ),
                "get_context_usage": lambda: self.get_context_usage(),
                "compact": compact_action,
                "get_system_prompt": lambda: self.system_prompt,
                "get_system_prompt_options": lambda: self._base_system_prompt_options,
            },
            {
                "register_provider": lambda name, provider_config: (
                    self._model_runtime.register_provider(name, provider_config),
                    self._refresh_current_model_from_registry(),
                )[0],
                "register_native_provider": lambda provider: (
                    self._model_runtime.register_native_provider(provider),
                    self._refresh_current_model_from_registry(),
                )[0],
                "unregister_provider": lambda name: (
                    self._model_runtime.unregister_provider(name),
                    self._refresh_current_model_from_registry(),
                )[0],
            },
        )

    def _refresh_tool_registry(
        self, active_tool_names: list[str] | None = None, include_all_extension_tools: bool | None = None
    ) -> None:
        previous_registry_names = set(self._tool_registry.keys())
        previous_active_tool_names = self.get_active_tool_names()
        allowed_tool_names = self._allowed_tool_names
        excluded_tool_names = self._excluded_tool_names

        def is_allowed_tool(name: str) -> bool:
            return (allowed_tool_names is None or name in allowed_tool_names) and not (
                excluded_tool_names is not None and name in excluded_tool_names
            )

        # lazy: import cycle within core
        from .extensions.types import RegisteredTool

        registered_tools = self._extension_runner.get_all_registered_tools()
        all_custom_tools = [
            *registered_tools,
            *(
                RegisteredTool(
                    definition=definition,
                    source_info=create_synthetic_source_info(f"<sdk:{definition.name}>", source="sdk"),
                )
                for definition in self._custom_tools
            ),
        ]
        all_custom_tools = [tool for tool in all_custom_tools if is_allowed_tool(tool.definition.name)]

        definition_registry: dict[str, _ToolDefinitionEntry] = {
            name: _ToolDefinitionEntry(
                definition=definition,
                source_info=create_synthetic_source_info(f"<builtin:{name}>", source="builtin"),
            )
            for name, definition in self._base_tool_definitions.items()
            if is_allowed_tool(name)
        }
        for tool in all_custom_tools:
            definition_registry[tool.definition.name] = _ToolDefinitionEntry(
                definition=tool.definition, source_info=tool.source_info
            )
        self._tool_definitions = definition_registry
        self._tool_prompt_snippets = {}
        self._tool_prompt_guidelines = {}
        for entry in definition_registry.values():
            snippet = self._normalize_prompt_snippet(entry.definition.prompt_snippet)
            if snippet:
                self._tool_prompt_snippets[entry.definition.name] = snippet
            guidelines = self._normalize_prompt_guidelines(entry.definition.prompt_guidelines)
            if guidelines:
                self._tool_prompt_guidelines[entry.definition.name] = guidelines

        runner = self._extension_runner
        wrapped_extension_tools = wrap_registered_tools(all_custom_tools, runner)
        wrapped_built_in_tools = wrap_registered_tools(
            [
                RegisteredTool(
                    definition=definition,
                    source_info=create_synthetic_source_info(f"<builtin:{definition.name}>", source="builtin"),
                )
                for definition in self._base_tool_definitions.values()
                if is_allowed_tool(definition.name)
            ],
            runner,
        )

        tool_registry: dict[str, AgentTool] = {tool.name: tool for tool in wrapped_built_in_tools}
        for tool in wrapped_extension_tools:
            tool_registry[tool.name] = tool
        self._tool_registry = tool_registry

        next_active_tool_names = [
            name
            for name in (list(active_tool_names) if active_tool_names is not None else list(previous_active_tool_names))
            if is_allowed_tool(name)
        ]

        if allowed_tool_names is not None:
            for tool_name in self._tool_registry:
                if tool_name in allowed_tool_names:
                    next_active_tool_names.append(tool_name)
        elif include_all_extension_tools:
            for tool in wrapped_extension_tools:
                next_active_tool_names.append(tool.name)
        elif active_tool_names is None:
            for tool_name in self._tool_registry:
                if tool_name not in previous_registry_names:
                    next_active_tool_names.append(tool_name)

        self.set_active_tools_by_name(list(dict.fromkeys(next_active_tool_names)))

    def _build_runtime(
        self,
        active_tool_names: list[str] | None = None,
        flag_values: dict[str, Any] | None = None,
        include_all_extension_tools: bool | None = None,
    ) -> None:
        auto_resize_images = self.settings_manager.get_image_auto_resize()
        shell_command_prefix = self.settings_manager.get_shell_command_prefix()
        shell_path = self.settings_manager.get_shell_path()
        if self._base_tools_override is not None:
            base_tool_definitions = {
                name: create_tool_definition_from_agent_tool(tool) for name, tool in self._base_tools_override.items()
            }
        else:
            base_tool_definitions = create_all_tool_definitions(
                self._cwd,
                {
                    "read": {"auto_resize_images": auto_resize_images},
                    "bash": {"command_prefix": shell_command_prefix, "shell_path": shell_path},
                },
            )

        self._base_tool_definitions = dict(base_tool_definitions)

        extensions_result = self._resource_loader.get_extensions()
        if flag_values:
            for name, value in flag_values.items():
                extensions_result.runtime.flag_values[name] = value

        self._extension_runner = ExtensionRunner(
            extensions_result.extensions,
            extensions_result.runtime,
            self._cwd,
            self.session_manager,
            ModelRegistry(self._model_runtime),
        )
        if self._extension_runner_ref is not None:
            self._extension_runner_ref.current = self._extension_runner
        self._bind_extension_core(self._extension_runner)
        self._apply_extension_bindings(self._extension_runner)

        default_active_tool_names = (
            list(self._base_tools_override.keys())
            if self._base_tools_override is not None
            else ["read", "bash", "edit", "write"]
        )
        base_active_tool_names = active_tool_names if active_tool_names is not None else default_active_tool_names
        self._refresh_tool_registry(
            active_tool_names=base_active_tool_names,
            include_all_extension_tools=include_all_extension_tools,
        )

    async def reload(self, before_session_start: Callable[[], Awaitable[None]] | None = None) -> None:

        previous_flag_values = self._extension_runner.get_flag_values()
        await emit_session_shutdown_event(self._extension_runner, {"type": "session_shutdown", "reason": "reload"})
        await self.settings_manager.reload()  # drains queued writes first, like pi
        self.sync_queue_modes_from_settings()
        # pi calls resetApiProviders() (the pi-ai compat registry); pidrei's
        # adapters are stateless modules composed per ModelRuntime, so there is
        # no global provider cache to reset.
        await self._resource_loader.reload()
        self._build_runtime(
            active_tool_names=self.get_active_tool_names(),
            flag_values=previous_flag_values,
            include_all_extension_tools=True,
        )

        has_bindings = (
            self._extension_ui_context
            or self._extension_command_context_actions
            or self._extension_shutdown_handler
            or self._extension_error_listener
        )
        if has_bindings:
            if before_session_start is not None:
                await before_session_start()
            await self._extension_runner.emit({"type": "session_start", "reason": "reload"})
            await self._extend_resources_from_extensions("reload")

    # =========================================================================
    # Auto-Retry
    # =========================================================================

    def _is_retryable_error(self, message: AssistantMessage) -> bool:
        """Retryable = overloaded, rate limit, server errors. Context overflow is
        NOT retryable (handled by compaction instead)."""
        context_window = self.model.context_window if self.model is not None else 0
        if is_context_overflow(message, context_window or 0):
            return False
        return is_retryable_assistant_error(message)

    def _summarization_retry_callbacks(self, source: dict[str, Any]) -> RetryCallbacks:
        """Retry callbacks shared by compaction and branch-summary summarization
        calls. `source` carries the context the TUI needs to render the retry."""

        async def on_retry_scheduled(attempt: int, max_attempts: int, delay_ms: float, error_message: str) -> None:
            self._emit(
                SummarizationRetryScheduledEvent(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    delay_ms=delay_ms,
                    error_message=error_message,
                )
            )

        async def on_retry_attempt_start() -> None:
            self._emit(SummarizationRetryAttemptStartEvent(source=source["source"], reason=source.get("reason")))

        async def on_retry_finished(*_args: Any) -> None:
            self._emit(SummarizationRetryFinishedEvent())

        return RetryCallbacks(
            on_retry_scheduled=on_retry_scheduled,
            on_retry_attempt_start=on_retry_attempt_start,
            on_retry_finished=on_retry_finished,
        )

    async def _prepare_retry(self, message: AssistantMessage) -> bool:
        """Prepare a retryable error for continuation with exponential backoff.
        Returns True if the caller should continue the agent."""
        settings = self.settings_manager.get_retry_settings()
        if not settings["enabled"]:
            return False

        self._retry_attempt += 1

        if self._retry_attempt > settings["max_retries"]:
            # Preserve the completed attempt count so post-run handling can emit
            # the final failure.
            self._retry_attempt -= 1
            return False

        delay_ms = settings["base_delay_ms"] * 2 ** (self._retry_attempt - 1)

        self._emit(
            AutoRetryStartEvent(
                attempt=self._retry_attempt,
                max_attempts=settings["max_retries"],
                delay_ms=delay_ms,
                error_message=message.error_message or "Unknown error",
            )
        )

        # Remove error message from agent state (keep in session for history)
        messages = self.agent.state.messages
        if messages and getattr(messages[-1], "role", None) == "assistant":
            self.agent.state.messages = messages[:-1]

        # Wait with exponential backoff (abortable)
        self._retry_cancel = CancelToken()
        try:
            await sleep(delay_ms, self._retry_cancel)
        except Exception:
            # Aborted during sleep - emit end event so UI can clean up
            attempt = self._retry_attempt
            self._retry_attempt = 0
            self._emit(AutoRetryEndEvent(success=False, attempt=attempt, final_error="Retry cancelled"))
            return False
        finally:
            self._retry_cancel = None

        return True

    def abort_retry(self) -> None:
        """Cancel in-progress retry."""
        if self._retry_cancel is not None:
            self._retry_cancel.cancel()

    @property
    def is_retrying(self) -> bool:
        return self._retry_cancel is not None

    @property
    def auto_retry_enabled(self) -> bool:
        return self.settings_manager.get_retry_enabled()

    def set_auto_retry_enabled(self, enabled: bool) -> None:
        self.settings_manager.set_retry_enabled(enabled)

    # =========================================================================
    # Bash Execution
    # =========================================================================

    async def execute_bash(
        self,
        command: str,
        on_chunk: Callable[[str], None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> BashResult:
        """Execute a bash command. Adds result to agent context and session."""
        options = options or {}
        self._bash_cancel = CancelToken()

        # Apply command prefix if configured (e.g. "shopt -s expand_aliases")
        prefix = self.settings_manager.get_shell_command_prefix()
        shell_path = self.settings_manager.get_shell_path()
        resolved_command = f"{prefix}\n{command}" if prefix else command

        def chunk_callback(delta: str) -> None:
            if on_chunk is not None:
                on_chunk(delta)
            self._emit(BashExecutionUpdateEvent(id=options.get("id"), delta=delta))

        try:
            operations = options.get("operations")
            result = await execute_bash_with_operations(
                resolved_command,
                self.session_manager.get_cwd(),
                operations if operations is not None else create_local_bash_operations(shell_path=shell_path),
                on_chunk=chunk_callback,
                cancel=self._bash_cancel,
            )

            await self.record_bash_result(command, result, options)
            return result
        finally:
            self._bash_cancel = None

    async def record_bash_result(self, command: str, result: BashResult, options: dict[str, Any] | None = None) -> None:
        """Record a bash execution result in session history. Used by execute_bash
        and by extensions that handle bash execution themselves."""
        options = options or {}
        bash_message = BashExecutionMessage(
            command=command,
            output=result.output,
            exit_code=result.exit_code,
            cancelled=result.cancelled,
            truncated=result.truncated,
            full_output_path=result.full_output_path,
            timestamp=_now_ms(),
            exclude_from_context=options.get("exclude_from_context", options.get("excludeFromContext")),
        )

        # If agent is streaming, defer adding to avoid breaking tool_use/tool_result
        # ordering; flushed on agent settle. Decision and append happen under
        # the guard the settle path clears the flag under, so a recording that
        # saw the run as active is guaranteed visible to the settle flush.
        with self._state_guard:
            if self._is_agent_run_active:
                self._pending_bash_messages.append(bash_message)
                return
        # Add to agent state immediately
        self.agent.state.messages.append(bash_message)
        # Save to session
        await self.session_manager.append_message(bash_message)

    def abort_bash(self) -> None:
        """Cancel running bash command."""
        if self._bash_cancel is not None:
            self._bash_cancel.cancel()

    @property
    def is_bash_running(self) -> bool:
        return self._bash_cancel is not None

    @property
    def has_pending_bash_messages(self) -> bool:
        return len(self._pending_bash_messages) > 0

    async def _flush_pending_bash_messages(self) -> None:
        """Flush pending bash messages to agent state and session. Called after the
        agent turn completes to maintain proper message ordering. Drains with a
        swap-under-guard loop so a concurrent `record_bash_result` append can
        never be dropped by the list reset."""
        while True:
            with self._state_guard:
                pending, self._pending_bash_messages = self._pending_bash_messages, []
            if not pending:
                return
            for bash_message in pending:
                self.agent.state.messages.append(bash_message)
                await self.session_manager.append_message(bash_message)

    # =========================================================================
    # Session Management
    # =========================================================================

    async def set_session_name(self, name: str) -> None:
        """Set a display name for the current session."""
        await self.session_manager.append_session_info(name)
        event = SessionInfoChangedEvent(name=self.session_manager.get_session_name())
        self._emit(event)
        tonio.spawn.without_tracking(self._extension_runner.emit({"type": "session_info_changed", "name": event.name}))

    # =========================================================================
    # Tree Navigation
    # =========================================================================

    async def navigate_tree(self, target_id: str, options: dict[str, Any] | None = None) -> NavigateTreeResult:
        """Navigate to a different node in the session tree. Unlike fork() this
        stays in the same file."""
        options = options or {}
        old_leaf_id = self.session_manager.get_leaf_id()

        # No-op if already at target
        if target_id == old_leaf_id:
            return NavigateTreeResult(cancelled=False)

        # Model required for summarization
        if options.get("summarize") and self.model is None:
            raise Exception("No model available for summarization")

        target_entry = self.session_manager.get_entry(target_id)
        if target_entry is None:
            raise Exception(f"Entry {target_id} not found")

        # Collect entries to summarize (from old leaf to common ancestor)
        collected = collect_entries_for_branch_summary(self.session_manager, old_leaf_id, target_id)
        entries_to_summarize = collected.entries

        # Prepare event data - mutable so extensions can override
        custom_instructions = options.get("custom_instructions")
        replace_instructions = options.get("replace_instructions")
        label = options.get("label")

        preparation = {
            "targetId": target_id,
            "oldLeafId": old_leaf_id,
            "commonAncestorId": collected.common_ancestor_id,
            "entriesToSummarize": entries_to_summarize,
            "userWantsSummary": bool(options.get("summarize")),
            "customInstructions": custom_instructions,
            "replaceInstructions": replace_instructions,
            "label": label,
        }

        # Set up cancel token for summarization
        self._branch_summary_cancel = CancelToken()

        try:
            extension_summary: dict[str, Any] | None = None
            from_extension = False

            # Emit session_before_tree event
            if self._extension_runner.has_handlers("session_before_tree"):
                result = await self._extension_runner.emit(
                    {
                        "type": "session_before_tree",
                        "preparation": preparation,
                        "signal": self._branch_summary_cancel,
                    }
                )

                if isinstance(result, dict):
                    if result.get("cancel"):
                        return NavigateTreeResult(cancelled=True)

                    if result.get("summary") is not None and options.get("summarize"):
                        extension_summary = result["summary"]
                        from_extension = True

                    # Allow extensions to override instructions and label
                    if "customInstructions" in result:
                        custom_instructions = result["customInstructions"]
                    if "replaceInstructions" in result:
                        replace_instructions = result["replaceInstructions"]
                    if "label" in result:
                        label = result["label"]

            # Run default summarizer if needed
            summary_text: str | None = None
            summary_details: Any = None
            summary_usage: Usage | None = None
            if options.get("summarize") and entries_to_summarize and extension_summary is None:
                model = self.model
                auth = await self._get_summarization_request_auth(model)
                branch_summary_settings = self.settings_manager.get_branch_summary_settings()
                result = await generate_branch_summary(
                    entries_to_summarize,
                    model=model,
                    api_key=auth["api_key"],
                    headers=auth["headers"],
                    env=auth["env"],
                    cancel=self._branch_summary_cancel,
                    custom_instructions=custom_instructions,
                    replace_instructions=replace_instructions,
                    reserve_tokens=branch_summary_settings["reserve_tokens"],
                    stream_fn=self.agent.stream_function,
                    retry=_retry_policy_from(self.settings_manager.get_retry_settings()),
                    callbacks=self._summarization_retry_callbacks({"source": "branchSummary"}),
                )
                if result.aborted:
                    return NavigateTreeResult(cancelled=True, aborted=True)
                if result.error:
                    raise Exception(result.error)
                summary_text = result.summary
                summary_usage = result.usage
                summary_details = {
                    "readFiles": result.read_files or [],
                    "modifiedFiles": result.modified_files or [],
                }
            elif extension_summary is not None:
                summary_text = extension_summary.get("summary")
                summary_details = extension_summary.get("details")
                summary_usage = extension_summary.get("usage")

            # Determine the new leaf position based on target type
            editor_text: str | None = None

            if target_entry.get("type") == "message" and getattr(target_entry.get("message"), "role", None) == "user":
                # User message: leaf = parent (None if root), text goes to editor
                new_leaf_id = target_entry.get("parentId")
                editor_text = content_text(target_entry["message"].content, "")
            elif target_entry.get("type") == "custom_message":
                # Custom message: leaf = parent (None if root), text goes to editor
                new_leaf_id = target_entry.get("parentId")
                editor_text = content_text(target_entry.get("content"), "")
            else:
                # Non-user message: leaf = selected node
                new_leaf_id = target_id

            # Switch leaf (with or without summary). The summary is attached at the
            # navigation target position (new_leaf_id), not the old branch.
            summary_entry: dict[str, Any] | None = None
            if summary_text:
                summary_id = await self.session_manager.branch_with_summary(
                    new_leaf_id, summary_text, summary_details, from_extension, summary_usage
                )
                summary_entry = self.session_manager.get_entry(summary_id)

                # Attach label to the summary entry
                if label:
                    await self.session_manager.append_label_change(summary_id, label)
            elif new_leaf_id is None:
                # No summary, navigating to root - reset leaf
                self.session_manager.reset_leaf()
            else:
                # No summary, navigating to non-root
                self.session_manager.branch(new_leaf_id)

            # Attach label to target entry when not summarizing
            if label and not summary_text:
                await self.session_manager.append_label_change(target_id, label)

            # Update agent state
            session_context = self.session_manager.build_session_context()
            self.agent.state.messages = session_context.messages

            # Emit session_tree event
            await self._extension_runner.emit(
                {
                    "type": "session_tree",
                    "newLeafId": self.session_manager.get_leaf_id(),
                    "oldLeafId": old_leaf_id,
                    "summaryEntry": summary_entry,
                    "fromExtension": from_extension if summary_text else None,
                }
            )

            return NavigateTreeResult(cancelled=False, editor_text=editor_text, summary_entry=summary_entry)
        finally:
            self._branch_summary_cancel = None

    def get_user_messages_for_forking(self) -> list[dict[str, str]]:
        """All user messages from session for fork selector."""
        result: list[dict[str, str]] = []

        for entry in self.session_manager.get_entries():
            if entry.get("type") != "message":
                continue
            message = entry.get("message")
            if getattr(message, "role", None) != "user":
                continue

            text = content_text(message.content, "")
            if text:
                result.append({"entryId": entry["id"], "text": text})

        return result

    async def export_to_html(self, output_path: str | None = None) -> str:
        """Export session to HTML; returns the path to the exported file."""
        # lazy: core <-> modes import cycle (see modes/__init__.py); export_html
        # closes the same loop back through here
        from ..modes.interactive.theme import get_theme_by_name, theme
        from .export_html import create_tool_html_renderer, export_session_to_html

        configured_theme_name = self.settings_manager.get_theme()
        theme_name = (
            configured_theme_name if configured_theme_name and await get_theme_by_name(configured_theme_name) else None
        )

        # Create tool renderer for custom tool HTML rendering
        tool_renderer = create_tool_html_renderer(
            {
                "getToolDefinition": self.get_tool_definition,
                "theme": theme,
                "cwd": self.session_manager.get_cwd(),
            }
        )

        return await export_session_to_html(
            self.session_manager,
            self.state,
            {"outputPath": output_path, "themeName": theme_name, "toolRenderer": tool_renderer},
        )

    def get_session_stats(self) -> SessionStats:
        """Session statistics. Aggregates over ALL session entries (including
        history that was compacted away), so token/cost totals reflect what was
        actually billed."""
        # lazy: import cycle within core
        from .usage_totals import add_usage_to_totals, create_usage_totals

        user_messages = 0
        assistant_messages = 0
        tool_results = 0
        total_messages = 0
        tool_calls = 0
        usage_totals = create_usage_totals()

        for entry in self.session_manager.get_entries():
            if entry.get("type") in ("branch_summary", "compaction") and entry.get("usage"):
                add_usage_to_totals(usage_totals, entry["usage"])
            if entry.get("type") != "message":
                continue
            total_messages += 1
            message = entry.get("message")
            role = getattr(message, "role", None)
            if role == "user":
                user_messages += 1
            elif role == "toolResult":
                tool_results += 1
                if message.usage is not None:
                    add_usage_to_totals(usage_totals, message.usage)
            elif role == "assistant":
                assistant_messages += 1
                if isinstance(message.content, list):
                    tool_calls += sum(1 for block in message.content if getattr(block, "type", None) == "toolCall")
                add_usage_to_totals(usage_totals, message.usage)

        return SessionStats(
            session_file=self.session_file,
            session_id=self.session_id,
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            tool_calls=tool_calls,
            tool_results=tool_results,
            total_messages=total_messages,
            tokens=SessionTokens(
                input=usage_totals.input,
                output=usage_totals.output,
                cache_read=usage_totals.cache_read,
                cache_write=usage_totals.cache_write,
                total=usage_totals.input + usage_totals.output + usage_totals.cache_read + usage_totals.cache_write,
            ),
            cost=usage_totals.cost,
            context_usage=self.get_context_usage(),
        )

    def get_context_usage(self) -> ContextUsage | None:
        model = self.model
        if model is None:
            return None

        context_window = model.context_window or 0
        if context_window <= 0:
            return None

        # After compaction, the last assistant usage reflects pre-compaction context
        # size. We can only trust usage from an assistant that responded after the
        # latest compaction; until then the context token count is unknown.
        branch_entries = self.session_manager.get_branch()
        latest_compaction = get_latest_compaction_entry(branch_entries)

        if latest_compaction is not None:
            compaction_index = max(
                (i for i, entry in enumerate(branch_entries) if entry is latest_compaction), default=-1
            )
            has_post_compaction_usage = False
            for i in range(len(branch_entries) - 1, compaction_index, -1):
                entry = branch_entries[i]
                if entry.get("type") == "message" and getattr(entry.get("message"), "role", None) == "assistant":
                    assistant = entry["message"]
                    if (
                        assistant.stop_reason not in ("aborted", "error")
                        and calculate_context_tokens(assistant.usage) > 0
                    ):
                        has_post_compaction_usage = True
                        break

            if not has_post_compaction_usage:
                return ContextUsage(tokens=None, context_window=context_window, percent=None)

        estimate = estimate_context_tokens(self.messages)
        percent = (estimate.tokens / context_window) * 100

        return ContextUsage(tokens=estimate.tokens, context_window=context_window, percent=percent)

    async def export_to_jsonl(self, output_path: str | None = None) -> str:
        """Export the current session branch to a JSONL file: session header
        followed by all entries on the current branch path (re-chained linear)."""

        # lazy: import cycle within core
        from .session_manager import _dump_json, _entry_to_wire

        default_name = f"session-{_iso_now().replace(':', '-').replace('.', '-')}.jsonl"
        file_path = resolve_path(output_path if output_path is not None else default_name, os.getcwd())
        directory = os.path.dirname(file_path)
        if directory and not await fs.Path(directory).exists():
            await fs.Path(directory).mkdir(parents=True, exist_ok=True)

        header = {
            "type": "session",
            "version": CURRENT_SESSION_VERSION,
            "id": self.session_manager.get_session_id(),
            "timestamp": _iso_now(),
            "cwd": self.session_manager.get_cwd(),
        }

        branch_entries = self.session_manager.get_branch()
        lines = [json.dumps(header, ensure_ascii=False, separators=(",", ":"))]

        # Re-chain parentIds to form a linear sequence
        prev_id: str | None = None
        for entry in branch_entries:
            linear = dict(entry)
            linear["parentId"] = prev_id
            lines.append(_dump_json(_entry_to_wire(linear)))
            prev_id = entry["id"]

        await fs.Path(file_path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
        return file_path

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_last_assistant_text(self) -> str | None:
        """Text content of last assistant message (for /copy)."""
        last_assistant = None
        for message in reversed(self.messages):
            if getattr(message, "role", None) != "assistant":
                continue
            # Skip aborted messages with no content
            if message.stop_reason == "aborted" and len(message.content) == 0:
                continue
            last_assistant = message
            break

        if last_assistant is None:
            return None

        text = "".join(block.text for block in last_assistant.content if getattr(block, "type", None) == "text")
        return text.strip() or None

    # =========================================================================
    # Extension System
    # =========================================================================

    def create_replaced_session_context(self) -> Any:
        context = self._extension_runner.create_command_context()
        context.send_message = self.send_custom_message
        context.send_user_message = self.send_user_message
        return context

    def has_extension_handlers(self, event_type: str) -> bool:
        return self._extension_runner.has_handlers(event_type)

    @property
    def extension_runner(self) -> ExtensionRunner:
        return self._extension_runner
