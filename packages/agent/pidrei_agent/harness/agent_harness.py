"""AgentHarness (port of pi `harness/agent-harness.ts`).

Session-bound orchestration over the low-level agent loop: hooks, prompt
templates and skills invocation, provider-request hooks, tool-context binding,
pending session writes, compaction, and tree navigation.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import tonio.colored as tonio

from pidrei_ai.types import (
    AssistantMessage,
    ImageContent,
    Model,
    SimpleStreamOptions,
    TextContent,
    Usage,
    UserMessage,
)
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.retry import RetryCallbacks, RetryPolicy
from pidrei_ai.utils.text import content_text

from ..agent_loop import run_agent_loop
from ..types import (
    AfterToolCallResult,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentTool,
    BeforeToolCallResult,
    MessageEndEvent,
    MessageStartEvent,
    QueueMode,
    ThinkingLevel,
    TurnEndEvent,
)
from .compaction.branch_summarization import (
    GenerateBranchSummaryOptions,
    collect_entries_for_branch_summary,
    generate_branch_summary,
)
from .compaction.compaction import DEFAULT_COMPACTION_SETTINGS, CompactionResult, compact, prepare_compaction
from .messages import convert_to_llm
from .prompt_templates import format_prompt_template_invocation
from .session.session import Session
from .skills import format_skill_invocation
from .system_prompt import format_skills_for_system_prompt  # noqa: F401  (re-exported convenience)
from .types import (
    AbortResult,
    AfterProviderResponseEvent,
    AgentHarnessError,
    AgentHarnessResources,
    AgentHarnessStreamOptions,
    AgentHarnessStreamOptionsPatch,
    BeforeAgentStartEvent,
    BeforeProviderPayloadEvent,
    BeforeProviderRequestEvent,
    BranchSummaryError,
    CompactionError,
    HarnessAbortEvent,
    HarnessContextEvent,
    HarnessToolCallEvent,
    HarnessToolResultEvent,
    ModelUpdateEvent,
    NavigateTreeResult,
    QueueUpdateEvent,
    ResourcesUpdateEvent,
    RetryAttemptStartEvent,
    RetryFinishedEvent,
    RetryScheduledEvent,
    SavePointEvent,
    SessionBeforeCompactEvent,
    SessionBeforeTreeEvent,
    SessionCompactEvent,
    SessionError,
    SessionTreeEvent,
    SettledEvent,
    ThinkingLevelUpdateEvent,
    ToolsUpdateEvent,
    TreePreparation,
    to_error,
)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _create_user_message(text: str, images: list[ImageContent] | None = None) -> UserMessage:
    content: list[TextContent | ImageContent] = [TextContent(text=text)]
    if images:
        content.extend(images)
    return UserMessage(content=content, timestamp=_now_ms())


def _create_failure_message(model: Model, error: Any, aborted: bool) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text="")],
        api=model.api,
        provider=model.provider,
        model=model.id,
        stop_reason="aborted" if aborted else "error",
        error_message=str(to_error(error)),
        timestamp=_now_ms(),
        usage=Usage(),
    )


def clone_stream_options(stream_options: AgentHarnessStreamOptions | None) -> AgentHarnessStreamOptions:
    if stream_options is None:
        return AgentHarnessStreamOptions()
    return replace(
        stream_options,
        headers=dict(stream_options.headers) if stream_options.headers is not None else None,
        metadata=dict(stream_options.metadata) if stream_options.metadata is not None else None,
    )


def _find_duplicate_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return duplicates


def apply_stream_options_patch(
    base: AgentHarnessStreamOptions,
    patch: AgentHarnessStreamOptionsPatch | None,
) -> AgentHarnessStreamOptions:
    result = clone_stream_options(base)
    if not patch:
        return result

    for key in ("transport", "timeout_ms", "max_retries", "max_retry_delay_ms", "cache_retention"):
        if key in patch:
            setattr(result, key, patch[key])

    for key in ("headers", "metadata"):
        if key in patch:
            patch_value = patch[key]
            if patch_value is None:
                setattr(result, key, None)
            else:
                merged = dict(getattr(result, key) or {})
                for entry_key, entry_value in patch_value.items():
                    if entry_value is None:
                        merged.pop(entry_key, None)
                    else:
                        merged[entry_key] = entry_value
                setattr(result, key, merged if merged else None)

    return result


_SUBSCRIBER_EVENT_TYPE = "*"


def _normalize_harness_error(error: Any, fallback_code: str) -> AgentHarnessError:
    if isinstance(error, AgentHarnessError):
        return error
    cause = to_error(error)
    if isinstance(cause, SessionError):
        return AgentHarnessError("session", cause.message, cause)
    if isinstance(cause, CompactionError):
        return AgentHarnessError("compaction", cause.message, cause)
    if isinstance(cause, BranchSummaryError):
        return AgentHarnessError("branch_summary", cause.message, cause)
    return AgentHarnessError(fallback_code, str(cause), cause)


def _normalize_hook_error(error: Any) -> AgentHarnessError:
    return _normalize_harness_error(error, "hook")


@dataclass(slots=True)
class _TurnState:
    messages: list[AgentMessage]
    resources: AgentHarnessResources
    tool_context: Any
    stream_options: AgentHarnessStreamOptions
    session_id: str
    system_prompt: str
    model: Model
    thinking_level: ThinkingLevel
    tools: list[Any]
    active_tools: list[Any]


class _BoundTool(AgentTool):
    """Harness tool with the turn's tool context bound (pi: `bindToolContext`)."""

    def __init__(self, tool: Any, context: Any):
        self.name = tool.name
        self.label = tool.label
        self.description = tool.description
        self.parameters = tool.parameters
        self.execution_mode = getattr(tool, "execution_mode", None)
        self.prepare_arguments = getattr(tool, "prepare_arguments", None)
        self._tool = tool
        self._context = context

    async def execute(self, tool_call_id, params, cancel, on_update):
        return await self._tool.execute(tool_call_id, params, cancel, on_update, self._context)


@dataclass(slots=True)
class _ActiveRun:
    done: tonio.Event


@dataclass(slots=True, kw_only=True)
class AgentHarnessOptions:
    session: Session
    # Provider collection used for all model requests (turn streaming,
    # compaction, branch summarization). Auth resolves through the providers' auth.
    models: Any
    model: Model
    tools: list[Any] | None = None
    # Concrete resources available to explicit invocation methods and
    # system-prompt callbacks.
    resources: AgentHarnessResources | None = None
    # str or (context-dict) -> str | Awaitable[str]
    system_prompt: Any = None
    # Static context or zero-argument context provider resolved per turn snapshot.
    tool_context: Any = None
    # Curated stream/provider request options. Snapshotted at turn start.
    stream_options: AgentHarnessStreamOptions | None = None
    # Optional retry policy for generated compaction and branch-summary requests.
    retry: RetryPolicy | None = None
    thinking_level: ThinkingLevel | None = None
    active_tool_names: list[str] | None = None
    steering_mode: QueueMode | None = None
    follow_up_mode: QueueMode | None = None


@dataclass(slots=True)
class _SystemPromptContext:
    session: Session
    model: Model
    thinking_level: ThinkingLevel
    active_tools: list[Any]
    resources: AgentHarnessResources


class AgentHarness:
    """Session-bound stateful wrapper around the low-level agent loop."""

    def __init__(self, options: AgentHarnessOptions):
        self._session = options.session
        self.models = options.models
        self._phase: str = "idle"
        self._run_cancel: CancelToken | None = None
        self._active_run: _ActiveRun | None = None
        self._pending_session_writes: list[dict[str, Any]] = []
        self._resources = options.resources if options.resources is not None else AgentHarnessResources()
        self._stream_options = clone_stream_options(options.stream_options)
        self._retry = options.retry
        self._system_prompt = options.system_prompt
        self._tool_context = options.tool_context
        self._validate_unique_names([tool.name for tool in options.tools or []], "Duplicate tool name(s)")
        self._tools: dict[str, Any] = {tool.name: tool for tool in options.tools or []}
        self._model = options.model
        self._thinking_level: ThinkingLevel = options.thinking_level if options.thinking_level is not None else "off"
        self._active_tool_names = (
            list(options.active_tool_names)
            if options.active_tool_names is not None
            else [tool.name for tool in options.tools or []]
        )
        self._validate_unique_names(self._active_tool_names, "Duplicate active tool name(s)")
        self._validate_tool_names(self._active_tool_names)
        self._steering_queue_mode: QueueMode = options.steering_mode if options.steering_mode else "one-at-a-time"
        self._follow_up_queue_mode: QueueMode = options.follow_up_mode if options.follow_up_mode else "one-at-a-time"
        self._steer_queue: list[AgentMessage] = []
        self._follow_up_queue: list[AgentMessage] = []
        self._next_turn_queue: list[AgentMessage] = []
        self._handlers: dict[str, list[Callable[..., Any]]] = {}

    # --- events -----------------------------------------------------------------

    def _get_handlers(self, type: str) -> list[Callable[..., Any]]:
        return self._handlers.get(type, [])

    async def _emit_own(self, event: Any, signal: CancelToken | None = None) -> None:
        for listener in list(self._get_handlers(_SUBSCRIBER_EVENT_TYPE)):
            try:
                await listener(event, signal)
            except Exception as error:
                raise _normalize_hook_error(error) from error

    async def _emit_any(self, event: Any, signal: CancelToken | None = None) -> None:
        await self._emit_own(event, signal)

    async def _emit_hook(self, event: Any) -> Any:
        handlers = self._get_handlers(event.type)
        if not handlers:
            return None
        last_result: Any = None
        for handler in list(handlers):
            try:
                result = await handler(event)
                if result is not None:
                    last_result = result
            except Exception as error:
                raise _normalize_hook_error(error) from error
        return last_result

    def _retry_callbacks(self, operation: str) -> RetryCallbacks:
        return RetryCallbacks(
            on_retry_scheduled=lambda attempt, max_attempts, delay_ms, error_message: self._emit_own(
                RetryScheduledEvent(
                    operation=operation,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    delay_ms=delay_ms,
                    error_message=error_message,
                )
            ),
            on_retry_attempt_start=lambda: self._emit_own(RetryAttemptStartEvent(operation=operation)),
            on_retry_finished=lambda *args: self._emit_own(RetryFinishedEvent(operation=operation)),
        )

    async def _emit_before_provider_request(
        self, model: Model, session_id: str, stream_options: AgentHarnessStreamOptions
    ) -> AgentHarnessStreamOptions:
        handlers = self._get_handlers("before_provider_request")
        current = clone_stream_options(stream_options)
        if not handlers:
            return current
        for handler in list(handlers):
            try:
                result = await handler(
                    BeforeProviderRequestEvent(
                        model=model, session_id=session_id, stream_options=clone_stream_options(current)
                    )
                )
                if result is not None and result.stream_options is not None:
                    current = apply_stream_options_patch(current, result.stream_options)
            except Exception as error:
                raise _normalize_hook_error(error) from error
        return current

    async def _emit_before_provider_payload(self, model: Model, payload: Any) -> Any:
        handlers = self._get_handlers("before_provider_payload")
        current = payload
        if not handlers:
            return current
        for handler in list(handlers):
            try:
                result = await handler(BeforeProviderPayloadEvent(model=model, payload=current))
                if result is not None:
                    current = result.payload
            except Exception as error:
                raise _normalize_hook_error(error) from error
        return current

    async def _emit_queue_update(self) -> None:
        await self._emit_own(
            QueueUpdateEvent(
                steer=list(self._steer_queue),
                follow_up=list(self._follow_up_queue),
                next_turn=list(self._next_turn_queue),
            )
        )

    def _start_run_promise(self) -> Callable[[], None]:
        run = _ActiveRun(done=tonio.Event())
        self._active_run = run

        def finish() -> None:
            self._active_run = None
            run.done.set()

        return finish

    # --- turn state -------------------------------------------------------------

    async def _resolve_tool_context(self) -> Any:
        # Value-or-callable; the callable branch is async-only.
        if callable(self._tool_context):
            return await self._tool_context()
        return self._tool_context

    async def _create_turn_state(self) -> _TurnState:
        context = await self._session.build_context()
        resources = self.get_resources()
        session_metadata = await self._session.get_metadata()
        tool_context = await self._resolve_tool_context()
        tools = list(self._tools.values())
        active_tools = [self._tools[name] for name in self._active_tool_names if name in self._tools]
        system_prompt = "You are a helpful assistant."
        if isinstance(self._system_prompt, str):
            system_prompt = self._system_prompt
        elif self._system_prompt is not None:
            # String-or-callable; the callable branch is async-only.
            system_prompt = await self._system_prompt(
                _SystemPromptContext(
                    session=self._session,
                    model=self._model,
                    thinking_level=self._thinking_level,
                    active_tools=active_tools,
                    resources=resources,
                )
            )
        return _TurnState(
            messages=context.messages,
            resources=resources,
            tool_context=tool_context,
            stream_options=clone_stream_options(self._stream_options),
            session_id=session_metadata.id,
            system_prompt=system_prompt,
            model=self._model,
            thinking_level=self._thinking_level,
            tools=tools,
            active_tools=active_tools,
        )

    def _create_context(self, turn_state: _TurnState, system_prompt: str | None = None) -> AgentContext:
        return AgentContext(
            system_prompt=system_prompt if system_prompt is not None else turn_state.system_prompt,
            messages=list(turn_state.messages),
            tools=[_BoundTool(tool, turn_state.tool_context) for tool in turn_state.active_tools],
        )

    def _create_stream_fn(self, get_turn_state: Callable[[], _TurnState]):
        async def stream_fn(model, context, stream_options):
            turn_state = get_turn_state()
            snapshot_options = replace(turn_state.stream_options)
            request_options = await self._emit_before_provider_request(model, turn_state.session_id, snapshot_options)

            async def on_payload(payload, _model=None):
                return await self._emit_before_provider_payload(model, payload)

            async def on_response(response, _model=None):
                headers = dict(response.headers)
                await self._emit_own(
                    AfterProviderResponseEvent(status=response.status, headers=headers),
                    stream_options.cancel if stream_options is not None else None,
                )

            return self.models.stream_simple(
                model,
                context,
                SimpleStreamOptions(
                    cache_retention=request_options.cache_retention,
                    headers=request_options.headers,
                    max_retries=request_options.max_retries,
                    max_retry_delay_ms=request_options.max_retry_delay_ms,
                    metadata=request_options.metadata,
                    on_payload=on_payload,
                    on_response=on_response,
                    reasoning=stream_options.reasoning if stream_options is not None else None,
                    cancel=stream_options.cancel if stream_options is not None else None,
                    session_id=turn_state.session_id,
                    timeout_ms=request_options.timeout_ms,
                    transport=request_options.transport,
                ),
            )

        return stream_fn

    async def _drain_queued_messages(self, queue: list[AgentMessage], mode: QueueMode) -> list[AgentMessage]:
        if mode == "all":
            messages = queue[:]
            del queue[: len(messages)]
        else:
            messages = queue[:1]
            del queue[:1]
        if not messages:
            return messages
        try:
            await self._emit_queue_update()
            return messages
        except Exception as error:
            queue[:0] = messages
            raise _normalize_hook_error(error) from error

    def _create_loop_config(
        self,
        get_turn_state: Callable[[], _TurnState],
        set_turn_state: Callable[[_TurnState], None],
    ) -> AgentLoopConfig:
        turn_state = get_turn_state()

        async def transform_context(messages, _cancel):
            result = await self._emit_hook(HarnessContextEvent(messages=list(messages)))
            return result.messages if result is not None and result.messages is not None else messages

        async def before_tool_call(ctx, _cancel):
            result = await self._emit_hook(
                HarnessToolCallEvent(tool_call_id=ctx.tool_call.id, tool_name=ctx.tool_call.name, input=ctx.args)
            )
            if result is None:
                return None
            return BeforeToolCallResult(block=result.block, reason=result.reason)

        async def after_tool_call(ctx, _cancel):
            patch = await self._emit_hook(
                HarnessToolResultEvent(
                    tool_call_id=ctx.tool_call.id,
                    tool_name=ctx.tool_call.name,
                    input=ctx.args,
                    content=ctx.result.content,
                    details=ctx.result.details,
                    is_error=ctx.is_error,
                    usage=ctx.result.usage,
                )
            )
            if patch is None:
                return None
            return AfterToolCallResult(
                content=patch.content,
                details=patch.details,
                is_error=patch.is_error,
                usage=patch.usage,
                terminate=patch.terminate,
            )

        async def prepare_next_turn(_context) -> AgentLoopTurnUpdate:
            await self._flush_pending_session_writes()
            next_turn_state = await self._create_turn_state()
            set_turn_state(next_turn_state)
            return AgentLoopTurnUpdate(
                context=self._create_context(next_turn_state),
                model=next_turn_state.model,
                thinking_level=next_turn_state.thinking_level,
            )

        async def get_steering_messages():
            return await self._drain_queued_messages(self._steer_queue, self._steering_queue_mode)

        async def get_follow_up_messages():
            return await self._drain_queued_messages(self._follow_up_queue, self._follow_up_queue_mode)

        async def convert_context_to_llm(messages):
            return convert_to_llm(messages)

        return AgentLoopConfig(
            model=turn_state.model,
            reasoning=None if turn_state.thinking_level == "off" else turn_state.thinking_level,
            convert_to_llm=convert_context_to_llm,
            transform_context=transform_context,
            before_tool_call=before_tool_call,
            after_tool_call=after_tool_call,
            prepare_next_turn=prepare_next_turn,
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
        )

    # --- validation / session writes ---------------------------------------------

    def _validate_unique_names(self, names: list[str], message: str) -> None:
        duplicates = _find_duplicate_names(names)
        if duplicates:
            raise AgentHarnessError("invalid_argument", f"{message}: {', '.join(duplicates)}")

    def _validate_tool_names(self, tool_names: list[str], tools: dict[str, Any] | None = None) -> None:
        tools = tools if tools is not None else self._tools
        self._validate_unique_names(tool_names, "Duplicate active tool name(s)")
        missing = [name for name in tool_names if name not in tools]
        if missing:
            raise AgentHarnessError("invalid_argument", f"Unknown tool(s): {', '.join(missing)}")

    async def _flush_pending_session_writes(self) -> None:
        while self._pending_session_writes:
            write = self._pending_session_writes[0]
            if write["type"] == "message":
                await self._session.append_message(write["message"])
            elif write["type"] == "model_change":
                await self._session.append_model_change(write["provider"], write["model_id"])
            elif write["type"] == "thinking_level_change":
                await self._session.append_thinking_level_change(write["thinking_level"])
            elif write["type"] == "active_tools_change":
                await self._session.append_active_tools_change(write["active_tool_names"])
            elif write["type"] == "custom":
                await self._session.append_custom_entry(write["custom_type"], write.get("data"))
            elif write["type"] == "custom_message":
                await self._session.append_custom_message_entry(
                    write["custom_type"], write["content"], write["display"], write.get("details")
                )
            elif write["type"] == "label":
                await self._session.append_label(write["target_id"], write.get("label"))
            elif write["type"] == "session_info":
                await self._session.append_session_name(write.get("name") or "")
            elif write["type"] == "leaf":
                await self._session.get_storage().set_leaf_id(write["target_id"])
            self._pending_session_writes.pop(0)

    async def _handle_agent_event(self, event: AgentEvent, signal: CancelToken | None = None) -> None:
        if event.type == "message_end":
            await self._session.append_message(event.message)
            await self._emit_any(event, signal)
            return
        if event.type == "turn_end":
            event_error: Exception | None = None
            try:
                await self._emit_any(event, signal)
            except Exception as error:
                event_error = error
            had_pending_mutations = len(self._pending_session_writes) > 0
            await self._flush_pending_session_writes()
            if event_error is not None:
                raise event_error
            await self._emit_own(SavePointEvent(had_pending_mutations=had_pending_mutations))
            return
        if event.type == "agent_end":
            await self._flush_pending_session_writes()
            self._phase = "idle"
            await self._emit_any(event, signal)
            await self._emit_own(SettledEvent(next_turn_count=len(self._next_turn_queue)), signal)
            return
        await self._emit_any(event, signal)

    async def _emit_run_failure(
        self, model: Model, error: Any, aborted: bool, signal: CancelToken
    ) -> list[AgentMessage]:
        failure_message = _create_failure_message(model, error, aborted)
        await self._handle_agent_event(MessageStartEvent(message=failure_message), signal)
        await self._handle_agent_event(MessageEndEvent(message=failure_message), signal)
        await self._handle_agent_event(TurnEndEvent(message=failure_message, tool_results=[]), signal)
        await self._handle_agent_event(AgentEndEvent(messages=[failure_message]), signal)
        return [failure_message]

    # --- turn execution -----------------------------------------------------------

    async def _execute_turn(
        self, turn_state: _TurnState, text: str, images: list[ImageContent] | None = None
    ) -> AssistantMessage:
        active_turn_state = turn_state
        messages: list[AgentMessage] = [_create_user_message(text, images)]
        if self._next_turn_queue:
            queued_messages = self._next_turn_queue[:]
            self._next_turn_queue.clear()
            try:
                await self._emit_queue_update()
            except Exception as error:
                self._next_turn_queue[:0] = queued_messages
                raise _normalize_hook_error(error) from error
            messages = [*queued_messages, messages[0]]
        before_result = await self._emit_hook(
            BeforeAgentStartEvent(
                prompt=text,
                images=images,
                system_prompt=turn_state.system_prompt,
                resources=turn_state.resources,
            )
        )
        if before_result is not None and before_result.messages is not None:
            messages = [*messages, *before_result.messages]

        run_cancel = CancelToken()
        self._run_cancel = run_cancel

        def get_turn_state() -> _TurnState:
            return active_turn_state

        def set_turn_state(next_turn_state: _TurnState) -> None:
            nonlocal active_turn_state
            active_turn_state = next_turn_state

        async def run() -> list[AgentMessage]:
            try:
                return await run_agent_loop(
                    messages,
                    self._create_context(
                        turn_state, before_result.system_prompt if before_result is not None else None
                    ),
                    self._create_loop_config(get_turn_state, set_turn_state),
                    lambda event: self._handle_agent_event(event, run_cancel),
                    run_cancel,
                    self._create_stream_fn(get_turn_state),
                )
            except Exception as error:
                try:
                    return await self._emit_run_failure(
                        active_turn_state.model, error, run_cancel.cancelled, run_cancel
                    )
                except Exception as failure_error:
                    cause = ExceptionGroup(
                        "Agent run failed and failure reporting failed",
                        [to_error(error), to_error(failure_error)],
                    )
                    raise AgentHarnessError("unknown", str(cause), cause) from cause

        try:
            new_messages = await run()
            for message in reversed(new_messages):
                if getattr(message, "role", None) == "assistant":
                    return message
            raise AgentHarnessError("invalid_state", "AgentHarness prompt completed without an assistant message")
        finally:
            try:
                await self._flush_pending_session_writes()
            finally:
                self._run_cancel = None

    async def prompt(self, text: str, images: list[ImageContent] | None = None) -> AssistantMessage:
        if self._phase != "idle":
            raise AgentHarnessError("busy", "AgentHarness is busy")
        self._phase = "turn"
        finish_run_promise = self._start_run_promise()
        try:
            turn_state = await self._create_turn_state()
            return await self._execute_turn(turn_state, text, images)
        except Exception as error:
            self._phase = "idle"
            raise _normalize_harness_error(error, "unknown") from error
        finally:
            finish_run_promise()

    async def skill(self, name: str, additional_instructions: str | None = None) -> AssistantMessage:
        if self._phase != "idle":
            raise AgentHarnessError("busy", "AgentHarness is busy")
        self._phase = "turn"
        finish_run_promise = self._start_run_promise()
        try:
            turn_state = await self._create_turn_state()
            skill = next((candidate for candidate in turn_state.resources.skills or [] if candidate.name == name), None)
            if skill is None:
                raise AgentHarnessError("invalid_argument", f"Unknown skill: {name}")
            return await self._execute_turn(turn_state, format_skill_invocation(skill, additional_instructions))
        except Exception as error:
            self._phase = "idle"
            raise _normalize_harness_error(error, "unknown") from error
        finally:
            finish_run_promise()

    async def prompt_from_template(self, name: str, args: list[str] | None = None) -> AssistantMessage:
        if self._phase != "idle":
            raise AgentHarnessError("busy", "AgentHarness is busy")
        self._phase = "turn"
        finish_run_promise = self._start_run_promise()
        try:
            turn_state = await self._create_turn_state()
            template = next(
                (candidate for candidate in turn_state.resources.prompt_templates or [] if candidate.name == name),
                None,
            )
            if template is None:
                raise AgentHarnessError("invalid_argument", f"Unknown prompt template: {name}")
            return await self._execute_turn(
                turn_state, format_prompt_template_invocation(template, args if args is not None else [])
            )
        except Exception as error:
            self._phase = "idle"
            raise _normalize_harness_error(error, "unknown") from error
        finally:
            finish_run_promise()

    # --- queueing -----------------------------------------------------------------

    async def steer(self, text: str, images: list[ImageContent] | None = None) -> None:
        if self._phase == "idle":
            raise AgentHarnessError("invalid_state", "Cannot steer while idle")
        self._steer_queue.append(_create_user_message(text, images))
        await self._emit_queue_update()

    async def follow_up(self, text: str, images: list[ImageContent] | None = None) -> None:
        if self._phase == "idle":
            raise AgentHarnessError("invalid_state", "Cannot follow up while idle")
        self._follow_up_queue.append(_create_user_message(text, images))
        await self._emit_queue_update()

    async def next_turn(self, text: str, images: list[ImageContent] | None = None) -> None:
        self._next_turn_queue.append(_create_user_message(text, images))
        await self._emit_queue_update()

    async def append_message(self, message: AgentMessage) -> None:
        try:
            if self._phase == "idle":
                await self._session.append_message(message)
            else:
                self._pending_session_writes.append({"type": "message", "message": message})
        except Exception as error:
            raise _normalize_harness_error(error, "session") from error

    # --- compaction / tree --------------------------------------------------------

    async def compact(self, custom_instructions: str | None = None) -> CompactionResult:
        if self._phase != "idle":
            raise AgentHarnessError("busy", "compact() requires idle harness")
        self._phase = "compaction"
        try:
            model = self._model
            if model is None:
                raise AgentHarnessError("invalid_state", "No model set for compaction")
            branch_entries = await self._session.get_branch()
            preparation_result = prepare_compaction(branch_entries, DEFAULT_COMPACTION_SETTINGS)
            if not preparation_result.ok:
                raise preparation_result.error
            preparation = preparation_result.value
            if preparation is None:
                raise AgentHarnessError("compaction", "Nothing to compact")
            hook_result = await self._emit_hook(
                SessionBeforeCompactEvent(
                    preparation=preparation,
                    branch_entries=branch_entries,
                    custom_instructions=custom_instructions,
                    signal=CancelToken(),
                )
            )
            if hook_result is not None and hook_result.cancel:
                raise AgentHarnessError("compaction", "Compaction cancelled")
            provided = hook_result.compaction if hook_result is not None else None
            if provided is not None:
                result = provided
            else:
                compact_result = await compact(
                    preparation,
                    self.models,
                    model,
                    custom_instructions,
                    None,
                    self._thinking_level,
                    self._retry,
                    self._retry_callbacks("compaction"),
                )
                if not compact_result.ok:
                    raise compact_result.error
                result = compact_result.value
            entry_id = await self._session.append_compaction(
                result.summary,
                result.first_kept_entry_id,
                result.tokens_before,
                result.details,
                provided is not None,
                result.usage,
                result.retained_tail,
            )
            entry = await self._session.get_entry(entry_id)
            if entry is not None and entry.type == "compaction":
                await self._emit_own(SessionCompactEvent(compaction_entry=entry, from_hook=provided is not None))
            return result
        except Exception as error:
            raise _normalize_harness_error(error, "compaction") from error
        finally:
            self._phase = "idle"

    async def navigate_tree(
        self,
        target_id: str,
        summarize: bool = False,
        custom_instructions: str | None = None,
        replace_instructions: bool | None = None,
        label: str | None = None,
    ) -> NavigateTreeResult:
        if self._phase != "idle":
            raise AgentHarnessError("busy", "navigateTree() requires idle harness")
        self._phase = "branch_summary"
        try:
            old_leaf_id = await self._session.get_leaf_id()
            if old_leaf_id == target_id:
                return NavigateTreeResult(cancelled=False)
            target_entry = await self._session.get_entry(target_id)
            if target_entry is None:
                raise AgentHarnessError("invalid_argument", f"Entry {target_id} not found")
            collected = await collect_entries_for_branch_summary(self._session, old_leaf_id, target_id)
            preparation = TreePreparation(
                target_id=target_id,
                old_leaf_id=old_leaf_id,
                common_ancestor_id=collected.common_ancestor_id,
                entries_to_summarize=collected.entries,
                user_wants_summary=summarize,
                custom_instructions=custom_instructions,
                replace_instructions=replace_instructions,
                label=label,
            )
            hook_result = await self._emit_hook(SessionBeforeTreeEvent(preparation=preparation, signal=CancelToken()))
            if hook_result is not None and hook_result.cancel:
                return NavigateTreeResult(cancelled=True)
            summary_entry = None
            summary_text = hook_result.summary.summary if hook_result is not None and hook_result.summary else None
            summary_details = hook_result.summary.details if hook_result is not None and hook_result.summary else None
            summary_usage = hook_result.summary.usage if hook_result is not None and hook_result.summary else None
            if not summary_text and summarize and collected.entries:
                model = self._model
                if model is None:
                    raise AgentHarnessError("invalid_state", "No model set for branch summary")
                branch_summary = await generate_branch_summary(
                    collected.entries,
                    GenerateBranchSummaryOptions(
                        models=self.models,
                        model=model,
                        cancel=CancelToken(),
                        custom_instructions=(
                            hook_result.custom_instructions
                            if hook_result is not None and hook_result.custom_instructions is not None
                            else custom_instructions
                        ),
                        replace_instructions=(
                            hook_result.replace_instructions
                            if hook_result is not None and hook_result.replace_instructions is not None
                            else replace_instructions
                        ),
                        retry=self._retry,
                        callbacks=self._retry_callbacks("branch_summary"),
                    ),
                )
                if not branch_summary.ok:
                    if branch_summary.error.code == "aborted":
                        return NavigateTreeResult(cancelled=True)
                    raise AgentHarnessError("branch_summary", branch_summary.error.message, branch_summary.error)
                summary_text = branch_summary.value.summary
                summary_usage = branch_summary.value.usage
                summary_details = {
                    "readFiles": branch_summary.value.read_files,
                    "modifiedFiles": branch_summary.value.modified_files,
                }
            editor_text: str | None = None
            if target_entry.type == "message" and getattr(target_entry.message, "role", None) == "user":
                new_leaf_id = target_entry.parent_id
                editor_text = content_text(target_entry.message.content, "")
            elif target_entry.type == "custom_message":
                new_leaf_id = target_entry.parent_id
                editor_text = content_text(target_entry.content, "")
            else:
                new_leaf_id = target_id
            summary_id = await self._session.move_to(
                new_leaf_id,
                (
                    {
                        "summary": summary_text,
                        "details": summary_details,
                        "usage": summary_usage,
                        "from_hook": hook_result is not None and hook_result.summary is not None,
                    }
                    if summary_text
                    else None
                ),
            )
            if summary_id is not None:
                entry = await self._session.get_entry(summary_id)
                if entry is not None and entry.type == "branch_summary":
                    summary_entry = entry
            await self._emit_own(
                SessionTreeEvent(
                    new_leaf_id=await self._session.get_leaf_id(),
                    old_leaf_id=old_leaf_id,
                    summary_entry=summary_entry,
                    from_hook=hook_result is not None and hook_result.summary is not None,
                )
            )
            return NavigateTreeResult(cancelled=False, editor_text=editor_text, summary_entry=summary_entry)
        except Exception as error:
            raise _normalize_harness_error(error, "branch_summary") from error
        finally:
            self._phase = "idle"

    # --- accessors ----------------------------------------------------------------

    def get_model(self) -> Model:
        return self._model

    async def set_model(self, model: Model) -> None:
        try:
            previous_model = self._model
            if self._phase == "idle":
                await self._session.append_model_change(model.provider, model.id)
            else:
                self._pending_session_writes.append(
                    {"type": "model_change", "provider": model.provider, "model_id": model.id}
                )
            self._model = model
            await self._emit_own(ModelUpdateEvent(model=model, previous_model=previous_model, source="set"))
        except Exception as error:
            raise _normalize_harness_error(error, "session") from error

    def get_thinking_level(self) -> ThinkingLevel:
        return self._thinking_level

    async def set_thinking_level(self, level: ThinkingLevel) -> None:
        try:
            previous_level = self._thinking_level
            if self._phase == "idle":
                await self._session.append_thinking_level_change(level)
            else:
                self._pending_session_writes.append({"type": "thinking_level_change", "thinking_level": level})
            self._thinking_level = level
            await self._emit_own(ThinkingLevelUpdateEvent(level=level, previous_level=previous_level))
        except Exception as error:
            raise _normalize_harness_error(error, "session") from error

    def get_tools(self) -> list[Any]:
        return list(self._tools.values())

    async def set_tools(self, tools: list[Any], active_tool_names: list[str] | None = None) -> None:
        try:
            self._validate_unique_names([tool.name for tool in tools], "Duplicate tool name(s)")
            next_tools = {tool.name: tool for tool in tools}
            next_active_tool_names = list(active_tool_names) if active_tool_names else self._active_tool_names
            self._validate_tool_names(next_active_tool_names, next_tools)
            previous_tool_names = list(self._tools.keys())
            previous_active_tool_names = list(self._active_tool_names)
            if self._phase == "idle":
                await self._session.append_active_tools_change(next_active_tool_names)
            else:
                self._pending_session_writes.append(
                    {"type": "active_tools_change", "active_tool_names": list(next_active_tool_names)}
                )
            self._tools = next_tools
            self._active_tool_names = list(next_active_tool_names)
            await self._emit_own(
                ToolsUpdateEvent(
                    tool_names=list(self._tools.keys()),
                    previous_tool_names=previous_tool_names,
                    active_tool_names=list(self._active_tool_names),
                    previous_active_tool_names=previous_active_tool_names,
                    source="set",
                )
            )
        except Exception as error:
            raise _normalize_harness_error(error, "invalid_argument") from error

    def get_active_tools(self) -> list[Any]:
        return [self._tools[name] for name in self._active_tool_names]

    async def set_active_tools(self, tool_names: list[str]) -> None:
        try:
            self._validate_tool_names(tool_names)
            previous_tool_names = list(self._tools.keys())
            previous_active_tool_names = list(self._active_tool_names)
            if self._phase == "idle":
                await self._session.append_active_tools_change(tool_names)
            else:
                self._pending_session_writes.append(
                    {"type": "active_tools_change", "active_tool_names": list(tool_names)}
                )
            self._active_tool_names = list(tool_names)
            await self._emit_own(
                ToolsUpdateEvent(
                    tool_names=list(self._tools.keys()),
                    previous_tool_names=previous_tool_names,
                    active_tool_names=list(self._active_tool_names),
                    previous_active_tool_names=previous_active_tool_names,
                    source="set",
                )
            )
        except Exception as error:
            raise _normalize_harness_error(error, "invalid_argument") from error

    def get_steering_mode(self) -> QueueMode:
        return self._steering_queue_mode

    async def set_steering_mode(self, mode: QueueMode) -> None:
        self._steering_queue_mode = mode

    def get_follow_up_mode(self) -> QueueMode:
        return self._follow_up_queue_mode

    async def set_follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_queue_mode = mode

    def get_resources(self) -> AgentHarnessResources:
        return AgentHarnessResources(
            skills=list(self._resources.skills) if self._resources.skills is not None else None,
            prompt_templates=(
                list(self._resources.prompt_templates) if self._resources.prompt_templates is not None else None
            ),
        )

    async def set_resources(self, resources: AgentHarnessResources) -> None:
        previous_resources = self.get_resources()
        self._resources = AgentHarnessResources(
            skills=list(resources.skills) if resources.skills is not None else None,
            prompt_templates=list(resources.prompt_templates) if resources.prompt_templates is not None else None,
        )
        await self._emit_own(
            ResourcesUpdateEvent(resources=self.get_resources(), previous_resources=previous_resources)
        )

    def get_stream_options(self) -> AgentHarnessStreamOptions:
        return clone_stream_options(self._stream_options)

    async def set_stream_options(self, stream_options: AgentHarnessStreamOptions) -> None:
        self._stream_options = clone_stream_options(stream_options)

    # --- lifecycle ----------------------------------------------------------------

    async def abort(self) -> AbortResult:
        cleared_steer = list(self._steer_queue)
        cleared_follow_up = list(self._follow_up_queue)
        self._steer_queue = []
        self._follow_up_queue = []
        if self._run_cancel is not None:
            self._run_cancel.cancel()
        errors: list[Exception] = []
        try:
            await self._emit_queue_update()
        except Exception as error:
            errors.append(to_error(error))
        try:
            await self.wait_for_idle()
        except Exception as error:
            errors.append(to_error(error))
        try:
            await self._emit_own(HarnessAbortEvent(cleared_steer=cleared_steer, cleared_follow_up=cleared_follow_up))
        except Exception as error:
            errors.append(to_error(error))
        if errors:
            cause = errors[0] if len(errors) == 1 else ExceptionGroup("Abort completed with errors", errors)
            raise _normalize_harness_error(cause, "hook") from cause
        return AbortResult(cleared_steer=cleared_steer, cleared_follow_up=cleared_follow_up)

    async def wait_for_idle(self) -> None:
        run = self._active_run
        if run is None:
            return
        await run.done.wait(None)

    def subscribe(self, listener: Callable[..., Any]) -> Callable[[], None]:
        handlers = self._handlers.setdefault(_SUBSCRIBER_EVENT_TYPE, [])
        handlers.append(listener)

        def unsubscribe() -> None:
            try:
                handlers.remove(listener)
            except ValueError:
                pass

        return unsubscribe

    def on(self, type: str, handler: Callable[..., Any]) -> Callable[[], None]:
        handlers = self._handlers.setdefault(type, [])
        handlers.append(handler)

        def unsubscribe() -> None:
            try:
                handlers.remove(handler)
            except ValueError:
                pass

        return unsubscribe
