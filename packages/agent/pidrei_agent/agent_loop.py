"""Agent loop (port of pi `agent/src/agent-loop.ts`).

Works with `AgentMessage` throughout; transforms to `Message[]` only at the
LLM call boundary.

Concurrency notes (vs pi's single JS thread):
- The event sink may be a sync callable (e.g. `EventStream.push`) or an async
  one; the loop awaits awaitable sink results. Tool `on_update` callbacks
  invoke the sink directly (real-time for sync sinks) and buffer awaitable
  results, which are awaited before the tool call finalizes — mirroring pi's
  update-promise batch.
- Parallel tool execution is true parallelism: prepared calls run as tonio
  tasks. `tool_execution_end` is emitted in completion order; tool-result
  messages are persisted and emitted in assistant source order (pi's ordering
  contract, enforced by construction).
"""

import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

import tonio.colored as tonio

from pidrei_ai.types import AssistantMessage, Context, TextContent, ToolResultMessage
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.event_stream import EventStream
from pidrei_ai.utils.validation import validate_tool_arguments

from .stream_fn import get_default_stream_fn
from .types import (
    AfterToolCallContext,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentStartEvent,
    AgentTool,
    AgentToolCall,
    AgentToolResult,
    BeforeToolCallContext,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    PrepareNextTurnContext,
    ShouldStopAfterTurnContext,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)


# Event sink: sync callable or coroutine function; awaitable results are awaited.
type AgentEventSink = Callable[[AgentEvent], Awaitable[None] | None]


async def _emit(sink: AgentEventSink, event: AgentEvent) -> None:
    result = sink(event)
    if inspect.isawaitable(result):
        await result


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _call_optional(fn: Callable[..., Any] | None, *args: Any) -> Any:
    if fn is None:
        return None
    return await _maybe_await(fn(*args))


def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    cancel: CancelToken | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """Start an agent loop with new prompt messages.

    The prompts are added to the context and events are emitted for them.
    """
    stream = _create_agent_stream()

    async def run() -> None:
        messages = await run_agent_loop(prompts, context, config, stream.push, cancel, stream_fn)
        stream.end(messages)

    tonio.spawn.without_tracking(run())
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    cancel: CancelToken | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """Continue an agent loop from the current context without adding a new message.

    Used for retries — the context already has a user message or tool results.
    The last message in context must convert to a `user` or `toolResult`
    message via `convert_to_llm`; this cannot be validated here.
    """
    if len(context.messages) == 0:
        raise Exception("Cannot continue: no messages in context")

    if getattr(context.messages[-1], "role", None) == "assistant":
        raise Exception("Cannot continue from message role: assistant")

    stream = _create_agent_stream()

    async def run() -> None:
        messages = await run_agent_loop_continue(context, config, stream.push, cancel, stream_fn)
        stream.end(messages)

    tonio.spawn.without_tracking(run())
    return stream


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancel: CancelToken | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    new_messages: list[AgentMessage] = [*prompts]
    current_context = replace(context, messages=[*context.messages, *prompts])

    await _emit(emit, AgentStartEvent())
    await _emit(emit, TurnStartEvent())
    for prompt in prompts:
        await _emit(emit, MessageStartEvent(message=prompt))
        await _emit(emit, MessageEndEvent(message=prompt))

    await _run_loop(
        current_context,
        new_messages,
        config,
        cancel,
        emit,
        stream_fn if stream_fn is not None else get_default_stream_fn(),
    )
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    cancel: CancelToken | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    if len(context.messages) == 0:
        raise Exception("Cannot continue: no messages in context")

    if getattr(context.messages[-1], "role", None) == "assistant":
        raise Exception("Cannot continue from message role: assistant")

    new_messages: list[AgentMessage] = []
    current_context = replace(context)

    await _emit(emit, AgentStartEvent())
    await _emit(emit, TurnStartEvent())

    await _run_loop(
        current_context,
        new_messages,
        config,
        cancel,
        emit,
        stream_fn if stream_fn is not None else get_default_stream_fn(),
    )
    return new_messages


def _create_agent_stream() -> EventStream[AgentEvent, list[AgentMessage]]:
    return EventStream(
        lambda event: event.type == "agent_end",
        lambda event: event.messages if event.type == "agent_end" else [],
    )


async def _run_loop(
    initial_context: AgentContext,
    new_messages: list[AgentMessage],
    initial_config: AgentLoopConfig,
    cancel: CancelToken | None,
    emit: AgentEventSink,
    stream_function: StreamFn,
) -> None:
    """Main loop logic shared by `agent_loop` and `agent_loop_continue`."""
    current_context = initial_context
    config = initial_config
    first_turn = True
    # Check for steering messages at start (user may have typed while waiting).
    pending_messages: list[AgentMessage] = (await _call_optional(config.get_steering_messages)) or []

    # Outer loop: continues when queued follow-up messages arrive after the agent would stop.
    while True:
        has_more_tool_calls = True

        # Inner loop: process tool calls and steering messages.
        while has_more_tool_calls or pending_messages:
            if not first_turn:
                await _emit(emit, TurnStartEvent())
            else:
                first_turn = False

            # Process pending messages (inject before next assistant response).
            if pending_messages:
                for message in pending_messages:
                    await _emit(emit, MessageStartEvent(message=message))
                    await _emit(emit, MessageEndEvent(message=message))
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            # Stream assistant response.
            message = await _stream_assistant_response(current_context, config, cancel, emit, stream_function)
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                await _emit(emit, TurnEndEvent(message=message, tool_results=[]))
                await _emit(emit, AgentEndEvent(messages=new_messages))
                return

            # Check for tool calls.
            tool_calls = [block for block in message.content if block.type == "toolCall"]

            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False
            if tool_calls:
                # A "length" stop means the output was cut off by the token limit, so
                # every tool call in the message may carry truncated arguments. Fail
                # them all instead of executing potentially borked calls.
                executed_tool_batch = (
                    await _fail_tool_calls_from_truncated_message(tool_calls, emit)
                    if message.stop_reason == "length"
                    else await _execute_tool_calls(current_context, message, config, cancel, emit)
                )
                tool_results.extend(executed_tool_batch.messages)
                has_more_tool_calls = not executed_tool_batch.terminate

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await _emit(emit, TurnEndEvent(message=message, tool_results=tool_results))

            next_turn_context = PrepareNextTurnContext(
                message=message,
                tool_results=tool_results,
                context=current_context,
                new_messages=new_messages,
            )
            next_turn_snapshot = await _call_optional(config.prepare_next_turn, next_turn_context)
            if next_turn_snapshot:
                current_context = (
                    next_turn_snapshot.context if next_turn_snapshot.context is not None else current_context
                )
                config = replace(
                    config,
                    model=next_turn_snapshot.model if next_turn_snapshot.model is not None else config.model,
                    reasoning=(
                        config.reasoning
                        if next_turn_snapshot.thinking_level is None
                        else (None if next_turn_snapshot.thinking_level == "off" else next_turn_snapshot.thinking_level)
                    ),
                )

            if await _call_optional(
                config.should_stop_after_turn,
                ShouldStopAfterTurnContext(
                    message=message,
                    tool_results=tool_results,
                    context=current_context,
                    new_messages=new_messages,
                ),
            ):
                await _emit(emit, AgentEndEvent(messages=new_messages))
                return

            pending_messages = (await _call_optional(config.get_steering_messages)) or []

        # Agent would stop here. Check for follow-up messages.
        follow_up_messages = (await _call_optional(config.get_follow_up_messages)) or []
        if follow_up_messages:
            # Set as pending so the inner loop processes them.
            pending_messages = follow_up_messages
            continue

        # No more messages, exit.
        break

    await _emit(emit, AgentEndEvent(messages=new_messages))


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    cancel: CancelToken | None,
    emit: AgentEventSink,
    stream_function: StreamFn,
) -> AssistantMessage:
    """Stream an assistant response from the LLM.

    This is where AgentMessage[] gets transformed to Message[] for the LLM.
    """
    # Apply context transform if configured (AgentMessage[] → AgentMessage[]).
    messages = context.messages
    if config.transform_context is not None:
        messages = await _maybe_await(config.transform_context(messages, cancel))

    # Convert to LLM-compatible messages (AgentMessage[] → Message[]).
    llm_messages = await _maybe_await(config.convert_to_llm(messages))

    # Build LLM context.
    llm_context = Context(system_prompt=context.system_prompt, messages=llm_messages, tools=context.tools)

    # Resolve API key (important for expiring tokens).
    resolved_api_key = (await _call_optional(config.get_api_key, config.model.provider)) or config.api_key

    # pi spreads the whole config into the stream options; the dataclass copy
    # carries the same fields (config extends SimpleStreamOptions).
    response = await _maybe_await(
        stream_function(config.model, llm_context, replace(config, api_key=resolved_api_key, cancel=cancel))
    )

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in response:
        if event.type == "start":
            partial_message = event.partial
            context.messages.append(partial_message)
            added_partial = True
            await _emit(emit, MessageStartEvent(message=replace(partial_message)))
        elif event.type in (
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ):
            if partial_message is not None:
                partial_message = event.partial
                context.messages[-1] = partial_message
                await _emit(
                    emit,
                    MessageUpdateEvent(message=replace(partial_message), assistant_message_event=event),
                )
        elif event.type in ("done", "error"):
            final_message = await response.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
            if not added_partial:
                await _emit(emit, MessageStartEvent(message=replace(final_message)))
            await _emit(emit, MessageEndEvent(message=final_message))
            return final_message

    final_message = await response.result()
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await _emit(emit, MessageStartEvent(message=replace(final_message)))
    await _emit(emit, MessageEndEvent(message=final_message))
    return final_message


@dataclass(slots=True)
class _ExecutedToolCallBatch:
    messages: list[ToolResultMessage]
    terminate: bool


@dataclass(slots=True)
class _PreparedToolCall:
    tool_call: AgentToolCall
    tool: AgentTool
    args: Any


@dataclass(slots=True)
class _ImmediateToolCallOutcome:
    result: AgentToolResult[Any]
    is_error: bool


@dataclass(slots=True)
class _ExecutedToolCallOutcome:
    result: AgentToolResult[Any]
    is_error: bool


@dataclass(slots=True)
class _FinalizedToolCallOutcome:
    tool_call: AgentToolCall
    result: AgentToolResult[Any]
    is_error: bool


async def _fail_tool_calls_from_truncated_message(
    tool_calls: list[AgentToolCall],
    emit: AgentEventSink,
) -> _ExecutedToolCallBatch:
    """Fail all tool calls from an assistant message truncated by the token limit.

    Streamed tool-call arguments are finalized with a best-effort JSON salvage
    parser, so a truncated message can yield tool calls whose arguments parse
    and validate but are silently incomplete. None of them are safe to execute;
    report each as an error so the model can re-issue them.
    """
    messages: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        await _emit(
            emit,
            ToolExecutionStartEvent(tool_call_id=tool_call.id, tool_name=tool_call.name, args=tool_call.arguments),
        )
        finalized = _FinalizedToolCallOutcome(
            tool_call=tool_call,
            result=_create_error_tool_result(
                f'Tool call "{tool_call.name}" was not executed: the response hit the output token limit, '
                "so its arguments may be truncated. Re-issue the tool call with complete arguments."
            ),
            is_error=True,
        )
        await _emit_tool_execution_end(finalized, emit)
        tool_result_message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(tool_result_message, emit)
        messages.append(tool_result_message)
    return _ExecutedToolCallBatch(messages=messages, terminate=False)


async def _execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    cancel: CancelToken | None,
    emit: AgentEventSink,
) -> _ExecutedToolCallBatch:
    tool_calls = [block for block in assistant_message.content if block.type == "toolCall"]
    tools = current_context.tools or []
    has_sequential_tool_call = any(
        next((tool.execution_mode for tool in tools if tool.name == tool_call.name), None) == "sequential"
        for tool_call in tool_calls
    )
    if config.tool_execution == "sequential" or has_sequential_tool_call:
        return await _execute_tool_calls_sequential(
            current_context, assistant_message, tool_calls, config, cancel, emit
        )
    return await _execute_tool_calls_parallel(current_context, assistant_message, tool_calls, config, cancel, emit)


async def _execute_tool_calls_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[AgentToolCall],
    config: AgentLoopConfig,
    cancel: CancelToken | None,
    emit: AgentEventSink,
) -> _ExecutedToolCallBatch:
    finalized_calls: list[_FinalizedToolCallOutcome] = []
    messages: list[ToolResultMessage] = []

    for tool_call in tool_calls:
        await _emit(
            emit,
            ToolExecutionStartEvent(tool_call_id=tool_call.id, tool_name=tool_call.name, args=tool_call.arguments),
        )

        preparation = await _prepare_tool_call(current_context, assistant_message, tool_call, config, cancel)
        if isinstance(preparation, _ImmediateToolCallOutcome):
            finalized = _FinalizedToolCallOutcome(
                tool_call=tool_call, result=preparation.result, is_error=preparation.is_error
            )
        else:
            executed = await _execute_prepared_tool_call(preparation, cancel, emit)
            finalized = await _finalize_executed_tool_call(
                current_context, assistant_message, preparation, executed, config, cancel
            )

        await _emit_tool_execution_end(finalized, emit)
        tool_result_message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(tool_result_message, emit)
        finalized_calls.append(finalized)
        messages.append(tool_result_message)

        if cancel is not None and cancel.cancelled:
            break

    return _ExecutedToolCallBatch(messages=messages, terminate=_should_terminate_tool_batch(finalized_calls))


async def _execute_tool_calls_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[AgentToolCall],
    config: AgentLoopConfig,
    cancel: CancelToken | None,
    emit: AgentEventSink,
) -> _ExecutedToolCallBatch:
    finalized_calls: list[_FinalizedToolCallOutcome | Callable[[], Awaitable[_FinalizedToolCallOutcome]]] = []

    for tool_call in tool_calls:
        await _emit(
            emit,
            ToolExecutionStartEvent(tool_call_id=tool_call.id, tool_name=tool_call.name, args=tool_call.arguments),
        )

        preparation = await _prepare_tool_call(current_context, assistant_message, tool_call, config, cancel)
        if isinstance(preparation, _ImmediateToolCallOutcome):
            finalized = _FinalizedToolCallOutcome(
                tool_call=tool_call, result=preparation.result, is_error=preparation.is_error
            )
            await _emit_tool_execution_end(finalized, emit)
            finalized_calls.append(finalized)
            if cancel is not None and cancel.cancelled:
                break
            continue

        def make_entry(prepared: _PreparedToolCall) -> Callable[[], Awaitable[_FinalizedToolCallOutcome]]:
            async def run() -> _FinalizedToolCallOutcome:
                executed = await _execute_prepared_tool_call(prepared, cancel, emit)
                finalized = await _finalize_executed_tool_call(
                    current_context, assistant_message, prepared, executed, config, cancel
                )
                await _emit_tool_execution_end(finalized, emit)
                return finalized

            return run

        finalized_calls.append(make_entry(preparation))
        if cancel is not None and cancel.cancelled:
            break

    async def resolve_entry(
        entry: _FinalizedToolCallOutcome | Callable[[], Awaitable[_FinalizedToolCallOutcome]],
    ) -> _FinalizedToolCallOutcome:
        if isinstance(entry, _FinalizedToolCallOutcome):
            return entry
        return await entry()

    # True parallelism: prepared calls execute concurrently as tonio tasks;
    # `tonio.spawn` returns results in submission (source) order.
    coros = [resolve_entry(entry) for entry in finalized_calls]
    if not coros:
        ordered_finalized_calls: list[_FinalizedToolCallOutcome] = []
    elif len(coros) == 1:
        ordered_finalized_calls = [await tonio.spawn(coros[0])]
    else:
        ordered_finalized_calls = list(await tonio.spawn(*coros))

    messages: list[ToolResultMessage] = []
    for finalized in ordered_finalized_calls:
        tool_result_message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(tool_result_message, emit)
        messages.append(tool_result_message)

    return _ExecutedToolCallBatch(messages=messages, terminate=_should_terminate_tool_batch(ordered_finalized_calls))


def _should_terminate_tool_batch(finalized_calls: list[_FinalizedToolCallOutcome]) -> bool:
    return len(finalized_calls) > 0 and all(finalized.result.terminate is True for finalized in finalized_calls)


def _prepare_tool_call_arguments(tool: AgentTool, tool_call: AgentToolCall) -> AgentToolCall:
    if tool.prepare_arguments is None:
        return tool_call
    prepared_arguments = tool.prepare_arguments(tool_call.arguments)
    if prepared_arguments is tool_call.arguments:
        return tool_call
    return replace(tool_call, arguments=prepared_arguments)


async def _prepare_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: AgentToolCall,
    config: AgentLoopConfig,
    cancel: CancelToken | None,
) -> _PreparedToolCall | _ImmediateToolCallOutcome:
    tool = next((entry for entry in current_context.tools or [] if entry.name == tool_call.name), None)
    if tool is None:
        return _ImmediateToolCallOutcome(
            result=_create_error_tool_result(f"Tool {tool_call.name} not found"), is_error=True
        )

    try:
        prepared_tool_call = _prepare_tool_call_arguments(tool, tool_call)
        validated_args = validate_tool_arguments(tool, prepared_tool_call)
        if config.before_tool_call is not None:
            before_result = await _maybe_await(
                config.before_tool_call(
                    BeforeToolCallContext(
                        assistant_message=assistant_message,
                        tool_call=tool_call,
                        args=validated_args,
                        context=current_context,
                    ),
                    cancel,
                )
            )
            if cancel is not None and cancel.cancelled:
                return _ImmediateToolCallOutcome(result=_create_error_tool_result("Operation aborted"), is_error=True)
            if before_result is not None and before_result.block:
                return _ImmediateToolCallOutcome(
                    result=_create_error_tool_result(before_result.reason or "Tool execution was blocked"),
                    is_error=True,
                )
        if cancel is not None and cancel.cancelled:
            return _ImmediateToolCallOutcome(result=_create_error_tool_result("Operation aborted"), is_error=True)
        return _PreparedToolCall(tool_call=tool_call, tool=tool, args=validated_args)
    except Exception as error:
        return _ImmediateToolCallOutcome(result=_create_error_tool_result(str(error)), is_error=True)


async def _execute_prepared_tool_call(
    prepared: _PreparedToolCall,
    cancel: CancelToken | None,
    emit: AgentEventSink,
) -> _ExecutedToolCallOutcome:
    update_results: list[Awaitable[None]] = []
    accepting_updates = True

    def on_update(partial_result: AgentToolResult[Any]) -> None:
        if not accepting_updates:
            return
        sink_result = emit(
            ToolExecutionUpdateEvent(
                tool_call_id=prepared.tool_call.id,
                tool_name=prepared.tool_call.name,
                args=prepared.tool_call.arguments,
                partial_result=partial_result,
            )
        )
        if inspect.isawaitable(sink_result):
            update_results.append(sink_result)

    try:
        result = await prepared.tool.execute(prepared.tool_call.id, prepared.args, cancel, on_update)
        accepting_updates = False
        for update_result in update_results:
            await update_result
        return _ExecutedToolCallOutcome(result=result, is_error=False)
    except Exception as error:
        accepting_updates = False
        for update_result in update_results:
            await update_result
        return _ExecutedToolCallOutcome(result=_create_error_tool_result(str(error)), is_error=True)
    finally:
        accepting_updates = False


async def _finalize_executed_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    executed: _ExecutedToolCallOutcome,
    config: AgentLoopConfig,
    cancel: CancelToken | None,
) -> _FinalizedToolCallOutcome:
    result = executed.result
    is_error = executed.is_error

    if config.after_tool_call is not None:
        try:
            after_result = await _maybe_await(
                config.after_tool_call(
                    AfterToolCallContext(
                        assistant_message=assistant_message,
                        tool_call=prepared.tool_call,
                        args=prepared.args,
                        result=result,
                        is_error=is_error,
                        context=current_context,
                    ),
                    cancel,
                )
            )
            if after_result is not None:
                result = replace(
                    result,
                    content=after_result.content if after_result.content is not None else result.content,
                    details=after_result.details if after_result.details is not None else result.details,
                    usage=after_result.usage if after_result.usage is not None else result.usage,
                    terminate=after_result.terminate if after_result.terminate is not None else result.terminate,
                )
                is_error = after_result.is_error if after_result.is_error is not None else is_error
        except Exception as error:
            result = _create_error_tool_result(str(error))
            is_error = True

    return _FinalizedToolCallOutcome(tool_call=prepared.tool_call, result=result, is_error=is_error)


def _create_error_tool_result(message: str) -> AgentToolResult[Any]:
    return AgentToolResult(content=[TextContent(text=message)], details={})


async def _emit_tool_execution_end(finalized: _FinalizedToolCallOutcome, emit: AgentEventSink) -> None:
    await _emit(
        emit,
        ToolExecutionEndEvent(
            tool_call_id=finalized.tool_call.id,
            tool_name=finalized.tool_call.name,
            result=finalized.result,
            is_error=finalized.is_error,
        ),
    )


def _create_tool_result_message(finalized: _FinalizedToolCallOutcome) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=finalized.tool_call.id,
        tool_name=finalized.tool_call.name,
        # Untyped tools can return results without content; normalize so the
        # null never enters session history or provider payloads.
        content=finalized.result.content if finalized.result.content is not None else [],
        details=finalized.result.details,
        usage=finalized.result.usage,
        added_tool_names=(finalized.result.added_tool_names if finalized.result.added_tool_names else None),
        is_error=finalized.is_error,
        timestamp=int(time.time() * 1000),
    )


async def _emit_tool_result_message(tool_result_message: ToolResultMessage, emit: AgentEventSink) -> None:
    await _emit(emit, MessageStartEvent(message=tool_result_message))
    await _emit(emit, MessageEndEvent(message=tool_result_message))
