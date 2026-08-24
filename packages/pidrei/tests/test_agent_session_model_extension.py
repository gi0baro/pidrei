"""Mirror of pi's suite/agent-session-model-extension.test.ts.

Two shape differences run through the file:

- pi's `setModel`/`setThinkingLevel`/`cycleModel`/`cycleThinkingLevel` take a
  `ModelMutationOptions` object; the port takes a keyword-only `persist`.
- pi's faux response callables take `(context)`; pidrei's take
  `(context, options, state, model)`, so each one absorbs the extra arguments.

pi's harness leaves `bindExtensions` to the caller, so its lifecycle case calls
it explicitly; `create_harness` here already binds at construction, so that
case observes the startup event the harness itself produced.
"""

from dataclasses import replace

import pytest

from pidrei.core.agent_session import ExtensionBindings, ScopedModel
from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import TextContent, Usage, UsageCost

from .harness import create_harness, get_assistant_texts


TWO_REASONING_MODELS = [
    {"id": "faux-1", "name": "One", "reasoning": True},
    {"id": "faux-2", "name": "Two", "reasoning": True},
]


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


def _message_texts(message) -> str:
    content = getattr(message, "content", None)
    if content is None or isinstance(content, str):
        return content or ""
    return "\n".join(part.text for part in content if getattr(part, "type", None) == "text")


@pytest.mark.tonio
async def test_set_model_saves_the_model_to_the_session_and_emits_model_select(harnesses):
    model_events: list[str] = []

    def factory(pi) -> None:
        async def on_model_select(event, _ctx) -> None:
            previous = event.get("previousModel")
            model_events.append(
                f"{previous.id if previous is not None else 'none'}->{event['model'].id}:{event['source']}"
            )

        pi.on("model_select", on_model_select)

    harness = await create_harness(models=TWO_REASONING_MODELS, extension_factories=[factory])
    harnesses.append(harness)
    next_model = harness.get_model("faux-2")

    await harness.session.set_model(next_model)

    assert harness.session.model.id == "faux-2"
    assert model_events == ["faux-1->faux-2:set"]
    assert [
        f"{entry['provider']}/{entry['modelId']}"
        for entry in harness.session_manager.get_entries()
        if entry["type"] == "model_change"
    ] == [f"{next_model.provider}/{next_model.id}"]
    assert harness.settings_manager.get_default_provider() is None
    assert harness.settings_manager.get_default_model() is None


@pytest.mark.tonio
async def test_only_persists_model_and_thinking_defaults_when_requested(harnesses):
    harness = await create_harness(models=TWO_REASONING_MODELS)
    harnesses.append(harness)
    next_model = harness.get_model("faux-2")

    await harness.session.set_model(next_model)
    assert harness.settings_manager.get_default_provider() is None
    assert harness.settings_manager.get_default_model() is None

    await harness.session.set_thinking_level("low")
    assert harness.settings_manager.get_default_thinking_level() is None

    await harness.session.set_model(next_model, persist=True)
    assert harness.settings_manager.get_default_provider() == next_model.provider
    assert harness.settings_manager.get_default_model() == next_model.id

    await harness.session.set_thinking_level("high", persist=True)
    assert harness.settings_manager.get_default_thinking_level() == "high"


@pytest.mark.tonio
async def test_persists_the_requested_default_thinking_level_even_when_the_current_model_clamps_it(harnesses):
    harness = await create_harness(models=[{"id": "faux-1", "reasoning": True}])
    harnesses.append(harness)

    await harness.session.set_thinking_level("max", persist=True)

    assert harness.session.thinking_level == "high"
    assert harness.settings_manager.get_default_thinking_level() == "max"


@pytest.mark.tonio
async def test_cycle_model_and_cycle_thinking_level_are_session_only_by_default(harnesses):
    harness = await create_harness(
        models=TWO_REASONING_MODELS,
        settings={"defaultProvider": "faux", "defaultModel": "faux-1", "defaultThinkingLevel": "low"},
    )
    harnesses.append(harness)

    await harness.session.cycle_model()
    assert harness.session.model.id == "faux-2"
    assert harness.settings_manager.get_default_model() == "faux-1"

    await harness.session.set_thinking_level("off")
    assert await harness.session.cycle_thinking_level() == "minimal"
    assert harness.settings_manager.get_default_thinking_level() == "low"


@pytest.mark.tonio
async def test_applies_per_model_thinking_level_override_on_model_switch(harnesses):
    harness = await create_harness(models=TWO_REASONING_MODELS, settings={"defaultThinkingLevel": "medium"})
    harnesses.append(harness)

    # Set a per-model override for faux-2
    harness.settings_manager.set_model_thinking_level("faux", "faux-2", "low")

    # Session starts on faux-1 with default thinking
    await harness.session.set_thinking_level("high")
    assert harness.session.thinking_level == "high"

    # Switch to faux-2 → per-model override should apply
    await harness.session.set_model(harness.get_model("faux-2"))
    assert harness.session.thinking_level == "low"

    # Switch back to faux-1 → no per-model override, uses global default
    await harness.session.set_model(harness.get_model("faux-1"))
    assert harness.session.thinking_level == "medium"


@pytest.mark.tonio
async def test_falls_back_to_current_session_thinking_level_when_nothing_is_configured(harnesses):
    harness = await create_harness(models=TWO_REASONING_MODELS)
    harnesses.append(harness)

    await harness.session.set_thinking_level("high")
    await harness.session.set_model(harness.get_model("faux-2"))
    assert harness.session.thinking_level == "high"


@pytest.mark.tonio
async def test_per_model_override_takes_priority_over_global_default_during_model_switch(harnesses):
    harness = await create_harness(
        models=TWO_REASONING_MODELS,
        settings={"defaultThinkingLevel": "high", "modelThinkingLevels": {"faux/faux-2": "minimal"}},
    )
    harnesses.append(harness)

    await harness.session.set_model(harness.get_model("faux-2"))
    assert harness.session.thinking_level == "minimal"


@pytest.mark.tonio
async def test_cycles_through_scoped_models_and_preserves_the_scoped_thinking_preference(harnesses):
    harness = await create_harness(
        models=[
            {"id": "faux-1", "name": "One", "reasoning": True},
            {"id": "faux-2", "name": "Two", "reasoning": False},
        ]
    )
    harnesses.append(harness)
    harness.session.set_scoped_models(
        [
            ScopedModel(model=harness.get_model("faux-1"), thinking_level="high"),
            ScopedModel(model=harness.get_model("faux-2")),
        ]
    )
    await harness.session.set_thinking_level("high")

    await harness.session.cycle_model()
    assert harness.session.model.id == "faux-2"
    assert harness.session.thinking_level == "off"

    await harness.session.cycle_model()
    assert harness.session.model.id == "faux-1"
    assert harness.session.thinking_level == "high"


@pytest.mark.tonio
async def test_clamps_thinking_levels_to_model_capabilities_and_cycles_available_levels(harnesses):
    harness = await create_harness(models=[{"id": "faux-1", "reasoning": False}])
    harnesses.append(harness)

    await harness.session.set_thinking_level("high")
    assert harness.session.thinking_level == "off"
    assert await harness.session.cycle_thinking_level() is None


@pytest.mark.tonio
async def test_cycles_xhigh_before_max_when_both_are_supported(harnesses):
    harness = await create_harness(models=[{"id": "faux-1", "reasoning": True}])
    harnesses.append(harness)
    harness.session.model.thinking_level_map = {"xhigh": "xhigh", "max": "max"}

    assert harness.session.get_available_thinking_levels() == [
        "off",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    await harness.session.set_thinking_level("high")
    assert await harness.session.cycle_thinking_level() == "xhigh"
    assert await harness.session.cycle_thinking_level() == "max"
    assert await harness.session.cycle_thinking_level() == "off"


@pytest.mark.tonio
async def test_throws_when_set_model_is_called_without_configured_auth(harnesses):
    harness = await create_harness(models=TWO_REASONING_MODELS, with_configured_auth=False)
    harnesses.append(harness)

    with pytest.raises(Exception, match=f"No API key for {harness.get_model().provider}/faux-2"):
        await harness.session.set_model(harness.get_model("faux-2"))


@pytest.mark.tonio
async def test_allows_extension_tool_call_handlers_to_block_tool_execution(harnesses):
    async def execute(*_args):
        raise Exception("tool should have been blocked")

    echo_tool = ToolDefinition(
        name="echo",
        label="Echo",
        description="Echo text back",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        execute=execute,
    )

    def factory(pi) -> None:
        async def block(_event, _ctx):
            return {"block": True, "reason": "Blocked by test"}

        pi.on("tool_call", block)

    harness = await create_harness(tools=[echo_tool], extension_factories=[factory])
    harnesses.append(harness)

    async def echo_error(context, *_rest):
        tool_result = next((m for m in context.messages if getattr(m, "role", None) == "toolResult"), None)
        return faux_assistant_message(_message_texts(tool_result) if tool_result is not None else "")

    harness.set_responses(
        [
            faux_assistant_message([faux_tool_call("echo", {"text": "hello"})], stop_reason="toolUse"),
            echo_error,
        ]
    )

    await harness.session.prompt("hi")

    assert "Blocked by test" in get_assistant_texts(harness)
    assert any(getattr(m, "role", None) == "toolResult" and m.is_error for m in harness.session.messages)


@pytest.mark.tonio
async def test_allows_extension_tool_result_handlers_to_modify_tool_results(harnesses):
    tool_usage = Usage(
        input=1,
        output=2,
        cache_read=3,
        cache_write=4,
        total_tokens=10,
        cost=UsageCost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1),
    )
    patched_tool_usage = Usage(
        input=5,
        output=6,
        cache_read=7,
        cache_write=8,
        total_tokens=26,
        cost=UsageCost(input=0.5, output=0.6, cache_read=0.7, cache_write=0.8, total=2.6),
    )
    observed: dict = {}

    async def execute(_tool_call_id, params, *_rest):
        text = str(params["text"]) if isinstance(params, dict) and "text" in params else ""
        return AgentToolResult(content=[TextContent(text=text)], details={"text": text}, usage=tool_usage)

    echo_tool = ToolDefinition(
        name="echo",
        label="Echo",
        description="Echo text back",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        execute=execute,
    )

    def factory(pi) -> None:
        async def on_tool_result(event, _ctx):
            observed["usage"] = event.get("usage")
            return {
                "content": [TextContent(text="patched result")],
                "details": {"patched": True},
                "usage": patched_tool_usage,
            }

        pi.on("tool_result", on_tool_result)

    harness = await create_harness(tools=[echo_tool], extension_factories=[factory])
    harnesses.append(harness)

    async def echo_back(context, *_rest):
        tool_result = next((m for m in context.messages if getattr(m, "role", None) == "toolResult"), None)
        return faux_assistant_message(_message_texts(tool_result) if tool_result is not None else "")

    harness.set_responses(
        [
            faux_assistant_message([faux_tool_call("echo", {"text": "hello"})], stop_reason="toolUse"),
            echo_back,
        ]
    )

    await harness.session.prompt("hi")

    assert "patched result" in get_assistant_texts(harness)
    tool_result = next(
        (
            m
            for m in harness.session.messages
            if getattr(m, "role", None) == "toolResult" and (m.details or {}).get("patched") is True
        ),
        None,
    )
    assert observed["usage"] == tool_usage
    assert tool_result is not None
    assert tool_result.usage == patched_tool_usage


@pytest.mark.tonio
async def test_allows_extension_context_handlers_to_modify_messages_before_the_llm_call(harnesses):
    def factory(pi) -> None:
        async def on_context(event, _ctx):
            return {
                "messages": [
                    replace(message, content=[TextContent(text="rewritten")])
                    if getattr(message, "role", None) == "user"
                    else message
                    for message in event["messages"]
                ]
            }

        pi.on("context", on_context)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    seen: dict = {"user": ""}

    async def capture(context, *_rest):
        user = next((m for m in context.messages if getattr(m, "role", None) == "user"), None)
        seen["user"] = _message_texts(user) if user is not None else ""
        return faux_assistant_message("done")

    harness.set_responses([capture])

    await harness.session.prompt("original")

    assert seen["user"] == "rewritten"
    stored = next((m for m in harness.session.messages if getattr(m, "role", None) == "user"), None)
    assert stored is not None
    assert stored.content == [TextContent(text="original")]


@pytest.mark.tonio
async def test_allows_extension_input_handlers_to_transform_or_handle_input(harnesses):
    seen_api: dict = {}

    def factory(pi) -> None:
        seen_api["pi"] = pi

        async def on_input(event, _ctx):
            if event["text"] == "ping":
                return {"action": "handled"}
            return {"action": "transform", "text": f"transformed:{event['text']}"}

        pi.on("input", on_input)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    seen: dict = {"user": ""}

    async def capture(context, *_rest):
        user = next((m for m in context.messages if getattr(m, "role", None) == "user"), None)
        seen["user"] = _message_texts(user) if user is not None else ""
        return faux_assistant_message("done")

    harness.set_responses([capture])

    await harness.session.prompt("hello")
    await harness.session.prompt("ping")

    assert seen["user"] == "transformed:hello"
    assert len([m for m in harness.session.messages if getattr(m, "role", None) == "user"]) == 1
    assert seen_api["pi"] is not None


@pytest.mark.tonio
async def test_allows_extension_commands_to_inspect_live_system_prompt_options(harnesses):
    seen_options: list = []

    def factory(pi) -> None:
        async def handler(_args, ctx) -> None:
            options = ctx.get_system_prompt_options()
            seen_options.append(options)
            if options.selected_tools is not None:
                options.selected_tools.append("mutated_tool")

        pi.register_command("inspect-options", handler=handler, description="Inspect system prompt options")

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)

    await harness.session.prompt("/inspect-options")
    await harness.session.prompt("/inspect-options")

    assert len(seen_options) == 2
    assert seen_options[0] is seen_options[1]
    assert seen_options[0].cwd == harness.temp_dir
    assert "read" in seen_options[0].selected_tools
    assert "mutated_tool" in seen_options[1].selected_tools


@pytest.mark.tonio
async def test_allows_before_agent_start_handlers_to_inject_messages_and_modify_the_system_prompt(harnesses):
    def factory(pi) -> None:
        async def on_before_agent_start(event, _ctx):
            return {
                "message": {
                    "customType": "before-start",
                    "content": "injected",
                    "display": True,
                    "details": {"injected": True},
                },
                "systemPrompt": f"{event['systemPrompt']}\n\nextra instructions",
            }

        pi.on("before_agent_start", on_before_agent_start)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    seen: dict = {"systemPrompt": "", "injected": False}

    async def capture(context, *_rest):
        seen["systemPrompt"] = context.system_prompt or ""
        seen["injected"] = any(
            getattr(message, "role", None) == "user" and "injected" in _message_texts(message)
            for message in context.messages
        )
        return faux_assistant_message("done")

    harness.set_responses([capture])

    await harness.session.prompt("hello")

    assert "extra instructions" in seen["systemPrompt"]
    assert seen["injected"] is True
    assert any(
        getattr(m, "role", None) == "custom" and m.custom_type == "before-start" for m in harness.session.messages
    )


@pytest.mark.tonio
async def test_bind_extensions_emits_session_start_and_reload_emits_shutdown_then_start(harnesses):
    lifecycle_events: list[str] = []

    def factory(pi) -> None:
        async def on_start(event, _ctx) -> None:
            lifecycle_events.append(f"start:{event['reason']}")

        async def on_shutdown(event, _ctx) -> None:
            lifecycle_events.append(f"shutdown:{event['reason']}")

        pi.on("session_start", on_start)
        pi.on("session_shutdown", on_shutdown)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    # `create_harness` already bound the extensions once (pi's harness leaves
    # that to the caller), so drop that startup event and run pi's sequence.
    lifecycle_events.clear()

    await harness.session.bind_extensions(ExtensionBindings(shutdown_handler=lambda: None))
    await harness.session.reload()

    assert lifecycle_events == ["start:startup", "shutdown:reload", "start:reload"]
