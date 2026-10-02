"""Mirror of pi's virtual-models.test.ts and suite/virtual-models.test.ts.

Both upstream files map here. pi's routers may return a route or a promise;
pidrei's are async-only, so every route below is an `async def`.
"""

import shutil
import tempfile
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pidrei.core.auth_storage import AuthStorage
from pidrei.core.compaction import CompactionResult
from pidrei.core.extensions import ExtensionVirtualModel, ToolDefinition
from pidrei.core.model_runtime import ModelRuntime
from pidrei.core.sdk import CreateAgentSessionOptions, create_agent_session
from pidrei.core.session_manager import SessionManager
from pidrei.core.virtual_models import (
    VIRTUAL_MODEL_STATE_ENTRY,
    FailedRoute,
    ModelRoute,
    RoutedResponse,
    VirtualModelDefinition,
    get_branch_selection,
)
from pidrei_agent.types import AgentToolResult
from pidrei_ai.models_store import InMemoryModelsStore
from pidrei_ai.providers.faux import FauxModelDefinition, faux_assistant_message, faux_provider, faux_tool_call
from pidrei_ai.registry import ModelsRefreshOptions, get_supported_thinking_levels
from pidrei_ai.types import Context, SimpleStreamOptions, TextContent, UserMessage
from pidrei_ai.utils import clock

from .agent_session_helpers import create_test_resource_loader
from .harness import create_harness


# --- ModelRuntime -------------------------------------------------------------


async def create_runtime(requests: list | None = None):
    requests = requests if requests is not None else []
    runtime = await ModelRuntime(
        credentials=AuthStorage.in_memory(),
        models_store=InMemoryModelsStore(),
        models_path=None,
        allow_model_network=False,
    )
    faux = faux_provider(
        models=[
            FauxModelDefinition(id="small", context_window=1000, max_tokens=100, input=["text"]),
            FauxModelDefinition(
                id="large", context_window=50_000, max_tokens=5000, input=["text", "image"], reasoning=True
            ),
        ]
    )
    runtime.register_native_provider(faux.provider)

    async def route(request):
        requests.append(request)
        model = runtime.get_model("faux", "large" if request.thinking_level == "high" else "small")
        return ModelRoute(model=model, thinking_level="high")

    definition = VirtualModelDefinition(
        provider="router", id="auto", name="Auto", thinking_levels=["low", "high"], route=route
    )
    runtime.register_virtual_model(definition)
    await runtime.refresh(ModelsRefreshOptions(allow_network=False))
    return SimpleNamespace(
        runtime=runtime, faux=faux, definition=definition, virtual=runtime.get_model("router", "auto")
    )


def assistant_from(model, text: str):
    return replace(faux_assistant_message(text), api=model.api, provider=model.provider, model=model.id)


def user(text: str, timestamp: int):
    return UserMessage(content=text, timestamp=timestamp)


# --- get_branch_selection -----------------------------------------------------


async def select(build, runtime):
    session_manager = SessionManager.in_memory()
    await build(session_manager)
    lookups: list[str] = []

    def get_model(provider, model_id):
        lookups.append(f"{provider}/{model_id}")
        return runtime.get_model(provider, model_id)

    return get_branch_selection(session_manager.get_branch(), get_model), lookups


# #10198: the selection must not cost one catalog lookup per assistant message.
@pytest.mark.tonio
async def test_branch_selection_looks_up_only_the_last_model_change():
    setup = await create_runtime()
    runtime, virtual = setup.runtime, setup.virtual
    small = runtime.get_model("faux", "small")
    large = runtime.get_model("faux", "large")

    async def physical_branch(session_manager):
        await session_manager.append_model_change(small.provider, small.id)
        for _ in range(100):
            await session_manager.append_message(assistant_from(large, "ok"))

    selection, lookups = await select(physical_branch, runtime)
    assert selection == ("faux", "large")
    assert lookups == ["faux/small"]

    async def routed_branch(session_manager):
        await session_manager.append_model_change(small.provider, small.id)
        await session_manager.append_message(assistant_from(small, "ok"))
        await session_manager.append_model_change(virtual.provider, virtual.id)
        for _ in range(100):
            await session_manager.append_message(assistant_from(large, "ok"))

    selection, lookups = await select(routed_branch, runtime)
    assert selection == ("router", "auto")
    assert lookups == ["router/auto"]


@pytest.mark.tonio
async def test_branch_selection_uses_the_last_model_change_without_responses_after_it():
    setup = await create_runtime()
    runtime, virtual = setup.runtime, setup.virtual
    small = runtime.get_model("faux", "small")

    async def build(session_manager):
        await session_manager.append_model_change(virtual.provider, virtual.id)
        await session_manager.append_message(assistant_from(small, "ok"))
        await session_manager.append_model_change(small.provider, small.id)

    selection, lookups = await select(build, runtime)
    assert selection == ("faux", "small")
    assert lookups == []


@pytest.mark.tonio
async def test_branch_selection_uses_the_latest_physical_response_without_a_model_change():
    setup = await create_runtime()
    runtime, virtual = setup.runtime, setup.virtual
    small = runtime.get_model("faux", "small")
    large = runtime.get_model("faux", "large")

    async def build(session_manager):
        await session_manager.append_message(assistant_from(small, "ok"))
        await session_manager.append_message(assistant_from(large, "ok"))
        # Failed routing leaves the virtual model on its message.
        await session_manager.append_message(replace(assistant_from(virtual, ""), stop_reason="error"))

    selection, lookups = await select(build, runtime)
    assert selection == ("faux", "large")
    assert lookups == []


@pytest.mark.tonio
async def test_lists_a_virtual_model_and_routes_it_to_a_physical_model_with_a_clamped_thinking_level():
    requests: list = []
    setup = await create_runtime(requests)
    runtime, virtual = setup.runtime, setup.virtual
    assert (virtual.provider, virtual.id, virtual.context_window, virtual.max_tokens) == ("router", "auto", 0, 0)
    assert virtual.input == ["text", "image"]
    assert get_supported_thinking_levels(virtual) == ["low", "high"]
    large = runtime.get_model("faux", "large")
    messages = [user("first", 1), replace(assistant_from(large, "answer"), thinking_level="medium"), user("second", 2)]

    low = await runtime.resolve_model(virtual, messages, reason="user", thinking_level="low")
    assert low.model.id == "small"
    # The router asked for "high", but the small model does not reason.
    assert low.thinking_level == "off"
    assert requests[0].previous == RoutedResponse(model=large, thinking_level="medium")

    high = await runtime.resolve_model(virtual, messages, reason="user", thinking_level="high")
    assert high == ModelRoute(model=large, thinking_level="high")


@pytest.mark.tonio
async def test_reports_the_failed_request_of_a_retry_separately_from_the_latest_successful_response():
    requests: list = []
    setup = await create_runtime(requests)
    runtime, virtual = setup.runtime, setup.virtual
    small = runtime.get_model("faux", "small")
    large = runtime.get_model("faux", "large")
    failed = replace(
        assistant_from(large, ""), thinking_level="high", stop_reason="error", error_message="overloaded_error"
    )
    messages = [user("first", 1), assistant_from(small, "answer"), user("second", 2)]

    await runtime.resolve_model(virtual, messages, reason="retry", thinking_level="low", failed=failed)
    # A routing failure names the virtual model, so there is no failed physical request to report.
    failed_route = replace(assistant_from(virtual, ""), stop_reason="error")
    await runtime.resolve_model(virtual, messages, reason="retry", thinking_level="low", failed=failed_route)

    assert requests[0].previous.model is small
    assert requests[0].failed == FailedRoute(model=large, thinking_level="high", message=failed)
    assert requests[1].failed is None


@pytest.mark.tonio
async def test_lists_several_virtual_models_under_a_provider_with_physical_models():
    setup = await create_runtime()
    runtime, definition = setup.runtime, setup.definition

    async def route(_request):
        return ModelRoute(model=runtime.get_model("faux", "small"), thinking_level="off")

    runtime.register_virtual_model(replace(definition, provider="faux", id="auto", route=route))
    runtime.register_virtual_model(replace(definition, provider="faux", id="fast", name="Fast", route=route))
    runtime.register_virtual_model(replace(definition, id="second", name="Second"))

    assert [model.id for model in runtime.get_models("faux")] == ["small", "large", "auto", "fast"]
    assert [model.id for model in runtime.get_models("router")] == ["auto", "second"]
    # Virtual models on a physical provider are available when the provider is.
    available = await runtime.get_available()
    assert [model.id for model in available if model.provider == "faux"] == ["small", "large", "auto", "fast"]
    fast = runtime.get_model("faux", "fast")
    routed = await runtime.resolve_model(fast, [], reason="user", thinking_level="off")
    assert (routed.model.provider, routed.model.id) == ("faux", "small")

    with pytest.raises(Exception, match="conflicts with a physical model"):
        runtime.register_virtual_model(replace(definition, provider="faux", id="large"))
    runtime.unregister_virtual_model("faux", "fast")
    runtime.unregister_virtual_model("router", "auto")
    assert [model.id for model in runtime.get_models("faux")] == ["small", "large", "auto"]
    assert [model.id for model in runtime.get_models("router")] == ["second"]


@pytest.mark.tonio
async def test_rejects_routes_to_virtual_or_unknown_models():
    setup = await create_runtime()
    runtime, definition, virtual = setup.runtime, setup.definition, setup.virtual
    unknown = replace(virtual, provider="faux", id="missing")

    for model in (virtual, unknown):

        async def route(_request, model=model):
            return ModelRoute(model=model, thinking_level="off")

        runtime.register_virtual_model(replace(definition, route=route))
        with pytest.raises(Exception, match="which is not a physical model"):
            await runtime.resolve_model(virtual, [], reason="user", thinking_level="low")


@pytest.mark.tonio
async def test_routes_direct_stream_simple_calls_within_the_routed_models_limits():
    requests: list = []
    setup = await create_runtime(requests)
    runtime, faux, virtual = setup.runtime, setup.faux, setup.virtual
    max_tokens: list = []

    async def respond(_context, options, _state, _model):
        max_tokens.append(options.max_tokens if options is not None else None)
        return faux_assistant_message("hello")

    faux.set_responses([respond])

    # The caller sized the request without knowing the routed model.
    message = await runtime.complete_simple(
        virtual, Context(messages=[user("hi", 1)]), SimpleStreamOptions(reasoning="high", max_tokens=20_000)
    )

    assert [request.reason for request in requests] == ["direct"]
    assert (message.provider, message.model, message.stop_reason) == ("faux", "large", "stop")
    assert max_tokens == [5000]


@pytest.mark.tonio
async def test_does_not_forward_caller_credentials_to_a_routed_model_of_another_provider():
    setup = await create_runtime()
    runtime, faux, definition, virtual = setup.runtime, setup.faux, setup.definition, setup.virtual
    seen: list = []

    async def respond(_context, options, _state, _model):
        seen.append((options.api_key, options.headers))
        return faux_assistant_message("hello")

    faux.set_responses([respond, respond])
    context = Context(messages=[user("hi", 1)])
    options = SimpleStreamOptions(api_key="caller-key", headers={"x-caller": "1"})
    runtime.register_virtual_model(replace(definition, provider="faux", id="auto"))

    await runtime.complete_simple(virtual, context, options)
    await runtime.complete_simple(runtime.get_model("faux", "auto"), context, options)

    # The router provider's credentials stay with it; the faux provider's own virtual model keeps them.
    assert seen[0][0] != "caller-key"
    assert "x-caller" not in (seen[0][1] or {})
    assert seen[1][0] == "caller-key"
    assert seen[1][1]["x-caller"] == "1"


@pytest.mark.tonio
async def test_fails_unrouted_stream_calls_on_virtual_models():
    setup = await create_runtime()

    message = await setup.runtime.complete(setup.virtual, Context(messages=[user("hi", 1)]))

    assert message.stop_reason == "error"
    assert "must be routed before streaming" in message.error_message


# --- create_agent_session -----------------------------------------------------


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp(prefix="pidrei-virtual-models-")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def sessions(request):
    opened: list = []
    request.addfinalizer(lambda: [session.dispose() for session in opened])
    return opened


async def open_session(temp_dir, runtime, session_manager, model=None):
    return await create_agent_session(
        CreateAgentSessionOptions(
            cwd=temp_dir,
            agent_dir=temp_dir,
            model_runtime=runtime,
            session_manager=session_manager,
            resource_loader=create_test_resource_loader(),
            model=model,
        )
    )


async def resume(temp_dir, sessions, runtime, model=None):
    """Resume a transcript where the virtual model was selected and the large model answered."""
    session_manager = SessionManager.in_memory(temp_dir)
    await session_manager.append_model_change("router", "auto")
    await session_manager.append_message(user("hi", 1))
    await session_manager.append_message(assistant_from(runtime.get_model("faux", "large"), "hello"))
    session = (await open_session(temp_dir, runtime, session_manager, model)).session
    sessions.append(session)
    return session, session_manager


def model_changes(session_manager) -> list:
    return [entry for entry in session_manager.get_branch() if entry["type"] == "model_change"]


@pytest.mark.tonio
async def test_restores_the_virtual_selection_instead_of_the_physical_model_that_answered(temp_dir, sessions):
    setup = await create_runtime()

    session, _ = await resume(temp_dir, sessions, setup.runtime)

    assert (session.model.provider, session.model.id) == ("router", "auto")
    assert (session.routed_model.model.provider, session.routed_model.model.id) == ("faux", "large")


@pytest.mark.tonio
async def test_restores_a_virtual_selection_registered_right_before_the_session_opens(temp_dir, sessions):
    setup = await create_runtime()
    runtime = setup.runtime
    # Extensions register while the session is created, without waiting for the availability refresh.
    runtime.register_virtual_model(replace(setup.definition, provider="late"))
    session_manager = SessionManager.in_memory(temp_dir)
    await session_manager.append_model_change("late", "auto")
    await session_manager.append_message(user("hi", 1))
    await session_manager.append_message(assistant_from(runtime.get_model("faux", "large"), "hello"))

    result = await open_session(temp_dir, runtime, session_manager)
    sessions.append(result.session)

    assert result.model_fallback_message is None
    assert (result.session.model.provider, result.session.model.id) == ("late", "auto")


@pytest.mark.tonio
async def test_falls_back_to_the_physical_model_when_the_virtual_model_is_not_registered(temp_dir, sessions):
    setup = await create_runtime()
    setup.runtime.unregister_virtual_model("router", "auto")

    session, _ = await resume(temp_dir, sessions, setup.runtime)

    assert (session.model.provider, session.model.id) == ("faux", "large")
    assert session.routed_model is None


@pytest.mark.tonio
async def test_falls_back_to_the_last_physical_response_when_the_transcript_ends_with_a_routing_failure(
    temp_dir, sessions
):
    setup = await create_runtime()
    runtime = setup.runtime
    session_manager = SessionManager.in_memory(temp_dir)
    await session_manager.append_model_change("router", "auto")
    await session_manager.append_message(user("hi", 1))
    await session_manager.append_message(assistant_from(runtime.get_model("faux", "large"), "hello"))
    await session_manager.append_message(user("again", 2))
    await session_manager.append_message(
        replace(assistant_from(setup.virtual, ""), stop_reason="error", error_message="router failed")
    )
    runtime.unregister_virtual_model("router", "auto")

    result = await open_session(temp_dir, runtime, session_manager)
    sessions.append(result.session)

    assert (result.session.model.provider, result.session.model.id) == ("faux", "large")
    assert result.model_fallback_message is None


@pytest.mark.tonio
async def test_resumes_the_selection_made_before_tree_navigation_left_its_model_change_on_another_branch(
    temp_dir, sessions
):
    setup = await create_runtime()
    runtime = setup.runtime
    setup.faux.set_responses([faux_assistant_message("ok") for _ in range(6)])
    large = runtime.get_model("faux", "large")

    for before, after in ((setup.virtual, large), (large, setup.virtual)):
        session_manager = SessionManager.in_memory(temp_dir)
        session = (await open_session(temp_dir, runtime, session_manager, before)).session
        await session.prompt("one")
        first_answer = session_manager.get_leaf_id()
        await session.set_model(after)
        await session.prompt("two")
        # Navigating back to before the switch keeps `after` selected, but its model_change is on the old branch.
        await session.navigate_tree(first_answer)
        await session.prompt("three")
        session.dispose()

        resumed = (await open_session(temp_dir, runtime, session_manager)).session
        sessions.append(resumed)
        assert (resumed.model.provider, resumed.model.id) == (after.provider, after.id)


@pytest.mark.tonio
async def test_does_not_record_a_physical_selection_on_every_prompt_while_requests_are_redirected(temp_dir, sessions):
    setup = await create_runtime()
    runtime = setup.runtime
    setup.faux.set_responses([faux_assistant_message("ok"), faux_assistant_message("ok")])
    small = runtime.get_model("faux", "small")
    session_manager = SessionManager.in_memory(temp_dir)
    session = (await open_session(temp_dir, runtime, session_manager, runtime.get_model("faux", "large"))).session
    sessions.append(session)
    prepare_request = session.agent.prepare_request

    async def redirected(request, cancel=None):
        return replace(await prepare_request(request, cancel), model=small)

    session.agent.prepare_request = redirected
    initial = len(model_changes(session_manager))

    await session.prompt("one")
    await session.prompt("two")

    assert [message.model for message in session.messages if message.role == "assistant"] == ["small", "small"]
    assert len(model_changes(session_manager)) == initial


@pytest.mark.tonio
async def test_records_an_explicit_model_override_on_resume_with_the_next_prompt(temp_dir, sessions):
    setup = await create_runtime()
    setup.faux.set_responses([faux_assistant_message("ok")])

    session, session_manager = await resume(temp_dir, sessions, setup.runtime, setup.runtime.get_model("faux", "small"))
    # Opening the session does not write to it.
    last = model_changes(session_manager)[-1]
    assert (last["provider"], last["modelId"]) == ("router", "auto")

    await session.prompt("again")
    last = model_changes(session_manager)[-1]
    assert (last["provider"], last["modelId"]) == ("faux", "small")


# --- AgentSession (suite) -----------------------------------------------------


async def _echo(*_args):
    return AgentToolResult(content=[TextContent(text="echoed")], details={})


ECHO_TOOL = ToolDefinition(
    name="echo",
    label="Echo",
    description="Echo text back",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    execute=_echo,
)


async def default_route(request, ctx):
    """Router used by these tests: new user turns pick by thinking level, everything else stays put."""

    def find(model_id: str):
        return ctx.model_registry.find("faux", model_id)

    if request.reason == "direct":
        return ModelRoute(model=find("large"), thinking_level="low")
    sticky = request.failed or request.previous
    if request.reason != "user" and sticky is not None:
        return ModelRoute(model=sticky.model, thinking_level=sticky.thinking_level or "high")
    if request.thinking_level == "high":
        return ModelRoute(model=find("large"), thinking_level="high")
    return ModelRoute(model=find("small"), thinking_level="off")


async def _compaction_factory(pi, summary: str = "compacted") -> None:
    async def on_before_compact(event, _ctx):
        preparation = event["preparation"]
        return {
            "compaction": CompactionResult(
                summary=summary,
                first_kept_entry_id=preparation.first_kept_entry_id,
                tokens_before=preparation.tokens_before,
            )
        }

    pi.on("session_before_compact", on_before_compact)


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


async def create_routed_harness(harnesses, route=default_route, *, settings=None, extension_factories=None):
    requests: list = []

    async def factory(pi) -> None:
        async def recording_route(request, ctx):
            requests.append(request)
            return await route(request, ctx)

        pi.register_virtual_model(
            ExtensionVirtualModel(
                provider="router",
                id="auto",
                name="Auto",
                thinking_levels=["low", "high"],
                context_window=1000,
                route=recording_route,
            )
        )

    harness = await create_harness(
        settings=settings,
        models=[
            {"id": "small", "context_window": 1000},
            {"id": "large", "context_window": 50_000, "max_tokens": 4000, "reasoning": True},
        ],
        tools=[ECHO_TOOL],
        extension_factories=[*(extension_factories or []), factory],
    )
    harnesses.append(harness)
    # The harness streams through the runtime, which records the thinking level on responses.
    runtime = harness.session.model_runtime
    await harness.session.set_model(runtime.get_model("router", "auto"))
    await harness.session.set_thinking_level("high")

    def reasons() -> list[str]:
        return [request.reason for request in requests]

    def dispatched() -> list[str]:
        """Physical model and thinking level recorded on each response."""
        return [
            f"{message.provider}/{message.model}:{message.thinking_level}"
            for message in harness.session.messages
            if message.role == "assistant"
        ]

    return SimpleNamespace(harness=harness, requests=requests, reasons=reasons, dispatched=dispatched)


@pytest.mark.tonio
async def test_routes_each_request_including_retries_while_the_selection_stays_virtual(harnesses):
    routed = await create_routed_harness(
        harnesses, settings={"retry": {"enabled": True, "maxRetries": 3, "baseDelayMs": 1}}
    )
    harness = routed.harness
    harness.set_responses(
        [
            faux_assistant_message("", stop_reason="error", error_message="overloaded_error"),
            faux_assistant_message(faux_tool_call("echo", {"text": "hi"}), stop_reason="toolUse"),
            faux_assistant_message("done"),
        ]
    )

    await harness.session.prompt("hello")

    assert routed.reasons() == ["user", "retry", "continuation"]
    assert routed.requests[1].failed.message.error_message == "overloaded_error"
    assert routed.requests[2].previous.model.id == "large"
    assert routed.dispatched() == ["faux/large:high", "faux/large:high"]
    assert (harness.session.model.provider, harness.session.model.id) == ("router", "auto")
    assert harness.session.thinking_level == "high"
    # Limits come from the physical model that produced the latest response, not the virtual model.
    assert harness.session.get_context_usage().context_window == 50_000


@pytest.mark.tonio
async def test_retries_the_first_request_of_a_turn_on_the_model_routed_for_that_turn(harnesses):
    routed = await create_routed_harness(
        harnesses, settings={"retry": {"enabled": True, "maxRetries": 3, "baseDelayMs": 1}}
    )
    harness = routed.harness
    await harness.session.set_thinking_level("low")
    harness.set_responses(
        [
            faux_assistant_message("easy answer"),
            faux_assistant_message("", stop_reason="error", error_message="overloaded_error"),
            faux_assistant_message("hard answer"),
        ]
    )
    await harness.session.prompt("easy")
    await harness.session.set_thinking_level("high")

    await harness.session.prompt("hard")

    assert routed.reasons() == ["user", "user", "retry"]
    # The retry reports the failed request on large next to the small response of the previous turn.
    assert routed.requests[2].failed.model.id == "large"
    assert routed.requests[2].previous.model.id == "small"
    assert routed.dispatched() == ["faux/small:off", "faux/large:high"]


@pytest.mark.tonio
async def test_routes_the_compact_and_retry_after_a_truncated_response_as_a_retry(harnesses):
    routed = await create_routed_harness(
        harnesses,
        settings={"compaction": {"keepRecentTokens": 1, "reserveTokens": 0}},
        extension_factories=[lambda pi: _compaction_factory(pi, "overflow compacted")],
    )
    harness = routed.harness

    async def truncated(*_args):
        return faux_assistant_message("x" * 64, stop_reason="length", timestamp=clock.now_ms() + 10_000)

    harness.set_responses([truncated, faux_assistant_message("done")])

    await harness.session.prompt("x" * 5000)

    assert [event.reason for event in harness.events_of_type("compaction_start")] == ["overflow"]
    # Compaction may fold the prompt into the summary, so the retry is not a new user turn.
    assert routed.reasons() == ["user", "retry"]
    assert routed.requests[1].failed.model.id == "large"
    assert routed.requests[1].failed.message.stop_reason == "length"


@pytest.mark.tonio
async def test_routes_requests_after_extension_messages_as_continuations(harnesses):
    continued = False

    async def factory(pi) -> None:
        async def on_before_settle(_event, _ctx):
            nonlocal continued
            if continued:
                return None
            continued = True
            return {
                "entries": [
                    {"type": "custom_message", "customType": "nudge", "content": "Keep going.", "display": False}
                ],
                "continue": True,
            }

        pi.on("agent_before_settle", on_before_settle)

    routed = await create_routed_harness(harnesses, extension_factories=[factory])
    routed.harness.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])

    await routed.harness.session.prompt("hello")

    # The hidden custom message becomes a user message for the model, but the user did not write it.
    assert routed.reasons() == ["user", "continuation"]


@pytest.mark.tonio
async def test_routes_the_first_request_of_a_prompt_as_a_user_turn_when_extension_messages_follow_the_prompt(
    harnesses,
):
    async def factory(pi) -> None:
        async def on_before_agent_start(_event, _ctx):
            return {"message": {"customType": "context", "content": "Extra context.", "display": False}}

        pi.on("before_agent_start", on_before_agent_start)

    routed = await create_routed_harness(harnesses, extension_factories=[factory])
    harness = routed.harness
    harness.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])

    await harness.session.prompt("hello")
    await harness.session.prompt("again")

    # The context ends with the extension message, but the request answers the user's prompt.
    assert harness.session.messages[-2].role == "custom"
    assert routed.reasons() == ["user", "user"]


@pytest.mark.tonio
async def test_ends_the_run_with_an_error_response_when_routing_fails_and_keeps_the_last_physical_limits(
    harnesses,
):
    fail = False

    async def route(request, ctx):
        if fail:
            raise Exception("router unavailable")
        return await default_route(request, ctx)

    routed = await create_routed_harness(harnesses, route)
    harness = routed.harness
    harness.set_responses([faux_assistant_message("answer")])
    await harness.session.prompt("hello")

    fail = True
    await harness.session.prompt("again")

    last = harness.session.messages[-1]
    assert (last.role, last.provider, last.model, last.stop_reason) == ("assistant", "router", "auto", "error")
    assert "router unavailable" in last.error_message
    assert harness.faux.state.call_count == 1
    # The failed attempt names the virtual model, whose declared window is 1k; the large model's 50k applies.
    assert harness.session.get_context_usage().context_window == 50_000


@pytest.mark.tonio
async def test_checks_compaction_against_the_physical_model_that_produced_the_response(harnesses):
    routed = await create_routed_harness(harnesses)
    harness = routed.harness
    harness.set_responses([faux_assistant_message("short answer"), faux_assistant_message("long answer")])
    await harness.session.prompt("hello")

    # About 20k tokens exceed the virtual model's declared 1k window but fit the large model's 50k.
    await harness.session.prompt("x" * 80_000)

    assert harness.events_of_type("compaction_start") == []


@pytest.mark.tonio
async def test_compacts_before_a_request_routed_to_a_model_with_a_smaller_window(harnesses):
    routed = await create_routed_harness(
        harnesses,
        settings={"compaction": {"keepRecentTokens": 1, "reserveTokens": 0}},
        extension_factories=[_compaction_factory],
    )
    harness = routed.harness
    compacted_before_small = False

    async def small_answer(*_args):
        nonlocal compacted_before_small
        compacted_before_small = len(harness.events_of_type("compaction_end")) == 1
        return faux_assistant_message("small answer")

    harness.set_responses([faux_assistant_message("y" * 8000), small_answer])
    await harness.session.prompt("hello")
    assert harness.events_of_type("compaction_start") == []

    # About 2k tokens fit the large model that answered last, but not the small model's 1k window.
    await harness.session.set_thinking_level("low")
    await harness.session.prompt("next")

    assert [event.reason for event in harness.events_of_type("compaction_start")] == ["threshold"]
    assert compacted_before_small is True
    assert routed.dispatched()[-1] == "faux/small:off"


@pytest.mark.tonio
async def test_compacts_between_turns_of_a_run_when_the_next_request_is_routed_to_a_smaller_window(harnesses):
    async def route(request, ctx):
        if request.reason == "continuation":
            return ModelRoute(model=ctx.model_registry.find("faux", "small"), thinking_level="off")
        return await default_route(request, ctx)

    routed = await create_routed_harness(
        harnesses,
        route,
        settings={"compaction": {"keepRecentTokens": 1, "reserveTokens": 0}},
        extension_factories=[_compaction_factory],
    )
    harness = routed.harness
    compacted_before_small = False

    async def small_answer(*_args):
        nonlocal compacted_before_small
        compacted_before_small = len(harness.events_of_type("compaction_end")) == 1
        return faux_assistant_message("small answer")

    harness.set_responses(
        [faux_assistant_message(faux_tool_call("echo", {"text": "hi"}), stop_reason="toolUse"), small_answer]
    )

    # About 2k tokens fit the large model of the first turn, but not the small model of the second.
    await harness.session.prompt("x" * 8000)

    assert [event.reason for event in harness.events_of_type("compaction_start")] == ["threshold"]
    assert compacted_before_small is True
    assert routed.dispatched() == ["faux/large:high", "faux/small:off"]


@pytest.mark.tonio
async def test_projects_the_session_once_per_request_under_a_virtual_selection(harnesses, monkeypatch):
    routed = await create_routed_harness(harnesses)
    harness = routed.harness
    harness.set_responses(
        [
            faux_assistant_message(faux_tool_call("echo", {"text": "hi"}), stop_reason="toolUse"),
            faux_assistant_message("done"),
        ]
    )
    build_session_projection = harness.session_manager.build_session_projection
    calls = 0

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return build_session_projection(*args, **kwargs)

    monkeypatch.setattr(harness.session_manager, "build_session_projection", counting)

    await harness.session.prompt("hello")

    # One per request, one between the turns, and one for the compaction check after the run.
    assert calls == 4


@pytest.mark.tonio
async def test_stores_router_state_on_the_branch_and_passes_it_to_later_requests(harnesses):
    states: list = []

    async def route(request, ctx):
        states.append(request.state)
        turns = request.state["turns"] if request.state is not None else 0
        chosen = await default_route(request, ctx)
        # Returning request.state keeps it without storing it again. Direct requests neither get nor store state.
        if request.reason == "continuation":
            return replace(chosen, state=request.state)
        return replace(chosen, state={"turns": turns + 1})

    routed = await create_routed_harness(harnesses, route, settings={"compaction": {"keepRecentTokens": 1}})
    harness = routed.harness
    harness.set_responses(
        [
            faux_assistant_message(faux_tool_call("echo", {"text": "hi"}), stop_reason="toolUse"),
            faux_assistant_message("first"),
            faux_assistant_message("second"),
            faux_assistant_message("summary"),
            faux_assistant_message("summary"),
        ]
    )

    await harness.session.prompt("one")
    await harness.session.prompt("two")

    def stored() -> list:
        return [
            entry.get("data")
            for entry in harness.session_manager.get_branch()
            if entry["type"] == "custom" and entry.get("customType") == VIRTUAL_MODEL_STATE_ENTRY
        ]

    assert stored() == [
        {"provider": "router", "modelId": "auto", "state": {"turns": 1}},
        {"provider": "router", "modelId": "auto", "state": {"turns": 2}},
    ]

    await harness.session.compact()

    assert routed.reasons() == ["user", "continuation", "user", "direct"]
    assert states == [None, {"turns": 1}, {"turns": 1}, None]
    assert len(stored()) == 2


@pytest.mark.tonio
async def test_does_not_route_compactions_that_an_extension_supplies(harnesses):
    async def route(request, ctx):
        if request.reason == "direct":
            raise Exception("router unavailable")
        return await default_route(request, ctx)

    routed = await create_routed_harness(
        harnesses,
        route,
        settings={"compaction": {"keepRecentTokens": 1}},
        extension_factories=[lambda pi: _compaction_factory(pi, "extension summary")],
    )
    harness = routed.harness
    harness.set_responses([faux_assistant_message("first answer"), faux_assistant_message("second answer")])
    await harness.session.prompt("first")
    await harness.session.prompt("second")

    result = await harness.session.compact()

    assert result.summary == "extension summary"
    assert routed.reasons() == ["user", "user"]


@pytest.mark.tonio
async def test_routes_compaction_summaries_before_sizing_them(harnesses):
    summaries: list[str] = []
    routed = await create_routed_harness(harnesses, settings={"compaction": {"keepRecentTokens": 1}})
    harness = routed.harness

    async def summary(_context, options, _state, model):
        reasoning = options.reasoning if options is not None and options.reasoning else "off"
        summaries.append(f"{model.id}:{reasoning}:{options.max_tokens}")
        return faux_assistant_message("summary")

    # Compaction summarizes the history and the split turn prefix with one routed model.
    harness.set_responses(
        [faux_assistant_message("first answer"), faux_assistant_message("second answer"), summary, summary]
    )
    await harness.session.prompt("first")
    await harness.session.prompt("second")

    result = await harness.session.compact()

    assert "summary" in result.summary
    assert routed.reasons() == ["user", "user", "direct"]
    # The router's thinking level applies, and the output budget respects the large model's 4000 tokens.
    assert summaries == ["large:low:4000", "large:low:4000"]
