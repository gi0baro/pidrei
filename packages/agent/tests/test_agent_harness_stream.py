"""Mirror of pi agent/test/harness/agent-harness-stream.test.ts."""

from dataclasses import replace

import pytest

from pidrei_agent.harness.agent_harness import AgentHarness, AgentHarnessOptions
from pidrei_agent.harness.session.memory_storage import InMemorySessionStorage
from pidrei_agent.harness.session.session import Session
from pidrei_agent.harness.types import (
    AgentHarnessStreamOptions,
    BeforeProviderPayloadResult,
    BeforeProviderRequestResult,
    SessionMetadata,
)
from pidrei_ai.providers.faux import faux_assistant_message, faux_provider, faux_tool_call
from pidrei_ai.registry import create_models
from tests.harness_tool_fixtures import calculate_tool


# Shared collection; each faux provider gets a unique id so coexisting fakes route correctly.
models = create_models()
_faux_count = 0


def new_faux():
    global _faux_count
    _faux_count += 1
    faux = faux_provider(provider=f"faux-stream-{_faux_count}")
    models.set_provider(faux.provider)
    return faux


def capture_options(options):
    return replace(
        options,
        headers=dict(options.headers) if options.headers is not None else None,
        metadata=dict(options.metadata) if options.metadata is not None else None,
    )


@pytest.mark.tonio
async def test_snapshots_stream_options_before_provider_request_hooks():
    captured_options = None
    registration = new_faux()

    def respond(_context, options, _state, _model):
        nonlocal captured_options
        captured_options = options
        return faux_assistant_message("ok")

    registration.set_responses([respond])

    session = Session(InMemorySessionStorage(metadata=SessionMetadata(id="session-1", created_at="now")))
    harness = AgentHarness(
        AgentHarnessOptions(
            models=models,
            session=session,
            model=registration.get_model(),
            stream_options=AgentHarnessStreamOptions(
                timeout_ms=1000,
                max_retries=2,
                max_retry_delay_ms=3000,
                headers={"x-base": "base"},
                metadata={"base": True},
                cache_retention="none",
            ),
        )
    )

    def hook(event):
        assert event.session_id == "session-1"
        assert event.stream_options.headers == {"x-base": "base"}
        return BeforeProviderRequestResult(stream_options={"headers": {"x-hook": "hook"}, "metadata": {"hook": True}})

    harness.on("before_provider_request", hook)

    await harness.prompt("hello")

    assert captured_options is not None
    assert captured_options.timeout_ms == 1000
    assert captured_options.max_retries == 2
    assert captured_options.max_retry_delay_ms == 3000
    assert captured_options.session_id == "session-1"
    assert captured_options.cache_retention == "none"
    assert captured_options.headers == {"x-base": "base", "x-hook": "hook"}
    assert captured_options.metadata == {"base": True, "hook": True}


@pytest.mark.tonio
async def test_chains_provider_request_patches_and_supports_deletion_semantics():
    captured_options = None
    registration = new_faux()

    def respond(_context, options, _state, _model):
        nonlocal captured_options
        captured_options = options
        return faux_assistant_message("ok")

    registration.set_responses([respond])

    harness = AgentHarness(
        AgentHarnessOptions(
            models=models,
            session=Session(InMemorySessionStorage()),
            model=registration.get_model(),
            stream_options=AgentHarnessStreamOptions(
                timeout_ms=1000,
                max_retries=2,
                headers={"keep": "base", "remove": "base"},
                metadata={"keep": "base", "remove": "base"},
            ),
        )
    )

    def first_hook(event):
        assert event.stream_options.headers == {"keep": "base", "remove": "base"}
        return BeforeProviderRequestResult(
            stream_options={
                "headers": {"first": "1", "remove": None},
                "metadata": {"first": 1, "remove": None},
            }
        )

    def second_hook(event):
        assert event.stream_options.headers == {"keep": "base", "first": "1"}
        assert event.stream_options.metadata == {"keep": "base", "first": 1}
        return BeforeProviderRequestResult(
            stream_options={"timeout_ms": None, "headers": {"second": "2"}, "metadata": None}
        )

    harness.on("before_provider_request", first_hook)
    harness.on("before_provider_request", second_hook)

    await harness.prompt("hello")

    assert captured_options is not None
    assert captured_options.timeout_ms is None
    assert captured_options.max_retries == 2
    assert captured_options.headers == {"keep": "base", "first": "1", "second": "2"}
    assert captured_options.metadata is None


@pytest.mark.tonio
async def test_uses_updated_stream_options_for_save_point_snapshots_without_mutating_the_active_request():
    captured_options = []
    registration = new_faux()

    def first_respond(_context, options, _state, _model):
        captured_options.append(capture_options(options))
        return faux_assistant_message(
            faux_tool_call("calculate", {"expression": "1 + 1"}, id="call-1"), stop_reason="toolUse"
        )

    def second_respond(_context, options, _state, _model):
        captured_options.append(capture_options(options))
        return faux_assistant_message("done")

    registration.set_responses([first_respond, second_respond])

    harness = AgentHarness(
        AgentHarnessOptions(
            models=models,
            session=Session(InMemorySessionStorage()),
            model=registration.get_model(),
            tools=[calculate_tool],
            stream_options=AgentHarnessStreamOptions(timeout_ms=1000, headers={"turn": "first"}),
        )
    )

    async def listener(event, _signal):
        if event.type == "tool_execution_start":
            await harness.set_stream_options(AgentHarnessStreamOptions(timeout_ms=2000, headers={"turn": "second"}))

    harness.subscribe(listener)

    await harness.prompt("hello")

    assert len(captured_options) == 2
    assert captured_options[0].timeout_ms == 1000
    assert captured_options[0].headers == {"turn": "first"}
    assert captured_options[1].timeout_ms == 2000
    assert captured_options[1].headers == {"turn": "second"}


@pytest.mark.tonio
async def test_chains_provider_payload_hooks():
    seen_payloads = []
    final_payload = None
    registration = new_faux()

    async def respond(_context, options, _state, model):
        nonlocal final_payload
        final_payload = await options.on_payload({"steps": ["provider"]}, model)
        return faux_assistant_message("ok")

    registration.set_responses([respond])

    harness = AgentHarness(
        AgentHarnessOptions(models=models, session=Session(InMemorySessionStorage()), model=registration.get_model())
    )

    def first_hook(event):
        seen_payloads.append(event.payload)
        return BeforeProviderPayloadResult(payload={"steps": ["provider", "first"]})

    def second_hook(event):
        seen_payloads.append(event.payload)
        return BeforeProviderPayloadResult(payload={"steps": ["provider", "first", "second"]})

    harness.on("before_provider_payload", first_hook)
    harness.on("before_provider_payload", second_hook)

    await harness.prompt("hello")

    assert seen_payloads == [{"steps": ["provider"]}, {"steps": ["provider", "first"]}]
    assert final_payload == {"steps": ["provider", "first", "second"]}
