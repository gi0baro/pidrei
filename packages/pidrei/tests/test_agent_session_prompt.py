"""Partial mirror of pi's suite/agent-session-prompt.test.ts: the 0.87.0
image-normalization case and the 1.2.0 cases for runs started by a custom
message. The rest of the prompt characterization suite is not mirrored here.

pi mocks `processImage`; here `agent_session.process_image` is swapped, which
is the name the prompt path calls.
"""

import base64

import pytest
import tonio.colored as tonio

from pidrei.core import agent_session as agent_session_module
from pidrei.core.agent_session import PromptOptions
from pidrei.core.extensions import ToolDefinition
from pidrei.utils.image_process import ProcessImageResult
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import (
    ImageContent,
    ModelImageInputLimits,
    ModelImageResizeOptions,
    ModelInputLimits,
    TextContent,
)
from pidrei_ai.utils.transcript import get_current_system_prompt

from .harness import create_harness


TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
NORMALIZED_BASE64 = base64.b64encode(b"normalized").decode("ascii")


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


@pytest.fixture
def process_image_calls(monkeypatch) -> list[tuple]:
    calls: list[tuple] = []

    def process_image(data, mime_type, **options):
        calls.append((data, mime_type, options))
        return ProcessImageResult(ok=True, data=NORMALIZED_BASE64, mime_type=mime_type, hints=[])

    monkeypatch.setattr(agent_session_module, "process_image", process_image)
    return calls


# Regression test for https://github.com/earendil-works/pi/issues/9631
@pytest.mark.tonio
async def test_uses_the_model_selected_by_before_agent_start_for_image_normalization(harnesses, process_image_calls):
    strict_model = None

    async def factory(pi) -> None:
        async def on_before_agent_start(_event, _ctx):
            if strict_model is None:
                raise Exception("Expected strict model")
            await pi.set_model(strict_model)

        pi.on("before_agent_start", on_before_agent_start)

    resize_options = ModelImageResizeOptions(max_width=1000, max_height=1000, max_bytes=500000, jpeg_quality=70)
    # The limits are part of the model definition: `Model`s are shared with the
    # provider and the model runtime, so tests never assign to one in place.
    harness = await create_harness(
        models=[
            {"id": "wide"},
            {"id": "strict", "input_limits": ModelInputLimits(images=ModelImageInputLimits(resize=resize_options))},
        ],
        extension_factories=[factory],
    )
    harnesses.append(harness)
    strict_model = harness.get_model("strict")
    harness.set_responses([faux_assistant_message("done")])

    await harness.session.prompt(
        "inspect", PromptOptions(images=[ImageContent(data=TINY_PNG_BASE64, mime_type="image/png")])
    )

    assert harness.session.model.id == "strict"
    assert len(process_image_calls) == 1
    data, mime_type, options = process_image_calls[0]
    assert isinstance(data, bytes)
    assert mime_type == "image/png"
    assert options == {"auto_resize_images": True, "resize_options": resize_options}
    user_message = next(message for message in harness.session.messages if message.role == "user")
    assert ImageContent(data=NORMALIZED_BASE64, mime_type="image/png") in user_message.content


# Issue #10267, #5581: a run started by send_message(..., {"triggerTurn": True}) skipped
# before_agent_start, so its second request patched away the sections handlers had added.


@pytest.mark.tonio
async def test_a_run_started_by_a_custom_message_keeps_the_same_system_prompt_for_every_request(harnesses):
    events: list[dict] = []

    async def noop(*_args):
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    noop_tool = ToolDefinition(
        name="noop",
        label="Noop",
        description="Does nothing",
        parameters={"type": "object", "properties": {}},
        execute=noop,
    )

    async def factory(pi) -> None:
        async def on_before_agent_start(event, _ctx):
            events.append(event)
            event["systemPromptOptions"].sections["extension_rules"] = "Answer in one word."

        pi.on("before_agent_start", on_before_agent_start)

    harness = await create_harness(tools=[noop_tool], extension_factories=[factory])
    harnesses.append(harness)
    prompts: list[str] = []

    def record(response):
        async def respond(context, *_rest):
            prompts.append(get_current_system_prompt(context.messages))
            return response

        return respond

    harness.set_responses(
        [
            record(faux_assistant_message("hi")),
            record(faux_assistant_message(faux_tool_call("noop", {}), stop_reason="toolUse")),
            record(faux_assistant_message("done")),
        ]
    )

    await harness.session.prompt("hello")
    await harness.session.send_custom_message(
        {"customType": "task-done", "content": "background task finished", "display": True},
        {"triggerTurn": True},
    )

    assert len(prompts) == 3
    assert "Answer in one word." in prompts[0]
    assert len(set(prompts)) == 1
    assert [event["messageType"] for event in events] == ["user", "custom"]
    assert events[1]["message"].custom_type == "task-done"


def _user_texts(context) -> list[str]:
    return [
        part.text
        for message in context.messages
        if message.role == "user" and not isinstance(message.content, str)
        for part in message.content
        if part.type == "text"
    ]


@pytest.mark.tonio
async def test_queues_a_custom_message_sent_while_before_agent_start_runs_into_the_starting_run(harnesses):
    handler_started = tonio.Event()
    release_handler = tonio.Event()

    async def factory(pi) -> None:
        async def on_before_agent_start(event, _ctx):
            if event["messageType"] != "custom" or event["message"].custom_type != "first":
                return
            handler_started.set()
            await release_handler.wait(5)

        pi.on("before_agent_start", on_before_agent_start)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    requests: list = []

    async def respond(context, *_rest):
        requests.append(context)
        return faux_assistant_message("handled both")

    harness.set_responses([respond])

    first = tonio.spawn(
        harness.session.send_custom_message(
            {"customType": "first", "content": "task one finished", "display": True}, {"triggerTurn": True}
        )
    )
    await handler_started.wait(5)
    assert handler_started.is_set()
    assert harness.session.is_idle is False
    # Queued in the call's synchronous prefix, before the handler is released, as in pi.
    await harness.session.send_custom_message(
        {"customType": "second", "content": "task two finished", "display": True}, {"triggerTurn": True}
    )
    release_handler.set()
    await first

    assert len(requests) == 1
    assert _user_texts(requests[0]) == ["task one finished", "task two finished"]


@pytest.mark.tonio
async def test_includes_a_custom_message_sent_without_trigger_turn_while_the_run_starts_in_its_first_request(
    harnesses,
):
    async def factory(pi) -> None:
        async def on_before_agent_start(_event, _ctx):
            pi.send_message(
                {"customType": "note", "content": "context from handler", "display": True}, {"triggerTurn": False}
            )

        pi.on("before_agent_start", on_before_agent_start)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    requests: list = []

    async def respond(context, *_rest):
        requests.append(context)
        return faux_assistant_message("done")

    harness.set_responses([respond])

    await harness.session.prompt("hello")

    assert len(requests) == 1
    assert "context from handler" in _user_texts(requests[0])


@pytest.mark.tonio
async def test_aborts_a_run_when_abort_is_requested_while_before_agent_start_runs(harnesses):
    handler_started = tonio.Event()
    release_handler = tonio.Event()
    contexts: list = []

    async def factory(pi) -> None:
        async def on_before_agent_start(_event, ctx):
            contexts.append(ctx)
            handler_started.set()
            await release_handler.wait(5)

        pi.on("before_agent_start", on_before_agent_start)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message("should not complete")])

    run = tonio.spawn(
        harness.session.send_custom_message(
            {"customType": "task-done", "content": "background task finished", "display": True},
            {"triggerTurn": True},
        )
    )
    await handler_started.wait(5)
    assert handler_started.is_set()
    # pi calls session.abort() without awaiting it, so its synchronous prefix lands
    # before the handler is released; ctx.abort() is that prefix.
    contexts[0].abort()
    release_handler.set()
    await run
    await harness.session.wait_for_idle()

    # pi's harness streams straight into the faux provider, which answers a fired
    # signal with stopReason "aborted". Here the request goes through the model
    # runtime, whose auth resolution sees the fired token first and fails the
    # stream's setup, as pi's runtime does: the request is never sent.
    last = harness.session.messages[-1]
    assert last.role == "assistant"
    assert last.error_message == "Operation was aborted"
    assert harness.get_pending_response_count() == 1
    assert harness.session.is_idle is True
