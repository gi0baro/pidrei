"""Mirror of pi's openai-responses-input-transformations.test.ts.

The shared stream processor is driven with a builder and an async generator of
dict events; the Codex case goes over SSE through
`OpenAICodexResponsesOptions.client` (pi stubs the global fetch). In the
malformed-reports case pi's `undefined` is the key left out and `null` is None.
"""

import json
from dataclasses import replace

import pytest

from pidrei_ai.api.openai_codex_responses import OpenAICodexResponsesOptions, stream as stream_codex
from pidrei_ai.api.openai_responses_shared import process_responses_stream
from pidrei_ai.builders import AssistantMessageBuilder, UsageBuilder
from pidrei_ai.types import AssistantMessageDiagnostic, Context, Model, ModelCost, UserMessage
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.utils.transcript import normalize_context
from pidrei_utils import clock
from tests.test_openai_codex_stream import FakeCodexClient, FakeCodexResponse, mock_token


MODEL = Model(
    id="gpt-6.2-sol",
    name="GPT-6.2 Sol",
    api="openai-responses",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=True,
    input=["text"],
    cost=ModelCost(),
    context_window=272000,
    max_tokens=128000,
)

_ABSENT = object()


def create_output() -> AssistantMessageBuilder:
    return AssistantMessageBuilder(
        content=[],
        api=MODEL.api,
        provider=MODEL.provider,
        model=MODEL.id,
        usage=UsageBuilder(),
        stop_reason="pending",
        timestamp=clock.now_ms(),
    )


async def process_terminal_response(input_transformations=_ABSENT, status: str = "completed"):
    """Run one terminal response event carrying `input_transformations` through the shared stream processor."""
    response: dict = {"id": "resp_transformations", "status": status, "output": []}
    if status == "incomplete":
        response["incomplete_details"] = {"reason": "max_output_tokens"}
    if input_transformations is not _ABSENT:
        response["input_transformations"] = input_transformations

    async def events():
        yield {
            "type": "response.incomplete" if status == "incomplete" else "response.completed",
            "sequence_number": 0,
            "response": response,
        }

    output = create_output()
    await process_responses_stream(events(), output, AssistantMessageEventStream(), MODEL)
    return output


PREFIX_MISMATCH = {"type": "reasoning", "id": "rs_edited", "action": "removed", "reason": "prefix_mismatch"}
ORGANIZATION_MISMATCH = {"type": "reasoning", "id": "rs_org", "action": "removed", "reason": "organization_mismatch"}
TURN_BOUNDARY = {"type": "reasoning", "id": "rs_old_turn", "action": "removed", "reason": "turn_boundary"}


def _only_diagnostic(diagnostics) -> AssistantMessageDiagnostic:
    assert diagnostics is not None and len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.type == "openai_input_transformations"
    assert isinstance(diagnostic.timestamp, int)
    assert diagnostic.error is None
    return diagnostic


@pytest.mark.tonio
async def test_records_removed_reasoning_items_with_their_reasons():
    output = await process_terminal_response([PREFIX_MISMATCH, TURN_BOUNDARY, ORGANIZATION_MISMATCH])

    assert output.stop_reason == "stop"
    assert _only_diagnostic(output.diagnostics).details == {"transformations": [PREFIX_MISMATCH, ORGANIZATION_MISMATCH]}


@pytest.mark.tonio
async def test_records_removals_on_incomplete_responses():
    output = await process_terminal_response([PREFIX_MISMATCH], "incomplete")

    assert output.stop_reason == "length"
    assert output.diagnostics[0].details == {"transformations": [PREFIX_MISMATCH]}


@pytest.mark.tonio
async def test_keeps_reasons_the_sdk_does_not_know_yet():
    future_reason = {"type": "reasoning", "id": "rs_future", "action": "removed", "reason": "future_reason"}
    output = await process_terminal_response([future_reason])

    assert output.diagnostics[0].details == {"transformations": [future_reason]}


@pytest.mark.tonio
async def test_ignores_turn_boundaries_other_transformations_and_malformed_reports():
    # Models before GPT-5.6 report every earlier turn's reasoning as a turn boundary on each request.
    reports = [
        _ABSENT,
        None,
        [],
        [TURN_BOUNDARY, TURN_BOUNDARY],
        {"malformed": True},
        [None, "rs_text", {"type": "message", "id": "msg_1", "action": "removed", "reason": "prefix_mismatch"}],
        [{"type": "reasoning", "id": "rs_changed", "action": "changed", "reason": "prefix_mismatch"}],
    ]
    for report in reports:
        output = await process_terminal_response(report)
        assert output.stop_reason == "stop", report
        assert not output.diagnostics, report


@pytest.mark.tonio
async def test_records_removals_that_codex_reports_in_response_done():
    terminal = {
        "type": "response.done",
        "response": {
            "id": "resp_codex",
            "status": "completed",
            "output": [],
            "input_transformations": [PREFIX_MISMATCH, TURN_BOUNDARY],
        },
    }
    client = FakeCodexClient(lambda _request: FakeCodexResponse([f"data: {json.dumps(terminal)}\n\n".encode()]))
    codex_model = replace(
        MODEL, api="openai-codex-responses", provider="openai-codex", base_url="https://chatgpt.com/backend-api"
    )

    result = await stream_codex(
        codex_model,
        normalize_context(Context(messages=[UserMessage(content="Hello", timestamp=1)])),
        OpenAICodexResponsesOptions(api_key=mock_token(), transport="sse", client=client),
    ).result()

    assert result.stop_reason == "stop", result.error_message
    assert _only_diagnostic(result.diagnostics).details == {"transformations": [PREFIX_MISMATCH]}
