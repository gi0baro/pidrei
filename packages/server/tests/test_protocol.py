"""Port of pi server `test/protocol.test.ts`.

JS-only sanitize/JSON cases (bigint, symbol, undefined, sparse arrays) are
replaced with Python-appropriate invalid payloads: unsupported object types,
non-finite floats, out-of-safe-range ints, and callables.
"""

from dataclasses import replace

import pytest

from pidrei_ai.types import (
    AssistantMessage,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)
from pidrei_protocol import PROTOCOL_VERSION, encode_server_message
from pidrei_server import (
    AssistantTranscriptOptions,
    ToolTranscriptOptions,
    UserTranscriptOptions,
    sanitize_protocol_details,
    to_protocol_assistant_message,
    to_protocol_json_value,
    to_protocol_model_metadata,
    to_protocol_tool_result_message,
    to_protocol_user_message,
)


MODEL = Model(
    id="model-1",
    name="Model One",
    api="test-api",
    provider="test-provider",
    base_url="https://example.test",
    reasoning=True,
    input=["text", "image"],
    cost=ModelCost(input=1, output=2, cache_read=0.1, cache_write=0.2),
    context_window=100_000,
    max_tokens=10_000,
)


def _zero_usage() -> Usage:
    return Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0, cost=UsageCost())


def assert_valid_server_payload(item):
    encode_server_message(
        {
            "type": "hello",
            "version": PROTOCOL_VERSION,
            "connectionId": "connection-1",
            "snapshot": {
                "serverId": "server-1",
                "protocolVersion": PROTOCOL_VERSION,
                "revision": 0,
                "sessions": [
                    {
                        "id": "session-1",
                        "createdAt": 1,
                        "updatedAt": 1,
                        "sessionName": "Session one",
                        "cwd": "/workspace",
                    }
                ],
                "models": [to_protocol_model_metadata(MODEL, True)],
            },
        }
    )

    encode_server_message(
        {
            "type": "event",
            "event": {
                "type": "session_snapshot",
                "snapshot": {
                    "id": "session-1",
                    "cwd": "/workspace",
                    "createdAt": 1,
                    "updatedAt": 1,
                    "phase": "idle",
                    "model": {"provider": "test-provider", "id": "model-1"},
                    "thinkingLevel": "off",
                    "attached": True,
                    "locked": True,
                    "revision": 1,
                    "transcript": [item],
                    "queuedSteer": [],
                    "queuedSteerCount": 0,
                },
            },
        }
    )


def test_maps_model_metadata_and_produces_protocol_valid_output():
    result = to_protocol_model_metadata(MODEL, True)

    assert result["provider"] == "test-provider"
    assert result["id"] == "model-1"
    assert result["api"] == "test-api"
    assert result["input"] == ["text", "image"]
    assert result["authenticated"] is True
    assert "off" in result["supportedThinkingLevels"]


def test_exhaustively_maps_assistant_content_and_stop_reasons():
    message = AssistantMessage(
        content=[
            TextContent(text="hello"),
            ThinkingContent(thinking="hmm", redacted=False),
            ToolCall(id="call-1", name="read", arguments={"path": "README.md"}),
        ],
        api="test-api",
        provider="test-provider",
        model="model-1",
        usage=Usage(
            input=1,
            output=2,
            cache_read=3,
            cache_write=4,
            total_tokens=10,
            cost=UsageCost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1),
        ),
        stop_reason="toolUse",
        timestamp=123,
    )

    result = to_protocol_assistant_message(message, AssistantTranscriptOptions(id="message-1"))

    assert result["id"] == "message-1"
    assert result["status"] == "complete"
    assert result["stopReason"] == "toolUse"
    assert result["model"] == {"provider": "test-provider", "id": "model-1"}
    assert result["content"] == [
        {"type": "text", "text": "hello"},
        {"type": "thinking", "thinking": "hmm", "redacted": False},
        {"type": "toolCall", "toolCallId": "call-1", "toolName": "read", "input": {"path": "README.md"}},
    ]
    assert_valid_server_payload(result)


def test_maps_user_and_tool_messages_without_leaking_non_json_details():
    user = UserMessage(content="hello", timestamp=1)
    circular = {}
    circular["self"] = circular
    tool = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="read",
        content=[TextContent(text="result")],
        details=circular,
        is_error=False,
        timestamp=2,
    )
    call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})

    user_result = to_protocol_user_message(user, UserTranscriptOptions(id="user-1"))
    assert user_result["id"] == "user-1"
    assert user_result["content"] == [{"type": "text", "text": "hello"}]
    assert_valid_server_payload(user_result)

    tool_result = to_protocol_tool_result_message(tool, ToolTranscriptOptions(id="tool-1", call=call))
    assert tool_result["id"] == "tool-1"
    assert tool_result["toolName"] == "read"
    assert tool_result["input"] == {"path": "README.md"}
    assert tool_result["details"] == {"self": "[Circular]"}
    assert tool_result["status"] == "complete"
    assert_valid_server_payload(tool_result)


def test_rejects_tool_results_associated_with_a_different_call():
    call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
    result = ToolResultMessage(
        tool_call_id="call-2",
        tool_name="read",
        content=[TextContent(text="result")],
        is_error=False,
        timestamp=2,
    )

    with pytest.raises(TypeError, match="tool call"):
        to_protocol_tool_result_message(result, ToolTranscriptOptions(id="tool-1", call=call))
    with pytest.raises(TypeError, match="tool call"):
        to_protocol_tool_result_message(
            replace(result, tool_call_id="call-1", tool_name="write"),
            ToolTranscriptOptions(id="tool-1", call=call),
        )


def test_derives_streaming_status_from_a_pending_stop_reason():
    message = AssistantMessage(
        content=[TextContent(text="partial")],
        api="test-api",
        provider="test-provider",
        model="model-1",
        usage=_zero_usage(),
        stop_reason="pending",
        timestamp=123,
    )

    result = to_protocol_assistant_message(message, AssistantTranscriptOptions(id="message-pending"))
    assert result["status"] == "streaming"
    assert "stopReason" not in result
    assert_valid_server_payload(result)


def test_preserves_optional_non_empty_assistant_error_messages():
    message = AssistantMessage(
        content=[],
        api="test-api",
        provider="test-provider",
        model="model-1",
        usage=_zero_usage(),
        stop_reason="error",
        timestamp=123,
    )

    result_without_message = to_protocol_assistant_message(message, AssistantTranscriptOptions(id="message-error"))
    assert result_without_message["status"] == "error"
    assert result_without_message["stopReason"] == "error"
    assert "errorMessage" not in result_without_message
    assert_valid_server_payload(result_without_message)
    with pytest.raises(TypeError):
        to_protocol_assistant_message(
            replace(message, error_message=""), AssistantTranscriptOptions(id="message-error")
        )
    result_with_message = to_protocol_assistant_message(
        replace(message, error_message="failed"), AssistantTranscriptOptions(id="message-error")
    )
    assert result_with_message["status"] == "error"
    assert result_with_message["stopReason"] == "error"
    assert result_with_message["errorMessage"] == "failed"
    assert_valid_server_payload(result_with_message)


def test_rejects_invalid_source_identifiers_and_timestamps():
    message = AssistantMessage(
        content=[ToolCall(id="", name="read", arguments={})],
        api="test-api",
        provider="test-provider",
        model="model-1",
        usage=_zero_usage(),
        stop_reason="toolUse",
        timestamp=1,
    )

    with pytest.raises(TypeError, match="[Tt]ool call id"):
        to_protocol_assistant_message(message, AssistantTranscriptOptions(id="assistant-1"))
    with pytest.raises(TypeError, match="timestamp"):
        to_protocol_user_message(
            UserMessage(content="hello", timestamp=float("nan")), UserTranscriptOptions(id="user-1")
        )


def test_rejects_lossy_tool_input_conversions():
    circular = {}
    circular["self"] = circular

    with pytest.raises(TypeError):
        to_protocol_json_value(float("inf"))
    with pytest.raises(TypeError):
        to_protocol_json_value(b"raw bytes")
    with pytest.raises(TypeError):
        to_protocol_json_value(object())
    with pytest.raises(TypeError):
        to_protocol_json_value({1, 2})
    with pytest.raises(TypeError):
        to_protocol_json_value(circular)


def test_rejects_unsupported_execution_data_and_normalizes_diagnostic_details():
    with pytest.raises(TypeError, match="Unsupported"):
        to_protocol_json_value([1, object()])
    assert sanitize_protocol_details([None, "value"]) == [None, "value"]
    assert sanitize_protocol_details([lambda: None, "value"]) == [None, "value"]
    assert sanitize_protocol_details({"fn": lambda: None, "kept": 1}) == {"kept": 1}
    assert sanitize_protocol_details(float("nan")) == "nan"
    assert sanitize_protocol_details(2**60) == str(2**60)
