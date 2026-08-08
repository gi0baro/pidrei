"""Wire message schemas (port of pi protocol `schemas.ts`).

pi validates with TypeBox; the established pidrei mapping (model_config.py,
pidrei_ai/utils/validation.py) is hand-rolled JSON Schema dicts run through
`jsonschema`. Wire values stay raw camelCase dicts — there are no typed
message classes, mirroring how pi passes the TypeBox-checked plain objects
around. Every object schema is strict (`additionalProperties: false`,
non-optional properties required), matching TypeBox `Check` semantics —
including the bool/int distinction and integral floats counting as integers.

The recursive JSON-value schema lives in `$defs` on the two top-level message
schemas (the only ones handed to a validator); component schemas reference it
as `#/$defs/jsonValue`.
"""

from typing import Any, Literal


PROTOCOL_VERSION = 1

_ID = {"type": "string", "minLength": 1}
_TIMESTAMP = {"type": "integer", "minimum": 0}
_STRING = {"type": "string"}
_BOOLEAN = {"type": "boolean"}
_NON_EMPTY_STRING = {"type": "string", "minLength": 1}
_NON_NEGATIVE_INTEGER = {"type": "integer", "minimum": 0}
_NON_NEGATIVE_NUMBER = {"type": "number", "minimum": 0}


def _strict_object(properties: dict[str, Any], *, optional: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": [name for name in properties if name not in optional],
        "additionalProperties": False,
    }


type JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

_JSON_VALUE_DEFINITION = {
    "anyOf": [
        {"type": "null"},
        _BOOLEAN,
        {"type": "number"},
        _STRING,
        {"type": "array", "items": {"$ref": "#/$defs/jsonValue"}},
        {"type": "object", "additionalProperties": {"$ref": "#/$defs/jsonValue"}},
    ]
}
JSON_VALUE_SCHEMA = {"$ref": "#/$defs/jsonValue"}

THINKING_LEVEL_SCHEMA = {"enum": ["off", "minimal", "low", "medium", "high", "xhigh", "max"]}
type ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]

# Matches AgentHarnessPhase so adapters do not need a second phase vocabulary.
SESSION_PHASE_SCHEMA = {"enum": ["idle", "turn", "compaction", "branch_summary", "retry"]}
type SessionPhase = Literal["idle", "turn", "compaction", "branch_summary", "retry"]

MODEL_REF_SCHEMA = _strict_object({"provider": _ID, "id": _ID})

MODEL_COST_SCHEMA = _strict_object(
    {
        "input": _NON_NEGATIVE_NUMBER,
        "output": _NON_NEGATIVE_NUMBER,
        "cacheRead": _NON_NEGATIVE_NUMBER,
        "cacheWrite": _NON_NEGATIVE_NUMBER,
    }
)

MODEL_METADATA_SCHEMA = _strict_object(
    {
        "provider": _ID,
        "id": _ID,
        "name": _NON_EMPTY_STRING,
        "api": _ID,
        "reasoning": _BOOLEAN,
        "input": {"type": "array", "items": {"enum": ["text", "image"]}},
        "contextWindow": {"type": "integer", "minimum": 1},
        "maxTokens": {"type": "integer", "minimum": 1},
        "cost": MODEL_COST_SCHEMA,
        "supportedThinkingLevels": {"type": "array", "items": THINKING_LEVEL_SCHEMA, "minItems": 1},
        "authenticated": _BOOLEAN,
    }
)

TEXT_CONTENT_SCHEMA = _strict_object({"type": {"const": "text"}, "text": _STRING})
THINKING_CONTENT_SCHEMA = _strict_object(
    {"type": {"const": "thinking"}, "thinking": _STRING, "redacted": _BOOLEAN}, optional=("redacted",)
)
IMAGE_CONTENT_SCHEMA = _strict_object({"type": {"const": "image"}, "data": _STRING, "mimeType": _NON_EMPTY_STRING})
TOOL_CALL_CONTENT_SCHEMA = _strict_object(
    {"type": {"const": "toolCall"}, "toolCallId": _ID, "toolName": _ID, "input": JSON_VALUE_SCHEMA}
)
USER_CONTENT_SCHEMA = {"anyOf": [TEXT_CONTENT_SCHEMA, IMAGE_CONTENT_SCHEMA]}
ASSISTANT_CONTENT_SCHEMA = {"anyOf": [TEXT_CONTENT_SCHEMA, THINKING_CONTENT_SCHEMA, TOOL_CALL_CONTENT_SCHEMA]}
TOOL_CONTENT_SCHEMA = {"anyOf": [TEXT_CONTENT_SCHEMA, IMAGE_CONTENT_SCHEMA]}

USAGE_SCHEMA = _strict_object(
    {
        "input": _NON_NEGATIVE_INTEGER,
        "output": _NON_NEGATIVE_INTEGER,
        "cacheRead": _NON_NEGATIVE_INTEGER,
        "cacheWrite": _NON_NEGATIVE_INTEGER,
        "reasoning": _NON_NEGATIVE_INTEGER,
        "totalTokens": _NON_NEGATIVE_INTEGER,
        "cost": _strict_object(
            {
                "input": _NON_NEGATIVE_NUMBER,
                "output": _NON_NEGATIVE_NUMBER,
                "cacheRead": _NON_NEGATIVE_NUMBER,
                "cacheWrite": _NON_NEGATIVE_NUMBER,
                "total": _NON_NEGATIVE_NUMBER,
            }
        ),
    },
    optional=("reasoning",),
)

USER_TRANSCRIPT_ITEM_SCHEMA = _strict_object(
    {
        "id": _ID,
        "role": {"const": "user"},
        "content": {"type": "array", "items": USER_CONTENT_SCHEMA},
        "timestamp": _TIMESTAMP,
    }
)

_ASSISTANT_TRANSCRIPT_ITEM_PROPERTIES = {
    "id": _ID,
    "role": {"const": "assistant"},
    "content": {"type": "array", "items": ASSISTANT_CONTENT_SCHEMA},
    "model": MODEL_REF_SCHEMA,
    "responseModel": _NON_EMPTY_STRING,
    "usage": USAGE_SCHEMA,
    "timestamp": _TIMESTAMP,
}
_ASSISTANT_OPTIONAL = ("responseModel", "usage")
_STREAMING_ASSISTANT_TRANSCRIPT_ITEM_SCHEMA = _strict_object(
    {**_ASSISTANT_TRANSCRIPT_ITEM_PROPERTIES, "status": {"const": "streaming"}},
    optional=_ASSISTANT_OPTIONAL,
)
_COMPLETE_ASSISTANT_TRANSCRIPT_ITEM_SCHEMA = _strict_object(
    {
        **_ASSISTANT_TRANSCRIPT_ITEM_PROPERTIES,
        "status": {"const": "complete"},
        "stopReason": {"enum": ["stop", "length", "toolUse"]},
    },
    optional=_ASSISTANT_OPTIONAL,
)
_ERROR_ASSISTANT_TRANSCRIPT_ITEM_SCHEMA = _strict_object(
    {
        **_ASSISTANT_TRANSCRIPT_ITEM_PROPERTIES,
        "status": {"const": "error"},
        "stopReason": {"const": "error"},
        "errorMessage": _NON_EMPTY_STRING,
    },
    optional=(*_ASSISTANT_OPTIONAL, "errorMessage"),
)
_ABORTED_ASSISTANT_TRANSCRIPT_ITEM_SCHEMA = _strict_object(
    {
        **_ASSISTANT_TRANSCRIPT_ITEM_PROPERTIES,
        "status": {"const": "aborted"},
        "stopReason": {"const": "aborted"},
        "errorMessage": _STRING,
    },
    optional=(*_ASSISTANT_OPTIONAL, "errorMessage"),
)
ASSISTANT_TRANSCRIPT_ITEM_SCHEMA = {
    "anyOf": [
        _STREAMING_ASSISTANT_TRANSCRIPT_ITEM_SCHEMA,
        _COMPLETE_ASSISTANT_TRANSCRIPT_ITEM_SCHEMA,
        _ERROR_ASSISTANT_TRANSCRIPT_ITEM_SCHEMA,
        _ABORTED_ASSISTANT_TRANSCRIPT_ITEM_SCHEMA,
    ]
}

_TOOL_TRANSCRIPT_ITEM_PROPERTIES = {
    "id": _ID,
    "role": {"const": "tool"},
    "toolCallId": _ID,
    "toolName": _ID,
    "input": JSON_VALUE_SCHEMA,
    "content": {"type": "array", "items": TOOL_CONTENT_SCHEMA},
    "details": JSON_VALUE_SCHEMA,
    "usage": USAGE_SCHEMA,
    "timestamp": _TIMESTAMP,
}
_TOOL_OPTIONAL = ("details", "usage")
_RUNNING_TOOL_TRANSCRIPT_ITEM_SCHEMA = _strict_object(
    {**_TOOL_TRANSCRIPT_ITEM_PROPERTIES, "status": {"const": "running"}, "isError": {"const": False}},
    optional=_TOOL_OPTIONAL,
)
_COMPLETE_TOOL_TRANSCRIPT_ITEM_SCHEMA = _strict_object(
    {**_TOOL_TRANSCRIPT_ITEM_PROPERTIES, "status": {"const": "complete"}, "isError": {"const": False}},
    optional=_TOOL_OPTIONAL,
)
_ERROR_TOOL_TRANSCRIPT_ITEM_SCHEMA = _strict_object(
    {**_TOOL_TRANSCRIPT_ITEM_PROPERTIES, "status": {"const": "error"}, "isError": {"const": True}},
    optional=_TOOL_OPTIONAL,
)
TOOL_TRANSCRIPT_ITEM_SCHEMA = {
    "anyOf": [
        _RUNNING_TOOL_TRANSCRIPT_ITEM_SCHEMA,
        _COMPLETE_TOOL_TRANSCRIPT_ITEM_SCHEMA,
        _ERROR_TOOL_TRANSCRIPT_ITEM_SCHEMA,
    ]
}
TRANSCRIPT_ITEM_SCHEMA = {
    "anyOf": [USER_TRANSCRIPT_ITEM_SCHEMA, ASSISTANT_TRANSCRIPT_ITEM_SCHEMA, TOOL_TRANSCRIPT_ITEM_SCHEMA]
}

# Normalized incremental activity. Snapshots remain authoritative.
TRANSCRIPT_PROGRESS_SCHEMA = {
    "anyOf": [
        _strict_object({"type": {"const": "item_started"}, "item": TRANSCRIPT_ITEM_SCHEMA}),
        _strict_object(
            {
                "type": {"const": "assistant_delta"},
                "messageId": _ID,
                "contentIndex": _NON_NEGATIVE_INTEGER,
                "kind": {"enum": ["text", "thinking", "toolCall"]},
                "delta": _STRING,
            }
        ),
        _strict_object(
            {
                "type": {"const": "item_updated"},
                "item": {"anyOf": [ASSISTANT_TRANSCRIPT_ITEM_SCHEMA, TOOL_TRANSCRIPT_ITEM_SCHEMA]},
            }
        ),
        _strict_object(
            {
                "type": {"const": "item_finished"},
                "item": {
                    "anyOf": [
                        _COMPLETE_ASSISTANT_TRANSCRIPT_ITEM_SCHEMA,
                        _ERROR_ASSISTANT_TRANSCRIPT_ITEM_SCHEMA,
                        _ABORTED_ASSISTANT_TRANSCRIPT_ITEM_SCHEMA,
                        _COMPLETE_TOOL_TRANSCRIPT_ITEM_SCHEMA,
                        _ERROR_TOOL_TRANSCRIPT_ITEM_SCHEMA,
                    ]
                },
            }
        ),
    ]
}

SESSION_METADATA_SCHEMA = _strict_object(
    {
        "id": _ID,
        "createdAt": _TIMESTAMP,
        "updatedAt": _TIMESTAMP,
        "parentSessionId": _ID,
        "sessionName": _STRING,
        "cwd": _NON_EMPTY_STRING,
    },
    optional=("updatedAt", "parentSessionId", "sessionName", "cwd"),
)
SESSION_SNAPSHOT_SCHEMA = _strict_object(
    {
        "id": _ID,
        "name": _STRING,
        "cwd": _NON_EMPTY_STRING,
        "createdAt": _TIMESTAMP,
        "updatedAt": _TIMESTAMP,
        "phase": SESSION_PHASE_SCHEMA,
        "model": MODEL_REF_SCHEMA,
        "thinkingLevel": THINKING_LEVEL_SCHEMA,
        "attached": _BOOLEAN,
        "locked": _BOOLEAN,
        "revision": _NON_NEGATIVE_INTEGER,
        "transcript": {"type": "array", "items": TRANSCRIPT_ITEM_SCHEMA},
        "queuedSteer": {"type": "array", "items": USER_TRANSCRIPT_ITEM_SCHEMA},
        "queuedSteerCount": _NON_NEGATIVE_INTEGER,
    },
    optional=("name",),
)

SERVER_SNAPSHOT_SCHEMA = _strict_object(
    {
        "serverId": _ID,
        "protocolVersion": {"const": PROTOCOL_VERSION},
        "revision": _NON_NEGATIVE_INTEGER,
        "sessions": {"type": "array", "items": SESSION_METADATA_SCHEMA},
        "models": {"type": "array", "items": MODEL_METADATA_SCHEMA},
    }
)

PROTOCOL_ERROR_CODE_SCHEMA = {
    "enum": ["version", "busy", "session_locked", "not_found", "invalid_request", "not_implemented", "internal_error"]
}
type ProtocolErrorCode = Literal[
    "version", "busy", "session_locked", "not_found", "invalid_request", "not_implemented", "internal_error"
]
PROTOCOL_ERROR_SCHEMA = _strict_object(
    {"code": PROTOCOL_ERROR_CODE_SCHEMA, "message": _STRING, "details": JSON_VALUE_SCHEMA},
    optional=("details",),
)

_PROMPT_PAYLOAD_PROPERTIES = {"sessionId": _ID, "text": _STRING}

LIST_COMMAND_SCHEMA = _strict_object({"command": {"const": "list"}})
CREATE_COMMAND_SCHEMA = _strict_object(
    {
        "command": {"const": "create"},
        "cwd": _NON_EMPTY_STRING,
        "name": _STRING,
        "model": MODEL_REF_SCHEMA,
        "thinkingLevel": THINKING_LEVEL_SCHEMA,
    },
    optional=("cwd", "name", "model", "thinkingLevel"),
)
ATTACH_COMMAND_SCHEMA = _strict_object({"command": {"const": "attach"}, "sessionId": _ID})
DETACH_COMMAND_SCHEMA = _strict_object({"command": {"const": "detach"}, "sessionId": _ID})
PROMPT_COMMAND_SCHEMA = _strict_object({"command": {"const": "prompt"}, **_PROMPT_PAYLOAD_PROPERTIES})
STEER_COMMAND_SCHEMA = _strict_object({"command": {"const": "steer"}, **_PROMPT_PAYLOAD_PROPERTIES})
ABORT_COMMAND_SCHEMA = _strict_object({"command": {"const": "abort"}, "sessionId": _ID})
SET_MODEL_COMMAND_SCHEMA = _strict_object(
    {"command": {"const": "set_model"}, "sessionId": _ID, "model": MODEL_REF_SCHEMA}
)
SET_THINKING_COMMAND_SCHEMA = _strict_object(
    {"command": {"const": "set_thinking"}, "sessionId": _ID, "thinkingLevel": THINKING_LEVEL_SCHEMA}
)
COMMAND_SCHEMA = {
    "anyOf": [
        LIST_COMMAND_SCHEMA,
        CREATE_COMMAND_SCHEMA,
        ATTACH_COMMAND_SCHEMA,
        DETACH_COMMAND_SCHEMA,
        PROMPT_COMMAND_SCHEMA,
        STEER_COMMAND_SCHEMA,
        ABORT_COMMAND_SCHEMA,
        SET_MODEL_COMMAND_SCHEMA,
        SET_THINKING_COMMAND_SCHEMA,
    ]
}
type CommandName = Literal[
    "list", "create", "attach", "detach", "prompt", "steer", "abort", "set_model", "set_thinking"
]

CREATE_RESULT_SCHEMA = _strict_object({"command": {"const": "create"}, "session": SESSION_SNAPSHOT_SCHEMA})
ATTACH_RESULT_SCHEMA = _strict_object({"command": {"const": "attach"}, "session": SESSION_SNAPSHOT_SCHEMA})
PROMPT_RESULT_SCHEMA = _strict_object({"command": {"const": "prompt"}, "session": SESSION_SNAPSHOT_SCHEMA})
STEER_RESULT_SCHEMA = _strict_object({"command": {"const": "steer"}, "session": SESSION_SNAPSHOT_SCHEMA})
ABORT_RESULT_SCHEMA = _strict_object({"command": {"const": "abort"}, "session": SESSION_SNAPSHOT_SCHEMA})
SET_MODEL_RESULT_SCHEMA = _strict_object({"command": {"const": "set_model"}, "session": SESSION_SNAPSHOT_SCHEMA})
SET_THINKING_RESULT_SCHEMA = _strict_object({"command": {"const": "set_thinking"}, "session": SESSION_SNAPSHOT_SCHEMA})

LIST_RESULT_SCHEMA = _strict_object(
    {"command": {"const": "list"}, "sessions": {"type": "array", "items": SESSION_METADATA_SCHEMA}}
)
DETACH_RESULT_SCHEMA = _strict_object({"command": {"const": "detach"}, "sessionId": _ID})
COMMAND_RESULT_SCHEMA = {
    "anyOf": [
        LIST_RESULT_SCHEMA,
        CREATE_RESULT_SCHEMA,
        ATTACH_RESULT_SCHEMA,
        DETACH_RESULT_SCHEMA,
        PROMPT_RESULT_SCHEMA,
        STEER_RESULT_SCHEMA,
        ABORT_RESULT_SCHEMA,
        SET_MODEL_RESULT_SCHEMA,
        SET_THINKING_RESULT_SCHEMA,
    ]
}

# Must be the first frame sent by a client. Version is intentionally an
# integer, not a coercible string.
CLIENT_HELLO_SCHEMA = _strict_object({"type": {"const": "hello"}, "version": _NON_NEGATIVE_INTEGER})

REQUEST_ENVELOPE_SCHEMA = _strict_object({"type": {"const": "request"}, "id": _ID, "request": COMMAND_SCHEMA})
CLIENT_MESSAGE_SCHEMA = {
    "anyOf": [CLIENT_HELLO_SCHEMA, REQUEST_ENVELOPE_SCHEMA],
    "$defs": {"jsonValue": _JSON_VALUE_DEFINITION},
}

SERVER_EVENT_SCHEMA = {
    "anyOf": [
        _strict_object({"type": {"const": "server_snapshot"}, "snapshot": SERVER_SNAPSHOT_SCHEMA}),
        _strict_object({"type": {"const": "session_snapshot"}, "snapshot": SESSION_SNAPSHOT_SCHEMA}),
        _strict_object(
            {"type": {"const": "session_progress"}, "sessionId": _ID, "progress": TRANSCRIPT_PROGRESS_SCHEMA}
        ),
        _strict_object({"type": {"const": "session_removed"}, "sessionId": _ID}),
    ]
}

SERVER_HELLO_SCHEMA = _strict_object(
    {
        "type": {"const": "hello"},
        "version": {"const": PROTOCOL_VERSION},
        "connectionId": _ID,
        "snapshot": SERVER_SNAPSHOT_SCHEMA,
    }
)
SERVER_HELLO_ERROR_SCHEMA = _strict_object({"type": {"const": "hello_error"}, "error": PROTOCOL_ERROR_SCHEMA})
RESPONSE_ENVELOPE_SCHEMA = {
    "anyOf": [
        _strict_object(
            {"type": {"const": "response"}, "id": _ID, "ok": {"const": True}, "result": COMMAND_RESULT_SCHEMA}
        ),
        _strict_object(
            {"type": {"const": "response"}, "id": _ID, "ok": {"const": False}, "error": PROTOCOL_ERROR_SCHEMA}
        ),
    ]
}
EVENT_ENVELOPE_SCHEMA = _strict_object({"type": {"const": "event"}, "event": SERVER_EVENT_SCHEMA})
SERVER_MESSAGE_SCHEMA = {
    "anyOf": [
        SERVER_HELLO_SCHEMA,
        SERVER_HELLO_ERROR_SCHEMA,
        RESPONSE_ENVELOPE_SCHEMA,
        EVENT_ENVELOPE_SCHEMA,
    ],
    "$defs": {"jsonValue": _JSON_VALUE_DEFINITION},
}

# Validated wire messages stay plain camelCase dicts.
type ClientMessage = dict[str, Any]
type ServerMessage = dict[str, Any]
type Command = dict[str, Any]
type CommandResult = dict[str, Any]
type ServerEvent = dict[str, Any]
type TranscriptItem = dict[str, Any]
type TranscriptProgress = dict[str, Any]
type SessionMetadata = dict[str, Any]
type SessionSnapshot = dict[str, Any]
type ServerSnapshot = dict[str, Any]
type ModelRef = dict[str, Any]
type ModelMetadata = dict[str, Any]
type ProtocolError = dict[str, Any]
type Usage = dict[str, Any]
