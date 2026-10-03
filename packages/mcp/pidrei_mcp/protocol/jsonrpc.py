"""Mirror of pi mcp src/protocol/jsonrpc.ts.

Messages stay the JSON objects they are on the wire (`dict`s, typed as
`TypedDict`s), validated by hand as pi does. JavaScript's `typeof` checks are
ported literally: a JSON number is an `int` or a `float`, never a `bool`.
"""

import json
import math
from typing import Any, NotRequired, TypedDict

from pidrei_utils.cancel import AbortError


type JsonRpcId = str | int | float


class JsonRpcRequest(TypedDict):
    jsonrpc: str
    id: JsonRpcId
    method: str
    params: NotRequired[Any]


class JsonRpcNotification(TypedDict):
    jsonrpc: str
    method: str
    params: NotRequired[Any]


class JsonRpcErrorObject(TypedDict):
    code: int
    message: str
    data: NotRequired[Any]


class JsonRpcSuccessResponse(TypedDict):
    jsonrpc: str
    id: JsonRpcId
    result: Any


class JsonRpcErrorResponse(TypedDict):
    jsonrpc: str
    id: JsonRpcId
    error: JsonRpcErrorObject


type JsonRpcResponse = JsonRpcSuccessResponse | JsonRpcErrorResponse
type JsonRpcMessage = JsonRpcRequest | JsonRpcNotification | JsonRpcResponse


class _JsonRpcErrorCodes:
    __slots__ = ()
    parse_error = -32700
    invalid_request = -32600
    method_not_found = -32601
    invalid_params = -32602
    internal_error = -32603


JSON_RPC_ERROR_CODES = _JsonRpcErrorCodes()


class McpError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class McpConnectionClosedError(Exception):
    def __init__(self, message: str = "MCP connection closed") -> None:
        super().__init__(message)


class McpTimeoutError(Exception):
    def __init__(self, timeout_ms: float) -> None:
        super().__init__(f"MCP request timed out after {_format_ms(timeout_ms)}ms")
        self.timeout_ms = timeout_ms


class McpAbortError(AbortError):
    """pi names it `AbortError`, so code that recognizes an abort recognizes
    this one: here it is one."""

    def __init__(self, message: str = "MCP request aborted") -> None:
        super().__init__(message)


def _format_ms(value: float) -> str:
    return js_string(value)


def js_string(value: Any) -> str:
    """`String(value)` for a JSON-RPC id or a number: JavaScript prints an
    integral number without a fraction (`5`, not `5.0`)."""
    if is_number(value) and math.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"Unexpected token {name} in JSON")


def parse_json(text: str | bytes) -> Any:
    """`JSON.parse`: unlike `json.loads`, `NaN` and `Infinity` are not JSON."""
    return json.loads(text, parse_constant=_reject_constant)


def stringify(value: Any) -> str:
    """`JSON.stringify(value)`: compact, non-ASCII kept."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def is_object(value: Any) -> bool:
    """`typeof value === "object" && value !== null && !Array.isArray(value)`
    for parsed JSON."""
    return isinstance(value, dict)


def is_number(value: Any) -> bool:
    """`typeof value === "number"` for parsed JSON."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def is_string(value: Any) -> bool:
    return isinstance(value, str)


def is_json_rpc_id(value: Any) -> bool:
    return isinstance(value, str) or (is_number(value) and math.isfinite(value))


def is_json_rpc_request(message: Any) -> bool:
    return (
        is_object(message)
        and message.get("jsonrpc") == "2.0"
        and is_json_rpc_id(message.get("id"))
        and isinstance(message.get("method"), str)
    )


def is_json_rpc_notification(message: Any) -> bool:
    return (
        is_object(message)
        and message.get("jsonrpc") == "2.0"
        and "id" not in message
        and isinstance(message.get("method"), str)
    )


def is_json_rpc_response(message: Any) -> bool:
    if not is_object(message) or message.get("jsonrpc") != "2.0" or not is_json_rpc_id(message.get("id")):
        return False
    if "result" in message:
        return "error" not in message
    if "error" not in message or not is_object(message["error"]):
        return False
    error = message["error"]
    return is_number(error.get("code")) and isinstance(error.get("message"), str)


def parse_json_rpc_message(value: Any) -> JsonRpcMessage:
    if is_json_rpc_request(value) or is_json_rpc_notification(value) or is_json_rpc_response(value):
        return value
    raise McpError(JSON_RPC_ERROR_CODES.invalid_request, "Invalid JSON-RPC message")
