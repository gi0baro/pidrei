"""Mirror of pi mcp src/client.ts.

The client owns request correlation, initialization, timeouts, cancellation,
server requests and the protocol-level helpers; the transport owns framing
and I/O.

What a server sends reaches the client one message at a time, in arrival
order (see `transports/transport.py`), and the client awaits its own
listeners for each message before the next: notification, progress and
close listeners are async and awaited in registration order. A listener
must not wait for a response from its own client, since the delivery that
would bring it is the one running the listener: start such work detached,
as pi's extension does for its tool refresh.

A request's pending entry is settled at most once, by whichever comes
first: its response, its timeout, its cancel token, a failed send, or the
close. Its message is queued on the transport under the client's lock, in
the same step as registering the entry, and a `notifications/cancelled` is
queued only after its request was claimed, so a cancellation never goes out
before the request it cancels.
"""

import math
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import tonio.colored as tonio
from tonio.colored.exceptions import CancelledError

from pidrei_utils import timers
from pidrei_utils.cancel import AbortError, CancelToken

from .protocol.content import CallToolResult
from .protocol.jsonrpc import (
    JSON_RPC_ERROR_CODES,
    JsonRpcId,
    JsonRpcMessage,
    McpAbortError,
    McpConnectionClosedError,
    McpError,
    McpTimeoutError,
    is_json_rpc_id,
    is_json_rpc_notification,
    is_json_rpc_request,
    is_json_rpc_response,
    is_number,
    is_object,
    js_string,
)
from .protocol.types import (
    LATEST_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    ClientCapabilities,
    Implementation,
    InitializeResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ProgressNotification,
    ReadResourceResult,
    Resource,
    ResourceTemplate,
    Root,
    ServerCapabilities,
    Tool,
)
from .transports.transport import McpTransport, SendResult


DEFAULT_REQUEST_TIMEOUT_MS = 30_000
MAX_LIST_PAGES = 1_000

type ClientState = Literal["idle", "connecting", "connected", "closed"]
type NotificationListener = Callable[[Any], Awaitable[None]]
type ErrorListener = Callable[[Exception], Awaitable[None]]
type CloseListener = Callable[[], Awaitable[None]]
type ProgressListener = Callable[[ProgressNotification], Awaitable[None]]
type Roots = Sequence[Root] | Callable[[], Awaitable[Sequence[Root]]]


@dataclass(frozen=True, slots=True)
class RequestContext:
    """What a handler of a server request gets besides the params: a token
    cancelled when the server sends `notifications/cancelled` for the
    request, or the connection closes."""

    cancel: CancelToken


type RequestHandler = Callable[[Any, RequestContext], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class McpClientOptions:
    name: str
    version: str
    title: str | None = None
    capabilities: ClientCapabilities | None = None
    protocol_version: str | None = None
    request_timeout_ms: float | None = None
    roots: Roots | None = None


@dataclass(frozen=True, slots=True)
class _Session:
    protocol_version: str
    server_info: Implementation
    server_capabilities: ServerCapabilities
    instructions: str | None


@dataclass(frozen=True, slots=True)
class _View:
    """The connection state and what initialization learned, published as one
    value so a reader never sees one without the other."""

    state: ClientState
    session: _Session | None


class _Pending:
    __slots__ = (
        "cancellable",
        "generation",
        "on_progress",
        "outcome",
        "progress_token",
        "settled",
        "timeout_ms",
        "timer",
        "unsubscribe",
    )

    def __init__(
        self,
        timeout_ms: float,
        cancellable: bool,
        on_progress: ProgressListener | None,
        progress_token: JsonRpcId | None,
    ) -> None:
        self.timeout_ms = timeout_ms
        self.cancellable = cancellable
        self.on_progress = on_progress
        self.progress_token = progress_token
        self.timer: timers.Timeout | None = None
        # Identifies the armed timer: a fire that finds another one was
        # superseded by a progress renewal (a fire that checked just before
        # the renewal's cancel still runs).
        self.generation: object | None = None
        self.unsubscribe: Callable[[], None] | None = None
        self.settled = tonio.Event()
        self.outcome: tuple[bool, Any] = (False, None)


def validate_initialize_result(value: Any) -> InitializeResult:
    if (
        not is_object(value)
        or not isinstance(value.get("protocolVersion"), str)
        or not is_object(value.get("capabilities"))
        or not is_object(value.get("serverInfo"))
        or not isinstance(value["serverInfo"].get("name"), str)
        or not isinstance(value["serverInfo"].get("version"), str)
        or ("instructions" in value and not isinstance(value["instructions"], str))
    ):
        raise McpError(JSON_RPC_ERROR_CODES.invalid_request, "Invalid MCP initialize result")
    return value


def _invalid(message: str) -> McpError:
    return McpError(JSON_RPC_ERROR_CODES.invalid_request, message)


def _validate_list_page(
    method: str, key: str, value: Any, is_item: Callable[[dict[str, Any]], bool]
) -> tuple[list[dict[str, Any]], str | None]:
    """One page of a paginated list: the items under `key`, each checked by
    `is_item`, and the next cursor."""
    items = value.get(key) if is_object(value) else None
    if not is_object(value) or not isinstance(items, list):
        raise _invalid(f"Invalid MCP {method} result")
    for item in items:
        if not is_object(item) or not is_item(item):
            raise _invalid(f"Invalid entry in MCP {method} result")
    # Some servers end pagination with `null` or `""` instead of omitting the cursor.
    next_cursor = value.get("nextCursor")
    if next_cursor == "":
        next_cursor = None
    if next_cursor is not None and not isinstance(next_cursor, str):
        raise _invalid(f"Invalid MCP {method} cursor")
    return items, next_cursor


def _is_tool(tool: dict[str, Any]) -> bool:
    return isinstance(tool.get("name"), str) and is_object(tool.get("inputSchema"))


# `name` is required by the spec, but some servers omit it; the URI stands in.
def _is_resource(resource: dict[str, Any]) -> bool:
    return isinstance(resource.get("uri"), str) and ("name" not in resource or isinstance(resource["name"], str))


def _is_resource_template(template: dict[str, Any]) -> bool:
    return isinstance(template.get("uriTemplate"), str) and (
        "name" not in template or isinstance(template["name"], str)
    )


def _to_resource(item: dict[str, Any]) -> Resource:
    return {**item, "name": item.get("name", item["uri"])}


def _to_resource_template(item: dict[str, Any]) -> ResourceTemplate:
    return {**item, "name": item.get("name", item["uriTemplate"])}


def _validate_read_resource_result(value: Any) -> ReadResourceResult:
    if not is_object(value) or not isinstance(value.get("contents"), list):
        raise _invalid("Invalid MCP resources/read result")
    for contents in value["contents"]:
        if (
            not is_object(contents)
            or not isinstance(contents.get("uri"), str)
            or (not isinstance(contents.get("text"), str) and not isinstance(contents.get("blob"), str))
        ):
            raise _invalid("Invalid contents in MCP resources/read result")
    return value


def _validate_call_tool_result(value: Any) -> CallToolResult:
    """`content` is required by the spec, but servers that only return
    `structuredContent` omit it (the SDK defaults it too)."""
    if not is_object(value) or ("content" in value and not isinstance(value["content"], list)):
        raise McpError(JSON_RPC_ERROR_CODES.invalid_request, "Invalid MCP tools/call result")
    if "structuredContent" in value and not is_object(value["structuredContent"]):
        raise McpError(JSON_RPC_ERROR_CODES.invalid_request, "Invalid MCP tools/call structured content")
    return {**value, "content": []} if "content" not in value else value


class McpClient:
    def __init__(
        self,
        *,
        name: str,
        version: str,
        title: str | None = None,
        capabilities: ClientCapabilities | None = None,
        protocol_version: str | None = None,
        request_timeout_ms: float | None = None,
        roots: Roots | None = None,
    ) -> None:
        self.options = McpClientOptions(
            name=name,
            version=version,
            title=title,
            capabilities=capabilities,
            protocol_version=protocol_version,
            request_timeout_ms=request_timeout_ms,
            roots=roots,
        )
        self._lock = threading.Lock()
        self._view = _View("idle", None)
        self._transport: McpTransport | None = None
        self._next_request_id = 1
        self._pending: dict[JsonRpcId, _Pending] = {}
        self._progress_requests: dict[JsonRpcId, JsonRpcId] = {}
        self._incoming: dict[JsonRpcId, CancelToken] = {}
        self._request_handlers: dict[str, RequestHandler] = {"ping": _answer_ping}
        # Insertion-ordered sets, as pi's `Set`s.
        self._notification_listeners: dict[str, dict[NotificationListener, None]] = {}
        self._error_listeners: dict[ErrorListener, None] = {}
        self._close_listeners: dict[CloseListener, None] = {}
        self._disposers: list[Callable[[], None]] = []
        if roots is not None:
            self._request_handlers["roots/list"] = self._answer_roots

    @property
    def connection_state(self) -> ClientState:
        return self._view.state

    @property
    def server_info(self) -> Implementation | None:
        session = self._view.session
        return session.server_info if session is not None else None

    @property
    def server_capabilities(self) -> ServerCapabilities | None:
        session = self._view.session
        return session.server_capabilities if session is not None else None

    @property
    def instructions(self) -> str | None:
        session = self._view.session
        return session.instructions if session is not None else None

    @property
    def protocol_version(self) -> str | None:
        session = self._view.session
        return session.protocol_version if session is not None else None

    async def connect(self, transport: McpTransport) -> InitializeResult:
        with self._lock:
            if self._view.state != "idle":
                raise RuntimeError(f"Cannot connect MCP client in {self._view.state} state")
            self._view = _View("connecting", None)
            self._transport = transport
            self._disposers = [
                transport.on_message(self._handle_message),
                # Transport errors are reported only. Pending requests fail when the transport closes.
                transport.on_error(self._emit_error),
                transport.on_close(self._handle_transport_close),
            ]
        try:
            await transport.start()
            options = self.options
            capabilities: ClientCapabilities = dict(options.capabilities or {})  # type: ignore[assignment]
            if options.roots is not None and "roots" not in capabilities:
                capabilities["roots"] = {}
            client_info: dict[str, Any] = {"name": options.name, "version": options.version}
            if options.title is not None:
                client_info["title"] = options.title
            result = validate_initialize_result(
                await self._request_internal(
                    "initialize",
                    {
                        "protocolVersion": options.protocol_version or LATEST_PROTOCOL_VERSION,
                        "capabilities": capabilities,
                        "clientInfo": client_info,
                    },
                    allow_connecting=True,
                )
            )
            if result["protocolVersion"] not in SUPPORTED_PROTOCOL_VERSIONS:
                raise RuntimeError(f"MCP server selected unsupported protocol version {result['protocolVersion']}")
            session = _Session(
                protocol_version=result["protocolVersion"],
                server_info=result["serverInfo"],
                server_capabilities=result["capabilities"],
                instructions=result.get("instructions"),
            )
            with self._lock:
                if self._view.state == "connecting":
                    self._view = _View("connecting", session)
            transport.set_protocol_version(result["protocolVersion"])
            await self._notify_internal("notifications/initialized", None, allow_connecting=True)
            with self._lock:
                # The transport may have dropped since the notification went out.
                if self._view.state != "connecting":
                    raise McpConnectionClosedError(f"MCP client is {self._view.state}")
                self._view = _View("connected", session)
            return result
        except CancelledError:
            # Nothing can be awaited on a cancelled chain: the close runs detached.
            tonio.spawn.without_tracking(self._close_quietly())
            raise
        except Exception:
            await self._close_quietly()
            raise

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        cancel: CancelToken | None = None,
        timeout_ms: float | None = None,
        on_progress: ProgressListener | None = None,
    ) -> Awaitable[Any]:
        return self._request_internal(
            method, params, cancel=cancel, timeout_ms=timeout_ms, on_progress=on_progress, allow_connecting=False
        )

    def notify(self, method: str, params: dict[str, Any] | None = None) -> Awaitable[None]:
        return self._notify_internal(method, params, allow_connecting=False)

    def set_request_handler(self, method: str, handler: RequestHandler) -> Callable[[], None]:
        with self._lock:
            self._request_handlers[method] = handler

        def dispose() -> None:
            with self._lock:
                if self._request_handlers.get(method) is handler:
                    del self._request_handlers[method]

        return dispose

    def on_notification(self, method: str, listener: NotificationListener) -> Callable[[], None]:
        with self._lock:
            listeners = self._notification_listeners.setdefault(method, {})
            listeners[listener] = None

        def dispose() -> None:
            with self._lock:
                listeners.pop(listener, None)
                if not listeners and self._notification_listeners.get(method) is listeners:
                    del self._notification_listeners[method]

        return dispose

    def on_error(self, listener: ErrorListener) -> Callable[[], None]:
        with self._lock:
            self._error_listeners[listener] = None

        def dispose() -> None:
            with self._lock:
                self._error_listeners.pop(listener, None)

        return dispose

    def on_close(self, listener: CloseListener) -> Callable[[], None]:
        """Called once when the connection closes, whether the transport
        dropped or `close()` was called."""
        with self._lock:
            self._close_listeners[listener] = None

        def dispose() -> None:
            with self._lock:
                self._close_listeners.pop(listener, None)

        return dispose

    async def ping(self, *, cancel: CancelToken | None = None, timeout_ms: float | None = None) -> None:
        await self.request("ping", None, cancel=cancel, timeout_ms=timeout_ms)

    async def list_tools(self, *, cancel: CancelToken | None = None, timeout_ms: float | None = None) -> list[Tool]:
        return await self._list_all("tools/list", "tools", _is_tool, cancel, timeout_ms)  # type: ignore[return-value]

    async def list_resources(
        self, *, cancel: CancelToken | None = None, timeout_ms: float | None = None
    ) -> list[Resource]:
        """Every resource, following `nextCursor` through all pages."""
        items = await self._list_all("resources/list", "resources", _is_resource, cancel, timeout_ms)
        return [_to_resource(item) for item in items]

    async def list_resources_page(
        self, cursor: str | None = None, *, cancel: CancelToken | None = None, timeout_ms: float | None = None
    ) -> ListResourcesResult:
        """One page of resources, starting at `cursor`."""
        items, next_cursor = await self._list_page(
            "resources/list", "resources", _is_resource, cursor, cancel, timeout_ms
        )
        page: ListResourcesResult = {"resources": [_to_resource(item) for item in items]}
        if next_cursor is not None:
            page["nextCursor"] = next_cursor
        return page

    async def list_resource_templates(
        self, *, cancel: CancelToken | None = None, timeout_ms: float | None = None
    ) -> list[ResourceTemplate]:
        """Every resource template, following `nextCursor` through all pages."""
        items = await self._list_all(
            "resources/templates/list", "resourceTemplates", _is_resource_template, cancel, timeout_ms
        )
        return [_to_resource_template(item) for item in items]

    async def list_resource_templates_page(
        self, cursor: str | None = None, *, cancel: CancelToken | None = None, timeout_ms: float | None = None
    ) -> ListResourceTemplatesResult:
        """One page of resource templates, starting at `cursor`."""
        items, next_cursor = await self._list_page(
            "resources/templates/list", "resourceTemplates", _is_resource_template, cursor, cancel, timeout_ms
        )
        page: ListResourceTemplatesResult = {"resourceTemplates": [_to_resource_template(item) for item in items]}
        if next_cursor is not None:
            page["nextCursor"] = next_cursor
        return page

    async def read_resource(
        self, uri: str, *, cancel: CancelToken | None = None, timeout_ms: float | None = None
    ) -> ReadResourceResult:
        return _validate_read_resource_result(
            await self.request("resources/read", {"uri": uri}, cancel=cancel, timeout_ms=timeout_ms)
        )

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        *,
        cancel: CancelToken | None = None,
        timeout_ms: float | None = None,
        on_progress: ProgressListener | None = None,
    ) -> CallToolResult:
        params: dict[str, Any] = {"name": name}
        if args is not None:
            params["arguments"] = args
        return _validate_call_tool_result(
            await self.request("tools/call", params, cancel=cancel, timeout_ms=timeout_ms, on_progress=on_progress)
        )

    async def close(self) -> None:
        with self._lock:
            transport = self._transport
            self._transport = None
            disposers, self._disposers = self._disposers, []
        for dispose in disposers:
            dispose()
        await self._mark_closed(McpConnectionClosedError())
        if transport is not None:
            await transport.close()

    async def _close_quietly(self) -> None:
        try:
            await self.close()
        except Exception:
            pass

    async def _list_page(
        self,
        method: str,
        key: str,
        is_item: Callable[[dict[str, Any]], bool],
        cursor: str | None,
        cancel: CancelToken | None,
        timeout_ms: float | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        params = None if cursor is None else {"cursor": cursor}
        return _validate_list_page(
            method, key, await self.request(method, params, cancel=cancel, timeout_ms=timeout_ms), is_item
        )

    async def _list_all(
        self,
        method: str,
        key: str,
        is_item: Callable[[dict[str, Any]], bool],
        cancel: CancelToken | None,
        timeout_ms: float | None,
    ) -> list[dict[str, Any]]:
        """Every item of a paginated list method."""
        items: list[dict[str, Any]] = []
        cursors: set[str] = set()
        cursor: str | None = None
        for _page in range(MAX_LIST_PAGES):
            page_items, next_cursor = await self._list_page(method, key, is_item, cursor, cancel, timeout_ms)
            items.extend(page_items)
            if next_cursor is None:
                return items
            if next_cursor in cursors:
                raise RuntimeError(f"MCP {method} returned duplicate cursor: {next_cursor}")
            cursors.add(next_cursor)
            cursor = next_cursor
        raise RuntimeError(f"MCP {method} exceeded {MAX_LIST_PAGES} pages")

    async def _request_internal(
        self,
        method: str,
        params: dict[str, Any] | None,
        *,
        cancel: CancelToken | None = None,
        timeout_ms: float | None = None,
        on_progress: ProgressListener | None = None,
        allow_connecting: bool,
    ) -> Any:
        # The spec forbids cancelling `initialize`.
        cancellable = method != "initialize"
        with self._lock:
            transport = self._require_transport_locked(allow_connecting)
            if cancel is not None and cancel.cancelled:
                raise McpAbortError()
            request_id = self._next_request_id
            self._next_request_id += 1
            progress_token = request_id if on_progress is not None else None
            request_params = params
            if progress_token is not None:
                meta = params.get("_meta") if params is not None else None
                request_params = {
                    **(params or {}),
                    "_meta": {**(meta if is_object(meta) else {}), "progressToken": progress_token},
                }
            message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
            if request_params is not None:
                message["params"] = request_params
            if timeout_ms is None:
                timeout_ms = self.options.request_timeout_ms
            entry = _Pending(
                timeout_ms if timeout_ms is not None else DEFAULT_REQUEST_TIMEOUT_MS,
                cancellable,
                on_progress,
                progress_token,
            )
            self._pending[request_id] = entry
            if progress_token is not None:
                self._progress_requests[progress_token] = request_id
            self._arm_timeout_locked(request_id, entry)
            sent = transport.send(message)  # type: ignore[arg-type]

        if cancel is not None:
            # Outside the lock: a token that already fired runs this at once.
            unsubscribe: Callable[[], None] | None = cancel.on_cancel(
                lambda reason: self._cancel_pending(
                    request_id, McpAbortError(), notify_server=cancellable, reason=str(reason)
                )
            )
            with self._lock:
                if self._pending.get(request_id) is entry:
                    entry.unsubscribe, unsubscribe = unsubscribe, None
            if unsubscribe is not None:
                unsubscribe()

        if sent.done:
            if sent.error is not None:
                self._cancel_pending(request_id, sent.error, notify_server=False)
        else:
            tonio.spawn.without_tracking(self._watch_request_send(request_id, sent))

        try:
            await entry.settled.wait()
        finally:
            if not entry.settled.is_set():
                # The caller was cancelled while waiting: the request is
                # abandoned, so the server is told and the entry dropped
                # (synchronously, on a cancelled chain).
                self._cancel_pending(request_id, McpAbortError(), notify_server=cancellable)
        failed, value = entry.outcome
        if failed:
            raise value
        return value

    async def _watch_request_send(self, request_id: JsonRpcId, sent: SendResult) -> None:
        try:
            await sent
        except Exception as error:
            self._cancel_pending(request_id, error, notify_server=False)

    async def _notify_internal(self, method: str, params: dict[str, Any] | None, *, allow_connecting: bool) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        with self._lock:
            sent = self._require_transport_locked(allow_connecting).send(message)  # type: ignore[arg-type]
        await sent

    def _require_transport_locked(self, allow_connecting: bool) -> McpTransport:
        state = self._view.state
        if self._transport is not None and (state == "connected" or (allow_connecting and state == "connecting")):
            return self._transport
        raise McpConnectionClosedError(f"MCP client is {state}")

    async def _handle_message(self, message: JsonRpcMessage) -> None:
        if is_json_rpc_response(message):
            await self._handle_response(message)
            return
        if is_json_rpc_request(message):
            self._handle_request(message)
            return
        if is_json_rpc_notification(message):
            await self._handle_notification(message["method"], message.get("params"))
            return
        await self._emit_error(McpError(JSON_RPC_ERROR_CODES.invalid_request, "Received invalid JSON-RPC message"))

    async def _handle_response(self, message: Any) -> None:
        request_id = message["id"]
        with self._lock:
            entry = self._pending.get(request_id)
            if entry is not None:
                self._remove_pending_locked(request_id, entry)
        if entry is None:
            await self._emit_error(RuntimeError(f"Received response for unknown MCP request {js_string(request_id)}"))
            return
        if "error" in message:
            error = message["error"]
            _settle(entry, True, McpError(error["code"], error["message"], error.get("data")))
        else:
            _settle(entry, False, message["result"])

    def _handle_request(self, message: Any) -> None:
        """Runs the handler detached. The request's cancel token is registered
        here, before the next message is handled, so a `notifications/cancelled`
        right behind the request finds it."""
        with self._lock:
            transport = self._transport
            if transport is None:
                return
            handler = self._request_handlers.get(message["method"])
            token = None
            if handler is not None:
                token = CancelToken()
                self._incoming[message["id"]] = token
        if handler is None or token is None:
            tonio.spawn.without_tracking(
                self._send_or_report(
                    transport,
                    {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "error": {
                            "code": JSON_RPC_ERROR_CODES.method_not_found,
                            "message": f"Method not found: {message['method']}",
                        },
                    },
                )
            )
            return
        tonio.spawn.without_tracking(self._serve_request(transport, handler, token, message))

    async def _serve_request(
        self, transport: McpTransport, handler: RequestHandler, token: CancelToken, message: Any
    ) -> None:
        request_id = message["id"]
        try:
            result = await handler(message.get("params"), RequestContext(token))
            await transport.send({"jsonrpc": "2.0", "id": request_id, "result": {} if result is None else result})
        except Exception as error:
            if isinstance(error, McpError):
                response_error: dict[str, Any] = {"code": error.code, "message": error.message}
                if error.data is not None:
                    response_error["data"] = error.data
            else:
                response_error = {"code": JSON_RPC_ERROR_CODES.internal_error, "message": str(error)}
            await self._send_or_report(transport, {"jsonrpc": "2.0", "id": request_id, "error": response_error})
        finally:
            with self._lock:
                if self._incoming.get(request_id) is token:
                    del self._incoming[request_id]

    async def _send_or_report(self, transport: McpTransport, message: dict[str, Any]) -> None:
        try:
            await transport.send(message)  # type: ignore[arg-type]
        except Exception as error:
            await self._emit_error(error)

    async def _handle_notification(self, method: str, params: Any) -> None:
        if method == "notifications/progress":
            await self._handle_progress(params)
        elif method == "notifications/cancelled":
            self._handle_cancelled(params)
        with self._lock:
            listeners = list(self._notification_listeners.get(method, ()))
        for listener in listeners:
            try:
                await listener(params)
            except Exception as error:
                await self._emit_error(error)

    async def _handle_progress(self, params: Any) -> None:
        if (
            not is_object(params)
            or not is_json_rpc_id(params.get("progressToken"))
            or not is_number(params.get("progress"))
        ):
            return
        with self._lock:
            request_id = self._progress_requests.get(params["progressToken"])
            entry = self._pending.get(request_id) if request_id is not None else None
            if request_id is None or entry is None:
                return
            self._arm_timeout_locked(request_id, entry)
            on_progress = entry.on_progress
        if on_progress is None:
            return
        try:
            await on_progress(params)
        except Exception as error:
            await self._emit_error(error)

    def _handle_cancelled(self, params: Any) -> None:
        if not is_object(params) or not is_json_rpc_id(params.get("requestId")):
            return
        with self._lock:
            token = self._incoming.get(params["requestId"])
        if token is not None:
            reason = params.get("reason")
            token.cancel(AbortError(reason) if isinstance(reason, str) else None)

    def _arm_timeout_locked(self, request_id: JsonRpcId, entry: _Pending) -> None:
        if entry.timer is not None:
            entry.timer.cancel()
            entry.timer = None
        entry.generation = None
        if not math.isfinite(entry.timeout_ms) or entry.timeout_ms <= 0:
            return
        generation = object()
        entry.generation = generation
        entry.timer = timers.Timeout(
            entry.timeout_ms,
            lambda: self._cancel_pending(
                request_id,
                McpTimeoutError(entry.timeout_ms),
                notify_server=entry.cancellable,
                reason="Request timed out",
                generation=generation,
            ),
        )

    def _cancel_pending(
        self,
        request_id: JsonRpcId,
        error: Exception,
        *,
        notify_server: bool,
        reason: str | None = None,
        generation: object | None = None,
    ) -> None:
        """Fail a pending request (abort, timeout, failed send, abandoned
        caller) and, when asked, tell the server. Synchronous: it runs from
        timers, token callbacks and cancelled chains."""
        sent = None
        with self._lock:
            entry = self._pending.get(request_id)
            if entry is None or (generation is not None and entry.generation is not generation):
                return
            self._remove_pending_locked(request_id, entry)
            if notify_server and self._transport is not None:
                params: dict[str, Any] = {"requestId": request_id}
                if reason:
                    params["reason"] = reason
                sent = self._transport.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": params})
        _settle(entry, True, error)
        if sent is not None:
            self._report_send_failure(sent)

    def _report_send_failure(self, sent: SendResult) -> None:
        if sent.done:
            if sent.error is not None:
                tonio.spawn.without_tracking(self._emit_error(sent.error))
            return

        async def watch() -> None:
            try:
                await sent
            except Exception as error:
                await self._emit_error(error)

        tonio.spawn.without_tracking(watch())

    def _remove_pending_locked(self, request_id: JsonRpcId, entry: _Pending) -> None:
        del self._pending[request_id]
        if entry.timer is not None:
            entry.timer.cancel()
            entry.timer = None
        entry.generation = None
        if entry.progress_token is not None:
            self._progress_requests.pop(entry.progress_token, None)
        if entry.unsubscribe is not None:
            entry.unsubscribe()
            entry.unsubscribe = None

    async def _handle_transport_close(self) -> None:
        await self._mark_closed(McpConnectionClosedError())

    async def _mark_closed(self, error: Exception) -> None:
        """Idempotent: fails in-flight requests, cancels the server requests
        being served, and flips the state; the close listeners run once."""
        with self._lock:
            was_closed = self._view.state == "closed"
            self._view = _View("closed", self._view.session)
            pending = list(self._pending.items())
            for request_id, entry in pending:
                self._remove_pending_locked(request_id, entry)
            incoming = list(self._incoming.values())
            self._incoming.clear()
            listeners = [] if was_closed else list(self._close_listeners)
        for _request_id, entry in pending:
            _settle(entry, True, error)
        for token in incoming:
            token.cancel(error)
        for listener in listeners:
            try:
                await listener()
            except Exception as listener_error:
                await self._emit_error(listener_error)

    async def _emit_error(self, error: Exception) -> None:
        with self._lock:
            listeners = list(self._error_listeners)
        for listener in listeners:
            try:
                await listener(error)
            except Exception:
                # Nowhere left to report it (pi lets it escape as an uncaught exception).
                pass

    async def _answer_roots(self, _params: Any, _context: RequestContext) -> dict[str, Any]:
        roots = self.options.roots
        listed = await roots() if callable(roots) else roots
        return {"roots": list(listed or ())}


async def _answer_ping(_params: Any, _context: RequestContext) -> dict[str, Any]:
    return {}


def _settle(entry: _Pending, failed: bool, value: Any) -> None:
    """Called by whoever removed the entry from the pending map, so exactly
    once per entry."""
    entry.outcome = (failed, value)
    entry.settled.set()
