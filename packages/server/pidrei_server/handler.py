"""Mirror of pi server src/handler.ts."""

from typing import Any

from .supervisor import supervisor
from .types import InstanceRecord


def to_instance_summary(instance: InstanceRecord) -> dict[str, Any]:
    # Optional fields are included only when present, matching JSON.stringify
    # dropping undefined properties on the wire.
    summary = {
        "id": instance["id"],
        "status": instance["status"],
        "cwd": instance["cwd"],
    }
    # pi also carries "radiusPiId"; nothing sets it since radius was dropped.
    for key in ("label", "sessionId", "sessionFile"):
        if instance.get(key) is not None:
            summary[key] = instance[key]
    return summary


def _unknown_instance_error(instance_id: str) -> dict[str, Any]:
    return {
        "type": "error",
        "ok": False,
        "error": f"Unknown instance: {instance_id}",
    }


async def handle_ipc_request(request: dict[str, Any]) -> dict[str, Any]:
    request_type = request.get("type")

    if request_type == "spawn":
        instance = await supervisor.spawn_instance(
            {
                "cwd": request["cwd"],
                "label": request.get("label"),
            }
        )
        return {
            "type": "spawn_result",
            "ok": True,
            "instance": to_instance_summary(instance),
        }

    if request_type == "list":
        return {
            "type": "list_result",
            "ok": True,
            "instances": [to_instance_summary(instance) for instance in await supervisor.list_instances()],
        }

    if request_type == "status":
        instance = await supervisor.get_instance(request["instanceId"])
        if instance is None:
            return _unknown_instance_error(request["instanceId"])

        return {
            "type": "status_result",
            "ok": True,
            "instance": to_instance_summary(instance),
        }

    if request_type == "stop":
        instance = await supervisor.stop_instance(request["instanceId"])
        if instance is None:
            return _unknown_instance_error(request["instanceId"])

        return {
            "type": "stop_result",
            "ok": True,
            "instanceId": request["instanceId"],
        }

    if request_type == "rpc":
        response = await supervisor.handle_rpc(request["instanceId"], request["command"])
        if response is None:
            return _unknown_instance_error(request["instanceId"])

        return {
            "type": "rpc_result",
            "ok": True,
            "response": response,
        }

    if request_type == "rpc_stream":
        instance = await supervisor.get_instance(request["instanceId"])
        if instance is None:
            return _unknown_instance_error(request["instanceId"])
        return {
            "type": "rpc_ready",
            "ok": True,
            "instance": to_instance_summary(instance),
        }

    # pi's switch is exhaustive over the compile-time request union; an
    # unknown type at runtime maps to an error response here.
    return {"type": "error", "ok": False, "error": f"Unknown request: {request_type}"}


class _OpenedRpcStream:
    __slots__ = ("_handle", "_on_response")

    def __init__(self, handle: Any, on_response: Any):
        self._handle = handle
        self._on_response = on_response

    async def handle_request(self, request: dict[str, Any]) -> None:
        if request.get("type") == "extension_ui_response":
            await self._handle.handle_ui_response(request)
            return
        response = await self._handle.handle_rpc(request)
        self._on_response(response)

    def close(self) -> None:
        self._handle.close()


def open_rpc_stream(instance_id: str, on_response: Any, on_session_event: Any, on_ui_request: Any) -> Any:
    handle = supervisor.open_rpc_stream(instance_id, on_session_event, on_ui_request)
    if handle is None:
        return None

    return _OpenedRpcStream(handle, on_response)
