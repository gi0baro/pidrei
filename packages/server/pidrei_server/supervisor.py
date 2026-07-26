"""Mirror of pi server src/supervisor.ts.

Instance records are plain camelCase dicts (see types.py). None stands in
for JS undefined: merging updates drops None-valued keys, matching pi's
spread + JSON.stringify-drops-undefined behavior on persisted records.

pi registers, updates and disconnects each instance with its radius presence
service throughout this file, and exposes a coordinator back to it; the
integration was dropped in Phase 7 step 1, so none of that is here.
"""

import threading
import uuid
from typing import Any

import tonio.colored as tonio

from .rpc_process import RpcProcessInstance, RpcProcessOptions, create_rpc_process_instance
from .storage import get_instance, load_instances, remove_instance, save_instances, upsert_instance
from .types import InstanceRecord, InstanceStatus, timestamp


class _LiveInstanceResources:
    __slots__ = ("rpc_process", "session_id")

    def __init__(self) -> None:
        self.rpc_process: RpcProcessInstance | None = None
        self.session_id: str | None = None


class _LiveInstance:
    __slots__ = ("on_ui_request", "record", "resources", "subscribers", "unsubscribe_events", "unsubscribe_exit")

    def __init__(self, record: InstanceRecord):
        self.record = record
        self.resources = _LiveInstanceResources()
        self.subscribers: list[Any] = []
        self.on_ui_request: Any = None
        self.unsubscribe_events: Any = None
        self.unsubscribe_exit: Any = None


def _clone_instance(record: InstanceRecord) -> InstanceRecord:
    return dict(record)


def _merge_record(record: InstanceRecord, updates: dict[str, Any]) -> InstanceRecord:
    merged = {**record, **updates}
    return {key: value for key, value in merged.items() if value is not None}


# Only refresh persisted session metadata after commands that can plausibly change
# the instance identity/details we store in instances.json. Most RPCs mutate transient
# runtime state only, so forcing a follow-up get_state after every command is wasted IO.
#
# - new_session / switch_session / fork / clone can change sessionId/sessionFile
# - set_session_name changes a persisted session detail we may want reflected externally
# - prompt can materialize or advance persisted session state after the child processes it
SESSION_METADATA_COMMANDS = frozenset(
    [
        "new_session",
        "switch_session",
        "fork",
        "clone",
        "set_session_name",
        "prompt",
    ]
)


def _should_refresh_session_metadata(command: dict[str, Any]) -> bool:
    return command.get("type") in SESSION_METADATA_COMMANDS


def _is_get_state_success(response: dict[str, Any]) -> bool:
    return response.get("success") is True and response.get("command") == "get_state" and "data" in response


class _RpcStreamHandle:
    __slots__ = ("_live", "_on_event", "_on_ui_request", "_rpc_process", "_supervisor")

    def __init__(self, supervisor, live: _LiveInstance, rpc_process: RpcProcessInstance, on_event, on_ui_request):
        self._supervisor = supervisor
        self._live = live
        self._rpc_process = rpc_process
        self._on_event = on_event
        self._on_ui_request = on_ui_request

    async def handle_rpc(self, command: dict[str, Any]) -> dict[str, Any]:
        response = await self._rpc_process.send(command)
        if _should_refresh_session_metadata(command):
            await self._supervisor._sync_instance_record(self._live)
        return response

    async def handle_ui_response(self, response: dict[str, Any]) -> None:
        await self._rpc_process.handle_ui_response(response)

    def close(self) -> None:
        if self._live.on_ui_request is self._on_ui_request:
            self._live.on_ui_request = None
        try:
            self._live.subscribers.remove(self._on_event)
        except ValueError:
            pass


class ServerSupervisor:
    def __init__(self) -> None:
        self._live_instances: dict[str, _LiveInstance] = {}
        # pi's single thread serializes the unexpected-exit handler against
        # the spawn-failure/stop paths (microtask ordering); under tonio's
        # parallel workers the guarded check-then-set sections need this lock
        # so a failed spawn deterministically ends "stopped", never "error".
        self._status_guard = threading.Lock()

    def _set_status(self, live: _LiveInstance, status: InstanceStatus) -> None:
        live.record = _merge_record(live.record, {"status": status, "lastSeenAt": timestamp()})
        upsert_instance(live.record)

    def _update_record(self, live: _LiveInstance, updates: dict[str, Any]) -> None:
        live.record = _merge_record(live.record, {**updates, "lastSeenAt": timestamp()})
        if updates.get("sessionId") is not None:
            live.resources.session_id = updates["sessionId"]
        upsert_instance(live.record)

    def _clear_bindings(self, live: _LiveInstance) -> None:
        if live.unsubscribe_events is not None:
            live.unsubscribe_events()
        if live.unsubscribe_exit is not None:
            live.unsubscribe_exit()
        live.unsubscribe_events = None
        live.unsubscribe_exit = None
        live.on_ui_request = None
        if live.resources.rpc_process is not None:
            live.resources.rpc_process.set_ui_request_handler(None)

    def _bind_rpc_process(self, live: _LiveInstance, rpc_process: RpcProcessInstance) -> None:
        self._clear_bindings(live)
        live.resources.rpc_process = rpc_process

        def on_event(event: dict[str, Any]) -> None:
            for subscriber in list(live.subscribers):
                subscriber(event)

        def on_exit(error: Exception | None) -> None:
            tonio.spawn.without_tracking(self._handle_unexpected_rpc_exit(live, error))

        def on_ui_request(request: dict[str, Any]) -> None:
            if live.on_ui_request is not None:
                live.on_ui_request(request)

        live.unsubscribe_events = rpc_process.on_event(on_event)
        live.unsubscribe_exit = rpc_process.on_exit(on_exit)
        rpc_process.set_ui_request_handler(on_ui_request)

    async def _handle_unexpected_rpc_exit(self, live: _LiveInstance, _error: Exception | None = None) -> None:
        with self._status_guard:
            if self._live_instances.get(live.record["id"]) is not live:
                return
            if live.record.get("status") in ("stopping", "stopped"):
                return
            self._set_status(live, "error")
        self._clear_bindings(live)
        live.resources.rpc_process = None
        self._live_instances.pop(live.record["id"], None)

    def _get_rpc_process(self, live: _LiveInstance) -> RpcProcessInstance | None:
        return live.resources.rpc_process

    async def _sync_instance_record(self, live: _LiveInstance) -> None:
        rpc_process = self._get_rpc_process(live)
        if rpc_process is None:
            self._update_record(live, {})
            return
        response = await rpc_process.send({"type": "get_state"})
        if not _is_get_state_success(response):
            self._update_record(live, {})
            return
        data = response["data"]
        self._update_record(live, {"sessionId": data.get("sessionId"), "sessionFile": data.get("sessionFile")})

    async def _cleanup_acquired_resources(self, live: _LiveInstance) -> None:
        rpc_process = live.resources.rpc_process
        self._clear_bindings(live)
        live.resources.session_id = None
        if rpc_process is not None:
            live.resources.rpc_process = None
            await rpc_process.dispose()

    async def _fail_spawn(self, live: _LiveInstance, error: Exception) -> None:
        self._set_status(live, "error")
        try:
            await self._cleanup_acquired_resources(live)
        finally:
            with self._status_guard:
                self._set_status(live, "stopped")
                self._live_instances.pop(live.record["id"], None)
        raise error

    def update_instance(self, instance: InstanceRecord) -> None:
        live = self._live_instances.get(instance["id"])
        if live is not None:
            live.record = instance
            live.resources.session_id = instance.get("sessionId")
        upsert_instance(instance)

    def open_rpc_stream(self, instance_id: str, on_event: Any, on_ui_request: Any) -> _RpcStreamHandle | None:
        live = self._live_instances.get(instance_id)
        rpc_process = self._get_rpc_process(live) if live is not None else None
        if live is None or rpc_process is None:
            return None
        live.subscribers.append(on_event)
        live.on_ui_request = on_ui_request
        return _RpcStreamHandle(self, live, rpc_process, on_event, on_ui_request)

    def get_live_instance(self, instance_id: str) -> InstanceRecord | None:
        live = self._live_instances.get(instance_id)
        return _clone_instance(live.record) if live is not None else None

    def list_live_instances(self) -> list[InstanceRecord]:
        return [_clone_instance(live.record) for live in self._live_instances.values()]

    async def recover_after_restart(self) -> None:
        recovered_at = timestamp()
        instances = [
            _merge_record(
                instance,
                {
                    "status": "stopped" if instance.get("status") in ("online", "starting") else instance.get("status"),
                    "lastSeenAt": recovered_at,
                },
            )
            for instance in load_instances()
        ]
        save_instances(instances)

    def list_instances(self) -> list[InstanceRecord]:
        return [_clone_instance(instance) for instance in load_instances()]

    def get_instance(self, instance_id: str) -> InstanceRecord | None:
        live = self._live_instances.get(instance_id)
        if live is not None:
            return _clone_instance(live.record)
        stored = get_instance(instance_id)
        return _clone_instance(stored) if stored is not None else None

    async def spawn_instance(self, options: dict[str, Any]) -> InstanceRecord:
        now = timestamp()
        record: InstanceRecord = {
            "id": str(uuid.uuid4()),
            "status": "starting",
            "cwd": options["cwd"],
            "createdAt": now,
            "lastSeenAt": now,
        }
        if options.get("label") is not None:
            record["label"] = options["label"]
        live = _LiveInstance(record)
        self._live_instances[record["id"]] = live
        upsert_instance(live.record)

        try:
            rpc_process = await create_rpc_process_instance(
                RpcProcessOptions(cwd=options["cwd"], cli_path=options.get("cli_path"))
            )
            self._bind_rpc_process(live, rpc_process)
            await self._sync_instance_record(live)
            self._set_status(live, "online")
            return _clone_instance(live.record)
        except Exception as error:
            await self._fail_spawn(live, error)
            raise  # unreachable; _fail_spawn always raises

    async def stop_instance(self, instance_id: str) -> InstanceRecord | None:
        live = self._live_instances.get(instance_id)
        if live is None:
            return None

        self._set_status(live, "stopping")
        try:
            await self._cleanup_acquired_resources(live)
        finally:
            live.record = _merge_record(live.record, {"status": "stopped", "lastSeenAt": timestamp()})
            self._live_instances.pop(instance_id, None)
            remove_instance(instance_id)
        return _clone_instance(live.record)

    async def handle_rpc(self, instance_id: str, command: dict[str, Any]) -> dict[str, Any] | None:
        live = self._live_instances.get(instance_id)
        rpc_process = self._get_rpc_process(live) if live is not None else None
        if live is None or rpc_process is None:
            return None

        response = await rpc_process.send(command)
        if _should_refresh_session_metadata(command):
            await self._sync_instance_record(live)
        return response

    async def shutdown(self) -> None:
        for instance_id in list(self._live_instances.keys()):
            await self.stop_instance(instance_id)


supervisor = ServerSupervisor()
