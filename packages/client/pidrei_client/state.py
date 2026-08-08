"""Client-side snapshot/event state with four listener registries (port of pi client `state.ts`).

Listeners are synchronous callables, mirroring pi: they run inline during
message dispatch, and failures are reported through `on_listener_error`
without corrupting client state.
"""

from collections.abc import Callable, Iterable

from pidrei_protocol import CommandResult, ServerEvent, ServerSnapshot, SessionSnapshot

from .types import ListenerErrorHandler, Unsubscribe


class ClientState:
    def __init__(self, on_listener_error: ListenerErrorHandler | None = None) -> None:
        self._session_snapshots: dict[str, SessionSnapshot] = {}
        self._attached_session_ids: set[str] = set()
        self._snapshot_listeners: set[Callable[[ServerSnapshot], None]] = set()
        self._event_listeners: set[Callable[[ServerEvent], None]] = set()
        self._session_snapshot_listeners: dict[str, set[Callable[[SessionSnapshot], None]]] = {}
        self._session_event_listeners: dict[str, set[Callable[[ServerEvent], None]]] = {}
        self._on_listener_error = on_listener_error
        self._snapshot: ServerSnapshot | None = None

    @property
    def snapshot(self) -> ServerSnapshot | None:
        return self._snapshot

    def reset(self) -> None:
        self._snapshot = None
        self._session_snapshots.clear()
        self._attached_session_ids.clear()

    def clear_attachments(self) -> None:
        self._attached_session_ids.clear()

    def dispose(self) -> None:
        self.reset()
        self._snapshot_listeners.clear()
        self._event_listeners.clear()
        self._session_snapshot_listeners.clear()
        self._session_event_listeners.clear()

    def get_session_snapshot(self, session_id: str) -> SessionSnapshot | None:
        return self._session_snapshots.get(session_id)

    def is_session_attached(self, session_id: str) -> bool:
        return session_id in self._attached_session_ids

    def forget_session_snapshot(self, session_id: str) -> SessionSnapshot | None:
        return self._session_snapshots.pop(session_id, None)

    def restore_session_snapshot(self, snapshot: SessionSnapshot) -> None:
        if snapshot["id"] not in self._session_snapshots:
            self._session_snapshots[snapshot["id"]] = snapshot

    def subscribe(self, listener: Callable[[ServerSnapshot], None]) -> Unsubscribe:
        self._snapshot_listeners.add(listener)
        return lambda: self._snapshot_listeners.discard(listener)

    def on_event(self, listener: Callable[[ServerEvent], None]) -> Unsubscribe:
        self._event_listeners.add(listener)
        return lambda: self._event_listeners.discard(listener)

    def subscribe_session(self, session_id: str, listener: Callable[[SessionSnapshot], None]) -> Unsubscribe:
        return _add_mapped_listener(self._session_snapshot_listeners, session_id, listener)

    def on_session_event(self, session_id: str, listener: Callable[[ServerEvent], None]) -> Unsubscribe:
        return _add_mapped_listener(self._session_event_listeners, session_id, listener)

    def apply_result(self, result: CommandResult) -> None:
        if result["command"] == "list":
            return
        if result["command"] == "detach":
            session_id = result["sessionId"]
            self._attached_session_ids.discard(session_id)
            snapshot = self._session_snapshots.get(session_id)
            if snapshot is not None:
                self._apply_session_snapshot({**snapshot, "attached": False}, force=True)
            return
        self._apply_session_snapshot(result["session"])

    def apply_event(self, event: ServerEvent) -> None:
        if event["type"] == "server_snapshot":
            self.apply_server_snapshot(event["snapshot"])
        if event["type"] == "session_snapshot":
            self._apply_session_snapshot(event["snapshot"])
        if event["type"] == "session_removed":
            self._session_snapshots.pop(event["sessionId"], None)
            self._attached_session_ids.discard(event["sessionId"])
        self._notify(self._event_listeners, event)
        session_id = _get_event_session_id(event)
        if session_id is not None:
            self._notify(self._session_event_listeners.get(session_id), event)

    def apply_server_snapshot(self, snapshot: ServerSnapshot) -> None:
        if self._snapshot is not None and snapshot["revision"] < self._snapshot["revision"]:
            return
        self._snapshot = snapshot
        self._notify(self._snapshot_listeners, snapshot)

    def _apply_session_snapshot(self, snapshot: SessionSnapshot, force: bool = False) -> None:
        current = self._session_snapshots.get(snapshot["id"])
        if not force and current is not None and snapshot["revision"] < current["revision"]:
            return
        self._session_snapshots[snapshot["id"]] = snapshot
        if snapshot["attached"]:
            self._attached_session_ids.add(snapshot["id"])
        else:
            self._attached_session_ids.discard(snapshot["id"])
        self._notify(self._session_snapshot_listeners.get(snapshot["id"]), snapshot)

    def _notify(self, listeners: Iterable[Callable[..., None]] | None, value: object) -> None:
        for listener in list(listeners) if listeners is not None else []:
            try:
                listener(value)
            except Exception as error:
                self._report_listener_error(error)

    def _report_listener_error(self, error: Exception) -> None:
        if self._on_listener_error is None:
            return
        try:
            self._on_listener_error(error)
        except Exception:
            # Diagnostics cannot affect client state.
            pass


def _add_mapped_listener(
    listeners_by_id: dict[str, set[Callable[..., None]]],
    listener_id: str,
    listener: Callable[..., None],
) -> Unsubscribe:
    listeners = listeners_by_id.get(listener_id)
    if listeners is None:
        listeners = set()
        listeners_by_id[listener_id] = listeners
    listeners.add(listener)

    def unsubscribe() -> None:
        listeners.discard(listener)
        if not listeners and listeners_by_id.get(listener_id) is listeners:
            del listeners_by_id[listener_id]

    return unsubscribe


def _get_event_session_id(event: ServerEvent) -> str | None:
    if event["type"] == "session_snapshot":
        return event["snapshot"]["id"]
    if event["type"] in ("session_progress", "session_removed"):
        return event["sessionId"]
    return None
