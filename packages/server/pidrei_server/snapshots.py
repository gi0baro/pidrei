"""Server snapshot revisions and broadcast serialization (port of pi server `snapshots.ts`).

`broadcast()` replaces pi's promise-chain queue with a lock: revision bump and
delivery happen under it, so revisions are delivered in monotonic order even
though every caller voids the returned awaitable.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import tonio.colored as tonio
from tonio.colored import sync

from pidrei_ai.utils.tasks import gather
from pidrei_protocol import PROTOCOL_VERSION, ModelMetadata, ServerMessage, ServerSnapshot, SessionMetadata

from .connection import ConnectionState
from .promise import Deferred
from .types import PiServerService


@dataclass(slots=True, frozen=True)
class ServerSnapshotPublisherOptions:
    server_id: str
    service: PiServerService
    connections: set[ConnectionState]
    is_closing: Callable[[], bool]
    list_sessions: Callable[[], Awaitable[list[SessionMetadata]]]
    send_message: Callable[[ConnectionState, ServerMessage], Awaitable[bool]]
    report_error: Callable[[object], None]


class ServerSnapshotPublisher:
    def __init__(self, options: ServerSnapshotPublisherOptions) -> None:
        self._options = options
        self._revision = 0
        self._broadcast_lock = sync.Lock()

    @property
    def current_revision(self) -> int:
        return self._revision

    async def get(self, models: list[ModelMetadata] | None = None) -> ServerSnapshot:
        return {
            "serverId": self._options.server_id,
            "protocolVersion": PROTOCOL_VERSION,
            "revision": self._revision,
            "sessions": await self._options.list_sessions(),
            "models": models if models is not None else await self._options.service.list_models(),
        }

    def broadcast(self) -> Awaitable[None]:
        # pi chains broadcasts on a promise tail so revisions go out in order.
        # Swapping a tail field is not atomic across worker threads; a lock
        # around "bump revision + send" gives the same ordering guarantee.
        result = Deferred()

        async def _run() -> None:
            async with self._broadcast_lock:
                try:
                    await self._perform_broadcast()
                except BaseException as error:
                    # BaseException: `result` must settle — an unsettled
                    # broadcast wedges any caller that awaits it.
                    self._options.report_error(error)
                    result.reject(error)
                    if isinstance(error, GeneratorExit):
                        raise
                    return
            result.resolve(None)

        tonio.spawn.without_tracking(_run())
        return result

    async def _perform_broadcast(self) -> None:
        ready_connections = [
            connection
            for connection in self._options.connections
            if connection.stage == "ready" and not connection.disconnected
        ]
        if not ready_connections or self._options.is_closing():
            return
        self._revision += 1
        revision = self._revision
        models = await self._options.service.list_models()
        current = await self.get(models)
        snapshot: ServerSnapshot = {**current, "revision": revision}
        envelope: ServerMessage = {"type": "event", "event": {"type": "server_snapshot", "snapshot": snapshot}}
        # Fan out so one slow client's flush does not delay the others; each
        # connection's writer queue keeps its own message order.
        await gather(*(self._options.send_message(connection, envelope) for connection in ready_connections))
