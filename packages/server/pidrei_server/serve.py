"""Mirror of pi server src/serve.ts.

pi parks forever on a never-resolving promise and exits from its signal
handlers; here the serve loop waits on a tonio signal receiver and returns
the exit code to the CLI. node's uncaughtException/unhandledRejection
global handlers have no direct equivalent — exceptions propagate to the
CLI entry (a crashed background task is reported by the runtime).
"""

import os
import signal as signal_module

from tonio.colored import signals as tonio_signals

from .config import get_socket_path
from .handler import handle_ipc_request, open_rpc_stream
from .ipc.server import IpcRequestHandler, start_ipc_server
from .supervisor import supervisor


async def serve() -> int:
    socket_path = get_socket_path()
    os.makedirs(os.path.dirname(socket_path), exist_ok=True)
    server = await start_ipc_server(
        IpcRequestHandler(
            handle_request=handle_ipc_request,
            open_rpc_stream=open_rpc_stream,
        )
    )

    try:
        # pi starts its radius presence heartbeat here; the integration was
        # dropped in Phase 7 step 1.
        await supervisor.recover_after_restart()
    except BaseException:
        server.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        raise

    print(f"server listening on {socket_path}", flush=True)

    # Keep serving until a signal triggers shutdown.
    with tonio_signals.signal_receiver(signal_module.SIGINT, signal_module.SIGTERM) as receiver:
        async for _sig in receiver:
            break

    server.close()
    await supervisor.shutdown()
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    return 0
