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
from .radius import get_radius_server_base_url, is_radius_enabled, radius_presence
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
        await supervisor.recover_after_restart()
        if is_radius_enabled():
            machine = await radius_presence.start()
            # console.log flushes immediately; Python block-buffers stdout on pipes
            print(f"radius integration enabled: {socket_path} -> {get_radius_server_base_url()}", flush=True)
            if machine is not None:
                print(f"radius machine id: {machine['id']}", flush=True)
        else:
            print(
                "radius integration disabled: login radius in ~/.pidrei/agent/auth.json or set RADIUS_API_KEY",
                flush=True,
            )
    except BaseException:
        server.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        raise

    print(f"server listening on {socket_path}", flush=True)

    # Keep serving until a signal triggers shutdown.
    receiver = tonio_signals.signal_receiver(signal_module.SIGINT, signal_module.SIGTERM)
    receiver.__enter__()
    try:
        async for _sig in receiver:
            break
    finally:
        try:
            receiver.__exit__(None, None, None)
        except AttributeError:
            # Workaround for a tonio bug: the colored signal_receiver's
            # __exit__ crashes cancelling its inner tasks (`_SpawnJoinCollect`
            # has no `throw`; see TONIO_BUGS.md). The signals are already
            # deregistered at that point, so suppressing is safe. Use the
            # plain `with` statement once fixed in tonio.
            pass

    server.close()
    await supervisor.shutdown()
    await radius_presence.stop()
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    return 0
