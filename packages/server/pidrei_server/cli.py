"""Mirror of pi server src/cli.ts."""

import json
import os
import sys
from typing import Any

import tonio.colored as tonio
from tonio.colored import net

from .config import VERSION, get_socket_path
from .ipc.client import send_ipc_request
from .ipc.protocol import encode_message
from .serve import serve


def _print_help() -> None:
    print(
        f"pidrei-server v{VERSION}\n\nUsage:\n  pidrei-server serve\n  pidrei-server list\n"
        "  pidrei-server spawn [--cwd <path>] [--label <label>]\n  pidrei-server status <instance-id>\n"
        "  pidrei-server stop <instance-id>\n  pidrei-server rpc <instance-id> <json-command>\n"
        "  pidrei-server rpc-stream <instance-id>\n  pidrei-server --help\n  pidrei-server --version\n\n"
        "RPC stream stdin expects JSONL RpcCommand or extension_ui_response messages."
    )


def _print_response(response: Any) -> None:
    print(json.dumps(response, indent=2))


def _get_flag_value(args: list[str], flag: str) -> str | None:
    try:
        index = args.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(args):
        return None
    return args[index + 1]


async def _rpc_stream(instance_id: str) -> int:
    stream = await net.open_unix_socket(get_socket_path())
    try:
        await stream.send_all(encode_message({"type": "rpc_stream", "instanceId": instance_id}).encode("utf-8"))
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1

    print(
        f"connected to rpc stream {instance_id}; send JSONL RpcCommand or extension_ui_response on stdin",
        file=sys.stderr,
    )

    async def pump_stdin() -> None:
        try:
            while True:
                line = await tonio.spawn_blocking(sys.stdin.readline)
                if not line:
                    return
                stripped = line.strip()
                if not stripped:
                    continue
                parsed = json.loads(stripped)
                await stream.send_all(encode_message(parsed).encode("utf-8"))
        except Exception:
            pass

    tonio.spawn.without_tracking(pump_stdin())

    try:
        while True:
            chunk = await stream.receive_some()
            if not chunk:
                return 0
            sys.stdout.write(chunk.decode("utf-8", "replace"))
            sys.stdout.flush()
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        stream.close()


async def _main(argv: list[str]) -> int:
    args = list(argv)

    if len(args) == 0 or args[0] in ("--help", "-h"):
        _print_help()
        return 0

    if args[0] in ("--version", "-v"):
        print(VERSION)
        return 0

    if args[0] == "serve":
        return await serve()

    if args[0] == "list":
        _print_response(await send_ipc_request({"type": "list"}))
        return 0

    if args[0] == "spawn":
        spawn_cwd = _get_flag_value(args, "--cwd") or os.getcwd()
        label = _get_flag_value(args, "--label")
        request: dict[str, Any] = {"type": "spawn", "cwd": spawn_cwd}
        if label is not None:
            request["label"] = label
        _print_response(await send_ipc_request(request))
        return 0

    if args[0] == "status":
        if len(args) < 2:
            print("Usage: pidrei-server status <instance-id>", file=sys.stderr)
            return 1
        _print_response(await send_ipc_request({"type": "status", "instanceId": args[1]}))
        return 0

    if args[0] == "stop":
        if len(args) < 2:
            print("Usage: pidrei-server stop <instance-id>", file=sys.stderr)
            return 1
        _print_response(await send_ipc_request({"type": "stop", "instanceId": args[1]}))
        return 0

    if args[0] == "rpc":
        if len(args) < 3:
            print("Usage: pidrei-server rpc <instance-id> <json-command>", file=sys.stderr)
            return 1
        _print_response(await send_ipc_request({"type": "rpc", "instanceId": args[1], "command": json.loads(args[2])}))
        return 0

    if args[0] == "rpc-stream":
        if len(args) < 2:
            print("Usage: pidrei-server rpc-stream <instance-id>", file=sys.stderr)
            return 1
        return await _rpc_stream(args[1])

    print(f"Unknown command: {args[0]}", file=sys.stderr)
    _print_help()
    return 1


async def main(argv: list[str]) -> int:
    return await _main(argv)


def run() -> None:
    import signal as signal_module

    argv = sys.argv[1:]
    # serve() waits on a tonio signal receiver, which requires the runtime to
    # be created listening for those signals; other commands keep the default
    # dispositions (Ctrl-C stays a KeyboardInterrupt).
    signals = [signal_module.SIGINT, signal_module.SIGTERM] if argv[:1] == ["serve"] else None
    sys.exit(tonio.run(main(argv), signals=signals))
