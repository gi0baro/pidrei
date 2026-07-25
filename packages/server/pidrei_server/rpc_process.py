"""Mirror of pi server src/rpc-process.ts.

pi spawns the RPC child as either the sibling `pi` binary (Bun build) or
`node .../rpc-entry`; pidrei always spawns `python -m pidrei --mode rpc`.
The `cli_path` option is a test seam matching the RpcClientOptions
convention (spawn `python <cli_path>` instead).

Invalid JSON on the child's stdout is ignored (the rpc-client convention);
pi lets JSON.parse throw into node's uncaughtException handler, which would
tear the whole server down on one bad line.
"""

import json
import signal as signal_module
import subprocess
import sys
import uuid
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio


@dataclass(slots=True, kw_only=True)
class RpcProcessOptions:
    # Working directory for the spawned instance
    cwd: str
    # Test seam: spawn `python <cli_path>` instead of `python -m pidrei`
    cli_path: str | None = None


class _PendingRequest:
    __slots__ = ("error", "event", "response")

    def __init__(self) -> None:
        self.event = tonio.Event()
        self.response: dict[str, Any] | None = None
        self.error: Exception | None = None

    def resolve(self, response: dict[str, Any]) -> None:
        self.response = response
        self.event.set()

    def reject(self, error: Exception) -> None:
        self.error = error
        self.event.set()


class RpcProcessInstance:
    def __init__(self, process: Any):
        self.process = process
        self._exited = False
        self._exit_event = tonio.Event()
        self._next_request_id = 0
        self._stderr_buffer = ""
        self._pending_requests: dict[str, _PendingRequest] = {}
        self._event_listeners: list[Any] = []
        self._exit_listeners: list[Any] = []
        self._ui_request_handler: Any = None
        self._attach_listeners()

    def _attach_listeners(self) -> None:
        process = self.process

        async def pump_stdout() -> None:
            buffer = ""
            try:
                while True:
                    chunk = await process.stdout.receive_some()
                    if not chunk:
                        break
                    buffer += chunk.decode("utf-8", "replace")
                    while True:
                        newline_index = buffer.find("\n")
                        if newline_index == -1:
                            break
                        line = buffer[:newline_index].strip()
                        buffer = buffer[newline_index + 1 :]
                        if not line:
                            continue
                        self._handle_line(line)
            except Exception:
                pass
            code = await process.wait()
            self._exited = True
            error = self._process_exit_error(code)
            self._reject_all_pending(error)
            self._notify_exit(error)
            self._exit_event.set()

        async def pump_stderr() -> None:
            try:
                while True:
                    chunk = await process.stderr.receive_some()
                    if not chunk:
                        return
                    self._stderr_buffer += chunk.decode("utf-8", "replace")
            except Exception:
                pass

        tonio.spawn.without_tracking(pump_stderr())
        tonio.spawn.without_tracking(pump_stdout())

    def _process_exit_error(self, returncode: int | None) -> Exception:
        if returncode is not None and returncode < 0:
            code: Any = "null"
            try:
                sig: Any = signal_module.Signals(-returncode).name
            except ValueError:
                sig = -returncode
        else:
            code = returncode
            sig = "null"
        return Exception(f"RPC process exited (code={code} signal={sig}). Stderr: {self._stderr_buffer}")

    def _handle_line(self, line: str) -> None:
        try:
            parsed = json.loads(line)
        except Exception:
            return
        if not isinstance(parsed, dict):
            return

        message_type = parsed.get("type")
        if message_type == "response":
            request_id = parsed.get("id")
            if not request_id:
                return
            pending = self._pending_requests.pop(request_id, None)
            if pending is None:
                return
            pending.resolve(parsed)
            return

        if message_type == "extension_ui_request":
            if self._ui_request_handler is not None:
                self._ui_request_handler(parsed)
            return

        for listener in list(self._event_listeners):
            listener(parsed)

    def _reject_all_pending(self, error: Exception) -> None:
        pending = list(self._pending_requests.values())
        self._pending_requests.clear()
        for request in pending:
            request.reject(error)

    def _notify_exit(self, error: Exception | None) -> None:
        for listener in list(self._exit_listeners):
            listener(error)

    async def send(self, command: dict[str, Any]) -> dict[str, Any]:
        if self._exited:
            raise Exception(f"RPC process is not running. Stderr: {self._stderr_buffer}")
        self._next_request_id += 1
        request_id = command.get("id") or f"server_{self._next_request_id}_{uuid.uuid4()}"
        full_command = {**command, "id": request_id}

        pending = _PendingRequest()
        self._pending_requests[request_id] = pending
        try:
            await self.process.stdin.send_all((json.dumps(full_command, ensure_ascii=False) + "\n").encode("utf-8"))
        except Exception as error:
            self._pending_requests.pop(request_id, None)
            raise Exception(str(error))

        await pending.event.wait()
        if pending.error is not None:
            raise pending.error
        assert pending.response is not None
        return pending.response

    async def handle_ui_response(self, response: dict[str, Any]) -> None:
        if self._exited:
            return
        try:
            await self.process.stdin.send_all((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
        except Exception:
            pass

    def set_ui_request_handler(self, handler: Any = None) -> None:
        self._ui_request_handler = handler

    def on_event(self, listener: Any) -> Any:
        self._event_listeners.append(listener)

        def unsubscribe() -> None:
            try:
                self._event_listeners.remove(listener)
            except ValueError:
                pass

        return unsubscribe

    def on_exit(self, listener: Any) -> Any:
        self._exit_listeners.append(listener)

        def unsubscribe() -> None:
            try:
                self._exit_listeners.remove(listener)
            except ValueError:
                pass

        return unsubscribe

    async def dispose(self) -> None:
        self._ui_request_handler = None
        self._reject_all_pending(Exception("RPC process disposed"))
        if self._exited:
            return
        try:
            self.process.send_signal(signal_module.SIGTERM)
        except Exception:
            pass
        await self._exit_event.wait()


async def create_rpc_process_instance(options: RpcProcessOptions) -> RpcProcessInstance:
    if options.cli_path is not None:
        command = [sys.executable, options.cli_path]
    else:
        command = [sys.executable, "-m", "pidrei", "--mode", "rpc"]

    process = await tonio.open_process(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=options.cwd,
    )
    return RpcProcessInstance(process)
