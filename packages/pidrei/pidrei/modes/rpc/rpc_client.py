"""Mirror of pi coding-agent src/modes/rpc/rpc-client.ts.

RPC Client for programmatic access to the coding agent. Spawns the agent
in RPC mode and provides a typed API for all operations.

pi spawns `node dist/cli.js`; pidrei spawns `python -m pidrei` (or
`python <cli_path>` when a script path is given, which the process-failure
tests use). Wire payloads are returned as parsed JSON dicts.
"""

import json
import os
import signal as signal_module
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

import tonio.colored as tonio

from .jsonl import JsonlLineDecoder, serialize_json_line


_REQUEST_TIMEOUT_S = 30.0
_DEFAULT_IDLE_TIMEOUT_MS = 60000


@dataclass(slots=True, kw_only=True)
class RpcClientOptions:
    # Path to a CLI entry-point script (default: run `python -m pidrei`)
    cli_path: str | None = None
    # Working directory for the agent
    cwd: str | None = None
    # Environment variables (merged over the current environment)
    env: dict[str, str] | None = None
    # Provider to use
    provider: str | None = None
    # Model ID to use
    model: str | None = None
    # Additional CLI arguments
    args: list[str] = field(default_factory=list)


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


class RpcClient:
    def __init__(self, options: RpcClientOptions | None = None):
        self._options = options or RpcClientOptions()
        self._process: Any = None
        self._event_listeners: list[Any] = []
        self._pending_requests: dict[str, _PendingRequest] = {}
        self._request_id = 0
        self._stderr = ""
        self._exit_error: Exception | None = None
        self._reader_handle: Any = None
        self._stderr_handle: Any = None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start the RPC agent process."""
        if self._process is not None:
            raise Exception("Client already started")

        self._exit_error = None

        if self._options.cli_path is not None:
            command = [sys.executable, self._options.cli_path]
        else:
            command = [sys.executable, "-m", "pidrei"]
        args = ["--mode", "rpc"]

        if self._options.provider:
            args.extend(["--provider", self._options.provider])
        if self._options.model:
            args.extend(["--model", self._options.model])
        if self._options.args:
            args.extend(self._options.args)

        env = None
        if self._options.env is not None:
            env = {**os.environ, **self._options.env}

        process = await tonio.open_process(
            [*command, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self._options.cwd,
            env=env,
        )
        self._process = process

        # Collect stderr for debugging
        async def pump_stderr() -> None:
            try:
                while True:
                    chunk = await process.stderr.receive_some()
                    if not chunk:
                        return
                    text = chunk.decode("utf-8", "replace")
                    self._stderr += text
                    sys.stderr.write(text)
            except Exception:
                pass

        # Strict JSONL reader for stdout
        async def pump_stdout() -> None:
            decoder = JsonlLineDecoder()
            try:
                while True:
                    chunk = await process.stdout.receive_some()
                    if not chunk:
                        break
                    for line in decoder.feed(chunk):
                        self._handle_line(line)
                for line in decoder.end():
                    self._handle_line(line)
            except Exception:
                pass
            # Stdout EOF implies process exit; reject in-flight requests.
            if self._process is not process:
                return
            await process.wait()
            if self._process is not process:
                return
            error = self._create_process_exit_error(process)
            self._exit_error = error
            self._reject_pending_requests(error)

        self._stderr_handle = tonio.spawn.without_tracking(pump_stderr())
        self._reader_handle = tonio.spawn.without_tracking(pump_stdout())

        # Wait a moment for the process to initialize
        await tonio.sleep(0.1)

        if process.poll() is not None:
            error = self._exit_error or self._create_process_exit_error(process)
            self._exit_error = error
            raise error

    async def stop(self) -> None:
        """Stop the RPC agent process."""
        process = self._process
        if process is None:
            return

        try:
            process.send_signal(signal_module.SIGTERM)
        except Exception:
            pass

        # Wait for process to exit, force kill after 1 second
        _result, completed = await tonio.time.timeout(process.wait(), 1.0)
        if not completed:
            try:
                process.kill()
            except Exception:
                pass
            await process.wait()

        self._process = None
        self._pending_requests.clear()

    def on_event(self, listener) -> Any:
        """Subscribe to agent events. Returns an unsubscribe callable."""
        self._event_listeners.append(listener)

        def unsubscribe() -> None:
            try:
                self._event_listeners.remove(listener)
            except ValueError:
                pass

        return unsubscribe

    def get_stderr(self) -> str:
        """Get collected stderr output (useful for debugging)."""
        return self._stderr

    # =========================================================================
    # Command Methods
    # =========================================================================

    async def prompt(self, message: str, images: list[Any] | None = None) -> None:
        """Send a prompt to the agent.

        Returns after the preflight response arrives; use on_event() to
        receive streaming events and wait_for_idle() to wait for completion."""
        await self._send({"type": "prompt", "message": message, "images": images})

    async def steer(self, message: str, images: list[Any] | None = None) -> None:
        """Queue a steering message to interrupt the agent mid-run."""
        await self._send({"type": "steer", "message": message, "images": images})

    async def follow_up(self, message: str, images: list[Any] | None = None) -> None:
        """Queue a follow-up message to be processed after the agent finishes."""
        await self._send({"type": "follow_up", "message": message, "images": images})

    async def abort(self) -> None:
        """Abort current operation."""
        await self._send({"type": "abort"})

    async def new_session(self, parent_session: str | None = None) -> dict[str, Any]:
        """Start a new session, optionally with parent tracking."""
        response = await self._send({"type": "new_session", "parentSession": parent_session})
        return self._get_data(response)

    async def get_state(self) -> dict[str, Any]:
        """Get current session state."""
        response = await self._send({"type": "get_state"})
        return self._get_data(response)

    async def set_model(self, provider: str, model_id: str) -> dict[str, Any]:
        """Set model by provider and ID."""
        response = await self._send({"type": "set_model", "provider": provider, "modelId": model_id})
        return self._get_data(response)

    async def cycle_model(self) -> dict[str, Any] | None:
        """Cycle to next model."""
        response = await self._send({"type": "cycle_model"})
        return self._get_data(response)

    async def get_available_models(self) -> list[dict[str, Any]]:
        """Get list of available models."""
        response = await self._send({"type": "get_available_models"})
        return self._get_data(response)["models"]

    async def set_thinking_level(self, level: str) -> None:
        """Set thinking level."""
        await self._send({"type": "set_thinking_level", "level": level})

    async def cycle_thinking_level(self) -> dict[str, Any] | None:
        """Cycle thinking level."""
        response = await self._send({"type": "cycle_thinking_level"})
        return self._get_data(response)

    async def get_available_thinking_levels(self) -> list[str]:
        """Get list of available thinking levels for the current model."""
        response = await self._send({"type": "get_available_thinking_levels"})
        return self._get_data(response)["levels"]

    async def set_steering_mode(self, mode: str) -> None:
        """Set steering mode."""
        await self._send({"type": "set_steering_mode", "mode": mode})

    async def set_follow_up_mode(self, mode: str) -> None:
        """Set follow-up mode."""
        await self._send({"type": "set_follow_up_mode", "mode": mode})

    async def compact(self, custom_instructions: str | None = None) -> dict[str, Any]:
        """Compact session context."""
        response = await self._send({"type": "compact", "customInstructions": custom_instructions})
        return self._get_data(response)

    async def set_auto_compaction(self, enabled: bool) -> None:
        """Set auto-compaction enabled/disabled."""
        await self._send({"type": "set_auto_compaction", "enabled": enabled})

    async def set_auto_retry(self, enabled: bool) -> None:
        """Set auto-retry enabled/disabled."""
        await self._send({"type": "set_auto_retry", "enabled": enabled})

    async def abort_retry(self) -> None:
        """Abort in-progress retry."""
        await self._send({"type": "abort_retry"})

    async def bash(self, command: str) -> dict[str, Any]:
        """Execute a bash command."""
        response = await self._send({"type": "bash", "command": command})
        return self._get_data(response)

    async def abort_bash(self) -> None:
        """Abort running bash command."""
        await self._send({"type": "abort_bash"})

    async def get_session_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        response = await self._send({"type": "get_session_stats"})
        return self._get_data(response)

    async def export_html(self, output_path: str | None = None) -> dict[str, Any]:
        """Export session to HTML."""
        response = await self._send({"type": "export_html", "outputPath": output_path})
        return self._get_data(response)

    async def switch_session(self, session_path: str) -> dict[str, Any]:
        """Switch to a different session file."""
        response = await self._send({"type": "switch_session", "sessionPath": session_path})
        return self._get_data(response)

    async def fork(self, entry_id: str) -> dict[str, Any]:
        """Fork from a specific message."""
        response = await self._send({"type": "fork", "entryId": entry_id})
        return self._get_data(response)

    async def clone(self) -> dict[str, Any]:
        """Clone the current active branch into a new session."""
        response = await self._send({"type": "clone"})
        return self._get_data(response)

    async def get_fork_messages(self) -> list[dict[str, Any]]:
        """Get messages available for forking."""
        response = await self._send({"type": "get_fork_messages"})
        return self._get_data(response)["messages"]

    async def get_entries(self, since: str | None = None) -> dict[str, Any]:
        """Get session entries in append order, optionally after `since`."""
        response = await self._send({"type": "get_entries", "since": since})
        return self._get_data(response)

    async def get_tree(self) -> dict[str, Any]:
        """Get the session entry tree."""
        response = await self._send({"type": "get_tree"})
        return self._get_data(response)

    async def get_last_assistant_text(self) -> str | None:
        """Get text of last assistant message."""
        response = await self._send({"type": "get_last_assistant_text"})
        return self._get_data(response)["text"]

    async def set_session_name(self, name: str) -> None:
        """Set the session display name."""
        await self._send({"type": "set_session_name", "name": name})

    async def get_messages(self) -> list[dict[str, Any]]:
        """Get all messages in the session."""
        response = await self._send({"type": "get_messages"})
        return self._get_data(response)["messages"]

    async def get_commands(self) -> list[dict[str, Any]]:
        """Get available commands (extension commands, prompt templates, skills)."""
        response = await self._send({"type": "get_commands"})
        return self._get_data(response)["commands"]

    # =========================================================================
    # Helpers
    # =========================================================================

    async def wait_for_idle(self, timeout: float = _DEFAULT_IDLE_TIMEOUT_MS) -> None:
        """Wait for agent to become idle (agent_settled event), timeout in ms."""
        settled = tonio.Event()

        def listener(event: dict[str, Any]) -> None:
            if event.get("type") == "agent_settled":
                settled.set()

        unsubscribe = self.on_event(listener)
        try:
            await settled.wait(timeout / 1000)
            if not settled.is_set():
                raise Exception(f"Timeout waiting for agent to become idle. Stderr: {self._stderr}")
        finally:
            unsubscribe()

    async def collect_events(self, timeout: float = _DEFAULT_IDLE_TIMEOUT_MS) -> list[dict[str, Any]]:
        """Collect events until agent becomes idle, timeout in ms."""
        events: list[dict[str, Any]] = []
        settled = tonio.Event()

        def listener(event: dict[str, Any]) -> None:
            events.append(event)
            if event.get("type") == "agent_settled":
                settled.set()

        unsubscribe = self.on_event(listener)
        try:
            await settled.wait(timeout / 1000)
            if not settled.is_set():
                raise Exception(f"Timeout collecting events. Stderr: {self._stderr}")
        finally:
            unsubscribe()
        return events

    async def prompt_and_wait(
        self, message: str, images: list[Any] | None = None, timeout: float = _DEFAULT_IDLE_TIMEOUT_MS
    ) -> list[dict[str, Any]]:
        """Send prompt and wait for completion, returning all events."""
        events_handle = tonio.spawn(self.collect_events(timeout))
        await self.prompt(message, images)
        return await events_handle

    # =========================================================================
    # Internal
    # =========================================================================

    def _handle_line(self, line: str) -> None:

        try:
            data = json.loads(line)
        except Exception:
            # Ignore non-JSON lines
            return

        if not isinstance(data, dict):
            return

        # Check if it's a response to a pending request
        if data.get("type") == "response" and data.get("id") in self._pending_requests:
            pending = self._pending_requests.pop(data["id"])
            pending.resolve(data)
            return

        # Otherwise it's an event
        for listener in list(self._event_listeners):
            listener(data)

    def _create_process_exit_error(self, process: Any) -> Exception:
        returncode = process.poll()
        if returncode is not None and returncode < 0:
            code: Any = "null"
            try:
                sig: Any = signal_module.Signals(-returncode).name
            except ValueError:
                sig = -returncode
        else:
            code = returncode
            sig = "null"
        return Exception(f"Agent process exited (code={code} signal={sig}). Stderr: {self._stderr}")

    def _reject_pending_requests(self, error: Exception) -> None:
        pending = list(self._pending_requests.values())
        self._pending_requests.clear()
        for request in pending:
            request.reject(error)

    async def _send(self, command: dict[str, Any]) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None:
            raise Exception("Client not started")
        if self._exit_error is not None:
            raise self._exit_error
        if process.poll() is not None:
            error = self._create_process_exit_error(process)
            self._exit_error = error
            raise error

        self._request_id += 1
        request_id = f"req_{self._request_id}"
        full_command = {key: value for key, value in command.items() if value is not None}
        full_command["id"] = request_id

        pending = _PendingRequest()
        self._pending_requests[request_id] = pending

        try:
            await process.stdin.send_all(serialize_json_line(full_command).encode("utf-8"))
        except Exception as write_error:
            self._pending_requests.pop(request_id, None)
            raise Exception(f"Agent process stdin error: {write_error}. Stderr: {self._stderr}") from None

        await pending.event.wait(_REQUEST_TIMEOUT_S)
        if not pending.event.is_set():
            self._pending_requests.pop(request_id, None)
            raise Exception(f"Timeout waiting for response to {command.get('type')}. Stderr: {self._stderr}")
        if pending.error is not None:
            raise pending.error
        assert pending.response is not None
        return pending.response

    def _get_data(self, response: dict[str, Any]) -> Any:
        if not response.get("success"):
            raise Exception(response.get("error"))
        return response.get("data")
