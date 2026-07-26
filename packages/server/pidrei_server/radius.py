"""Mirror of pi server src/radius.ts.

Radius presence integration: machine/instance registration and heartbeats
against the radius service. Fully inert without stored radius credentials
or RADIUS_API_KEY.

HTTP goes through the punkreq seam (pidrei_ai.utils.http.shared_client),
matching the remote-catalog convention. setTimeout/clearTimeout maps to the
Event-based one-shot _Timer below.

Node's `process.arch` reports "x64"/"arm64"; platform.machine() reports
"x86_64"/"aarch64" — the raw Python value is sent.
"""

import json
import os
import platform
import random
import socket as stdlib_socket
import sys
import urllib.parse
from typing import Any

import tonio.colored as tonio

from pidrei.core.auth_storage import read_stored_credential
from pidrei_ai.auth.types import OAuthCredential

from .config import VERSION, get_server_dir, get_socket_path
from .storage import load_machine, save_machine
from .types import InstanceRecord, MachineRecord, timestamp


DEFAULT_RADIUS_URL = "https://radius.pi.dev/"
DEFAULT_SERVER_BASE_PATH = "/v1/"
NOT_FOUND_RETRY_THRESHOLD = 3
HEARTBEAT_BACKOFF_BASE_MS = 1_000
HEARTBEAT_BACKOFF_MAX_MS = 30_000
RADIUS_PROVIDER = "radius"


class _Timer:
    """setTimeout/clearTimeout equivalent: cancellable one-shot on an Event."""

    __slots__ = ("_cancelled",)

    def __init__(self, delay_s: float, fn: Any):
        self._cancelled = tonio.Event()
        tonio.spawn.without_tracking(self._run(delay_s, fn))

    async def _run(self, delay_s: float, fn: Any) -> None:
        await self._cancelled.wait(delay_s)
        if self._cancelled.is_set():
            return
        await fn()

    def cancel(self) -> None:
        self._cancelled.set()


class _PiHeartbeatState:
    __slots__ = ("consecutive_not_found_count", "interval_ms", "radius_pi_id", "timer", "transient_failure_count")

    def __init__(self, interval_ms: float, radius_pi_id: str):
        self.timer: _Timer | None = None
        self.interval_ms = interval_ms
        self.radius_pi_id = radius_pi_id
        self.consecutive_not_found_count = 0
        self.transient_failure_count = 0


class RadiusHttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _compact(body: dict[str, Any]) -> dict[str, Any]:
    """Drop None values like JSON.stringify drops undefined properties."""
    return {key: value for key, value in body.items() if value is not None}


async def _post(path: str, body: Any) -> Any:
    from pidrei_ai.utils.http import shared_client

    response = await shared_client().post(
        urllib.parse.urljoin(get_radius_server_base_url(), path),
        headers={
            "Authorization": f"Bearer {get_radius_access_token()}",
            "Content-Type": "application/json",
        },
        content=json.dumps(body).encode("utf-8"),
    )

    body_bytes = await response.read()
    text = body_bytes.decode("utf-8", "replace") if isinstance(body_bytes, bytes) else body_bytes
    if response.status_code < 200 or response.status_code >= 300:
        raise RadiusHttpError(response.status_code, f"Radius request failed: {response.status_code} {text}")

    return json.loads(text)


async def _maybe_post(path: str, body: Any) -> None:
    from pidrei_ai.utils.http import shared_client

    response = await shared_client().post(
        urllib.parse.urljoin(get_radius_server_base_url(), path),
        headers={
            "Authorization": f"Bearer {get_radius_access_token()}",
            "Content-Type": "application/json",
        },
        content=json.dumps(body).encode("utf-8"),
    )
    if response.status_code < 200 or response.status_code >= 300:
        body_bytes = await response.read()
        text = body_bytes.decode("utf-8", "replace") if isinstance(body_bytes, bytes) else body_bytes
        raise RadiusHttpError(response.status_code, f"Radius request failed: {response.status_code} {text}")


def _is_not_found_error(error: Exception) -> bool:
    return isinstance(error, RadiusHttpError) and error.status == 404


def compute_backoff_delay_ms(failure_count: int) -> float:
    exponential_delay = min(HEARTBEAT_BACKOFF_MAX_MS, HEARTBEAT_BACKOFF_BASE_MS * 2 ** max(0, failure_count - 1))
    jitter_ms = int(random.random() * max(250, exponential_delay / 4))  # noqa: S311
    return min(HEARTBEAT_BACKOFF_MAX_MS, exponential_delay + jitter_ms)


def _format_radius_error(error: Exception) -> str:
    if isinstance(error, RadiusHttpError):
        return f"HTTP {error.status}: {error}"
    return str(error)


def _log_radius_retry(scope: str, action: str, delay_ms: float, failure_count: int, error: Exception) -> None:
    print(
        f"{scope} {action} failed (attempt {failure_count}); retrying in {delay_ms}ms: {_format_radius_error(error)}",
        file=sys.stderr,
    )


def get_radius_url() -> str:
    return os.environ.get("PIDREI_RADIUS_URL") or DEFAULT_RADIUS_URL


def get_radius_server_base_url() -> str:
    explicit_url = os.environ.get("PIDREI_RADIUS_SERVER_URL")
    if explicit_url:
        return explicit_url

    return urllib.parse.urljoin(get_radius_url(), DEFAULT_SERVER_BASE_PATH)


def _get_stored_radius_credential() -> OAuthCredential | None:
    credential = read_stored_credential(RADIUS_PROVIDER)
    return credential if isinstance(credential, OAuthCredential) else None


def get_radius_access_token() -> str:
    stored_credential = _get_stored_radius_credential()
    if stored_credential is not None and isinstance(stored_credential.access, str) and stored_credential.access:
        return stored_credential.access

    api_key = os.environ.get("RADIUS_API_KEY")
    if api_key:
        return api_key

    raise Exception("Radius credentials are required in ~/.pidrei/agent/auth.json or RADIUS_API_KEY")


def is_radius_enabled() -> bool:
    stored = _get_stored_radius_credential()
    return bool(stored is not None and stored.access) or bool(os.environ.get("RADIUS_API_KEY"))


class RadiusPresence:
    def __init__(self) -> None:
        self._machine_heartbeat_timer: _Timer | None = None
        self._machine_heartbeat_interval_ms: float = 0
        self._machine_consecutive_not_found_count = 0
        self._machine_transient_failure_count = 0
        self._pi_heartbeat_states: dict[str, _PiHeartbeatState] = {}
        self._machine: MachineRecord | None = None
        self._coordinator: Any = None

    def set_coordinator(self, coordinator: Any) -> None:
        """Coordinator protocol: get_live_instance(instance_id), list_live_instances(), update_instance(instance)."""
        self._coordinator = coordinator

    async def start(self, label: str | None = None) -> MachineRecord | None:
        if not is_radius_enabled():
            return None

        registered = await self._register_machine(label)
        self._start_machine_heartbeat(registered["heartbeatIntervalMs"])
        return self._machine

    async def stop(self) -> None:
        if self._machine_heartbeat_timer is not None:
            self._machine_heartbeat_timer.cancel()
            self._machine_heartbeat_timer = None
        for instance_id, state in list(self._pi_heartbeat_states.items()):
            if state.timer is not None:
                state.timer.cancel()
            self._pi_heartbeat_states.pop(instance_id, None)
        if self._machine is None or not is_radius_enabled():
            return
        try:
            await _maybe_post(f"machines/{self._machine['id']}/disconnect", {})
        except Exception as error:
            if not _is_not_found_error(error):
                raise

    async def register_pi(self, instance: InstanceRecord) -> InstanceRecord:
        if not is_radius_enabled():
            return instance
        machine = self._machine if self._machine is not None else load_machine()
        if machine is None:
            raise Exception("No registered machine available for Pi registration")
        registered = await _post(
            "pis/register",
            _compact(
                {
                    "machineId": machine["id"],
                    "label": instance.get("label"),
                    "cwd": instance["cwd"],
                    "hostname": stdlib_socket.gethostname(),
                    "pid": os.getpid(),
                    "transport": "local-rpc",
                    "capabilities": {"rpc": True, "relay": False, "iroh": False},
                    "sessionId": instance.get("sessionId"),
                }
            ),
        )
        registered_instance = {**instance, "radiusPiId": registered["id"]}
        self._start_pi_heartbeat(instance["id"], registered["heartbeatIntervalMs"], registered["id"])
        return registered_instance

    async def disconnect_pi(self, instance: InstanceRecord) -> None:
        state = self._pi_heartbeat_states.get(instance["id"])
        if state is not None:
            if state.timer is not None:
                state.timer.cancel()
            self._pi_heartbeat_states.pop(instance["id"], None)
        if not is_radius_enabled() or not instance.get("radiusPiId"):
            return
        try:
            await _maybe_post(f"pis/{instance['radiusPiId']}/disconnect", {})
        except Exception as error:
            if not _is_not_found_error(error):
                raise

    async def _register_machine(self, label: str | None = None) -> dict[str, Any]:
        existing_machine = self._machine if self._machine is not None else load_machine()
        registered = await _post(
            "machines/register",
            _compact(
                {
                    "machineId": existing_machine["id"] if existing_machine else None,
                    "label": label,
                    "hostname": stdlib_socket.gethostname(),
                    "platform": sys.platform,
                    "arch": platform.machine(),
                    "version": VERSION,
                    "capabilities": {"spawn": True, "relay": False, "iroh": False},
                }
            ),
        )

        now = timestamp()
        machine: MachineRecord = {
            "id": registered["id"],
            "createdAt": existing_machine["createdAt"] if existing_machine else now,
            "lastSeenAt": now,
        }
        if label is not None:
            machine["label"] = label
        self._machine = machine
        save_machine(machine)
        self._machine_consecutive_not_found_count = 0
        self._machine_transient_failure_count = 0
        return registered

    def _start_machine_heartbeat(self, interval_ms: float) -> None:
        self._machine_heartbeat_interval_ms = interval_ms
        self._schedule_machine_heartbeat(interval_ms)

    def _schedule_machine_heartbeat(self, delay_ms: float) -> None:
        if self._machine_heartbeat_timer is not None:
            self._machine_heartbeat_timer.cancel()
        self._machine_heartbeat_timer = _Timer(delay_ms / 1000, self._heartbeat_machine)

    def _start_pi_heartbeat(self, instance_id: str, interval_ms: float, radius_pi_id: str) -> None:
        existing_state = self._pi_heartbeat_states.get(instance_id)
        if existing_state is not None and existing_state.timer is not None:
            existing_state.timer.cancel()
        state = existing_state if existing_state is not None else _PiHeartbeatState(interval_ms, radius_pi_id)
        state.interval_ms = interval_ms
        state.radius_pi_id = radius_pi_id
        state.consecutive_not_found_count = 0
        state.transient_failure_count = 0
        self._pi_heartbeat_states[instance_id] = state
        self._schedule_pi_heartbeat(instance_id, interval_ms)

    def _schedule_pi_heartbeat(self, instance_id: str, delay_ms: float) -> None:
        state = self._pi_heartbeat_states.get(instance_id)
        if state is None:
            return
        if state.timer is not None:
            state.timer.cancel()

        async def fire() -> None:
            await self._heartbeat_pi(instance_id)

        state.timer = _Timer(delay_ms / 1000, fire)

    async def _heartbeat_machine(self) -> None:
        if self._machine is None or not is_radius_enabled():
            return

        try:
            await _maybe_post(
                f"machines/{self._machine['id']}/heartbeat",
                {"cwd": get_server_dir(), "socketPath": get_socket_path()},
            )
            self._machine_consecutive_not_found_count = 0
            self._machine_transient_failure_count = 0
            self._schedule_machine_heartbeat(self._machine_heartbeat_interval_ms)
        except Exception as error:
            if not _is_not_found_error(error):
                self._machine_transient_failure_count += 1
                delay_ms = compute_backoff_delay_ms(self._machine_transient_failure_count)
                _log_radius_retry("Radius machine", "heartbeat", delay_ms, self._machine_transient_failure_count, error)
                self._schedule_machine_heartbeat(delay_ms)
                return

            self._machine_transient_failure_count = 0
            self._machine_consecutive_not_found_count += 1
            if self._machine_consecutive_not_found_count < NOT_FOUND_RETRY_THRESHOLD:
                self._schedule_machine_heartbeat(self._machine_heartbeat_interval_ms)
                return

            try:
                await self._re_register_machine_and_pis()
            except Exception as recovery_error:
                self._machine_transient_failure_count += 1
                delay_ms = compute_backoff_delay_ms(self._machine_transient_failure_count)
                _log_radius_retry(
                    "Radius machine",
                    "re-registration",
                    delay_ms,
                    self._machine_transient_failure_count,
                    recovery_error,
                )
                self._schedule_machine_heartbeat(delay_ms)

    async def _heartbeat_pi(self, instance_id: str) -> None:
        if not is_radius_enabled():
            return

        state = self._pi_heartbeat_states.get(instance_id)
        if state is None:
            return

        try:
            await _maybe_post(f"pis/{state.radius_pi_id}/heartbeat", {})
            state.consecutive_not_found_count = 0
            state.transient_failure_count = 0
            self._schedule_pi_heartbeat(instance_id, state.interval_ms)
        except Exception as error:
            if not _is_not_found_error(error):
                state.transient_failure_count += 1
                delay_ms = compute_backoff_delay_ms(state.transient_failure_count)
                _log_radius_retry(
                    f"Radius Pi {instance_id}", "heartbeat", delay_ms, state.transient_failure_count, error
                )
                self._schedule_pi_heartbeat(instance_id, delay_ms)
                return

            state.transient_failure_count = 0
            state.consecutive_not_found_count += 1
            if state.consecutive_not_found_count < NOT_FOUND_RETRY_THRESHOLD:
                self._schedule_pi_heartbeat(instance_id, state.interval_ms)
                return

            try:
                recovered = await self._re_register_pi(instance_id)
                if not recovered:
                    delay_ms = compute_backoff_delay_ms(1)
                    print(f"Radius Pi {instance_id} re-registration skipped; retrying in {delay_ms}ms", file=sys.stderr)
                    self._schedule_pi_heartbeat(instance_id, delay_ms)
            except Exception as recovery_error:
                state.transient_failure_count += 1
                delay_ms = compute_backoff_delay_ms(state.transient_failure_count)
                _log_radius_retry(
                    f"Radius Pi {instance_id}",
                    "re-registration",
                    delay_ms,
                    state.transient_failure_count,
                    recovery_error,
                )
                self._schedule_pi_heartbeat(instance_id, delay_ms)

    async def _re_register_machine_and_pis(self) -> None:
        registered = await self._register_machine(self._machine.get("label") if self._machine else None)
        self._start_machine_heartbeat(registered["heartbeatIntervalMs"])

        instances = self._coordinator.list_live_instances() if self._coordinator is not None else []
        for instance in instances:
            try:
                await self._re_register_pi(instance["id"])
            except Exception as error:
                print(
                    f"Radius Pi {instance['id']} re-registration failed: {_format_radius_error(error)}", file=sys.stderr
                )

    async def _re_register_pi(self, instance_id: str) -> bool:
        instance = self._coordinator.get_live_instance(instance_id) if self._coordinator is not None else None
        if instance is None:
            state = self._pi_heartbeat_states.get(instance_id)
            if state is not None:
                if state.timer is not None:
                    state.timer.cancel()
                self._pi_heartbeat_states.pop(instance_id, None)
            return False

        if self._machine is None:
            await self._re_register_machine_and_pis()
            return True

        registered_instance = await self.register_pi(instance)
        if self._coordinator is not None:
            self._coordinator.update_instance(registered_instance)
        return True


radius_presence = RadiusPresence()
