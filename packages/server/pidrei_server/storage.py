"""Mirror of pi server src/storage.ts.

Sync by design: pi uses readFileSync/writeFileSync for machine.json and
instances.json (small records, ordering guaranteed by synchronous writes),
so these stay synchronous stdlib calls rather than tonio.fs hops.

pi's node process is single-threaded, so each sync storage call is
effectively atomic against every other one. tonio tasks run on parallel
worker threads, so the equivalent atomicity (in particular read-modify-write
upserts vs concurrent loads) needs the module lock below.

These functions stay sync deliberately: `supervisor` calls them through
`spawn_blocking`, so both the file I/O and the lock are held pool-side and
never on a runtime worker. Any new caller must do the same.
"""

import json
import os
import threading

from .config import get_instances_path, get_machine_path, get_server_dir
from .types import InstanceRecord, MachineRecord


_lock = threading.Lock()


def _ensure_server_dir() -> None:
    server_dir = get_server_dir()
    if not os.path.exists(server_dir):
        os.makedirs(server_dir, exist_ok=True)


def _load_machine() -> MachineRecord | None:
    machine_path = get_machine_path()
    if not os.path.exists(machine_path):
        return None

    with open(machine_path, encoding="utf-8") as f:
        return json.load(f)


def load_machine() -> MachineRecord | None:
    with _lock:
        return _load_machine()


def save_machine(machine: MachineRecord) -> None:
    with _lock:
        _ensure_server_dir()
        with open(get_machine_path(), "w", encoding="utf-8") as f:
            f.write(json.dumps(machine, indent=2))


def delete_machine() -> None:
    with _lock:
        machine_path = get_machine_path()
        if not os.path.exists(machine_path):
            return
        os.remove(machine_path)


def _load_instances() -> list[InstanceRecord]:
    instances_path = get_instances_path()
    if not os.path.exists(instances_path):
        return []

    with open(instances_path, encoding="utf-8") as f:
        return json.load(f)


def _save_instances(instances: list[InstanceRecord]) -> None:
    _ensure_server_dir()
    with open(get_instances_path(), "w", encoding="utf-8") as f:
        f.write(json.dumps(instances, indent=2))


def load_instances() -> list[InstanceRecord]:
    with _lock:
        return _load_instances()


def save_instances(instances: list[InstanceRecord]) -> None:
    with _lock:
        _save_instances(instances)


def get_instance(instance_id: str) -> InstanceRecord | None:
    with _lock:
        return next((instance for instance in _load_instances() if instance["id"] == instance_id), None)


def upsert_instance(instance: InstanceRecord) -> None:
    with _lock:
        instances = _load_instances()
        index = next((i for i, existing in enumerate(instances) if existing["id"] == instance["id"]), -1)
        if index == -1:
            instances.append(instance)
            _save_instances(instances)
            return

        instances[index] = instance
        _save_instances(instances)


def remove_instance(instance_id: str) -> None:
    with _lock:
        instances = [instance for instance in _load_instances() if instance["id"] != instance_id]
        _save_instances(instances)
