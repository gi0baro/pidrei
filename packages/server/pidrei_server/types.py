"""Mirror of pi server src/types.ts.

pi declares these as TypeScript interfaces over plain JSON objects; the
records are persisted to machine.json / instances.json and spread-merged
throughout the supervisor, so they stay plain camelCase dicts here (the
settings-manager precedent: open JS object semantics, unknown keys survive
round-trips).

InstanceStatus: "starting" | "online" | "stopping" | "stopped" | "error"

MachineRecord: {id, createdAt, lastSeenAt?, label?}

InstanceRecord: {id, status, cwd, createdAt, lastSeenAt?, label?,
sessionId?, sessionFile?}

pi's RadiusRegistration and the InstanceRecord `radiusPiId` field are absent:
the radius integration was dropped in Phase 7 step 1.
"""

from datetime import UTC, datetime
from typing import Any, Literal


InstanceStatus = Literal["starting", "online", "stopping", "stopped", "error"]
MachineRecord = dict[str, Any]
InstanceRecord = dict[str, Any]


def timestamp() -> str:
    """Record timestamp: millisecond precision like JS Date.toISOString()."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
