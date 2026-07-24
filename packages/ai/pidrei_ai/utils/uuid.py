"""UUIDv7 generation (pi: packages/ai/src/utils/uuid.ts).

CPython 3.14 ships `uuid.uuid7()` with in-process monotonicity (42-bit RFC 9562
counter), but its module-level counter state is not lock-guarded — on the
free-threaded build concurrent callers can race it. Callers may run on any
tonio worker thread, so pidrei serializes generation under a lock.
"""

import threading
import uuid


_lock = threading.Lock()


def uuidv7() -> str:
    """Generate a time-ordered UUIDv7 string."""
    with _lock:
        return str(uuid.uuid7())
