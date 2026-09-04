"""UUIDv7 generation (pi: packages/ai/src/utils/uuid.ts).

pi's generator (0.85.0): a 48-bit millisecond timestamp, a 41-bit sequence
seeded from randomness and incremented on every call — so ids order strictly
within the process even inside one millisecond — and fresh random tail bits.
A supplied timestamp is preserved verbatim for follower ids; ordinary calls
never let the clock run backwards (`max` with the last ordinary timestamp).

The generator state is process-global and callers run on any tonio worker
thread, so generation is serialized under a lock (pi's single thread gets
that for free). The stdlib `uuid.uuid7()` used until 0.84.4 cannot take a
caller-supplied timestamp, which is why pi's algorithm is carried here.
"""

import os
import threading
import time


MAX_UUID_V7_TIMESTAMP = 0xFFFFFFFFFFFF
_MAX_SEQUENCE = (1 << 41) - 1

_lock = threading.Lock()
_last_ordinary_timestamp = -1
_sequence: int | None = None


def _now_ms() -> int:
    """`Date.now()`; a module seam so tests can pin the clock."""
    return time.time_ns() // 1_000_000


def _random_bytes(count: int) -> bytes:
    """`crypto.getRandomValues`; a module seam so tests can stub randomness."""
    return os.urandom(count)


def uuidv7(timestamp_ms: int | None = None) -> str:
    """Generate a time-ordered UUIDv7. A supplied timestamp is preserved for follower ids."""
    global _last_ordinary_timestamp, _sequence
    requested = timestamp_ms if timestamp_ms is not None else _now_ms()
    # `Number.isInteger`: integral floats pass, everything else (NaN, ±inf,
    # fractions, bools) is rejected. pi throws RangeError.
    if isinstance(requested, bool) or not (
        isinstance(requested, int) or (isinstance(requested, float) and requested.is_integer())
    ):
        raise ValueError(f"UUIDv7 timestamp must be an integer between 0 and {MAX_UUID_V7_TIMESTAMP}")
    requested = int(requested)
    if requested < 0 or requested > MAX_UUID_V7_TIMESTAMP:
        raise ValueError(f"UUIDv7 timestamp must be an integer between 0 and {MAX_UUID_V7_TIMESTAMP}")

    with _lock:
        effective = max(requested, _last_ordinary_timestamp) if timestamp_ms is None else requested
        if timestamp_ms is None:
            _last_ordinary_timestamp = effective

        data = bytearray(_random_bytes(16))
        if _sequence is None:
            _sequence = int.from_bytes(data[1:6], "big")
        else:
            if _sequence == _MAX_SEQUENCE:
                raise ValueError("UUIDv7 generator sequence exhausted")
            _sequence += 1
        sequence = _sequence

    data[0:6] = effective.to_bytes(6, "big")
    data[6] = 0x70 | ((sequence >> 37) & 0x0F)
    data[7] = (sequence >> 29) & 0xFF
    data[8] = 0x80 | ((sequence >> 23) & 0x3F)
    data[9] = (sequence >> 15) & 0xFF
    data[10] = (sequence >> 7) & 0xFF
    data[11] = ((sequence & 0x7F) << 1) | (data[11] & 0x01)

    hex_digits = data.hex()
    return f"{hex_digits[0:8]}-{hex_digits[8:12]}-{hex_digits[12:16]}-{hex_digits[16:20]}-{hex_digits[20:]}"
