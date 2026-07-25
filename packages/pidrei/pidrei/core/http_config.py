"""HTTP timeout settings parsing (from pi coding-agent src/core/http-dispatcher.ts).

Only the settings-facing pieces are ported here; the undici global-dispatcher
installation has no pidrei equivalent (HTTP transport is punkreq's concern).
"""

import math
from typing import Any


DEFAULT_HTTP_IDLE_TIMEOUT_MS = 300_000

HTTP_IDLE_TIMEOUT_CHOICES = [
    {"label": "30 sec", "timeout_ms": 30_000},
    {"label": "1 min", "timeout_ms": 60_000},
    {"label": "2 min", "timeout_ms": 120_000},
    {"label": "5 min", "timeout_ms": 300_000},
    {"label": "disabled", "timeout_ms": 0},
]


def parse_http_idle_timeout_ms(value: Any) -> int | None:
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed.lower() == "disabled":
            return 0
        if len(trimmed) == 0:
            return None
        try:
            return parse_http_idle_timeout_ms(float(trimmed))
        except ValueError:
            return None

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return math.floor(value)


def format_http_idle_timeout_ms(timeout_ms: int) -> str:
    for choice in HTTP_IDLE_TIMEOUT_CHOICES:
        if choice["timeout_ms"] == timeout_ms:
            return choice["label"]
    return f"{timeout_ms / 1000:g} sec"
