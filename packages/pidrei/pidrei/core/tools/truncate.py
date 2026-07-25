"""Mirror of pi coding-agent src/core/tools/truncate.ts.

pi keeps two byte-identical truncation implementations (agent harness and
coding-agent); the Phase 2 port in pidrei-agent is the single implementation
here, re-exported under the coding-agent module path.
"""

from pidrei_agent.harness.utils.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    GREP_MAX_LINE_LENGTH,
    TruncatedLine,
    TruncationResult,
    format_size,
    truncate_head,
    truncate_line,
    truncate_tail,
    utf8_byte_length,
)


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "GREP_MAX_LINE_LENGTH",
    "TruncatedLine",
    "TruncationResult",
    "format_size",
    "truncate_head",
    "truncate_line",
    "truncate_tail",
    "utf8_byte_length",
]
