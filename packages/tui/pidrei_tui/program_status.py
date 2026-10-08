"""Mirror of pi tui src/program-status.ts.

Program Status Protocol (OSC 7501): a program tells the terminal whether it is
idle, working, blocked on the user, done, or failed. Only the root record is
supported.

Spec: https://www.superlogical.com/rex/docs/build/program-status
"""

import base64
import re
from dataclasses import dataclass
from typing import Literal


type ProgramStatusState = Literal["idle", "working", "blocked", "done", "error", "clear"]
type ProgramStatusKind = Literal["permission", "question", "auth"]


@dataclass(frozen=True, slots=True)
class ProgramStatus:
    # `clear` removes the status instead of reporting one.
    state: ProgramStatusState
    # Stable program name, `[A-Za-z0-9_.+-]{1,32}`. Other values are omitted.
    app: str | None = None
    # What a blocked program waits for. Omitted for other states.
    kind: ProgramStatusKind | None = None
    # One human-readable line. Control characters become spaces; longer text is cut to the spec limit.
    message: str | None = None


# Feature detection query. A supporting terminal replies with the same body.
PROGRAM_STATUS_QUERY = "\x1b]7501;?\x1b\\"

# `\Z` is JavaScript's `$` (Python's also matches before a final "\n").
_REPLY_RE = re.compile(r"^\x1b\]7501;\?[^\x07\x1b]*(?:\x07|\x1b\\)\Z")
_APP_RE = re.compile(r"^[A-Za-z0-9_.+-]{1,32}\Z")
_CONTROL_CHARACTERS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]+")
# Decoded `msg` limit. Its base64 encoding stays under the 2732-byte encoded limit.
_MAX_MESSAGE_BYTES = 2048


def is_program_status_reply(sequence: str) -> bool:
    """The reply to `PROGRAM_STATUS_QUERY`. Later spec revisions may add pairs after the `?`."""
    return _REPLY_RE.match(sequence) is not None


def _utf8(text: str) -> bytes:
    # Node's `Buffer.from(text, "utf8")` writes a lone surrogate as U+FFFD instead of failing.
    return text.encode("utf-8", "replace")


def _truncate_utf8(text: str, max_bytes: int) -> str:
    encoded = _utf8(text)
    if len(encoded) <= max_bytes:
        return text
    # Cut at a character boundary: a character the limit splits is dropped whole.
    return encoded[:max_bytes].decode("utf-8", "ignore")


def format_program_status(status: ProgramStatus) -> str:
    """Encode a status report. Terminals discard reports whose text contains
    control characters, so they are replaced."""
    pairs = [f"state={status.state}"]
    if status.app is not None and _APP_RE.match(status.app):
        pairs.append(f"app={status.app}")
    if status.state == "blocked" and status.kind:
        pairs.append(f"kind={status.kind}")
    message = _truncate_utf8(_CONTROL_CHARACTERS_RE.sub(" ", status.message or "").strip(), _MAX_MESSAGE_BYTES)
    if message:
        pairs.append(f"msg={base64.b64encode(_utf8(message)).decode('ascii')}")
    return f"\x1b]7501;{':'.join(pairs)}\x1b\\"
