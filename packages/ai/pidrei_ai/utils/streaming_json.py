"""Port of pi's streaming JSON parser (packages/ai/src/utils/streaming-json.ts).

Incremental, tolerant parsing of JSON text that arrives in pieces, such as
streamed tool-call arguments.

`StreamingJsonParser.append()` reads each piece once, so a stream costs time
and memory linear in its length, and `snapshot()` returns the value of the text
read so far. Snapshots are plain JSON values (dicts, lists, strings, numbers,
booleans, `None`) and never change once returned: the parser copies the
containers that are still open (the "open path") into every new snapshot and
shares completed containers and strings. Callers must not mutate them either.
The rules for incomplete text:

- An unterminated string shows the characters read so far; an unterminated
  escape is held back until complete.
- An unterminated container shows the members and elements read so far.
- A member whose key or value has not started is omitted.
- An unterminated literal shows its value once it is a prefix of `true`,
  `false` or `null`; an unterminated number shows its value when the digits
  read so far form a number (an exponent without digits is ignored).

Malformed text is read leniently, the way `repair_json()` and the finished
parse treat it: raw control characters in strings are kept, an invalid escape
such as `\\d` stays literal text, missing commas and colons are tolerated,
values in invalid positions and unknown words (`NaN`, `Infinity` among them)
are dropped, and text after the root value is ignored. The parser never
raises. Callers that need the exact value of the finished text parse it once,
e.g. with `parse_tool_call_arguments()`.

Numbers follow `json.loads`: a token without a fraction or an exponent is an
`int`, any other a `float`. A `\\uD800`-`\\uDBFF` escape followed by a
`\\uDC00`-`\\uDFFF` escape is one character, as `json.loads` reads the pair;
Python strings hold code points, not UTF-16 units, so the two escapes are
joined here instead of by string concatenation.

`append()` also returns the changes the piece made, for a writer that keeps
its own copy of the value and applies them instead of re-assigning snapshots:
`JsonSet` when a key, an array item or a finished value appears (a new string
is set once with its first text, a container empty), `JsonAppend` when the
piece extended an open string. Unterminated numbers and literals are not
reported until they end, and a high surrogate escape is not reported until
the next character shows whether its low half follows (snapshots show it
alone meanwhile, as `json.loads` reads a lone one). A container nested more
than 32 levels deep is reported as one `JsonSet` of its whole value when it
closes, so paths stay short.

Cost: copying the open path costs its member and element count. Python
strings are not ropes, so showing an open string builds it whole: that costs
one more unit per `CHARACTERS_PER_CREDIT` characters of it. Each `append()`
earns copy credit proportional to its length plus a constant; `snapshot()`
spends it and returns the previous snapshot while a fresh one costs more than
the credit. So a document that keeps a very large container or string open
(thousands of elements, tens of kilobytes) shows a preview that lags a little
behind the text, and no input makes the total work grow faster than the text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar, Final, Literal

from pidrei_ai.utils.json_parse import parse_tool_call_arguments


type JsonPath = tuple[str | int, ...]
"""A path from the root value: member keys and array indices."""


@dataclass(frozen=True, slots=True)
class JsonSet:
    """A key, an array item or a finished value appeared at `path`."""

    kind: ClassVar[Literal["set"]] = "set"
    path: JsonPath
    value: Any


@dataclass(frozen=True, slots=True)
class JsonAppend:
    """The piece extended the open string at `path` by `text`."""

    kind: ClassVar[Literal["append"]] = "append"
    path: JsonPath
    text: str


type JsonChange = JsonSet | JsonAppend
"""One change a piece of text made to the value; see the module docstring."""


class _NoValue:
    """The snapshot before a value starts (pi's `undefined`)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "NO_VALUE"

    def __reduce__(self) -> str:
        # Copies and pickles stay the one sentinel.
        return "NO_VALUE"


NO_VALUE: Final = _NoValue()

# Where a value sits in its parent: `(kind, key or index)`. `None` for a value
# in an invalid position, which is read but dropped.
type _Slot = tuple[Literal["root"], None] | tuple[Literal["member"], str] | tuple[Literal["element"], int]

_ROOT_SLOT: Final[_Slot] = ("root", None)
# Changes are reported at most this deep, so a path costs O(1) to build.
MAX_REPORTED_DEPTH: Final = 32
# Copy credit per append and per appended character. The floor keeps small
# documents fresh at every snapshot.
CREDIT_PER_APPEND: Final = 256
CREDIT_PER_CHARACTER: Final = 8
# Characters of an open string that cost one unit of credit to show.
CHARACTERS_PER_CREDIT: Final = 16
# An unterminated number longer than this is not shown until it ends, so
# `snapshot()` stays cheap.
MAX_PENDING_SCALAR_LENGTH: Final = 64

_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_NUMBER_WITHOUT_EXPONENT_DIGITS = re.compile(r"(-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?)[eE][+-]?")
_LITERALS: Final[tuple[tuple[str, Any], ...]] = (("true", True), ("false", False), ("null", None))
_STRING_STOP = re.compile(r'["\\]')
_DELIMITER = re.compile(r'[ \t\n\r,:{}\[\]"]')
_WHITESPACE: Final = frozenset(" \t\n\r")
_HEX_DIGITS: Final = frozenset("0123456789abcdefABCDEF")
_SIMPLE_ESCAPES: Final = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


class _OpenChange:
    """A change still being extended by the piece being read.

    `parts` holds the text of a string `set` or of an `append`; `None` for a
    `set` of any other value.
    """

    __slots__ = ("kind", "parts", "path", "value")

    def __init__(self, kind: Literal["set", "append"], path: JsonPath, value: Any = None, parts=None) -> None:
        self.kind = kind
        self.path = path
        self.value = value
        self.parts: list[str] | None = parts

    def close(self) -> JsonChange:
        if self.kind == "append":
            return JsonAppend(self.path, "".join(self.parts or ()))
        return JsonSet(self.path, self.value if self.parts is None else "".join(self.parts))


class _Frame:
    """An open object or array. Its `container` is never handed out while open;
    snapshots get copies."""

    __slots__ = ("container", "is_array", "key", "path", "phase", "reports_inside", "size", "slot")

    def __init__(self, container: dict | list, slot: _Slot | None, path: JsonPath | None, reports_inside: bool):
        self.container = container
        self.is_array = isinstance(container, list)
        self.slot = slot
        # Where the container sits, when its changes are reported.
        self.path = path
        # Whether changes inside it are reported; when not, its whole value is
        # reported as it closes.
        self.reports_inside = reports_inside
        # Members or elements; what a copy of the container costs.
        self.size = 0
        # Objects: the key read last, where the member value goes.
        self.key: str | None = None
        self.phase: Literal["key", "colon", "value", "comma"] = "value" if self.is_array else "key"


class _OpenString:
    """An unterminated string. Its text is kept in pieces and joined only when
    shown or finished, so reading it costs its length once."""

    __slots__ = ("change", "escape", "high", "length", "parts", "path", "slot", "target")

    def __init__(self, target: Literal["key", "value"], slot: _Slot | None, path: JsonPath | None) -> None:
        self.target = target
        self.slot = slot
        self.parts: list[str] = []
        self.length = 0
        # Escape read so far: "" for none, otherwise a backslash and up to five
        # more characters.
        self.escape = ""
        # A decoded high surrogate escape waiting for its low half; shown, not
        # yet reported.
        self.high = ""
        # Where a value string sits, when its changes are reported.
        self.path = path
        # The change this string extends, while it is the newest change of the piece.
        self.change: _OpenChange | None = None

    def text(self) -> str:
        parts = self.parts
        if len(parts) > 1:
            parts[:] = ["".join(parts)]
        return parts[0] if parts else ""


class StreamingJsonParser:
    """Incremental, tolerant JSON parser; see the module docstring for its rules and costs.

    One parser reads one stream: it is not safe to share between threads.
    """

    __slots__ = (
        "_changes",
        "_credit",
        "_dirty",
        "_done",
        "_path_cost",
        "_root",
        "_scalar",
        "_scalar_slot",
        "_snapshot",
        "_stack",
        "_string",
    )

    def __init__(self) -> None:
        self._root: Any = NO_VALUE
        self._done = False
        self._stack: list[_Frame] = []
        self._string: _OpenString | None = None
        self._scalar: list[str] = []
        self._scalar_slot: _Slot | None = None
        # Sum of `size + 1` over the open path: what a fresh snapshot copies.
        self._path_cost = 0
        self._credit = 0
        self._dirty = False
        self._snapshot: Any = NO_VALUE
        # Changes of the piece being read.
        self._changes: list[_OpenChange] = []

    def append(self, text: str) -> list[JsonChange]:
        """Read the next piece of text; returns the changes it made, in order.

        Costs time linear in `len(text)`.
        """
        self._credit += CREDIT_PER_APPEND + CREDIT_PER_CHARACTER * len(text)
        self._changes = []
        if self._string is not None:
            self._string.change = None
        index = 0
        length = len(text)
        while index < length and not self._done:
            open_string = self._string
            if open_string is not None:
                index = self._read_string(open_string, text, index)
            elif self._scalar:
                index = self._read_scalar(text, index)
            else:
                self._structural(text[index])
                index += 1
        changes = [change.close() for change in self._changes]
        self._changes = []
        return changes

    def snapshot(self) -> Any:
        """The value of the text read so far: `NO_VALUE` before a value starts,
        otherwise a plain JSON value that never changes afterwards.

        Returns the previous snapshot while the text read since did not change
        the value, or while a fresh copy of the open path is not yet paid for
        (see the module docstring).
        """
        if not self._dirty:
            return self._snapshot
        cost = self._path_cost
        pending_scalar = self._pending_scalar()
        if pending_scalar is not None:
            cost += 1
        open_string = self._string
        shows_string = open_string is not None and open_string.target == "value" and open_string.slot is not None
        if shows_string:
            cost += 1 + open_string.length // CHARACTERS_PER_CREDIT
        if cost > self._credit:
            return self._snapshot
        self._credit -= cost
        self._dirty = False
        if shows_string:
            pending = (open_string.slot, open_string.text() + open_string.high)
        else:
            pending = pending_scalar
        self._snapshot = self._copy_open_path(pending)
        return self._snapshot

    def _read_string(self, open_string: _OpenString, text: str, start: int) -> int:
        """Read string characters from `start` until the string ends or the text
        does; returns the next index."""
        index = start
        length = len(text)
        while index < length:
            if open_string.escape and self._continue_escape(open_string, text[index]):
                index += 1
                continue
            # Otherwise the character is not part of an escape: read it as a normal string character.
            stop = _STRING_STOP.search(text, index)
            end = length if stop is None else stop.start()
            if end > index:
                self._append_string(open_string, text[index:end])
            if stop is None:
                return length
            if text[end] == '"':
                self._end_string(open_string)
                return end + 1
            open_string.escape = "\\"
            index = end + 1
        return length

    def _read_scalar(self, text: str, start: int) -> int:
        """Read number or literal characters from `start` up to the next
        delimiter; returns the delimiter's index."""
        delimiter = _DELIMITER.search(text, start)
        end = len(text) if delimiter is None else delimiter.start()
        if end > start:
            self._scalar.append(text[start:end])
            if self._scalar_slot is not None:
                self._dirty = True
        if delimiter is not None:
            self._end_scalar()
        return end

    def _structural(self, char: str) -> None:
        """Handle one character outside strings."""
        if char in _WHITESPACE:
            return
        if char == '"':
            self._start_string()
        elif char == "{":
            self._open_container({})
        elif char == "[":
            self._open_container([])
        elif char in "}]":
            self._close_container()
        elif char == ":":
            frame = self._stack[-1] if self._stack else None
            if frame is not None and not frame.is_array and frame.phase == "colon":
                frame.phase = "value"
        elif char == ",":
            frame = self._stack[-1] if self._stack else None
            if frame is not None:
                frame.phase = "value" if frame.is_array else "key"
        else:
            self._scalar = [char]
            self._scalar_slot = self._value_slot()
            if self._scalar_slot is not None:
                self._dirty = True

    def _value_slot(self) -> _Slot | None:
        """Where a value starting now goes, or `None` when this position takes no value."""
        if not self._stack:
            return _ROOT_SLOT if self._root is NO_VALUE else None
        frame = self._stack[-1]
        if frame.is_array:
            return ("element", len(frame.container))
        # A missing colon is tolerated: `{"a" 1}` reads as `{"a": 1}`.
        if frame.phase in ("value", "colon") and frame.key is not None:
            return ("member", frame.key)
        return None

    def _start_string(self) -> None:
        frame = self._stack[-1] if self._stack else None
        # A missing comma is tolerated: `{"a": 1 "b": 2}` reads the second member.
        if frame is not None and not frame.is_array and frame.phase in ("key", "comma"):
            self._string = _OpenString("key", None, None)
            return
        slot = self._value_slot()
        path = self._reported_path(slot)
        open_string = _OpenString("value", slot, path)
        self._string = open_string
        self._place(slot, "")
        if path is not None:
            open_string.change = self._report(_OpenChange("set", path, parts=[]))

    def _reported_path(self, slot: _Slot | None) -> JsonPath | None:
        """The path of a value at `slot` in the innermost open container, when its changes are reported."""
        if slot is None:
            return None
        if slot[0] == "root":
            return ()
        frame = self._stack[-1]
        if not frame.reports_inside or frame.path is None:
            return None
        return (*frame.path, slot[1])

    def _report(self, change: _OpenChange) -> _OpenChange:
        self._changes.append(change)
        return change

    def _continue_escape(self, open_string: _OpenString, char: str) -> bool:
        """Continue an escape with `char`; returns False when `char` is not part
        of it and must be read normally."""
        if open_string.escape == "\\":
            decoded = _SIMPLE_ESCAPES.get(char)
            if decoded is not None:
                open_string.escape = ""
                self._append_string(open_string, decoded)
                return True
            if char == "u":
                open_string.escape = "\\u"
                return True
            # Invalid escape: keep the backslash as text and read the character
            # normally, as repair_json() does.
            open_string.escape = ""
            self._append_string(open_string, "\\")
            return False
        if char in _HEX_DIGITS:
            open_string.escape += char
            if len(open_string.escape) < 6:
                return True
            hex_digits = open_string.escape[2:]
            code = int(hex_digits, 16)
            open_string.escape = ""
            self._append_code_point(open_string, code)
            return True
        # `\u` without four hex digits: keep it as text and read the character
        # normally, as repair_json() does.
        literal = open_string.escape
        open_string.escape = ""
        self._append_string(open_string, literal)
        return False

    def _append_code_point(self, open_string: _OpenString, code: int) -> None:
        """Append a decoded `\\uXXXX` escape, joining a surrogate pair the way `json.loads` does."""
        if 0xDC00 <= code <= 0xDFFF and open_string.high:
            high = ord(open_string.high)
            open_string.high = ""
            self._append_string(open_string, chr(0x10000 + ((high - 0xD800) << 10) + (code - 0xDC00)))
        elif 0xD800 <= code <= 0xDBFF:
            # A previous high surrogate stays alone.
            self._append_string(open_string, "")
            open_string.high = chr(code)
            if open_string.target == "value" and open_string.slot is not None:
                self._dirty = True
        else:
            self._append_string(open_string, chr(code))

    def _append_string(self, open_string: _OpenString, text: str) -> None:
        if open_string.high:
            # Whatever follows a high surrogate escape, it is not its low half.
            text = open_string.high + text
            open_string.high = ""
        if not text:
            return
        open_string.parts.append(text)
        open_string.length += len(text)
        if open_string.target != "value" or open_string.slot is None:
            return
        self._dirty = True
        path = open_string.path
        if path is None:
            return
        # Extend this string's newest change while nothing came after it;
        # otherwise report an append.
        change = open_string.change
        if change is not None and self._changes and self._changes[-1] is change:
            change.parts.append(text)  # type: ignore[union-attr]
            return
        open_string.change = self._report(_OpenChange("append", path, parts=[text]))

    def _end_string(self, open_string: _OpenString) -> None:
        self._append_string(open_string, "")
        self._string = None
        frame = self._stack[-1] if self._stack else None
        if open_string.target == "key":
            if frame is None:
                return
            frame.key = open_string.text()
            frame.phase = "colon"
            return
        self._place(open_string.slot, open_string.text())
        self._value_ended(frame)

    def _end_scalar(self) -> None:
        token = "".join(self._scalar)
        slot = self._scalar_slot
        self._scalar = []
        self._scalar_slot = None
        value = _scalar_value(token)
        if value is not NO_VALUE:
            self._place(slot, value)
            path = self._reported_path(slot)
            if path is not None:
                self._report(_OpenChange("set", path, value))
        self._value_ended(self._stack[-1] if self._stack else None)

    def _pending_scalar(self) -> tuple[_Slot, Any] | None:
        """The value of the unterminated scalar at the end of the text, if it shows."""
        slot = self._scalar_slot
        if slot is None or not self._scalar:
            return None
        if len(self._scalar) > 1:
            self._scalar[:] = ["".join(self._scalar)]
        token = self._scalar[0]
        if len(token) > MAX_PENDING_SCALAR_LENGTH:
            return None
        value = _pending_scalar_value(token)
        return None if value is NO_VALUE else (slot, value)

    def _open_container(self, container: dict | list) -> None:
        slot = self._value_slot()
        path = self._reported_path(slot)
        self._place(slot, container)
        self._stack.append(_Frame(container, slot, path, path is not None and len(path) < MAX_REPORTED_DEPTH))
        self._path_cost += 1
        self._dirty = True
        # A fresh container: the working one keeps changing, and a change never does.
        if path is not None:
            self._report(_OpenChange("set", path, [] if isinstance(container, list) else {}))

    def _close_container(self) -> None:
        # A stray closer outside any container is ignored.
        if not self._stack:
            return
        frame = self._stack.pop()
        self._path_cost -= frame.size + 1
        self._dirty = True
        # Too deep to report inside: the whole value, final now that it is closed.
        if frame.path is not None and not frame.reports_inside:
            self._report(_OpenChange("set", frame.path, frame.container))
        self._value_ended(self._stack[-1] if self._stack else None)

    def _value_ended(self, frame: _Frame | None) -> None:
        """A value in `frame`, or the root value when there is no frame, has ended."""
        if frame is None:
            # Text after the root value is ignored, but only once a root value exists.
            if self._root is not NO_VALUE:
                self._done = True
            return
        frame.phase = "comma"

    def _place(self, slot: _Slot | None, value: Any) -> None:
        """Set the value at `slot` in the open path; nothing for a value in an invalid position."""
        if slot is None:
            return
        self._dirty = True
        kind, where = slot
        if kind == "root":
            self._root = value
            return
        frame = self._stack[-1]
        container = frame.container
        if kind == "element":
            if where == len(container):
                container.append(value)  # type: ignore[union-attr]
                frame.size += 1
                self._path_cost += 1
            else:
                container[where] = value  # type: ignore[index]
            return
        if where not in container:
            frame.size += 1
            self._path_cost += 1
        container[where] = value  # type: ignore[index]

    def _copy_open_path(self, pending: tuple[_Slot, Any] | None) -> Any:
        """A fresh snapshot: copies of the open containers, innermost first, each
        holding the copy of its open child."""
        stack = self._stack
        if not stack:
            return pending[1] if pending is not None and pending[0][0] == "root" else self._root
        child: Any = NO_VALUE
        child_slot: _Slot | None = None
        innermost = len(stack) - 1
        for depth in range(innermost, -1, -1):
            frame = stack[depth]
            copy = list(frame.container) if frame.is_array else dict(frame.container)
            if depth == innermost and pending is not None:
                _set_slot(copy, pending[0], pending[1])
            if child is not NO_VALUE and child_slot is not None:
                _set_slot(copy, child_slot, child)
            child = copy
            child_slot = frame.slot
        # The outermost open container is the root unless it sits in an invalid position.
        return child if child_slot is not None and child_slot[0] == "root" else self._root


class ToolCallArgumentsStream:
    """The arguments of one streaming tool call.

    It holds the raw text, for the one full parse when the call ends, and a
    `StreamingJsonParser` for the partial arguments in between. Providers keep
    one per call in their own stream state, never on the content block, so the
    raw text never reaches partial messages or transcripts.
    """

    __slots__ = ("_arguments", "_parser", "_parts", "_snapshot")

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._parser = StreamingJsonParser()
        self._snapshot: Any = NO_VALUE
        self._arguments: dict[str, Any] = {}

    @property
    def text(self) -> str:
        """The raw argument text received so far."""
        parts = self._parts
        if len(parts) > 1:
            parts[:] = ["".join(parts)]
        return parts[0] if parts else ""

    def append(self, delta: str) -> dict[str, Any]:
        """Append a piece of argument text; returns the partial arguments, a new
        dict whenever they changed."""
        if delta:
            self._parts.append(delta)
        self._parser.append(delta)
        return self._partial()

    def replace(self, text: str) -> tuple[str | None, dict[str, Any]]:
        """Take the provider's complete argument text.

        When it continues the text so far, returns the missing piece as the
        delta; otherwise starts over with `text` and returns `None` as the
        delta. Either way returns the partial arguments of `text`:
        `(delta, arguments)`.
        """
        current = self.text
        if text.startswith(current):
            delta = text[len(current) :]
            return delta, (self._partial() if delta == "" else self.append(delta))
        self._parts = []
        self._parser = StreamingJsonParser()
        return None, self.append(text)

    def finish(self) -> tuple[dict[str, Any], str | None]:
        """The final arguments: one full parse of the complete text, which
        reports malformed arguments. Returns `(arguments, arguments_error)`."""
        return parse_tool_call_arguments(self.text)

    def _partial(self) -> dict[str, Any]:
        snapshot = self._parser.snapshot()
        if snapshot is self._snapshot:
            return self._arguments
        self._snapshot = snapshot
        # Tool arguments are an object; any other partial value shows as no arguments yet.
        self._arguments = snapshot if isinstance(snapshot, dict) else {}
        return self._arguments


def _scalar_value(token: str) -> Any:
    """The value of a complete number or literal token; `NO_VALUE` for anything else, which is dropped."""
    for literal, value in _LITERALS:
        if token == literal:
            return value
    return _number(token) if _NUMBER.fullmatch(token) else NO_VALUE


def _pending_scalar_value(token: str) -> Any:
    """The value an unterminated number or literal token shows."""
    for literal, value in _LITERALS:
        if literal.startswith(token):
            return value
    if _NUMBER.fullmatch(token):
        return _number(token)
    mantissa = _NUMBER_WITHOUT_EXPONENT_DIGITS.fullmatch(token)
    return NO_VALUE if mantissa is None else _number(mantissa[1])


def _number(token: str) -> Any:
    """A number token as `json.loads` reads it. An integer past Python's
    digit limit, which `json.loads` rejects too, is dropped."""
    if "." in token or "e" in token or "E" in token:
        return float(token)
    try:
        return int(token)
    except ValueError:
        return NO_VALUE


def _set_slot(container: dict | list, slot: _Slot, value: Any) -> None:
    kind, where = slot
    if kind == "element" and where == len(container):
        # An unterminated scalar is not in the working array yet.
        container.append(value)  # type: ignore[union-attr]
    elif kind != "root":
        container[where] = value  # type: ignore[index]
