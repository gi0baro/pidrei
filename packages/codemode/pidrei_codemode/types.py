"""Mirror of pi codemode src/types.ts.

pi has one `CodemodeTool` type for tools and globals, with `spread` choosing
whether a global receives its argument list or its first argument. Python calls
carry positional and keyword arguments separately, so the two kinds are two
types here: a tool takes keyword arguments only (`CodemodeTool.execute(args)`),
a global takes both (`CodemodeGlobal.execute(args, kwargs)`).
"""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


# A JSON Schema document. Only used to render declarations; values are not
# validated against it.
type CodemodeJsonSchema = Mapping[str, Any] | bool


@dataclass(frozen=True, slots=True)
class CodemodeTool:
    # The script calls tools as `await tools.<id>(arg=value, ...)`, where `<id>`
    # is the name made a valid identifier (see `to_codemode_identifier`).
    name: str
    # Receives the keyword arguments the script passed, after a JSON round
    # trip. The return value must be JSON-serializable; a raised error surfaces
    # in the script as a `RuntimeError` with the same message.
    execute: Callable[[dict[str, Any]], Awaitable[Any]]
    # Shown before the declaration in `render_tool_samples`, and listed in
    # `ALL_TOOLS`.
    description: str | None = None
    # Schema of the keyword arguments; `**args: Any` when omitted.
    input_schema: CodemodeJsonSchema | None = None
    # Schema of the returned value; `Any` when omitted.
    output_schema: CodemodeJsonSchema | None = None


@dataclass(frozen=True, slots=True)
class CodemodeGlobal:
    # A top-level function (`search_tools`), or `<namespace>.<member>`, which
    # groups members into a namespace object (`models.classify`). Scripts await
    # every global.
    name: str
    # Receives the positional and keyword arguments the script passed, after a
    # JSON round trip; the same contract as `CodemodeTool.execute` otherwise.
    execute: Callable[[tuple[Any, ...], dict[str, Any]], Awaitable[Any]]
    # The parameter list and return type in the type-check stubs, for example
    # `(query: str, limit: int = 8) -> list[ToolInfo]`. Types it names that are
    # not built in come from the sandbox's `stubs_preamble`.
    signature: str = "(*args: Any, **kwargs: Any) -> Any"


@dataclass(frozen=True, slots=True)
class CodemodeTextItem:
    text: str
    # `print()` output (pi: `console?: true`).
    console: bool = False
    type: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class CodemodeImageItem:
    # Base64.
    data: str
    mime_type: str
    type: Literal["image"] = "image"


# One item of the script's output, in the order the script produced it:
# `text()` and `print()` produce text items, with `console=True` for `print()`,
# and `image()` image items.
type CodemodeOutputItem = CodemodeTextItem | CodemodeImageItem

type CodemodeCallStatus = Literal["ok", "error", "cancelled"]


@dataclass(frozen=True, slots=True)
class CodemodeCall:
    name: str
    status: CodemodeCallStatus
    duration_ms: float


type CodemodeErrorKind = Literal[
    # The script raised, failed to parse, or failed the type check. `name` and
    # `stack` come from the script's error.
    "script",
    # The overall deadline expired, or the script used up its execution time.
    # The worker was terminated.
    "timeout",
    # The caller cancelled or the sandbox was closed. The worker was terminated.
    "aborted",
    # The worker failed outside the script's control (it could not start, or
    # it died).
    "sandbox",
]


@dataclass(frozen=True, slots=True)
class CodemodeError:
    kind: CodemodeErrorKind
    message: str
    name: str | None = None
    stack: str | None = None


@dataclass(frozen=True, slots=True)
class CodemodeStoreWrites:
    """Keys the script changed with `store()`. Only successful executions
    report writes."""

    set: Mapping[str, Any] = field(default_factory=dict)
    # Keys stored as `None`.
    delete: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CodemodeResult:
    """`output` is kept for failed executions too, up to the failure. `exit()`
    completes with `value` `None`. `store_writes` is set when `ok`, `error`
    when not."""

    ok: bool
    output: tuple[CodemodeOutputItem, ...]
    calls: tuple[CodemodeCall, ...]
    value: Any = None
    store_writes: CodemodeStoreWrites | None = None
    error: CodemodeError | None = None
