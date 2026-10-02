"""Mirror of pi codemode src/runtime/host.ts, running scripts on Monty.

pi runs each script in a fresh QuickJS VM in a worker thread and talks to it
through messages. Here each script checks out a session from a Monty pool (a
`monty` worker subprocess) and is driven by its snapshots: every
`feed_start`/`resume` is one short blocking job on the pool, and between jobs
the script is suspended on the host, so no thread is held while it waits on
tools.

- A call the script awaits (`tools.x(...)`, `call_tool`, the configured
  globals) is answered with a future and runs as a child coroutine of the
  execution's scope; a `FutureSnapshot` is resumed with whichever calls have
  settled. Calls start when made, so two calls made before an await run
  concurrently.
- `text`, `image`, `store`, `load` and `exit` are answered on the spot. OS
  calls (sleep, environment, files) are refused.
- Stopping a script early (the caller's cancel, `timeout_ms`, `close()`, or the
  owner being cancelled) kills its worker with SIGKILL, synchronously: a
  running feed cannot be interrupted otherwise. The pool replaces the worker.
- At the end, calls still running are cancelled with the scope.
"""

import functools
import math
import os
import re
import signal
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import tonio.colored as tonio
from pydantic_monty import (
    ClassInstance,
    ClassType,
    FunctionSnapshot,
    FutureSnapshot,
    MontyComplete,
    MontyCrashedError,
    MontyError,
    MontyRuntimeError,
    MontySyntaxError,
    MontyTypingError,
    NameLookupSnapshot,
)
from tonio.colored.sync import channel

from .. import clock
from ..declarations import (
    RenderedTool,
    namespace_class_name,
    render_stubs,
    render_tools,
    reserved_type_names,
)
from ..identifier import is_identifier
from ..types import (
    CodemodeCall,
    CodemodeCallStatus,
    CodemodeError,
    CodemodeErrorKind,
    CodemodeGlobal,
    CodemodeOutputItem,
    CodemodeResult,
    CodemodeTextItem,
    CodemodeTool,
)
from .pool import CodemodePool
from .prelude import PRELUDE, ScriptStore, image_output, json_round_trip, output_text


DEFAULT_TIMEOUT_MS = 300_000
# Execution time a script may use, not counting time suspended on tools. pi has
# no counterpart: without it a spinning script outlives a killed pidrei.
DEFAULT_MAX_EXECUTION_SECS = 60.0
# Host calls, name lookups and future resolutions per script (Monty's default
# is 1000: one text() call per item of a long list would exceed it).
MAX_SUSPENSIONS = 1_000_000

RESERVED_GLOBALS = frozenset(
    {
        "tools",
        "ALL_TOOLS",
        "text",
        "image",
        "exit",
        "store",
        "load",
        "has_tool",
        "all_settled",
        "call_tool",
        "asyncio",
        "print",
    }
)
# Host functions a script may also read as values (`f = text`).
_HOST_FUNCTIONS = ("text", "image", "exit", "store", "load", "call_tool")

_TOOLS_HINT = "ALL_TOOLS lists every tool; search_tools(query) finds tools by topic."
_FEED_TIME_LIMIT_PREFIX = "feed time limit exceeded"
_TYPE_CHECK_FAILED = "The script did not run: type checking failed."
_GATHER_HINT = (
    "Hint: asyncio.gather() does not support return_exceptions here. Use await all_settled(...) to keep "
    "the results of the calls that succeed."
)


class CancelSignal(Protocol):
    """What `execute` needs of the caller's cancel token (pidrei_ai's
    `CancelToken` fits): pi's `AbortSignal`."""

    @property
    def cancelled(self) -> bool: ...

    @property
    def reason(self) -> BaseException | None: ...

    def on_cancel(self, callback: Callable[[BaseException], None]) -> Callable[[], None]: ...


_UNSET: Any = object()
_STOP = object()
_EXIT = object()


def _comparable(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def missing_member_message(label: str, member: str, names: list[str], *, tools: bool) -> str:
    """The error for a member that does not exist, naming close matches."""
    wanted = _comparable(member)
    exact = [name for name in names if _comparable(name) == wanted]
    close = exact or [name for name in names if wanted and (wanted in _comparable(name) or _comparable(name) in wanted)]
    message = f"{label}.{member} does not exist."
    if close:
        message += f" Did you mean {', '.join(f'{label}.{name}' for name in close[:5])}?"
    elif len(names) <= 20:
        message += f" Available: {', '.join(names)}."
    if tools:
        message += f' {_TOOLS_HINT} Check for a tool with has_tool("{member}").'
    return message


def _keyword_example(rendered: RenderedTool) -> str:
    schema = rendered.tool.input_schema
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    required = schema.get("required") if isinstance(schema, Mapping) else None
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        return ""
    names = [name for name in properties if name in required]
    if all(is_identifier(name) for name in names):
        return ", ".join(f"{name}=..." for name in names)
    return "**{" + ", ".join(f"{name!r}: ..." for name in names) + "}"


@functools.cache
def _host_function_value(name: str) -> Callable[..., Any]:
    """What a script gets when it reads a host function as a value: a host
    function Monty routes back by name, so calling it is the same call."""

    def host_function(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"{name}() runs on the codemode host")

    host_function.__name__ = host_function.__qualname__ = name
    return host_function


@functools.cache
def _host_object_type(name: str) -> ClassType:
    """An empty class per host object (`Tools`, `Models`), so Monty's own
    messages name the class the stubs declare. Its id is pinned: without one,
    Monty assigns it through a process-wide cache that concurrent scripts
    would race on."""
    cls = type(name, (), {"__module__": __name__})
    return ClassType(cls, id=uuid.uuid5(uuid.NAMESPACE_URL, f"pidrei-codemode:{name}"))


def _host_object(name: str) -> ClassInstance:
    host_type = _host_object_type(name)
    return ClassInstance(host_type.value(), class_type=host_type)


def _format_timeout_ms(timeout_ms: float) -> str:
    return str(int(timeout_ms)) if float(timeout_ms).is_integer() else str(timeout_ms)


@dataclass(slots=True)
class _CallRecord:
    name: str
    started_at: float
    status: CodemodeCallStatus = "cancelled"
    duration_ms: float | None = None

    def freeze(self, now: float) -> CodemodeCall:
        duration = self.duration_ms if self.duration_ms is not None else (now - self.started_at) * 1000
        return CodemodeCall(name=self.name, status=self.status, duration_ms=duration)


@dataclass(slots=True)
class _PendingCall:
    call_id: int
    invoke: Callable[[], Any]
    record: _CallRecord | None
    # The child coroutine, closed after the scope if it never started.
    coroutine: Any = None
    started: bool = False


@dataclass(frozen=True, slots=True)
class _ExecutionSetup:
    pool: CodemodePool
    tools: tuple[RenderedTool, ...]
    globals: tuple[CodemodeGlobal, ...]
    stubs: str | None
    limits: Mapping[str, Any]


class _Execution:
    """One script run on one Monty session."""

    def __init__(
        self,
        setup: _ExecutionSetup,
        code: str,
        *,
        timeout_ms: float,
        cancel: CancelSignal | None,
        store: Mapping[str, Any] | None,
    ) -> None:
        self._setup = setup
        self._code = code
        self._timeout_ms = timeout_ms
        self._cancel = cancel
        self._store = ScriptStore(dict(store) if store is not None else None)
        self._tools = {tool.identifier: tool for tool in setup.tools}
        self._tool_names = [tool.identifier for tool in setup.tools]
        self._globals = {item.name: item for item in setup.globals}
        self._tools_object = _host_object("Tools")
        self._namespace_objects: dict[str, ClassInstance] = {}
        for item in setup.globals:
            namespace, dot, _member = item.name.partition(".")
            if dot and namespace not in self._namespace_objects:
                self._namespace_objects[namespace] = _host_object(namespace_class_name(namespace))
        self._namespaces_by_id = {wrapper.id: name for name, wrapper in self._namespace_objects.items()}

        # Guards the outcome: the flags, the session hand-over and the kill,
        # the records and the output. Never held across an await.
        self._guard = threading.Lock()
        self._finished = False
        self._stop: tuple[CodemodeErrorKind, str] | None = None
        self._session: Any = None
        self._pid: int | None = None
        self._step_done: tonio.Event | None = None
        self._calls: list[_CallRecord] = []
        self._pending: list[_PendingCall] = []
        self._output: list[CodemodeOutputItem] = []
        self._printed: list[str] = []
        self._sender, self._receiver = channel.unbounded()
        self._scope: Any = None
        self._deadline_coroutine: Any = None
        self._deadline_started = False
        # Set when `run` has returned or raised; `close()` waits on it.
        self.done = tonio.Event()

    # -- stopping -----------------------------------------------------------

    def stop(self, kind: CodemodeErrorKind, message: str) -> None:
        """End the script early with `kind`. Synchronous and callable from any
        thread: kills the worker if it has one."""
        with self._guard:
            if self._finished or self._stop is not None:
                return
            self._stop = (kind, message)
            # Under the guard: after `_finished` the worker may already be back
            # in the pool, serving someone else.
            self._kill_locked()
            self._sender.send(_STOP)

    def _kill_locked(self) -> None:
        if self._pid is None:
            return
        try:
            os.kill(self._pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _on_cancel(self, reason: BaseException) -> None:
        message = str(reason) if reason is not None and str(reason) else "Execution aborted"
        self.stop("aborted", message)

    async def _deadline(self, seconds: float) -> None:
        self._deadline_started = True
        # Parked on an Event nothing sets: the scope cancels it when the script
        # ends first.
        await tonio.Event().wait(seconds)
        self.stop("timeout", f"Execution timed out after {_format_timeout_ms(self._timeout_ms)} ms")

    def _abandon(self) -> None:
        """The owner was cancelled (or failed) mid-script: synchronous cleanup.
        Kills the worker and leaves the session's release to a detached
        coroutine, which waits for the step in flight first."""
        with self._guard:
            if self._finished:
                return
            self._finished = True
            self._kill_locked()
            step_done = self._step_done
        tonio.spawn.without_tracking(self._release_detached(step_done))

    async def _release_detached(self, step_done: tonio.Event | None) -> None:
        try:
            if step_done is not None:
                await step_done.wait()
            with self._guard:
                session, self._session = self._session, None
            if session is not None:
                await tonio.spawn_blocking(session.__exit__, None, None, None)
        except Exception:
            # The worker is dead or the pool closed: nothing left to release.
            pass

    # -- blocking steps -----------------------------------------------------

    async def _step(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run one blocking Monty call on the pool. The job runs detached, so
        cancelling the owner never interrupts a call into Monty halfway; the
        owner only stops waiting for it."""
        done = tonio.Event()
        outcome = tonio.Result()

        async def job() -> None:
            try:
                outcome.store((True, await tonio.spawn_blocking(fn, *args, **kwargs)))
            except Exception as error:
                outcome.store((False, error))
            finally:
                done.set()

        with self._guard:
            self._step_done = done
        tonio.spawn.without_tracking(job())
        await done.wait()
        stored = outcome.fetch()
        if stored is None:
            raise _StepInterrupted("the sandbox step was interrupted")
        ok, value = stored
        if not ok:
            raise value
        return value

    def _open_blocking(self) -> None:
        """Check out a session, hand it over, and feed the prelude."""
        session = self._setup.pool.monty.checkout(
            limits=dict(self._setup.limits),
            type_check=self._setup.stubs is not None,
            type_check_stubs=self._setup.stubs,
            type_check_format="concise",
        )
        session.__enter__()
        # `worker_pid` reads None while a feed is in flight: read it here.
        pid = session.worker_pid
        with self._guard:
            abandoned = self._finished
            if not abandoned:
                self._session, self._pid = session, pid
        if abandoned:
            session.__exit__(None, None, None)
            return
        all_tools = [
            {"name": tool.identifier, "description": tool.tool.description or ""} for tool in self._setup.tools
        ]
        session.feed_run(
            PRELUDE,
            inputs={"ALL_TOOLS": all_tools, "tools": self._tools_object, **self._namespace_objects},
            skip_type_check=True,
        )

    async def _release(self, session: Any) -> None:
        if session is None:
            return
        try:
            await self._step(session.__exit__, None, None, None)
        except Exception:
            pass

    # -- output -------------------------------------------------------------

    def _on_print(self, _stream: str, text: str) -> None:
        with self._guard:
            self._printed.append(text)

    def _flush_prints_locked(self) -> None:
        """Consecutive `print()` output becomes one text item, without its final
        newline, placed before the next `text()`/`image()` item."""
        if not self._printed:
            return
        text = "".join(self._printed).removesuffix("\n")
        self._printed.clear()
        self._output.append(CodemodeTextItem(text))

    def _emit(self, item: CodemodeOutputItem) -> None:
        with self._guard:
            self._flush_prints_locked()
            self._output.append(item)

    # -- calls --------------------------------------------------------------

    def _spawn(self, call: _PendingCall) -> dict[str, Any]:
        call.coroutine = self._run_call(call)
        with self._guard:
            self._pending.append(call)
            if call.record is not None:
                self._calls.append(call.record)
        self._scope.spawn(call.coroutine)
        return {"future": ...}

    async def _run_call(self, call: _PendingCall) -> None:
        call.started = True
        try:
            value = json_round_trip(await call.invoke())
            settled: dict[str, Any] = {"return_value": value}
            status: CodemodeCallStatus = "ok"
        except Exception as error:
            settled = {"exception": RuntimeError(str(error))}
            status = "error"
        with self._guard:
            # After the script ended the record keeps "cancelled".
            if self._finished:
                return
            if call.record is not None:
                call.record.status = status
                call.record.duration_ms = (clock.monotonic() - call.record.started_at) * 1000
            self._sender.send((call.call_id, settled))

    def _call_tool(self, call_id: int, identifier: str, args: tuple[Any, ...], kwargs: dict[str, Any], label: str):
        rendered = self._tools.get(identifier)
        if rendered is None:
            return {
                "exception": AttributeError(missing_member_message("tools", identifier, self._tool_names, tools=True))
            }
        if args:
            example = _keyword_example(rendered)
            if label == "call_tool":
                example = f"{identifier!r}{', ' if example else ''}{example}"
            return {"exception": TypeError(f"{label}() takes keyword arguments only, for example {label}({example}).")}
        if rendered.parameters is not None:
            dropped = {
                parameter.name
                for parameter in rendered.parameters
                if not parameter.required and not parameter.accepts_null
            }
            kwargs = {name: value for name, value in kwargs.items() if not (value is None and name in dropped)}
        try:
            payload = json_round_trip(kwargs)
        except TypeError as error:
            return {"exception": error}
        execute = rendered.tool.execute
        record = _CallRecord(name=rendered.tool.name, started_at=clock.monotonic())
        return self._spawn(_PendingCall(call_id, lambda: execute(payload), record))

    def _call_global(self, call_id: int, item: CodemodeGlobal, args: tuple[Any, ...], kwargs: dict[str, Any]):
        try:
            positional = tuple(json_round_trip(list(args)))
            keywords = json_round_trip(kwargs)
        except TypeError as error:
            return {"exception": error}
        execute = item.execute
        return self._spawn(_PendingCall(call_id, lambda: execute(positional, keywords), None))

    def _host_functions(self) -> dict[str, Callable[..., Any]]:
        def text(value: Any) -> None:
            self._emit(CodemodeTextItem(output_text(value)))

        def image(value: Any) -> None:
            self._emit(image_output(value))

        def store(key: Any, value: Any) -> None:
            self._store.store(key, value)

        def load(key: Any) -> Any:
            return self._store.load(key)

        return {"text": text, "image": image, "store": store, "load": load}

    def _answer(self, snapshot: FunctionSnapshot, host_functions: Mapping[str, Callable[..., Any]]) -> Any:
        """The answer to one call the script made: a value, an exception, a
        future, or `_EXIT`."""
        name = snapshot.function_name
        args, kwargs = snapshot.args, snapshot.kwargs
        if snapshot.object_id is not None:
            if snapshot.object_id == self._tools_object.id:
                return self._call_tool(snapshot.call_id, name, args, kwargs, f"tools.{name}")
            namespace = self._namespaces_by_id.get(snapshot.object_id)
            if namespace is None:
                return {"exception": AttributeError(f"object has no attribute {name!r}")}
            item = self._globals.get(f"{namespace}.{name}")
            if item is None:
                members = [key.partition(".")[2] for key in self._globals if key.startswith(f"{namespace}.")]
                return {"exception": AttributeError(missing_member_message(namespace, name, members, tools=False))}
            return self._call_global(snapshot.call_id, item, args, kwargs)
        if name == "exit":
            return _EXIT
        if name == "call_tool":
            if not args or not isinstance(args[0], str):
                return {"exception": TypeError("call_tool() expects a tool name")}
            return self._call_tool(snapshot.call_id, args[0], args[1:], kwargs, "call_tool")
        if name in host_functions:
            try:
                return {"return_value": host_functions[name](*args, **kwargs)}
            except Exception as error:
                return {"exception": error}
        item = self._globals.get(name)
        if item is not None:
            return self._call_global(snapshot.call_id, item, args, kwargs)
        return {"exception": NameError(f"name {name!r} is not defined")}

    async def _settled_calls(self) -> dict[int, Any] | None:
        """The calls that settled since the last resume (waiting for at least
        one), or None when the script was stopped."""
        items = [await self._receiver.receive()]
        while (item := self._receiver.receive_nowait()) is not self._receiver.Empty:
            items.append(item)
        if any(item is _STOP for item in items):
            return None
        return dict(items)

    # -- outcome ------------------------------------------------------------

    def _finish(self, *, value: Any = None, error: CodemodeError | None = None) -> tuple[CodemodeResult, Any]:
        """Settle the outcome (a stop that came first wins) and take the session
        for release."""
        with self._guard:
            self._finished = True
            if self._stop is not None:
                kind, message = self._stop
                error = CodemodeError(kind=kind, message=message)
            self._flush_prints_locked()
            now = clock.monotonic()
            calls = tuple(record.freeze(now) for record in self._calls)
            output = tuple(self._output)
            session, self._session = self._session, None
        if error is not None:
            return CodemodeResult(ok=False, output=output, calls=calls, error=error), session
        writes = self._store.writes()
        return CodemodeResult(ok=True, output=output, calls=calls, value=value, store_writes=writes), session

    def _complete(self, value: Any) -> tuple[CodemodeResult, Any]:
        """The script's value is its last line when that is an expression (pi:
        its `return` value). Monty's checker rejects a top-level `return`."""
        try:
            value = json_round_trip(value)
        except TypeError as error:
            return self._finish(error=CodemodeError(kind="script", name="TypeError", message=str(error)))
        return self._finish(value=value)

    def _failure(self, error: Exception) -> tuple[CodemodeResult, Any]:
        if isinstance(error, MontyTypingError):
            diagnostics = error.display().strip()
            return self._finish(
                error=CodemodeError(kind="script", message=diagnostics, stack=f"{_TYPE_CHECK_FAILED}\n{diagnostics}")
            )
        if isinstance(error, MontySyntaxError):
            return self._finish(
                error=CodemodeError(
                    kind="script", name="SyntaxError", message=error.display("msg"), stack=error.display()
                )
            )
        if isinstance(error, MontyRuntimeError):
            name = type(error.exception()).__name__
            message = error.display("msg")
            if name == "TimeoutError" and message.startswith(_FEED_TIME_LIMIT_PREFIX):
                seconds = self._setup.limits["max_feed_duration_secs"]
                return self._finish(
                    error=CodemodeError(
                        kind="timeout",
                        message=(
                            f"Execution timed out after {seconds:g} seconds of script execution "
                            "(time spent waiting on tools does not count)"
                        ),
                    )
                )
            stack = error.display()
            if name == "NotImplementedError" and "gather()" in message:
                stack = f"{stack}\n\n{_GATHER_HINT}"
            return self._finish(error=CodemodeError(kind="script", name=name, message=message, stack=stack))
        if isinstance(error, (MontyCrashedError, _StepInterrupted)):
            return self._finish(
                error=CodemodeError(kind="sandbox", message="The sandbox worker exited before the script settled")
            )
        return self._finish(error=CodemodeError(kind="sandbox", message=str(error)))

    # -- driving ------------------------------------------------------------

    async def run(self) -> CodemodeResult:
        unsubscribe: Callable[[], None] | None = None
        try:
            async with tonio.scope(cancel_on_exc=True) as scope:
                self._scope = scope
                try:
                    if self._cancel is not None:
                        unsubscribe = self._cancel.on_cancel(self._on_cancel)
                    if math.isfinite(self._timeout_ms):
                        self._deadline_coroutine = self._deadline(self._timeout_ms / 1000)
                        scope.spawn(self._deadline_coroutine)
                    result, session = await self._drive()
                except BaseException:
                    self._abandon()
                    raise
                scope.cancel()
            await self._release(session)
            return result
        finally:
            if unsubscribe is not None:
                unsubscribe()
            # Children the scope dropped before their first step: close them
            # rather than leave never-awaited coroutines to the collector.
            for call in self._pending:
                if not call.started:
                    call.coroutine.close()
            if self._deadline_coroutine is not None and not self._deadline_started:
                self._deadline_coroutine.close()
            self.done.set()

    async def _drive(self) -> tuple[CodemodeResult, Any]:
        if self._stop is not None:
            return self._finish()
        try:
            await self._step(self._open_blocking)
        except Exception as error:
            if isinstance(error, MontyError) and not isinstance(error, MontyCrashedError):
                return self._failure(error)
            return self._finish(
                error=CodemodeError(kind="sandbox", message=f"Failed to start the sandbox worker: {error}")
            )
        if self._stop is not None or self._session is None:
            return self._finish()

        session = self._session
        host_functions = self._host_functions()
        try:
            snapshot = await self._step(session.feed_start, self._code, print_callback=self._on_print)
            while True:
                if isinstance(snapshot, MontyComplete):
                    return self._complete(snapshot.output)
                if isinstance(snapshot, FunctionSnapshot):
                    if snapshot.is_os_function:
                        snapshot = await self._step(snapshot.resume_not_handled)
                        continue
                    answer = self._answer(snapshot, host_functions)
                    if answer is _EXIT:
                        return self._finish()
                    snapshot = await self._step(snapshot.resume, answer)
                elif isinstance(snapshot, NameLookupSnapshot):
                    name = snapshot.variable_name
                    if snapshot.object_id is None and (name in _HOST_FUNCTIONS or name in self._globals):
                        snapshot = await self._step(snapshot.resume, value=_host_function_value(name))
                    else:
                        snapshot = await self._step(snapshot.resume)
                elif isinstance(snapshot, FutureSnapshot):
                    settled = await self._settled_calls()
                    if settled is None:
                        return self._finish()
                    snapshot = await self._step(snapshot.resume, settled)
                else:
                    return self._finish(
                        error=CodemodeError(kind="sandbox", message=f"Unexpected sandbox state: {snapshot!r}")
                    )
        except Exception as error:
            return self._failure(error)


class _StepInterrupted(Exception):
    """A step whose job never reported back (the runtime shutting down)."""


def _validate_globals(globals: Iterable[CodemodeGlobal]) -> tuple[CodemodeGlobal, ...]:
    by_name: dict[str, CodemodeGlobal] = {}
    namespaces: set[str] = set()
    for item in globals:
        parts = item.name.split(".")
        if len(parts) > 2 or not all(is_identifier(part) for part in parts) or parts[0] in RESERVED_GLOBALS:
            raise ValueError(f'Invalid global name "{item.name}"')
        if item.name in by_name:
            raise ValueError(f'Global "{item.name}" is already registered')
        if len(parts) == 2:
            namespaces.add(parts[0])
        by_name[item.name] = item
    for name in namespaces:
        if name in by_name:
            raise ValueError(f'Global "{name}" conflicts with the namespace "{name}"')
    return tuple(by_name.values())


class CodemodeSandbox:
    """Runs Python scripts in Monty sessions from `pool`. The script sees
    `await tools.<name>(...)` for every registered tool, `ALL_TOOLS`,
    `has_tool`, `call_tool`, `all_settled`, the output helpers `text`, `image`,
    `exit` and `print`, `store`/`load`, and the configured globals; nothing
    else (no files, network, environment, subprocesses or sleep).

    Each `execute()` gets its own session; the sandbox only holds the tool
    table and defaults. `close()` aborts in-flight executions.

    With `type_check`, every script is checked against stubs rendered from the
    tools and globals before it runs; `stubs_preamble` declares the types the
    globals' signatures name.
    """

    def __init__(
        self,
        pool: CodemodePool,
        *,
        tools: Iterable[CodemodeTool] = (),
        globals: Iterable[CodemodeGlobal] = (),
        stubs_preamble: str = "",
        timeout_ms: float = DEFAULT_TIMEOUT_MS,
        memory_limit_bytes: int | None = None,
        max_execution_secs: float = DEFAULT_MAX_EXECUTION_SECS,
        type_check: bool = True,
    ) -> None:
        self._pool = pool
        self._guard = threading.Lock()
        self._tools: dict[str, CodemodeTool] = {}
        for tool in tools:
            self.register_tool(tool)
        self._globals = _validate_globals(globals)
        self._stubs_preamble = stubs_preamble
        self._timeout_ms = timeout_ms
        self._type_check = type_check
        self._limits: dict[str, Any] = {
            "max_feed_duration_secs": max_execution_secs,
            "max_suspensions": MAX_SUSPENSIONS,
        }
        if memory_limit_bytes is not None:
            self._limits["max_memory"] = memory_limit_bytes
        self._running: set[_Execution] = set()
        self._closed = False

    def register_tool(self, tool: CodemodeTool) -> None:
        """Raises if a tool with the same name is already registered."""
        with self._guard:
            if tool.name in self._tools:
                raise ValueError(f'Tool "{tool.name}" is already registered')
            self._tools[tool.name] = tool

    def unregister_tool(self, name: str) -> bool:
        with self._guard:
            return self._tools.pop(name, None) is not None

    @property
    def tools(self) -> list[CodemodeTool]:
        with self._guard:
            return list(self._tools.values())

    @property
    def globals(self) -> list[CodemodeGlobal]:
        return list(self._globals)

    def render_tools(self) -> list[RenderedTool]:
        """The registered tools' declarations, named as the stubs name them."""
        reserved = reserved_type_names((item.name for item in self._globals), self._stubs_preamble)
        return render_tools(self.tools, reserved_names=reserved)

    async def execute(
        self,
        code: str,
        *,
        cancel: CancelSignal | None = None,
        timeout_ms: float = _UNSET,
        store: Mapping[str, Any] | None = None,
    ) -> CodemodeResult:
        """`code` runs as a script: top-level `await` works, and a last line
        that is an expression is the script's value (`result.value`). Never
        raises for script failures; those come back as `ok=False`. The script
        uses `store(key, value)` and `load(key)` on `store`. `timeout_ms`
        (default: the sandbox's) is a deadline for the whole script, tool time
        included; `math.inf` disables it."""
        rendered = tuple(self.render_tools())
        stubs = render_stubs(rendered, self._globals, preamble=self._stubs_preamble) if self._type_check else None
        setup = _ExecutionSetup(
            pool=self._pool, tools=rendered, globals=self._globals, stubs=stubs, limits=self._limits
        )
        execution = _Execution(
            setup,
            code,
            timeout_ms=self._timeout_ms if timeout_ms is _UNSET else timeout_ms,
            cancel=cancel,
            store=store,
        )
        with self._guard:
            if self._closed:
                raise RuntimeError("Sandbox is closed")
            self._running.add(execution)
        try:
            return await execution.run()
        finally:
            with self._guard:
                self._running.discard(execution)

    async def close(self) -> None:
        """Abort in-flight executions (they finish with kind `aborted`) and
        reject new ones. Does not close the pool."""
        with self._guard:
            self._closed = True
            running = list(self._running)
        for execution in running:
            execution.stop("aborted", "Sandbox closed")
        for execution in running:
            await execution.done.wait()
