# AGENTS.md

Rules and context for coding agents working on PiDrei. Read it in full before
changing anything: most of these rules exist because breaking them caused a
real defect, and several are easy to break without noticing.

If `.agents/working-agreements.md` exists, read it before starting.

## What PiDrei is

PiDrei is a behavioural port of the [Pi coding agent](https://github.com/earendil-works/pi)
(TypeScript) to free-threaded CPython 3.14+, running on the
[TonIO](https://github.com/gi0baro/tonio) runtime, with
[httpunk](https://github.com/gi0baro/httpunk) and
[punkreq](https://github.com/gi0baro/punkreq) for HTTP.

Two goals carry equal weight:

1. **Behavioural parity with Pi.** Same observable behaviour, same ordering
   contracts, same strings the model sees, same session file format. Pi's own
   test suites, mirrored module by module, are the acceptance spec.
2. **Exercising the TonIO ecosystem on a large, genuinely multi-threaded
   program.** PiDrei is multi-threaded by design: coroutines run in parallel on
   real threads. A port that serialises everything to imitate JavaScript's
   single thread defeats the purpose.

What is ported is Pi's *behaviour*, never its execution model or code shape.
JavaScript turns, microtask ordering and "this runs synchronously so nothing
can interleave" do not exist here. State the Pi promise as observable
behaviour, then deliver it with TonIO primitives. "Closer to Pi's shape" is
never, on its own, an argument for a design.

Platform: POSIX only (TonIO is Unix-only). Pi's win32 branches are never
ported.

## Repository layout

uv workspace with a virtual root; members under `packages/` mirror Pi's
monorepo:

| Package | Import | Pi counterpart |
|---|---|---|
| `packages/ai` (`pidrei-ai`) | `pidrei_ai` | `packages/ai` |
| `packages/agent` (`pidrei-agent`) | `pidrei_agent` | `packages/agent` |
| `packages/tui` (`pidrei-tui`) | `pidrei_tui` | `packages/tui` (must never import `pidrei_ai`) |
| `packages/pidrei` (`pidrei`) | `pidrei` | `packages/coding-agent` |
| `packages/{protocol,client,server}` | `pidrei_protocol` etc. | transport remnants, unpublished workspace members |

Other places:

- `scripts/`: classifier for upstream deltas (`upstream_diff.py`), static
  audit (`audit.py`), release gate, release notes, brew formula, blocking-fs
  detector.
- `packages/ai/pidrei_ai/providers/data/`: the generated model catalog,
  committed (never generated at build time).
- `packages/pidrei/pidrei/docs/`: user documentation shipped in the wheel.
- `packages/pidrei/pidrei/examples/extensions/`: example extensions.
- `packages/pidrei/pidrei/CHANGELOG.md`: the changelog.
- `.last_upstream_ref` + `packages/pidrei/pidrei/upstream.py`: the Pi commit
  this tree is a port of (a test asserts they agree).

## Commands

- Environment: `uv sync --all-packages` from the repo root. A plain `uv sync`
  drops the workspace members.
- `make lint` (ruff check + format check on `packages`), `make format`,
  `make audit` (static checks for defects the suites structurally cannot see),
  `make test`.
- `make release-check`, `make models-data` (catalog regen, needs network),
  `make upstream-diff` / `make upstream-bump REF=<sha>` (needs a Pi checkout;
  `PI_ROOT` defaults to `~/Downloads/code/pi`).
- `make test-fs-detect`: the suite with the blocking-fs detector, which
  reports every filesystem call made on a runtime worker.

### Running tests

- **Every test command runs under `timeout 60`.** The full suite takes about
  35–45 s. Anything past 60 s is a hang: never rerun with a bigger budget.
- Run the suite in slices:
  - `packages/ai/tests`
  - `packages/{agent,tui,server,client,protocol}/tests`
  - `packages/pidrei/tests/test_[a-f]*.py`, `test_[g-r]*.py`,
    `test_[s-z]*.py`, `test_[0-9]*.py` (the numbered regression files are
    easy to forget).

  Type the globs unquoted in each command: a shell loop over quoted glob
  strings silently runs nothing ("no tests ran").
- When a run hangs, rerun the smallest enclosing scope with `-v`: locally the
  last test name printed is the stuck one. In CI logs, where a line appears
  only once complete, it is the test after the last finished one.
  `-o faulthandler_timeout=N` (N well under 60) dumps stacks.
- CI runs `make test PYTEST_ARGS=-s` on Linux and macOS, for 3.14t and 3.15t.
  Read failures with `gh run view <id> --log-failed`, and grep CI logs for
  `UNHANDLED`: a crashed detached coroutine never fails the run by itself,
  and its report goes to stdout, which capture swallows without `-s`.
- Hang, flake and freeze techniques (forcing a suspected race, macOS
  differences and how to reproduce them, the pty stress driver):
  `spec/debugging.md`.

## The runtime rules

The architecture behind these rules is in `spec/concurrency.md`; the
terminal UI's is in `spec/ui-island.md`.

### Never block the runtime

Filesystem I/O goes through `tonio.colored.fs` (`fs.Path`, `fs.open`,
`wrap_file`) or an explicit `spawn_blocking` / `map_blocking` dispatch. Never
stdlib `open`, `os.path.exists`, `os.makedirs` and friends in code reachable
from the runtime.

- The argument is safety, not speed. A blocked worker is one the scheduler
  cannot use, and nobody can enumerate what else is in flight. `fs` being
  slower than stdlib is expected and is not an argument against it.
- `fs` and `spawn_blocking` are the same mechanism (`fs.Path` methods are
  `spawn_blocking` underneath), so choosing between them is style, never
  correctness.
- When a value is an `fs.Path`, use it as one: join with `/`, walk with
  `.parent`, await its methods. Don't unwrap it into `os.path.join` plus a
  blocking helper.
- **The `*_blocking` convention.** Anything doing I/O the runtime has
  primitives for (fs, net, subprocesses, pipes, fds) is `async def` or
  returns an awaitable. The raw blocking calls live only in functions named
  `*_blocking` (the one suffix: no `_sync`, no `_inner`), which run on the
  pool and are never called from anywhere else; an `async def *_blocking`
  is a contradiction. `make audit` (check 7) enforces it.
- Readiness-driven fd I/O (`os.read`/`os.write` on a non-blocking fd after
  its readiness wait, `fcntl`, `termios`, `isatty`, `fstat`) is not blocking
  I/O and needs no offload.
- **After `EAGAIN`, clear readiness with the tick-guarded
  `ScheduledIO.clear_r()`/`clear_w()`, never `consume_r()`/`consume_w()`.**
  `consume_*` drains readiness unconditionally, so an edge arriving between
  the failed read/write and the consume is lost and the next wait parks
  forever: an fd pump wedged this way froze the whole TUI silently.
- `importlib.resources` reads count, and get wrapped in `spawn_blocking`.
- `tempfile.gettempdir()` probes the filesystem on its first call. It is
  resolved once at import as an `fs.Path` constant (`pidrei.config.TEMP_DIR`;
  the TUI keeps its own), never called at a use site.
- Outside the rule by construction: module-level code that runs at import,
  before `tonio.run`, and bodies already on the blocking pool. Making such an
  import lazy moves its body onto a worker.
- Python's own internal file access (module imports, traceback/linecache
  formatting) is not PiDrei's I/O. Never offload it.
- Stdio goes through `pidrei.core.output_guard`; ruff bans `sys.stdout`,
  `sys.stderr` and `traceback.print_*`.
- When converting a sync accessor that does I/O, prefer hoisting the I/O into
  an explicit load/save step over making the accessor async.

### Concurrency

- **Races are never optional.** Every race, ordering or thread-safety issue
  gets fixed, however narrow the window or benign the outcome looks. In
  reviews they are findings, never "hardening". Frame fixes around
  publication order and guards, never around removing an await that is
  legitimately needed.
- **Locks:**
  - `threading.Lock` / `RLock` guard short synchronous critical sections and
    are **never held across an `await`**, anywhere.
  - A TonIO `sync.Lock` is for critical sections that suspend.
  - Use `RLock` only where user callbacks can synchronously re-enter.
  - Keep lock order consistent.
- **Wait vs fire-and-forget.** For each Pi call, classify it as run-and-wait
  or fire-and-forget, then choose the PiDrei construct with the same
  semantics. Function colour is just how each language encodes the
  distinction; matching colours is never the goal.
  - A Pi sync call becoming `await` here (because it does I/O) is fine: it
    still waits.
  - Fire-and-forget is `tonio.spawn.without_tracking`. Never drop a
    coroutine, never inline-await what Pi detached.
  - An un-awaited Pi async call (`void f()`) still runs its body up to the
    first `await` in the caller. Port that prefix inline; spawn only the rest.
    Capture the arguments at event time, not inside the spawned body.
- **Published values are immutable.** Messages, content blocks and usage are
  frozen dataclasses. Producers build through the builders in
  `pidrei_ai/builders.py`, and a Pi mutation translates to constructing a new
  value (`dataclasses.replace`).
- **Shared state is published as one value.** Config services and session
  state publish immutable snapshots ("epochs") that are swapped atomically;
  readers pin one read. Never publish a pair as two separate writes.
- **No unguarded process globals.** The whole suite runs in one TonIO
  runtime, so a test that leaves module-level state set changes every later
  test. New module-level mutable state ships with a fail-loud autouse guard in
  the conftest of every package whose tests can reach its setter. The guard
  resets the state and warns, naming the polluting test (see
  `_capability_overrides_guard`).
- **Clocks:** production code reads time only through `pidrei_ai.utils.clock`
  and `pidrei_tui.clock` (ruff TID251 enforces this). Those are the test
  seams.
- Work still running after `main()` returns is a leak to fix where it leaks,
  not a lifecycle case to design or document around.

### The TonIO contract

Current as of TonIO 0.10.x; recheck these when bumping TonIO. When a design
depends on a TonIO behaviour not covered here, ask; don't infer it from
experiments.

- **There are no tasks**, only coroutines. `tonio.spawn(coro)` runs a
  coroutine concurrently and returns a handle. The work starts before anyone
  awaits the handle.
- **Awaiting a spawn handle re-raises the child's exception**, wrapped in
  `ExceptionGroup("SpawnExceptionGroup", [err])`; unwrap the leaf where needed.
  The *scope join* and `spawn.without_tracking` re-raise nothing. A crash is
  only reported on stdout (`UNHANDLED ...`, in debug builds). Children report
  outcomes through Events or channels they settle themselves. An exception in
  a scope body while a child waits on work the body was meant to produce is
  a deadlock. Where a scope's body can raise while its children are parked
  (a `try` inside the body, a test's assertions), use
  `tonio.scope(cancel_on_exc=True)`: the children are cancelled instead.
- **`scope.cancel()` only sets a flag.** Cancellation happens when the scope
  exits, and cancels the whole coroutine chain. A parked child gets
  `CancelledError` and unwinds; a running child gets it at its next
  suspension point. From then on every `await` in that chain raises
  `CancelledError` again (TonIO ≥ 0.10.2). `finally` blocks do run, but an
  `await` inside one raises too, so the cleanup after it never happens.
  Therefore **cancel-time cleanup is synchronous** (or a detached
  `spawn.without_tracking`), never an `await`.
- **`tonio.time.timeout` is sound.** A coroutine still running at the
  deadline completes later and its result is discarded, so it must never
  *return* a resource. Hand the resource over under a lock instead, and close
  it yourself if the deadline won.
- **Cancelling an awaited `spawn_blocking`** interrupts the job by injecting
  an exception into its thread, but does not wait for it to finish unwinding
  (TonIO ≥ 0.10.3).
  The exception lands only when the thread next runs Python bytecode. The
  awaiter must not free state the job still uses.
- **Bounded waits:** `await event.wait(timeout)` then read `event.is_set()`.
  Don't wrap `event.wait()` in `tonio.time.timeout`. Composed waits use
  `await tonio.Waiter.any(ev1, ev2, ...)`; note that its `timeout` is in
  **microseconds**, while `Event.wait` takes seconds. `Waiter(*events)` is an
  AND.
- **Never put a timeout on a channel `receive()`.** If a design seems to need
  "receive or deadline", restructure it.
- **Channels:**
  - `channel.unbounded()`: `send` is sync and thread-safe.
  - `channel.channel(size)`: async `send`, which suspends while the channel
    is full. Use it for backpressure.
  - After `sender.close()` (idempotent), `receive()` drains what is queued,
    then raises `BrokenPipeError`. It never returns `None`.
  - `receive_nowait()` returns the `receiver.Empty` sentinel when nothing is
    queued.
- `tonio.Result` replaces one-element list boxes handed between coroutines.
  Switching to it does not change locking needs.
- `Event` and `ScheduledIO` are bare Rust classes: no `__dict__`, no
  subclassing, no attribute patching.
- `CancelledError` is the only cancellation signal. TonIO never raises
  `GeneratorExit` into coroutines.
- Import TonIO submodules as modules (`from tonio.colored.sync import
  channel`, then `channel.unbounded()`). Never import from `tonio._*`
  (`make audit` checks this).
- **Never monkeypatch or alias-swap TonIO**, not even a module's `tonio`
  name in a test. It is one runtime for the whole process. To observe spawned
  work, hook PiDrei code instead (wrap a method, set an Event).
- `UNHANDLED ...` lines are an alarm, not the defect. Find the coroutine
  whose exception went unretrieved and guard it. Long-lived loops must never
  die silently.

### Exceptions

- Catch `CancelledError` only to deal with it. `except CancelledError: raise`
  is pointless.
- Never catch `GeneratorExit`; cleanup goes in `finally`.
- No `isinstance`/`type` checks inside `except BaseException`: use dedicated
  arms, and `except*` for exception groups.
- `except BaseException: <sync, error-only cleanup>; raise` is fine. Arms that
  forward or swallow errors use `except Exception`.
- Never catch pyo3/TonIO panics.
- No `await` in a `finally`/`except` reachable by cancellation: on a
  cancelled chain it raises `CancelledError` again (see the TonIO contract).
  Branch on whether the `finally` is unwinding a `CancelledError` and do only
  sync work there, as `pidrei_ai.utils.http.finish_body` does. The same holds
  for code that catches `CancelledError`: it cannot go on awaiting.
- Errors never cross the `tonio.run` boundary: `main()` reports and returns
  an exit code.

### Code conventions

- **Callbacks are async-only** in the extension API (`pi.on`, command
  handlers, factories, routers), and generally:
  `Callable[..., Awaitable[T]]`, never `T | Awaitable[T]` unions, never
  `inspect.isawaitable` checks. Exception: the interactive UI's input path
  (`handle_input`/`handle_mouse` and input-path callbacks) is synchronous;
  slow work is spawned.
- **No useless async wrappers.** `async def f(): return await g()` becomes
  `def f(): return g()`, closures included. Rewrapping a bound coroutine
  method becomes `staticmethod(...)`.
- **Async construction** uses `obj = await Cls(...)`: a plain `__init__`
  plus `__await__` delegating to an `async def _start()` that returns `self`.
  No `open()`/`create()` classmethods. `close()` tolerates a never-awaited
  instance.
- **Prefer the ecosystem's primitives** (TonIO, httpunk, punkreq, stdlib)
  over hand-rolled protocol or runtime code. Enumerate what the stack provides
  first. HTTP goes through the seam in `pidrei_ai/utils/http.py`.
- **Extend the existing mechanism first.** When X doesn't follow the rules Y
  already follows, route X through Y's mechanism. No new modules, package
  moves or unifications as part of the fix.
- **No test-shaped production code.** No `getattr`/`hasattr` defences or
  indirection so stubs pass. Write against the real invariants; fix the test
  double (DI, monkeypatch, extend the stub).
- Prefer stdlib when 3.14 provides it. Check free-threading safety: for
  example `uuid.uuid7()` is wrapped in a lock because its counter is not
  thread-safe.

## Dependencies

- The stack is TonIO, httpunk and punkreq. HTTP goes through
  `pidrei_ai/utils/http.py`; WebSockets are `websockets`' sans-io protocol
  over httpunk's upgraded connection (`pidrei_ai/utils/websocket.py`). No
  asyncio/anyio-based libraries (httpx, vendor provider SDKs, pydantic-ai):
  PiDrei ports Pi's own adapters and SSE decoding, as Pi uses SDKs only as
  HTTP clients.
- Every native dependency must support free-threading (cp314t wheels,
  `gil_used=false`). One that re-enables the GIL on import makes TonIO refuse
  to start. Check before adding a dependency.
- Streaming requests set a `read` timeout and leave `total=None`: a total
  timeout would kill a legitimately long stream.
- A suspected TonIO/httpunk/punkreq bug gets a one-shot repro outside the
  repo and is reported upstream. PiDrei carries a workaround only until the
  fixed release is pinned.

## Ported code

- Model-visible strings are byte-identical to Pi; PiDrei renames (app name,
  config dir, env vars) apply only to user-facing text.
- Session files keep Pi's JSONL format and the Pi identifiers in it; wire
  shapes (sessions, settings, RPC) stay camelCase with explicit field maps.
- Pi's mirrored tests are the spec: when one fails, check Pi's behaviour
  before changing the test.
- Regions listed in the `DIVERGED` table (`scripts/upstream_diff.py`) are
  deliberately reshaped; changes there follow the region's recipe.
- Porting an upstream delta: `spec/upstream-sync.md`.

## Tests

- `@pytest.mark.tonio` goes on test functions only: never on fixtures, never
  on nested helpers. Yield fixtures (`tmp_path`, `monkeypatch`) and async
  fixtures work. `tonio.run()` inside a sync test is never a workaround: it
  fails once any earlier test has built a runtime.
- **No sleep-polling.** Never `while ...: await sleep(x)` or a bare sleep to
  wait for something. Give the fake an Event it sets, and
  `await event.wait(timeout)`. Bound every wait so a failure fails fast.
  Never `sleep(0)`.
- **Parallelism is real in tests too:**
  - A promise Pi settles inside a sync listener delivers only after the
    listener returns; here the awaiter wakes immediately on another thread.
    Have the listener set an Event as its last statement and wait on it
    before asserting.
  - Two spawned callers: waiting for "the work started" proves only that the
    first one joined.
  - The render loop and other loops write concurrently. Select by content,
    not position (`writes[-1]`).
- Fake time instead of sleeping:
  - Swap `clock.now_ms` (wall time) or `clock.monotonic` (deadlines); conftest
    guards reset them.
  - The agent package has `FakeTimers`.
  - Modules importing `Timeout` at top level can have it rebound to a fake
    that the test fires by hand.
  - Manual clocks seeded from real time must advance *past* thresholds with
    margin: float rounding makes exact advances fall short.
- **No vacuous tests.** A test earns its place by failing under a meaningful
  mutation. Don't restate constants or test stdlib behaviour.
- Don't test dependency behaviour (TonIO, httpunk, punkreq); they're assumed
  correct.
- No repo tests for example extensions: verify them with lint and a
  throwaway run.
- Pi's `toMatchObject` is a subset check; mirror it as one (`_assert_matches`
  helpers), never `==`. Catalog-backed assertions otherwise break on the next
  regen.
- Tests are hermetic: the pidrei tests' conftest sets `PIDREI_OFFLINE=1`, and
  the root conftest's network guard refuses any non-loopback DNS lookup or
  connection and fails the test that tried. A code path that sends a real
  request (a missing client stub) needs a fake behind its seam, not an
  exemption. Process-global
  registries mutated in a test are restored in `finally`.
- Extension tools in tests return `AgentToolResult`, not dicts.
- macOS CI turns latent races into failures (line-sized pipe reads, long
  temp paths, slow pool I/O, reachable startup windows). Reproduce locally
  through the seams in `spec/debugging.md` rather than guessing.

## Changelog, versions, releases

- Versions are `<pi version>.<pidrei build>` (e.g. `0.99.1.0`). The four
  published packages (`ai`, `agent`, `tui`, `pidrei`) share the version and
  pin each other exactly; the transport packages are not bumped.
- `UPSTREAM_VERSION`/`UPSTREAM_REF` in `upstream.py` and `.last_upstream_ref`
  move together (`make upstream-bump`), by convention to the commit right
  after Pi's release tag.
- CHANGELOG:
  - Until `vX.Y.Z.W` is tagged, every entry goes in that version's section;
    `[Unreleased]` is only for work after a tag.
  - A `.0` section opens with `Tracks [Pi X.Y.Z](...)`.
  - Entries are user-facing prose under Added/Changed/Removed/Fixed (and
    "Not ported" where relevant).
  - A dependency bump is one line.
- User-facing text (CHANGELOG, `docs/`, README, anything in a wheel) never
  references repository-internal planning documents.
- The release gate is `make release-check`, which fails only on a missing
  tag before tagging.
