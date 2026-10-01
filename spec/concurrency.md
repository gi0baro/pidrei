# Concurrency design

How PiDrei is built to be correct on a multi-threaded runtime, and why it
differs from Pi where it does. `AGENTS.md` has the rules; this document has
the architecture behind them. The translation recipes for upstream diffs
that land in these areas are in `spec/upstream-sync.md` §7.

## The starting point

Pi is a JavaScript program: one thread, an event loop, and the guarantee that
nothing interleaves between two `await`s. A great deal of Pi's design leans
on that guarantee without saying so. Messages are live objects shared by
reference, services mutate shared state and rely on nobody reading
mid-update, listeners run inline, and promise chains provide ordering.

PiDrei runs on TonIO under free-threaded CPython: coroutines run in parallel
on a pool of worker threads, and there is no GIL to fall back on. A
convention like "only X looks at this" is load-bearing here in a way it never
is in JavaScript, and nothing enforces it. PiDrei is also a *tracking* port:
upstream commits keep arriving, and every place PiDrei differs from Pi costs
something on every future sync. Two principles follow.

1. **Domain logic stays 1:1; the runtime layer is PiDrei's.**
   - Provider payload building, agent-loop control flow, the session tree,
     compaction, component output and error texts port side by side, and
     Pi's mirrored tests are their spec.
   - Promise chains, `setTimeout`, `AbortSignal`, microtask ordering and
     `void promise()` are facts about JavaScript's runtime, not product
     behaviour. PiDrei owns them outright with a small set of native
     primitives (below), rather than emulating them.
2. **Divergence lives at seams.** Where PiDrei must differ from Pi's shape,
   the difference is placed at a boundary upstream rarely changes (the
   runtime layer, a publication point) rather than smeared across the
   high-churn files (adapters, agent loop, components). Each such region is
   listed in the `DIVERGED` table with a translation recipe, so an upstream
   diff landing there is still a mechanical translation.

### Ordering contracts

Parity means Pi's observable behaviour, not its schedule. The orderings Pi
got for free from one thread are part of that behaviour, so PiDrei encodes
each one with a primitive and tests it. Everything else runs in parallel.

- Tool bodies run in parallel: `tool_execution_end` is emitted in completion
  order, while tool-result messages are persisted and emitted in assistant
  source order.
- Agent listeners are awaited in registration order, one event at a time
  (see "The agent").
- Steering and follow-up queues are drained only at the agent loop's defined
  points.
- File mutations queue per path (`with_file_mutation_queue`), so two edits
  to one file never interleave and edits to different files don't wait on
  each other.
- Session appends have a single writer (`SessionManager`'s FIFO I/O lock),
  so entries reach the file in append order.

## Runtime primitives

The vocabulary the rest of the design is written in.

### Cancellation: tokens at the edges, scopes underneath

Pi threads an `AbortSignal` through everything and uses it as the
cancellation *mechanism*. In PiDrei, `CancelToken` is only the Pi-visible
*surface* (`cancel: CancelToken | None`, where Pi has a signal). The
mechanism is TonIO scope cancellation:

- Work that Pi would hand a signal runs as the child of a scope. The owner
  waits inside the scope for a condition that both completion and
  cancellation set; the token's `on_cancel` cancels the scope and sets the
  condition. Leaving the scope is what cancels the child at its current
  suspension point. Nothing is polled per chunk or per iteration.
- The two implementations of that shape are
  `EventStream.spawn_producer` (streams) and
  `pidrei_ai.utils.abort.run_cancellable` (operations). Use them rather than
  wiring a scope by hand.
- `race_with_cancel` is the other shape: the caller stops waiting but the
  operation keeps running detached. It is for state mutations (credential
  and model-store writes) that must not be torn by an abort.
- `.cancelled` checks remain only at Pi-observable decision points, where Pi
  checks `signal.aborted` and behaviour branches on it.
- A `None` token flows through `NEVER_CANCELLED` and costs nothing.
- The aborted result (a message carrying the partial content) is produced
  by the stream's *owner*, from what the producer published, never by the
  cancelled child.

### Ordering: locks, single consumers, tails

Pi gets "first come, first served" from promise chains. PiDrei uses:

- **A TonIO `sync.Lock`** where callers must run one at a time in call order
  and the critical section suspends. Release in `finally` means a failed
  predecessor never blocks its successors, and with `async with` a caller
  cancelled while it waits leaves the queue (the nested tool-call exclusive
  queue).
- **One long-lived consumer over a channel** where work has a natural queue
  and a flush point (stdio output, the agent mailbox, the TUI's
  terminal-event loop).
- **An Event-tail chain** where taking a place in line and waiting for the
  turn are separate steps, which a lock cannot express: a newcomer reads the
  current tail and installs its own under a short thread lock, waits for the
  old tail later, and sets its own when done. FIFO by construction (the two
  `file_mutation_queue.py`: the place is taken at call time, or under a
  registration lock). Unlike Pi's promise wait, a coroutine waiting for the
  old tail can be cancelled, and a tail that is never set strands the queue,
  while one set at cancel time lets the successor overlap the predecessor.
  So the wait must not die with the caller: from the moment the place is
  taken, with no suspension in between, the work runs detached
  (`spawn.without_tracking`) and stores its outcome, and the caller only
  waits for that outcome. The work runs in its turn whatever happens to the
  caller, as in Pi.
- **Thread locks** (`threading.Lock`/`RLock`) for short synchronous critical
  sections only, never held across an `await`.

### Hand-off and publication

- `tonio.Result` carries a value between coroutines (instead of one-element
  list boxes); `tonio.Event` signals; `Waiter.any(...)` composes waits.
- State that readers on other threads must see consistently is published
  as **one immutable value, rebound atomically** (the epoch pattern, below).
  Readers pin one read; writers copy, change privately, and rebind under the
  writer's lock.

### Timers

`pidrei_tui._timers.Timeout`/`Interval` are plain primitives: a coroutine
parked on its cancel Event with the delay as timeout. The callback is
synchronous and runs on the timer's own coroutine; it guards what it touches
itself. `cancel()` is atomic with the fire's check. Unlike JavaScript's
`clearTimeout`, though, a fire that checked just before the cancel still
runs, so a callback whose late run matters re-checks its own state.

Production code reads time only through `pidrei_ai.utils.clock` and
`pidrei_tui.clock`, which are also the tests' seams.

### Blocking work

Filesystem and other blocking I/O run on TonIO's blocking pool
(`tonio.colored.fs` or `spawn_blocking`), never on a runtime worker. This is
a safety rule, not a performance preference: a blocked worker is unavailable
to everything else in flight, and nobody can enumerate what that is.
Granularity is tuning: one coarse job shipped to the pool beats many fine
hops (listing a directory, loading a session, decoding JSONL in chunks).

### Stdio

Every write to stdout and stderr goes through `pidrei.core.output_guard`:
the writes are sent down one channel to a single writer coroutine, so the
two streams keep one order and a slow reader parks the writer, not the
caller. While the TUI owns the terminal, writes that share its output go
through the terminal's own queue, so the tty has exactly one writer.

## The data plane: messages are values

Pi's streaming partial is the provider's live object: adapters mutate it per
SSE event (`block.text += delta`) while consumers hold references to it. In
PiDrei that would be a data race on every token.

PiDrei freezes at the publication seam:

- Message, content-block and usage types (`pidrei_ai/types.py`) are frozen
  dataclasses.
- Producers build into mutable **builders** (`pidrei_ai/builders.py`) that
  mirror the frozen types field for field. Adapters keep Pi's mutation lines
  verbatim; only construction sites name the builder type.
- `AssistantMessageEventStream.push()` freezes each event's message payload
  as it is published. The cadence is **per delta**: every event carries an
  independent snapshot that any consumer may keep indefinitely. A full
  freeze of a typical streaming message measured around 3 µs, so the
  per-token cost is negligible.
- Consumers that change a message construct a new one
  (`dataclasses.replace`).

Two deliberately shallow edges:
- `content`/`diagnostics` stay lists, rebuilt per snapshot, so they compare
  equal to the lists Pi's mirrored tests construct.
- User-owned dicts (`arguments`, `details`, `data`, `metadata`) are shared by
  reference. Producers rebind them, never mutate them in place.

Extension handlers that hold a message can no longer mutate it in place:
they get a loud `FrozenInstanceError` instead of a silent race. Returning a
replacement message works as in Pi.

## The agent

### Parallel work, serialized observation

Pi's `await emit(event)` means every listener has run, in order, and no two
listeners overlap. With parallel tool execution, emits arrive concurrently
from several coroutines, so PiDrei serializes *observation* rather than
work:

- Tool bodies run genuinely in parallel; results persist in source order.
- Each run has one **dispatcher** coroutine (`Agent._dispatch_events`). It
  is fed through a channel of tickets (`_PendingEvent`), reduces
  `AgentState`, and awaits the listeners in order. The emitter awaits its
  ticket, so "`emit` returned" still means "listeners settled", and a
  listener's error re-raises at the emitter.
- `Agent.observe(event)` puts an observe-only ticket (no reducer) on the same
  dispatcher. Nested tool calls use it so that their events join the one
  serialized stream. Once the dispatcher has closed, an observe is dropped.
- The pipeline stays **fused** on purpose: Pi's code relies on it as a
  programming model. The loop reads state a listener just set, `message_end`
  handlers replace a message before it enters state, `before_tool_call`
  gates execution. The fusion is also real backpressure. Decoupling the
  observers would turn every hook into an interceptor-or-observer
  classification whose mistakes are silent ordering bugs. The stall meter
  (`PIDREI_DISPATCH_STALL_LOG` set to a file path) measures per-event
  observation latency by event type, logs each event over 50 ms, and appends
  a per-type summary when the run ends; with the variable unset it costs
  nothing. Only a trace showing the dispatcher stalled behind a
  slow listener in normal use would reopen the question.

### The mailbox

Pi keeps the agent's queues and run lifecycle as loose fields plus
"already processing" guards. PiDrei folds them into `_AgentMailbox`, an
actor: a standing consumer coroutine per Agent owns the steering and
follow-up queues and run admission, and every operation is a synchronous job
on its channel, run in FIFO order.

- Admission is a job, so concurrent `prompt()` calls admit exactly one run,
  in channel order. The others raise Pi's "Agent is already processing."
- Queue mutators (`steer`, `follow_up`, the clears) keep Pi's synchronous
  signatures as fire-and-forget posts. That is enough: every observer of the
  queues also goes through the channel, so FIFO puts it behind the mutation.
  Only `has_queued_messages` is awaited.
- The run record (`mailbox.current`) is published for lock-free readers
  (`abort`, `signal`, `wait_for_idle`); aborting fires the record's cancel
  token directly.
- The consumer exits when the Agent is garbage-collected: dropping the last
  sender closes the channel.

### State epochs

`AgentState.messages` is a tuple that is only ever rebound: the setter stores
a tuple, the reducer publishes `(*old, message)`, and mid-run writers rebind.
Readers pin one read. The session's identity side tables (Pi's
`WeakMap`/`WeakSet` keyed by message) are run-scoped dicts keyed by `id()`.
They hold strong references (so an id can't be reused mid-run) and are
cleared when the prompt-loop iteration ends; a miss falls back to Pi's
positional lookup. No weakrefs, no finalizers.

The session's tool loadout (the declared tools and the hidden declarations)
is likewise one frozen value (`_ToolLoadoutEpoch`), published under a
synchronous guard and pinned by each request.

### Nested tool calls

A tool calling other tools (`ctx.execute_tool`) records them on its own
result. Pi does this with a per-session recorder that nested calls mutate.
PiDrei uses one channel per top-level call: nested calls send started and
finished messages, and the owning tool wrapper folds them into a bounded
snapshot when the tool returns. The snapshot rides on the result into the
frozen tool-result message. Details are in the `nested-calls-channel`
recipe.

## Configuration: epochs instead of reload chains

Pi's config services (settings, auth storage, model stores, the model
registry, the model runtime) are shared mutable objects kept consistent by
write queues and re-checks. PiDrei's publish **immutable snapshots swapped
atomically**:

- Settings setters run Pi's mutation lines on a private deep copy and
  publish by rebinding under the writer lock. Compound getters pin one read,
  so a single call can't mix two versions.
- Auth and model-store read caches are one frozen `(data, revision)` value,
  pinned before the revision comparison. The cross-process revision checks
  stay; they are about freshness, not threads.
- In-memory stores and the provider map are swapped on write; their readers
  take no lock.
- The model runtime publishes its composition inputs (config, extension and
  native providers, virtual models, errors) as one `_CompositionEpoch`. A
  request pins it once, so provider, auth and headers always come from the
  same composition. Its availability view (all and available models,
  configured and stored providers, auth checks) is likewise one `_Snapshot`
  value, replaced whole on each refresh rather than updated in place; it was
  the first epoch in the codebase and the model for the others.

Consumers that read a service twice across an await (compaction and retry
in the session, header transforms in the SDK) are Pi's own shape: they can
tear in Pi too, so they stay 1:1 rather than being "fixed" into a pinned
read.

## The UI

The TUI and interactive mode follow their own design: passive UI state
guarded by one reentrant lock, with input, render, timers and terminal events
as independent loops. It is described in `spec/ui-island.md`.

## Decided, and not to be re-derived

- **Removing the `CancelToken` surface** would cost little and gain little.
  The token is no longer the mechanism; what remains is a legitimate
  "explicit token at the edge, structured concurrency underneath" shape. A
  package may drop it opportunistically.
- **Sync code on threads** ("run tool bodies as plain sync code, drop the
  async colouring") was considered and rejected. Runtime workers are a
  cooperative pool; blocking I/O on one removes it from async progress for
  everything else. The two-pool split is the same rule tokio has.
- **Elm-style UI** (immutable model, pure render) would be the right design
  for a standalone TUI library. As a retrofit on a tracking port it is
  prohibitive, because every upstream component diff would become a
  redesign. It belongs to a TUI that no longer tracks Pi.
- **Per-layer tracking value.** If PiDrei ever stops tracking Pi for some
  layer, the order is: the TUI first (mostly implementation churn), the
  agent and session layer next, the provider adapters last or never (they
  encode protocol facts that would otherwise be rediscovered against live
  APIs).
