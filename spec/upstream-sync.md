# Upstream sync

How PiDrei ports a range of upstream Pi commits, and the translation rules for
the regions where PiDrei deliberately differs from Pi. Read it in full before
starting a sync. The engineering rules in `AGENTS.md` apply throughout.

## What a sync is

Pi releases; PiDrei ports the delta between the last ported commit and the
new release. Upstream commits are the porting unit, worked **in upstream
order**, and **Pi's tests are the spec**. Each portable commit lands as one
faithful PiDrei change: production code and test mirrors together.

A sync has three phases:

1. **Sweep.** Classify the range, size it, find the design-level work, and
   list the decisions that need a ruling. Group the commits into units.
   Report the plan and wait for the decisions before porting anything.
2. **Port.** Execute the units in order, reporting after each one.
3. **Close-out.** Regenerate the catalog, bump the versions and the
   upstream ref, write the CHANGELOG, and run the verification gate.

Everything stays uncommitted for review throughout.

## Prerequisites

- A Pi checkout (`PI_ROOT`, default `~/Downloads/code/pi`; override with
  `make upstream-diff PI_ROOT=...`), checked out at the target commit.
- **Ref convention.** The ported ref is the commit *one past* the release
  tag (Pi's "Add [Unreleased] section for next cycle"), so `git describe`
  reads `vX.Y.Z-1-g<sha>`.
- The current ported ref is in `.last_upstream_ref`, mirrored in
  `packages/pidrei/pidrei/upstream.py`.

## 1. Sweep

### Run the classifier until its report is clean

```
make upstream-diff        # exit 2 => UNMAPPED paths need table entries
```

`scripts/upstream_diff.py` lists every commit in the range and classifies each
changed file:

- **portable**: a mechanical mapping (kebab-case → snake_case, `.ts` →
  `.py`, Pi's nested test directories flatten into `tests/`; mirrored example
  extensions map too, with non-code assets keeping their names). The tables
  below override it. A portable file listed in `DIVERGED` also prints
  `⚠ diverged [recipe-id]`: hunks inside that region follow the recipe in
  §7, and hunks elsewhere in the file port normally.
- **dropped**: documented non-ports (§6), each with its reason.
- **noise**: lockfiles, changelogs, CI, test-runner configs, READMEs, Pi's
  internal design docs. A `package.json` of a ported package is noise plus a
  summary line; check those commits for new runtime dependencies (usually
  JS-only).
- **merges** are skipped: every commit a merge brings in is listed
  individually.

The tables (all in `scripts/upstream_diff.py`):

| Table | Holds |
|---|---|
| `PREFIX_MAP` | upstream directory → PiDrei directory |
| `RENAMES` | hand-verified name divergences, one per entry |
| `TEST_HOMES` | Pi test files whose PiDrei coverage is consolidated, partial, or a recorded **PARITY GAP** (ported production code with no mirror; backfill when a commit touches it) |
| `DROPPED_PREFIXES`, `DROPPED_BASENAME_RE`, `LIVE_API_AI_TESTS` | non-ported paths, with a reason |
| `NOISE_PREFIXES`, `NOISE_BASENAMES` | paths with nothing to port |
| `DIVERGED` | upstream files containing a diverged region, with the recipe id |

Triage every `[MISSING — rename?]` marker before porting anything:

1. Grep PiDrei for the *snake_cased test names* (`it("does the thing")` →
   `test_does_the_thing`), not for phrases or file names: many Pi test files
   were consolidated into module-organized ones.
2. Consolidated → a `TEST_HOMES` "covered by ..." entry. Genuinely absent
   with ported production code → a `TEST_HOMES` PARITY GAP entry, or a
   backfill if the range touches it. Deliberately unported →
   `DROPPED_PREFIXES`. True rename → `RENAMES`.
3. `[MISSING]` on a file *added earlier in the same range* heals itself as
   you port in order; no entry needed.
4. `[not in curated docs subset]`: PiDrei ships a curated subset of Pi's
   docs. Leave it unless the doc became relevant.

Re-run until the classifier exits 0.

### Size the work and find the decisions

Read the range; for large ranges read every commit touching ported code.
Separate:

- **mechanical** commits, which port side-by-side;
- **design-level** series, where the Pi change depends on single-thread
  assumptions (shared mutable state, synchronous listeners, ordering by
  event-loop turn) or adds a new subsystem;
- **new upstream packages or subsystems**, which need a feasibility look
  before anything else;
- **candidates to drop or defer**, each with a reason.

Every judgement call becomes a numbered decision for the maintainer to rule
on. Design-level series get a proposed translation, written as a recipe
(§7) once ruled. Then group the commits into units: related commits
together, upstream order preserved within and across units, a design-level
series in a unit of its own.

Report the plan before executing: the unit list, the design items, and
everything you intend to drop or defer, so the maintainer can veto it. The
sweep's plan and rulings are working notes kept outside the repository.
Rules that outlive the sync land in this document.

## 2. Port

For each unit, commit by commit:

1. `git -C $PI_ROOT show <sha>`: read the whole diff, tests included. The
   tests tell you what the production change actually means.
2. Land the production change in the mapped PiDrei files. Mirror structure
   and comments, and keep model-visible strings byte-identical. Hunks in a
   diverged region follow its recipe.
3. Land the test changes in their homes. New Pi test files get same-named
   mirrors, which also heals the classifier. Cases touching an unmirrored
   PARITY GAP file: port the *delta* somewhere sensible if it specifies new
   behaviour, otherwise leave the gap entry.
4. Run the touched test files (`timeout 60 uv run pytest -q <files>`).
5. Move on. The full suite runs at close-out.

After each unit: lint, report what landed and every judgement call, and wait
for review.

Judgement calls with precedent:

- **Observable JavaScript semantics are mirrored deliberately**, not
  translated into the Python idiom:
  - `||` (any falsy value) and `??` (only `null`/`undefined`) stay distinct:
    `or` chains for the first, explicit `is None` checks for the second.
  - Truthiness quirks are preserved where behaviour depends on them (an empty
    string or `0` is falsy, an empty object is truthy).
  - `undefined` (key absent) and `null` (key present, `None`) stay distinct
    wherever Pi distinguishes them. Pi's JSON output drops `undefined` and
    keeps `null`, and settings and wire dicts track key presence.

  Leave a comment where the mirrored form would look like a mistake to a
  Python reader.
- **Port the final state of a series.** When commits in the range rewrite
  the same strings or code, land the end state, not each step.
- A commit and its revert in the same range cancel; skip both.
- Commits marked "WIP" or "PoC" upstream that shipped in the release **are
  ported**: released tests are the spec.
- Fixture-only upstream changes (offline by default, test isolation) often
  describe hygiene PiDrei already has. Verify, adapt comments, move on.
- JS-only surface with no consumer in the product (a per-request `fetch`
  injection, bundler or runtime packaging) may be recorded as a documented
  deviation instead of ported. Say so in the report; the maintainer can veto
  it.
- **npm**: PiDrei installs packages from git and local paths only.
  npm-flavoured changes map onto the git path or are dropped with a note.
- **Windows** branches in a diff are dropped silently.
- A diff outside a dropped path that only makes sense with dropped surface
  (e.g. consumers of a non-ported subsystem): port the product half, drop the
  rest with a one-line note.
- **Catalog commits** (`generate-models.ts`): port the generator change.
  The catalog is regenerated once at close-out, so tests asserting catalog
  contents fail until then; note them, don't chase them.
- **Cross-commit dependencies.** A commit can build on another one placed in
  a later unit, or on one silently skipped earlier. Before porting a commit,
  check where the symbols its context lines use came from
  (`git -C $PI_ROOT log -S<symbol>`), and diff an earlier commit's hunks
  against PiDrei when a later commit builds on it.
- When a unit exposes something the rulings don't cover, stop and ask.

## 3. Writing test mirrors

The `AGENTS.md` test rules apply. Sharp edges specific to mirroring Pi:

- **`vi.mock` becomes a module or attribute swap.** Rebind the class the
  code under test imports (e.g. an adapter's client class) inside a context
  manager that restores it, or use `monkeypatch`. Copy the nearest sibling:
  `test_bedrock_endpoint_resolution.py`, `test_anthropic_sse_parsing.py`,
  the `test_openai_completions.py` FakeClient, `test_google_stream.py`,
  `oauth_helpers.py`.
- **JS microtask timing does not transfer.** A Pi test that reads a shared
  object mid-iteration (e.g. `partial.stopReason` off a start event) races
  here. Either sample at push time (a recording-stream subclass) or gate the
  producer on an Event the consumer sets; `test_openai_responses.py` has
  both.
- **Pi's stubs are invoked synchronously in call order.** Spawning twice and
  then indexing `invocations[0]` is a coin flip. Spawn, wait for the first
  invocation, then spawn the next.
- **`vi.useFakeTimers()`** maps onto the clock seams and `FakeTimers`
  (`AGENTS.md`), never real sleeps.
- **`toMatchObject` is a subset check.** Mirror it with an `_assert_matches`
  helper (see `test_google_vertex_api_key_resolution.py`). An `==` mirror
  breaks on the next catalog regen.
- **`register_native_provider` spawns a detached full refresh.** A test that
  then races a provider-scoped refresh must first drain it with an unscoped
  `await runtime.refresh(ModelsRefreshOptions(allow_network=False))`; only an
  unscoped refresh clears the request flag. Bound the waits so a recurrence
  fails fast.
- Pi tests that only exercise JS-specific behaviour (prototype keys, getter
  reassignment, `undefined` vs missing where PiDrei has no such distinction)
  are not mirrored; note them in the mirror's docstring.
- A mirror needing a relaxation (a builder instead of a mutated message, an
  `await` on a call that is async here) carries a one-line note naming the
  recipe or reason.

## 4. Close-out

In order:

1. **Catalog regen**: `make models-data`, then rerun `packages/ai/tests`.
   The regen-dependent failures noted during the port must clear. One run
   covers chat, image and classifier models.
2. **Versions**: the four published packages (`packages/{ai,agent,tui,pidrei}/pyproject.toml`)
   move to `<pi version>.0`: their `version` *and* their exact cross-pins.
   `protocol`/`client`/`server` are not bumped.
3. **Upstream ref**: set `UPSTREAM_VERSION` in
   `packages/pidrei/pidrei/upstream.py` to the Pi version, then
   `make upstream-bump REF=<one-past-tag sha>`. The bump is ancestry-checked
   and rewrites `.last_upstream_ref` and `UPSTREAM_REF`.
4. **CHANGELOG** (`packages/pidrei/pidrei/CHANGELOG.md`): a new
   `## [X.Y.Z.0] - <date>` section opening with
   `Tracks [Pi X.Y.Z](https://github.com/earendil-works/pi/releases/tag/vX.Y.Z).`
   Then user-facing prose grouped Added / Changed / Removed / Fixed, and
   **Not ported** for upstream features left out. Write it from Pi's
   changelog for the range, checking each entry against what actually
   landed. Dependency bumps are one line. Check that
   `uv run python scripts/release_notes.py X.Y.Z.0` extracts it.
5. **Diverged regions**: every recipe ruled during the sync is written into
   §7, and its files get `DIVERGED` entries.
6. **Verification gate**, all of it:
   - `make upstream-diff`: exit 0, "0 commits to port".
   - `make lint`. New mirrored ladders may need
     `# noqa: C901 (mirrors pi's ...)`, matching existing style.
   - `make audit`: 0 findings.
   - The full suite, in slices (`AGENTS.md`), under `timeout 60`.
   - **Never block the runtime**: eyeball every new production I/O site,
     then run the blocking-fs detector per slice
     (`PIDREI_FS_DETECT=1 timeout 60 uv run pytest -q <slice>`) and compare
     the shipped-code sites against the previous release. The port must add
     no runtime-worker fs sites driven by production code. To compare
     reliably, run the detector on the previous release in a scratch
     `git worktree`; sync that worktree's venv with the main repo's
     free-threaded interpreter (`uv sync --python $(readlink -f .venv/bin/python)`),
     or every test fails with "GIL detected". Reading the report:
     - The detector attributes an import-time site to whichever test
       imports the module first, so counts shift with test order.
     - A site reached through a spawned coroutine can show as
       "driven by production code" even when a test drives it.
     - Python's own reads (imports, traceback/linecache) are not PiDrei's
       I/O.
   - `make release-check` fails **only** on the missing `vX.Y.Z.0` tag.
7. **Report and stop.** Lead with the verification state, list what landed,
   and surface every judgement call: deviations, drops, deferrals, parity
   gaps. The maintainer reviews, commits, watches CI on both platforms,
   tags, and publishes.

### When the suite hangs

No hang is expected; treat one as a fresh bug. Rerun once (exit 124 means
the timeout fired). If it recurs, capture before anything else: the
`-v` ordering to name the test, `py-spy dump`, per-thread CPU from
`/proc/<pid>/task/*/stat`. Then deselect the victim to unblock the port and
report the capture. Don't chase it mid-port.

## 5. Dependency bumps

TonIO, httpunk and punkreq releases are ported as their own small change,
usually a `.N` PiDrei release:

1. Bump the pins in every `pyproject.toml` that names the dependency
   (TonIO: all six packages), then `uv sync --all-packages`.
2. For httpunk/punkreq, diff the two tags for the names PiDrei uses. PiDrei
   imports httpunk in exactly one place, `pidrei_ai/utils/http.py`
   (`Backend`, `H1Connection`, `H1Server`), and sees its exceptions only
   through punkreq's mapper (`map_httpunk_exception`) and the messages
   `pidrei_ai/utils/retry.py` matches. The flow tests stub the network; the
   real coverage is `packages/ai/tests/test_oauth_callback_server.py` and
   the loopback cases in `test_websocket_connect.py`. Run those first. When
   in doubt, run them on the old pins too (`uv pip install httpunk==X
   punkreq==Y`, `uv run --no-sync pytest ...`, then
   `uv sync --all-packages`).
3. For TonIO, recheck the TonIO contract section of `AGENTS.md` against the
   release.
4. The full suite in slices, `make lint`, `make audit`, the blocking-fs
   detector (same sites as before), `make release-check`.
5. A one-line CHANGELOG entry in the current untagged section, or a new
   `.N` section.

## 6. What is not ported

Each has entries in the classifier's dropped tables, with the reason.

- **Windows**: TonIO is POSIX-only.
- **Pi's experimental stack**: the durable harness runtime, Chord, the
  experimental coding agent (`coding-agent/src/experimental`,
  `cli/experimental`, `mini/`, facet plugins), the protocol/server/client
  rewrite, session backends, benchmarks. It is an unfinished rewrite the
  product does not run on; porting it would track churn on a moving target.
  Not porting it is a deferral, not a saving.
  - Still ported: the transport layer (CBOR codec, framing, unix socket
    listener and transports, unix client connect) in the unpublished
    `protocol`/`client`/`server` packages. Pi 1.0.0 removed the harness
    helper layer from `pi-agent-core`; the parts the product used live in
    `pidrei` (`core/messages.py`, `core/message_wire.py`,
    `core/tools/edit_diff.py`, `core/tools/truncate.py`, `utils/mime.py`).
  - **Reopen** when Pi's product (`AgentSession` or the default interactive
    mode) starts running on the harness, when upstream declares session
    format 4 stable, or when `pi server`/`pi client` leave the
    `PI_EXPERIMENTAL` gate. The port is then one end-state unit (never
    commit-by-commit), preceded by a design pass for its single-thread
    assumptions, with the transport survivors as the first layer.
- **RPC mode as a mode**: it is not run or audited, but its code is kept and
  kept correct; contract changes propagate to it.
- **Self-update** machinery, installer-managed updates, and Node/Bun
  packaging.
- **Telemetry** of any kind (the `no-telemetry` recipe).
- **Radius**: provider, presence and session sharing.
- **The llama.cpp extension**, and its consumers (llama.cpp classify).
- **Native helpers** (`tui/native`, darwin clipboard file paths): code that
  calls them gets a `None` answer.
- Live-API tests, Pi's eval harness, storage backends, manual probe scripts,
  Pi's TypeScript SDK examples and npm/wasm example extensions.

Deferred (ported later as one unit): codemode, tool search and MCP (client,
extension, CLI). The core they build on (tool exposure, nested tool calls,
the MCP server registry) is ported.

## 7. Diverged regions

Regions where PiDrei deliberately differs from Pi's shape, so that it is
sound on a multi-threaded runtime; `spec/concurrency.md` explains the design
behind them. The `DIVERGED` table flags the upstream
files that contain them. When a flagged hunk lands *inside* the region,
apply the recipe instead of porting side by side, and extend the recipe here
if the diff exposed a case it doesn't cover. Hunks elsewhere in the same
file port normally. Pattern-shaped recipes (`cancel-token`) have no
`DIVERGED` entry and apply wherever the pattern appears.

### `cancel-token` (pattern-shaped)

Pi threads `AbortSignal` everywhere and uses it as the cancellation
*mechanism*. PiDrei keeps `cancel: CancelToken | None` only as the
Pi-visible *surface*; scopes are the mechanism.

- Upstream threads a signal into new work → own that work in a scope and
  wire the token at the edge (`cancel.on_cancel(...)` cancels the scope and
  settles the owner's wait condition). Reach for the existing
  implementations first: `EventStream.spawn_producer` and
  `pidrei_ai.utils.abort.run_cancellable`.
- Keep `.cancelled` / `raise_if_cancelled()` checks **only** at Pi-observable
  decision points, where Pi itself checks `signal.aborted` and behaviour
  branches on it. Never add polling checks inside scope-owned work: a
  cancelled `await` already raises.
- `None` tokens flow through `NEVER_CANCELLED` and cost nothing; don't
  special-case them.
- Mirrored signatures keep the token parameter where Pi has the signal.

### `dispatch-observe` (`agent/src/agent.ts`)

Pi emits agent events by iterating listeners inline on its single thread.
PiDrei serializes observation on a per-run dispatcher coroutine
(`Agent._dispatch_events`, fed by `_process_events` through `_PendingEvent`
tickets in `pidrei_agent/agent.py`), which also carries the debug-gated stall
meter (`PIDREI_DISPATCH_STALL_LOG`).

- Changes to *what* is emitted, event payloads, or listener order port 1:1:
  the reducer and the listener loop mirror Pi's semantics ("listeners
  settled" when `await emit(...)` returns).
- Changes to *how* Pi delivers events (inline iteration, promise handling,
  listener error propagation) land in the dispatcher machinery: listener
  errors travel back through `_PendingEvent` and re-raise at the emitter.
- `Agent.observe(event)` (PiDrei-only) puts an observe-only ticket on the
  same dispatcher: no reducer, same listener stream (used by nested tool
  events, see `nested-calls-channel`).
- Keep the stall meter's hooks around the observation loop.

### `agent-mailbox` (`agent/src/agent.ts`)

Pi keeps the agent's queue and lifecycle state as loose fields (two
`PendingMessageQueue`s, an `activeRun` record, "already processing" guards).
PiDrei folds them into `_AgentMailbox` (`pidrei_agent/agent.py`): a standing
consumer coroutine per Agent that owns the queues and run admission. Every
operation is a synchronous closure job on the mailbox channel, run in FIFO
order. The consumer starts at construction and exits when the Agent is
garbage-collected (dropping the last sender closes the channel).

- **One signature divergence**: Pi's sync `hasQueuedMessages()` is awaited
  in PiDrei. An upstream call site or test gains an `await` plus a one-line
  note. Every queue *mutator* keeps Pi's sync signature: `steer`,
  `followUp` and the clears are fire-and-forget posts. This is enough
  because every observer also goes through the channel, so FIFO puts any
  later drain or query behind the mutation. Don't upgrade a mutator to an
  awaited call.
- **Queue diffs** (`PendingMessageQueue`, drain modes, clear/has-items) land
  in `_PendingMessages`. It is confined to the mailbox consumer, so Pi's list
  logic ports verbatim with no lock. `mode` is the one published field.
- **Lifecycle diffs** (`activeRun`, `waitForIdle`, `abort`, `signal`,
  admission errors):
  - Admission is a mailbox job: `claim()` raises Pi's "Agent is already
    processing." verbatim, so concurrent `prompt()` calls admit exactly one
    run, in channel order.
  - The admit job spawns `_run_lifecycle` detached, and the lifecycle ends
    by posting `_finish_run`: release the slot, *then* set `done`. Keep that
    order; it is what makes an admission sent after `done` win by FIFO.
  - `mailbox.current` is the published run record. `signal`, `abort`,
    `wait_for_idle` and the entry points' guards read it with no lock;
    `abort` fires the record's cancel token directly.
  - A cancelled `prompt()` awaiter fires the token and unwinds without
    waiting for the run; idleness is `wait_for_idle()`'s job.
- **New entry points that start a run**: an early caller-specific check
  against `mailbox.current` (Pi's error message, best-effort), then the
  admit job as the authoritative gate.
- Mailbox jobs never await (a job waiting on the mailbox would deadlock
  itself), and posted jobs must not raise.

### `state-epochs` (`agent/src/agent.ts`, `coding-agent/src/core/agent-session.ts`)

Pi's `agent.state.messages` is a live array that callers push onto and
assign. PiDrei publishes it as a **tuple that is only ever rebound**.

- **Writes**: `state.messages = xs` ports as-is (the setter stores
  `tuple(xs)`); `state.messages.push(x)` becomes
  `state.messages = [*state.messages, x]`; `splice`/`slice`-and-assign
  become a rebind. The agent's `_reduce` publishes `(*old, message)`.
- **Reads**: pin one read when a block reads the list more than once, and
  never assume a value read before an await is still current: mid-run
  writers (`prepare_request`, `finish_turn` via
  `_refresh_finalized_context`) rebind it.
- **Identity maps** (`WeakMap`/`WeakSet` keyed by message object) land in the
  session's run-scoped `id()`-keyed tables (`_entry_ids_by_message`,
  `_message_replacements`), which hold strong references, are written where
  upstream writes, and are cleared at the end of each prompt-loop iteration.
  Lookups that miss fall back to upstream's positional walk. Never use
  weakrefs or finalizers for these.
- **The tool loadout** (`_applyToolLoadout`) publishes
  `(tools, hidden_declarations)` as one frozen `_ToolLoadoutEpoch` under
  `_tool_loadout_guard` (an RLock held in sync code only, also around the
  tool registry rebind); readers pin one epoch.
- Mirrored tests that push onto `agent.state.messages` get the one-line
  rebind translation plus a comment naming this recipe.

### `freeze-at-seam` (`ai/src/types.ts`, the streaming adapters, `ai/src/utils/event-stream.ts`, consumer code)

Pi's messages are live mutable objects shared by reference. PiDrei's
message, content-block and usage types (`pidrei_ai/types.py`) are **frozen
dataclasses**. Producers build through the mutable mirrors in
`pidrei_ai/builders.py` (same field names, plus `freeze()`), and
`AssistantMessageEventStream.push()` freezes every event's message payload
at publication, so each pushed event carries an independent snapshot.

- **`types.ts` field changes**: a new or renamed message, content-block or
  usage field lands in the frozen type *and* its builder
  (`AssistantMessageBuilder`, `TextContentBuilder`,
  `ThinkingContentBuilder`, `ToolCallBuilder`, `UsageBuilder`,
  `UsageCostBuilder`), including the builder's `freeze()`, and in the session
  serde. Non-message types (options, compat, events, `Model`) stay plain.
- **Adapter diffs**: mutation lines (`block.text +=`, `output.usage.x = …`,
  `content.append`, `calculate_cost(model, output.usage)`) port verbatim,
  because the builders duck-type the field names. Only *construction* sites
  differ: where Pi constructs the streamed partial or its blocks, PiDrei
  constructs the `*Builder`. Complete values that are never mutated after
  construction (request-side conversion, final reconstruction) use the
  frozen types directly.
- **`event-stream.ts` diffs**: PiDrei's `push()` override is the seam.
  Delivery mechanics land under it (in `EventStream.push`); anything
  touching published payloads goes through the freeze. `stream.partial` is
  the producer-private builder.
- **Consumer diffs** (any package): code that *mutates* a message after
  publication (session edits, compaction, extension handlers, UI
  decoration) becomes constructing a new value (`dataclasses.replace`).
  Where Pi relies on the mutation being visible through the shared
  reference, decide explicitly what PiDrei does and document it (precedent:
  interactive mode's `message_end` abort decoration is a display-only copy;
  the persisted message keeps the provider's error text).
- **Tests** that drive producer internals construct builders (precedents in
  `test_openai_responses.py`, `test_registry.py`). Tests that shape a
  scenario by mutating a constructed message switch to `replace(...)` with a
  one-line note (precedent: `test_openai_completions_reasoning_details.py`).
- **Never** reintroduce a mutable message type or `getattr` probing for
  builders in consumer code.

### `config-epochs` (`settings-manager.ts`, `auth-storage.ts`, both `models-store.ts`, `runtime-credentials.ts`, `model-runtime.ts`, `ai/src/models.ts`)

Pi's config services are shared mutable objects, safe on one thread.
PiDrei's publish **immutable snapshots swapped atomically**: readers pin one
attribute read and take no lock; writers copy, mutate privately, and rebind
under the service's writer lock. Public APIs are unchanged.

- **`settings-manager.ts` setters**: Pi's setter body
  (`this.globalSettings.x = v; markModified; save`) becomes
  `self._set_global("x", v)` for a single key,
  `self._set_global_nested(field, key, v)` for one nested key, or an
  `update(settings)` closure passed to `_update_global_settings` for
  multi-key or conditional logic. The closure runs Pi's mutation lines
  verbatim on a private deep copy (precedent: `set_model_thinking_level`).
  Never mutate the published settings dicts outside those helpers. Getters
  port 1:1; a getter returning several values from one group pins one
  settings read (precedent: `get_retry_settings`).
- **Read caches** (`auth-storage.ts`, coding-agent `models-store.ts`): the
  `(data, revision)` pair is one frozen `_AuthFileSnapshot` /
  `_ModelsFileSnapshot`, rebound whole; readers pin it before comparing
  revisions.
- **In-memory stores**: `_entries` is swapped on write, never mutated in
  place.
- **`ai/src/models.ts`**: the provider map is swapped on write and read
  lock-free. `_apply_auth` takes the caller-pinned provider;
  `get_auth_for_provider` is the PiDrei-only pinned variant that `get_auth`
  delegates to, so an upstream diff to `getAuth` lands in its body.
- **`model-runtime.ts`**: the composition inputs (config, extension
  providers, native providers, virtual models, composition errors) publish
  as one `_CompositionEpoch` via `_publish_composition()` under
  `_composition_guard`. Reader diffs land against a pinned
  `composition = self._composition`; request auth resolves provider, auth
  and headers from one pin. A new mutable composition input becomes an
  epoch field (inner dicts replaced, never mutated).
- Multi-reads across awaits in consumers (agent-session compaction/retry
  reads, SDK header-transform timing) are **Pi's own shape** and port 1:1.
  Don't "fix" them into pinned reads; that would change observable
  mid-operation settings semantics.

### `nested-calls-channel` (`core/nested-tool-calls.ts`, `core/agent-session.ts`, `tools/tool-definition-wrapper.ts`, `agent/src/agent-loop.ts`, `agent/src/agent.ts`)

Pi records a tool's nested calls (`ctx.executeTool`) in a per-session runner
holding a `scopes` map, a mutable recorder per parent that nested calls write
into, and a `takeRecord()` that the session's `message_start` listener uses
to mutate `nestedCalls`/`usage` onto the tool-result message. On TonIO the
nested calls run in parallel, the parent can finish while some are still
writing, and the tool-result message is frozen at publication. PiDrei ports
the *promise*: the message carries a bounded snapshot of the nested calls
as of when the parent finished, `unfinished` for what was still running, and
nested usage summed into the parent's usage. It does this with a
producer/consumer channel and no shared mutable record
(`core/nested_tool_calls.py`).

- **One feed per top-level tool call.** The tool wrapper
  (`tool_definition_wrapper.py`) is the owner when its context carries no
  inherited scope. It opens a `NestedCallFeed` (unbounded channel plus a
  close guard) and builds the call's context through
  `ToolContextFactory(tool_call_id, cancel, scope)`. The `NestedCallScope`
  (`parent_id`, `feed`, `holds_queue`, a guarded counter for `<parent>/<n>`
  ids) rides in the context. Pi's `scopes` map, `takeRecord` and `clear()`
  have no counterpart.
- **Producers** send `NestedCallStarted` before a nested call runs and
  `NestedCallFinished` after. They never touch a record. A send after the
  owner closed the feed is a no-op: close and "open? then send" are one
  step under the feed's guard.
- **Consumer**: when `execute` returns, the owner drains the feed and
  `fold_nested_calls` folds the messages in order:
  - `NESTED_CALL_LIMITS` (256 calls, 8 KiB per call, 32 KiB total, 500
    error characters) are applied in `started` order, so the same calls drop
    as in Pi.
  - A `started` without a `finished` is `unfinished`; `complete` means not
    truncated and nothing unfinished.
  - Usage is summed from `finished`.
  - If `execute` raises with calls recorded, the wrapper returns the error
    result the loop would build, so the record survives; a raise with no
    nested calls propagates unchanged.
- **Onto the message**: the fold rides on `AgentToolResult` (PiDrei-only
  `nested_calls`/`nested_usage`). The loop moves it off the result into a
  frozen `_NestedCallsRecord` before the hooks and `tool_execution_end`
  (they see the tool's own usage, as in Pi), and
  `_create_tool_result_message` combines the usage (`combine_usage` in
  `pidrei_ai/utils/usage.py`) and sets `nested_calls`.
- **Depth**: Pi's record is flat at the top, with caps applied across the
  tree. A nested tool runs through a bound copy (`with_nested_scope`) whose
  execute carries the inherited scope; the wrapper sees it and acts as
  producer, never owner.
- **Cancellation** follows the tree by scope: each nested call is awaited in
  its caller's coroutine.
- **Events**: every level's `tool_execution_start/update/end` (with
  `parent_tool_call_id` set to the immediate parent) goes through
  `await agent.observe(event)` (see `dispatch-observe`). The dispatcher's
  close and observe's "open? then send" are one step under `events_guard`;
  an observe with no run, or after the close, is dropped. Never emit nested
  events from the nested call's coroutine directly.
- **Exclusive queue**: Pi's `queueTail` promise chain is a FIFO mutex, so it
  is a TonIO `sync.Lock` held with `async with` around the tool call; a call
  cancelled while it waits leaves the queue. (Not an Event-tail chain: the
  place in line is taken after an await, and a chain strands on a cancelled
  waiter.) `holds_queue` rides in the scope, so a nested call inside an
  exclusive call never re-queues.
- The runner exists from session construction; Pi creates it lazily, which
  would race between parallel tools.
- **Unchanged from Pi**: ids, event payloads, caps, error truncation,
  `run_tool_call`, hook plumbing with the parent id, the sequential decision.
- **Deviation to keep**: the record is taken when `execute` returns, one
  step before Pi's `message_start`. A call still unwinding during the
  `tool_result` hooks is `unfinished` here, where Pi may report it
  `ok`/`error`.
- **Tests**: Pi's `nested-tool-calls.test.ts` drives the recorder directly;
  the mirror drives the fold with the same message sequence. Runner tests
  port through the wrapper.

### `tui-island` (`tui/src/tui.ts`, `terminal.ts`, `stdin-buffer.ts`, `tui-main-screen.ts`, `tui-alt-screen.ts`, `coding-agent/src/modes/interactive/interactive-mode.ts`, `core/extensions/runner.ts`)

PiDrei's UI is **passive state guarded by one reentrant thread lock**
(`state_lock`, with the terminal's lifetime), with independent loops (input,
render, timers, terminal events) working on it in parallel. The concurrency
contract is in `pidrei_tui/tui.py`'s module docstring. Component code is
untouched: component diffs port 1:1. What diverges:

- **Input** (`terminal.ts`, `stdin-buffer.ts`, `tui.ts` `handleInput`):
  - A read-ahead reader (`ProcessTerminal._read_input`) parses stdin with a
    synchronous `StdinBuffer`, whose flush timeout is a deadline the reader
    expires.
  - The reader completes the Kitty/DA negotiation (the Kitty activation is
    queued as an item and applied in input order) and hands terminal replies
    to the TUI's `_consume_terminal_reply`. Everything else is queued.
  - One consumer per terminal (`_consume_input`) runs Pi's `handleInput`
    stages under the lock (`_route_input`).
  - Diffs to parsing or to `handleInput`'s stages port into
    `stdin_buffer.py` / `_route_input`; diffs to how stdin is read or
    delivered land in the reader or consumer.
  - `stop()` drops queued items and never waits on input handling.
- **Synchronous input**: `handle_input`/`handle_mouse` and their callbacks
  are synchronous, as in Pi. Where Pi does I/O synchronously inside a key
  handler, PiDrei spawns it and registers the handle with
  `finish_before_next_input`, so the next key waits for it.
- **Rendering** (`requestRender`/`doRender`, both renderers):
  - `request_render()` sends a request object on a one-slot channel; a
    render loop draws one frame per request (no throttle; `force` rides in
    the request).
  - A frame is `_compose_frame` (tree walk and what input reads back, under
    the state lock), then `_write_frame` (diff and output, render lock
    only). Pi's `doRender` body splits across the two: reads of live state go
    in the compose half, the diff against the previous frame in the write
    half.
  - Diffs to *when* a frame is scheduled land in
    `request_render`/`_render_loop`.
- **Timers**: `pidrei_tui._timers.Timeout`/`Interval` are plain primitives.
  A callback that touches UI state takes the lock itself. A callback whose
  late run (a fire racing `cancel()`) would undo newer state either
  re-checks its own state, or mutates nothing and computes from a deadline
  instead.
- **`interactive-mode.ts`**:
  - Helpers mutate in place under `with self.ui.state_lock:`. The lock is
    reentrant, so a diff inside a helper ports 1:1 inside the hold.
  - The session listener applies `_handle_event` under the lock, dropping
    events from a session that is no longer current.
  - Flows Pi runs across awaits apply each stretch between two waits in one
    hold. A new awaited step splits the hold there; PiDrei-only I/O (a Pi
    sync call that is async here) is prefetched before the hold.
  - Handlers Pi writes as `async` but that are reached from sync callbacks
    run their pre-await prefix synchronously and spawn the rest with
    `_spawn_flow`.
- **Errors** that nothing up the stack can take go to `TuiBase.report_error`
  (the installed crash handler).
- **Extensions**:
  - `ctx.ui` setters and getters are synchronous, one hold each;
    `ctx.ui.apply(fn)` groups changes.
  - Dialogs mount at call time and return a `tonio.spawn` handle for the
    answer.
  - Component factories are synchronous and run under the lock.
  - Factories receive `ExtensionTui` (`modes/interactive/extension_tui.py`),
    a guarded wrapper, not the TUI. A new `TUI` member that Pi extensions use
    lands there, guarded.
  - Pi's global `setTimeout`/`setInterval` in an example become
    `tui.timeout`/`tui.interval`; examples never import private modules.

### `terminal-colors-loop` (`tui/src/tui.ts`, `terminal.ts`, `theme/theme-controller.ts`, `theme/theme.ts`)

Pi's `applyTerminalColors(reported)` is synchronous and called from three
places: the settled query, a late reply, and a scheme change. In PiDrei the
apply is async (`set_theme` reads theme files) and those callers are
separate coroutines, so interleaved applies could land an older report after
a newer one. Every colour report therefore goes through the TUI's
terminal-event loop.

- **TUI surface**: `query_terminal_colors(timeout_ms) -> tonio.Event` and
  `on_terminal_colors(listener)`. The sync prefix registers the query; the
  burst write (OSC 10/11/4;0–15 plus DA1) and the timeout run on their own
  coroutine. The returned Event is set by the terminal-event loop after the
  listeners handled the query's first report, or directly when the TUI is
  stopped. Startup awaits it before building the header.
- **Reader side** (`_consume_terminal_reply`): it accumulates OSC answers
  into the pending query under `_query_lock`. On the DA1 reply or the 18th
  colour it settles the query and sends the report; a reply after
  settlement is one more (late) report. A failed burst write reports an
  empty report. Nothing theme-related runs on the reader.
- **Every report** is sent under `_query_lock` to `_terminal_events` as
  `("colors", report, applied | None)`; `stop()` swaps and closes the
  channel under the same lock. The single consumer delivers colour reports
  to the `on_terminal_colors` listener and scheme reports to
  `on_terminal_color_scheme_change`, in arrival order.
- **Controller**: every theme application (settings, selections, previews,
  instance sets, both terminal listeners) runs under `_apply_lock` (a TonIO
  lock), so a terminal report never lands between a user's resolve and
  apply. No TUI lock is held across its awaits.
- **Theme globals** (`_terminal_colors`, `_terminal_colors_pending`,
  `_terminal_color_scheme`) live under `_theme_state_lock`, guarded by the
  conftest's `_terminal_colors_guard`. `Theme.colors` publishes one
  `(terminal, colors)` tuple.
- **DA1**: only the DA1 owed to the Kitty query is swallowed; later DA1
  replies reach `on_reply` so the colour query can settle on them.
- **Tests**: the colour, detection, controller and TUI query tests drive the
  reader with reply bytes and assert on what the consumer applied; the
  timeout path uses a manual clock.

### `subagent-inprocess` (`coding-agent/examples/extensions/subagent/index.ts`)

Pi's subagent example spawns a `pi` child process per task and parses its
JSONL stream. PiDrei runs each task as an **in-process `AgentSession`**
(`examples/extensions/subagent/__init__.py`, `run_single_agent`): on TonIO
the sessions run in parallel and a process buys nothing. The result-dict
contract is the seam: `messages` stay camelCase wire dicts, so mode logic,
params and rendering port side by side.

- **Field renames**: `exitCode`/`stderr` don't exist. `exitCode !== 0` →
  `is_failed_result(r)`; `exitCode === 0` → `not is_failed_result(r)`;
  `exitCode === -1` (running) → `is_running_result(r)` / `"status":
  "running"`; `stderr` → `errorMessage`.
- **Hunks in `runSingleAgent`** or the subprocess plumbing translate by
  intent into the in-process runner:
  - A new CLI flag becomes the matching `CreateAgentSessionOptions` field or
    construction argument.
  - JSONL-parsing changes map onto the typed session events.
  - Process and watchdog changes usually have no counterpart; cancellation
    is `cancel.on_cancel` → a spawned `session.abort()`.
- **Divergences to keep**:
  - Subagent sessions load no extensions.
  - An unknown frontmatter `model:` fails the task.
  - `on_update` streams per delta.
  - A PiDrei-only `concurrency` parameter bounds parallel mode.
- **Updates are snapshots**: `emit_update` sends `snapshot_result()`, since
  the runner keeps appending after handing an update off.
- **Parallel updates go through one aggregator**: each subagent stores its
  slot in a `tonio.Result` and ticks a channel; a single `aggregate()`
  coroutine builds every combined view and is the only caller of
  `on_update`. Upstream hunks to `emitParallelUpdate` port into
  `emit_parallel_update`, per-child update hunks into `task_update`.
- No tests (example extensions have none); Pi's subagent project-trust
  regression is dropped, since it depends on a child process failing at
  spawn.

### `no-telemetry` (`first-time-setup.ts`, settings-manager analytics accessors)

PiDrei sends no telemetry, so it doesn't ask about it either: the first-time
setup is theme-only, and `SettingsManager` has no
`enableAnalytics`/`trackingId` accessors. Upstream hunks in the analytics
step of `first-time-setup.ts`, those accessors, the startup wiring that
shares analytics, and their test cases are skipped; theme-step and
`shouldRunFirstTimeSetup` hunks port 1:1. A new upstream consumer of those
settings (a privacy command, an analytics sink) is a telemetry feature:
drop it the same way.
