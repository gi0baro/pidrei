# UI island

How PiDrei's terminal UI (`pidrei_tui` and interactive mode) stays correct
when input, rendering, timers, terminal replies and agent events all run in
parallel. `spec/concurrency.md` has the general model; the translation
recipes for upstream diffs are `tui-island` and `terminal-colors-loop` in
`spec/upstream-sync.md`. The contract is also stated, briefly, in
`pidrei_tui/tui.py`'s module docstring.

## What Pi promises, and what PiDrei delivers

Pi runs its whole UI on JavaScript's one thread. PiDrei ports the behaviour
that thread gives, not the thread itself:

1. **Input order.** Items (keys, pastes, clicks) are handled in arrival
   order, and each one sees the previous one's effects.
2. **Input never freezes on Pi's async work.** Work Pi puts behind an `await`
   doesn't hold input. Pi's *synchronous* I/O does hold input while it runs,
   and so may PiDrei's.
3. **UI changes are whole.** A frame never shows a half-applied change.
4. **Per-source order.** One source's changes appear in the order it made
   them. For agent events, the UI already reflects an event when the emit
   returns.

## The design

- **The island is passive.** It is UI state plus one guard. It is not a
  coroutine, it runs nobody's code, and it knows nothing about its callers.
  There are no jobs posted to an owner.
- **Independent loops run in parallel**: input, render, timers and terminal
  events. None of them waits on another.
- **A mutation is a call through the guard.** Calling it and waiting is
  run-and-wait; spawning the call is fire-and-forget.
- **Each function is one kind of code**: either a synchronous mutation that
  runs under the guard, or coroutine code that awaits work and takes the
  guard for its synchronous stretches. A function reached from both sides
  is a design gap to fix, never something to patch with an "am I on the UI
  side?" flag.

### The guards

- **The state lock** (`state_lock`) is a reentrant thread lock with the
  terminal's lifetime, shared by every TUI on that terminal (the main-screen
  and alt-screen renderers). It guards all component and TUI state. Every
  mutation or read is a short synchronous section, and **nothing is ever
  awaited while holding it**. Blocking a thread on it is acceptable. It is
  reentrant, so a helper that mutates can take it itself.
- **The render lock** is a TonIO lock, one per TUI. It allows one frame at a
  time and guards render-private state: the previous frame, diff
  bookkeeping, the cursor, the frame writer.
- **One-at-a-time locks** (TonIO locks) serialize operations that must not
  overlap: `AgentSession._model_change_lock` around model changes and
  cycles, `AgentSessionRuntime._replacement_lock` around session
  replacement (new, switch, fork, import). Extension callbacks that can
  re-enter (`model_select` handlers, `with_session`) run after the lock is
  released. The bash "already running" check-then-act is a claim flag,
  checked at submit and released in the flow's `finally`.

## Rendering

- **Requests**: `request_render()` is synchronous, lock-free and callable
  from anywhere, including a component's own `render`. It sends a request
  object on a channel of size 1 with `send_nowait`. If the channel is full,
  a request is already pending and covers this one; if it is closed, the TUI
  is stopped.
- **The render loop** (one per start) receives a request, installs a fresh
  request object, reads its `force` flag once, and draws one frame. Because
  it receives *before* walking the tree, any change made after the receive
  queues exactly one more frame: no update is lost, and duplicates collapse.
- **`force`** (drop the differential state and repaint everything) is a
  field of the pending request object. A forced request sets it before its
  send, so even a send dropped as full lands in the pending request. A race
  can repeat a full redraw, never lose one.
- **No throttle.** The loop is receive, frame, receive. While a frame runs,
  requests collapse into the pending one, and the frame writer paces a slow
  terminal. Pi's 16 ms throttle and its keyboard fast path have no
  counterpart, and nothing in the loop waits on time.
- **A frame** is two halves:
  - `_compose_frame` runs under the state lock: the tree walk (components'
    synchronous `render(width)`, which reads live fields and fills caches),
    overlays, the cursor, and publishing what input reads back (layout,
    overlay bounds, the mouse map, the alt screen's visible screen).
  - `_write_frame` runs under the render lock only: line resets, the diff
    against the previous frame, output.

  In this component model the tree walk *is* the snapshot. Moving it out
  from under the state lock would need immutable per-component render data:
  possible, not required.
- **Frame cost comes from caches, never from partial work.** A streaming
  message's Markdown is re-lexed whole on every frame on purpose: the tail can
  still change what came before it (lazy continuation, reference
  definitions, an unclosed fence), so a cached lex prefix would render wrong
  output. What is cached is each block's rendered lines, keyed on the lexed
  token, the width and the next block's type, so an unchanged block costs a
  dictionary lookup instead of highlighting and styling
  (`pidrei_tui/components/markdown.py`).
- **Stop** closes the request channel and joins the loop. Render requests
  made while stopped are dropped; `start()` makes Pi's plain first request,
  and restarts after a terminal handoff force a full redraw, as in Pi.

## Input

- **The reader reads ahead.** `ProcessTerminal._read_input` (one per start)
  reads stdin, parses it with a synchronous `StdinBuffer`, and pushes items
  into an ordered channel. It never waits for the consumer.
  - Parser deadlines belong to the reader: the lone-ESC flush and the Kitty
    negotiation's split-reply flush. The reader waits for "more bytes, or the
    next deadline", so a flushed ESC enters the channel in its place.
  - The reader completes the Kitty/DA negotiation. The Kitty activation is
    queued as an item, so it applies in input order.
  - Terminal replies are split off to the TUI's `_consume_terminal_reply`:
    colour-query answers and colour-scheme reports go to the terminal-event
    loop (see the `terminal-colors-loop` recipe). Mouse, focus in/out and
    paste stay in the key stream; a paste is one item. Cell-size replies are
    handled in the consumer, after the input filters, as in Pi.
- **One consumer per terminal** (`_consume_input`, for the terminal's
  lifetime) takes each item and, under the state lock, runs Pi's
  `handleInput` stages in Pi's order (`_route_input`): mouse and focus,
  input filters, the cell-size reply, the debug key, overlay focus routing,
  then the focused component. "The component asking for input" is the
  focus, resolved per item. Filters and routing read focus, so they run
  inside this consumer, never ahead of it.
- **`handle_input` and `handle_mouse` are synchronous**, as in Pi, and so are
  the callbacks they invoke (`on_select`, `on_cancel`, `on_submit`, key
  actions). Each does its immediate effects in the call and spawns slow work.
  A synchronous handler can't wait on anything, which makes "never hold input
  on Pi's async work" structural.
- **Generations.** Every handler change (`start`, `stop`, `drain_input`)
  bumps an input generation under the state lock, and the consumer drops
  items tagged with an older one.

### When a key must wait for I/O

Pi's synchronous I/O and PiDrei's `await` mean the same thing, run and wait,
and it is the only way to guarantee the next key sees the effect. A
synchronous handler under a thread lock can't await, so:

1. The handler does its synchronous part and starts the I/O with
   `tonio.spawn()` (not `spawn.without_tracking`).
2. It registers the handle with `finish_before_next_input(handle)`.
3. When the current item ends, the consumer releases the state lock and
   awaits every registered handle before taking the next item. Awaiting a
   handle re-raises its exception into the consumer's error routing.

`finish_before_next_input` raises when called outside input handling: a
handle registered elsewhere would be awaited by an unrelated key, or never.
While the consumer waits, render and agent events can still take the lock;
in Pi nothing runs during a synchronous write, but the next key still sees
the effect, which is the promise.

Examples: cycling the thinking level, theme previews, trust confirmation,
tree label edits, commands that are synchronous in Pi but start with I/O in
PiDrei (`/settings`, `/trust`, `/changelog`, `/name`, `/debug`).

### Stopping the UI from inside a key

Ctrl+Z, the external editor, the settings renderer switch and the config
selector's exit all stop the UI from a key handler.

- **Stop never involves the input consumer.** It stops the reader (taking
  its parser state with it), gates render, drains output and restores the
  terminal. The consumer stays alive across stop and start; with the reader
  stopped, its channel is simply empty. There is no self-join.
- **The stopping key waits only for the stop**, through a completion.
  Ctrl+Z's completion stops and sends SIGTSTP; a spawned flow
  (`_resume_after_suspend`) owns the SIGCONT receiver and restarts the TUI.
  The external editor's flow is stop, run the editor, restore the text,
  start; the key waits for the stop part only.
- **Items queued at stop are dropped**, with the parser's partial state.
  Because the reader reads ahead, typeahead that Pi would leave in the
  kernel buffer for the editor or the shell is lost. This is an accepted
  divergence. `drain_input` drops items queued behind the one being handled,
  for the same reason.
- **Timers and agent events keep running while stopped**, as in Pi.

## Whole changes

### Flows that await several times

Session replacement (`/new`, `/resume`, `/fork`, `/clone`, `/import`),
`/reload`, tree navigation, login and model selection all await several
times. The rules:

- **One hold per stretch.** Everything a flow changes between two of its
  waits is applied in *one* hold of the state lock. A frame can then only
  show the states Pi can show, the ones at the flow's waits: `/new` never
  shows the new footer over the old chat.
- **Waits only PiDrei has.** Where one of Pi's no-await blocks contains I/O
  that PiDrei awaits (a cwd resolution, a trust save, keybinding loads), do
  the I/O first, then apply the whole block in one hold. Split the block
  only where the I/O depends on the block's earlier changes.
- **State render reads that isn't UI state** changes only under the state
  lock, in the same hold as the rest of the change: the current session
  reference and the global theme. The session swap is handed to the UI as a
  callback (`rebind_session(new_session, swap)`) so it lands in the same hold
  as the rebind. The theme swap goes through the `on_theme_change` callback
  in the same way. Render then sees the old pair or the new pair, never a mix.
- Callbacks into the UI from slow work take their own hold.

### Agent events

Applying an agent event is a direct guarded call inside the session
listener, as synchronous as Pi's listener. The dispatcher awaits the
listeners, so when an emit returns the UI already reflects the event (the
fused-emit contract). Each apply checks, inside its hold, that its session
is still current (Pi's `this.session !== session`). An exception propagates
to the run, as in Pi.

## Timers

`Timeout` and `Interval` (`pidrei_utils.timers`) know nothing about the UI,
as `setTimeout` doesn't. A callback that touches UI state takes the lock
itself (components use `apply`). A callback whose late run matters (a fire
racing `cancel()`) either re-checks its own state, or mutates nothing and
computes from a deadline at render time. For example, the scroll view's
auto-hiding scrollbar and the session selector's status line are shown until
a deadline that render compares with the clock; their timers only request
frames.

## Terminal-level state

- The Kitty protocol flag is set by the consumer, under the state lock, when
  the reader's activation item reaches it.
- Terminal capabilities (overrides and the detection cache) are one
  `_CapabilityState`, replaced whole under a module lock;
  `replace_capabilities` is a compare-and-swap. A cache computed from old
  overrides is not stored.
- Cell dimensions are read and written under the state lock.
- Terminal colours and the colour scheme: see `terminal-colors-loop`.

## Errors

Follow Pi, caller by caller:

- **Loops** (the input consumer, the output pump, the terminal-event loop,
  render) catch around each step and hand the error to
  `TuiBase.report_error`, which routes it to the installed crash handler.
  `ProcessTerminal` gets it through `start(on_input, on_resize, on_reply,
  on_error)`.
- **Coroutine code that mutates directly** (applying agent events, session
  emits, flow stretches) lets the exception propagate to its caller, as Pi
  does. A listener's error fails the run.
- **Spawned flows** route whatever escapes them to the crash handler at
  their top level (interactive mode's `_spawn_flow`), the equivalent of Pi's
  unhandled rejection.
- A timer callback's exception goes to its `on_error`; without one it is
  dropped (an `Interval` stops).

## Extensions

Extension code comes in two kinds, and each follows the same rules as
PiDrei's own code:

| Kind | What | Runs on | Gets |
|---|---|---|---|
| Handlers | event handlers, commands, shortcuts, tools | their own coroutines, async | `ctx.ui` |
| Components | factories, `render`, `handle_input`, timers, input listeners, renderers | the island's loops, under the lock | the guarded `tui`, `done` |

Extensions are given only safe primitives: everything they can reach either
takes the lock itself or needs none. The raw state lock is not reachable.

### `ctx.ui`

- **Setters and getters are synchronous**, each one call through the lock:
  the change is whole, lands in call order, and the extension's next line
  sees it. `get_editor_text()` is synchronous. `paste_to_editor(text)` calls
  the editor's `handle_input` with the bracketed paste directly, under the
  lock, whatever has focus (Pi's behaviour); it doesn't go through the input
  stream.
- **`ctx.ui.apply(fn)`** runs a synchronous `fn` under the lock and returns
  its result. It raises `TypeError` for an `async def` or a returned
  coroutine (`call_sync`), so awaiting under the lock is impossible. It is
  the only way to group several changes into one frame: Pi's implicit
  grouping would need the lock held across the extension's awaits.
- **Dialogs** (`select`, `confirm`, `input`, `editor`, `custom`) are plain
  functions. They mount at call time, under the lock, and return the handle
  of `tonio.spawn(<wait for the answer>)`. `await ctx.ui.select(...)` reads
  as in Pi. Called from a synchronous `handle_input`, the dialog is mounted
  before the next key is routed, which Pi guarantees by mounting inside its
  promise executor. `cancel` and `timeout` close the dialog through the lock.
- **`custom(factory, options)`**: the factory is synchronous and runs under
  the lock, in one hold with the editor-text snapshot and the mount. Async
  preparation goes before the `custom()` call; async work the component
  needs is spawned from the factory. A terminal handoff (stop, run a child,
  start) is spawned from the factory, which returns an empty component.
  `done(result)` is synchronous, takes the lock itself, and is ignored once
  closed.
- **`on_terminal_input`** listeners are synchronous, run inside input routing
  under the lock, and return `{consume, data}`.
- The no-op (print/JSON) and RPC contexts have the same shape: synchronous
  getters and `paste_to_editor`, `apply`, dialogs returning an awaitable
  handle, and `theme` as the global theme. `get_all_themes`/`get_theme` are
  awaitable everywhere (they read theme files).

### The `tui` extensions receive

Every factory an extension provides (`custom`, widget, footer, header,
editor) receives `ExtensionTui` (`modes/interactive/extension_tui.py`), a
guarded wrapper, not the TUI. Components built with it hold the wrapper too.
Its surface:

- `request_render(force=False)` and `report_error(error)`, which need no lock;
- `apply(fn)`, the same primitive as `ctx.ui.apply`;
- `spawn(coro)`: fire-and-forget, with escaping errors going to
  `report_error`;
- `set_focus`, `show_overlay`, `hide_overlay`, each taking the lock;
  `show_overlay` returns a guarded overlay handle (`guard_overlay_handle`);
- `timeout(ms, fn)` and `interval(ms, fn)`, returning a handle with an exact
  `cancel()`. The callback runs under the lock, and an `async def` callback
  is refused at creation. `cancel()` marks the handle under the lock and
  the fire re-checks the mark under the lock, so a fire already waiting for
  the lock doesn't run, matching `clearTimeout`;
- `finish_before_next_input(handle)`, async `stop()`/`start()`, and the
  terminal's size (`rows`, `columns`).

A new `TUI` member that Pi extensions use lands on the wrapper, either
guarded (`apply` around any mutation) or needing no lock. Examples never
import private modules, and never arm `pidrei_utils.timers` directly: they
use `tui.timeout`/`tui.interval` (callbacks run under the UI state lock, and
errors go to `report_error`) where Pi uses the global timers.

### Component code

- `render`, `handle_input` and `handle_mouse` are synchronous; so are input
  listeners, tool/message/entry renderers and markdown transformers, all
  called under the lock.
- The one rule Pi doesn't need: an extension changes its own component's
  state from a spawned coroutine only through `apply`. Pi gets that from
  its single thread; PiDrei can't guard the fields of arbitrary objects.
- Calling `ctx.ui` from component code is allowed: setters are synchronous,
  the lock is reentrant, and dialogs mount eagerly.

## Accepted divergences from Pi

- Typeahead queued when the UI stops (Ctrl+Z, external editor, drain) is
  dropped instead of being left for the next reader.
- Pi's 16 ms render throttle and keyboard fast path are gone.
- During a terminal handoff, the other loops keep running (Pi's
  `spawnSync` freezes everything); render requests are dropped while stopped,
  so nothing reaches the screen early.
- PiDrei-only API: `apply`, `tui.spawn`, `tui.timeout`/`tui.interval`,
  `finish_before_next_input`, async `stop()`/`start()`, `Terminal.close()`/
  `TuiBase.close()`, `Terminal.start`'s `on_reply`/`on_error`.
- A `custom` factory returning a promise (Pi) has no counterpart: factories
  are synchronous.

## Testing the island

- `VirtualTerminal` (`packages/tui/tests/virtual_terminal.py`) renders
  frames into a pyte screen. `wait_for_render(since)` waits for a frame
  after `since`, then `settle()`s: settling counts render requests and the
  requests each frame covers, so tests wait on real frames, not time.
- Timer-driven components are tested with manual timers or a manual clock,
  never real sleeps.
- pty tests drive a real `ProcessTerminal` from the master side. Their scopes
  use `cancel_on_exc=True`, so a failing assertion fails instead of hanging.
- New ordering guarantees get a test that fails under the mutation that
  breaks them.
