# Changelog

Versions are `<pi version>.<pidrei build>`. The first three segments name the
Pi release this port tracks; the fourth counts PiDrei's own releases against it,
so `0.82.0.1` would be a PiDrei fix on top of the same Pi 0.82.0.

## [Unreleased]

## [0.84.1.0] - 2026-08-09

Tracks [Pi 0.84.1](https://github.com/earendil-works/pi/releases/tag/v0.84.1),
and folds in [Pi 0.84.0](https://github.com/earendil-works/pi/releases/tag/v0.84.0).

### Added

- Fullscreen TUI mode: `--tui-mode fullscreen`, the `tuiMode` setting, and a
  **TUI mode** row in `/settings` that switches renderers without restarting.
  Fullscreen keeps a sticky editor, status, widget and footer dock while the
  transcript scrolls independently, and adds a draggable scrollbar
  (`auto`/`always`/`hidden`), page and half-page scrolling, marked-message
  navigation, stacked transient notifications, double-click word and
  triple-click line selection, and an optional `scrollbarThumb` theme color
  that falls back to `selectedBg`.
- Remote sessions. Two new packages — `pidrei-protocol` (CBOR codec plus
  length-prefixed framing) and `pidrei-client` (transport-neutral `PiClient`
  and the `RemoteSession` controller with transcript reducers) — and a
  `pidrei-server` rebuilt on top of them, serving sessions over a Unix socket
  with durable snapshots and single-flight acquisition. This replaces the old
  supervisor/IPC server wholesale: the `pidrei-server` console script and its
  `serve`/`list`/`stop` commands are gone.
- `pidrei auth check`: verify a provider's or model's credentials before a
  run, with `--json` and an optional `--credentials` dump of the resolved
  credential.
- Qwen Token Plan Individual as a built-in provider, sharing the
  international `QWEN_TOKEN_PLAN_API_KEY`, and Baseten with
  `BASETEN_API_KEY` and `zai-org/GLM-5.2` as its default model. Model catalog
  refreshed from models.dev.
- `pi.register_markdown_transformer()`: chainable, display-only transforms
  over user and assistant Markdown, applied per render with the message type,
  streaming flag, and available width.
- Per-directory `AGENTS.override.md` context files, which replace `AGENTS.md`
  or `CLAUDE.md` in the same directory while leaving other directories'
  context intact.
- Arbitrary OpenAI-compatible sampling parameters through `samplingParams` in
  `models.json`, model overrides, extension providers and stream options,
  plus opt-in vLLM `thinking_token_budget` for models that share their output
  budget between reasoning and the answer.
- Extension `tool_call` handlers can set `terminate` on a blocked call, so a
  batch where every call is terminated skips the automatic follow-up model
  call.
- Opt-in `Ctrl+P`/`Ctrl+N` prompt-history navigation; explicit history
  bindings win over application shortcuts while the editor is focused.
- Terminal-friendly Unicode rendering for LaTeX expressions in Markdown.
- `AI_AGENT=pidrei` in CLI and RPC child-process environments, for tools that
  detect a coding agent generically. The variable keeps upstream's name on
  purpose — renaming it would defeat a cross-tool signal.
- Deferred provider requests (durable response handles, authenticated
  fetch/cancel dispatch, faux-provider support for pending, ready, failed and
  cancelled responses), support for OpenAI-compatible streams that omit
  `finish_reason` via `compat.supportsFinishReason`, structured Amazon
  Bedrock failure diagnostics (HTTP status, modeled error code, AWS request
  id), and `AgentOptions.should_stop_after_turn`.
- The agent harness moves to the v4 lane-based `Session`, `SessionStorage`
  and `SessionRepo` APIs — durable operation records, global facts, shared
  sequence numbers, tree-scoped lane views, an append-only
  `JsonlSessionRepo`, and bounded branch-entry and open-operation recovery
  queries.

### Changed

- **Breaking.** JSON and RPC `message_update` events carry only
  `assistantMessageEvent` deltas. The cumulative `message` and
  `assistantMessageEvent.partial` fields are gone — they made output grow
  quadratically. Clients that need a partial message assemble deltas between
  `message_start` and `message_end`; `message_end` remains authoritative.
- **Breaking.** `ModelRegistry.get_api_key_and_headers()` returns headers
  whose values may be `None`, preserving deletion markers instead of dropping
  them; this stops placeholder OpenAI credentials from reaching Cloudflare AI
  Gateway. `ModelRegistry.refresh()` takes refresh options and returns a
  result rather than discarding cancellation and provider errors, and
  `ModelRuntime.set_runtime_api_key()` now takes auth cancellation options —
  call `refresh(providers=[provider_id], cancel=...)` separately when remote
  freshness matters.
- **Breaking.** Dynamic providers read the `context.stored` snapshot and
  commit through the generation-checked `context.publish()` instead of
  touching the store directly. Providers built with `create_provider(...)`
  that only return fetched models need no change.
- **Breaking.** The legacy in-memory and JSONL harness repositories are
  removed in favour of the v4 `SessionRepo` implementations, and harness
  filesystems must implement `rename_file()` with same-filesystem replacement
  semantics for atomic JSONL publication.
- The bash tool's guideline about `PIDREI_*` environment variables is
  softened, to cut down on unnecessary inspection commands.
- Automatic terminal theme detection probes color-scheme and background
  support concurrently, halving its worst case from 200 ms to 100 ms.
- The fullscreen mouse wheel steps one line instead of three.
- `ModelsStore` reads, writes and deletions accept cancellation, and catalog
  orchestration binds those waits to the provider refresh token.
- Runtime dependencies updated: tonio 0.9.6 (cancelling a task that is
  waiting to acquire a lock or semaphore no longer strands the primitive, and
  two waker-cleanup fixes), httpunk 0.1.3 and punkreq 0.1.2.

### Fixed

- Responses truncated below their intended output limit compact and retry
  once instead of ending the run.
- Manual `/compact` no longer races threshold auto-compaction, and messages
  queued during a manual compaction are sent afterwards instead of failing.
- Extension TUI method wrappers no longer recurse indefinitely when
  delegating to the original method.
- Extension event-bus listeners are disposed with the session instead of
  surviving reloads.
- `set_tools_expanded(False)` is a no-op when tool output is already
  collapsed, so extensions stop emitting redundant `Tool output: collapsed`
  notices at startup.
- Oversized images returned by extension and built-in tools go through
  automatic resizing instead of bypassing it.
- Bare exact `--model` ids shared by several providers resolve to the sole
  authenticated provider, or fail with an explicit ambiguity error naming the
  candidates, instead of silently taking the first catalog entry.
- GitHub Copilot compaction and branch summaries use the credential-resolved
  Business or Enterprise endpoint rather than the Individual one, and
  extension model calls keep credential-resolved endpoints when forwarding
  request authentication.
- Project-level nested provider retry settings merge into global settings
  instead of replacing them.
- Session discovery finds sessions stored behind symlinked directories, and
  JSONL session ids are unique per working directory rather than globally.
- JSONL session forks and torn-tail repairs publish atomically, so an
  interrupted write cannot leave a partial session behind.
- `find` results at a filesystem root keep their first path segment and gain
  no duplicate separators.
- Tool-argument validation preserves values that already match an
  `anyOf`/`oneOf` arm before coercion, so a nullable union no longer turns
  `null` into another primitive.
- Fireworks GLM 5.2 requests drop the unsupported `prompt_cache_retention`
  field when long cache retention is on, and enable session affinity for
  automatic prompt caching. GitHub Copilot Grok 4.5 uses the Responses API.
- Malformed resource arrays in package manifests no longer crash session
  startup; invalid fields are ignored and the rest of the manifest is used.
- Long-running sessions pick up credentials written by another process
  instead of using stale ones, concurrent credential mutations no longer lose
  unrelated providers' updates, and concurrent model-store reads no longer
  form a lock convoy that delays startup.
- OAuth token refreshes release the credential-store lock when a request
  stalls, and waiting on a file-backed credential or catalog lock is
  cancellable — a cancelled mutation cannot commit later.
- Provider login no longer hangs after credentials are saved when a catalog
  refresh stalls; forced availability refreshes no longer queue behind a
  stalled earlier one; stale availability, pi.dev, llama.cpp and extension
  catalog results no longer publish after a newer pass; `/model` reports
  every catalog that failed; `/model <name>` and `/scoped-models` answer from
  the cache instead of waiting for a refresh.
- A provider stays authenticated right after logging into it: a background
  availability refresh running concurrently with the login no longer discards
  the login's result and republishes the state from before the credential
  existed.
- Concurrent writes to the same session no longer collide on the next entry
  sequence, which could abort a write with a non-consecutive-sequence error.
- A remote session runtime is disposed once when it reports a terminal error,
  rather than twice when the disconnect and the termination race.
- The model and scoped-model selectors no longer flicker their list empty when
  a background catalog refresh repopulates it mid-frame.
- Fullscreen shutdown no longer leaks terminal capability-query replies into
  the parent shell prompt, `Ctrl+X` copy confirmations show the transient
  `Copied!` marker instead of a transcript line, Kitty image previews stop
  overlapping the sticky dock while scrolling, image-heavy sessions no longer
  retransmit visible image payloads on every layout change, and fullscreen
  transcript navigation leaves `Ctrl`-modified `Home`/`End`/`PageUp`/
  `PageDown` available to the editor.
- Spaces typed in a `/settings` search no longer toggle the highlighted
  setting, so multi-word queries such as **TUI mode** work.
- Custom editors inherit the default editor's autocomplete dropdown item
  limit.
- The footer no longer shows `(sub)` for generic OAuth sign-ins without a
  known subscription; extension OAuth providers opt in with
  `is_subscription`.
- `/copy` reads clipboard text on Wayland when no X11 clipboard is available.

### Not ported

- Upstream's vendor-neutral telemetry contracts. pidrei ships no telemetry.
- Mermaid diagram rendering, which upstream implements with a JavaScript
  dependency.
- Windows fixes (drive-path resolution, right-click paste, `find` globs) and
  Bun/npm packaging fixes: pidrei is POSIX-only and is not distributed
  through npm.

## [0.83.0.1] - 2026-08-07

PiDrei fixes on top of Pi 0.83.0.

### Changed

- tonio updated to 0.9.4: fixes a per-task memory leak in the blocking thread
  pool and its thread-spawn accounting, so `spawn_blocking`-heavy sessions
  (file I/O, settings, auth storage) no longer leak references.
- The runtime is now sized for pidrei's I/O-bound workload instead of tonio's
  generic defaults: worker threads are the CPU count clamped to 2–8 (was
  uncapped, honoring affinity/cgroup limits), and the blocking pool caps at
  8 threads per worker (16–64, was a fixed 128). `PIDREI_THREADS` and
  `PIDREI_BLOCKING_THREADS` override either value, taken as-is without the
  clamp.

## [0.83.0.0] - 2026-07-30

Tracks [Pi 0.83.0](https://github.com/earendil-works/pi/releases/tag/v0.83.0).

### Added

- `pidrei auth print-api-key` / `pidrei auth print-bearer-token`: print a
  configured credential alone on stdout for external clients. Bearer tokens
  refresh through the normal request-auth path and honor `--min-expiry`
  (30 minutes by default).
- OpenRouter sign-in works over SSH: the PKCE flow races the loopback
  callback against a manual prompt, so pasting the final redirect URL (or the
  bare authorization code) completes login on remote/headless machines.
- Extensions can read `ctx.scoped_models` — the models scoped to the session
  (the `/scoped-models` set) — instead of enumerating the whole catalogue.
- Assistant messages carry `rawStopReason`, the provider's untranslated stop
  reason, alongside the mapped one; unmapped provider stops now surface as
  "Provider stopped with: ..." errors instead of silent stops or generic
  messages.
- Streaming partials expose a `pending` stop reason until the provider sends a
  real one, and a stream that ends without a stop reason is an error instead
  of a fake success.
- GitHub Copilot gains Claude Opus 5 (Anthropic Messages API, adaptive
  thinking, 1M context, `minimal` thinking-level override). Qwen Token Plan
  reasoning models expose `reasoning_effort` thinking levels. Model catalog
  refreshed from models.dev.

### Fixed

- OAuth credentials with less than five minutes of validity refresh before a
  request instead of at expiry, so tokens no longer die mid-request.
- Aborting via session switch, resume, or tree navigation settles the active
  response first: the aborted turn and its tool results persist instead of
  leaving a dangling tool call. Navigating the session tree during a response
  is rejected with a clear message.
- Concurrent `!` bash commands can each be cancelled; aborting cancels all of
  them and finishing one no longer marks the rest as done.
- RPC-mode `bash` commands now emit the `user_bash` extension event instead of
  bypassing extension interception.
- Switching sessions during startup no longer duplicates messages (the stale
  startup rebind is dropped when a replacement session takes over).
- The `/model` selector highlights the top (best) match when typing a query
  instead of leaving the highlight where it was.
- Failed git installs clean up their partial checkout instead of leaving a
  broken package behind.
- Z.AI providers receive `max_tokens` (they ignore `max_completion_tokens`),
  so the configured output cap applies again.
- Explicitly configured Bedrock profiles win over ambient
  `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`.
- Extension-provided skills and prompts no longer erase package metadata on
  reload (autocomplete source tags stay correct).
- AGENTS.md is no longer loaded twice when running from a linked git worktree
  nested under its main repository.
- File-backed SYSTEM.md and APPEND_SYSTEM.md show up in the startup
  `[Context]` section.
- Toggling tool-output expansion shows a status line.
- Long image-fallback paths are home-shortened, hyperlinked (OSC 8) when
  supported, and clamped to the terminal width instead of crashing the TUI.
- Malformed OpenAI-compatible tool-call deltas that carry an empty `custom`
  object no longer lose their parsed function arguments.

## [0.82.1.0] - 2026-07-28

Tracks [Pi 0.82.1](https://github.com/earendil-works/pi/releases/tag/v0.82.1)
— the first upstream sync, and mercifully shorter than the last entry.

### Added

- `ANTHROPIC_AUTH_TOKEN` support: a bearer token from the environment (or a
  gateway) now authenticates Anthropic requests via an `Authorization` header,
  without OAuth request shaping. `ANTHROPIC_OAUTH_TOKEN` and
  `ANTHROPIC_API_KEY` keep their precedence for API-key auth.
- Claude Opus 5: model settings (adaptive thinking, `xhigh`/`max` effort, no
  temperature) on Anthropic and Amazon Bedrock, exposed through an inference
  profile on Bedrock. Model catalog refreshed from models.dev.
- Custom message renderers receive `outputPad` in their options, so extension
  messages can line up with the configured transcript padding.
- Remote model catalogs revalidate with ETag/`If-None-Match`: unchanged
  catalogs cost a 304 instead of a download, and a transient failure keeps the
  cached overlay and its validator.

### Fixed

- The scoped-models selector keeps configured-but-unavailable models listed
  (marked `[unavailable]`) and editable instead of silently dropping them —
  removing a model's provider no longer erases your saved selection.
- `ModelsError` messages keep the underlying cause (an OAuth refresh failure
  now says why).
- A directory named `AGENTS.md` (or `CLAUDE.md`) no longer breaks context-file
  discovery; it is skipped and the next candidate loads.
- Interactive sessions no longer error with "Object of type EditToolDetails is
  not JSON serializable" after the edit tool runs: tool details now persist to
  the session file as plain camelCase objects, like Pi's.

## [0.82.0.0] - 2026-07-27

First release, tracking
[Pi 0.82.0](https://github.com/earendil-works/pi/releases/tag/v0.82.0).

Itemising what changed would mean itemising the whole project, so: all of it.
[The README](README.md) is the honest summary — what works, what doesn't yet,
and where PiDrei differs from Pi on purpose. Later entries will have the decency
to be shorter.
