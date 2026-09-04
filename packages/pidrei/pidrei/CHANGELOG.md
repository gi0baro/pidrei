# Changelog

Versions are `<pi version>.<pidrei build>`. The first three segments name the
Pi release this port tracks; the fourth counts PiDrei's own releases against it,
so `0.82.0.1` would be a PiDrei fix on top of the same Pi 0.82.0.

## [Unreleased]

### Removed

- The remote-session packages `pidrei_protocol`, `pidrei_client` and
  `pidrei_server`, together with the agent harness runtime scaffold and the
  experimental `server`/`client` CLI mirrors. They tracked pi's experimental
  server/worker architecture, which pi rewrote from scratch in 0.85 and which
  no pidrei feature uses; only the transport layer (CBOR, framing, unix
  sockets) stays in the workspace, unpublished. See
  `UPSTREAM_EXPERIMENTAL_RULING.md` in the repository.

## [0.84.4.1] - 2026-09-04

### Fixed

- Model catalog refreshes no longer fail when the pi.dev catalog carries a
  compat field this release does not declare (such as `supportsMidConvoEffort`
  from pi 0.85). Unknown compat keys are dropped, as pi does, instead of
  rejecting the whole provider catalog on every refresh.

## [0.84.4.0] - 2026-08-30

### Added

- Terminal capability overrides: force or disable inline images, truecolor,
  and OSC 8 hyperlinks via the `terminal.images` / `terminal.trueColor` /
  `terminal.hyperlinks` settings or the `PIDREI_IMAGE_PROTOCOL` /
  `PIDREI_TRUE_COLOR` / `PIDREI_HYPERLINKS` environment variables.
- A `fullscreenCopyOnSelect` setting: disable automatic copy-on-select in
  fullscreen mode and copy the active selection with `ctrl+x` instead.
- `ui_prompt_start` / `ui_prompt_end` extension events fire around blocking
  extension UI prompts so host integrations can report "waiting for user".
- An RPC `clear_queue` command (and `RpcClient.clear_queue()`) that clears
  queued steering and follow-up messages and returns their text.
- New models: DeepSeek V4 Flash Vision Exp, Cloudflare Workers AI
  passthroughs on the AI Gateway catalog, and a refreshed image model catalog.
- OpenRouter reasoning controls are derived from OpenRouter's model metadata,
  so reasoning-mandatory models no longer receive `effort: "none"`.

### Fixed

- Compaction now runs between a turn's tool results and the next provider
  request in the same run, instead of waiting for the run to finish; steering
  queued while compaction runs is included in the resumed request.
- Truncated compaction and branch summaries (a `length` stop) are rejected
  instead of being persisted as session checkpoints.
- Custom messages sent while the agent is running are appended after the
  turn's tool results instead of landing between a tool call and its result.
- Persisting a default model with `--models` in effect adds it to the scoped
  model list and `enabledModels` instead of silently escaping the scope.
- OpenRouter `reasoning_details` deltas are merged into logical entries and
  the thinking signature is serialized once when the block completes.
- Explicitly requested `toolChoice` is honored even when no tools are defined,
  and compaction requests no longer force `toolChoice: "none"`.
- Fragmented Mistral tool calls no longer split when continuation chunks omit
  the tool-call ID.
- Valid session files with an unterminated final line are repaired on load so
  later entries cannot fuse with the tail.
- Fullscreen double-click word selection keeps paths and kebab-case tokens
  whole, and large main-screen renders are written in bounded 1 MiB chunks.
- Toggling thinking-block visibility no longer rebuilds live tool components,
  preserving partial bash output.
- `@` autocomplete ranks direct children of the scoped directory before deeper
  same-score matches, even when recursive results are flooded.

## [0.84.3.3] - 2026-08-29

### Changed

- Bumped tonio to 0.9.14.

### Fixed

- Terminal teardown no longer self-sends SIGWINCH to unpark the resize
  watcher (the watcher now unwinds through scope cancellation), and the
  watcher is only started when a resize handler is actually provided.

## [0.84.3.2] - 2026-08-28

### Changed

- Refactored internals for multi-threading: the TUI runs as a single-task
  island that owns all component mutation and rendering, agent events are
  frozen into immutable snapshots at the session seam, settings/models/auth
  and the provider registry publish atomic epoch snapshots with lock-free
  readers, and the agent's run lifecycle and pending-message queues are folded
  into one internal mailbox task.
- Reworked the subagent example extension to run subagents as parallel
  in-process agent sessions instead of `pidrei` subprocesses.

## [0.84.3.1] - 2026-08-26

### Changed

- Updated punkreq and httpunk dependencies.

## [0.84.3.0] - 2026-08-24

Tracks [Pi 0.84.3](https://github.com/earendil-works/pi/releases/tag/v0.84.3).

### Added

- A `/thinking` selector, searchable default choices in the model and
  thinking selectors, and `Ctrl+S` to save the selected model as the global
  default.
- `session_compact_failed` extension events, exposing a compaction failure's
  reason, retry state, source and error message to handlers.
- Transcript usage notices for compaction and branch summaries when cache
  miss notices are enabled.
- Optional routing session IDs on the exported compaction summary helpers, so
  callers can preserve provider routing without enabling prompt cache writes.
- Provider-neutral `tool_choice` on simple stream requests, honored by every
  adapter.
- Automatic Anthropic server-side refusal fallback for supported first-party
  models, priced with the model that actually answered.
- Configurable OpenAI-compatible thinking-token budget fields for vLLM,
  Qwen/SGLang and llama.cpp servers.
- China-specific Z.AI Coding Plan models, including GLM-4.6V vision support
  and API-equivalent usage cost estimates.
- `deepseek-v4-pro-0813` in the Qwen Token Plan Individual catalog.
- Settings diagnostics: invalid settings files are reported with their path
  inside the TUI instead of scrolling past during startup.

### Changed

- **Breaking (SDK types):** the inherited `GoogleThinkingLevel` type is now
  `GoogleApiThinkingLevel`, and `ResolvedGoogleThinkingLevel` names the
  normalized adapter level.
- `/model` and `/thinking` selections are session-scoped: they no longer
  persist globally unless explicitly saved with `Ctrl+S`.
- Built-in xAI models use the Responses API with encrypted reasoning replay,
  and Grok 4.6 is the default xAI model.
- The Anthropic, Azure OpenAI, Google, Mistral and OpenAI adapters send
  pidrei's default `User-Agent` unless a caller overrides it.
- Keybinding defaults shift under WSL, where the Windows terminal reserves
  several chords: `alt+p` cycles to the previous model, `ctrl+q`/`alt+q`
  queue and restore follow-ups, `alt+v` pastes an image, `alt+z` undoes an
  edit, and fullscreen uses `ctrl+f` to search and `ctrl+up`/`ctrl+down` to
  jump between marked messages. `ctrl+up`/`ctrl+down` also work as marked-
  message navigation everywhere, alongside the existing `ctrl+shift` chords.
- Compaction and branch summarization requests no longer expose tools to
  providers.
- Model catalog refreshed from models.dev, OpenRouter and the Vercel AI
  Gateway.
- The default mistral model is `mistral-medium-latest`: upstream still names
  `devstral-medium-latest`, which models.dev retired before this refresh.

### Fixed

- Extension factories that fail partway through no longer leave their event
  subscriptions, provider registrations and default flag state behind, and
  the API object they captured is disabled.
- `models.json` accepts the documented OpenAI-compatible
  `compat.supportsFinishReason` provider and model override.
- JSON and RPC `toolcall_start` events carry the tool call id and name.
- Nested Markdown skills inside `.agents/skills/` grouping directories are
  discovered, and root Markdown files such as `README.md` and `AGENTS.md` in
  skill directories are no longer reported as broken skills.
- Single-object `edit` tool inputs validate as a one-edit array, in both the
  coding-agent and harness edit tools.
- `pi.register_flag()` rejects default values that do not match the declared
  flag type.
- The default Cerebras model no longer references an unavailable Z.AI model,
  and the Z.AI Coding Plan defaults no longer reference the removed GLM-5.1.
- OpenAI-compatible Chat Completions reasoning replay preserves and resends
  assistant-level `reasoning_details` verbatim and in order.
- GitHub Copilot login no longer trips model-policy rate limits: policy
  updates are sequential, model discovery retries once, and server retry
  delays are honored.
- Amazon Bedrock replays opaque redacted reasoning from non-Anthropic models,
  and its response hooks receive the raw response headers instead of a
  synthesized request id.
- Z.AI Coding Plan models derive complete reasoning-effort metadata,
  including GLM-5.3's low, high and max levels.
- DeepSeek V4 Flash on OpenCode and OpenCode Go exposes its low thinking
  level.
- Azure OpenAI Responses honors `tool_choice` in provider-specific stream
  requests.
- Kimi usage reporting counts top-level `cached_tokens` as cache reads
  instead of normal input tokens.
- Google custom models honor `thinkingLevelMap`, restoring extended thinking
  controls.
- Writes to `auth.json` and `models-store.json` preserve
  administrator-managed file permissions.
- UTF-8 BOM markers no longer prevent frontmatter and user configuration
  files from loading.
- Repeated ambiguous truncated-response recovery is no longer mislabeled as
  context overflow.
- Threshold auto-compaction still runs when providers omit streaming usage
  data.
- Branch summary entries record the pre-navigation source leaf in `fromId`
  instead of the navigation destination.
- Package update checks no longer treat older registry versions as available
  updates.
- Hung pi.dev model catalog requests retry instead of consuming the whole
  refresh deadline.
- Xiaomi model catalogs no longer list shut-down MiMo V2 models.
- llama.cpp login guidance points at `/llama` before `/model` when no local
  models are loaded.
- Dash-prefixed prompts are no longer parsed as options: `--` ends the option
  list.
- Padded text no longer exceeds narrow terminal widths, and wrapped Markdown
  table links no longer leak color into borders and neighboring cells,
  including tables inside blockquotes.
- Duplicate fullscreen right-click paste in VS Code-based terminals.
- RPC `get_available_models` serializes models whose compat carries
  `allowedFallbackModels` instead of failing the request.

### Not ported

- Upstream's optional `powershell` tool: it is Windows-only by construction
  and pidrei is POSIX-only. The refactor that lets one implementation back
  several shell tools did land.
- Upstream's Radius session-sharing changes (clickable links, canonical
  artifact URL, system prompt and tool definitions in shares): Radius is
  long-dropped surface. The JSONL session export they were built on is now
  reusable as `core.session_export`, and `/share` still creates a gist.
- Installer-managed updates (staged, verified, atomically activated
  releases): pidrei has no self-update machinery.
- The Node.js/Bun packaging work — bundled runtime and CLI entrypoints,
  single-executable extension loading, lazy jiti/Babel, deferred
  highlight.js grammars, Bun archive layout, and the npm dependency-tree
  reductions. None of it has a Python analogue; pidrei highlights with
  pygments.

## [0.84.2.8] - 2026-08-24

### Fixed

- The remaining permanent TUI freeze (hit mostly on macOS, where the ~1KB tty
  output queue makes the window frequent): the terminal pumps' `EAGAIN` paths
  called `consume_r`/`consume_w`, which drop readiness unconditionally — a
  readiness edge landing between the failed `os.read`/`os.write` and the
  consume was eaten, and the next `arm_r`/`arm_w` parked forever on an
  already-ready fd (edge-triggered, so no further event ever comes). A wedged
  output pump froze the whole UI — writes, rendering and input all funnel
  into it — with no exception for the 0.84.2.7 crash guards to catch. The
  pumps and `FdReader`/`FdWriter` (output guard, RPC stdin) now use the
  tick-guarded `clear_r`/`clear_w` from tonio 0.9.12, which keep readiness
  that arrived after the last arm.

### Changed

- tonio requirement bumped to `~=0.9.12` (exposes `ScheduledIO.clear_r`/
  `clear_w`).

## [0.84.2.7] - 2026-08-24

### Fixed

- The random-looking permanent TUI freeze introduced in 0.84.2.5: detached
  handlers (editor submit, follow-up, external editor, clipboard paste,
  extension and selector actions) mutated editor state while the input owner
  was mid-keystroke; the resulting `IndexError` killed the stdin pump and
  input was dead for good. Editor mutations from off the owner now go through
  the owner task (`_post_editor_mutation`), restoring the single-writer
  contract (concurrency audit §4.4) across the coding-agent layer too.
- Long-lived tasks no longer die silently on an exception: the stdin pump,
  the output pump, the frame writer, the render loop, posted owner work and
  the autocomplete request task route errors to the crash handler (and the
  pumps keep running), instead of leaving a frozen UI with a live agent.
  `BaseException` is caught where a pyo3 `PanicException` would otherwise
  slip past `except Exception`.

## [0.84.2.6] - 2026-08-24

### Fixed

- `/login` no longer fails with `'CancelToken' object has no attribute
  'raise_if_cancelled'`: the login dialog's abort signal is now a real
  ai-layer `CancelToken`, and Escape aborts with a "Login cancelled" reason
  instead of surfacing as a login failure.
- The tui-local `CancelToken` mirror now implements the full ai token
  contract (`reason`, `never`, `raise_if_cancelled`, reason-carrying
  `on_cancel` callbacks), fixing the same crash for loader signals passed
  into streaming APIs (e.g. the `qna` and `handoff` example extensions).
- Cancellation callbacks registered by the extension selector/input UI and
  the auth prompt race no longer raise `TypeError` when their token is
  cancelled (they ignored the abort reason argument).

## [0.84.2.5] - 2026-08-23

Runtime-layer rework: the JS promise/abort idioms kept from Pi are replaced by
tonio-native structure. No user-visible behaviour change is intended beyond
the performance and ordering items below.

### Changed

- Cancellation (Escape, `abort`) now unwinds the streaming request through
  tonio scopes instead of polling an abort flag: one streaming request is one
  scope, with teardown owned by the caller. The `CancelToken` API is unchanged.
- Work that Pi fans out concurrently (model availability, provider refresh,
  session listing, tool bodies) now runs in parallel via `tonio.map` instead
  of sequential awaits with locks.
- TUI input, timers and state mutations run on a single owner task, in order;
  the per-object locks and identity re-checks that emulated the JS event loop
  are gone. Agent events are dispatched by one ordered dispatcher.
- Long answers render much faster: finished Markdown blocks are cached across
  streaming updates (code highlighting happens once per block), terminal line
  resets are memoized, footer usage totals are recomputed only when the
  session changes, and the editor layout is cached per input state.
- The render loop computes the next frame while the previous one is still
  being written to the terminal; frame throttling is measured from the end of
  a frame rather than its start, so slow terminals no longer compound delays.

### Fixed

- Races in the agent loop, event stream, auth/models/settings stores, bash
  output accumulation and the assistant message component that the previous
  lock-based emulation did not cover.

## [0.84.2.4] - 2026-08-23

### Changed

- Updated tonio to 0.9.11 (fixes a memory leak).
- Provider-facing client identity is now pidrei's everywhere: the `User-Agent`
  sent to Anthropic-compatible (Kimi Coding) and OpenAI Codex endpoints, the
  Codex `originator` (requests and OAuth), and the xAI OAuth `referrer` all
  said `pi` before.

### Fixed

- A provider login could return before the model snapshot reflected the new
  credential when it overlapped a catalog refresh for the same provider:
  the login's own availability pass was superseded by the refresh's
  unawaited tail. Per-provider availability passes are now serialized.
- TUI: terminal output goes through a single pump task — writers enqueue
  whole sequences, the pump emits them in order using write readiness. This
  removes the busy wait on a full terminal buffer, makes write ordering
  total, paces rendering to a slow terminal, and (with cursor visibility now
  emitted as part of each frame) means overlay open/close can no longer tear
  a frame in flight.

## [0.84.2.3] - 2026-08-19

### Changed

- Updated tonio to 0.9.10.

## [0.84.2.2] - 2026-08-18

PiDrei examples and extension-API fixes on top of Pi 0.84.2.

### Added

- All portable example extensions from upstream Pi are now ported: 70 new
  examples under `examples/extensions/`, from the minimal `hello.py` up to
  custom providers, overlay games, the `subagent/` orchestrator, and remote
  tool operations over SSH. The examples README is now a themed index of all
  74 examples. The only upstream examples not ported are the four that
  depend on the Node ecosystem itself (doom-overlay, gondolin, sandbox,
  with-deps).
- The plan mode example gained the upstream README it was missing.

### Fixed

- Extension calls to `ctx.new_session()` and `ctx.fork()` raised a swallowed
  `TypeError` in interactive mode, surfacing as "Failed to create session" /
  "Failed to fork session"; both worked in print and RPC modes.
- The plan mode example injected its plan/execution prompts with stray
  leading whitespace on every line.
- The merge-and-resolve example crashed on conflicted files containing
  non-UTF-8 bytes instead of skipping them.
- `docs/extensions.md` disagreed with the code in several places: the
  custom-tool snippet used import paths that do not resolve, a two-argument
  `execute` signature (the real contract is `(tool_call_id, params, cancel,
  on_update, ctx)`), and omitted the required `label` field; the `tool_call`
  event example read `event["args"]` where the event carries `input`; the
  event catalogue listed `model_change`/`thinking_level_change` (session
  entry types) instead of the actual `model_select`/`thinking_level_select`
  events, and omitted `after_provider_response`.

## [0.84.2.1] - 2026-08-18

PiDrei dependency updates on top of Pi 0.84.2.

### Changed

- tonio updated to 0.9.7, httpunk to 0.1.4 and punkreq to 0.1.4.

### Fixed

- A race between a full availability refresh and a concurrent login/logout
  could republish stale provider availability, briefly reporting a provider
  as unconfigured right after login (or configured right after logout).

## [0.84.2.0] - 2026-08-16

Tracks [Pi 0.84.2](https://github.com/earendil-works/pi/releases/tag/v0.84.2).

### Added

- Fullscreen transcript search: `Ctrl+Shift+F` opens an incremental search
  over the transcript with match highlighting, next/previous navigation
  (`Enter`/`Ctrl+G` and `Shift+Enter`/`Ctrl+Shift+G`), and configurable
  search-match theme colors.
- Experimental strict JSON-schema constrained sampling for the default
  `read`, `bash`, `edit` and `write` tools under `PIDREI_EXPERIMENTAL=1`.
- A fullscreen exit output setting to choose between printing the final
  transcript and only a session resume hint.
- The `defaultTools` setting for configuring the initial built-in tool
  selection globally or per project.
- `--use-theme <name[/name]>` to choose an initial per-run interactive theme
  without changing saved settings.
- `expandPromptTemplates` in extension `pi.send_user_message()` options for
  explicitly dispatching commands and expanding skills and prompt templates.
- `AssistantMessage.end_turn`, preserving OpenAI Codex's terminal `end_turn`
  signal for diagnostics.
- Unbound single-line transcript scrolling actions, `tui.altScreen.lineUp`
  and `tui.altScreen.lineDown`, for fullscreen keybindings.

### Changed

- Kimi Coding requests send pi's runtime `User-Agent` header (deliberately
  still named `pi`: the backend may gate on the client identity).
- The Mistral adapter streams Chat Completions over HTTP directly instead of
  going through the SDK-shaped transport.
- OpenAI Responses deferred tool loading prefers message-anchored
  `additional_tools` where supported, keeping the tool-search and top-level
  fallbacks.
- Fullscreen rendering paints full-width layout rows directly instead of
  recompositing them on every frame.
- Model catalog refreshed from models.dev and OpenRouter.

### Fixed

- Managed-tool resolution no longer delays TUI startup: the TUI mounts
  first, download progress and warnings show inside it, and a prompt
  submitted during startup is restored instead of lost.
- Opening a model selector immediately after startup joins the in-progress
  model catalog refresh instead of cancelling and restarting it.
- GitHub Copilot login no longer triggers API rate limits while enabling
  model policies; concurrent policy updates are limited.
- Fullscreen transcript search no longer snaps back to the current match
  during manual scrolling, and fragmented SGR mouse input no longer leaks
  into the search query.
- Required LaTeX arguments starting on a new line are no longer parsed as
  empty, and LaTeX control spaces split across line endings no longer make
  complete expressions fall back to raw source.
- Fallback rendering for extension tool results collapses long output and
  honors tool expansion.
- JSON and RPC `message_update` events carry cumulative usage during
  streaming instead of dropping it.
- `pi.send_message(..., {"triggerTurn": False})` records the custom message
  instead of steering an active run.
- The `defaultTools` setting no longer drops extension and SDK custom tools
  when selecting built-in defaults.
- Custom system prompts no longer concatenate the current working directory
  with later appended prompt content.
- OpenAI Responses function and custom tool calls keep their namespaces
  during streaming, proxying and replay.
- Request buffer failures trigger automatic assistant retries.
- Built-in and custom DeepSeek API models no longer send output limits
  through an unsupported field, and DeepSeek compatibility detection works
  for base URLs whose hostname contains uppercase letters.
- Amazon Bedrock replay accepts tool arguments that contain empty object
  keys, preserving all valid nested values.
- Google Generative AI and Vertex AI responses with tool calls keep
  output-limit and provider-error stop reasons instead of reporting normal
  tool use.
- Fullscreen mouse drag selection and OSC 8 link activation work in
  terminals that report generic SGR mouse release button codes.
- Focused fullscreen overlays receive mouse wheel and viewport scroll keys
  such as PageUp and PageDown.
- Split `Alt+Enter` input over SSH is no longer misread as Escape;
  `PIDREI_TUI_ESC_TIMEOUT` tunes the lone-Escape timeout for high-latency
  terminals.
- Idle fullscreen sessions no longer repaint and clear text selection when
  the terminal loses focus.
- Fullscreen selection copy goes through the host clipboard and reports
  failure instead of claiming success when OSC 52 is unsupported.

### Not ported

- `createGatewayBindingFetch()`, upstream's Cloudflare AI Gateway
  Workers-binding adapter: a JavaScript/Workers `env.AI` shim with no Python
  consumer.
- The subagent example fixes and the `nanoid` dependency update: pidrei
  ships neither upstream's examples nor its JavaScript dependency tree.

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
