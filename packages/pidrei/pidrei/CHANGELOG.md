# Changelog

Versions are `<pi version>.<pidrei build>`. The first three segments name the
Pi release this port tracks; the fourth counts PiDrei's own releases against it,
so `0.82.0.1` would be a PiDrei fix on top of the same Pi 0.82.0.

## [Unreleased]

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
