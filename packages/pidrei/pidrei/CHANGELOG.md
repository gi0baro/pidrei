# Changelog

Versions are `<pi version>.<pidrei build>`. The first three segments name the
Pi release this port tracks; the fourth counts PiDrei's own releases against it,
so `0.82.0.1` would be a PiDrei fix on top of the same Pi 0.82.0.

## [Unreleased]

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

## [0.82.0.0] - 2026-07-27

First release, tracking
[Pi 0.82.0](https://github.com/earendil-works/pi/releases/tag/v0.82.0).

Itemising what changed would mean itemising the whole project, so: all of it.
[The README](README.md) is the honest summary — what works, what doesn't yet,
and where PiDrei differs from Pi on purpose. Later entries will have the decency
to be shorter.
