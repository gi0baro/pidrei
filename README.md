# pidrei

A rewrite of the [pi coding agent](https://github.com/badlogic/pi-mono) in Python,
built on [tonio](https://github.com/gi0baro/tonio) (free-threaded CPython 3.14+)
and [punkreq](https://github.com/gi0baro/punkreq).

Goals, in order:

1. Mirror pi's observable behavior 1:1 — pi's test suites are the spec.
2. Validate the tonio ecosystem (tonio, httpunk, punkreq) on a complete,
   demanding, genuinely multi-threaded project.

See [FEASIBILITY.md](FEASIBILITY.md) for the analysis and decisions,
[PLAN.md](PLAN.md) for the phased implementation plan.

## Status

Phase 0 (foundations): core primitives (`EventStream`, `CancelToken`, SSE
decoding) and project scaffolding.

The repository is a uv workspace mirroring pi's monorepo; packages live under
`packages/` (`pidrei-ai` is the first) and join the workspace as their
implementation phase begins.

## Development

Requires free-threaded CPython 3.14+ and [uv](https://docs.astral.sh/uv/).

```
make          # format + lint + test
make test
```
