# PiDrei

A Python port of the [Pi coding agent](https://github.com/earendil-works/pi),
built on top of the [TonIO runtime](https://github.com/gi0baro/tonio).

> **Note:** PiDrei is built with substantial help from LLMs, under human supervision.

## Rationale

The main reasons behind PiDrei existence are:

- *for fun* :)
- the desire to dig into and understand Pi internals
- exercising TonIO against a large project
- stress-test free-threaded CPython
- give [me](https://github.com/gi0baro) a chance to join the *Austrian AI mafia*

## Status

Alpha, and the honest kind. PiDrei is a port in progress, validated the only way
a port sensibly can be: by porting Pi's own test suites module by module and
keeping them green (2,958 mirrored cases so far).

Working today: the agent loop and its tools, the TUI and interactive mode, the
headless CLI and RPC server, all 37 of Pi's providers, image generation, OAuth
login for the ones that need it, and extensions — loading, the full hook bus,
and packages installed from git.

Not yet: Pi's bundled llama.cpp extension. PiDrei also never updates itself —
`pidrei update` handles packages and model catalogs, and tells you which install
command to re-run for PiDrei.

## Installation

If you're brave enough to actually use this, despite the alpha state and all the
premises above, you will need a free-threaded CPython 3.14. `uv` fetches one for
you, so this is the whole procedure:

```
uv tool install -p 3.14t git+https://github.com/gi0baro/pidrei@v0.82.0.0
```

Or, if you would rather Homebrew did the honours:

```
brew install gi0baro/tap/pidrei
```

Wheels and source tarballs are attached to every
[GitHub release](https://github.com/gi0baro/pidrei/releases), if you prefer to
install from those.

## Differences with Pi

Deliberate, and not going away:

- **POSIX only.** TonIO is Unix-only, so Pi's Windows code paths are not ported
  and never will be. If you need this on Windows, WSL works — and so does Pi.
- **Free-threaded CPython 3.14+ is mandatory**, not an option. Running a real
  workload on it was half the point.
- **Extensions are Python**, loaded through `importlib`, rather than TypeScript
  modules loaded through jiti. The hook bus semantics mirror Pi's 1:1; the
  extension artifacts themselves obviously cannot.
- **Its own config**: `~/.pidrei/` and `PIDREI_*` environment variables. Session
  files keep Pi's JSONL format, so transcripts stay interchangeable.
- **Syntax highlighting is Pygments**, not highlight.js — close enough to look
  right, not close enough to diff.
- **No `radius`**, provider or presence integration. A Pi-specific service that
  does nothing without Pi's own credentials — so it does nothing here, either.
- **Nothing phones home.** Pi pings `pi.dev` on install and asks it for the
  latest version; PiDrei does neither. Model catalogs are still fetched, because
  that is where the models are. And where Pi tells OpenRouter which app is
  calling, PiDrei takes the blame itself.

Everything else is meant to be behaviourally identical, down to the strings the
model sees. Where it isn't, that's a bug, not a design decision.

## Versioning

PiDrei mirrors Pi releases, by expanding the original version with an additional
group: the first 3 groups of the versioning scheme reflect the upstream Pi
version, while the latter is for PiDrei specific patches.

So `0.82.0.3` is the fourth PiDrei release tracking Pi 0.82.0 — the first being
`0.82.0.0`, because we are programmers. A bump in the first three groups means
upstream moved; a bump in the last one means only PiDrei did.

Which also means PiDrei trails Pi, usually by a release. Porting takes as long as
it takes; the number is at least honest about which Pi you got.

## Credits & License

PiDrei is MIT licensed, which is convenient, because so is Pi.

It is worth being blunt about what that means here: this is a translation, not an
invention. **Pi** is © 2025 [Mario Zechner](https://github.com/badlogic) and
[Earendil Works](https://github.com/earendil-works), and it is the design, the
architecture and very nearly every behavioural decision in this repository.

If you like how PiDrei works, that's their doing. If you don't like how it runs,
that one's mine.

Model and provider metadata is generated from [models.dev](https://models.dev),
the same catalog Pi uses.
