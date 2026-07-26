# pidrei documentation

pidrei is a terminal coding agent: a small core extended through Python
extensions, skills, prompt templates, themes, and packages. It is a port of
[pi](https://github.com/earendil-works/pi) to free-threaded CPython on the
[tonio](https://github.com/gi0baro/tonio) runtime, and behaves like pi except
where this documentation says otherwise.

## Start here

- [Providers](providers.md) — subscription and API-key setup for the built-in providers.
- [Custom models](models.md) — add model entries in `models.json`.
- [Environment variables](environment-variables.md) — process configuration, and what bash tools see.
- [Keybindings](keybindings.md) — default shortcuts and how to rebind them.

## Customization

- [Extensions](extensions.md) — Python modules adding tools, commands, events, and UI.
- [Skills](skills.md) — reusable on-demand capabilities.
- [Prompt templates](prompt-templates.md) — reusable prompts behind slash commands.
- [Themes](themes.md) — built-in and custom terminal themes.
- [Packages](packages.md) — bundle and share extensions, skills, prompts, and themes.
- [Custom providers](custom-provider.md) — register a provider from an extension.

## Programmatic use

- [Library use](sdk.md) — embed pidrei's packages in your own Python program.
- [TUI components](tui.md) — build terminal UI for extensions.

## Differences from pi

pidrei aims at behavioural parity with pi, including the strings the model
sees. The deliberate exceptions:

- **POSIX only.** tonio is Unix-only, so pi's Windows paths are not ported.
- **Free-threaded CPython 3.14+ is required**, not optional.
- **Extensions are Python modules**, not TypeScript. The hook bus mirrors pi's;
  the extension artifacts cannot. See [extensions.md](extensions.md).
- **Packages install from git or a local path.** pi also supports `npm:`
  sources; pidrei refuses them with a clear error. See [packages.md](packages.md).
- **Its own config**: `~/.pidrei/` and `PIDREI_*` variables. Session files keep
  pi's JSONL format, so transcripts stay interchangeable.
- **Syntax highlighting is Pygments**, not highlight.js.
- **No `radius`** provider or presence integration — a pi-specific service.
- **Nothing phones home.** No install ping; the update check reads pidrei's own
  GitHub releases. Model catalogs still refresh from the public catalog and
  models.dev.

Anything else that differs from pi is a bug.
