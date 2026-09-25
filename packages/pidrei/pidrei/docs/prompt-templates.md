# Prompt templates

A prompt template is a markdown file that becomes a slash command. Running the
command expands the file into your message before it reaches the model. Use
one to reuse a prompt without adding executable behavior or a larger set of
supporting instructions.

## Writing one

Create `.pidrei/prompts/review.md`:

```markdown
---
description: Review the current diff
argument-hint: "[base-branch]"
---

Review the diff against $1 for correctness and test coverage.
Point out anything that would fail in production.
```

Then type `/review main`.

| Frontmatter | Meaning |
|-------------|---------|
| `description` | Shown in the command list; defaults to the first non-empty line |
| `argument-hint` | Shown before the description in autocomplete; use `<required>` and `[optional]` |

Frontmatter is optional — a plain markdown file works and takes its name from
the filename. Typing `/` lists templates with their hints and descriptions.
Run `/reload` after adding or changing a template in a running session.

## Arguments

| Placeholder | Expands to |
|-------------|-----------|
| `$1`, `$2`, … | Individual arguments |
| `$@` or `$ARGUMENTS` | All arguments, space-separated |
| `${1:-default}` | Argument 1, or `default` when missing or empty |
| `${@:-default}` | All arguments, or `default` when there are none |
| `${@:N}` | Arguments from position `N` (1-based) on |
| `${@:N:L}` | `L` arguments starting at position `N` |

Arguments are split on whitespace, with quotes honoured, so
`/review "feature branch"` passes one argument. A placeholder with no matching
argument expands to nothing.

An extension command with the same name wins over a template. Otherwise
extensions see the raw `/name args` text through the `input` event first, and
the template is expanded afterwards.

## Locations

| Location | Scope |
|----------|-------|
| `~/.pidrei/agent/prompts/` | User |
| `<project>/.pidrei/prompts/` | Project, once the project is trusted |
| `prompts` array in settings | Files or directories |
| `--prompt-template <path>` | This run (repeatable) |

Only `.md` files directly in a prompt directory are loaded — discovery is not
recursive. To use nested files, name them in the `prompts` setting or a package
manifest; see [packages.md](packages.md). `--no-prompt-templates` disables
discovery; paths passed with `--prompt-template` still load.

Project templates become commands only after you trust the project. Read them
before trusting an unfamiliar repository.

## Templates versus skills

They look similar and are not:

- A **template** is expanded by *you*, deterministically, when you type the
  command.
- A **skill** is read by the *agent*, when it decides the skill is relevant.

Use a template for a prompt you type often; use a [skill](skills.md) for
knowledge the agent should reach for on its own.
