# Prompt templates

A prompt template is a markdown file that becomes a slash command. Running the
command expands the file into your message before it reaches the model.

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
| `argument-hint` | Argument hint shown alongside the command |

Frontmatter is optional — a plain markdown file works and takes its name from
the filename.

## Arguments

| Placeholder | Expands to |
|-------------|-----------|
| `$1`, `$2`, … | Individual arguments |
| `$@` | All arguments, space-separated |

Arguments are split on whitespace, with quotes honoured, so
`/review "feature branch"` passes one argument. A placeholder with no matching
argument expands to nothing.

## Locations

| Location | Scope |
|----------|-------|
| `~/.pidrei/agent/prompts/` | User |
| `<project>/.pidrei/prompts/` | Project |
| `--prompt-template <path>` | This run |

Only `.md` files directly in the directory are loaded — discovery is not
recursive. Packages may ship templates; see [packages.md](packages.md).
`--no-prompt-templates` disables all of them.

`/prompts` lists what loaded and where each came from.

## Templates versus skills

They look similar and are not:

- A **template** is expanded by *you*, deterministically, when you type the
  command.
- A **skill** is read by the *agent*, when it decides the skill is relevant.

Use a template for a prompt you type often; use a [skill](skills.md) for
knowledge the agent should reach for on its own.
