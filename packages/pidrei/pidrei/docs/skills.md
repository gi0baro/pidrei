# Skills

A skill is a folder of instructions the agent can pull in on demand. Skills keep
specialised knowledge out of the system prompt until it is actually needed: the
name and description are always visible, the body is read only when the agent
decides the skill applies. Use one when a workflow needs more context than a
[prompt template](prompt-templates.md) but no new executable integration point.

pidrei implements the [Agent Skills specification](https://agentskills.io/specification),
leniently: most violations produce warnings rather than stopping startup.

## Layout

A skill is any directory containing `SKILL.md`:

```
.pidrei/skills/
└── code-review/
    ├── SKILL.md
    └── checklist.md
```

Discovery treats a directory with `SKILL.md` as a skill root and does not
recurse further, so a skill may keep whatever supporting files it likes
alongside. Directories without `SKILL.md` are searched recursively, skipping
dot-directories, `node_modules`, and anything matched by `.gitignore`,
`.ignore` or `.fdignore`.

Direct `.md` files at the root of `~/.pidrei/agent/skills/` or
`.pidrei/skills/` are loaded as skills too, but only when they carry skill
frontmatter with a non-empty `description`. Other Markdown there —
`README.md`, `AGENTS.md` — is ignored silently, even if it fails to parse. In
the `.agents/skills/` locations it is the reverse: root `.md` files are
ignored, while `.md` files inside grouping folders load when they declare skill
frontmatter. A directory with `SKILL.md` is the portable form; prefer it.

## SKILL.md

YAML frontmatter, then the body:

```markdown
---
name: code-review
description: Review a diff for correctness, tests, and style. Use when asked to review code.
---

# Code review

1. Read the diff in full before commenting.
2. Check the tests cover the change.
3. See `checklist.md` for the full list.
```

| Field | Meaning |
|-------|---------|
| `name` | Skill name; defaults to the directory name |
| `description` | What it does and **when to use it** — this is what the agent matches on |
| `disable-model-invocation` | `true` to hide it from the system prompt, so it runs only via `/skill:name` |

The spec's `license`, `compatibility`, `metadata` and `allowed-tools` fields
are accepted; unknown fields are ignored.

The description is the single most important field. Write it as "does X; use
when Y" rather than a bare noun phrase ("Helps with PDFs" gives the agent
nothing to route on).

Names use lowercase letters, digits and hyphens, with no leading, trailing or
consecutive hyphens, up to 64 characters; descriptions are capped at 1024.
Violations warn but the skill still loads. pidrei does not require the name to
match the directory, though other Agent Skills implementations may, so matching
names stay the portable choice. A malformed `SKILL.md`, or one without a
description, is not loaded. On a name collision the first skill found wins and
a warning is shown.

## Locations

| Location | Scope |
|----------|-------|
| `~/.pidrei/agent/skills/`, `~/.agents/skills/` | User |
| `<project>/.pidrei/skills/` | Project, once the project is trusted |
| `.agents/skills/` in the working directory and its ancestors, up to the git repository root | Project, once the project is trusted |
| `skills` array in settings | Files or directories |
| `--skill <path>` | This run (repeatable) |

The ancestor search means a `.agents/skills/` at the root of a monorepo applies
in every subdirectory. Packages may ship skills; see [packages.md](packages.md).
`--no-skills` disables discovery; paths passed with `--skill` still load.

To reuse another harness's skills, list its directory in the `skills` setting
(for example `"~/.claude/skills"`).

Project skills can tell the model to run scripts or modify files. Review an
unfamiliar project's skills and their supporting files before trusting it.

## Using them

Skills appear in the system prompt as a name, description and path list. The
agent reads the body itself when it judges the skill relevant — with the `read`
tool, or with `bash` when `read` is unavailable. With neither tool the skills
section is omitted.

A model may fail to load a relevant skill; `/skill:name` forces it. Anything
after the name is appended to the skill's instructions as your request:

```text
/skill:code-review focus on error handling
```

The `enableSkillCommands` setting (toggle it in `/settings`) controls whether
skill commands show up in autocomplete; typing `/skill:name` works either way.
Run `/reload` after editing a skill in a running session.

## Paths inside a skill

Reference supporting files relative to the skill directory. The agent is told
to resolve them against the skill root rather than the working directory, so
`checklist.md` means the one next to `SKILL.md`.
