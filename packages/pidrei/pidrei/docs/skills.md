# Skills

A skill is a folder of instructions the agent can pull in on demand. Skills keep
specialised knowledge out of the system prompt until it is actually needed: the
name and description are always visible, the body is read only when the agent
decides the skill applies.

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
alongside. Directories without `SKILL.md` are searched recursively.

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
| `disable-model-invocation` | `true` to make it available only via `/skill`, never picked automatically |

The description is the single most important field. Write it as "does X; use
when Y" rather than a bare noun phrase.

## Locations

| Location | Scope |
|----------|-------|
| `~/.pidrei/agent/skills/` | User |
| `<project>/.pidrei/skills/` | Project |
| `--skill <path>` | This run |

Ancestor directories of the working directory are also searched, so a skill at
the root of a monorepo applies in every subdirectory. Packages may ship skills;
see [packages.md](packages.md). `--no-skills` disables all of them.

## Using them

Skills appear in the system prompt as a name and description list. The agent
reads the body itself, with the read tool, when it judges the skill relevant —
which is why the `read` tool is required for skills to work at all, and why the
skills section is omitted when it is unavailable.

Invoke one explicitly with `/skill <name>`. `/skills` lists what loaded and
where each came from.

## Paths inside a skill

Reference supporting files relative to the skill directory. The agent is told
to resolve them against the skill root rather than the working directory, so
`checklist.md` means the one next to `SKILL.md`.
