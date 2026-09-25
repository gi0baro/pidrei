# Configuration

pidrei has user-level and project configuration. User-level configuration lives
in the agent directory, `~/.pidrei/agent` by default. Project configuration
lives in `.pidrei/` under the working directory and loads only once the project
is trusted. The one exception is `sessionDir`, which pidrei reads from project
settings before resolving trust so it can find sessions.

In interactive mode, `/settings` changes common preferences. For anything else,
ask pidrei to update the configuration or edit the files directly, then run
`/reload` to pick up changes to settings, keybindings, instructions, or
resources.

## Agent directory

The agent directory is shown as `<agent-dir>` below. Move it with the
`PIDREI_CODING_AGENT_DIR` environment variable, or with `agent_dir` in
`CreateAgentSessionOptions` when embedding pidrei ([sdk.md](sdk.md)).

| Path | Responsibility |
|---|---|
| `<agent-dir>/settings.json` | User settings: preferences, defaults, resource paths, and [package](packages.md) declarations |
| `<agent-dir>/keybindings.json` | Custom [keybindings](keybindings.md), a flat map of action id to key(s) |
| `<agent-dir>/models.json` | [Custom providers, models, and overrides](models.md) |
| `<agent-dir>/auth.json` | Saved API keys and OAuth credentials ([providers.md](providers.md)) |
| `<agent-dir>/AGENTS.override.md`, `AGENTS.md`, `AGENTS.MD`, `CLAUDE.md`, or `CLAUDE.MD` | User instructions applied in every working directory |
| `<agent-dir>/SYSTEM.md` | Replaces pidrei's default system prompt |
| `<agent-dir>/APPEND_SYSTEM.md` | Adds instructions to the system prompt |
| `<agent-dir>/extensions/` | User [extensions](extensions.md) |
| `<agent-dir>/skills/` | User [skills](skills.md) and their supporting files |
| `<agent-dir>/prompts/` | User [prompt templates](prompt-templates.md), exposed as slash commands |
| `<agent-dir>/themes/` | User [themes](themes.md) |
| `<agent-dir>/sessions/` | Default session storage, one subdirectory per working directory |
| `<agent-dir>/bin/` | Optional `fd`/`rg` binaries, searched before `PATH` |

`sessions/` is only the default: `--session-dir`,
`PIDREI_CODING_AGENT_SESSION_DIR`, and the `sessionDir` setting override it, in
that order.

## Project `.pidrei` directory

| Path | Responsibility |
|---|---|
| `.pidrei/settings.json` | Project settings, resource paths, and package declarations |
| `.pidrei/SYSTEM.md` | Replaces the system prompt for this project |
| `.pidrei/APPEND_SYSTEM.md` | Adds project-specific instructions to the system prompt |
| `.pidrei/extensions/` | Project extensions |
| `.pidrei/skills/` | Project skills and their supporting files |
| `.pidrei/prompts/` | Project prompt templates |
| `.pidrei/themes/` | Project themes |

Project settings are deep-merged over user settings: project values win, and
nested objects merge key by key.

For `SYSTEM.md` and `APPEND_SYSTEM.md`, the trusted project file takes
precedence over the agent-directory file of the same name. The two are not
combined. The `--system-prompt` and `--append-system-prompt` flags take the
place of the discovered `SYSTEM.md` and `APPEND_SYSTEM.md` respectively.

## Project trust

Any of the entries above, or a `.agents/skills` directory in the working
directory or one of its parents, makes the project need trust. Trusting it
lets pidrei load `.pidrei` settings and resources, install missing project
packages, and run project extensions — which is running their code.

Interactive mode asks the first time. `/trust` saves the decision for future
sessions, `--approve` (`-a`) and `--no-approve` (`-na`) decide for one run, and
the user-level `defaultProjectTrust` setting (`"ask"`, `"always"`, or
`"never"`) sets the default.

## Context files

Context files are separate from `.pidrei` configuration. pidrei loads them from
the agent directory, then from the working directory and each of its parents,
outermost first. A context file applies whenever pidrei runs in its directory
or anywhere below it.

Each directory contributes at most one file, the first of `AGENTS.override.md`,
`AGENTS.md`, `AGENTS.MD`, `CLAUDE.md`, `CLAUDE.MD` that exists. So
`AGENTS.override.md` replaces `AGENTS.md` or `CLAUDE.md` only in its own
directory; it does not suppress context files anywhere else.

Context-file discovery does not require project trust. `--no-context-files`
(`-nc`) disables it.
