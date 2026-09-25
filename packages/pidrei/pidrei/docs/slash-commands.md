# Slash commands

Type `/` in the editor to search the commands available in the current session.
Extensions, prompt templates, and skills add their own, so that menu is the
exact reference for your setup; this page lists the built-ins.

## Models and settings

| Command | Description |
|---------|-------------|
| `/settings` | Open settings |
| `/model [provider/model]` | Select a model; an exact match switches for this session, anything else opens the picker filtered by it |
| `/thinking [level]` | Set the thinking level |
| `/scoped-models` | Choose the models Ctrl+P cycles through |
| `/login [provider]` | Add provider authentication ([providers.md](providers.md)) |
| `/logout` | Remove provider authentication |

## Sessions and context

| Command | Description |
|---------|-------------|
| `/new` | Start a new session |
| `/resume` | Switch to another saved session |
| `/name [name]` | Set the session display name; without one, show it |
| `/session` | Session info and statistics |
| `/tree` | Navigate the session tree |
| `/fork` | New session from an earlier user message |
| `/clone` | Duplicate the session at its current position |
| `/compact [instructions]` | Compact the context, optionally with custom instructions |
| `/import <path>` | Import and resume a JSONL session |

## Export and share

| Command | Description |
|---------|-------------|
| `/copy` | Copy the last assistant message |
| `/export [path]` | Export as HTML, or JSONL when `path` ends in `.jsonl` |
| `/share` | Upload the HTML export as a secret GitHub gist |

`/share` needs the GitHub CLI (`gh`) installed and logged in. It prints the gist
URL, plus a viewer link when `PIDREI_SHARE_VIEWER_URL` is set
([environment-variables.md](environment-variables.md)).

Review a session before exporting or sharing it: it can hold prompts, tool
arguments, command output, file contents, and any credentials that surfaced in
the conversation.

## Runtime and project

| Command | Description |
|---------|-------------|
| `/trust` | Save a project trust decision for future runs |
| `/reload` | Reload keybindings, extensions, skills, prompts, themes, and context files |
| `/hotkeys` | Show active shortcuts ([keybindings.md](keybindings.md)) |
| `/changelog` | Show pidrei's changelog |
| `/quit` | Quit pidrei |

## Commands added by resources

- Extensions register commands with `register_command`, including argument
  completion ([extensions.md](extensions.md)).
- Each prompt template is a command named after the template
  ([prompt-templates.md](prompt-templates.md)).
- Skills are available as `/skill:name` unless the `enableSkillCommands` setting
  is off ([skills.md](skills.md)).

Run `/reload` after adding or changing a discovered resource.
