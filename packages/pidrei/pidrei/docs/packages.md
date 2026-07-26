# Packages

A package bundles extensions, skills, prompt templates and themes so they can be
installed as a unit.

## Sources

pidrei installs packages from **git** or a **local path**:

```jsonc
// ~/.pidrei/agent/settings.json
{
  "packages": [
    "https://github.com/someone/pidrei-goodies",
    "git:github.com/someone/other#v1.2.0",
    "git@github.com:private/pack.git",
    "./local/checkout"
  ]
}
```

Git sources may pin a ref with `#<ref>`. Pinned packages are never updated
automatically. SSH forms work and share package identity with their HTTPS
equivalent, so the same repository is one package however it is addressed.

pi additionally supports `npm:` sources. pidrei does not, and says so rather
than silently resolving to nothing:

```
npm package sources are not supported: npm:foo. Use a git source
(git:… or an https:// URL) or a local path.
```

Checkouts live under the agent directory, and pidrei refuses to write outside
its own install roots.

## Layout

A package is a directory. Anything found in the conventional locations loads
automatically:

```
my-package/
├── pyproject.toml          # optional manifest
├── extensions/
├── skills/
├── prompts/
└── themes/
```

To place resources elsewhere, declare them:

```toml
[tool.pidrei]
extensions = ["src/my_package/agent_ext.py"]
skills = ["resources/skills/"]
prompts = ["resources/prompts/"]
themes = ["resources/themes/"]
```

Declared entries replace auto-discovery for that resource type. A directory
containing `__init__.py` is loaded as a single package extension.

## Filtering

Take part of a package with a filter:

```jsonc
{
  "packages": [
    { "source": "https://github.com/someone/pack", "extensions": ["git-*", "!git-danger"] },
    { "source": "./local", "autoload": false, "skills": ["review"] }
  ]
}
```

- Patterns match resource filenames; `*` is a wildcard.
- `!name` excludes; `+name` and `-name` add to or remove from what is already
  selected.
- `autoload: false` disables everything the package ships *except* what a
  pattern names, so nothing loads unless you ask for it by name.

## Commands

```bash
pidrei install <source> [-l]     # add a source and install it
pidrei remove <source> [-l]      # remove it again (alias: uninstall)
pidrei list                      # what is configured, and where it lives
pidrei update [source]           # update installed packages
pidrei update --models           # refresh model catalogs
pidrei update --all              # both
pidrei config [-l]               # enable/disable individual resources (TUI)
```

`-l` targets project settings instead of global; it needs project trust, so
`--approve` / `--no-approve` decide that for a single command. Every subcommand
takes `--help`.

**pidrei does not update itself.** pi's `update` also reinstalls pi through
whichever package manager installed it; pidrei installs from git or Homebrew,
where updating means re-running the install command with a new version — so
`pidrei update --self` tells you the command rather than guessing at your
installation.

Package resources also appear in `/extensions`, `/skills`, `/prompts` and
`/themes` with the package as their source. `/reload` re-reads them without
restarting.

## Offline

With `PIDREI_OFFLINE=1` (or `--offline`) pidrei never contacts a git remote:
already-installed packages load, missing ones are reported and skipped.
