# Packages

A package bundles extensions, skills, prompt templates and themes so they can be
installed as a unit. Use one when a customization should be shared through git,
or when several resources belong together.

Packages run extension code in the pidrei process and can ship skills that tell
the model to run programs. Review a third-party package's source before
installing it, and a project's package declarations before trusting the
project.

## Sources

pidrei installs packages from **git** or a **local path**:

```jsonc
// ~/.pidrei/agent/settings.json
{
  "packages": [
    "https://github.com/someone/pidrei-goodies",
    "git:github.com/someone/other@v1.2.0",
    "git:git@github.com:private/pack.git",
    "./local/checkout"
  ]
}
```

| Source | Behavior |
|--------|----------|
| `git:host/path`, `git:git@host:path`, `git:user/repo` (GitHub) | Cloned under the agent directory (project installs under `.pidrei/`) |
| `https://`, `ssh://` URL | Treated as a git source |
| Local path | Loaded in place, not copied |

Git sources may pin a ref with `@<ref>`. A pinned tag or commit is never moved:
updates reconcile the checkout to it. The scp-like `git@host:path` form needs
the `git:` prefix — without it the value is read as a local path. SSH forms
share package identity with their HTTPS equivalent, so the same repository is
one package however it is addressed.

Relative local paths resolve from the settings file that contains them. A path
to a `.py` file loads one extension; a directory follows the layout rules
below.

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

Paths are relative to the package root, and entries may use glob patterns.
Globs discover visible paths in lexical order: list dot-prefixed paths
directly, and list the resource root directly when a glob would have to
continue through a symlink.

pidrei does not install a package's Python dependencies. Extensions run in
pidrei's own interpreter, so they can import `pidrei`, `pidrei_ai`,
`pidrei_tui` and `tonio` directly; anything else must be vendored in the
package or installed by the user, and the package should say so.

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
- `[]` loads none of that type; omitting the key loads everything.
- `autoload: false` disables everything the package ships *except* what a
  pattern names, so nothing loads unless you ask for it by name.

Filters narrow what the package declares; they never expose resources its
manifest leaves out.

The same package may appear in user and project settings. The project entry
normally replaces the user one; with `autoload: false` it instead layers over
the user entry as a filtering delta. Git packages are identified by host and
repository path (ignoring the ref), local ones by resolved path, so equivalent
declarations never load a package twice.

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

`install` writes to `~/.pidrei/agent/settings.json`; `-l` targets the
project's `.pidrei/settings.json` instead. Project declarations are read, and
project packages installed and loaded, only once the project is trusted, so
`--approve` / `--no-approve` decide that for a single command. Every subcommand
takes `--help`.

To try a package for one run without adding it to settings, pass it to
`-e`/`--extension`:

```bash
pidrei -e git:github.com/someone/pack
```

**pidrei does not update itself.** pi's `update` also reinstalls pi through
whichever package manager installed it; pidrei installs from git or Homebrew,
where updating means re-running the install command with a new version — so
`pidrei update --self` tells you the command rather than guessing at your
installation.

`/reload` re-reads package resources without restarting.

## Offline

With `PIDREI_OFFLINE=1` (or `--offline`) pidrei never contacts a git remote:
already-installed packages load, missing ones are reported and skipped.
