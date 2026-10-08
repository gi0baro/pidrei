# Command line

`pidrei --help` prints the exact interface of your installed version, including
flags registered by loaded extensions; append `--help` to a subcommand for its
usage.

```sh
pidrei [options] [--] [@files...] [messages...]
pidrei install <source> [-l]
pidrei remove <source> [-l]         # alias: uninstall
pidrei update [source] [--extensions|--models|--all]
pidrei list
pidrei config [-l]
pidrei auth <check|print-api-key|print-bearer-token> [options]
pidrei mcp <add|remove|list|login|logout> [options]
```

## Invocation and output

```sh
pidrei
pidrei -p "Summarize this repository"
git diff | pidrei -p "Review this change"
pidrei --mode json "Inspect this repository" > events.jsonl
```

With a terminal on both stdin and stdout, pidrei opens the TUI unless `--print`
or `--mode json` asks for something else. When either stream is redirected,
print mode is used instead. [CLI integration](cli-integration.md) compares the
modes.

| Input | Behavior |
|-------|----------|
| `message` | Initial prompt; further messages are sent in order |
| `@path` | Include a text file or image in the first prompt |
| Piped stdin | Prepended to the first prompt |
| `--` | End option parsing, so a prompt can start with `-` |

`@path` resolves from the current directory, which also decides project
configuration, resource discovery, and which project the session belongs to.

| Option | Behavior |
|--------|----------|
| `-p`, `--print` | Run the prompts, write the final assistant text to stdout, exit |
| `--mode text` | Text output; still opens the TUI when both streams are terminals |
| `--mode json` | Run the prompts, write JSONL events to stdout, exit |
| `--export <input> [output]` | Export a session file to HTML and exit |

`--print` decides whether pidrei runs once and exits; `--mode` decides the output
format. Outside the TUI, stdout carries only the result — diagnostics go to
stderr.

## Models

```sh
pidrei --model sonnet:high
```

| Option | Behavior |
|--------|----------|
| `--provider <name>` | Restrict `--model` lookup to one provider; requires `--model` |
| `--model <pattern>` | Exact or fuzzy ID/name match; accepts `provider/id` and a `:<thinking>` suffix |
| `--api-key <key>` | Non-persistent key for this run; needs a model from `--model` or `--models` |
| `--thinking <level>` | `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`; overrides a `:<thinking>` suffix and is clamped to the model |
| `--models <patterns>` | Comma-separated scope for startup and Ctrl+P cycling: IDs, fuzzy matches, case-insensitive globs (`anthropic/*`), optional `:<thinking>` |
| `--list-models [search]` | List available models, optionally fuzzy-filtered, and exit |

Credentials are covered in [providers.md](providers.md), custom model entries in
[models.md](models.md).

## Sessions

```sh
pidrei --continue
```

| Option | Behavior |
|--------|----------|
| `-c`, `--continue` | Continue the most recent session for this project |
| `-r`, `--resume` | Open the session picker |
| `--session <path\|id>` | Open by path, exact ID, or ID prefix; a match in another project offers to fork it here |
| `--session-id <id>` | Open the project session with this exact ID, creating it if absent |
| `--fork <path\|id>` | Fork an existing session into a new one for this project |
| `--session-dir <dir>` | Storage and lookup directory; beats `PIDREI_CODING_AGENT_SESSION_DIR` and the `sessionDir` setting |
| `--no-session` | In-memory session, never written to disk |
| `-n`, `--name <name>` | Set the session display name |

Session IDs use letters, digits, `.`, `_`, and `-`, and start and end with a
letter or digit. `--fork` cannot be combined with `--session`, `--continue`,
`--resume`, or `--no-session`. `--session-id` cannot be combined with
`--session`, `--continue`, or `--resume`; with `--fork` it picks the new ID.

## Tools

```sh
pidrei --tools read,grep,find,ls -p "Review this project"
```

| Option | Behavior |
|--------|----------|
| `-t`, `--tools <list>` | Allowlist of built-in, extension, or custom tools: names, or patterns where `*` matches any characters. MCP tools are kept unless an entry starts with `mcp__` (see [MCP tools](#mcp-tools)). A list of only `+name` and `-name` entries is not an allowlist; it changes the default selection instead |
| `-xt`, `--exclude-tools <list>` | Disable these names or patterns after every other selection, MCP tools included |
| `-nbt`, `--no-builtin-tools` | Disable built-in tools, keep extension and custom ones |
| `-nt`, `--no-tools` | Start with every tool disabled, MCP tools included |

`read`, `bash`, `edit`, and `write` are enabled by default; the `defaultTools`
setting changes that. `--tools` with plain names replaces the whole selection,
so name every tool you want; `defaultTools` also accepts `+name` and `-name` to
change the defaults instead (for example `["-bash", "+grep"]`). A project list
with only `+name` and `-name` entries changes the user's selection; a list with
a plain name replaces it.

Like `defaultTools`, `--tools` also accepts a list of only `+name` and `-name`
entries, which adds tools to or removes them from the default selection:
`pidrei --tools +codemode,-write` keeps the other default tools, enables
`codemode`, and disables `write`. These entries take exact tool names, not `*`
patterns; use `--exclude-tools` to disable tools by pattern. Plain names and
`+name`/`-name` entries cannot be mixed.

`/reload` enables tools newly added to `defaultTools`. It does not disable
tools removed from it or re-enable unchanged tools you turned off; a tool
removed with `--tools -name` stays removed. `--tools` with plain names,
`--no-tools`, and `--no-builtin-tools` override `defaultTools`, also on reload.

<a id="mcp-tools"></a>

`--tools` selects the tools declared to the model. It does not remove MCP
tools, whose reach is set by their [exposure](mcp.md#control-tool-exposure):
`pidrei --tools read,codemode` keeps every MCP tool callable from codemode
scripts. An MCP tool that no entry names or matches is never declared directly,
whatever its exposure; only `tool_search`, if listed, can load it. Once an
entry starts with `mcp__`, `--tools` filters MCP tools too, so this keeps only
the tools of the `radius` server:

```sh
pidrei --tools read,bash,codemode,'mcp__radius__*'
```

The MCP resource tools (`list_mcp_resources`, `list_mcp_resource_templates`,
`read_mcp_resource`) count as MCP tools. To remove MCP tools, use
`--exclude-tools 'mcp__*'` or [`--no-mcp`](#resources).

| Built-in | Purpose |
|----------|---------|
| `read` | Read text files and supported images |
| `bash` | Run shell commands |
| `edit` | Exact text replacements in an existing file |
| `write` | Create or overwrite a file |
| `grep` | Search file contents (needs `ripgrep` on `PATH`) |
| `find` | Find paths by glob (needs `fd` on `PATH`) |
| `ls` | List directory contents |

Built-in extensions add two more tools. They are off by default; the MCP
extension turns them on when an MCP server needs them (see
[MCP](mcp.md#control-tool-exposure)). To enable them yourself, name them in
`--tools` or `defaultTools`.

| Built-in extension | Purpose |
|--------------------|---------|
| `codemode` | Run Python that calls the other tools, for example in parallel with `all_settled`; only the script's output reaches the model |
| `tool_search` | Search tools that are not declared to the model (`codemode` and `deferred` exposure, such as MCP tools) and declare the matches for the next call |

### Enable codemode

To turn on `codemode` for every session, add it to the default tools in
`~/.pidrei/agent/settings.json` or a project's `.pidrei/settings.json`:

```json
{
  "defaultTools": ["+codemode"]
}
```

This keeps `read`, `bash`, `edit`, and `write` and adds `codemode`. For one
invocation, add it with `--tools`:

```sh
pidrei --tools +codemode
```

Scripts can run several tool calls in parallel, filter large output before it
reaches the model, call classifier models such as TypeSafe's Jev through
`models.classify()`, and generate images through `models.generate_images()`
(see [Classifier models](models.md#classifier-models)).

### How codemode works

Scripts are Python, not JavaScript as in pi. They run in a sandboxed
interpreter, are type-checked before they run, and reach the other tools
through `await tools.<name>(arg=value)`. [Codemode](codemode.md) describes the
script API, how tools are listed and found, the `store()` and `models`
globals, the settings, and the limits.

### Tool search

`tool_search` is off by default; enable it with
`"defaultTools": ["+tool_search"]` or `--tools`. It uses the same ranking as
`search_tools()` over tools that are not declared yet and declares the matches
for the next model call. Loaded tools are recorded in the session like other
tool changes, so they stay declared on that branch.

## Resources

```sh
pidrei -e ./review.py
```

| Option | Behavior |
|--------|----------|
| `-e`, `--extension <path>` | Load an extension file or directory, or a built-in extension such as `builtin:mcp`; repeatable |
| `-ne`, `--no-extensions` | Skip discovered, configured, and built-in extensions; `-e` still loads, so `pidrei -ne -e builtin:mcp` keeps only the built-in MCP support |
| `--no-mcp` | Disable the built-in MCP support for this run: no servers connect, and there are no MCP tools or `/mcp`. An extension that replaces the built-in MCP support is not affected |
| `--skill <path>` | Load a skill file or directory; repeatable |
| `-ns`, `--no-skills` | Skip discovered and configured skills; `--skill` still loads |
| `--prompt-template <path>` | Load a prompt template file or directory; repeatable |
| `-np`, `--no-prompt-templates` | Skip discovered and configured templates; `--prompt-template` still loads |
| `--theme <path>` | Load a theme file or directory; repeatable |
| `--use-theme <name[/name]>` | Initial TUI theme for this run |
| `--no-themes` | Skip discovered and configured themes; `--theme` still loads |
| `-nc`, `--no-context-files` | Skip `AGENTS.md` / `CLAUDE.md` discovery |

These apply to the current process only; relative paths resolve from the current
directory. See [extensions.md](extensions.md), [skills.md](skills.md),
[prompt-templates.md](prompt-templates.md), and [themes.md](themes.md).

## Prompts and process

| Option | Behavior |
|--------|----------|
| `--system-prompt <text\|path>` | Replace the default system prompt with text or an existing file's contents |
| `--append-system-prompt <text\|path>` | Append text or a file to the system prompt; repeatable |
| `--tui-mode <mode>` | `fullscreen` (default) or `regular` |
| `--verbose` | Verbose startup, overriding `quietStartup` |
| `-a`, `--approve` | Trust project-local configuration and resources for this process |
| `-na`, `--no-approve` | Ignore trust-gated project-local configuration and resources |
| `--offline` | No startup network activity, catalog refreshes included; same as `PIDREI_OFFLINE=1` |
| `-h`, `--help` | Help, including extension flags, then exit |
| `-v`, `--version` | Print the version and exit |

Extensions can register further long options (`register_flag`, see
[extensions.md](extensions.md)); unknown short options are rejected. Project
trust and config locations are in [configuration.md](configuration.md); process
variables in [environment-variables.md](environment-variables.md).

## Package commands

| Task | Command |
|------|---------|
| Install a package | `pidrei install <source>` |
| List configured packages | `pidrei list` |
| Remove a package and its settings entry | `pidrei remove <source>` |
| Enable or disable package resources | `pidrei config` |
| Update all installed packages | `pidrei update` |
| Update one package | `pidrei update <source>` |
| Refresh model catalogs | `pidrei update --models` |
| Packages and model catalogs | `pidrei update --all` |

Sources are git URLs or local paths — see [packages.md](packages.md).
`-l`/`--local` on `install`, `remove`, and `config` uses project settings
(`.pidrei/settings.json`) instead of global ones. `-a`/`-na` apply project trust
for one command, as above. `update --extension <source>` is the long form of
`update <source>`.

pidrei never updates itself: `update --self`, `update self`, and `update --force`
are refused with the reinstall command to run instead.

## Credential commands

```sh
pidrei auth check --provider openai --json
```

Each command needs `--provider <provider>`, `--model <model>`, or both.

| Command | Output |
|---------|--------|
| `pidrei auth check` | `ready`, `not_ready`, or `invalid`; exit status `0`, `1`, or `2` |
| `pidrei auth print-api-key` | The resolved API key |
| `pidrei auth print-bearer-token` | A resolved OAuth bearer token |

| Option | Applies to | Effect |
|--------|------------|--------|
| `--json` | `check` | Structured JSON result |
| `--credentials` | `check` | Also emit the credential when ready |
| `--no-refresh` | `check` | Don't refresh expired OAuth credentials (refresh is the default) |
| `--min-expiry <duration>` | `print-bearer-token` | Require remaining lifetime, e.g. `30m` (`ms`, `s`, `m`, `h`) |

The printing commands write secrets to stdout.

## MCP commands

These commands work outside a session, so agents can run them through `bash`.
See [MCP Servers](mcp.md).

| Command | Behavior |
|---------|----------|
| `pidrei mcp add <server> [options] -- <command> [args...]` | Add or replace a stdio server in `mcp.json`; `--env KEY=VALUE` (repeatable) and `--cwd <dir>` set its environment and working directory. Arguments after the command are passed to it |
| `pidrei mcp add <server> [options] --url <url>` | Add or replace a streamable HTTP server; `--header KEY=VALUE` (repeatable), `--bearer-token-env-var <NAME>` (sends `Authorization: Bearer ${NAME}`), `--oauth-client-id`, `--oauth-client-secret`, `--oauth-callback-port`, and `--oauth-client-name` configure authentication |
| `pidrei mcp remove <server>` | Remove a server from `mcp.json`; stored OAuth credentials are kept |
| `pidrei mcp list [--json]` | Connect to every enabled server and print its state, tools, and errors; exit with `1` when a config entry is invalid or an enabled server is not connected |
| `pidrei mcp login <server> [--timeout <seconds>]` | Sign in to an OAuth server: open the authorization page and wait for the browser (default 300 seconds); a terminal also accepts the pasted redirect URL |
| `pidrei mcp logout <server>` | Delete the stored OAuth credentials of a server |

`add` and `remove` change `~/.pidrei/agent/mcp.json`, or `.pidrei/mcp.json` in
the current directory with `--local` (`-l`). `add` also takes
`--exposure <mode>` (see [Exposure](mcp.md#control-tool-exposure)) and
`--description <text>` and does not connect; run `pidrei mcp list` to check
the server.

Project `.pidrei/mcp.json` files are only read for projects that are already
trusted.
