# Environment variables

## Configuration

| Variable | Effect |
|----------|--------|
| `PIDREI_CODING_AGENT_DIR` | Config directory (default `~/.pidrei/agent`) |
| `PIDREI_CODING_AGENT_SESSION_DIR` | Session storage (overridden by `--session-dir`) |
| `PIDREI_CONFIG_DIR` | Base config directory |
| `PIDREI_PACKAGE_DIR` | Override where pidrei looks for its own shipped files (Nix/Guix store paths) |
| `PIDREI_OFFLINE` | Disable every startup network operation when `1`/`true`/`yes` |
| `PIDREI_SKIP_VERSION_CHECK` | Skip the update check only |
| `PIDREI_PROVIDER_ATTRIBUTION` | Force provider attribution headers on or off |
| `PIDREI_SHARE_VIEWER_URL` | Base URL of a session viewer for `/share` (default: none) |
| `PIDREI_PROVIDER` / `PIDREI_MODEL` | Default provider and model |
| `PIDREI_REASONING_LEVEL` | Default thinking level |
| `PIDREI_OAUTH_CALLBACK_HOST` | Host the OAuth callback server binds |
| `PIDREI_THREADS` | Runtime worker threads (default: CPU count clamped to 2–8) |
| `PIDREI_BLOCKING_THREADS` | Blocking thread pool cap (default: 8 per worker) |

Provider credentials are listed in [providers.md](providers.md); `pidrei --help`
prints them all.

## Available to bash tools

pidrei exports these into the environment of every command the `bash` tool
runs, so scripts and hooks can find the session they belong to:

| Variable | Value |
|----------|-------|
| `PIDREI_CODING_AGENT` | Set when running under pidrei |
| `PIDREI_SESSION_ID` | Current session id |
| `PIDREI_SESSION_FILE` | Path to the session JSONL file |

## External tools

The `find` and `grep` tools shell out to `fd` and `ripgrep`, which must be
installed and on `PATH` — pidrei never downloads binaries. pidrei also looks in
its own `bin` directory under the agent dir first, if you put them there.

## Terminal and display

| Variable | Effect |
|----------|--------|
| `PIDREI_HARDWARE_CURSOR` | Use the terminal's own cursor |
| `PIDREI_CLEAR_ON_SHRINK` | Clear the screen when the terminal shrinks |
| `PIDREI_CACHE_RETENTION` | Provider cache retention behaviour |

## Debugging

| Variable | Effect |
|----------|--------|
| `PIDREI_TUI_DEBUG` | TUI diagnostics |
| `PIDREI_TUI_WRITE_LOG` | Log every terminal write |
| `PIDREI_DEBUG_REDRAW` | Highlight redraws |
| `PIDREI_INPUT_EVENT_LOG` | Log decoded input events |
| `PIDREI_TIMING` | Print startup phase timings |
| `PIDREI_STARTUP_BENCHMARK` | Startup benchmark mode |
| `PIDREI_EXPERIMENTAL` | Enable experimental features |

Debug output goes to `~/.pidrei/agent/pidrei-debug.log`.
