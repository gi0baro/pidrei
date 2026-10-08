# Environment variables

## Configuration

| Variable | Effect |
|----------|--------|
| `PIDREI_CODING_AGENT_DIR` | Config directory (default `~/.pidrei/agent`) |
| `PIDREI_CODING_AGENT_SESSION_DIR` | Session storage (overridden by `--session-dir`) |
| `PIDREI_CONFIG_DIR` | Base config directory |
| `PIDREI_PACKAGE_DIR` | Override where pidrei looks for its own shipped files (Nix/Guix store paths) |
| `PIDREI_OFFLINE` | Disable automatic network activity — model catalog refreshes, the version check, package update checks — when `1`/`true`/`yes` |
| `PIDREI_SKIP_VERSION_CHECK` | Skip the update check only |
| `PIDREI_PROVIDER_ATTRIBUTION` | Force provider attribution headers on or off |
| `PIDREI_SHARE_VIEWER_URL` | Base URL of a session viewer for `/share` (default: none) |
| `PIDREI_PROVIDER` / `PIDREI_MODEL` | Default provider and model |
| `PIDREI_REASONING_LEVEL` | Default thinking level |
| `PIDREI_OAUTH_CALLBACK_HOST` | Host the OAuth callback server binds |
| `PIDREI_HYPERLINKS` | Override OSC 8 hyperlink detection with `1`, `0`, or `auto` |
| `PIDREI_PROGRAM_STATUS` | Override OSC 7501 program status reporting: `1` always reports, `0` never reports; otherwise pidrei reports only after the terminal confirms support. |
| `PIDREI_IMAGE_PROTOCOL` | Override inline image detection with `kitty`, `iterm2`, `none`, or `auto` |
| `PIDREI_TRUE_COLOR` | Override truecolor detection with `1`, `0`, or `auto` |
| `PIDREI_TUI_ESC_TIMEOUT` | How long to wait after a lone ESC before treating it as Escape, in milliseconds; defaults to `100` over SSH and `10` otherwise. Increase if Alt-key input is misread as Escape |
| `PIDREI_THREADS` | Runtime worker threads (default: CPU count clamped to 2–8) |
| `PIDREI_BLOCKING_THREADS` | Blocking thread pool cap (default: 8 per worker) |

Provider credentials and provider-specific configuration are listed in
[providers.md](providers.md#api-keys); `pidrei --help` prints them all.

## Available to bash tools

pidrei exports these into the environment of every command the `bash` tool
runs, so scripts and hooks can find the session they belong to:

| Variable | Value |
|----------|-------|
| `AI_AGENT` | `pidrei` — generic marker identifying the launching agent |
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

## Program status

pidrei reports its state with the [Program Status Protocol (OSC 7501)](https://www.superlogical.com/rex/docs/build/program-status), so terminals and agent dashboards can show whether it is working, waiting for you, done, or failed:

| State | When |
|---|---|
| `working` | An agent run or compaction is in progress. The message is the session name. |
| `blocked` | An extension dialog or login waits for you. The message is the dialog title. |
| `done` | A run finished. The message is the session name. |
| `error` | A run ended with an error that is not retried. The message is the first line of the error. |
| `idle` | pidrei started, or you cancelled the run. |

Reports never contain prompts or model output. pidrei sends them only after the terminal answers the protocol's support query; tmux and screen do not forward them. Set `PIDREI_PROGRAM_STATUS=1` to send reports without asking, or `PIDREI_PROGRAM_STATUS=0` to turn them off.

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
