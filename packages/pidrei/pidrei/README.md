# pidrei

A terminal coding agent — a Python port of
[pi](https://github.com/earendil-works/pi) running on free-threaded CPython and
the [tonio](https://github.com/gi0baro/tonio) runtime.

This is the documentation shipped inside the installed package. The
[full documentation index](docs/index.md) lists everything; the
[examples directory](examples/) has working extensions you can run directly.

## Running

```bash
pidrei                       # interactive mode in the current directory
pidrei "explain this repo"   # interactive, with a first message
pidrei -p "list the tests"   # print mode: run once, print, exit
pidrei --mode json -p "..."  # print mode with a structured event stream
pidrei --mode rpc            # JSONL request/response over stdin/stdout
```

`pi3` is installed as a shorter alias for the same entry point.

Authenticate with `/login` for subscription providers, or export an API key
such as `ANTHROPIC_API_KEY` before starting. See [docs/providers.md](docs/providers.md).

## Useful flags

| Flag | Meaning |
|------|---------|
| `-c`, `--continue` | Resume the most recent session in this directory |
| `-r`, `--resume` | Pick a session to resume |
| `--provider` / `--model` | Choose the model for this run |
| `--thinking <level>` | `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` |
| `-t`, `--tools` | Restrict to a comma-separated tool list |
| `-e`, `--extension <path>` | Load an extension for this run |
| `-ne`, `--no-extensions` | Skip all extensions |
| `--offline` | Disable startup network operations |

`pidrei --help` prints the complete list, including every provider environment
variable.

## Configuration

Everything lives under `~/.pidrei/agent/`:

| File | Purpose |
|------|---------|
| `settings.json` | Global settings; a project may add `.pidrei/settings.json` |
| `models.json` | Custom model and provider entries — [docs/models.md](docs/models.md) |
| `auth.json` | Credentials, written by `/login`, mode `0600` |
| `models-store.json` | Cached provider catalogs for offline use |

Project-scoped resources live in `.pidrei/` inside the project:
`extensions/`, `skills/`, `prompts/`, `themes/`.

## Extending

pidrei extensions are Python modules that define an `extension(pi)` function:

```python
def extension(pi):
    @pi.on("session_start")
    async def on_start(event, ctx):
        ctx.ui.notify("hello from an extension", "info")
```

Drop that in `.pidrei/extensions/hello.py` and it loads on the next start.
[docs/extensions.md](docs/extensions.md) documents the whole API; the
[examples](examples/extensions/) are working extensions covering the range of it.

## Credits

pidrei is a translation, not an invention. **pi** is © 2025
[Mario Zechner](https://github.com/badlogic) and
[Earendil Works](https://github.com/earendil-works), and is the design,
architecture and nearly every behavioural decision here. Both projects are MIT
licensed; see `LICENSE`, which carries both notices.

Model and provider metadata is generated from [models.dev](https://models.dev).
