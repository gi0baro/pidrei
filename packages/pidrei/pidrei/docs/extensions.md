# Extensions

Extensions are Python modules that add tools, slash commands, keyboard
shortcuts, CLI flags, event handlers, providers, and custom UI. They run
in-process, on the same interpreter as pidrei itself.

This is the largest deliberate divergence from pi, whose extensions are
TypeScript modules loaded through jiti. The hook bus, the event payloads and
the `pi` object mirror pi's one-for-one; the module format cannot.

## Table of contents

- [Quick start](#quick-start)
- [The extension ABI](#the-extension-abi)
- [Where extensions live](#where-extensions-live)
- [Imports](#imports)
- [Events](#events)
- [ExtensionContext](#extensioncontext)
- [The `pi` object](#the-pi-object)
- [Custom tools](#custom-tools)
- [Commands, shortcuts and flags](#commands-shortcuts-and-flags)
- [State](#state)
- [Custom UI](#custom-ui)
- [Providers](#providers)
- [Error handling](#error-handling)
- [Mode behaviour](#mode-behaviour)
- [Examples](#examples)

## Quick start

Create `.pidrei/extensions/hello.py` in your project:

```python
def extension(pi):
    async def handle(_args, ctx):
        ctx.ui.notify("hello", "info")

    pi.register_command("hello", handler=handle, description="Say hello")
```

Start pidrei and type `/hello`. To load an extension without installing it:

```bash
pidrei -e ./path/to/hello.py
```

## The extension ABI

Three rules, and they are the whole contract.

**1. An extension is a module that defines `extension`.** pi uses
`export default function (pi) { ... }`; Python has no default export, so the
module-level name `extension` is the factory:

```python
def extension(pi):
    pi.on("session_start", lambda event, ctx: None)
```

The factory may be `async def` — it is awaited if so. It receives one argument,
the [`pi` object](#the-pi-object). Its return value is ignored.

A module that does not define `extension`, or defines it as something not
callable, fails to load with:

```
Extension does not define a valid extension() factory function: <path>
```

**2. A directory extension is a package.** pi resolves a directory through
`package.json`'s `main`, falling back to `index.ts`. pidrei uses `__init__.py`:

```
.pidrei/extensions/my-extension/
├── __init__.py        # defines extension(pi)
└── helpers.py         # imported as `from . import helpers`
```

Relative imports work inside both directory extensions and single-file ones —
a standalone `hello.py` is given a synthetic parent package rooted at its own
directory, so `from . import helpers` resolves against the file's neighbours.

**3. The manifest is `pyproject.toml`.** For packages that ship extensions
alongside other resources, declare them under `[tool.pidrei]`:

```toml
[tool.pidrei]
extensions = ["src/my_package/agent_ext.py"]
skills = ["skills/"]
prompts = ["prompts/"]
themes = ["themes/"]
```

Paths are relative to the file. See [packages.md](packages.md).

### No module alias table

pi ships a virtual-module table so `import "@earendil-works/pi-ai"` resolves
both inside its compiled binary and in a source checkout. pidrei needs no
equivalent: an extension runs on the same interpreter, on the same `sys.path`,
so it just imports normally.

### Module identity

Every load executes the module under a fresh synthetic name, so an extension
is re-evaluated on `/reload` rather than served from `sys.modules`. This
mirrors pi passing `moduleCache: false` to jiti. Two consequences worth
knowing:

- Module-level state does **not** survive a reload. Keep state in the closure
  or on an object (see [State](#state)).
- Two extensions may both contain a `helpers.py` without colliding.

Files beginning with `_` are skipped by discovery, so helper modules can sit
next to an extension without being loaded as one.

## Where extensions live

Loaded in this order; later entries override earlier ones by filename:

| Location | Scope |
|----------|-------|
| `~/.pidrei/agent/extensions/` | User, all projects |
| `<project>/.pidrei/extensions/` | Project (requires project trust) |
| `-e <path>` | This run only |

Package-provided extensions load with the scope of the package that declares
them. `--no-extensions` skips all of them.

Project extensions are code, and loading them runs that code. pidrei asks
before trusting a project the first time; see the `defaultProjectTrust`
setting.

## Imports

Everything pidrei itself uses is importable:

```python
from pidrei_ai.types import TextContent, ToolDefinition
from pidrei_agent import AgentToolResult
from pidrei_tui import Container, Text, SelectList
import tonio.colored as tonio
```

- `pidrei_ai` — model types, providers, wire APIs
- `pidrei_agent` — the agent loop, tool result types
- `pidrei_tui` — terminal UI components ([tui.md](tui.md))
- `pidrei` — the CLI internals; stable enough to read, not a stable API
- `tonio` — the async runtime. Use `tonio.spawn_blocking` for blocking I/O.

Third-party packages work if they are installed in the same environment.

## Events

Register with `pi.on(name, handler)`. Handlers must be `async def` (or
return an awaitable) and are awaited; a plain sync function is not accepted.
Each receives `(event, ctx)` where `ctx` is an
[ExtensionContext](#extensioncontext).

### Lifecycle

```
pidrei starts
  ├─► project_trust        (user/global and -e extensions only, before project resources)
  ├─► session_start        { reason: "startup" }
  └─► resources_discover   { reason: "startup" }

user sends a prompt
  ├─► (extension commands are checked first and bypass the rest)
  ├─► input                (intercept, transform, or handle)
  ├─► (skill / template expansion, if not handled)
  ├─► before_agent_start   (inject a message, modify the system prompt)
  ├─► agent_start
  │
  │   ┌── turn (repeats while the model calls tools) ──┐
  │   ├─► turn_start                                   │
  │   ├─► context                  (modify messages)   │
  │   ├─► before_provider_headers  (mutate headers)    │
  │   ├─► before_provider_request  (inspect/replace)   │
  │   ├─► message_start / message_update / message_end │
  │   ├─► tool_call                (block or rewrite)  │
  │   ├─► tool_result              (rewrite output)    │
  │   └─► turn_end                                     │
  │
  └─► agent_end
```

### Catalogue

| Event | Fires | Can change |
|-------|-------|-----------|
| `project_trust` | Before project resources load | Grant or refuse trust |
| `session_start` | Startup, and on new/switched sessions | — |
| `session_shutdown` | Before the process exits | Block shutdown |
| `session_before_compact` | Before compaction runs | — |
| `session_before_fork` / `session_before_switch` / `session_before_tree` | Before the matching session action | — |
| `resources_discover` | Startup and `/reload` | Add resource paths |
| `input` | User submitted input | Transform, or handle it entirely |
| `before_agent_start` | Before the loop starts | Message, system prompt |
| `agent_start` / `agent_end` | Around the whole run | — |
| `turn_start` / `turn_end` | Around each provider round-trip | — |
| `context` | Before each request | The message list |
| `before_provider_headers` | Before each request | Request headers |
| `before_provider_request` | Before each request | The payload |
| `message_start` / `message_update` / `message_end` | Assistant message stream | `message_end` may rewrite |
| `tool_call` | Before a tool runs | Block it, or rewrite arguments |
| `tool_result` | After a tool runs | Rewrite the result |
| `user_bash` | User ran a `!` command | — |
| `model_change` / `thinking_level_change` | Selection changed | — |

Handlers that return `None` leave the event unchanged. Handlers that return a
value replace the corresponding field — the table's "can change" column says
which. Ordering follows extension load order.

### Blocking a tool call

```python
def extension(pi):
    async def guard(event, _ctx):
        if event["toolName"] == "bash" and "rm -rf" in event["args"].get("command", ""):
            return {"block": True, "reason": "refusing a destructive command"}

    pi.on("tool_call", guard)
```

### Transforming input

```python
def extension(pi):
    async def expand(event, _ctx):
        if event["text"].startswith("??"):
            return {"text": f"Explain in detail: {event['text'][2:]}"}

    pi.on("input", expand)
```

## ExtensionContext

The second handler argument. The useful members:

| Member | Purpose |
|--------|---------|
| `ctx.cwd` | Working directory of the session |
| `ctx.has_ui` | False in print and RPC modes — check before touching `ctx.ui` |
| `ctx.ui` | UI surface (below) |
| `ctx.session_manager` | Session entries and metadata |
| `ctx.is_idle()` | Whether the agent is between runs |
| `ctx.has_pending_messages()` | Whether queued messages are waiting |

`ctx.ui` offers `notify(text, level)`, `set_status(...)`, `set_widget(...)`,
`select(title, options)`, `editor(title, prefill)`, `prompt(...)` and `theme`.
The `select`/`editor`/`prompt` calls are awaitable and return the user's
choice, or `None` if dismissed.

In print and RPC modes `ctx.has_ui` is False and `ctx.ui` is a no-op object, so
handlers stay safe to call unconditionally — but a handler that *waits* on user
input should check `ctx.has_ui` first, or it will wait forever on nothing.

Command handlers receive a slightly wider context that also carries the command
arguments.

## The `pi` object

### Registration

| Method | Purpose |
|--------|---------|
| `on(event, handler)` | Subscribe to an event |
| `register_tool(tool)` | Add a tool the model can call |
| `register_command(name, *, handler, description=None, get_argument_completions=None)` | Add a slash command |
| `register_shortcut(shortcut, *, handler, description=None)` | Bind a key |
| `register_flag(name, *, type, description=None, default=None)` | Add a CLI flag |
| `register_message_renderer(custom_type, renderer)` | Render a custom message |
| `register_entry_renderer(custom_type, renderer)` | Render a custom session entry |
| `get_flag(name)` | Read a registered flag's value |

### Actions

| Method | Purpose |
|--------|---------|
| `send_message(message, options=None)` | Queue a message (custom types allowed) |
| `send_user_message(content, options=None)` | Queue a user message |
| `await append_entry(custom_type, data=None)` | Append a custom session entry |
| `await set_session_name(name)` / `get_session_name()` | Session name |
| `await set_label(entry_id, label)` | Label a session entry |
| `await exec(command, args, *, cwd=None, **options)` | Run a subprocess |
| `get_active_tools()` / `set_active_tools(names)` | The active tool set |
| `get_all_tools()` | Every registered tool |
| `get_commands()` | Every registered command |
| `await set_model(model)` | Switch model |
| `get_thinking_level()` / `await set_thinking_level(level)` | Reasoning level |
| `events` | Shared bus for extension-to-extension messages |

The rows written with `await` return an awaitable and do nothing until you await
it. Forgetting the `await` is silent — the call appears to succeed and the work
never happens — so reach for it whenever the table shows it.

`append_entry`, `set_session_name`, `set_label` and `set_thinking_level` write
to the session or settings file, and awaiting them is what guarantees the write
landed before the next statement runs. `pi.exec` likewise starts the command
only when awaited, and resolves to a result with `stdout`, `stderr`, `code` and
`killed`.

Both `send_message` and `send_user_message` accept `{"deliverAs": "followUp"}`
to queue behind the current run, and `{"triggerTurn": True}` to start one.

### Providers

`register_provider(name, config)` registers by name and config;
`register_provider(provider)` registers a constructed provider object.
`unregister_provider(name)` removes one. Registrations made while extensions
are still loading are queued and flushed once the model registry exists, so an
extension does not need to defer them. See [custom-provider.md](custom-provider.md).

## Custom tools

A tool is a `ToolDefinition` with a JSON-schema parameter spec and an execute
function. It must return an `AgentToolResult` — a bare dict is not accepted.

```python
from pidrei_agent import AgentToolResult
from pidrei_ai.types import TextContent, ToolDefinition


def extension(pi):
    async def execute(args, _ctx):
        return AgentToolResult(
            content=[TextContent(type="text", text=f"echo: {args['text']}")],
            details={},
        )

    pi.register_tool(
        ToolDefinition(
            name="echo",
            description="Echo the given text back",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            execute=execute,
        )
    )
```

The `content` list is what the model sees. `details` is arbitrary data for your
own renderers and is not sent to the model.

Extension tools respect the same filters as built-in ones: `--tools` restricts
to a list, `--exclude-tools` removes some, and `--no-builtin-tools` drops
pidrei's own tools while keeping extension tools.

## Commands, shortcuts and flags

```python
def extension(pi):
    async def handle(args, ctx):
        ctx.ui.notify(f"ran with {args!r}", "info")

    pi.register_command("mycmd", handler=handle, description="Does a thing")
    pi.register_shortcut("ctrl+shift+k", handler=handle, description="Same, by key")
    pi.register_flag("aggressive", type="boolean", description="Go faster", default=False)
```

Command handlers receive `(args_string, ctx)`. Supply
`get_argument_completions` to offer completions after the command name.

Shortcut strings are `ctrl+`, `alt+`, `shift+` prefixes plus a key. A shortcut
that collides with a reserved binding is refused with a diagnostic — see
[keybindings.md](keybindings.md) for the reserved set.

Flags become `--aggressive` on the command line, readable with
`pi.get_flag("aggressive")`.

## State

Because a reload re-executes the module, module-level globals reset. Keep state
in the factory closure:

```python
def extension(pi):
    seen = []

    async def remember(event, _ctx):
        seen.append(event["text"])

    pi.on("input", remember)
```

or, when an extension grows past a few handlers, on an object — which is what
the `plan_mode` example does:

```python
class MyExtension:
    def __init__(self, pi):
        self.pi = pi
        self.count = 0

    def wire(self):
        self.pi.on("turn_end", self.on_turn_end)

    async def on_turn_end(self, _event, _ctx):
        self.count += 1


def extension(pi):
    MyExtension(pi).wire()
```

For state that must outlive the process, write to a file under `ctx.cwd` or the
agent directory. For state shared between extensions, use `pi.events`.

## Custom UI

`ctx.ui.set_widget(...)` places a persistent component; `ctx.ui.set_status(...)`
sets a one-line status; renderers registered with
`register_message_renderer` / `register_entry_renderer` control how custom
messages and entries appear in the transcript.

Components come from `pidrei_tui` — `Container`, `Text`, `Spacer`,
`SelectList`, `SettingsList` and friends. See [tui.md](tui.md).

Guard UI work with `ctx.has_ui`.

## Error handling

An exception raised while loading an extension is reported as a diagnostic and
that extension is skipped; the rest still load. An exception inside a handler is
reported and the run continues — one broken extension does not take down the
session. Diagnostics appear at startup and in `/extensions`.

## Mode behaviour

| Mode | Extensions | UI |
|------|-----------|-----|
| Interactive | Loaded | Full |
| Print (`-p`) | Loaded | `has_ui` False |
| JSON (`--mode json`) | Loaded | `has_ui` False |
| RPC (`--mode rpc`) | Loaded | Requests forwarded to the client |

Commands and shortcuts only make sense in interactive mode; tools and event
handlers work everywhere.

## Examples

Working extensions live in [`examples/extensions/`](../examples/extensions/):

| Example | Shows |
|---------|-------|
| `trigger_compact.py` | Triggering compaction from an event |
| `input_transform_streaming.py` | Rewriting user input, streaming output |
| `git_merge_and_resolve.py` | `pi.exec`, follow-up messages, blocking I/O off the loop |
| `plan_mode/` | Flags, commands, shortcuts, tool gating, context filtering, widgets — the widest use of the API |

Run any of them with `pidrei -e <path>`.
