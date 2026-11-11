# Extensions

> pidrei can create extensions. Ask it to build one for your use case.

Extensions are Python modules that extend pidrei: they subscribe to lifecycle
events, register tools the model can call, and add slash commands, keyboard
shortcuts, CLI flags, providers and custom UI. They run in-process, on the
same interpreter as pidrei itself, with your operating system permissions: an
extension can read prompts, tool calls, files, credentials and session
history, so load them only from sources you trust.

This is the largest deliberate divergence from pi, whose extensions are
TypeScript modules loaded through jiti. The hook bus, the event payloads and
the `pi` object mirror pi's one-for-one; the module format cannot.

> **Placement:** put extensions in `~/.pidrei/agent/extensions/` (global) or
> `.pidrei/extensions/` (project) for auto-discovery, and use
> `pidrei -e ./path.py` for quick tests. `/reload` reloads extensions from
> every location.

**Key capabilities:**

- **Custom tools**: register tools the model can call with `pi.register_tool()`
- **Event interception**: block or modify tool calls, inject context,
  customize compaction
- **User interaction**: prompt the user through `ctx.ui` (select, confirm,
  input, notify)
- **Custom UI components**: full TUI components with keyboard input through
  `ctx.ui.custom()` for complex interactions
- **Custom commands**: register commands like `/mycommand` with
  `pi.register_command()`
- **Session persistence**: store state that survives restarts with
  `pi.append_entry()`
- **Custom rendering**: control how tool calls, results and messages appear
  in the TUI
- **Providers and MCP servers**: register model providers, virtual models and
  MCP servers

**Example use cases:**

- Permission gates (confirm before `rm -rf`, `sudo`, and so on)
- Git checkpointing (stash at each turn, restore on branch)
- Path protection (block writes to `.env`, `node_modules/`)
- Custom compaction (summarize the conversation your way)
- Conversation summaries (see the `summarize.py` example)
- Interactive tools (questions, wizards, custom dialogs)
- Stateful tools (todo lists, connection pools)
- External integrations (file watchers, webhooks, CI triggers)
- Games while you wait (see the `snake.py` example)

See [examples/extensions/](../examples/extensions/) for working
implementations.

## Table of contents

- [Quick start](#quick-start)
- [Where extensions live](#where-extensions-live)
- [Available imports](#available-imports)
- [Writing an extension](#writing-an-extension)
  - [Extension styles](#extension-styles)
- [Events](#events)
  - [Lifecycle overview](#lifecycle-overview)
  - [Handler composition](#handler-composition)
  - [Startup events](#startup-events)
  - [Resource events](#resource-events)
  - [Session events](#session-events)
  - [Agent events](#agent-events)
  - [Model events](#model-events)
  - [Tool events](#tool-events)
  - [User bash events](#user-bash-events)
  - [Input events](#input-events)
- [ExtensionContext](#extensioncontext)
- [ExtensionCommandContext](#extensioncommandcontext)
- [The `pi` object](#the-pi-object)
- [State management](#state-management)
- [Custom tools](#custom-tools)
  - [Tool exposure](#tool-exposure)
  - [Calling other tools](#calling-other-tools)
  - [Dynamic tool loading](#dynamic-tool-loading)
- [MCP servers](#mcp-servers)
- [Custom UI](#custom-ui)
- [Error handling](#error-handling)
- [Mode behaviour](#mode-behaviour)
- [Examples](#examples)

## Quick start

Create `~/.pidrei/agent/extensions/my_extension.py`:

```python
from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent


async def extension(pi):
    # React to events
    async def on_session_start(_event, ctx):
        ctx.ui.notify("Extension loaded!", "info")

    async def on_tool_call(event, ctx):
        if event["toolName"] == "bash" and "rm -rf" in event["input"].get("command", ""):
            ok = await ctx.ui.confirm("Dangerous!", "Allow rm -rf?")
            if not ok:
                return {"block": True, "reason": "Blocked by user"}
        return None

    pi.on("session_start", on_session_start)
    pi.on("tool_call", on_tool_call)

    # Register a custom tool
    async def greet(_tool_call_id, params, _cancel, _on_update, _ctx):
        return AgentToolResult(content=[TextContent(text=f"Hello, {params['name']}!")], details={})

    pi.register_tool(
        ToolDefinition(
            name="greet",
            label="Greet",
            description="Greet someone by name",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Name to greet"}},
                "required": ["name"],
            },
            execute=greet,
        )
    )

    # Register a command
    async def hello(args, ctx):
        ctx.ui.notify(f"Hello {args or 'world'}!", "info")

    pi.register_command("hello", handler=hello, description="Say hello")
```

Test it with the `--extension` (or `-e`) flag:

```bash
pidrei -e ./my_extension.py
```

## Where extensions live

Extensions run with your full system permissions and can execute arbitrary
code. Install them only from sources you trust.

Extensions are discovered automatically from these locations:

| Location | Scope |
|----------|-------|
| `~/.pidrei/agent/extensions/*.py` | Global (all projects) |
| `~/.pidrei/agent/extensions/*/__init__.py` | Global (package directory) |
| `~/.pidrei/agent/extensions/*/pyproject.toml` with a `[tool.pidrei] extensions` list | Global (directory with a manifest) |
| `<project>/.pidrei/extensions/...` | Project, same layouts (requires project trust) |
| `-e <path>` | This run only |

Discovery goes one level deep; larger packages declare their entry points in
a `pyproject.toml` manifest. Files beginning with `_` are skipped, so helper
modules can sit next to an extension without being loaded as one.

Settings add more paths: `packages` takes package sources and `extensions`
takes local extension files or directories:

```json
{
  "packages": ["git:github.com/user/repo@v1"],
  "extensions": ["/path/to/local/extension.py", "/path/to/local/extension/dir"]
}
```

Package-provided extensions load with the scope of the package that declares
them; see [packages.md](packages.md) to share extensions through git as
pidrei packages. `--no-extensions` skips all of them.

pidrei also ships built-in extensions, named `builtin:mcp`,
`builtin:codemode`, and `builtin:tool-search` in the `extensions` setting.
They load by default, after the others; `-builtin:mcp` disables one. A
`+builtin:<name>` or `-builtin:<name>` entry in project settings overrides
the user setting. `pidrei config` lists them under Built-in.
`--no-extensions` disables them too, and `-e builtin:<name>` loads one
explicitly.

Project extensions are code, and loading them runs that code. pidrei asks
before trusting a project the first time; see the `defaultProjectTrust`
setting. Only user and `-e` extensions load early enough to handle the
`project_trust` event.

## Available imports

Everything pidrei itself uses is importable:

```python
from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent
from pidrei_tui import Container, Text, SelectList
import tonio.colored as tonio
```

| Package | Purpose |
|---------|---------|
| `pidrei` | Extension types (`pidrei.core.extensions`: `ToolDefinition`, the event and context types) and the CLI internals; stable enough to read, not a stable API |
| `pidrei_agent` | Agent message and tool result types |
| `pidrei_ai` | Model types, providers, wire APIs, transcript helpers |
| `pidrei_http` | The HTTP client pidrei uses (pooled, proxy-aware) |
| `pidrei_tui` | Terminal UI components for custom rendering ([tui.md](tui.md)) |
| `tonio` | The async runtime. Use `tonio.spawn_blocking` for blocking I/O |

There is no import alias table to configure: an extension runs on the same
interpreter and the same `sys.path` as pidrei, so it imports these packages
directly and gets the very objects pidrei is using.

Third-party packages work if they are installed in the same environment.
pidrei does not install an extension's or a package's Python dependencies:
vendor them or install them yourself (see [packages.md](packages.md)).

Tool parameters are JSON Schema dicts; see
[Tool definition](#tool-definition) for how pidrei validates and coerces the
arguments.

## Writing an extension

An extension is a module that defines `extension`. pi uses
`export default function (pi) { ... }`; Python has no default export, so the
module-level name `extension` is the factory. It receives one argument, the
[`pi` object](#the-pi-object):

```python
async def extension(pi):
    # Subscribe to events
    async def on_event(event, ctx):
        # ctx.ui for user interaction
        ok = await ctx.ui.confirm("Title", "Are you sure?")
        ctx.ui.notify("Done!", "info")
        ctx.ui.set_status("my-ext", "Processing...")  # Footer status
        ctx.ui.set_widget("my-ext", ["Line 1", "Line 2"])  # Widget above the editor

    pi.on("event_name", on_event)

    # Register tools, commands, shortcuts, flags
    pi.register_tool(ToolDefinition(...))
    pi.register_command("name", handler=...)
    pi.register_shortcut("ctrl+x", handler=...)
    pi.register_flag("my-flag", type="boolean")
```

The factory must be an `async def`; it is awaited. A plain `def extension(pi)`
is not accepted (pi's `void | Promise<void>` union is not ported) and fails to
load like any other factory error. Its return value is ignored. A module that
does not define `extension`, or defines it as something not callable, fails
to load with:

```
Extension does not define a valid extension() factory function: <path>
```

pidrei awaits the factory before startup continues, so its initialization
completes before `session_start`, before `resources_discover`, and before the
provider registrations it queued with `pi.register_provider()` are flushed.

### Async factory functions

Use the factory for one-time startup work such as fetching remote
configuration or discovering the available models:

```python
import json

from pidrei_http import http


async def extension(pi):
    response = await http.default_fetch(http.build_request("GET", "http://localhost:1234/v1/models"))
    payload = json.loads(await response.read())

    pi.register_provider(
        "local-openai",
        {
            "baseUrl": "http://localhost:1234/v1",
            "apiKey": "$LOCAL_OPENAI_API_KEY",
            "api": "openai-completions",
            "models": [
                {
                    "id": model["id"],
                    "name": model.get("name", model["id"]),
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": model.get("context_window", 128000),
                    "maxTokens": model.get("max_tokens", 4096),
                }
                for model in payload["data"]
            ],
        },
    )
```

This makes the fetched models available during normal startup and to
`pidrei --list-models`. For model lists that change while pidrei runs,
implement a model refresh instead; see
[custom-provider.md](custom-provider.md#model-refresh).

### Long-lived resources and shutdown

The factory runs whenever extensions load, and some invocations load them
without ever starting a session. So don't start processes, sockets, watchers
or timers in the factory: start them from `session_start`, or from the
command, tool or event that needs them, and release them in a
`session_shutdown` handler. Keep that cleanup idempotent: quit, reload and
session replacement all converge on it.

`/reload` (or `await ctx.reload()` from a command) replaces the whole
extension runtime: code that runs after the reload must not reuse objects or
the `ctx` from the old one.

### Extension styles

**Single file**, the simplest, for small extensions:

```
~/.pidrei/agent/extensions/
└── my_extension.py
```

**A package directory**, for multi-file extensions. pi resolves a directory
through `package.json`'s `main`, falling back to `index.ts`; pidrei uses
`__init__.py`:

```
~/.pidrei/agent/extensions/
└── my_extension/
    ├── __init__.py     # defines extension(pi)
    ├── tools.py        # imported as `from . import tools`
    └── utils.py
```

Relative imports work inside both package directories and single files: a
standalone `hello.py` is given a synthetic parent package rooted at its own
directory, so `from . import helpers` resolves against the file's neighbours.

**A directory with a manifest**, for packages that ship extensions alongside
other resources. The manifest is `pyproject.toml`; declare the entry points
under `[tool.pidrei]`:

```toml
[tool.pidrei]
extensions = ["src/my_package/agent_ext.py"]
skills = ["skills/"]
prompts = ["prompts/"]
themes = ["themes/"]
```

Paths are relative to the file. See [packages.md](packages.md).

### Module identity

Every load executes the module under a fresh synthetic name, so an extension
is re-evaluated on `/reload` rather than served from `sys.modules`. This
mirrors pi passing `moduleCache: false` to jiti. Two consequences worth
knowing:

- Module-level state does **not** survive a reload. Keep state in the closure
  or on an object (see [State management](#state-management)).
- Two extensions may both contain a `helpers.py` without colliding.

## Events

Register with `pi.on(name, handler)`. Handlers must be `async def` (or
return an awaitable) and are awaited; a plain function is not accepted, and
calling one fails when the event fires. Each receives `(event, ctx)`: `event`
is a dict with pi's camelCase keys (`event["toolName"]`), and `ctx` is an
[ExtensionContext](#extensioncontext). Handlers return `None` or a dict, also
with camelCase keys (`{"block": True, "reason": ...}`).

`pi.on` returns an unsubscribe function that removes only that registration:

```python
async def on_agent_end(event, ctx):
    unsubscribe()
    await update_integration(event["messages"])


unsubscribe = pi.on("agent_end", on_agent_end)
```

### Lifecycle overview

A prompt starts a run. Input handling comes first:

1. Extension commands (`/name`) run their handler, and the `input` event does
   not fire. Typed and RPC prompts always check for commands;
   `pi.send_user_message()` checks only with `{"expandPromptTemplates": True}`.
2. `input` handlers can intercept, transform, or handle the text.
3. Skill commands (`/skill:name`) and prompt templates (`/template`) expand.
4. `before_agent_start` runs, then the agent loop starts.

The run repeats turns, then checks whether more work starts another
low-level agent run. Each turn sends one model request and executes the tool
calls in the assistant response:

```
user sends a prompt
  ├─► (extension commands are checked first and bypass the rest)
  ├─► input                (intercept, transform, or handle)
  ├─► (skill / template expansion, if not handled)
  ├─► before_agent_start   (inject a message, modify the system prompt)
  ├─► agent_start  ◄──────────────────────────────┐
  │     turns, see below                          │
  ├─► agent_end                                   │
  │     more work? (retry, recovery, compaction   │
  │     retry, queued messages) ── yes ───────────┤
  ├─► agent_before_settle  (append entries)       │
  │     continuation requested? ── yes ───────────┘
  └─► agent_settled        (final, notification only)

each turn (repeats while the model calls tools)
  ├─► turn_start
  ├─► model request
  │     ├─► context                  (modify conversation)
  │     ├─► context_with_system      (modify full transcript)
  │     ├─► before_provider_headers  (mutate headers)
  │     ├─► before_provider_request  (inspect/replace)
  │     ├─► after_provider_response  (status/headers)
  │     └─► provider_stream_event, message_start / message_update / message_end
  ├─► each tool call
  │     ├─► tool_execution_start
  │     ├─► tool_call                (block or rewrite)
  │     ├─► tool_execution_update    (while the tool runs)
  │     ├─► tool_result              (rewrite output)
  │     └─► tool_execution_end
  └─► turn_end                       (append entries, continue)
        └─► threshold compaction before a naturally required next turn
```

More work means a retry, final-attempt recovery, a compaction retry, or queued
messages. `agent_end` is therefore not a final-idle signal: use
`agent_before_settle` to append final entries or request one last
continuation, and `agent_settled` for notification-only work once nothing
continues automatically. An aborted run skips `agent_before_settle` and goes
from `agent_end` to `agent_settled`. pidrei counts as busy while a run starts,
including the compaction check and `before_agent_start`, so messages sent
meanwhile go into the starting run.

The model request runs its hooks in the order the diagram shows: `context`
(can modify the conversation messages), `context_with_system` (can modify the
full transcript), `before_provider_headers` (can mutate headers),
`before_provider_request` (can inspect or replace the payload),
`after_provider_response` (status and headers, before the stream is consumed),
then `provider_stream_event` (raw parsed stream events) alongside
`message_start` / `message_update` / `message_end` for the assistant message.
Provider hooks run only when the transport exposes the matching stage.

A tool call that executes runs `tool_execution_start`, `tool_call` (can block
or mutate the input), `tool_execution_update` while the tool runs,
`tool_result` (can modify the result) and `tool_execution_end`, in that order.
Calls that do not execute end early. An unknown tool or invalid arguments skip
`tool_call`; a blocked or aborted call skips execution and `tool_result`. Each
of these still gets `tool_execution_end` with an error result, so use
`tool_execution_end` to observe every finalized call.

`turn_end` can append entries and request a continuation. Threshold compaction
runs before a naturally required next turn. Message events also fire for
other messages, such as user, tool-result and custom messages; the diagram
shows only the assistant stream, where it occurs in each turn.

Session startup and replacement follow a separate lifecycle:

```
pidrei starts
  ├─► project_trust        (when the project needs trust; user and -e extensions only)
  ├─► session_start        { reason: "startup" }
  └─► resources_discover   { reason: "startup" }

/new, /resume, /import, /fork or /clone
  ├─► session_before_switch (/new, /resume, /import) or session_before_fork (/fork, /clone): may cancel
  ├─► session_shutdown     { reason: "new" | "resume" | "fork" }  (the old runtime)
  ├─► project_trust        (when the replacement enters a working directory not yet resolved)
  ├─► (the replacement extension runtime loads)
  ├─► session_start        { reason: "new" | "resume" | "fork" }
  └─► resources_discover   { reason: "startup" }

/reload
  ├─► session_shutdown     { reason: "reload" }
  ├─► (the extension runtime reloads)
  ├─► session_start        { reason: "reload" }
  └─► resources_discover   { reason: "reload" }

quit
  └─► session_shutdown     { reason: "quit" }
```

`session_start` carries `reason` `"startup"`, `"reload"`, `"new"`, `"resume"`
or `"fork"`, and `session_shutdown` carries `"quit"`, `"reload"`, `"new"`,
`"resume"` or `"fork"`. `resources_discover` uses `"reload"` after a reload
and `"startup"` otherwise. Quitting includes SIGTERM and SIGHUP, in every
mode. `project_trust` runs before trusted project resources load
and can run again when a session replacement enters a working directory whose
trust this process has not resolved yet.

Other operations change the current session without replacing the runtime:

| Operation | Events |
|---|---|
| `/compact` or automatic compaction | `session_before_compact` (can cancel or customize), then `session_compact` or `session_compact_failed` |
| `/tree` navigation | `session_before_tree` (can cancel or customize), then `session_tree` |
| `/name` or `pi.set_session_name()` | `session_info_changed` |
| `/model`, model cycling (`Ctrl+P`) or `pi.set_model()` | `thinking_level_select` if the level changes (for example, clamped to the new model), and `model_select` |
| Thinking-level change (`Shift+Tab`, `pi.set_thinking_level()`) | `thinking_level_select` |
| `!` or `!!` command | `user_bash` (can intercept) |
| Extension dialog or custom screen waits for the user | `ui_prompt_start`, `ui_prompt_end` |
| Idle prompt-cache refresh | `cache_warming_decision` |
| Extension registers or removes an MCP server after startup | `mcp_servers_change` |

### Choosing an event

| Goal | Event |
|------|-------|
| Inspect, transform or consume raw user input | `input` |
| Change prompt sections or active tools, or add context, before a run | `before_agent_start` |
| Observe one low-level agent run | `agent_start`, `agent_end` |
| Act after one model response and its tool calls | `turn_end` |
| Filter the conversation for one provider request | `context` |
| Transform the complete transcript, system messages included | `context_with_system` |
| Change the request headers or the serialized provider payload | `before_provider_headers`, `before_provider_request` |
| Observe provider response metadata or parsed stream events | `after_provider_response`, `provider_stream_event` |
| Authorize or modify a tool call | `tool_call` |
| Redact, annotate or replace a tool result | `tool_result` |
| Append final entries or request one last continuation | `agent_before_settle` |
| Run notification work after automatic processing has stopped | `agent_settled` |
| Set up or release session-scoped resources | `session_start`, `session_shutdown` |

### Catalogue

| Event | Fires | Can change |
|-------|-------|-----------|
| `project_trust` | Before project resources load | Grant or refuse trust |
| `session_start` | Startup, reload, and new/resumed/forked sessions (`reason`) | — |
| `session_shutdown` | Before the session is torn down: quit, reload, or session replacement (`reason`) | — (release resources here) |
| `session_info_changed` | The session name was set or cleared (`name`) | — |
| `session_before_compact` | Before compaction runs | `{"cancel": True}`, or supply `{"compaction": CompactionResult(...)}` |
| `session_compact` / `session_compact_failed` | After compaction succeeds / after it fails or is aborted | — |
| `session_before_fork` / `session_before_switch` | Before the matching session action | `{"cancel": True}` |
| `session_before_tree` | Before `/tree` navigation | `{"cancel": True}`, a `summary`, or `customInstructions` / `replaceInstructions` / `label` |
| `resources_discover` | After every `session_start` (`reason` is `"reload"` after a reload, else `"startup"`) | Add resource paths |
| `input` | User submitted input | Transform, or handle it entirely |
| `before_agent_start` | Before the loop starts | Message; prompt sections, tools and rules via the mutable `event["systemPromptOptions"]` (sent as a patch); or the whole system prompt for the run |
| `agent_start` / `agent_end` | Around each low-level run (pidrei may still retry, compact, or continue) | — |
| `turn_start` | Before each provider round-trip | — |
| `turn_end` | After a turn's assistant and tool-result messages are persisted | Append session entries; request one continuation (see [turn_start / turn_end](#turn_start--turn_end)) |
| `agent_before_settle` | Last actionable point before the run settles | Append session entries; request one continuation |
| `agent_settled` | The run has settled; nothing continues automatically (`aborted`) | — (runs started here begin after every settled handler finishes) |
| `context` | Before each request | The conversation, without system messages (see [context](#context)) |
| `context_with_system` | Before each request, after every `context` handler | The full transcript, sent as returned |
| `before_provider_headers` | Before each request | Request headers |
| `before_provider_request` | Before each request | The payload |
| `after_provider_response` | After each response arrives | — (inspect status/headers) |
| `provider_stream_event` | Each parsed provider stream event, before pidrei normalizes it (`provider`, `api`, `model`, `data`) | — (notification-only, not persisted) |
| `cache_warming_decision` | Before each prompt-cache refresh, with pidrei's decision (`warmCost`, `missCost`, `continuationProbability`, `action`) | Return `{"action": "warm"}` or `{"action": "stop"}`; the last handler that returns an action wins; `"stop"` ends warming until the next real request |
| `message_start` / `message_update` / `message_end` | Each user, assistant, tool-result and custom message; `message_update` streams the assistant's | `message_end` may replace the message |
| `tool_execution_start` / `tool_execution_update` / `tool_execution_end` | Around each tool call: before `tool_call`, while the tool runs, after `tool_result` | — |
| `tool_call` | Before a tool runs (`parentToolCallId` set for calls another tool made) | Block it, or rewrite arguments |
| `tool_result` | After a tool runs (`parentToolCallId` as for `tool_call`; `structuredContent` when the tool returned one) | Rewrite the result; replacing `content` without `structuredContent` drops it |
| `mcp_servers_change` | An extension registered or unregistered an MCP server after load (`servers`) | — |
| `user_bash` | User ran a `!` or `!!` command | Return exactly one of `{"operations": ...}` (run through that backend) or a complete `{"result": ...}` (record without running), which stops propagation; `None` passes to the next handler, then to local execution; an invalid result or a raise blocks the command |
| `ui_prompt_start` / `ui_prompt_end` | Around a blocking `ctx.ui` prompt (`select`, `confirm`, `input`, `editor`, `custom`) — nested prompts coalesce into one outer waiting span; handlers are best-effort and not awaited | — |
| `model_select` / `thinking_level_select` | Selection changed | — |

Handlers that return `None` leave the event unchanged. Handlers that return a
value replace the corresponding field — the table's "can change" column says
which; a return value on a notification-only event has no effect.

### Handler composition

Handlers run in extension load order, then registration order within each
extension. Adding or removing a handler does not affect a dispatch already in
progress. Some events notify; others transform data, replace results, or
cancel an operation:

| Pattern | Behaviour |
|---------|-----------|
| Notification | Return values are ignored; the event is for observation only. |
| In-place mutation | Later handlers see changes made to `event["headers"]` in `before_provider_headers`, `event["systemPromptOptions"]` in `before_agent_start` and `event["input"]` in `tool_call`. |
| Chained transformation | Later handlers receive what earlier `input`, `context`, `context_with_system`, `message_end`, `tool_result` and `before_provider_request` handlers returned. |
| Gate | Handled input, a blocked tool call, a cancelled session action or a handled `user_bash` command stops that operation's normal path. |
| Actionable boundary | Each `turn_end` or `agent_before_settle` handler sees the entries and continuation decision proposed so far. |
| Last override | The last `cache_warming_decision` handler that returns an action wins. |

Don't do slow work in a notification handler on a streaming event. A
transforming handler should keep the fields it does not mean to own, and a
gate handler should fail closed when it cannot make its safety decision.

A handler that raises is reported as an extension error and the remaining
handlers still run, with two exceptions: a raising `tool_call` handler blocks
the call (its message becomes the tool's error result) and later `tool_call`
handlers do not run, and a raising `user_bash` handler blocks the command.

The emitter waits for the handlers of most events before it goes on. Agent
loop events (`agent_*`, `turn_*`, `message_*`, `tool_*`) are delivered one at
a time, in the order they occur, even when tool calls run in parallel.
`session_info_changed`, `thinking_level_select`, `ui_prompt_start`,
`ui_prompt_end` and `mcp_servers_change` are dispatched without waiting:
their handlers run alongside the code that triggered them.

### Startup events

#### project_trust

Fired before pidrei decides whether to trust a project that has project
resources (`.pidrei` entries or `.agents/skills`; see
[configuration.md](configuration.md#project-trust)). It runs during startup,
and when a session replacement (for example `/resume`) enters a working
directory whose trust this process has not resolved yet. Only user/global
extensions and `-e` extensions take part: project-local extensions, and the
built-in ones, load only after trust is resolved.

```python
async def extension(pi):
    async def on_project_trust(event, ctx):
        # event["cwd"]: the working directory
        # ctx is a limited trust context: cwd, mode, has_ui, and the
        # select/confirm/input/notify UI helpers
        if ctx.has_ui and await ctx.ui.confirm("Trust project?", event["cwd"]):
            return {"trusted": "yes", "remember": True}
        return {"trusted": "undecided"}

    pi.on("project_trust", on_project_trust)
```

A `project_trust` handler returns `{"trusted": "yes" | "no" | "undecided"}`.
The first handler that returns `"yes"` or `"no"` owns the decision: later
handlers do not run and the built-in trust prompt is skipped. Add
`"remember": True` to save the decision to `~/.pidrei/agent/trust.json`;
otherwise it applies only to the current process. Return `"undecided"` to let
later handlers or the built-in flow decide. Check `ctx.has_ui` before
prompting. When no handler decides, normal trust resolution continues: a saved
decision applies first, then the `defaultProjectTrust` setting controls
whether pidrei asks, trusts, or declines. See
`examples/extensions/project_trust.py`.

### Resource events

#### resources_discover

Fired after `session_start` so extensions can contribute additional skill,
prompt and theme paths. The startup path, and new, resumed and forked
sessions, use `reason: "startup"`; reload uses `reason: "reload"`.

```python
async def extension(pi):
    async def on_resources_discover(event, _ctx):
        # event["cwd"]: the working directory
        # event["reason"]: "startup" | "reload"
        return {
            "skillPaths": ["/path/to/skills"],
            "promptPaths": ["/path/to/prompts"],
            "themePaths": ["/path/to/themes"],
        }

    pi.on("resources_discover", on_resources_discover)
```

See `examples/extensions/dynamic_resources/`.

#### mcp_servers_change

Fired when an extension registers or unregisters an MCP server after
extensions are bound. `event["servers"]` lists every registered server after
the change (each with `name`, `config` and `extension_path`). Handling this
event marks the extension as the one that connects registered servers; see
[MCP servers](#mcp-servers).

### Session events

Session entries are dicts in the session file's format; messages are the
types in [message-types.md](message-types.md).

#### session_start

Fired when a session is started, loaded, or reloaded.

```python
async def on_session_start(event, ctx):
    # event["reason"]: "startup" | "reload" | "new" | "resume" | "fork"
    # event["previousSessionFile"]: present for "new", "resume" and "fork"
    ctx.ui.notify(f"Session: {ctx.session_manager.get_session_file() or 'ephemeral'}", "info")
```

#### session_info_changed

Fired when the current session's display name is set through `/name`, RPC,
or `pi.set_session_name()`.

```python
async def on_session_info_changed(event, ctx):
    # event["name"]: the current normalized name, or None if cleared
    ctx.ui.notify(f"Session renamed: {event['name'] or '(none)'}", "info")
```

#### session_before_switch

Fired before starting a new session (`/new`) or switching to another one
(`/resume`, `/import`).

```python
async def on_session_before_switch(event, ctx):
    # event["reason"]: "new" or "resume"
    # event["targetSessionFile"]: the session being switched to ("resume" only)
    if event["reason"] == "new":
        if not await ctx.ui.confirm("Clear?", "Delete all messages?"):
            return {"cancel": True}
    return None
```

After a successful switch or new session, pidrei emits `session_shutdown` for
the old extension instance, loads and binds the extensions again for the new
session, then emits `session_start` with `reason` `"new"` or `"resume"` and
`previousSessionFile`. Do cleanup work in `session_shutdown`, then re-establish
any in-memory state in `session_start`.

#### session_before_fork

Fired when forking with `/fork` or cloning with `/clone`.

```python
async def on_session_before_fork(event, ctx):
    # event["entryId"]: ID of the selected entry
    # event["position"]: "before" for /fork, "at" for /clone
    return {"cancel": True}  # cancel the fork or clone
```

After a successful fork or clone, pidrei emits `session_shutdown` for the old
extension instance, loads and binds the extensions again for the new session,
then emits `session_start` with `reason: "fork"` and `previousSessionFile`. Do
cleanup work in `session_shutdown`, then re-establish any in-memory state in
`session_start`.

#### session_before_compact / session_compact / session_compact_failed

Fired on compaction.

```python
from pidrei.core.compaction import CompactionResult


async def extension(pi):
    async def on_before_compact(event, ctx):
        preparation = event["preparation"]  # a CompactionPreparation
        # event["branchEntries"]: the entries of the current branch
        # event["customInstructions"]: /compact's instructions, or None
        # event["reason"]: "manual" (/compact), "threshold" or "overflow"
        # event["willRetry"]: whether the aborted turn is retried afterwards (overflow recovery)
        # event["signal"]: cancel token of the compaction

        # Cancel:
        #     return {"cancel": True}

        # Custom summary:
        return {
            "compaction": CompactionResult(
                summary="...",
                first_kept_entry_id=preparation.first_kept_entry_id,
                tokens_before=preparation.tokens_before,
                # usage=summary_usage,  # optional; included in session totals
            )
        }

    async def on_compact(event, ctx):
        # event["compactionEntry"]: the saved compaction entry
        # event["fromExtension"]: whether an extension provided it
        # event["reason"], event["willRetry"]: as above
        ...

    async def on_compact_failed(event, ctx):
        # event["reason"]: "manual", "threshold" or "overflow"
        # event["errorMessage"]: set for failures that were not aborts
        # event["aborted"]: True for cancelled or aborted compactions
        # event["willRetry"]: whether the aborted turn would have been retried
        # event["fromExtension"]: whether extension-provided content was being used
        ...

    pi.on("session_before_compact", on_before_compact)
    pi.on("session_compact", on_compact)
    pi.on("session_compact_failed", on_compact_failed)
```

`preparation` carries `first_kept_entry_id`, `messages_to_summarize`,
`turn_prefix_messages`, `is_split_turn`, `tokens_before`, `file_ops`,
`settings` and `previous_summary`. Cancelling from `session_before_compact`
reports `session_compact_failed` with `aborted` set. See
`examples/extensions/custom_compaction.py`.

#### session_before_tree / session_tree

Fired on `/tree` navigation (see [how-pi-works.md](how-pi-works.md#sessions)
for the session tree).

```python
async def on_before_tree(event, ctx):
    preparation = event["preparation"]
    # preparation keys: targetId, oldLeafId, commonAncestorId, entriesToSummarize,
    # userWantsSummary, customInstructions, replaceInstructions, label
    # event["signal"]: cancel token of the navigation's summary
    return {"cancel": True}
    # OR provide a custom summary (used only when preparation["userWantsSummary"] is true):
    return {
        "summary": {
            "summary": "...",
            # "usage": summary_usage,  # optional; included in session totals
            "details": {},
        },
    }
    # OR override summarization options: "customInstructions", "replaceInstructions", "label"


async def on_tree(event, ctx):
    # event["newLeafId"], event["oldLeafId"], event["summaryEntry"], event["fromExtension"]
    ...
```

#### session_shutdown

Fired before a started session runtime is torn down. Use it to clean up
resources opened from `session_start` or other session-scoped hooks, and keep
the cleanup idempotent.

```python
async def on_session_shutdown(event, ctx):
    # event["reason"]: "quit" | "reload" | "new" | "resume" | "fork"
    # event["targetSessionFile"]: destination session for session replacement flows
    ...  # clean up, save state, etc.
```

### Agent events

#### before_agent_start

Fired before a run starts, after input handling and expansion. Can inject a
message and modify the system prompt.

`event["messageType"]` says what started the run: `"user"` for a prompt or
`pi.send_user_message()`, with `prompt` and `images`; `"custom"` for
`pi.send_message(..., {"triggerTurn": True})` while idle, with the starting
`message` and no `prompt`.

```python
async def extension(pi):
    async def on_before_agent_start(event, ctx):
        # event["messageType"]: "user" | "custom"
        # event["prompt"], event["images"]: the prompt text and images ("user")
        # event["message"]: the custom message that starts the run ("custom")
        # event["systemPrompt"]: the system prompt as of this handler,
        #   including changes from earlier before_agent_start handlers
        # event["systemPromptOptions"]: the structured options the prompt is built from
        options = event["systemPromptOptions"]
        options.prompt_guidelines.append("Prefer small, reviewable diffs.")

        return {
            # Inject a persistent message (stored in the session, sent to the model)
            "message": {
                "customType": "my-extension",
                "content": "Additional context for the LLM",
                "display": True,
            },
            # Or replace the system prompt for this run (chained across extensions):
            # "systemPrompt": event["systemPrompt"] + "\n\nExtra instructions for this turn...",
        }

    pi.on("before_agent_start", on_before_agent_start)
```

`event["systemPromptOptions"]` is a `BuildSystemPromptOptions` with these
attributes:

| Attribute | Contents |
|-----------|----------|
| `custom_prompt` | Exact prompt prefix from `--system-prompt`, `SYSTEM.md` or custom templates |
| `force_system_prompt` | Optional exact replacement for the complete prompt |
| `selected_tools` | Tools currently active in the prompt |
| `hidden_tools` | Selected tools whose declarations requests leave out |
| `tool_snippets` | One-line descriptions, keyed by tool name |
| `tool_guidelines` | Rule bullets, keyed by tool name |
| `prompt_guidelines` | Additional custom rule bullets |
| `sections` | Custom XML-wrapped sections, keyed by tag name |
| `append_system_prompt` | Text from `--append-system-prompt` |
| `cwd` | Working directory |
| `context_files` | `AGENTS.md` files and other loaded context files |
| `skills` | Loaded skills |

A returned `message` is added to either kind of run; check `messageType` to
skip runs started by custom messages.

The options give extensions the same structured data pidrei builds the system
prompt from, and every collection on them is present and mutable. Prefer
editing `sections`, `selected_tools` or `prompt_guidelines` in place: pidrei
diffs the resulting prompt sections against what the model already has and
adds one system message patching only the changed sections. Returning
`{"systemPrompt": ...}`, or setting `force_system_prompt`, replaces the whole
prompt for the run: every provider receives the forced text as its leading
system prompt (a cache miss when it changes), and the session transcript keeps
recording the structured sections. Changing the tool selection updates both
the prompt contributions and the tools sent to the provider; calling
`pi.set_active_tools()` inside the handler has the same effect as editing
`selected_tools`. Models that accept system messages mid-conversation receive
the patch in place and keep their cached prefix; other models get the
replayed prompt as their system prompt, which is a cache miss once per change.

Inside `before_agent_start`, `event["systemPrompt"]` and
`ctx.get_system_prompt()` both reflect the chained system prompt as of the
current handler. Later `before_agent_start` handlers can still change it.

#### agent_start / agent_end / agent_before_settle / agent_settled

`agent_start` fires when a low-level agent run begins. `agent_end` fires when
that run ends, but pidrei may still retry, compact and retry, or continue with
queued follow-up messages. `agent_before_settle` is the final actionable
boundary: it can append session entries and request one continuation.
`agent_settled` is final and notification-only; use it for status
integrations that need to know pidrei will not continue running
automatically.

```python
async def extension(pi):
    added_review_reminder = False

    async def on_agent_end(event, ctx):
        # event["messages"]: the messages of this low-level run
        ...

    async def on_before_settle(event, _ctx):
        nonlocal added_review_reminder
        if added_review_reminder:
            return None
        added_review_reminder = True
        return {
            "entries": [
                *event["entries"],
                {
                    "type": "custom_message",
                    "customType": "review-reminder",
                    "content": "Review the final diff before replying.",
                    "display": False,
                },
            ],
            "continue": True,
        }

    async def on_settled(event, ctx):
        # event["aborted"]: True when the run ended because it was aborted, for example with Escape
        # ctx.is_idle() is True; runs requested here start after every settled handler finishes.
        ...

    pi.on("agent_end", on_agent_end)
    pi.on("agent_before_settle", on_before_settle)
    pi.on("agent_settled", on_settled)
```

`agent_before_settle` receives the same boundary fields as `turn_end` (see
below), without the turn's message fields. If the run is aborted while
`agent_before_settle` handlers are running, valid returned entries are still
committed, but the requested continuation is dropped. Work requested from
`agent_settled` (a prompt, a triggered custom message) is deferred until every
settled handler completes, so the dispatch is not re-entered.

#### ui_prompt_start / ui_prompt_end

Notification-only events around blocking user-facing extension prompts. They
fire around `ctx.ui.select()`, `ctx.ui.confirm()`, `ctx.ui.input()`,
`ctx.ui.editor()` and `ctx.ui.custom()`, so status integrations can report
"waiting for user" instead of just "running".

Nested or overlapping prompts coalesce into one outer waiting span. Handlers
are invoked best-effort and are not awaited before the prompt shows or
closes.

```python
async def on_ui_prompt_start(event, ctx):
    # event["reason"] == "ui_prompt"
    # event["kind"]: "select" | "confirm" | "input" | "editor" | "custom"
    # event.get("title"): the prompt title, when there is one
    ...


async def on_ui_prompt_end(event, ctx):
    # pidrei is no longer waiting on that prompt span (same keys as ui_prompt_start).
    ...
```

#### turn_start / turn_end

Fired for each turn (one model response and its tool calls).

```python
async def extension(pi):
    replaced = False

    async def on_turn_start(event, ctx):
        # event["turnIndex"], event["timestamp"]
        ...

    async def on_turn_end(event, _ctx):
        # event["turnIndex"], event["message"], event["toolResults"]
        # event["entries"]: the session entries proposed so far
        nonlocal replaced
        if replaced or event["outcome"] != "completed" or event["toolResults"]:
            return None
        replaced = True
        return {
            "entries": [
                *event["entries"],
                {"type": "context_edit", "targetId": event["messageEntryId"], "replacement": None},
                {
                    "type": "custom_message",
                    "customType": "replacement-instruction",
                    "content": "Answer again using the persisted user request.",
                    "display": False,
                },
            ],
            "continue": True,
        }

    pi.on("turn_start", on_turn_start)
    pi.on("turn_end", on_turn_end)
```

`turn_end` and `agent_before_settle` are actionable boundaries. `turn_end`
runs after the turn's assistant and tool-result messages have been persisted.
Retry backoff and final-attempt recovery still happen after `agent_end`,
keeping their lifecycle and queue ordering; `agent_before_settle` sees the
repaired context after that work completes. Each handler receives:

- `event["entries"]` — the session entries proposed so far, as dicts:
  `{"type": "custom", "customType", "data"}`,
  `{"type": "custom_message", "customType", "content", "display", "details"}`,
  `{"type": "context_edit", "targetId", "replacement"}` (`None` omits the
  target from model context; `{"content": ...}` replaces its content) or
  `{"type": "compaction", "summary", "firstKeptEntryId"}` (`None` keeps no
  earlier entries)
- `event["continue"]` — whether a continuation has been requested
- `event["context"]` — a `BoundaryContextPreview` (`context_entries`,
  `context_messages`, `llm_messages`, `pending_messages`, `can_continue`)
  rebuilt from the proposals
- `event["outcome"]` — `"completed"`, `"aborted"` or `"error"`; `turn_end`
  also carries `messageEntryId` and `toolResultEntryIds`

Return `{"entries": ..., "continue": ...}`; an omitted field keeps the current
proposal. Handlers run in extension load and registration order, each seeing
the previous proposals in `event["entries"]` and `event["context"]` rebuilt
from them. The complete proposal is validated, then appended in list order
after all handlers finish; a handler error is reported and later handlers
still run. Validation prevents partially applied semantic errors, but
persistence is not transactional.

`"continue": True` ensures one next provider request for that boundary. If
tool results, steering or a follow-up already cause that request, they
satisfy the decision and no additional request is made; otherwise pidrei
makes one context-only request. Error and aborted responses remain hard
exits, and `"continue": False` never suppresses natural work. Guard the
condition: an unconditional `"continue": True` is evaluated again after the
next response and can loop forever. A `custom_message` draft contributes a
user-role model message but is extension-authored: it does not run input
hooks, slash commands, skills or prompt templates.

#### message_start / message_update / message_end

Fired for message lifecycle updates.

- `message_start` and `message_end` fire for every message, including user,
  assistant, tool-result and custom messages.
- `message_update` fires for assistant streaming updates.
- `message_end` handlers can return `{"message": ...}` to replace the
  finalized message. The replacement must keep the same `role`; one that does
  not is reported and ignored.

Messages are frozen dataclasses, so a replacement is built with
`dataclasses.replace`:

```python
import dataclasses


async def extension(pi):
    async def on_message_update(event, ctx):
        # event["message"]
        # event["assistantMessageEvent"]: the token-by-token stream event
        ...

    async def on_message_end(event, _ctx):
        message = event["message"]
        if message.role != "assistant":
            return None
        cost = dataclasses.replace(message.usage.cost, total=0.123)
        return {"message": dataclasses.replace(message, usage=dataclasses.replace(message.usage, cost=cost))}

    pi.on("message_update", on_message_update)
    pi.on("message_end", on_message_end)
```

#### tool_execution_start / tool_execution_update / tool_execution_end

Fired for tool execution lifecycle updates.

In parallel tool mode:

- `tool_execution_start` is emitted in assistant source order while the calls
  are prepared
- `tool_execution_update` events may interleave across tools
- `tool_execution_end` is emitted in tool completion order after each tool is
  finalized
- the final tool-result message events are still emitted later, in assistant
  source order

```python
async def on_start(event, ctx):
    # event["toolCallId"], event["toolName"], event["args"]
    ...


async def on_update(event, ctx):
    # event["toolCallId"], event["toolName"], event["args"], event["partialResult"]
    ...


async def on_end(event, ctx):
    # event["toolCallId"], event["toolName"], event["result"], event["isError"]
    # event.get("durationMs"): milliseconds execute() took; absent when the tool did not run
    ...
```

All three events, `tool_call` and `tool_result` carry `parentToolCallId` when
another tool made the call through `ctx.execute_tool()`; see
[Calling other tools](#calling-other-tools).

#### context

Fired before each model request. Modify messages non-destructively.

```python
async def on_context(event, ctx):
    # event["messages"]: a deep copy of the conversation without system messages, safe to modify
    return {"messages": [message for message in event["messages"] if not should_prune(message)]}
```

A handler may return `{"messages": [...]}` or edit `event["messages"]` in
place. The prompt and tool declarations belong to pidrei and are not part of
this hook: when the handler changes the list, pidrei replays the current
prompt sections and tool declarations into one leading system message ahead
of the returned messages, so filtering, windowing or slicing from a
compaction summary cannot drop the prompt or the tools. An unchanged list
keeps mid-conversation system messages in place, so models that accept them
keep their cached prefix. System messages a handler adds are kept after
pidrei's head. To change the prompt or the tool set durably, use
[`before_agent_start`](#before_agent_start) or `pi.set_active_tools()`; to
edit system messages for one request, use
[`context_with_system`](#context_with_system).

#### context_with_system

Fired before each model request, after every `context` handler has run and
pidrei has restored the prompt and tool state. `event["messages"]` is the
full transcript, including the leading system message and any
mid-conversation prompt or tool patches (see
[SystemMessage](message-types.md#systemmessage)). The returned messages are
sent as they are: this hook owns the prompt and tool declarations for the
request.

```python
from pidrei_ai.utils.transcript import get_current_system_message


async def on_context_with_system(event, ctx):
    cut = find_cut_index(event["messages"])
    # Fold the dropped prefix so its prompt and tool state survives as the new head.
    head = get_current_system_message(event["messages"][:cut])
    kept = event["messages"][cut:]
    return {"messages": [head, *kept] if head is not None else kept}
```

Keep a system message at index 0: providers read the prompt and initial tool
declarations there, and dropping it is reported as an extension error (the
output is still sent). Removing a system message removes the tool
declarations and section patches it carries. Check the output with
`get_current_system_prompt()` and `get_current_tools()` from
`pidrei_ai.utils.transcript`. Handlers run in extension load order; a
`systemPrompt` forced from `before_agent_start` is still projected onto the
request afterwards.

#### before_provider_headers

Fired after the outgoing HTTP headers are assembled. Use it to add, override
or remove request headers.

Handlers mutate `event["headers"]` in place; the return value is ignored. Set
a key to a string to add or override it, or to `None` to delete it.

```python
async def on_provider_headers(event, ctx):
    # Add or override, e.g. a session id for gateway tracing/attribution
    event["headers"]["x-session-id"] = ctx.session_manager.get_session_id()

    # Drop an attribution header pidrei adds for this call
    event["headers"]["X-OpenRouter-Title"] = None
```

It runs once per provider request; retries reuse the same headers rather than
firing the hook again.

#### before_provider_request

Fired after the provider-specific payload is built, right before the request
is sent. Handlers run in extension load order. Returning `None` keeps the
payload unchanged; returning any other value replaces the payload for later
handlers and for the actual request.

This hook can rewrite the provider-level system instructions or remove them
entirely. Those payload-level changes are not reflected by
`ctx.get_system_prompt()`, which reports pidrei's system prompt string rather
than the final serialized provider payload.

```python
async def on_provider_request(event, ctx):
    # event["payload"]: the provider payload
    # Optional: replace it
    return {**event["payload"], "temperature": 0}
```

This is mainly useful for debugging provider serialization and cache
behaviour; `examples/extensions/provider_payload.py` logs every payload and
response to a file.

The payload is the full request. Providers that continue from a previous
response (OpenAI Codex over WebSocket, Anthropic message threads) may send only
the part that is new since the previous request.

#### after_provider_response

Fired after an HTTP response is received and before its stream body is
consumed. Handlers run in extension load order.

```python
async def on_provider_response(event, ctx):
    # event["status"]: HTTP status code
    # event["headers"]: the response headers, as a dict
    if event["status"] == 429:
        ctx.ui.notify(f"rate limited, retry-after: {event['headers'].get('retry-after')}", "warning")
```

Header availability depends on the provider and transport. Providers that
abstract HTTP responses may not expose headers.

#### provider_stream_event

Fired for each parsed provider stream event before pidrei normalizes it.
Notification-only and not persisted.

```python
async def on_provider_stream_event(event, ctx):
    # event["provider"], event["api"], event["model"] (the model id)
    # event["data"]: the earliest structured value pidrei has (read-only)
    ...
```

`event["data"]` is a parsed SSE payload or stream chunk, not the raw HTTP
bytes. It is the adapter's own object: treat it as read-only, because
mutating it changes what the adapter normalizes. Handlers are awaited in
stream order, so a slow handler delays the stream; handler errors are
reported without changing the provider response.
`examples/extensions/debug_provider.py` is an opt-in viewer that groups the
raw events by assistant message.

#### cache_warming_decision

Fired before each prompt-cache refresh with pidrei's decision filled in. The
event carries only pidrei's cost estimates; use `ctx.model`, `ctx.is_idle()`
and `ctx.get_context_usage()` for everything else.

```python
async def on_cache_warming_decision(event, ctx):
    # event["warmCost"]: price of this refresh
    # event["missCost"]: extra price of the next request if the entry is lost
    # event["continuationProbability"]: pidrei's estimate that a request arrives in time
    # event["action"]: "warm" | "stop", pidrei's decision
    if ctx.model is not None and ctx.model.provider == "my-provider":
        return {"action": "stop"}
    return None
```

Return `{"action": "warm"}` or `{"action": "stop"}` to override; the last
handler that returns an action wins. `"stop"` ends warming until the next
real request. If a handler raises, pidrei keeps its own decision.

### Model events

#### model_select

Fired when the model changes through `/model`, model cycling (`Ctrl+P`), or
`pi.set_model()`. Selecting the model already in use does not fire it, and
neither does restoring a session: read `ctx.model` in `session_start` to
initialize model-dependent state.

```python
async def on_model_select(event, ctx):
    # event["model"]: the newly selected model
    # event["previousModel"]: the previous model (None if first selection)
    # event["source"]: "set" | "cycle"
    previous = event["previousModel"]
    prev = f"{previous.provider}/{previous.id}" if previous is not None else "none"
    model = event["model"]
    ctx.ui.notify(f"Model changed ({event['source']}): {prev} -> {model.provider}/{model.id}", "info")
```

Use it to update UI elements (status bars, footers) or run model-specific
initialization when the active model changes. See
`examples/extensions/model_status.py`.

#### thinking_level_select

Fired when the thinking level changes. Notification-only; handler return
values are ignored.

```python
async def on_thinking_level_select(event, ctx):
    # event["level"]: the newly selected thinking level
    # event["previousLevel"]: the previous thinking level
    ctx.ui.set_status("thinking", f"thinking: {event['level']}")
```

Use it to update extension UI when `pi.set_thinking_level()`, a model change,
or the built-in thinking-level controls change the active level. It is
dispatched without waiting, so on a model change its handlers can run
alongside `model_select`'s.

### Tool events

#### tool_call

Fired after `tool_execution_start`, before the tool executes. **Can block.**
There are no typed event variants: check `event["toolName"]` and read
`event["input"]` as the tool's arguments dict.

When `tool_call` runs, every earlier agent event has been handled, so
`ctx.session_manager` is up to date through the current assistant
tool-calling message. In the default parallel tool execution mode, sibling
tool calls from the same assistant message are prepared one after the other
(`tool_execution_start`, then `tool_call`, for each), then executed
concurrently. A `tool_call` handler is therefore not guaranteed to see
sibling tool results from that message in `ctx.session_manager`, and neither
`tool_call` nor `tool_result` handlers may assume a sibling call or its
result exists yet.

`event["input"]` is mutable. Mutate it in place to patch the tool arguments
before execution:

- Mutations to `event["input"]` affect the actual tool execution
- Later `tool_call` handlers see mutations made by earlier handlers
- No re-validation is performed after your mutation
- The return value controls blocking: `{"block": True, "reason": ..., "terminate": ...}`
  (`reason` and `terminate` optional); the first handler that blocks ends
  the dispatch, and `reason` becomes the call's error result
- `terminate` only applies to a blocked call; the agent stops after the
  current tool batch only when every finalized result in the batch is
  terminating

```python
async def extension(pi):
    async def on_tool_call(event, ctx):
        # event["toolName"]: "bash", "read", "write", "edit", etc.
        # event["toolCallId"]
        # event["input"]: the tool arguments (mutable)
        if event["toolName"] == "bash":
            # event["input"] is {"command": str, "timeout"?: number}
            event["input"]["command"] = f"source ~/.profile\n{event['input']['command']}"
            if "rm -rf" in event["input"]["command"]:
                return {"block": True, "reason": "Dangerous command", "terminate": True}

        if event["toolName"] == "read":
            # event["input"] is {"path": str, "offset"?: int, "limit"?: int}
            ctx.ui.notify(f"Reading: {event['input']['path']}", "info")
        return None

    pi.on("tool_call", on_tool_call)
```

See `examples/extensions/permission_gate.py` and
`examples/extensions/protected_paths.py`.

#### tool_result

Fired after tool execution finishes and before `tool_execution_end` and the
final tool-result message events are emitted. **Can modify the result.**

In parallel tool mode, `tool_result` and `tool_execution_end` may interleave
in tool completion order, while the final tool-result message events are
still emitted later in assistant source order.

`tool_result` handlers chain like middleware:

- Handlers run in extension load order
- Each handler sees the latest result after previous handler changes
- Handlers can return partial patches (`content`, `details`,
  `structuredContent`, `isError` or `usage`); omitted fields keep their
  current values
- Replacing `content` without returning `structuredContent` drops the
  structured content, because it may no longer match

For nested async work inside the handler, pass `ctx.signal` (the cancel token
of the active operation, or `None` when idle) to what you start, so Escape cancels model
calls, subprocesses and other cancel-aware operations the extension started.

```python
from pidrei_ai.types import TextContent


async def extension(pi):
    async def on_tool_result(event, ctx):
        # event["toolName"], event["toolCallId"], event["input"], event.get("parentToolCallId")
        # event["content"], event["details"], event.get("structuredContent"), event["isError"], event["usage"]
        if event["toolName"] != "bash" or event["isError"]:
            return None
        # For the built-in bash tool, event["details"] is a BashToolDetails
        # (pidrei.core.tools): truncation, full_output_path.
        text = "".join(block.text for block in event["content"] if block.type == "text")
        summary = await pi.exec("summarize", [text], cancel=ctx.signal)

        # Modify the result:
        return {"content": [TextContent(text=summary.stdout)], "isError": False}

    pi.on("tool_result", on_tool_result)
```

### User bash events

#### user_bash

Fired when the user runs a `!` or `!!` command. **Can intercept.**

```python
from pidrei.core.tools import create_local_bash_operations


local = create_local_bash_operations()


class ProfileBash:
    """Wraps pidrei's built-in local bash backend."""

    async def exec(self, command, cwd, **options):
        return await local.exec(f"source ~/.profile\n{command}", cwd, **options)


async def on_user_bash(event, ctx):
    # event["command"]: the bash command
    # event["excludeFromContext"]: True for the !! prefix
    # event["cwd"]: the working directory

    # Option 1: provide custom operations (e.g. SSH; see examples/extensions/ssh.py)
    #     return {"operations": remote_bash_ops}

    # Option 2: wrap the built-in local bash backend
    return {"operations": ProfileBash()}

    # Option 3: full replacement, return the result directly
    #     return {"result": {"output": "...", "exitCode": 0, "cancelled": False, "truncated": False}}
```

`operations` is any object with an `async exec(command, cwd, *, on_data,
cancel=None)` method that streams output as bytes to the synchronous
`on_data` callback and returns a result with `exit_code`. A `result` needs
`output` (str), `exitCode` (a number or `None`), `cancelled` and `truncated`
(bools), and optionally `fullOutputPath`.

Returning `None` continues to the next handler, then to local execution if
none handles the event. A valid result stops propagation: `operations` runs
the command through the supplied backend, while `result` records the
completed command without running it. A result that is neither, or a handler
that raises, blocks the command and is reported as an extension error.

### Input events

#### input

Fired when user input is received, after extension commands are checked but
before skill and template expansion. The event sees the raw input text, so
`/skill:foo` and `/template` are not yet expanded.

**Processing order:**

1. Extension commands (`/cmd`) are checked first: if one matches, its handler
   runs and the `input` event is skipped. Messages from
   `pi.send_user_message()` are checked only with
   `{"expandPromptTemplates": True}`.
2. The `input` event fires and can intercept, transform or handle the input.
3. If not handled: skill commands (`/skill:name`) expand to the skill content.
4. If not handled: prompt templates (`/template`) expand to the template
   content.
5. Agent processing begins (`before_agent_start`, etc.).

```python
async def extension(pi):
    async def on_input(event, ctx):
        # event["text"]: the raw input (before skill/template expansion)
        # event["images"]: attached images, if any
        # event["source"]: "interactive" (typed), "rpc" (API) or "extension" (send_user_message)
        # event["streamingBehavior"]: "steer" | "followUp" | None
        #   None when idle, "steer" for mid-stream interrupts,
        #   "followUp" for messages queued until the agent finishes

        # Route by source: skip processing for extension-injected messages
        if event["source"] == "extension":
            return {"action": "continue"}

        # Transform: rewrite the input before expansion
        if event["text"].startswith("?quick "):
            return {"action": "transform", "text": f"Respond briefly: {event['text'][7:]}"}

        # Handle: respond without the model (the extension shows its own feedback)
        if event["text"] == "ping":
            ctx.ui.notify("pong", "info")
            return {"action": "handled"}

        # Intercept skill commands before expansion
        if event["text"].startswith("/skill:"):
            ...  # could transform, handle, or let it pass

        return {"action": "continue"}  # default: pass through to expansion

    pi.on("input", on_input)
```

**Results:**

- `{"action": "continue"}` — pass through unchanged (also what returning
  `None` does)
- `{"action": "transform", "text": ..., "images": ...}` — replace the text
  (and optionally the images), then continue to expansion
- `{"action": "handled"}` — skip the agent entirely; the first handler to
  return it wins

A returned dict without `"action": "transform"` does not change the input.
Transforms chain across handlers: each handler sees the text the previous one
produced. See `examples/extensions/input_transform.py` and
`examples/extensions/input_transform_streaming.py` for
`streamingBehavior`-aware routing.

## ExtensionContext

Every event handler receives `ctx` as its second argument. A tool's `execute`
receives the tool context, which adds `ctx.tools` and `ctx.execute_tool()`
(see [Custom tools](#custom-tools)). Command handlers receive the
[command context](#extensioncommandcontext), which adds session control.
Shortcut handlers receive the plain context.

| Member | Purpose |
|--------|---------|
| `ctx.ui` | UI surface (below) |
| `ctx.mode` | `"tui"`, `"rpc"`, `"json"` or `"print"` |
| `ctx.has_ui` | True in TUI and RPC modes, False in print and JSON modes |
| `ctx.cwd` | Working directory of the session |
| `ctx.is_project_trusted()` | Whether project-local trust is active |
| `ctx.session_manager` | Session entries and metadata |
| `ctx.model_registry` | Models, providers and resolved authentication, plus model calls |
| `ctx.model` / `ctx.thinking_level` | Current model and its effective reasoning level |
| `ctx.scoped_models` | Models scoped to the session; empty when no scoping is configured |
| `ctx.signal` | Cancel token of the active agent run, or `None` when idle |
| `ctx.is_idle()` / `ctx.has_pending_messages()` | Whether the agent is between runs; whether queued messages are waiting |
| `ctx.abort()` / `ctx.shutdown()` | Abort the current operation; request an orderly shutdown |
| `ctx.get_context_usage()` / `ctx.compact(options)` | Context-window usage; start a compaction |
| `ctx.get_system_prompt()` | The current system prompt |

Every member checks that the context is still current: after a reload or a
session replacement, using an old `ctx` raises.

### ctx.ui

UI methods for user interaction; [Custom UI](#custom-ui) has the full
surface. `ctx.ui` offers `notify(text, level)`, `set_status(...)`,
`set_widget(...)`, `select(title, options)`, `confirm(title, message)`,
`input(title, placeholder)`, `editor(title, prefill)` and `theme`.

- The dialogs (`select`/`confirm`/`input`/`editor`/`custom`) open when
  called and return a handle: `await` it for the user's choice, or `None`
  if dismissed. Dropping the handle is fine — the dialog stays open.
- The setters and getters, `paste_to_editor` and `get_editor_text`
  included, are plain synchronous calls: each change is applied whole
  before the call returns, so your next line sees it.
- The theme accessors (`get_all_themes`, `get_theme`, `set_theme`) are
  awaitable: loading a theme reads files.
- `ctx.ui.apply(fn)` runs the synchronous `fn` as one change, so several
  updates show up in the same frame:

  ```python
  ctx.ui.apply(lambda: (ctx.ui.set_status("job", "done"), ctx.ui.set_widget("job", None)))
  ```

  Nothing can be awaited inside it: an `async def` is refused.

To send raw escape sequences to the terminal (desktop notifications via OSC
777/99, for instance), use `ctx.ui.write_terminal(sequence)`, not
`sys.stdout`: while the TUI runs, its output pump is the terminal's only
writer, and a direct write can land in the middle of a frame.
`write_terminal` queues behind the pump; it does nothing without a UI.
Other stdout/stderr output goes through `write_stdout`/`write_stderr` from
`pidrei.core.output_guard`: they queue the text for pidrei's stdio writer
instead of writing on the runtime (`print` and `sys.stdout` end up there too,
but only as a stand-in for code that cannot be changed).

Component code — a custom component's `handle_input`, a custom editor from
`set_editor_component` — is synchronous, so it cannot await a dialog's
answer. Calling the dialog opens it all the same (the next key already goes
to it); to act on the answer, await the handle in work started with
`tui.spawn(...)`. Event and command handlers run on their own coroutines and
can await freely.

### ctx.mode / ctx.has_ui

`ctx.mode` is the current run mode: `"tui"`, `"rpc"`, `"json"` or `"print"`.
Check `ctx.mode == "tui"` before terminal-only features: `custom()`,
component factories, terminal input and direct TUI rendering.

`ctx.has_ui` is True in TUI and RPC modes and False in print (`-p`) and JSON
modes. Check it before the dialogs (`select`, `confirm`, `input`, `editor`)
and before the fire-and-forget calls that work in both TUI and RPC modes
(`notify`, `set_status`, `set_widget`, `set_title`, `set_editor_text`). In
RPC mode the dialogs and those calls go to the RPC client; widgets accept
only string lists, and custom components, terminal input, custom editors and
theme switching are not supported (they do nothing or return a default).

In print and JSON modes `ctx.ui` is a no-op object, so handlers stay safe to
call it unconditionally: the setters do nothing and the dialogs resolve at
once with their dismissed value (`None`, or `False` for `confirm`). A handler
that needs an answer should check `ctx.has_ui` and take a non-interactive
path when it is False.

### ctx.cwd

The session's working directory.

Build project-local config paths with `CONFIG_DIR_NAME` from `pidrei.config`
rather than a hard-coded `.pidrei`, and read them through `tonio.colored.fs`:

```python
from tonio.colored import fs

from pidrei.config import CONFIG_DIR_NAME


async def extension(pi):
    async def on_session_start(_event, ctx):
        project_config = fs.Path(ctx.cwd) / CONFIG_DIR_NAME / "my-extension.json"
        ...

    pi.on("session_start", on_session_start)
```

### ctx.is_project_trusted()

Whether project-local trust is active for the current session. This includes
decisions made for this run only and the CLI trust overrides, not just
decisions saved in the global trust store.

Check it before reading project-local extension configuration that should be
honored only for trusted projects.

### ctx.session_manager

Session state: `get_cwd()`, `get_session_dir()`, `get_session_id()`,
`get_session_file()`, `get_session_name()`, `get_header()`, `get_entries()`,
`get_entry(id)`, `get_branch()`, `get_tree()`, `get_leaf_id()`,
`get_leaf_entry()`, `get_label(id)`, `build_context_entries()` and
`build_session_projection()`. Entries are dicts in the session file's format
(`entry["type"]`, `entry["customType"]`, `entry["data"]`); see
[Sessions](how-pi-works.md#sessions). Write to the session through the `pi`
methods (`append_entry`, `set_label`, `set_session_name`), not through the
manager.

For `tool_call`, this state includes the current assistant message before
the handlers run. When tools run in parallel, it is still not guaranteed to
include the results of sibling tool calls from the same assistant message.

```python
ctx.session_manager.get_entries()  # All entries
ctx.session_manager.get_branch()  # Current branch
ctx.session_manager.build_context_entries()  # Active branch entries with compaction applied
ctx.session_manager.get_leaf_id()  # Current leaf entry id
```

### ctx.model_registry / ctx.model / ctx.thinking_level / ctx.scoped_models

`ctx.model_registry` gives access to models, providers and resolved
authentication. `get_provider(id)` returns the effective pidrei-ai provider,
and `await get_provider_auth(id)` resolves its current API key, headers,
base URL and provider-scoped environment without needing a loaded model.
`find(provider, id)` looks up one model and `get_available()` lists the
models with working credentials. `ctx.model` is the active model, and
`ctx.thinking_level` its current effective reasoning level.

`ctx.scoped_models` is the read-only list of models scoped to the current
session — the set `/scoped-models` shows. It is resolved at session start
from the `--models` flag and the `enabledModels` setting (glob patterns
matched against `provider/id` or a bare model id) and is empty when no
scoping is configured, meaning every available model is usable. Each entry
has `model` and `thinking_level`, which is set only when a pattern pinned it
(`anthropic/*:high`, say) and `None` otherwise. Use it to fill a model picker
that mirrors the built-in one instead of listing the whole catalog with
`ctx.model_registry.get_available()`.

#### Streaming model calls

Use `ctx.model_registry.stream_simple(model, context, options)` for
provider-neutral options such as `reasoning`, `stream()` for API-specific
options, or `await complete(model, context, options)` for just the final
assistant message. All of them use the configured providers and resolve
authentication, including for providers registered with
`pi.register_provider()`. `stream_simple()` and `stream()` return an
`AssistantMessageEventStream`: iterate it with `async for` for response
events and `await stream.result()` for the final message. Setup failures
produce error events and error results. Pass `cancel=ctx.signal` in the
options to stop the call with the agent run.

For non-chat models, `find_of_type()`, `get_models_of_type()`,
`await get_available_of_type()` and `get_model_of_type()` look up image and
classifier models, and `classify()` and `generate_images()` call them; see
[Classifier models](models.md#classifier-models).

### ctx.signal

The cancel token (a `CancelToken` from `pidrei_utils.cancel`) of the active
agent run, or `None` when no run is active. Pass it to nested work started
from a handler so it stops when the run is aborted: `pi.exec(...,
cancel=ctx.signal)`, model calls that take `cancel`, and helpers that accept
a cancel token.

It is usually set during run events such as `tool_call`, `tool_result`,
`message_update` and `turn_end`, and usually `None` in idle or non-run
contexts such as session events, commands and shortcuts fired while pidrei is
idle.

```python
async def on_tool_result(event, ctx):
    result = await pi.exec("./scripts/audit", [event["toolName"]], cancel=ctx.signal, timeout=10_000)
    if result.code == 0:
        return {"details": {"audit": result.stdout}}


pi.on("tool_result", on_tool_result)
```

### ctx.is_idle() / ctx.abort() / ctx.has_pending_messages()

Control flow helpers. `ctx.is_idle()` is False while pidrei is processing an
agent run, an automatic retry, an auto-compaction retry or a queued
continuation, and while a compaction runs. `ctx.abort()` aborts the current
operation. `ctx.has_pending_messages()` reports whether queued steering or
follow-up messages are waiting.

### ctx.shutdown()

Request an orderly shutdown of pidrei.

- **Interactive mode:** deferred until the agent is idle (after every queued
  steering and follow-up message has been processed).
- **RPC mode:** deferred until the next idle state, once the current command
  response is complete.
- **Print and JSON modes:** no effect. The process exits by itself once every
  prompt has been processed.

The `session_shutdown` event reaches every extension before pidrei exits.
`ctx.shutdown()` is available everywhere: event handlers, tools, commands and
shortcuts.

```python
async def on_tool_call(event, ctx):
    if is_fatal(event["input"]):
        ctx.shutdown()


pi.on("tool_call", on_tool_call)
```

### ctx.get_context_usage()

Context usage for the active model, or `None` when the model has no known
context window. It uses the last assistant usage when available and
estimates the tokens of the messages after it. The result has `tokens`,
`context_window` and `percent`; `tokens` and `percent` are `None` after a
compaction, until an assistant has responded.

```python
usage = ctx.get_context_usage()
if usage is not None and usage.tokens is not None and usage.tokens > 100_000:
    ...
```

### ctx.compact(options)

Start a compaction without waiting for it. The options dict takes
`custom_instructions`, and `on_complete` and `on_error` for follow-up work:
plain functions, called with the compaction result or the exception once
the compaction ends.

```python
def on_complete(_result):
    ctx.ui.notify("Compaction completed", "info")


def on_error(error):
    ctx.ui.notify(f"Compaction failed: {error}", "error")


ctx.compact(
    {
        "custom_instructions": "Focus on recent changes",
        "on_complete": on_complete,
        "on_error": on_error,
    }
)
```

See `examples/extensions/trigger_compact.py`.

### ctx.get_system_prompt()

pidrei's current system prompt, as a string.

- During `before_agent_start`, it reflects the system prompt changes made so
  far for the current turn by the handlers that already ran.
- It does not include later `context` message changes.
- It does not include `before_provider_request` payload rewrites.
- Extensions loaded after yours can still change what is finally sent.

```python
async def on_before_agent_start(_event, ctx):
    ctx.ui.notify(f"System prompt length: {len(ctx.get_system_prompt())}", "info")


pi.on("before_agent_start", on_before_agent_start)
```

## ExtensionCommandContext

Command handlers receive a context that extends the
[extension context](#extensioncontext) with session control:
`get_system_prompt_options()`, `wait_for_idle()`, `new_session(options)`,
`fork(entry_id, options)`, `navigate_tree(target_id, options)`,
`switch_session(path, options)` and `reload()`. These are command-only
because calling them from an event handler can deadlock the runtime. Every
method except `get_system_prompt_options()` is awaitable and does nothing
until awaited.

### ctx.get_system_prompt_options()

The base inputs pidrei currently builds the system prompt from, as a
`BuildSystemPromptOptions` (from `pidrei.core.system_prompt`):

```python
options = ctx.get_system_prompt_options()
context_paths = [file.path for file in options.context_files]
```

It has the same shape as `event["systemPromptOptions"]` in
`before_agent_start`: custom or forced prompt, active tools, tool snippets,
per-tool and custom rules, custom sections, appended prompt text, `cwd`,
loaded context files and loaded skills. It can include full context file
contents, so treat it as sensitive extension-local data and keep it out of
command lists, logs and autocomplete metadata.

It reports the current base inputs: it does not include the per-turn
`before_agent_start` changes, later `context` message changes or
`before_provider_request` payload rewrites.

The command gets its own copy (the same one on every call during that
handler) and can change it; the changes are applied when the handler returns.
If pidrei rebuilt its base options while the handler ran, they are dropped.

### ctx.wait_for_idle()

Wait for the agent to settle completely, including automatic retries,
auto-compaction retries and queued continuations:

```python
async def my_cmd(_args, ctx):
    await ctx.wait_for_idle()
    # The agent is idle: safe to change the session


pi.register_command("my-cmd", handler=my_cmd)
```

### ctx.new_session(options)

Create a new session:

```python
from pidrei_ai.types import TextContent, UserMessage
from pidrei_utils import clock


async def handoff(_args, ctx):
    parent_session = ctx.session_manager.get_session_file()
    kickoff = "Continue in the replacement session"

    async def setup(session_manager):
        await session_manager.append_message(
            UserMessage(content=[TextContent(text="Context from previous session...")], timestamp=clock.now_ms())
        )

    async def with_session(new_ctx):
        # Use only the replacement-session context here.
        await new_ctx.send_user_message(kickoff)

    result = await ctx.new_session({"parent_session": parent_session, "setup": setup, "with_session": with_session})
    if result["cancelled"]:
        ...  # An extension cancelled the new session
```

Options:

- `parent_session`: the parent session file to record in the new session's
  header.
- `setup`: an async callable that receives the new session's
  `SessionManager` and can change it before `with_session` runs.
- `with_session`: an async callable that runs post-switch work against a
  fresh replacement-session context. Do not use the captured old `pi` or
  command `ctx` there; see [Session replacement](#session-replacement).

The result is a dict; `result["cancelled"]` is True when a
`session_before_switch` handler cancelled the switch. See
`examples/extensions/handoff.py`.

### ctx.fork(entry_id, options)

Fork from an entry, creating a new session file:

```python
async def with_session(new_ctx):
    # Use only the replacement-session context here.
    new_ctx.ui.notify("Now in the forked session", "info")


result = await ctx.fork("entry-id-123", {"with_session": with_session})
if result["cancelled"]:
    ...  # An extension cancelled the fork

clone_result = await ctx.fork("entry-id-456", {"position": "at"})
if clone_result["cancelled"]:
    ...  # An extension cancelled the clone
```

Options:

- `position`: `"before"` (default) forks before the selected user message
  and restores that prompt into the editor; `"at"` duplicates the active path
  through the selected entry without restoring editor text.
- `with_session`: as for `new_session`.

`result["cancelled"]` is True when a `session_before_fork` handler cancelled
the fork.

### ctx.navigate_tree(target_id, options)

Move to another point of the session tree, in the same session file:

```python
result = await ctx.navigate_tree(
    "entry-id-456",
    {
        "summarize": True,
        "custom_instructions": "Focus on error handling changes",
        "replace_instructions": False,  # True replaces the default prompt entirely
        "label": "review-checkpoint",
    },
)
```

Options:

- `summarize`: whether to summarize the abandoned branch.
- `custom_instructions`: custom instructions for the summarizer.
- `replace_instructions`: when True, `custom_instructions` replaces the
  default prompt instead of being appended to it.
- `label`: a label for the branch summary entry, or for the target entry
  when not summarizing.

`result["cancelled"]` is True when a `session_before_tree` handler cancelled
the navigation. The call raises, leaving the active branch unchanged, while
an agent response, a manual or automatic compaction, or another tree
navigation is running, even with `summarize` set to False: wait for it to
finish (with `await ctx.wait_for_idle()`, for instance) and retry. It also
raises when `summarize` is set and no model is available, and when the
target entry does not exist.

### ctx.switch_session(session_path, options)

Switch to another session file:

```python
async def with_session(new_ctx):
    await new_ctx.send_user_message("Resume work in the replacement session")


result = await ctx.switch_session("/path/to/session.jsonl", {"with_session": with_session})
if result["cancelled"]:
    ...  # An extension cancelled the switch via session_before_switch
```

Options:

- `with_session`: as for `new_session`.

To find sessions, use the static `SessionManager.list(cwd)` and
`SessionManager.list_all()` (both awaitable) from
`pidrei.core.session_manager`. Each entry has `path`, `id`, `cwd`, `name`,
`first_message`, `created` and `modified`, among others:

```python
from pidrei.core.session_manager import SessionManager


async def extension(pi):
    async def switch(_args, ctx):
        sessions = await SessionManager.list(ctx.cwd)
        if not sessions:
            return
        choice = await ctx.ui.select("Pick session:", [session.path for session in sessions])
        if choice is None:
            return

        async def with_session(new_ctx):
            new_ctx.ui.notify("Switched session", "info")

        await ctx.switch_session(choice, {"with_session": with_session})

    pi.register_command("switch", handler=switch, description="Switch to another session")
```

### Session replacement

`new_session`, `fork` and `switch_session` replace the session. The
`with_session` callback receives a fresh replacement context: a command
context for the new session, plus `send_message()` and `send_user_message()`
bound to it. Unlike the `pi` methods of the same name, these two are async:
await them.

Lifecycle and pitfalls:

- `with_session` runs only after the old session has emitted
  `session_shutdown`, the old runtime has been torn down, the replacement
  session is in place, and the new extension instance has received
  `session_start`.
- The callback still runs in your old closure, not in the new extension
  instance, so your old instance may already have run its shutdown cleanup.
- The captured old `pi` and the old command `ctx` are stale after the
  replacement and raise if used. Use only the context passed to
  `with_session` for session-bound work.
- Objects you took out of them earlier are still your responsibility: if you
  kept `sm = ctx.session_manager` before the replacement, `sm` is still the
  old session's manager. Do not reuse it afterwards.
- Assume anything your `session_shutdown` handler tears down is already gone.
  Capture only plain data that survives shutdown, such as strings, ids and
  serialized config.

Safe:

```python
async def handoff(_args, ctx):
    kickoff = "Continue from the replacement session"

    async def with_session(new_ctx):
        await new_ctx.send_user_message(kickoff)

    await ctx.new_session({"with_session": with_session})


pi.register_command("handoff", handler=handoff)
```

Unsafe:

```python
async def handoff(_args, ctx):
    old_session_manager = ctx.session_manager

    async def with_session(_new_ctx):
        # Stale old objects: do not do this
        old_session_manager.get_session_file()
        pi.send_user_message("wrong")  # raises: the old pi is stale

    await ctx.new_session({"with_session": with_session})
```

### ctx.reload()

Run the same reload as `/reload`.

```python
async def reload_runtime(_args, ctx):
    await ctx.reload()
    return


pi.register_command(
    "reload-runtime",
    handler=reload_runtime,
    description="Reload extensions, skills, prompts, themes, and context files",
)
```

How it behaves:

- `await ctx.reload()` emits `session_shutdown` for the current extension
  runtime.
- It then reloads the resources and emits `session_start` with
  `"reason": "reload"` and `resources_discover` with reason `"reload"`.
- The running command handler continues in the old call frame: code after
  `await ctx.reload()` still runs from the pre-reload version of the module.
- That code must not assume the old in-memory extension state is still valid,
  and the old `pi` and `ctx` raise if used.
- Once the handler returns, later commands, events and tool calls use the new
  extension version.

For predictable behavior, treat reload as terminal for the handler
(`await ctx.reload()`, then `return`).

pidrei refuses to reload while the agent is responding or compacting, and
tools receive the tool context, not the command context, so they cannot call
`ctx.reload()`. To let the model trigger a reload, have the tool record the
request and send the command from `agent_settled`: a message sent there is
processed after every settled handler has finished, with the agent idle.

```python
from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent


async def extension(pi):
    state = {"reload_requested": False}

    async def reload_runtime(_tool_call_id, _params, _cancel, _on_update, _ctx):
        state["reload_requested"] = True
        return AgentToolResult(
            content=[TextContent(text="Reload requested. pidrei reloads after this response.")],
            details={},
            terminate=True,
        )

    pi.register_tool(
        ToolDefinition(
            name="reload_runtime",
            label="Reload Runtime",
            description="Reload extensions, skills, prompts, themes, and context files",
            parameters={"type": "object", "properties": {}},
            execute=reload_runtime,
        )
    )

    async def on_agent_settled(_event, _ctx):
        if not state["reload_requested"]:
            return
        state["reload_requested"] = False
        pi.send_user_message("/reload-runtime", {"expandPromptTemplates": True})

    pi.on("agent_settled", on_agent_settled)
```

`expandPromptTemplates: True` dispatches the text as a command, at once, even
while the agent is running; without it the text reaches the model as a plain
user message.

## The `pi` object

The factory's argument. Registration methods record what the extension
provides; action methods act on the session.

### Registration

| Method | Purpose |
|--------|---------|
| `on(event, handler)` | Subscribe to an event; returns an unsubscribe function |
| `register_tool(tool)` | Add a tool the model can call |
| `register_tools(tools, *, update_active=None)` | Add several tools in one step: the session sees all of them or none. `update_active` is an `update_active_tools()` callback applied in the same step |
| `register_command(name, *, handler, description=None, get_argument_completions=None)` | Add a slash command |
| `register_shortcut(shortcut, *, handler, description=None)` | Bind a key |
| `register_flag(name, *, type, description=None, default=None)` | Add a CLI flag |
| `get_flag(name)` | Read a registered flag's value |
| `register_message_renderer(custom_type, renderer)` | Render a custom message |
| `register_markdown_transformer(transformer)` | Rewrite Markdown before pidrei renders it |
| `register_entry_renderer(custom_type, renderer)` | Render a custom session entry |
| `register_tool_renderer(resolver)` | Choose how tool calls are drawn |
| `register_provider(...)` / `unregister_provider(name)` | Add, override or remove a model provider |
| `register_virtual_model(model)` / `unregister_virtual_model(provider, id)` | Add or remove a routing model |
| `register_mcp_server(name, config)` / `unregister_mcp_server(name)` / `get_mcp_servers()` | MCP servers for the session |

### Actions

| Method | Purpose |
|--------|---------|
| `send_message(message, options=None)` | Queue a message (custom types allowed) |
| `send_user_message(content, options=None)` | Queue a user message |
| `await append_entry(custom_type, data=None)` | Append a custom session entry |
| `await set_session_name(name)` / `get_session_name()` | Session name |
| `await set_label(entry_id, label)` | Label a session entry |
| `await exec(command, args, *, cwd=None, cancel=None, timeout=None, max_buffer=...)` | Run a subprocess |
| `get_active_tools()` / `set_active_tools(names)` | The active tool set: the tools declared to the model |
| `update_active_tools(update)` | Change the active tool set based on its current value, in one step |
| `get_all_tools()` | Every registered tool, with its `exposure`, `namespace`, `annotations` and `reaches_tools` |
| `get_commands()` | Every command available to `send_user_message` |
| `await set_model(model)` | Switch the session's model; `False` when the provider has no auth configured |
| `get_thinking_level()` / `await set_thinking_level(level)` | Reasoning level, clamped to the model |
| `get_settings()` | A copy of the effective settings |
| `events` | Shared bus for extension-to-extension messages |

The rows written with `await` return an awaitable and do nothing until you
await it. Forgetting the `await` is silent — the call appears to succeed and
the work never happens — so reach for it whenever the table shows it.
`append_entry`, `set_session_name`, `set_label`, `set_model` and
`set_thinking_level` write to the session file, and awaiting them is what
guarantees the write landed before the next statement runs.

Actions need the session, so most of them raise when called from the factory
itself ("Extension runtime not initialized. Action methods cannot be called
during extension loading."); call them from handlers. While the factory runs,
`get_active_tools()`, `get_all_tools()` and `get_commands()` return empty
lists. Registration methods and `exec` work in the factory.

### pi.on(event, handler)

Subscribe to an event; see [Events](#events) for the event types and return
values. The handler is an `async def` receiving `(event, ctx)`. `on` returns
an unsubscribe function that removes only that registration (registering the
same handler twice gives two independent registrations):

```python
async def on_agent_end(event, _ctx):
    unsubscribe()
    await update_integration(event["messages"])


unsubscribe = pi.on("agent_end", on_agent_end)
```

Handlers run in extension load order, then in registration order within each
extension. Adding or removing a handler does not affect a dispatch already in
progress.

### pi.register_tool(definition)

Register a tool the model can call; see [Custom tools](#custom-tools) for the
details.

`pi.register_tool()` works during extension load and after startup: from
`session_start`, a command handler or any other handler. New tools take
effect in the same session at once and appear in `pi.get_all_tools()`
without `/reload`; a tool with a registered name replaces it. Registering a
tool activates it unless it sets `default_active=False` or an
[exposure](#tool-exposure) other than `"direct"` or `"model-only"`. To
register several tools, use `pi.register_tools(tools, update_active=...)`:
the session sees all of them or none, and `update_active` (an
`update_active_tools()` callback) is applied in the same step.

Use `pi.set_active_tools()` or `pi.update_active_tools()` to enable or
disable tools, dynamically registered ones included, at runtime.

Set `prompt_snippet` to give a tool a one-line entry in the `<tools>` section
of the default system prompt, and `prompt_guidelines` to append bullets to
its `<rules>` section while the tool is active.

**Important:** `prompt_guidelines` bullets are appended flat to the `<rules>`
section, with no tool name in front. Each guideline must name the tool it is
about: not "Use this tool when...", since the model cannot tell which tool
"this" is, but "Use my_tool when...".

```python
from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent


async def execute(tool_call_id, params, cancel, on_update, ctx):
    # Stream progress; partial results also need details
    if on_update is not None:
        on_update(AgentToolResult(content=[TextContent(text="Working...")], details={"result": "pending"}))
    return AgentToolResult(content=[TextContent(text="Done")], details={"result": "..."})


pi.register_tool(
    ToolDefinition(
        name="my_tool",
        label="My Tool",
        description="What this tool does",
        prompt_snippet="Summarize or transform text according to action",
        prompt_guidelines=["Use my_tool when the user asks to summarize previously generated text."],
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "add"]},
                "text": {"type": "string"},
            },
            "required": ["action"],
        },
        execute=execute,
        # Optional: custom rendering
        # render_call=render_call,
        # render_result=render_result,
    )
)
```

See `examples/extensions/dynamic_tools.py` for a full example.

### pi.send_message(message, options)

Inject a custom message into the session. Custom messages are part of the
model context. For durable content shown in the TUI but never sent to the
model, use [`pi.append_entry()`](#piappend_entrycustom_type-data) with
[`pi.register_entry_renderer()`](#piregister_entry_renderercustom_type-renderer).

```python
pi.send_message(
    {
        "customType": "my-extension",
        "content": "Message text",
        "display": True,
        "details": {...},
    },
    {"triggerTurn": True, "deliverAs": "steer"},
)
```

Options:

- `deliverAs`, the delivery mode:
  - `"steer"` (default): while the agent runs, the message is queued and
    delivered after the current assistant turn finishes executing its tool
    calls, before the next model call.
  - `"followUp"`: waits for the agent to finish; delivered only when the
    agent has no more tool calls.
  - `"nextTurn"`: queued for the next user prompt. Interrupts and triggers
    nothing.
- `triggerTurn: True`: when the agent is idle, start a model response at
  once. Applies only to `"steer"` and `"followUp"` (ignored for
  `"nextTurn"`). Without it, an idle session stores the message without
  starting a turn.
- `triggerTurn: False` while the agent runs: the message is neither steered
  nor queued as a follow-up; it is appended once the current turn ends.

`send_message` decides the delivery before it returns and does the rest in
the background; errors are reported as extension errors.

### pi.send_user_message(content, options)

Send a user message to the agent. Unlike `send_message()`, which sends custom
messages, this sends a real user message, as if the user had typed it. It
starts or queues a turn unless an `input` handler handles it or, with
`expandPromptTemplates: True`, it runs an extension command.

```python
from pidrei_ai.types import ImageContent, TextContent

# Simple text message
pi.send_user_message("What is 2+2?")

# With a content list (text + images)
pi.send_user_message(
    [
        TextContent(text="Describe this image:"),
        ImageContent(data="...", mime_type="image/png"),
    ]
)

# While the agent runs - a delivery mode is required
pi.send_user_message("Focus on error handling", {"deliverAs": "steer"})
pi.send_user_message("And then summarize", {"deliverAs": "followUp"})

# Opt in to extension command dispatch and skill/prompt template expansion
pi.send_user_message("/review src/index.py", {"expandPromptTemplates": True})
```

Options:

- `deliverAs`, required while the agent runs:
  - `"steer"`: queued for delivery after the current assistant turn finishes
    executing its tool calls.
  - `"followUp"`: waits for the agent to finish all tools.
- `expandPromptTemplates`: dispatch extension commands and expand skill
  commands and prompt templates. Defaults to `False`. An extension command
  runs at once, even while the agent runs.

When the agent is idle, the message is sent at once and starts a new turn.
While it runs, a message without `deliverAs` is rejected and reported as an
extension error. The call returns immediately and the message is processed in
the background. See `examples/extensions/send_user_message.py`.

### pi.append_entry(custom_type, data)

Persist extension data in the session. Custom entries are not part of the
model context. In interactive mode they can also appear in the transcript,
drawn by a renderer registered with `pi.register_entry_renderer()`.

```python
await pi.append_entry("my-state", {"count": 42})
await pi.append_entry("status-card", {"title": "Indexed files", "count": 17})


# Restore on reload
async def on_session_start(_event, ctx):
    for entry in ctx.session_manager.get_entries():
        if entry["type"] == "custom" and entry["customType"] == "my-state":
            ...  # Rebuild from entry["data"]


pi.on("session_start", on_session_start)
```

### pi.set_session_name(name) / pi.get_session_name()

`await pi.set_session_name(name)` sets the session's display name, shown in
the session selector instead of the first message, and emits
`session_info_changed`. `pi.get_session_name()` returns the current name, or
`None` when none is set.

```python
await pi.set_session_name("Refactor auth module")

name = pi.get_session_name()
if name:
    ctx.ui.notify(f"Session: {name}", "info")
```

### pi.set_label(entry_id, label)

Set or clear the label of an entry. Labels are user-defined markers for
bookmarking and navigation, shown in the `/tree` selector. Labeling an entry
that does not exist raises.

```python
# Set a label
await pi.set_label(entry_id, "checkpoint-before-refactor")

# Clear a label
await pi.set_label(entry_id, None)

# Read labels through the session manager
label = ctx.session_manager.get_label(entry_id)
```

Labels are stored in the session and survive restarts. Use them to mark
important points (turns, checkpoints) in the conversation tree.

### pi.register_command(name, *, handler, description, get_argument_completions)

Register a slash command. The handler is an `async def` receiving
`(args, ctx)`: the text after the command name, and the
[command context](#extensioncommandcontext).

```python
async def stats(_args, ctx):
    count = len(ctx.session_manager.get_entries())
    ctx.ui.notify(f"{count} entries", "info")


pi.register_command("stats", handler=stats, description="Show session statistics")
```

When several extensions register the same name, pidrei keeps them all and
numbers them in load order, for example `/review:1` and `/review:2`.

`get_argument_completions` offers completions after the command name. It is
an `async def` receiving the text typed so far and returning a list of items
(`{"value": ..., "label": ...}`), or `None` for none:

```python
async def deploy_completions(prefix):
    items = [{"value": env, "label": env} for env in ("dev", "staging", "prod")]
    return [item for item in items if item["value"].startswith(prefix)] or None


async def deploy(args, ctx):
    ctx.ui.notify(f"Deploying: {args}", "info")


pi.register_command(
    "deploy",
    handler=deploy,
    description="Deploy to an environment",
    get_argument_completions=deploy_completions,
)
```

See `examples/extensions/commands.py`.

### pi.get_commands()

The slash commands that `send_user_message(..., {"expandPromptTemplates":
True})` can invoke in the current session: extension commands, prompt
templates and skill commands, in that order. Skill commands are named
`skill:<name>`.

```python
commands = pi.get_commands()
from_extensions = [command for command in commands if command.source == "extension"]
user_scoped = [command for command in commands if command.source_info.scope == "user"]
```

Each entry has:

| Attribute | Value |
|-----------|-------|
| `name` | The invocable name without the leading slash; may carry a suffix such as `review:1` |
| `description` | The description, or `None` |
| `source` | `"extension"`, `"prompt"` or `"skill"` |
| `source_info` | `path`, `source`, `scope` (`"user"`, `"project"` or `"temporary"`), `origin` (`"package"` or `"top-level"`) and `base_dir` |

Use `source_info` as the provenance of a command. Do not infer ownership from
command names or by parsing paths.

Built-in interactive commands such as `/model` and `/settings` are not
included: they exist only in interactive mode and would not run if sent with
`send_user_message`.

### pi.register_message_renderer(custom_type, renderer)

Register a TUI renderer for custom messages of your `custom_type`. Custom
messages are created with `pi.send_message()` and are part of the model
context. See [Custom UI](#custom-ui) and
`examples/extensions/message_renderer.py`.

### pi.register_markdown_transformer(transformer)

Rewrite the Markdown of normal user text, assistant text, and thinking blocks
before pidrei renders it. Transformers run in extension load order and each
one receives the Markdown the previous one returned; the built-in renderer
then renders the result. Each extension has one transformer: registering
again replaces it.

The transformer is called with the Markdown string and a context dict:

- `messageType` — `"user"`, `"assistant"`, or `"assistant-thinking"`
- `isStreaming` — `True` for partial assistant updates; `False` for user,
  finalized assistant, and restored messages
- `availableWidth` — the exact terminal columns available for the content
- `messageId`, `timestamp` — the session entry's id and ISO timestamp when
  rendering from stored session entries; `None` for live messages

It returns the transformed Markdown:

```python
def transformer(markdown, context):
    if context["isStreaming"] or context["messageType"] == "assistant-thinking":
        return markdown
    return markdown.replace("-->", "→")


pi.register_markdown_transformer(transformer)
```

If a transformer raises, pidrei keeps the Markdown produced so far and
continues with the next one. The hook is display-only: the session and the
model context keep the original message. It runs for new user messages,
assistant streaming updates, restored session messages, and terminal width
changes — so unlike every other extension callback it is **synchronous**, and
it must stay inexpensive.

`pidrei_tui.lex_markdown(text)` exposes the bundled Markdown lexer if a
transformer needs to inspect the token stream.

### pi.register_entry_renderer(custom_type, renderer)

Register a TUI renderer for custom entries of your `custom_type`. Custom
entries are created with `pi.append_entry()` and are not part of the model
context. The renderer is a plain function receiving the entry dict, an
options dict (`options["expanded"]`) and the theme, and returns a component:

```python
import json

from pidrei_tui import Box, Text


def render_status_card(entry, options, theme):
    data = entry["data"]
    box = Box(1, 1, lambda text: theme.bg("customMessageBg", text))
    box.add_child(Text(f"{theme.bold(data['title'])}: {data['count']}", 0, 0))
    if options.get("expanded"):
        box.add_child(Text(theme.fg("dim", json.dumps(data, indent=2)), 0, 0))
    return box


pi.register_entry_renderer("status-card", render_status_card)

await pi.append_entry("status-card", {"title": "Indexed files", "count": 17})
```

See `examples/extensions/entry_renderer.py`.

### pi.register_shortcut(shortcut, *, handler, description)

Register a keyboard shortcut. The handler is an `async def` receiving `ctx`
(the plain context, without the command methods).

```python
async def toggle_plan_mode(ctx):
    ctx.ui.notify("Toggled!", "info")


pi.register_shortcut("ctrl+shift+p", handler=toggle_plan_mode, description="Toggle plan mode")
```

Shortcut strings are `ctrl+`, `alt+`, `shift+` prefixes plus a key. A
shortcut that collides with a reserved binding is refused with a diagnostic;
see [keybindings.md](keybindings.md#extension-shortcuts) for the format, the
built-in bindings and the reserved set.

### pi.register_flag(name, *, type, description, default)

Register a CLI flag. `type` is `"boolean"` or `"string"`, and a `default`
must match it.

```python
pi.register_flag("plan", type="boolean", description="Start in plan mode", default=False)


# Read the value once loading is done; in the factory, get_flag() returns the default
async def on_session_start(_event, _ctx):
    if pi.get_flag("plan"):
        ...  # Plan mode enabled


pi.on("session_start", on_session_start)
```

The flag becomes `--plan` on the command line. A boolean flag is True when
given; a string flag takes a value (`--name value` or `--name=value`).
`pi.get_flag(name)` reads only flags this extension registered, and returns
`None` for others.

### pi.exec(command, args, *, cwd, cancel, timeout, max_buffer)

Run a command without a shell. The command starts only when the call is
awaited.

```python
result = await pi.exec("git", ["status"], cancel=ctx.signal, timeout=5000)
# result.stdout, result.stderr, result.code, result.killed, result.truncated
```

Options: `cancel` (a cancel token) and `timeout` (in milliseconds) terminate
the command, `cwd` sets the working directory (the session's by default), and
`max_buffer` limits stdout and stderr combined, in bytes (default 256 MiB).
When the output exceeds `max_buffer`, pidrei terminates the command and
returns the output up to the limit with `truncated` and `killed` True and a
nonzero `code`. `killed` is True whenever pidrei terminated the command:
timeout, cancellation or `max_buffer`. A command that ignores SIGTERM gets
SIGKILL after 5 seconds. A command that cannot be started resolves with
`code` 1 and the error in `stderr`; only an invalid `max_buffer` raises.

### pi.get_active_tools() / pi.get_all_tools() / pi.set_active_tools(names) / pi.update_active_tools(update)

Manage the active tools, built-in and dynamically registered alike.
`pi.get_active_tools()` returns the active tool names as a list of strings;
`pi.get_all_tools()` returns metadata for every configured tool.

```python
active = pi.get_active_tools()  # ["read", "bash", ...]
all_tools = pi.get_all_tools()
# all_tools[0].name == "read", .description, .parameters,
# .prompt_guidelines == ["Use read to examine files instead of cat or sed."],
# .exposure == "direct",
# .source_info: path "builtin:read", source "builtin", scope "temporary", origin "top-level"

# Tools shipped with pidrei: core tools and tools of built-in extensions such as builtin:mcp
shipped = [tool for tool in all_tools if tool.source_info.source == "builtin"]
# Tools from installed or local extensions
from_extensions = [tool for tool in all_tools if tool.source_info.source not in ("builtin", "sdk")]

# Keep the current tools and enable my_custom_tool
pi.update_active_tools(lambda names: None if "my_custom_tool" in names else [*names, "my_custom_tool"])
pi.set_active_tools(["read", "bash"])  # Only read and bash
```

Each `get_all_tools()` entry has `name`, `description`, `parameters`,
`prompt_guidelines`, `exposure` and `source_info`, plus `namespace`,
`annotations` and `reaches_tools`, which are `None` when the tool does not
set them.

Typical `source_info` values:

- `source == "builtin"` for tools shipped with pidrei: core tools (`path`
  `"builtin:<tool>"`) and tools registered by built-in extensions such as
  `builtin:mcp` (`path` `"builtin:<extension>"`).
- `source == "sdk"` for tools passed as `custom_tools` to
  `create_agent_session()` (see [Library use](sdk.md)).
- The extension's own source metadata for tools registered by extensions.

`pi.set_active_tools()` ignores unknown and `"hidden"` tools. Tools with
`"codemode"` or `"deferred"` exposure stay callable from codemode scripts
whether active or not. The change is appended to the transcript as a system
message before the model's next request; a model that cannot take system
messages mid-conversation gets them folded into the leading one instead,
which can invalidate the cached prefix.

To add or remove tools relative to the current set, use
`pi.update_active_tools(update)` rather than
`pi.set_active_tools([*pi.get_active_tools(), ...])`. Tools, handlers and tool
registrations run in parallel, so another change can land between
`get_active_tools()` and `set_active_tools()`, and the write would undo it.
`update` receives the active tool names and returns the new ones, or `None`
to leave them as they are; no other change lands in between. It runs
synchronously while the session's tool loadout is locked, so keep it quick and
never block in it.

### pi.set_model(model)

Set the model for the current session. The change is recorded in the session
and restored when that session is resumed, but it does not change the
configured `defaultProvider` or `defaultModel` that new sessions use.
Resolves to `False` when no authentication is configured for the model's
provider, and emits `model_select` on success. See [models.md](models.md) for
configuring custom models.

```python
model = ctx.model_registry.find("anthropic", "claude-sonnet-4-5")
if model is not None:
    if not await pi.set_model(model):
        ctx.ui.notify("No API key for this model", "error")
```

### pi.get_thinking_level() / pi.set_thinking_level(level)

`pi.get_thinking_level()` returns the current thinking level: `"off"`,
`"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"` or `"max"`. Levels are
clamped to the model's capabilities (non-reasoning models always use
`"off"`), and a change emits `thinking_level_select`.

`await pi.set_thinking_level(level)` changes the level for the current
session. The change is recorded in the session and restored when that
session is resumed, but it does not change the configured default that new
sessions use.

```python
current = pi.get_thinking_level()
await pi.set_thinking_level("high")
```

### pi.events

A shared event bus for communication between extensions. Handlers are
`async def` functions receiving the emitted data; `emit` starts each of them
in the background and returns at once, and a handler's error is written to
stderr. `on` returns an unsubscribe function, and subscriptions end with the
extension runtime (on reload or session replacement).

```python
async def on_my_event(data): ...


pi.events.on("my:event", on_my_event)
pi.events.emit("my:event", {...})
```

See `examples/extensions/event_bus.py`.

### pi.register_provider(name, config) / pi.register_provider(provider)

Register or override a model provider: for proxies, custom endpoints, or
team-wide model configurations. [custom-provider.md](custom-provider.md)
covers both forms in full: complete providers built with `create_provider`,
model refresh, authentication, custom streaming, and the model definition
reference.

Registrations made while extensions are still loading are queued and applied
once the model registry exists, so an extension can register from its
factory without deferring to an event, and startup model selection sees the
provider. Later calls — from a command handler that ends a setup flow, say —
take effect immediately, without `/reload`.

`pi.register_provider(provider)` registers a complete pidrei-ai `Provider`,
for native auth, filtering, model refresh or stream behavior; it becomes the
base, and `models.json` overrides still apply above it. A provider with a
live catalog implements model refresh and receives a context whose `cancel`
token it must pass to blocking I/O; see
[Model refresh](custom-provider.md#model-refresh).

`pi.register_provider(name, config)` takes a camelCase config dict:

```python
from pidrei.core.provider_composer import ExtensionOAuthConfig
from pidrei_utils import clock

# A new provider with its own models
pi.register_provider(
    "my-proxy",
    {
        "name": "My Proxy",
        "baseUrl": "https://proxy.example.com",
        "apiKey": "$PROXY_API_KEY",  # environment variable reference
        "api": "anthropic-messages",
        "models": [
            {
                "id": "claude-sonnet-4-20250514",
                "name": "Claude 4 Sonnet (proxy)",
                "reasoning": False,
                "input": ["text", "image"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 200000,
                "maxTokens": 16384,
            }
        ],
    },
)

# Override the baseUrl of an existing provider (keeps all its models)
pi.register_provider("anthropic", {"baseUrl": "https://proxy.example.com"})


# A provider with OAuth support for /login
async def login(callbacks):
    # Custom OAuth flow
    callbacks.on_auth({"url": "https://sso.corp.com/..."})
    code = await callbacks.on_prompt({"message": "Enter code:"})
    return {"refresh": code, "access": code, "expires": clock.now_ms() + 3_600_000}


async def refresh_token(credential, cancel):
    # Refresh logic; pass `cancel` to network requests
    return credential


pi.register_provider(
    "corporate-ai",
    {
        "baseUrl": "https://ai.corp.com",
        "api": "openai-responses",
        "models": [...],
        "oauth": ExtensionOAuthConfig(
            name="Corporate AI (SSO)",
            login=login,
            refresh_token=refresh_token,
            get_api_key=lambda credential: credential.access,
        ),
    },
)
```

Config keys:

- `name`: display name for the provider in the UI, such as `/login`.
- `baseUrl`: API endpoint. Every model needs one, from its own definition or
  from the provider.
- `apiKey`: an API key literal, an environment reference (`$ENV_VAR` or
  `${ENV_VAR}`), or a leading `!command`. Optional: without it, users store a
  key with `/login`. `$$` escapes `$`, and `$!` escapes a literal `!` without
  running a command.
- `api`: the API type: `"anthropic-messages"`, `"openai-completions"`,
  `"openai-responses"`, and so on.
- `headers`: custom headers to include in requests.
- `authHeader`: when True, adds an `Authorization: Bearer` header
  automatically.
- `models`: model definitions. When given, they replace all the provider's
  existing models. A model definition can set `baseUrl` to override the
  provider's endpoint for that model.
- `refreshModels`: an async callable for dynamic discovery; the models it
  returns replace the models the registration provided.
- `oauth`: an `ExtensionOAuthConfig` for `/login` support; the provider then
  appears in the login menu. `login(callbacks)` is async and returns the
  credential, `refresh_token(credential, cancel)` is async, and
  `get_api_key(credential)` returns the key for a request.
- `streamSimple`: a custom streaming implementation for non-standard APIs;
  it requires `api`.
- `images` and `classifiers`: implementations for image and classifier
  models, keyed by their `api`.

### pi.unregister_provider(name)

Remove a registered provider and its models. Built-in models the provider
overrode are restored. Nothing happens if the provider was not registered.

Like `register_provider`, it takes effect immediately when called after
loading, without `/reload`.

```python
async def teardown(_args, _ctx):
    pi.unregister_provider("my-proxy")


pi.register_command("my-setup-teardown", handler=teardown, description="Remove the custom proxy provider")
```

### pi.register_virtual_model(model) / pi.unregister_virtual_model(provider, id)

Register a selectable catalog entry, an `ExtensionVirtualModel` (from
`pidrei.core.extensions`), that routes each request to a physical model.
`ctx.model` and the session's model selection name the virtual model;
assistant messages record the physical model and thinking level the router
picked. Registering the same provider and id again replaces the virtual
model. See [virtual-models.md](virtual-models.md).

### pi.register_mcp_server(name, config) / pi.unregister_mcp_server(name) / pi.get_mcp_servers()

Add, remove and list MCP servers for the current session; see
[MCP servers](#mcp-servers).

### pi.register_tool_renderer(resolver)

Choose how calls to any tool are drawn, including tools that are not
registered yet, such as MCP tools in a resumed session before their server
connected. The resolver is a plain function `resolver(tool_name, next)`
returning a `ToolRenderers` (from `pidrei.core.extensions.types`) or `None`;
`next()` returns what the remaining resolvers (in extension load order), then
the registered tool, would use, so returning `next()` when it is not `None`,
and yours otherwise, only fills in. See [Tool rendering](#tool-rendering).

```python
from pidrei.core.extensions.types import ToolRenderers
from pidrei_tui import Text


def resolver(tool_name, next):
    if not tool_name.startswith("mcp__"):
        return next()
    fallback = ToolRenderers(render_call=lambda args, theme, context: Text(theme.fg("toolTitle", tool_name), 0, 0))
    return next() or fallback


pi.register_tool_renderer(resolver)
```

### pi.get_settings()

A copy of the effective settings, with global and project settings merged and
overrides applied.

## State management

Because a reload re-executes the module, module-level globals reset. Keep state
in the factory closure:

```python
async def extension(pi):
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


async def extension(pi):
    MyExtension(pi).wire()
```

Session-bound state belongs in the session itself; pick the store by how it
relates to the conversation:

| State | Store |
|-------|-------|
| Tool state that should follow the active branch | The tool result's `details` |
| Durable data kept out of model context | `await pi.append_entry(...)` |
| Custom content stored *and* sent to the model | `pi.send_message(...)` |
| Data beyond one session | A file under `ctx.cwd` or the agent directory |

Tool state stored in `details` branches with the conversation: rebuild it from
the active branch when the session starts and when `/tree` moves to another
branch.

```python
import threading

from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent


async def extension(pi):
    state = {"items": []}
    # Tool calls run in parallel with each other and with event handlers.
    guard = threading.Lock()

    # Reconstruct state from the active branch
    def reconstruct(ctx):
        items = []
        for entry in ctx.session_manager.get_branch():
            if entry["type"] != "message":
                continue
            message = entry["message"]
            if message.role == "toolResult" and message.tool_name == "my_tool":
                items = list((message.details or {}).get("items", []))
        with guard:
            state["items"] = items

    async def on_branch(_event, ctx):
        reconstruct(ctx)

    pi.on("session_start", on_branch)
    pi.on("session_tree", on_branch)  # /tree changes the branch

    async def execute(_tool_call_id, _params, _cancel, _on_update, _ctx):
        with guard:
            state["items"].append("new item")
            items = list(state["items"])
        return AgentToolResult(
            content=[TextContent(text="Added")],
            details={"items": items},  # Stored for reconstruction
        )

    pi.register_tool(
        ToolDefinition(
            name="my_tool",
            label="My Tool",
            description="Add an item",
            parameters={"type": "object", "properties": {}},
            execute=execute,
        )
    )
```

Rebuild from `ctx.session_manager.get_branch()`, not from every entry in the
file: abandoned branches are alternative histories. `get_branch()` returns the
entries as dicts; a `"message"` entry's `entry["message"]` is the message
object (`role`, `tool_name`, `details`, ...).

Store `details` as plain JSON data (dicts, lists, strings, numbers). The
session file keeps them as they are, so a resumed session reads back what the
tool returned. A dataclass in `details` is written as a dict with camelCase
keys and comes back as that dict, not as the dataclass: code that reads
`details` from the session, and renderers of a resumed session, then see the
dict.

Register an entry or message renderer when stored custom content should appear
in the transcript. For state shared between extensions, use `pi.events`.

## Custom tools

Register tools the model can call with `pi.register_tool()`. Tools appear in
the system prompt and can have custom rendering.

A tool is a `ToolDefinition` with a JSON Schema parameter spec and an execute
function. It must return an `AgentToolResult` — a bare dict is not accepted.

```python
from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent


async def extension(pi):
    async def execute(_tool_call_id, params, _cancel, _on_update, _ctx):
        return AgentToolResult(
            content=[TextContent(text=f"echo: {params['text']}")],
            details={},
        )

    pi.register_tool(
        ToolDefinition(
            name="echo",
            label="Echo",
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

Use `prompt_snippet` for a short one-line entry in the `<tools>` section of
the default system prompt. If omitted, the tool is left out of that section.

Use `prompt_guidelines` to add tool-specific bullets to the `<rules>` section
of the default system prompt. These bullets are included only while the tool
is active (for example, after `pi.set_active_tools([...])`).

**Important:** `prompt_guidelines` bullets are appended flat to the `<rules>`
section with no tool name prefix or grouping. Each guideline must name the
tool it refers to — avoid "Use this tool when..." because the model cannot
tell which tool "this" means. Write "Use my_tool when..." instead.

Some models include the `@` prefix in tool path arguments. Built-in tools
strip a leading `@` before resolving paths. If your custom tool accepts a
path, normalize a leading `@` as well.

Tool calls run in parallel by default. If your custom tool mutates files, wrap
the mutation in `with_file_mutation_queue()` (from `pidrei.core.tools`) so it
joins the same per-file queue as the built-in `edit` and `write`. Without the
queue, two tools can read the same old file contents, compute different
updates, and then whichever write lands last overwrites the other. For
example, your tool edits `foo.py` while the built-in `edit` also changes
`foo.py` in the same assistant turn: both read the original `foo.py`, apply
separate changes, and one of those changes is lost.

Queue the real target file, not the raw user argument: resolve it to an
absolute path first, relative to `ctx.cwd` or your tool's working directory.
The queue key comes from `await resolve_mutation_queue_key(path)` (in
`pidrei.core.tools.file_mutation_queue`), which resolves the path off the
runtime. For existing files the key goes through `realpath()`, so symlink
aliases of the same file share one queue; for new files it is the absolute
path, since there is nothing to resolve yet. Queue the entire mutation window
on that path, including any read-modify-write logic, not just the final
write:

```python
import os

from tonio.colored import fs

from pidrei.core.tools import with_file_mutation_queue
from pidrei.core.tools.file_mutation_queue import resolve_mutation_queue_key


async def execute(_tool_call_id, params, _cancel, _on_update, ctx):
    absolute_path = os.path.join(ctx.cwd, params["path"])
    queue_key = await resolve_mutation_queue_key(absolute_path)

    async def mutate():
        path = fs.Path(absolute_path)
        await path.parent.mkdir(parents=True, exist_ok=True)
        current = await path.read_text(encoding="utf-8")
        updated = current.replace(params["oldText"], params["newText"], 1)
        await path.write_text(updated, encoding="utf-8")
        return AgentToolResult(content=[TextContent(text=f"Updated {params['path']}")], details={})

    return await with_file_mutation_queue(absolute_path, mutate, queue_key=queue_key)
```

`mutate` is an async function taking no arguments. Once queued it runs to the
end even if the calling tool is cancelled, so a write is never torn.

### Tool definition

```python
from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent


MY_TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["list", "add"]},
        "text": {"type": "string"},
    },
    "required": ["action"],
}


def prepare_arguments(args):
    # Fold a legacy `oldAction` field into `action`. Schema validation still runs afterwards.
    if isinstance(args, dict) and "action" not in args and "oldAction" in args:
        return {**args, "action": args["oldAction"]}
    return args


async def extension(pi):
    async def execute(tool_call_id, params, cancel, on_update, ctx):
        # Check for cancellation
        if cancel is not None and cancel.cancelled:
            return AgentToolResult(content=[TextContent(text="Cancelled")], details={})

        # Stream progress updates (a plain call); partial results carry details too
        if on_update is not None:
            on_update(AgentToolResult(content=[TextContent(text="Working...")], details={}))

        # Run commands with pi.exec (captured from the factory's closure)
        result = await pi.exec("some-command", [], cancel=cancel)

        return AgentToolResult(
            content=[TextContent(text="Done")],  # Sent to the model
            details={"code": result.code},  # For rendering and state
            # usage=nested_response.usage,  # Optional nested model usage
            # Optional: stop after this tool batch when every finished tool
            # result in the batch also sets terminate=True.
            terminate=True,
        )

    pi.register_tool(
        ToolDefinition(
            name="my_tool",
            label="My Tool",
            description="What this tool does (shown to the model)",
            prompt_snippet="List or add items in the project todo list",
            prompt_guidelines=[
                "Use my_tool for todo planning instead of direct file edits when the user asks for a task list."
            ],
            parameters=MY_TOOL_PARAMS,
            prepare_arguments=prepare_arguments,
            execute=execute,
            # Optional: custom rendering
            # render_call=render_call,
            # render_result=render_result,
        )
    )
```

`ToolDefinition` takes keyword arguments only. `execute` is an `async def`
that receives `(tool_call_id, params, cancel, on_update, ctx)`: the call id,
the validated arguments, a cancel token, a streaming-update callback, and the
extension context.

- `cancel` is a `CancelToken` (from `pidrei_utils.cancel`), pidrei's
  counterpart of an abort signal: read `cancel.cancelled`, call
  `cancel.raise_if_cancelled()`, register `cancel.on_cancel(callback)`, or hand
  it on (`pi.exec(..., cancel=cancel)`). It may be `None`.
- `on_update(partial_result)` is a plain synchronous call that takes an
  `AgentToolResult`; the UI renders the partial result. Calls made after the
  tool returned are ignored.
- `pi.exec(...)` and the other `pi` actions marked `await` in the
  [Actions](#actions) table do nothing until awaited.

**Result fields:** `content` (text and image blocks) is what the model sees.
`details` is arbitrary data for your own renderers and state reconstruction,
never sent to the model; it defaults to `None`, which is right when there is
nothing to store. Keep `details` the same shape across all results of the
tool, partial results passed to `on_update` included: renderers and state
reconstruction read it (see [State management](#state-management) for what
survives a resume).

**Usage accounting:** if a tool makes nested model calls, return their
combined `Usage` (from `pidrei_ai.types`; `combine_usage` in
`pidrei_ai.utils.usage` adds two) as `usage`. pidrei keeps it on the tool
result and includes it in the footer, `/session` and RPC session totals.
`tool_result` handlers can inspect or replace it (`event["usage"]`).

**Signaling errors:** to mark a tool execution as failed, raise from
`execute`, or return the result with `is_error=True`. Both report an error
result to the model; a returned result also keeps `details` and
`structured_content` for rendering and scripts. Do not only describe the
failure in `content`: other returned results never set the error flag.

```python
# Correct: raise to signal an error
async def execute(_tool_call_id, params, _cancel, _on_update, _ctx):
    if not is_valid(params["input"]):
        raise ValueError(f"Invalid input: {params['input']}")
    return AgentToolResult(content=[TextContent(text="OK")], details={})
```

The model sees the exception's message as the error result.

**Early termination:** set `terminate=True` on the result to skip the
automatic follow-up model request after the current tool batch. It takes
effect only when every finished tool result in that batch sets it. See
`examples/extensions/structured_output.py` for a minimal example where the
agent ends on a final structured-output tool call.

**String enums:** write them as `{"type": "string", "enum": [...]}`, a plain
`enum` schema that every provider accepts.

**Structured output:** declare `output_schema` and return a matching
`structured_content` when the result is data. The model still receives
`content`; programmatic callers such as codemode scripts receive
`structured_content` instead of the text. Tools without `output_schema` are
passed to scripts as their text content. To report a failure that still
carries data, return the result with `is_error=True` instead of raising: the
model sees an error, and scripts still receive `structured_content`. The
built-in `bash` tool returns `{"output", "truncated", "full_output_path"?,
"exit_code", "wall_time_seconds"}`, also for non-zero exit codes, with up to
1 MiB of output. `tool_result` handlers that redact `content` should also
replace `structuredContent`; replacing only `content` drops it.

**Execution mode:** tool calls from one assistant message run in parallel by
default. Set `execution_mode="sequential"` when a tool must not run
concurrently with other calls, for example because it shares mutable
in-memory state; a batch that contains such a call runs all of its calls one
at a time. Event handlers and other coroutines still run in parallel with the
tool, so state they share with it needs a `threading.Lock` held only around
synchronous code (see `examples/extensions/todo.py`).

**Other definition fields:**

| Field | Purpose |
|---|---|
| `default_active` | Set `False` to register the tool inactive; `--tools`, the `defaultTools` setting, or `pi.set_active_tools()` activates it |
| `exposure`, `namespace`, `annotations`, `reaches_tools`, `prepare_loadout` | Control how the model and other tools reach the tool; see [Tool exposure](#tool-exposure) |
| `render_shell`, `render_call`, `render_result` | Custom rendering; see [Custom rendering](#custom-rendering) |
| `constrained_sampling` | Optional provider-side constrained sampling request for the tool's arguments, such as `JsonSchemaConstrainedSampling(strict="prefer")` from `pidrei_ai.types` |

**Argument validation:** `parameters` must be a dict (registering anything
else raises). pidrei validates the arguments of every call against it as JSON
Schema (draft 2020-12), with pi's validation rules, before `execute` runs.
First, loosely typed arguments are coerced towards the schema:

- numeric strings to numbers (`"42"` becomes `42`; for an `integer`, `"3.5"`
  becomes `3`);
- `"true"` and `"false"` to booleans, and numbers to strings;
- a single value to a one-item array;
- a `null` in an optional property whose schema does not accept null is
  dropped, as if omitted;
- a union (`anyOf`/`oneOf`) takes the first branch the value converts to.

A value that cannot be coerced is left as it is, and the check reports it.
`execute` receives the coerced arguments as a copy, so changing `params`
never changes the tool call in the transcript. The standard formats are
checked, plus `url` and `json-pointer-uri-fragment`; any other `format` is an
annotation. `pattern` and `patternProperties` use Python's `re` syntax. A
`$ref` that names nothing rejects only arguments at its place, and nothing is
ever fetched over the network.

When validation fails the tool does not run, and the model gets an error
result with one line per failing keyword, at the dotted path of the value
(`root` for the arguments themselves), followed by the arguments it sent:

```
Validation failed for tool "my_tool":
  - action: missing
  - limit: expected number, got string

Received arguments:
{
  "limit": "x"
}
```

**Argument preparation:** `prepare_arguments(args)` is optional. If defined,
it runs before schema validation and before `execute`, as a plain synchronous
function. Use it to accept an older input shape when pidrei resumes an older
session whose stored tool call arguments no longer match the current schema.
Return the value you want validated against `parameters`. Return a new dict
rather than modifying `args`: `args` is the tool call stored in the
transcript. Keep the public schema strict: do not add deprecated
compatibility fields to `parameters` just to keep old resumed sessions
working.

Example: an older session may contain an `edit` tool call with top-level
`oldText` and `newText`, while the current schema only accepts
`edits: [{"oldText", "newText"}]`.

```python
EDIT_PARAMS = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"oldText": {"type": "string"}, "newText": {"type": "string"}},
                "required": ["oldText", "newText"],
            },
        },
    },
    "required": ["path", "edits"],
}


def prepare_edit_arguments(args):
    if not isinstance(args, dict):
        return args  # validation reports anything still invalid
    old_text, new_text = args.get("oldText"), args.get("newText")
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        return args
    rest = {key: value for key, value in args.items() if key not in ("oldText", "newText")}
    return {**rest, "edits": [*(rest.get("edits") or []), {"oldText": old_text, "newText": new_text}]}


async def extension(pi):
    async def execute(_tool_call_id, params, _cancel, _on_update, _ctx):
        # params now matches the current schema
        return AgentToolResult(
            content=[TextContent(text=f"Applying {len(params['edits'])} edit block(s)")],
            details={},
        )

    pi.register_tool(
        ToolDefinition(
            name="edit",
            label="Edit",
            description="Edit a single file using exact text replacement",
            parameters=EDIT_PARAMS,
            prepare_arguments=prepare_edit_arguments,
            execute=execute,
        )
    )
```

### Overriding built-in tools

Extensions can override built-in tools (`read`, `bash`, `edit`, `write`,
`grep`, `find`, `ls`) by registering a tool with the same name. The extension
tool replaces the built-in without a warning.

```bash
# Extension's read tool replaces built-in read
pidrei -e ./tool_override.py
```

Extension tools respect the same filters as built-in ones: `--tools` restricts
to a list, `--exclude-tools` removes some, and `--no-builtin-tools` drops
pidrei's own tools while keeping extension tools:

```bash
# No built-in tools, only extension tools
pidrei --no-builtin-tools -e ./my_extension.py
```

See `examples/extensions/tool_override.py` for a complete example that
overrides `read` with logging and access control.

**Rendering:** built-in renderer inheritance is resolved per slot. Execution
override and rendering override are independent. If your override omits
`render_call`, the built-in `render_call` is used; if it omits
`render_result`, the built-in `render_result` is used. If it omits both, the
built-in renderer is used automatically (syntax highlighting, diffs, etc.).
This lets you wrap built-in tools for logging or access control without
reimplementing the UI. `render_shell` is not inherited.

**Prompt metadata:** `prompt_snippet` and `prompt_guidelines` are not
inherited from the built-in tool. If your override should keep those prompt
instructions, define them on the override explicitly.

**Your implementation must match the exact result shape**, including the
`details` type. The UI and session logic depend on these shapes for rendering
and state tracking.

Built-in tool implementations:

- `pidrei/core/tools/read.py` - `ReadToolDetails`
- `pidrei/core/tools/bash.py` - `BashToolDetails`
- `pidrei/core/tools/edit.py` - `EditToolDetails`
- `pidrei/core/tools/write.py`
- `pidrei/core/tools/grep.py` - `GrepToolDetails`
- `pidrei/core/tools/find.py` - `FindToolDetails`
- `pidrei/core/tools/ls.py` - `LsToolDetails`

### Remote execution

Built-in tools support pluggable operations for delegating to remote systems
(SSH, containers, etc.). Each `create_*_tool_definition(cwd, ...)` factory in
`pidrei.core.tools` takes an `operations` object:

```python
import dataclasses
import os

from pidrei.core.tools import create_read_tool_definition


class SshReadOperations:
    def __init__(self, ssh):
        self._ssh = ssh

    async def read_file(self, absolute_path: str) -> bytes:
        return await ssh_read_bytes(self._ssh, absolute_path)

    async def access(self, absolute_path: str) -> None:
        await ssh_exec(self._ssh, f"test -r {absolute_path}")  # raises when not readable


async def extension(pi):
    cwd = os.getcwd()
    local_read = create_read_tool_definition(cwd)

    # Register, checking the remote configuration at execution time
    async def execute(tool_call_id, params, cancel, on_update, ctx):
        ssh = get_ssh_config()
        if ssh is not None:
            remote_read = create_read_tool_definition(cwd, operations=SshReadOperations(ssh))
            return await remote_read.execute(tool_call_id, params, cancel, on_update, ctx)
        return await local_read.execute(tool_call_id, params, cancel, on_update, ctx)

    pi.register_tool(dataclasses.replace(local_read, execute=execute))
```

The factories return complete tool definitions, including the built-in prompt
snippet and guidelines, so `dataclasses.replace(definition, execute=...)`
keeps them. Rendering does not need copying: a tool registered under a
built-in name inherits the built-in renderer for each slot it leaves out (see
[Overriding built-in tools](#overriding-built-in-tools)).

**Operations:** an `operations` object provides these methods, all `async`:

| Tool | Methods |
|---|---|
| `read` | `read_file(path) -> bytes`, `access(path)` (raises when the file is not readable), and optionally `detect_image_mime_type(path) -> str \| None` |
| `write` | `write_file(path, content)`, `mkdir(dir)` |
| `edit` | `read_file(path) -> bytes`, `write_file(path, content)`, `access(path)` |
| `bash` | `exec(command, cwd, *, on_data, cancel=None, timeout=None, env=None) -> BashExecResult`; `on_data` receives output chunks as `bytes`, `timeout` is in seconds |
| `ls` | `exists(path)`, `is_directory(path)`, `readdir(path) -> list[str]` |
| `grep` | `is_directory(path)`, `read_file(path) -> str` |
| `find` | `exists(path)`, `glob(pattern, path, *, ignore, limit)` returning the matching paths |

See `examples/extensions/ssh.py` for a complete SSH example with an `--ssh`
flag.

For `user_bash`, extensions can reuse pidrei's local shell backend with
`create_local_bash_operations()` (from `pidrei.core.tools`) instead of
reimplementing local process spawning, shell resolution and process-tree
termination.

The `bash` tool also supports a spawn hook to adjust the command, cwd, or env
before execution. The hook is a plain synchronous function that receives a
`BashSpawnContext(command, cwd, env)` and returns one:

```python
import dataclasses

from pidrei.core.tools import create_bash_tool_definition


def spawn_hook(context):
    return dataclasses.replace(
        context,
        command=f"source ~/.profile\n{context.command}",
        cwd=f"/mnt/sandbox{context.cwd}",
        env={**context.env, "CI": "1"},
    )


bash_tool = create_bash_tool_definition(cwd, spawn_hook=spawn_hook)
```

`create_bash_tool_definition()` exposes the current session to commands
through `PIDREI_SESSION_ID`, `PIDREI_SESSION_FILE` (when the session has a
file), `PIDREI_PROVIDER`, `PIDREI_MODEL` and `PIDREI_REASONING_LEVEL`.
Injection happens before `spawn_hook`, so hooks receive these values in `env`
and keep them when they spread the existing environment as above. Pass
`expose_session_environment=False` to disable them:

```python
bash_tool = create_bash_tool_definition(cwd, expose_session_environment=False)
```

See [Available to bash tools](environment-variables.md#available-to-bash-tools)
for the variables, and `examples/extensions/bash_spawn_hook.py` for a
complete example.

### Output truncation

**Tools MUST truncate their output** to avoid overwhelming the model's
context. Large outputs can cause:

- Context overflow errors (prompt too long)
- Compaction failures
- Degraded model performance

The built-in limit is **50KB** (~10k tokens) and **2000 lines**, whichever is
hit first. Use the truncation utilities from `pidrei.core.tools`:

| Name | Purpose |
|---|---|
| `truncate_head(content, max_lines=..., max_bytes=...)` | Keep the first lines/bytes (good for file reads, search results) |
| `truncate_tail(content, max_lines=..., max_bytes=...)` | Keep the last lines/bytes (good for logs, command output) |
| `truncate_line(line, max_chars=500)` | Truncate a single line to a character count; returns a `TruncatedLine` with `text` and `was_truncated` |
| `format_size(size_bytes)` | Human-readable size (`"50.0KB"`, `"1.5MB"`) |
| `DEFAULT_MAX_BYTES` | 50KB |
| `DEFAULT_MAX_LINES` | 2000 |

`truncate_head` and `truncate_tail` return a `TruncationResult` with
`content`, `truncated`, `truncated_by` (`"lines"`, `"bytes"` or `None`),
`total_lines`, `total_bytes`, `output_lines` and `output_bytes`, among others.

```python
import tempfile

import tonio.colored as tonio
from tonio.colored import fs

from pidrei.core.tools import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head


async def execute(_tool_call_id, params, _cancel, _on_update, ctx):
    output = await run_command()

    # Apply truncation
    truncation = truncate_head(output, max_lines=DEFAULT_MAX_LINES, max_bytes=DEFAULT_MAX_BYTES)
    result = truncation.content
    details = {"truncated": truncation.truncated}

    if truncation.truncated:
        # Write the full output to a temp file, off the runtime
        temp_dir = await tonio.spawn_blocking(tempfile.mkdtemp, prefix="my-tool-")
        temp_file = fs.Path(temp_dir) / "output.txt"
        await temp_file.write_text(output, encoding="utf-8")
        details["fullOutputPath"] = str(temp_file)

        # Tell the model where to find the complete output
        result += f"\n\n[Output truncated: {truncation.output_lines} of {truncation.total_lines} lines"
        result += f" ({format_size(truncation.output_bytes)} of {format_size(truncation.total_bytes)})."
        result += f" Full output saved to: {temp_file}]"

    return AgentToolResult(content=[TextContent(text=result)], details=details)
```

**Key points:**

- Use `truncate_head` for content where the beginning matters (search
  results, file reads).
- Use `truncate_tail` for content where the end matters (logs, command
  output).
- Always tell the model when output is truncated and where to find the full
  version.
- Document the truncation limits in your tool's description.

See `examples/extensions/truncated_tool.py` for a complete example wrapping
`rg` (ripgrep) with proper truncation.

### Multiple tools

One extension can register multiple tools with shared state.
`pi.register_tools([...])` adds them in one step: the session sees all of
them or none.

```python
async def extension(pi):
    state = {"connection": None}

    pi.register_tools(
        [
            ToolDefinition(name="db_connect", label=..., description=..., parameters=..., execute=connect),
            ToolDefinition(name="db_query", label=..., description=..., parameters=..., execute=query),
            ToolDefinition(name="db_close", label=..., description=..., parameters=..., execute=close),
        ]
    )

    async def on_shutdown(_event, _ctx):
        connection = state["connection"]
        if connection is not None:
            await connection.close()

    pi.on("session_shutdown", on_shutdown)
```

<a id="tool-rendering"></a>

### Custom rendering

Tools can provide `render_call` and `render_result` for custom display. A
tool's renderers draw its calls in the interactive transcript and in HTML
exports. See [tui.md](tui.md) for the full component API and
`pidrei/modes/interactive/components/tool_execution.py` for how tool rows are
composed.

Renderers are plain synchronous functions that return a component from
`pidrei_tui`; never block or await in them. By default, tool output is wrapped
in a `Box` that handles the background and the horizontal padding from the
`outputPad` setting. If a slot renderer is not defined, pidrei uses fallback
rendering for that slot.

Set `render_shell="self"` when the tool should render its own shell instead of
using the default `Box`. This is useful for tools that need complete control
over framing or background behavior, for example large previews that must
stay visually stable after the tool settles.

```python
from pidrei_tui import Text


async def extension(pi):
    async def execute(_tool_call_id, _params, _cancel, _on_update, _ctx):
        return AgentToolResult(content=[TextContent(text="ok")])

    def render_call(_args, theme, _context):
        return Text(theme.fg("accent", "my custom shell"), 0, 0)

    pi.register_tool(
        ToolDefinition(
            name="my_tool",
            label="My Tool",
            description="Custom shell example",
            parameters={"type": "object", "properties": {}},
            render_shell="self",
            execute=execute,
            render_call=render_call,
        )
    )
```

`render_call(args, theme, context)` and
`render_result(result, options, theme, context)` each receive a `context`
dict with these keys:

- `args` - the current tool call arguments
- `state` - a dict shared by `render_call` and `render_result` for this tool
  row
- `lastComponent` - the component this slot returned last time, if any
- `invalidate()` - request a rerender of this tool row; safe to call from
  your own coroutines
- `toolCallId`, `cwd`, `executionStarted`, `argsComplete`, `isPartial`,
  `expanded`, `showImages`, `isError`
- `durationMs` - how long execution took, from the final result; `None` while
  running
- `outputPad` - horizontal padding from the `outputPad` setting; renderers
  with `render_shell="self"` apply it themselves
- `keyText(id)`, `keyHint(id, description)` - format the user's configured
  keybindings, see [Keybinding hints](#keybinding-hints)

Use `context["state"]` for state shared across the two slots. Keep slot-local
caches on the returned component instance when you want to reuse and mutate
the same component across renders.

#### render_call

Renders the tool call or header. `args` can be partial, or empty, while the
model is still streaming the call:

```python
from pidrei_tui import Text


def render_call(args, theme, context):
    last = context["lastComponent"]
    text = last if isinstance(last, Text) else Text("", 0, 0)
    args = args or {}
    content = theme.fg("toolTitle", theme.bold("my_tool "))
    content += theme.fg("muted", args.get("action", ""))
    if args.get("text"):
        content += " " + theme.fg("dim", f'"{args["text"]}"')
    text.set_text(content)
    return text
```

#### render_result

Renders the tool result or output. `result` is a dict with `content` and
`details`; `options` is a dict with `expanded` and `isPartial`:

```python
def render_result(result, options, theme, context):
    if options["isPartial"]:
        return Text(theme.fg("warning", "Processing..."), 0, 0)

    details = result["details"] or {}
    if details.get("error"):
        return Text(theme.fg("error", f"Error: {details['error']}"), 0, 0)

    text = theme.fg("success", "✓ Done")
    if options["expanded"]:
        for item in details.get("items", []):
            text += "\n  " + theme.fg("dim", item)
    return Text(text, 0, 0)
```

`details` is what the tool returned while the session runs, and what the
session file stored once it is resumed (see
[State management](#state-management)); content blocks can be objects
(`block.text`) or dicts (`block["text"]`), so renderers that read them should
accept both, as `examples/extensions/truncated_tool.py` does.

If a slot intentionally has no visible content, return an empty component
such as an empty `Container()`.

#### Keybinding hints

Use `keyHint` and `keyText` from the render context to display keybinding
hints that respect the user's keybinding configuration:

```python
def render_result(result, options, theme, context):
    text = theme.fg("success", "✓ Done")
    if not options["expanded"]:
        text += f" ({context['keyHint']('app.tools.expand', 'to expand')})"
    return Text(text, 0, 0)
```

The same functions are in the `options` dict of message and entry renderers
(`options["keyHint"]`, `options["keyText"]`):

- `keyHint(keybinding, description)` - formats a configured keybinding id such
  as `"app.tools.expand"` or `"tui.select.confirm"`, followed by the
  description
- `keyText(keybinding)` - returns the raw configured key text for a keybinding
  id, or `""` if the id has no keys

They are the functions pidrei's own renderers use. To format a literal key
string, use `raw_key_hint(key, description)` from
`pidrei.modes.interactive.components`.

Use namespaced keybinding ids:

- Coding-agent ids use the `app.*` namespace, for example `app.tools.expand`,
  `app.editor.external`, `app.session.rename`
- Shared TUI ids use the `tui.*` namespace, for example `tui.select.confirm`,
  `tui.select.cancel`, `tui.input.tab`

For the exhaustive list of keybinding ids and defaults, see
[keybindings.md](keybindings.md). `keybindings.json` uses those same
namespaced ids.

Custom editors and `ctx.ui.custom()` components receive the keybindings
manager as an argument of their factory (`keybindings`); use it directly.

#### Best practices

- Use `Text` with padding `(0, 0)`. The default `Box` handles padding.
- Use `\n` for multi-line content.
- Handle `isPartial` for streaming progress.
- Support `expanded` for detail on demand.
- Keep the default view compact.
- Read `context["args"]` in `render_result` instead of copying args into
  `context["state"]`.
- Use `context["state"]` only for data that must be shared across the call
  and result slots.
- Reuse `context["lastComponent"]` when the same component instance can be
  updated in place.
- Use `render_shell="self"` only when the default boxed shell gets in the way.
  In self-shell mode the tool is responsible for its own framing, padding and
  background.

#### Fallback

If a slot renderer is not defined or raises:

- `render_call`: shows the tool name and its arguments
- `render_result`: shows the text from `content` (the first 10 lines while
  collapsed)

#### Renderers for any tool

To choose renderers for calls to any tool, including tools that are not
registered yet (MCP tools in a resumed session before their server
connected, say), register a resolver with
[`pi.register_tool_renderer()`](#piregister_tool_rendererresolver). Resolvers
are synchronous, like the renderers they return.

### Dynamic tool loading

Extensions can register many tools while keeping only a small initial set
active. A tool can then change the active set during execution. pidrei stores
the initial prompt and tool loadout in the transcript's first system message,
then appends tool and prompt changes as system messages before the next model
request. A model that cannot take system messages mid-conversation gets them
folded into the leading one instead, which can invalidate the cached prefix.

The lifecycle is:

1. Register every tool with `pi.register_tool()` so it appears in
   `pi.get_all_tools()`. Register searchable tools with `default_active=False`
   so they start inactive.
2. Keep loader tools, such as `search_tools`, active.
3. During loader execution, change the active tool names with
   `pi.update_active_tools()` (or `pi.set_active_tools()`). Names must
   already be registered; unknown and `"hidden"` names are ignored.

To add or remove tools relative to the current set, use
`pi.update_active_tools(update)` rather than
`pi.set_active_tools([*pi.get_active_tools(), ...])`. Tools, handlers and tool
registrations run in parallel, so another change can land between
`get_active_tools()` and `set_active_tools()`, and the write would undo it.
`update` receives the active tool names and returns the new ones, or `None`
to leave them as they are; no other change lands in between. It runs
synchronously while the session's tool loadout is locked, so keep it quick and
never block in it:

```python
pi.update_active_tools(lambda active: [*active, "docs_search"])
```

pidrei also ships a built-in `tool_search` tool. Register tools with
`exposure="deferred"` to make them findable by it instead of writing your own
loader; see [Tool exposure](#tool-exposure).

#### Search tool example

The following extension registers two inactive searchable tools and keeps
`search_tools` active as their loader. The example uses simple keyword
matching, but the search could use BM25, embeddings, a remote catalog, or
project-specific routing.

```python
import re

from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent


SEARCHABLE_TOOL_NAMES = {"lookup_weather", "search_issues"}


async def extension(pi):
    async def lookup_weather(_tool_call_id, params, _cancel, _on_update, _ctx):
        return AgentToolResult(content=[TextContent(text=f"Weather for {params['city']}: sunny")], details={})

    async def search_issues(_tool_call_id, params, _cancel, _on_update, _ctx):
        return AgentToolResult(content=[TextContent(text=f"No open issues matching {params['query']}")], details={})

    pi.register_tool(
        ToolDefinition(
            name="lookup_weather",
            label="Lookup Weather",
            default_active=False,
            description="Look up the current weather for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            execute=lookup_weather,
        )
    )

    pi.register_tool(
        ToolDefinition(
            name="search_issues",
            label="Search Issues",
            default_active=False,
            description="Search project issues by keyword",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            execute=search_issues,
        )
    )

    async def search_tools(_tool_call_id, params, _cancel, _on_update, _ctx):
        terms = [term for term in re.split(r"[^a-z0-9]+", params["query"].lower()) if term]
        scored = []
        for tool in pi.get_all_tools():
            if tool.name not in SEARCHABLE_TOOL_NAMES:
                continue
            haystack = f"{tool.name} {tool.description}".lower()
            score = sum(1 for term in terms if term in haystack)
            if score > 0:
                scored.append((score, tool.name))
        scored.sort(key=lambda match: -match[0])
        matches = [name for _score, name in scored[: params.get("limit", 3)]]

        if not matches:
            return AgentToolResult(
                content=[TextContent(text=f"No tools found for: {params['query']}")],
                details={"matches": [], "added": []},
            )

        added = []

        def add_matches(active):
            added.extend(name for name in matches if name not in active)
            return [*active, *added] if added else None

        pi.update_active_tools(add_matches)

        text = f"Loaded tools: {', '.join(added)}" if added else f"Matching tools already active: {', '.join(matches)}"
        return AgentToolResult(content=[TextContent(text=text)], details={"matches": matches, "added": added})

    pi.register_tool(
        ToolDefinition(
            name="search_tools",
            label="Search Tools",
            description="Search for and enable tools relevant to a task",
            prompt_snippet="Search for additional tools when the active tools cannot perform the task",
            prompt_guidelines=[
                "Use search_tools when a task requires a capability that is not currently available.",
            ],
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Capability or task to search for"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
            execute=search_tools,
        )
    )
```

When `search_tools` adds a match, the model receives the complete updated
tool list on the immediately following request.

### Tool exposure

`exposure` controls how the model reaches a tool. "Callable" means callable
from other tools through `ctx.execute_tool()` (`ctx.tools`), as the `codemode`
tool's scripts do:

- `"direct"` (default): declared to the model while active, and callable while
  active.
- `"model-only"`: declared to the model while active, never callable. Use it
  for tools that orchestrate other tools or ask the user.
- `"codemode"`: callable whenever registered, and listed by the `codemode`
  tool. Not declared to the model unless activated explicitly.
- `"deferred"`: like `"codemode"`, but the `codemode` tool does not list it;
  `tool_search` can find and activate it.
- `"hidden"`: registered but unreachable. Re-register a tool with
  `exposure="hidden"` to withdraw it, since tools cannot be unregistered.

`namespace=ToolNamespace(name=..., aliases=..., description=..., instructions=...)`
groups related tools, as MCP servers do. The `codemode` tool lists a namespace
under one heading with its `description`. `instructions` holds longer usage
guidance; it is not listed, and codemode scripts read it with
`describe_namespace(name)`. `aliases` are other names scripts may look the
namespace up by, for example the server name `docs` for `mcp__docs`; a name and
the same name as a script identifier (`my-docs`, `my_docs`) match alike.

Registering a `"direct"` or `"model-only"` tool activates it unless it sets
`default_active=False`; the other exposures are not activated on
registration. A tool with `default_active=False` is activated by naming it in
`--tools` or the `defaultTools` setting (`"+name"` adds it to the defaults),
or with `set_active_tools()`. The active set (`pi.get_active_tools()`,
`pi.set_active_tools()`) is the set of tools declared to the model, except
active tools whose declarations `prepare_loadout` hides. `pi.get_all_tools()`
reports each tool's `exposure`, `namespace`, `annotations` and
`reaches_tools`.

`annotations=ToolAnnotations(...)` are hints about what a tool does, with the
meaning of MCP tool annotations: `read_only_hint`, `destructive_hint`,
`idempotent_hint` and `open_world_hint`. MCP tools carry the hints their
server declares. A hint left `None` takes the MCP default: a tool is not
read-only, and may be destructive and reach an open world. The hints are not
verified, but a permission extension can use them to decide which calls to
confirm. This confirms the calls Codex asks approval for:

```python
from pidrei.core.extensions.types import ToolAnnotations


async def extension(pi):
    async def confirm_risky_calls(event, ctx):
        tool = next((tool for tool in pi.get_all_tools() if tool.name == event["toolName"]), None)
        hints = tool.annotations if tool is not None and tool.annotations is not None else ToolAnnotations()
        needs_approval = hints.destructive_hint is True or (
            not hints.read_only_hint
            and (hints.destructive_hint in (None, True) or hints.open_world_hint in (None, True))
        )
        if needs_approval and not await ctx.ui.confirm("Allow tool call?", event["toolName"]):
            return {"block": True, "reason": f"{event['toolName']} was not approved"}
        return None

    pi.on("tool_call", confirm_risky_calls)
```

A tool that orchestrates other tools can adjust what the model sees while it is
active with `prepare_loadout(loadout)`, a plain synchronous function. It runs
whenever the active tools change and receives a `ToolLoadout`: the declared
tools, the callable tools, and every registered tool, with
`get_exposure(name)`, `get_namespace(name)` and `get_prompt_guidelines(name)`.
It returns a `ToolLoadoutChanges` (or `None`) with replacement `descriptions`
for declared tools (including its own) and `hidden_declarations`: active tools
whose declarations requests leave out while they stay active and callable. The
default system prompt leaves hidden tools out of its tool list and rules, and
names no tool in the skills hint when the file reader is hidden, so the
orchestrating tool should present their guidelines itself. `codemode` uses
only this hook, `exposure`, `reaches_tools` and `ctx.execute_tool()`, so
another tool can implement the same behavior under a different name. The
exposure types, `ToolNamespace`, `ToolAnnotations`, `ToolLoadout` and
`ToolLoadoutChanges` come from `pidrei.core.extensions.types`.

A tool that reaches other tools declares which ones a call may reach with
`async reaches_tools(input)`: `"all"`, or the words of the input that may name
them, where a word refers to every tool or namespace whose name it contains.
`codemode` returns `"all"` for scripts that search, list, or call tools by name
and the words of the script otherwise; `tool_search` always returns `"all"`.
Extensions that register tools late use it from their `tool_call` handler to
finish registering the tools a call needs before it runs, as the MCP extension
does for servers that are still connecting. The MCP extension also activates
`codemode` or `tool_search` for MCP tools only when the tool of that name
declares `reaches_tools`.

### Calling other tools

A tool can run other tools with `await ctx.execute_tool(name, args,
ExecuteToolOptions(cancel=..., on_update=...))` (`ExecuteToolOptions` is in
`pidrei.core.nested_tool_calls`); the cancel token defaults to the calling
tool's. `ctx.tools` lists the tools it can call. The call returns an
`AgentToolCallOutcome` (`result`, `is_error`) and never raises for tool
failures: unknown tools, validation errors, blocked calls and raised errors
come back with `is_error=True`.

Nested calls go through argument validation and the `tool_call` and
`tool_result` handlers like model-issued calls, and emit
`tool_execution_start`, `tool_execution_update` and `tool_execution_end`. All
of these events carry `parentToolCallId`, and their `toolCallId` is assigned
by pidrei as `<parent id>/<n>`. These ids do not appear as tool calls or tool
results in the transcript.

Nested calls add no transcript entries: their results only reach the calling
tool, which reports them itself, for example through `on_update` and
`details`. The session keeps a bounded record of them (name, arguments,
status, duration, error; never results) as `nested_calls` on the calling
tool's result message. It is used for compaction file lists and shown in HTML
exports. Arguments over 8 KiB per call or 32 KiB per tool result are omitted,
at most 256 calls are kept, and `complete=False` marks a record that lost
anything; a call still running when the calling tool returned is
`unfinished`.

The `usage` of nested results, at every depth, is added to the calling tool's
result `usage`, so a tool reports only its own usage, not that of the tools it
called.

## MCP servers

`pi.register_mcp_server(name, config)` adds an MCP server for the current
session. `config` is a dict with the shape of an `mcpServers` entry in
[`mcp.json`](mcp.md): `command`, `args`, `env` and `cwd` for stdio servers,
`url`, `headers` and `oauth` for HTTP servers, plus `exposure`,
`toolExposure`, `description`, `enabled` and `timeout`.

```python
pi.register_mcp_server("jira", {"url": "https://mcp.example.com/jira", "exposure": "codemode"})
pi.unregister_mcp_server("jira")
```

Servers registered while the extension loads connect when the session starts,
together with the `mcp.json` servers; servers registered later connect right
away, and `pi.unregister_mcp_server(name)` closes the connection and makes the
server's tools unreachable. Registrations are not saved: register again on
every load, for example based on the extension's own settings. A server in
`mcp.json` with the same name takes precedence, and `/mcp` shows the
override. Registering the same name again replaces the extension's earlier
registration; names another extension registered, names that differ from a
registered server's only in `-` and `_`, invalid names and invalid configs
raise.

The built-in MCP support connects registered servers. When nothing does,
because another extension replaced it (see
[MCP](mcp.md#replace-the-built-in-mcp-support)), each registration is reported
as an extension error. Other MCP extensions can connect registered servers
too: read them with `pi.get_mcp_servers()` (each a `RegisteredMcpServer` with
`name`, `config` and `extension_path`) on `session_start`, and handle the
`mcp_servers_change` event (`{"servers": [...]}`) for later changes.

## Custom UI

Extensions talk to the user through `ctx.ui` and control how their messages,
entries and Markdown appear in the transcript. This section is the
extension-facing API: each `ctx.ui` method, its options and what it returns.
[tui.md](tui.md) covers the components themselves (`Container`, `Text`,
`SelectList`, `SettingsList` and friends from `pidrei_tui`), focus, overlays,
theming and rendering performance.

How the calls behave:

- The dialogs (`select`, `confirm`, `input`, `editor`, `custom`) open when
  called and return a handle: `await` it for the user's answer. Dropping the
  handle is fine; the dialog stays open and answerable.
- The setters and getters, `paste_to_editor` and `get_editor_text` included,
  are plain synchronous calls: each change is applied whole before the call
  returns, so your next line sees it.
- The theme accessors (`get_all_themes`, `get_theme`, `set_theme`) are
  awaitable: loading a theme reads files.
- Every component factory (`custom`, `set_widget`, `set_footer`,
  `set_header`, `set_editor_component`) is a plain synchronous function. An
  `async def` factory is refused.
- `ctx.ui` works from anywhere: extension handlers on their own coroutines,
  and component code (see
  [Where extension code and component code run](#where-extension-code-and-component-code-run)).

### Dialogs

```python
# Select from options: the chosen string, or None
choice = await ctx.ui.select("Pick one:", ["A", "B", "C"])

# Confirm dialog: True or False
ok = await ctx.ui.confirm("Delete?", "This cannot be undone")

# Single-line input: the text, or None
name = await ctx.ui.input("Name:", "placeholder")

# Multi-line editor, prefilled: the text, or None
text = await ctx.ui.editor("Edit:", "prefilled text")

# Notification (non-blocking)
ctx.ui.notify("Done!", "info")  # "info" | "warning" | "error"
```

`select`, `input` and `editor` resolve to `None` when the user dismisses
them; `confirm` resolves to `False`. The `editor` dialog opens the external
editor (`$VISUAL`, `$EDITOR`, or the configured command) on Ctrl+G. While a
dialog is open, terminals that support
[program status](environment-variables.md#program-status) show pidrei as
blocked: `confirm` as a permission request, the others as a question.

Dialogs stack: opening one from inside another (a confirm from a custom
component, say) hides the visible one, and closing it shows the previous one
again.

Component code (`handle_input`, `render`, callbacks) is synchronous, so it
cannot await a dialog's answer. Calling the dialog opens it all the same, and
the next key already goes to it; to act on the answer, await the handle in
work started with `tui.spawn(...)`.

#### Timed dialogs with countdown

`select`, `confirm` and `input` take an options dict as their third argument.
`timeout` (milliseconds) dismisses the dialog automatically and shows a live
countdown after its title (`(5s)`, `(4s)`, ...):

```python
confirmed = await ctx.ui.confirm(
    "Timed Confirmation",
    "This dialog will auto-cancel in 5 seconds. Confirm?",
    {"timeout": 5000},
)

if confirmed:
    ...  # The user confirmed
else:
    ...  # The user cancelled, or the dialog timed out
```

On timeout, `select` and `input` return `None` and `confirm` returns
`False`, exactly as if the user had dismissed the dialog.

#### Manual dismissal with a cancel token

To close a dialog yourself, or to tell a timeout from a user cancel, pass a
`CancelToken` (from `pidrei_utils.cancel`) as `signal`. Cancelling the token
closes the dialog with the dismissed value; a token already cancelled when
the dialog is called returns that value at once without showing anything:

```python
import tonio.colored as tonio

from pidrei_utils.cancel import CancelToken


async def run_timed_signal(_args, ctx):
    token = CancelToken()
    settled = tonio.Event()

    async def watchdog():
        await settled.wait(5)
        if not settled.is_set():
            token.cancel()

    watchdog_join = tonio.spawn(watchdog())
    try:
        confirmed = await ctx.ui.confirm(
            "Timed Confirmation",
            "This dialog will auto-cancel in 5 seconds. Confirm?",
            {"signal": token},
        )
    finally:
        settled.set()
        await watchdog_join

    if confirmed:
        ...  # The user confirmed
    elif token.cancelled:
        ...  # The dialog timed out
    else:
        ...  # The user cancelled (Escape, or "No")
```

See `examples/extensions/timed_confirm.py` for both approaches.

### Widgets, status and footer

```python
# Status in the footer (persistent until cleared)
ctx.ui.set_status("my-ext", "Processing...")
ctx.ui.set_status("my-ext", None)  # Clear

# Working message (shown while the agent streams)
ctx.ui.set_working_message("Thinking deeply...")
ctx.ui.set_working_message()  # Restore the default
ctx.ui.set_working_visible(False)  # Hide the built-in working row entirely
ctx.ui.set_working_visible(True)  # Show it again

# Working indicator (shown while the agent streams)
ctx.ui.set_working_indicator({"frames": [ctx.ui.theme.fg("accent", "●")]})  # Static dot
ctx.ui.set_working_indicator(
    {
        "frames": [
            ctx.ui.theme.fg("dim", "·"),
            ctx.ui.theme.fg("muted", "•"),
            ctx.ui.theme.fg("accent", "●"),
            ctx.ui.theme.fg("muted", "•"),
        ],
        "intervalMs": 120,
    }
)
ctx.ui.set_working_indicator({"frames": []})  # Hide the indicator
ctx.ui.set_working_indicator()  # Restore the default spinner

# Widget above the editor (default)
ctx.ui.set_widget("my-widget", ["Line 1", "Line 2"])
# Widget below the editor
ctx.ui.set_widget("my-widget", ["Line 1", "Line 2"], {"placement": "belowEditor"})
# Widget from a factory (tui, theme) -> component
ctx.ui.set_widget("my-widget", lambda tui, theme: Text(theme.fg("accent", "Custom"), 0, 0))
ctx.ui.set_widget("my-widget", None)  # Clear

# Custom footer (replaces the built-in footer entirely);
# footer_data provides the git branch and the set_status() texts
ctx.ui.set_footer(lambda tui, theme, footer_data: MyFooter(theme, footer_data))
ctx.ui.set_footer(None)  # Restore the built-in footer

# Custom header, shown at startup above the chat
ctx.ui.set_header(lambda tui, theme: Text(theme.fg("accent", "my-project"), 0, 0))
ctx.ui.set_header(None)  # Restore the built-in header

# Label shown for hidden thinking blocks
ctx.ui.set_hidden_thinking_label("Reasoning...")
ctx.ui.set_hidden_thinking_label()  # Restore the default ("Thinking...")

# Raw terminal input (interactive mode only); returns an unsubscribe function
unsubscribe = ctx.ui.on_terminal_input(lambda data: None)

# Terminal title
ctx.ui.set_title("pidrei - my-project")

# Raw escape sequences to the terminal
ctx.ui.write_terminal("\x1b]777;notify;pidrei;Done\x07")

# Editor text
ctx.ui.set_editor_text("Prefill text")
current = ctx.ui.get_editor_text()

# Paste into the editor (runs paste handling, including collapsing large content)
ctx.ui.paste_to_editor("pasted content")

# Stack custom autocomplete on top of the built-in provider (see below)
ctx.ui.add_autocomplete_provider(lambda current: IssueProvider(current))

# Tool output expansion
was_expanded = ctx.ui.get_tools_expanded()
ctx.ui.set_tools_expanded(True)
ctx.ui.set_tools_expanded(was_expanded)

# Custom editor (vim mode, emacs mode, ...; see below)
ctx.ui.set_editor_component(lambda tui, theme, keybindings: VimEditor(tui, theme, keybindings))
ctx.ui.set_editor_component(None)  # Restore the default editor

# Themes (see themes.md for writing one)
themes = await ctx.ui.get_all_themes()  # [{"name": "dark", "path": "/..." or None}, ...]
light_theme = await ctx.ui.get_theme("light")  # Load without switching; None if unknown
result = await ctx.ui.set_theme("light")  # Switch by name
if not result["success"]:
    ctx.ui.notify(f"Failed: {result['error']}", "error")
ctx.ui.theme.fg("accent", "styled text")  # The current theme

# Several changes as one frame
ctx.ui.apply(lambda: (ctx.ui.set_status("job", "done"), ctx.ui.set_widget("job", None)))
```

Notes on these calls:

- `notify` with `"warning"` or `"error"` shows a warning or error line; any
  other type (or none) shows a status line.
- A widget given as a list of strings shows one `Text` row per string, up to
  10 rows; longer lists end with `... (widget truncated)`. `set_widget` with
  an existing key replaces that widget, wherever it was placed. A component
  that defines `dispose()` gets it called when it is replaced or cleared; the
  same holds for custom footers and headers.
- Working-indicator frames are rendered verbatim. To colour them, put the
  colour in the frame strings yourself, for example with
  `ctx.ui.theme.fg(...)`. `intervalMs` sets the animation speed.
- The footer factory receives a footer data provider for what extensions
  cannot reach otherwise: `get_git_branch()`, `get_extension_statuses()`
  (the `set_status` texts) and `on_branch_change(callback)`, which returns an
  unsubscribe function. Context usage is on `ctx.get_context_usage()`, token
  statistics on `ctx.session_manager`, model information on `ctx.model`.
- The `on_terminal_input` handler is synchronous component code. It receives
  each raw input string before the focused component does, and returns
  `None` to let it through, `{"consume": True}` to swallow it, or
  `{"data": replacement}` to pass something else on.
- Use `ctx.ui.write_terminal(sequence)` for raw escape sequences (desktop
  notifications through OSC 777/99, for instance), never `sys.stdout`: while
  the TUI runs, its output pump is the terminal's only writer, and a direct
  write can land in the middle of a frame. `write_terminal` queues behind
  the pump, and does nothing without a UI. Other stdout/stderr output goes
  through `write_stdout`/`write_stderr` from `pidrei.core.output_guard`: they
  queue the text for pidrei's stdio writer instead of writing on the runtime
  (`print` and `sys.stdout` end up there too, but only as a stand-in for code
  that cannot be changed).
- `get_editor_text()` returns the text with collapsed pastes expanded.
- `set_theme` takes a theme name or a `Theme` object (from `get_theme`).
  Switching by name also saves it as the theme setting. It returns
  `{"success": True}`, or `{"success": False, "error": ...}`.
- `ctx.ui.apply(fn)` runs the synchronous `fn` as one change, so several
  updates show up in the same frame. Nothing can be awaited inside it: an
  `async def` is refused.

Extension statuses, widgets, the footer and header, the custom editor,
autocomplete providers, terminal input listeners and the working-row
settings are reset on `/reload` and when the session is replaced.

### Autocomplete providers

`ctx.ui.add_autocomplete_provider(factory)` stacks custom completion on top
of the built-in slash-command and path provider. The factory receives the
current provider and returns a provider that wraps it. pidrei calls the
factory again each time it rebuilds its provider (when another provider is
added, when the session is bound or reloaded, when a setting such as skill
commands changes), so keep it cheap and keep long-lived state outside it.

A provider is any object with:

- `async get_suggestions(lines, cursor_line, cursor_col, options)`: returns
  `{"items": [...], "prefix": ...}`, or `None` for no suggestions. Items are
  dicts with `value`, `label` and an optional `description`.
  `options["signal"]` is a `CancelToken`, cancelled when the request is
  superseded.
- `apply_completion(lines, cursor_line, cursor_col, item, prefix)`
  (synchronous): returns `{"lines", "cursorLine", "cursorCol"}`.
- optionally `should_trigger_file_completion(lines, cursor_line, cursor_col)`
  and `trigger_characters`, a list of characters that open completion as
  they are typed (such as `"$"` or `"#"`).

The usual pattern: inspect the text before the cursor, return your own
suggestions when your extension's syntax matches, delegate to
`current.get_suggestions(...)` otherwise, and delegate `apply_completion(...)`
unless you need custom insertion.

```python
import re

_ISSUE_RE = re.compile(r"(?:^|[ \t])#([^\s#]*)$")


class IssueProvider:
    trigger_characters = ["#"]

    def __init__(self, current):
        self._current = current

    async def get_suggestions(self, lines, cursor_line, cursor_col, options):
        line = lines[cursor_line] if cursor_line < len(lines) else ""
        match = _ISSUE_RE.search(line[:cursor_col])
        if match is None:
            return await self._current.get_suggestions(lines, cursor_line, cursor_col, options)
        return {
            "prefix": f"#{match.group(1)}",
            "items": [
                {"value": "#2983", "label": "#2983", "description": "Extension API for autocomplete"},
                {"value": "#2753", "label": "#2753", "description": "Reload stale resource settings"},
            ],
        }

    def apply_completion(self, lines, cursor_line, cursor_col, item, prefix):
        return self._current.apply_completion(lines, cursor_line, cursor_col, item, prefix)

    def should_trigger_file_completion(self, lines, cursor_line, cursor_col):
        should = getattr(self._current, "should_trigger_file_completion", None)
        return should(lines, cursor_line, cursor_col) if should is not None else True


async def extension(pi):
    async def on_session_start(_event, ctx):
        ctx.ui.add_autocomplete_provider(IssueProvider)

    pi.on("session_start", on_session_start)
```

`examples/extensions/github_issue_autocomplete.py` is a complete version: it
preloads the latest open GitHub issues with `gh issue list` once per session
and filters them locally for fast `#...` completion. It needs the GitHub CLI
(`gh`) and a GitHub repository checkout.

### Custom components

For complex UI, use `ctx.ui.custom(factory, options=None)`. By default the
component replaces the editor until `done(result)` is called; the editor and
its text come back afterwards. The factory is called as
`factory(tui, theme, keybindings, done)`:

- `tui`: a guarded handle to the TUI (the table below): renders, timers,
  background work, focus and overlays, the terminal size
- `theme`: the current theme, for styling
- `keybindings`: the application's keybinding manager, for matching keys
- `done(value)`: closes the component and resolves the handle with `value`.
  It is safe from anywhere: input handling, a timer, spawned work, even the
  factory itself.

The factory runs, and its component is shown, before `custom()` returns; the
handle it returns yields the value passed to `done`:

```python
from pidrei_tui import truncate_to_width


class ConfirmPrompt:
    def __init__(self, tui, theme, keybindings, done):
        self._theme = theme
        self._keybindings = keybindings
        self._done = done

    def render(self, width):
        return [truncate_to_width(self._theme.fg("accent", "Press Enter to confirm, Escape to cancel"), width)]

    def handle_input(self, data):
        if self._keybindings.matches(data, "tui.select.confirm"):
            self._done(True)
        elif self._keybindings.matches(data, "tui.select.cancel"):
            self._done(False)

    def invalidate(self):
        pass


result = await ctx.ui.custom(ConfirmPrompt)
if result:
    ...  # The user pressed Enter
```

Do any async preparation before calling `custom`, and start work the
component needs from the factory with `tui.spawn(...)`. A component that
defines `dispose()` gets it called when it closes. A custom component does
not report pidrei as waiting for the user; pass
`{"blocked": {"kind": "question", "message": "..."}}` when it does (see
[Overlays and prompts](tui.md#overlays-and-prompts)).
[A minimal custom screen](tui.md#a-minimal-custom-screen) is a complete
example with cached rendering.

#### Overlay mode

Pass `{"overlay": True}` to draw the component as a floating modal on top of
the existing content instead of replacing the editor:

```python
result = await ctx.ui.custom(
    lambda tui, theme, keybindings, done: MyOverlay(theme, on_close=done),
    {"overlay": True},
)
```

For positioning, pass `"overlayOptions"`: a dict with `width`, `minWidth`,
`maxHeight`, `anchor`, `offsetX`, `offsetY`, `row`, `col`, `margin`,
`visible` and `nonCapturing` (sizes and positions take numbers or
percentage strings), or a callable returning one. Without it, a `width`
attribute on the component sets the overlay's width. `"onHandle"` receives
the overlay's handle, for controlling focus and visibility:

```python
def on_handle(handle):
    handle.focus()  # Focus this overlay and bring it to the front
    # handle.unfocus({"target": editor})  # Hand input to a specific component
    # handle.set_hidden(True)  # Toggle visibility
    # handle.is_hidden(), handle.is_focused(), handle.get_bounds()


result = await ctx.ui.custom(
    lambda tui, theme, keybindings, done: MyOverlay(theme, on_close=done),
    {
        "overlay": True,
        "overlayOptions": {"anchor": "top-right", "width": "50%", "margin": 2},
        "onHandle": on_handle,
    },
)
```

Finish the interaction with `done()`; never call `handle.hide()` on an
overlay `custom()` created. Overlays ignore the `blocked` option.

A focused, visible overlay takes input back when temporary non-overlay custom
UI closes. To let another component keep input while the overlay stays
visible, call `handle.unfocus({"target": component})`;
`handle.unfocus({"target": None})` releases the overlay without focusing
anything else. [Overlays and prompts](tui.md#overlays-and-prompts) covers the
lifecycle, and `examples/extensions/overlay_qa_tests.py` exercises every
option.

### Where extension code and component code run

Extension handlers — commands, event handlers, shortcuts, tools — run on their
own coroutines and reach the UI through `ctx.ui`.

Component code is the rest: the factories, and everything the components
they build do — `handle_input`, `render`, timers, and the callbacks they
call. pidrei's UI runs it under its state lock, exactly like pidrei's own
components, and hands it `tui` — a guarded handle, not the TUI itself:

| `tui.` | |
|--------|---|
| `request_render(force=False)` | Ask for a frame |
| `apply(fn)` | Run the synchronous `fn` as one change (same as `ctx.ui.apply`) |
| `spawn(coro)` | Run `coro` in the background; an error it raises is reported like the UI's own |
| `timeout(ms, fn)` / `interval(ms, fn)` | Call the synchronous `fn` once / repeatedly, as one change each time; returns a handle with `cancel()` |
| `set_focus`, `show_overlay`, `hide_overlay` | Focus and overlays; the overlay handle is guarded too |
| `finish_before_next_input(handle)` | From `handle_input` only: hold the next key until the spawned work finishes |
| `await stop()` / `await start()` | Give the terminal away and take it back |
| `terminal.rows` / `terminal.columns` | The terminal's size |
| `terminal.write_sync(data)` | Write to the terminal, queued behind its other output |

The rules:

- `handle_input`, `render` and every callback are synchronous. Change your
  component's state directly there, and call `ctx.ui` freely.
- Anything slow goes to `tui.spawn(...)`. When that work changes your
  component's state, wrap the change in `tui.apply(...)` so a frame never
  shows it half done. `done(result)` needs no wrapping: it is safe from
  anywhere (see `examples/extensions/handoff.py`).
- Use `tui.timeout`/`tui.interval` for timers: their callbacks already run
  as one change.
- When the next key must see the effect of slow work (pi does such work
  synchronously), spawn it with `tonio.spawn(...)` and pass the handle to
  `tui.finish_before_next_input(...)`.
- To give the terminal to another program, spawn the handoff from the
  factory: `await tui.stop()`, run the program, `await tui.start()`, then
  `done(...)` (see `examples/extensions/interactive_shell.py`).

Guard UI work with `ctx.has_ui`, and terminal-only behavior with
`ctx.mode == "tui"`. Keep tool and event logic independent of rendering so it
still works without a UI.

### Custom editor

Replace the main input editor with your own (vim mode, emacs mode, ...):

```python
from pidrei.modes.interactive.components import CustomEditor
from pidrei_tui import matches_key


class VimEditor(CustomEditor):
    def __init__(self, tui, theme, keybindings):
        super().__init__(tui, theme, keybindings)
        self.mode = "insert"  # "normal" | "insert"

    def handle_input(self, data):
        if matches_key(data, "escape") and self.mode == "insert":
            self.mode = "normal"
            return
        if self.mode == "normal" and data == "i":
            self.mode = "insert"
            return
        super().handle_input(data)  # App keybindings and text editing


async def extension(pi):
    async def on_session_start(_event, ctx):
        ctx.ui.set_editor_component(lambda tui, theme, keybindings: VimEditor(tui, theme, keybindings))

    pi.on("session_start", on_session_start)
```

Key points:

- Subclass `CustomEditor` (from `pidrei.modes.interactive.components`), not
  the base `Editor`, to keep the application's keybindings: Escape to abort,
  Ctrl+D, model switching and the rest.
- Forward keys you don't handle to `super().handle_input(data)`.
- `handle_input` and `render` are synchronous component code (see above).
- The factory receives `tui`, the editor theme and `keybindings`.
- Custom editors keep the standalone working row by default. Pass
  `{"embedWorkingStatus": True}` as the fourth `CustomEditor` constructor
  argument (the editor options) to show working status in the editor's top
  border instead.
- The new editor takes over the current text. While an extension dialog is
  open, it appears when the last dialog closes.
- `ctx.ui.set_editor_component(None)` restores the default editor.

To compose with another extension that already replaced the editor, capture
its factory with `ctx.ui.get_editor_component()` before setting yours:

```python
previous = ctx.ui.get_editor_component()
ctx.ui.set_editor_component(
    lambda tui, theme, keybindings: MyEditor(
        tui, theme, keybindings, base=previous(tui, theme, keybindings) if previous else None
    )
)
```

See `examples/extensions/modal_editor.py` for a complete example with a mode
indicator.

### Message and entry rendering

`pi.register_message_renderer(custom_type, renderer)` controls how messages
with that `customType` appear. Use message renderers for content that is
also part of the model's context:

```python
import json

from pidrei_tui import Text


def render_message(message, options, theme):
    text = theme.fg("accent", f"[{message.custom_type}] ") + message.content
    if options["expanded"] and message.details:
        text += "\n" + theme.fg("dim", json.dumps(message.details, indent=2))
    elif message.details:
        text += f" ({options['keyHint']('app.tools.expand', 'to expand')})"
    return Text(text, options["outputPad"], 0)


pi.register_message_renderer("my-extension", render_message)
```

The renderer is synchronous and is called with the message (with
`custom_type`, `content`, `display` and `details` attributes), an options
dict and the theme. The options carry `expanded` (the tool-output expansion
state), `outputPad` (the horizontal padding configured by the outputPad
setting; use it as your component's left pad so custom messages line up with
the rest of the transcript), and the helpers `keyText(keybinding)` and
`keyHint(keybinding, description)` for showing key bindings. Return a
component, or `None` to fall back to the default rendering; a renderer that
raises falls back too.

Messages are sent with `pi.send_message()`:

```python
pi.send_message(
    {
        "customType": "my-extension",  # Matches register_message_renderer
        "content": "Status update",
        "display": True,  # Show in the transcript
        "details": {"level": "info"},  # Available to the renderer
    }
)
```

For content that belongs in the transcript but not in the model's context,
store a custom entry and register an entry renderer instead:

```python
def render_card(entry, options, theme):
    return Text(theme.fg("accent", json.dumps(entry["data"])), 0, 0)


pi.register_entry_renderer("my-card", render_card)

await pi.append_entry("my-card", {"status": "done"})
```

An entry renderer receives the entry as a dict (`customType`, `data`), the
same options without `outputPad`, and the theme. Returning `None` shows
nothing; a renderer that raises shows an error line in its place. See
`examples/extensions/message_renderer.py` and
`examples/extensions/entry_renderer.py`.

To rewrite the Markdown of user text, assistant text and thinking blocks
before pidrei renders it, register a transformer with
[`pi.register_markdown_transformer()`](#piregister_markdown_transformertransformer).

### Theme colours

Every renderer and factory receives a `theme`. [themes.md](themes.md) covers
writing themes and the full list of colour roles; [Theming](tui.md#theming)
covers using them correctly in components.

```python
# Foreground colours
theme.fg("toolTitle", text)  # Tool names
theme.fg("accent", text)  # Highlights
theme.fg("success", text)  # Success (green)
theme.fg("error", text)  # Errors (red)
theme.fg("warning", text)  # Warnings (yellow)
theme.fg("muted", text)  # Secondary text
theme.fg("dim", text)  # Tertiary text

# Text styles
theme.bold(text)
theme.italic(text)
theme.underline(text)
theme.inverse(text)
theme.strikethrough(text)

# Foreground, background and attributes combined
theme.style(text, {"fg": "success", "bg": "toolSuccessBg", "bold": True})
```

`theme.colors` maps every role to a concrete colour, and `theme.appearance`
is `"dark"` or `"light"`.

For syntax highlighting in custom tool renderers:

```python
from pidrei.modes.interactive.theme import get_language_from_path, highlight_code

# Highlight code in an explicit language; returns one string per line
lines = highlight_code("x = 1", "python")

# Detect the language from a file path
lang = get_language_from_path("/path/to/file.rs")  # "rust", or None
highlighted = highlight_code(code, lang)
```

Without a known language, `highlight_code` returns the lines in the plain
code-block colour.

## Error handling

An exception raised while loading an extension is reported as a diagnostic and
that extension is skipped; the rest still load. An exception inside a handler is
reported and the run continues — one broken extension does not take down the
session. Diagnostics appear at startup and after `/reload`. Interactive mode
shows handler errors in the transcript, print and JSON modes write them to
stderr, and RPC mode sends them to the client as `extension_error` records.

Two exceptions fail safe rather than continue: a `tool_call` handler that
raises blocks that tool call, and a `user_bash` handler that raises blocks the
command instead of falling through to local execution.

A tool reports failure by raising from `execute` or by returning an
`AgentToolResult` with `is_error=True`. A raised exception becomes an error
result for the model, and the run continues.

## Mode behaviour

| Mode | Started with | `ctx.mode` | `ctx.has_ui` | UI |
|------|--------------|------------|--------------|-----|
| Interactive | `pidrei` in a terminal | `"tui"` | True | The full TUI |
| RPC | `--mode rpc` | `"rpc"` | True | Dialogs and notifications go to the client over the JSON protocol; `custom()` returns `None`, component factories are ignored |
| JSON | `--mode json` | `"json"` | False | Event stream on stdout; `ctx.ui` is a no-op |
| Print | `-p`, or when stdin or stdout is not a terminal | `"print"` | False | Extensions run but cannot prompt; `ctx.ui` is a no-op |

Check `ctx.mode == "tui"` before TUI-only features: `custom()`, component
factories, custom editors, autocomplete providers, terminal input. Check
`ctx.has_ui` before dialogs and notifications, which work in both the TUI and
RPC mode.

In RPC mode, `select`, `confirm`, `input`, `editor`, `notify`, `set_status`,
`set_widget` with a list of strings, `set_title` and `set_editor_text` are
sent to the client as `extension_ui_request` records, and the dialogs resolve
with the client's `extension_ui_response`. The `timeout` and `signal` options
of `select`, `confirm` and `input` work as in the TUI: `timeout` is also sent
to the client, and both end the wait with the dismissed value. `paste_to_editor` falls back to
`set_editor_text`, and `get_editor_text()` returns `""`. Everything else is
ignored: `get_all_themes()` returns `[]`, `get_theme()` returns `None` and
`set_theme()` reports failure. `examples/extensions/rpc_demo.py` exercises
every method RPC mode supports.

In print and JSON modes every `ctx.ui` call is safe and does nothing; the
dialogs resolve at once with their dismissed value (`None`, or `False` for
`confirm`). A handler that needs an answer should check `ctx.has_ui` and take
a non-interactive path when it is False.

Shortcuts need the interactive TUI. Commands run wherever a prompt that
starts with `/name` arrives: typed in the TUI, sent by an RPC client, or
given as a print-mode message. Tools and event handlers work in every mode.

## Examples

Working extensions live in [`examples/extensions/`](../examples/extensions/)
and are the implementation reference. Start with the smallest one matching
the behaviour you need:

| Goal | Start with | Why |
|------|------------|-----|
| Confirm or block tool calls | `permission_gate.py` | Combines `tool_call`, mode-aware UI and fail-closed behaviour |
| Modify prompt sections for a run | `prompt_customizer.py` | Uses the structured prompt options instead of replacing the whole prompt |
| Keep branch-aware tool state | `todo.py` | Rebuilds its state from persisted entries and renders it in the TUI |
| Register tools after startup | `dynamic_tools.py` | Shares registration code between lifecycle and command handlers |
| Run cancellable model work from a command | `qna.py` | Combines a nested model call, cancellation and custom UI |
| Replace the main editor | `modal_editor.py` | Extends `CustomEditor` while keeping the application's controls |
| Run finalization at shutdown | `auto_commit_on_exit.py` | Does its work in the `session_shutdown` handler |
| Route built-in tools to another machine | `ssh.py` | Reuses the tools' operation interfaces instead of recreating their contracts |
| Add OAuth and provider-specific streaming | `custom_provider_gitlab_duo/` | A complete provider that delegates to built-in wire implementations |
| Trigger compaction from an event | `trigger_compact.py` | Calls compaction from an event handler |
| Rewrite user input while streaming | `input_transform_streaming.py` | Transforms input and streams output |
| Run git and send follow-up messages | `git_merge_and_resolve.py` | `pi.exec`, follow-up messages, blocking I/O off the loop |
| See most of the API in one place | `plan_mode/` | Flags, commands, shortcuts, tool gating, context filtering, widgets |

Run any of them with `pidrei -e <path>`. [Custom providers](custom-provider.md)
covers model-service integrations, [TUI components](tui.md) custom
components, and [packages.md](packages.md) installing or distributing
extensions with other resources.

### All examples

| Example | Description | Key APIs |
|---------|-------------|----------|
| **Tools** |||
| `hello.py` | Minimal tool registration | `register_tool` |
| `question.py` | Tool with user interaction | `register_tool`, `ui.custom`, `render_result` |
| `questionnaire.py` | Multi-step wizard tool | `register_tool`, `ui.custom` |
| `todo.py` | Stateful tool with branch-aware state in `details` | `register_tool`, `render_result`, `register_command`, `on("session_start")`, `on("session_tree")` |
| `dynamic_tools.py` | Register tools after startup and during commands | `register_tool`, `session_start`, `register_command` |
| `structured_output.py` | Final structured-output tool with `terminate=True` | `register_tool`, terminating tool results |
| `truncated_tool.py` | Output truncation | `register_tool`, `truncate_head` |
| `tic_tac_toe.py` | Tool that must run sequentially | `register_tool`, `execution_mode="sequential"`, `ui.custom` |
| `tool_override.py` | Override the built-in read tool | `register_tool` (same name as the built-in) |
| `built_in_tool_renderer.py` | Custom rendering for built-in tools | `create_*_tool_definition`, `render_call`, `render_result`, `render_shell` |
| `minimal_mode.py` | Collapsed display for built-in tools | Built-in tool overrides, `render_result` |
| **Commands** |||
| `pirate.py` | Modify the system prompt per turn | `register_command`, `before_agent_start` |
| `summarize.py` | Conversation summary command | `register_command`, `ui.custom` |
| `handoff.py` | Cross-provider model handoff | `register_command`, `ui.editor`, `ui.custom` |
| `qna.py` | Q&A with custom UI | `register_command`, `ui.custom`, `set_editor_text` |
| `send_user_message.py` | Inject user messages | `register_command`, `send_user_message` |
| `reload_runtime.py` | Reload command, and a tool that requests a reload once the agent settles | `register_command`, `ctx.reload()`, `register_tool`, `on("agent_settled")`, `send_user_message` |
| `shutdown_command.py` | Graceful shutdown command | `register_command`, `ctx.shutdown()` |
| **Events and gates** |||
| `permission_gate.py` | Block dangerous commands | `on("tool_call")`, `ui.select` |
| `project_trust.py` | Decide or defer project trust from a user/global or CLI extension | `on("project_trust")`, trust UI, required trust result |
| `protected_paths.py` | Block writes to specific paths | `on("tool_call")` |
| `confirm_destructive.py` | Confirm session changes | `on("session_before_switch")`, `on("session_before_fork")` |
| `dirty_repo_guard.py` | Warn on a dirty git repo | `on("session_before_*")`, `exec` |
| `input_transform.py` | Transform user input | `on("input")` |
| `input_transform_streaming.py` | Streaming-aware input transform | `on("input")`, `streamingBehavior` |
| `model_status.py` | React to model changes | `on("model_select")`, `set_status` |
| `provider_payload.py` | Inspect payloads and provider response headers | `on("before_provider_request")`, `on("after_provider_response")` |
| `debug_provider.py` | View raw provider stream events | `on("provider_stream_event")`, `append_entry`, `register_entry_renderer` |
| `system_prompt_header.py` | Display system prompt info | `on("agent_start")`, `get_system_prompt` |
| `claude_rules.py` | Load rules from files | `on("session_start")`, `on("before_agent_start")` |
| `prompt_customizer.py` | Add context-aware tool guidance using `systemPromptOptions` | `on("before_agent_start")` |
| `file_trigger.py` | A watched file triggers messages | `send_message` |
| **Compaction and sessions** |||
| `custom_compaction.py` | Custom compaction summary | `on("session_before_compact")` |
| `trigger_compact.py` | Trigger compaction manually | `compact()` |
| `git_checkpoint.py` | Git stash on turns | `on("turn_start")`, `on("tool_result")`, `on("agent_settled")`, `on("session_before_fork")`, `exec` |
| `git_merge_and_resolve.py` | Fetch, merge, and resolve conflicts | `on("agent_end")`, `exec`, `send_user_message` |
| `auto_commit_on_exit.py` | Commit on shutdown | `on("session_shutdown")`, `exec` |
| **UI components** |||
| `status_line.py` | Footer status indicator | `set_status`, session events |
| `working_indicator.py` | Customize the streaming working indicator | `set_working_indicator`, `register_command` |
| `working_message_test.py` | Custom working message and indicator that persist across turns | `set_working_message`, `set_working_indicator` |
| `github_issue_autocomplete.py` | Add `#1234` issue completions on top of the built-in autocomplete by preloading recent open issues from `gh issue list` | `add_autocomplete_provider`, `on("session_start")`, `exec` |
| `custom_footer.py` | Replace the footer entirely | `register_command`, `set_footer` |
| `custom_header.py` | Replace the startup header | `on("session_start")`, `set_header` |
| `modal_editor.py` | Vim-style modal editor | `set_editor_component`, `CustomEditor` |
| `rainbow_editor.py` | Custom editor styling | `set_editor_component` |
| `widget_placement.py` | Widget above/below the editor | `set_widget` |
| `overlay_test.py` | Overlay components | `ui.custom` with overlay options |
| `overlay_qa_tests.py` | Comprehensive overlay tests | `ui.custom`, all overlay options |
| `notify.py` | Native terminal notification when the agent is done | `on("agent_settled")`, `write_terminal` |
| `titlebar_spinner.py` | Spinner in the terminal title while working | `set_title` |
| `hidden_thinking_label.py` | Label for hidden thinking blocks | `set_hidden_thinking_label` |
| `border_status_editor.py` | Status in the editor border | `set_editor_component`, `CustomEditor` |
| `rpc_demo.py` | Every RPC-supported UI method | `ui.*` in RPC mode |
| `timed_confirm.py` | Dialogs with timeout | `ui.confirm` with `timeout`/`signal` |
| `mac_system_theme.py` | Auto-switch theme | `set_theme`, `exec` |
| **Complex extensions** |||
| `plan_mode/` | Full plan mode implementation | All event types, `register_command`, `register_shortcut`, `register_flag`, `set_status`, `set_widget`, `send_message`, `set_active_tools` |
| `preset.py` | Saveable presets (model, tools, thinking) | `register_command`, `register_shortcut`, `register_flag`, `set_model`, `set_active_tools`, `set_thinking_level`, `append_entry` |
| `tools.py` | Toggle tools on/off UI | `register_command`, `set_active_tools`, `SettingsList`, session events |
| **Remote and sandbox** |||
| `ssh.py` | SSH remote execution | `register_flag`, `on("user_bash")`, `on("before_agent_start")`, tool operations |
| `interactive_shell.py` | Run interactive programs from `!` commands | `on("user_bash")`, `ui.custom`, `tui.stop`/`tui.start` |
| `subagent/` | Delegate to sub-agents, each an in-process agent session | `register_tool`, `create_agent_session` |
| **Games** |||
| `snake.py` | Snake game | `register_command`, `ui.custom`, keyboard handling |
| `space_invaders.py` | Space Invaders game | `register_command`, `ui.custom` |
| **Providers** |||
| `custom_provider_anthropic/` | Custom Anthropic proxy | `register_provider` |
| `custom_provider_gitlab_duo/` | GitLab Duo integration | `register_provider` with OAuth |
| `jev_router.py` | Virtual model that routes planning and implementation to different models | `register_virtual_model` |
| **Messages and communication** |||
| `message_renderer.py` | Custom message rendering | `register_message_renderer`, `send_message` |
| `entry_renderer.py` | Transcript-only custom entry rendering | `register_entry_renderer`, `append_entry` |
| `event_bus.py` | Inter-extension events | `pi.events` |
| **Session metadata** |||
| `session_name.py` | Name sessions for the selector | `set_session_name`, `get_session_name` |
| `bookmark.py` | Bookmark entries for /tree | `set_label` |
| **Misc** |||
| `commands.py` | List available slash commands | `get_commands` |
| `dynamic_resources/` | Add skills, prompts, and themes at runtime | `on("resources_discover")` |
| `inline_bash.py` | Expand inline `!{command}` in user prompts | `on("input")`, `exec` |
| `bash_spawn_hook.py` | Adjust bash command, cwd, and env before execution | `create_bash_tool_definition`, `spawn_hook` |
