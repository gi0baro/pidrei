# Extensions

Extensions are Python modules that add tools, slash commands, keyboard
shortcuts, CLI flags, event handlers, providers, and custom UI. They run
in-process, on the same interpreter as pidrei itself, with your operating
system permissions: an extension can read prompts, tool calls, files,
credentials and session history, so load them only from sources you trust.

This is the largest deliberate divergence from pi, whose extensions are
TypeScript modules loaded through jiti. The hook bus, the event payloads and
the `pi` object mirror pi's one-for-one; the module format cannot.

## Table of contents

- [Quick start](#quick-start)
- [The extension ABI](#the-extension-abi)
- [Where extensions live](#where-extensions-live)
- [Runtime lifecycle](#runtime-lifecycle)
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
setting. Only user and `-e` extensions load early enough to handle the
`project_trust` event.

## Runtime lifecycle

The factory runs whenever extensions load, and some invocations load them
without ever starting a session. So don't start processes, sockets, watchers
or timers in the factory: start them from `session_start`, or from the
command or tool that needs them, and release them in a `session_shutdown`
handler. Keep that cleanup idempotent — quit, reload and session replacement
all converge on it.

An async factory is awaited before startup continues, so it can fetch
configuration or register providers that startup model selection needs.

`/reload` (or `await ctx.reload()` from a command) replaces the whole
extension runtime: code that runs after the reload must not reuse objects or
the `ctx` from the old one.

## Imports

Everything pidrei itself uses is importable:

```python
from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent
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

`pi.on` returns an unsubscribe function that removes only that registration:

```python
async def on_agent_end(event, ctx):
    unsubscribe()
    await update_integration(event["messages"])


unsubscribe = pi.on("agent_end", on_agent_end)
```

Handlers run in extension load order, then registration order within each
extension. Adding or removing a handler does not affect a dispatch already in
progress.

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
  │   ├─► context                  (modify conversation)
  │   ├─► context_with_system      (modify full transcript)
  │   ├─► before_provider_headers  (mutate headers)    │
  │   ├─► before_provider_request  (inspect/replace)   │
  │   ├─► after_provider_response  (status/headers)    │
  │   ├─► message_start / message_update / message_end │
  │   ├─► tool_call                (block or rewrite)  │
  │   ├─► tool_result              (rewrite output)    │
  │   └─► turn_end                 (append entries, continue)
  │       └─► threshold compaction before a naturally required next turn
  │
  ├─► agent_end
  ├─► retry backoff or final-attempt recovery (when selected)
  │   └─► fresh agent_start on successful recovery
  ├─► agent_before_settle          (append entries, continue)
  └─► agent_settled                (final, notification only)
```

### Catalogue

| Event | Fires | Can change |
|-------|-------|-----------|
| `project_trust` | Before project resources load | Grant or refuse trust |
| `session_start` | Startup, reload, and new/resumed/forked sessions (`reason`) | — |
| `session_shutdown` | Before the session is torn down: quit, reload, or session replacement (`reason`) | — (release resources here) |
| `session_before_compact` | Before compaction runs | `{"cancel": True}`, or supply `{"compaction": CompactionResult(...)}` |
| `session_compact` / `session_compact_failed` | After compaction succeeds / after it fails or is aborted | — |
| `session_before_fork` / `session_before_switch` | Before the matching session action | `{"cancel": True}` |
| `session_before_tree` | Before `/tree` navigation | `{"cancel": True}`, a `summary`, or `customInstructions` / `replaceInstructions` / `label` |
| `resources_discover` | Startup and `/reload` | Add resource paths |
| `input` | User submitted input | Transform, or handle it entirely |
| `before_agent_start` | Before the loop starts | Message; prompt sections, tools and rules via the mutable `event["systemPromptOptions"]` (sent as a patch); or the whole system prompt for the run |
| `agent_start` / `agent_end` | Around each low-level run (pidrei may still retry, compact, or continue) | — |
| `turn_start` | Before each provider round-trip | — |
| `turn_end` | After a turn's assistant and tool-result messages are persisted | Append session entries; request one continuation (see [Turn boundaries](#turn-boundaries)) |
| `agent_before_settle` | Last actionable point before the run settles | Append session entries; request one continuation |
| `agent_settled` | The run has settled; nothing continues automatically | — (runs started here begin after every settled handler finishes) |
| `context` | Before each request | The conversation, without system messages (see [Request context](#request-context)) |
| `context_with_system` | Before each request, after every `context` handler | The full transcript, sent as returned |
| `before_provider_headers` | Before each request | Request headers |
| `before_provider_request` | Before each request | The payload |
| `after_provider_response` | After each response arrives | — (inspect status/headers) |
| `cache_warming_decision` | Before each prompt-cache refresh, with pidrei's decision (`warmCost`, `missCost`, `continuationProbability`, `action`) | Return `{"action": "warm"}` or `{"action": "stop"}`; the last handler that returns an action wins; `"stop"` ends warming until the next real request |
| `message_start` / `message_update` / `message_end` | Assistant message stream | `message_end` may rewrite |
| `tool_call` | Before a tool runs | Block it, or rewrite arguments |
| `tool_result` | After a tool runs | Rewrite the result |
| `user_bash` | User ran a `!` command | Return exactly one of `{"operations": ...}` (run through that backend) or a complete `{"result": ...}` (record without running), which stops propagation; `None` passes to the next handler, then to local execution; an invalid result or a raise blocks the command |
| `ui_prompt_start` / `ui_prompt_end` | Around a blocking `ctx.ui` prompt (`select`, `confirm`, `input`, `editor`, `custom`) — nested prompts coalesce into one outer waiting span; handlers are best-effort and not awaited | — |
| `model_select` / `thinking_level_select` | Selection changed | — |

Handlers that return `None` leave the event unchanged. Handlers that return a
value replace the corresponding field — the table's "can change" column says
which; a return value on a notification-only event has no effect. Ordering
follows extension load order. `tool_result` handlers compose, each seeing the
previous handler's changes.

Tool calls from one assistant message can run in parallel, so a `tool_call` or
`tool_result` handler must not assume a sibling call or its result exists yet.
For nested work owned by the active turn use `ctx.signal`; commands and idle
session events usually have none.

### Changing the system prompt

`before_agent_start` receives the prompt as text (`event["systemPrompt"]`) and
as structured sections (`event["systemPromptOptions"]`). Prefer editing the
sections, selected tools or guidelines in place: pidrei then records only the
difference in the transcript. Returning `{"systemPrompt": ...}` (or setting
`force_system_prompt` on the options) replaces the whole prompt for that run;
providers receive the forced text as their leading system prompt.

### Blocking a tool call

```python
def extension(pi):
    async def guard(event, _ctx):
        if event["toolName"] == "bash" and "rm -rf" in event["input"].get("command", ""):
            return {"block": True, "reason": "refusing a destructive command"}

    pi.on("tool_call", guard)
```

A blocked call may also set `"terminate": True` to hint that the agent should
stop after the current tool batch. The turn only ends early when *every*
finalized tool result in that batch sets it.

### Transforming input

```python
def extension(pi):
    async def expand(event, _ctx):
        if event["text"].startswith("??"):
            return {"text": f"Explain in detail: {event['text'][2:]}"}

    pi.on("input", expand)
```

### Turn boundaries

`turn_end` and `agent_before_settle` are actionable boundaries. `turn_end`
runs after the turn's assistant and tool-result messages are persisted;
`agent_before_settle` runs after retries and recovery, just before the run
settles. Each handler receives:

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
proposal. Handlers run in load and registration order, each seeing the
previous proposals. The complete proposal is validated, then appended in
order after all handlers finish; a handler error is reported and later
handlers still run.

```python
def extension(pi):
    replaced = False

    async def on_turn_end(event, _ctx):
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

    pi.on("turn_end", on_turn_end)
```

`"continue": True` ensures one next provider request: tool results, steering
or a follow-up satisfy it, otherwise pidrei makes one context-only request.
Error and aborted responses remain hard exits, and `"continue": False` never
suppresses natural work. Guard the condition — an unconditional continuation
fires again after the next response. If the run is aborted while
`agent_before_settle` handlers run, valid entries are still committed but the
continuation is dropped.

### Request context

`context` handlers receive a deep copy of the conversation *without* system
messages; return `{"messages": [...]}` or edit `event["messages"]` in place. The
prompt and tool declarations belong to pidrei: when the list changed, the
current prompt sections and tool declarations are replayed into one leading
system message ahead of it, so filtering, windowing or slicing from a
compaction summary cannot drop them. An unchanged list keeps mid-conversation
system messages in place (preserving the cached prefix). System messages a
handler adds are kept after the head. To change the prompt or tools durably,
use `before_agent_start` or `pi.set_active_tools()`.

`context_with_system` runs after every `context` handler on the full
transcript, and its result is sent as returned — the handler owns the prompt
and tool declarations for that request:

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
output is still sent). A `systemPrompt` forced from `before_agent_start` is
still projected onto the request afterwards.

## ExtensionContext

The second handler argument. The useful members:

| Member | Purpose |
|--------|---------|
| `ctx.cwd` | Working directory of the session |
| `ctx.mode` | `"tui"`, `"print"` or `"json"` |
| `ctx.has_ui` | False in print and JSON modes — check before waiting on `ctx.ui` |
| `ctx.ui` | UI surface (below) |
| `ctx.model` / `ctx.thinking_level` | Current model and reasoning level |
| `ctx.signal` | Cancel token of the active operation, or `None` when idle |
| `ctx.session_manager` | Session entries and metadata |
| `ctx.scoped_models` | Read-only list of models scoped to the session (from `--models` / `enabledModels`, the set `/scoped-models` shows); empty when no scoping is configured |
| `ctx.model_registry` | Models, providers and resolved authentication, plus streaming model calls (below) |
| `ctx.is_idle()` | Whether the agent is between runs |
| `ctx.has_pending_messages()` | Whether queued messages are waiting |
| `ctx.get_context_usage()` / `ctx.compact(options)` | Context-window usage; start a compaction |
| `ctx.abort()` / `ctx.shutdown()` | Abort the current operation; request an orderly shutdown |

**Streaming model calls.** Use `ctx.model_registry.stream_simple(model,
context, options)` for provider-neutral options such as `reasoning`, or
`stream()` for API-specific options. Both use the configured providers and
resolve authentication, including for providers registered with
`pi.register_provider()`. Both return an `AssistantMessageEventStream`: iterate
it with `async for` for response events and `await stream.result()` for the
final message. Setup failures produce error events and error results.

`ctx.ui` offers `notify(text, level)`, `set_status(...)`, `set_widget(...)`,
`select(title, options)`, `confirm(title, message)`,
`input(title, placeholder)`, `editor(title, prefill)` and `theme`.
The `select`/`confirm`/`input`/`editor`/`custom` calls are awaitable and
return the user's choice, or `None` if dismissed. `paste_to_editor` and the
theme accessors (`get_all_themes`, `get_theme`, `set_theme`) are awaitable
too; the remaining setters are plain sync calls.

In print and JSON modes `ctx.has_ui` is False and `ctx.ui` is a no-op object,
so handlers stay safe to call unconditionally — but a handler that *waits* on
user input should check `ctx.has_ui` first, or it will wait forever on
nothing.

Command handlers receive a wider context that adds `wait_for_idle()`,
`reload()`, `navigate_tree(target_id, options)`, `new_session(options)`,
`switch_session(path, options)` and `fork(entry_id, options)`. These are
command-only because calling them from an event handler can deadlock the
runtime. Replacing the session invalidates the old context (using it raises):
capture plain data beforehand and do post-replacement work in
`{"withSession": callback}`, an async callable that receives the fresh
context.

## The `pi` object

### Registration

| Method | Purpose |
|--------|---------|
| `on(event, handler)` | Subscribe to an event; returns an unsubscribe function |
| `register_tool(tool)` | Add a tool the model can call |
| `register_command(name, *, handler, description=None, get_argument_completions=None)` | Add a slash command |
| `register_shortcut(shortcut, *, handler, description=None)` | Bind a key |
| `register_flag(name, *, type, description=None, default=None)` | Add a CLI flag |
| `register_message_renderer(custom_type, renderer)` | Render a custom message |
| `register_markdown_transformer(transformer)` | Rewrite Markdown before Pidrei renders it |
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
| `await set_model(model)` | Switch the model for the current session (recorded in the session, restored on resume; the configured default for new sessions is unchanged). Returns `False` when the provider has no auth configured |
| `get_thinking_level()` / `await set_thinking_level(level)` | Reasoning level, clamped to the model; the setter is session-scoped like `set_model` |
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
`send_user_message` also accepts `{"expandPromptTemplates": True}` to dispatch
extension commands and expand skill commands and prompt templates (default:
`False`).

### Providers

`register_provider(name, config)` registers by name and config;
`register_provider(provider)` registers a constructed provider object.
`unregister_provider(name)` removes one. Registrations made while extensions
are still loading are queued and flushed once the model registry exists, so an
extension does not need to defer them; later calls take effect immediately.
See [custom-provider.md](custom-provider.md).

## Custom tools

A tool is a `ToolDefinition` with a JSON-schema parameter spec and an execute
function. It must return an `AgentToolResult` — a bare dict is not accepted.

```python
from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent


def extension(pi):
    async def execute(_tool_call_id, params, _cancel, _on_update, _ctx):
        return AgentToolResult(
            content=[TextContent(type="text", text=f"echo: {params['text']}")],
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

Execute receives `(tool_call_id, params, cancel, on_update, ctx)`: the call id,
the validated arguments, a cancel token, a streaming-update callback, and the
extension context.

The `content` list is what the model sees. `details` is arbitrary data for your
own renderers and state reconstruction, not sent to the model; leave it `None`
when there is nothing structured. A tool that makes nested model calls should
put their `usage` on the result so session totals stay accurate.

Raise from `execute` to produce a failed tool result — returning a result
never marks it as an error. Set `terminate=True` only to skip the automatic
follow-up request; it takes effect only when every finished tool in the batch
sets it.

Tools run in parallel by default. Set `execution_mode="sequential"` when tools
share mutable in-memory state, and wrap a file tool's whole read-modify-write
in `with_file_mutation_queue(path, fn, queue_key=...)` from
`pidrei.core.tools`, resolving the key first with
`await resolve_mutation_queue_key(path)` (in
`pidrei.core.tools.file_mutation_queue`) so no filesystem call runs on the
runtime. Truncate large outputs and tell the model where to read the rest.

To offer tools on demand, register all of them up front, keep the optional
ones inactive, and switch with `pi.set_active_tools(names)` — from a loader
tool, say. Names must already be registered; unknown ones are ignored. The
change is appended to the transcript as a system message before the model's
next request; a model that cannot take system messages mid-conversation gets
them folded into the leading one instead, which can invalidate the cached
prefix.

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

Session-bound state belongs in the session itself; pick the store by how it
relates to the conversation:

| State | Store |
|-------|-------|
| Tool state that should follow the active branch | The tool result's `details` |
| Durable data kept out of model context | `await pi.append_entry(...)` |
| Custom content stored *and* sent to the model | `pi.send_message(...)` |
| Data beyond one session | A file under `ctx.cwd` or the agent directory |

Rebuild branch-sensitive state in `session_start` from
`ctx.session_manager.get_branch()`, not from every entry in the file —
abandoned branches are alternative histories. Register an entry or message
renderer when stored custom content should appear in the transcript. For
state shared between extensions, use `pi.events`.

## Custom UI

`ctx.ui.set_widget(...)` places a persistent component; `ctx.ui.set_status(...)`
sets a one-line status; renderers registered with
`register_message_renderer` / `register_entry_renderer` control how custom
messages and entries appear in the transcript. Message renderers receive an
`options` dict carrying `expanded` and `outputPad` (the horizontal padding
configured by the outputPad setting — use it as your component's left pad so
custom messages line up with the rest of the transcript).

Components come from `pidrei_tui` — `Container`, `Text`, `Spacer`,
`SelectList`, `SettingsList` and friends. Use `ctx.ui.custom()` only when an
interaction needs its own rendering and input; [tui.md](tui.md) covers
components, focus, overlays, theming and performance.

Guard UI work with `ctx.has_ui`, and terminal-only behavior with
`ctx.mode == "tui"`. Keep tool and event logic independent of rendering so it
still works without a UI.

### register_markdown_transformer(transformer)

Rewrite the Markdown of normal user text, assistant text, and thinking blocks
before Pidrei renders it. Transformers run in extension load order and each one
receives the Markdown the previous one returned; the built-in renderer then
renders the result.

The transformer is called with the Markdown string and a context dict:

- `messageType` — `"user"`, `"assistant"`, or `"assistant-thinking"`
- `isStreaming` — `True` for partial assistant updates; `False` for user,
  finalized assistant, and restored messages
- `availableWidth` — the exact terminal columns available for the content

```python
def transformer(markdown, context):
    if context["isStreaming"] or context["messageType"] == "assistant-thinking":
        return markdown
    return markdown.replace("-->", "→")


pi.register_markdown_transformer(transformer)
```

If a transformer raises, Pidrei keeps the Markdown produced so far and continues
with the next one. The hook is display-only: the session and the model context
keep the original message. It runs for new user messages, assistant streaming
updates, restored session messages, and terminal width changes — so unlike every
other extension callback it is **synchronous**, and it must stay inexpensive.

`pidrei_tui.lex_markdown(text)` exposes the bundled Markdown lexer (pi exports
its bundled `Marked` here) if a transformer needs to inspect the token stream.

## Error handling

An exception raised while loading an extension is reported as a diagnostic and
that extension is skipped; the rest still load. An exception inside a handler is
reported and the run continues — one broken extension does not take down the
session. Diagnostics appear at startup and after `/reload`.

Two exceptions fail safe rather than continue: a `tool_call` handler that
raises blocks that tool call, and a `user_bash` handler that raises blocks the
command instead of falling through to local execution. A tool whose `execute`
raises becomes an error result for the model.

## Mode behaviour

| Mode | Extensions | UI |
|------|-----------|-----|
| Interactive | Loaded | Full |
| Print (`-p`) | Loaded | `has_ui` False |
| JSON (`--mode json`) | Loaded | `has_ui` False |

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
