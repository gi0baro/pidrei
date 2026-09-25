# Library use

pidrei is a set of importable Python packages as well as a CLI. Nothing here is
a stable public API yet — pidrei is alpha, and these modules exist to serve the
CLI first — but they are usable, and extensions import them freely.

| Package | Contains |
|---------|----------|
| `pidrei_ai` | Model types, the provider registry, wire adapters, auth |
| `pidrei_agent` | The agent loop, tools, harness resources |
| `pidrei_tui` | Terminal UI ([tui.md](tui.md)) |
| `pidrei` | CLI internals: sessions, modes, extensions, config |

Every package is async and runs on [tonio](https://github.com/gi0baro/tonio),
not asyncio. Entry points look like:

```python
import tonio.colored as tonio


async def main(): ...


tonio.run(main())
```

Blocking work belongs in `tonio.spawn_blocking(fn, *args)`.

## Embedding a session

`create_agent_session` builds the same `AgentSession` the CLI runs: one
conversation with its model, tools, queues, compaction state and extensions.

```python
import tonio.colored as tonio
from pidrei.core.sdk import create_agent_session


async def main():
    session = (await create_agent_session()).session
    try:
        await session.prompt("What files are in the current directory?")
        print(session.get_last_assistant_text())
    finally:
        session.dispose()


tonio.run(main())
```

Without options it uses the current directory, discovered resources, stored
settings and configured credentials, and persists the session like the CLI
does. `prompt()` returns when the run finishes, automatic retries included.

`CreateAgentSessionOptions` replaces any boundary: `cwd` (project discovery,
context files, session grouping and tool paths — pass it when it differs from
the process's), `model_runtime`, `model`, `thinking_level`, `scoped_models`,
`settings_manager`, `session_manager`, `resource_loader`, and `tools`,
`no_tools`, `exclude_tools`, `custom_tools`. A `DefaultResourceLoader` keeps
normal discovery with selected overrides, and its `extension_factories` loads
in-process extensions (each factory an `async def` taking `pi`, like a module's
`extension`); supply your own loader only if the host owns resource
discovery entirely.

Read state through `session.messages`, `session.model`,
`session.thinking_level`, `session.system_prompt` (the effective prompt,
including changes not yet sent) and `session.get_active_tool_names()`.

### History

The `SessionManager` owns the entry tree and its active leaf, and is
authoritative for the model's context: to restore history, build the session
with a manager that holds it — assigning agent state messages does not
replace persisted context. `SessionManager.in_memory()` skips session files.
Sessions keep pi's JSONL format; [message-types.md](message-types.md)
describes the transcript values.

`session.dispose()` aborts active work, invalidates extension contexts and
drops listeners; call it when done. `create_agent_session_runtime` (in
`pidrei.core.agent_session_runtime`) adds `new_session`, `switch_session`,
`fork` and `import_from_jsonl`; each replaces the active `AgentSession`, so
subscriptions must be re-bound afterwards.

### Prompting and events

A prompt sent while the session is streaming must say whether it steers the
current run or follows it —
`PromptOptions(streaming_behavior="steer" | "followUp")` — or it is rejected.
`steer()` and `follow_up()` do the same directly. `abort()` stops the active
operation and waits for idle; `wait_for_idle()` waits without aborting.

`session.subscribe(listener)` takes a plain (non-async) callable and returns
an unsubscribe function; subscribe before prompting to stream output:

```python
def on_event(event):
    if event.type == "message_update" and event.assistant_message_event.type == "text_delta":
        print(event.assistant_message_event.delta, end="", flush=True)


unsubscribe = session.subscribe(on_event)
```

`message_end` carries the authoritative finished message. `agent_end` ends
one low-level run, but retries, compaction or queued messages may follow;
`agent_settled` means pidrei will not continue on its own.

## Talking to a model

```python
import time

import tonio.colored as tonio
from pidrei.core.model_runtime import ModelRuntime
from pidrei_ai.types import Context, SimpleStreamOptions, TextContent, UserMessage


async def main():
    runtime = await ModelRuntime.create()
    model = runtime.get_model("anthropic", "claude-sonnet-4-5")
    message = UserMessage(content=[TextContent(text="Hi")], timestamp=int(time.time() * 1000))

    stream = runtime.stream_simple(model, Context(messages=[message]), SimpleStreamOptions(reasoning="low"))
    async for event in stream:
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)


tonio.run(main())
```

`ModelRuntime` resolves credentials and `models.json` exactly as the CLI does
(see [models.md](models.md#keys-and-headers)). `stream_simple` takes a
provider-neutral thinking level; `stream` takes provider-specific options.
Both yield the same events, and `complete` / `complete_simple` await the
final message instead. `pidrei_ai.providers.all.builtin_models()` is the bare
collection without pidrei's config: credentials come only from the
environment.

## Registering a provider

`pidrei_ai.registry.create_provider(...)` builds one from parts:

```python
from pidrei_ai.registry import create_provider

provider = create_provider(
    id="my-provider",
    name="My Provider",
    base_url="https://api.example.com/v1",
    auth=...,
    models=[...],
    api={...},
)
```

From inside pidrei, register it with `pi.register_provider(provider)` — see
[custom-provider.md](custom-provider.md).

## Driving pidrei itself

To drive pidrei from another process or language rather than import it, run
it as a subprocess — `pidrei -p "..."` for text, `pidrei --mode json -p "..."`
for structured events on stdout. See [cli-integration.md](cli-integration.md).
The process interface is stable in a way the Python packages are not yet.

## Versioning

The packages version together and track pi's releases: `0.82.0.N` is the Nth
pidrei build tracking pi 0.82.0. Until 1.0, treat every import outside the
documented extension API as internal.
