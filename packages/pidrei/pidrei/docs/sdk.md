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

## Talking to a model

```python
import tonio.colored as tonio
from pidrei_ai.providers.all import get_builtin_provider
from pidrei_ai.types import Context, TextContent, UserMessage


async def main():
    provider = get_builtin_provider("anthropic")
    model = provider.models[0]
    context = Context(messages=[UserMessage(content=[TextContent(type="text", text="Hi")])])

    async for event in provider.api.stream_simple(model, context):
        if event.type == "text":
            print(event.text, end="", flush=True)


tonio.run(main())
```

`stream` yields full provider events; `stream_simple` yields a reduced set.
`complete` and `complete_simple` await the whole response instead.

Credentials resolve exactly as they do for the CLI — environment variable,
then `auth.json`. See [providers.md](providers.md).

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

To script a running pidrei rather than build on the libraries, use one of the
process interfaces instead:

- `pidrei --mode rpc` — JSONL requests and responses over stdin/stdout, the
  full session API including extension UI round-trips.
- `pidrei --mode json -p "..."` — one run, structured events on stdout.

These are stable in a way the Python packages are not yet, and they work from
any language.

## Versioning

The packages version together and track pi's releases: `0.82.0.N` is the Nth
pidrei build tracking pi 0.82.0. Until 1.0, treat every import outside the
documented extension API as internal.
