# Custom providers

There are two ways to add a provider, depending on whether it speaks a protocol
pidrei already implements.

## By configuration

If the endpoint is OpenAI- or Anthropic-compatible, no code is needed — add it
to `models.json`:

```jsonc
{
  "providers": {
    "my-gateway": {
      "name": "My Gateway",
      "baseUrl": "https://gateway.internal/v1",
      "api": "openai-completions",
      "models": [{ "id": "big-model", "contextWindow": 128000 }]
    }
  }
}
```

See [models.md](models.md) for every field. This covers most cases: the majority
of third-party endpoints are OpenAI-compatible.

## From an extension

For a provider with its own wire format, its own auth flow, or a dynamic model
list, register it from an extension.

```python
from pidrei_ai.registry import create_provider


def extension(pi):
    provider = create_provider(
        id="my-provider",
        name="My Provider",
        base_url="https://api.example.com",
        auth=...,
        models=[...],
        api=...,
    )
    pi.register_provider(provider)
```

`create_provider` takes:

| Argument | Purpose |
|----------|---------|
| `id` | Provider id, used everywhere else |
| `name` | Display name |
| `base_url` | API root |
| `headers` | Static headers |
| `auth` | A `ProviderAuth` describing how credentials are obtained |
| `models` | Static model list |
| `fetch_models` | Async callable returning models, for dynamic catalogs |
| `filter_models` | Narrow the list based on the resolved credential |
| `api` | The wire implementation |

`pi.register_provider(name, config)` is the by-name form, taking the same
config shape `models.json` uses. `pi.unregister_provider(name)` removes one.

Registrations made while extensions are still loading are queued and applied
once the model registry exists, so an extension can register at import time
without deferring to an event.

## Authentication

`ProviderAuth` describes what a credential is and how to obtain it — an API key
read from an environment variable, or an OAuth flow. For OAuth, the flow
receives UI callbacks so it can prompt through whatever interface is active,
including RPC clients; `pidrei_ai.auth.oauth` has the built-in flows to read as
prior art.

Credentials are stored in `~/.pidrei/agent/auth.json` at mode `0600` and
refreshed automatically when they expire.

## Attribution headers

pidrei identifies itself to providers that credit the calling application
(OpenRouter's leaderboard, NVIDIA's billing origin). A custom provider gets no
attribution headers unless it adds its own via `headers`. Users can disable
attribution entirely with `PIDREI_PROVIDER_ATTRIBUTION=0`.

## Testing

`pidrei --list-models` shows whether a provider loaded and which models it
offers. `/models` does the same interactively. A provider that raises during
registration is reported as a diagnostic and skipped, leaving the rest working.
