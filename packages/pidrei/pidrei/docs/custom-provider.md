# Custom providers

There are two ways to add a provider, depending on whether it speaks a protocol
pidrei already implements. Use the smallest one that works.

| Requirement | Use |
|-------------|-----|
| Models behind an API pidrei implements | `models.json` |
| Change an existing provider's endpoint or headers | `models.json`, or a by-name registration |
| Discover models from a live service | An extension provider with a model refresh |
| A `/login` flow | An extension provider with OAuth auth |
| A wire protocol pidrei does not implement | An extension provider with its own stream |

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
      "apiKey": "$GATEWAY_API_KEY",
      "models": [{ "id": "big-model", "contextWindow": 128000 }]
    }
  }
}
```

See [models.md](models.md) for every field. This covers most cases: the majority
of third-party endpoints are OpenAI-compatible.

## From an extension

For a provider with its own wire format, its own auth flow, or a dynamic model
list, register it from an [extension](extensions.md). It follows the same
loading, trust, reload and error rules as any extension, and it runs in
pidrei's process with access to credentials, prompts, tool definitions and
responses — treat it as trusted code, and never log tokens, authorization
headers or provider payloads.

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
camelCase config shape `models.json` uses (plus `oauth`, `refreshModels` and
`streamSimple`). Prefer a complete provider for anything beyond static endpoint
and model metadata; `models.json` overrides still compose on top of either
form. A by-name registration with only `baseUrl` or `headers` keeps the
provider's built-in models; one with `models` replaces the provider's model
list with its own.

Registrations made while extensions are still loading are queued and applied
once the model registry exists, so an extension can register from its factory
without deferring to an event, and those providers are visible to startup
model selection and `--list-models`. Calls made later take effect
immediately. `pi.unregister_provider(name)` removes the registration and
restores any built-in behavior it replaced.

## Model refresh

`fetch_models` is for catalogs that come from a live service. It receives a
`RefreshModelsContext` and returns the current model list, which pidrei
merges over the static `models` by id and persists in `models-store.json`, so
the last snapshot is restored at the next startup even offline. It is not
called when `context.allow_network` is false; pass `context.cancel` into
blocking I/O so a refresh can be cancelled. A provider that must not persist
its list (a local server's loaded models, say) implements `refresh_models`
itself and publishes only an in-memory `update` through `context.publish`.

The by-name form's `refreshModels` is async and returns camelCase model
definitions, which replace that registration's models.

Set a model's `prompt_cache` (`promptCache` in the by-name form) only when you
know the provider's cache lifetime; without it that model is never
cache-warmed. Likewise set `compat` switches only for differences verified
against the real server.

## Authentication

`ProviderAuth` describes what a credential is and how to obtain it — an API key
read from an environment variable, or an OAuth flow. In the by-name form,
`apiKey` and `headers` values use the `models.json` syntax — `$NAME`, a leading
`!command`, `$$` and `$!` escapes (see [models.md](models.md#keys-and-headers)).

An OAuth provider appears in `/login` once registered. Its flow is UI-neutral:
it receives callbacks to open an authorization URL, show a device code, report
progress, prompt for input or offer a choice of login methods, so it works in
whatever interface is active. Honor the interaction's cancel token during
network requests. `pidrei_ai.auth.oauth` has the built-in flows to read as
prior art.

Credentials are stored in `~/.pidrei/agent/auth.json` at mode `0600` and
refreshed automatically when they expire.

## Streaming

Reuse one of pidrei-ai's API implementations whenever the protocol matches:
pass `anthropic_messages_api()`, `openai_completions_api()`,
`openai_responses_api()`, `google_generative_ai_api()`, `google_vertex_api()`,
`azure_openai_responses_api()`, `mistral_conversations_api()` or
`bedrock_converse_stream_api()` (each from `pidrei_ai.api.<name>_lazy`) as
`api`. The provider still owns auth, base URL, headers, filtering and
discovery, while message conversion, tool handling, usage, cancellation and
compatibility behavior stay the built-in ones.

Write your own stream only when none fits, and study `pidrei_ai/api/` first.
The context it receives is a normalized transcript: read the system prompt and
tools with `get_current_system_prompt(context.messages)` and
`get_current_tools(context.messages)` from `pidrei_ai.utils.transcript`, and
call `collapse_system_messages(context)` if the model cannot take system
messages mid-conversation. A stream must:

1. Build an assistant message with provider, model, timestamp, content and
   zeroed usage.
2. Emit one `start` event once request setup succeeds (setup failures may go
   straight to `error`), then balanced text, thinking and tool-call events,
   updating the message before each event that exposes it. Tool-call
   arguments must be parsed by `toolcall_end`.
3. Finish with usage, cost and a concrete stop reason, then exactly one
   terminal `done` or `error` event; cancellation becomes an aborted result.
   Error and aborted messages need an `error_message`.
4. Call `options.on_payload` before sending (using any replacement payload it
   returns) and `options.on_response` before consuming the response body, and
   pass through `options.cancel` and `options.env`. Extensions' request hooks
   depend on these.

## Context overflow

pidrei compacts and retries when a request fails with a recognized overflow
error. If your service words it differently, rewrite only that provider's
overflow errors from a `message_end` handler in the same extension, prefixing
`error_message` with `context_length_exceeded`. Never rewrite rate limits or
transient failures — those take the normal retry path.

## Attribution headers

pidrei identifies itself to providers that credit the calling application
(OpenRouter's leaderboard, NVIDIA's billing origin). A custom provider gets no
attribution headers unless it adds its own via `headers`. Users can disable
attribution entirely with `PIDREI_PROVIDER_ATTRIBUTION=0`.

## Testing

`pidrei --list-models` shows whether a provider loaded and which models it
offers. `/model` does the same interactively. A provider that raises during
registration is reported as a diagnostic and skipped, leaving the rest working.

Beyond a few manual prompts, cover plain and empty responses, tool calls and
results, images if supported, usage and cost, abort, context overflow,
malformed or partial streams, Unicode boundaries, switching providers
mid-session, and auth refresh and cancellation; pidrei-ai's provider tests
show the behavior expected of built-in providers. `/reload` picks up changes
to a provider extension in a running session.
