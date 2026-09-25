# Custom models

`~/.pidrei/agent/models.json` adds providers and models, or overrides fields on
built-in ones. pidrei reads it at startup and never writes it, so it is yours to
hand-edit. `//` and `/* */` comments are allowed.

Keys are camelCase because they are the same shape as the upstream catalog
(models.dev and pi's), so definitions copy across between the two projects
unchanged.

## Shape

```jsonc
{
  "providers": {
    "my-provider": {
      "name": "My Provider",
      "baseUrl": "https://api.example.com/v1",
      "api": "openai-completions",
      "apiKey": "sk-...",           // prefer an env var or /login
      "headers": { "X-Extra": "1" },
      "models": [
        {
          "id": "my-model-large",
          "name": "My Model (large)",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 200000,
          "maxTokens": 8192,
          "cost": { "input": 3.0, "output": 15.0, "cacheRead": 0.3, "cacheWrite": 3.75 }
        }
      ]
    }
  }
}
```

Only `providers` is required at the top level, and only `id` is required on a
model.

## Provider fields

| Field | Meaning |
|-------|---------|
| `name` | Display name |
| `baseUrl` | API root |
| `api` | Wire protocol to speak (see below) |
| `apiKey` | Literal key — an environment variable or `/login` is safer |
| `headers` | Extra headers on every request |
| `authHeader` | Send the key as `Authorization` rather than the API's default |
| `compat` | Compatibility switches for near-compliant endpoints |
| `models` | Model definitions |
| `modelOverrides` | Patch fields on models that already exist |

`api` picks the wire format, not the vendor: `openai-completions`,
`openai-responses`, `anthropic-messages`, `google-generative-ai` and the other
adapters pidrei implements. Most third-party endpoints are
`openai-completions`.

## Model fields

| Field | Meaning |
|-------|---------|
| `id` | Model id sent to the API — **required** |
| `name` | Display name |
| `reasoning` | Whether the model supports reasoning |
| `thinkingLevelMap` | Map pidrei's levels onto the API's own values |
| `input` | Modalities: `text`, `image`, `audio` |
| `contextWindow` | Total context in tokens |
| `maxTokens` | Maximum output tokens |
| `cost` | `input` / `output` / `cacheRead` / `cacheWrite` per million tokens, and optional `tiers` |
| `inputLimits` | Request limits and image preprocessing for this model (see below) |
| `promptCache` | Best-effort prompt cache lifetime in seconds per retention tier (see below) |
| `headers` | Extra headers for this model only |
| `compat` | Per-model compatibility switches |

## Image input limits

`inputLimits.images.resize` configures how new images are encoded before they
enter conversation history:

```jsonc
{
  "id": "vision-model",
  "input": ["text", "image"],
  "inputLimits": {
    "images": { "resize": { "maxWidth": 1568, "maxHeight": 1568, "maxBytes": 524288, "jpegQuality": 75 } }
  }
}
```

`maxBytes` is the maximum base64-encoded payload size. Omitted fields keep the
conservative defaults (2000×2000, 4.5 MiB encoded, JPEG quality 80). The
selected model's profile applies to `@file` attachments, the `read` tool and
images returned by tools; images are encoded once, so switching models does
not rewrite history. `images.autoResize` in settings disables resizing. A
`modelOverrides` entry deep-merges `inputLimits`. The catalog may also record
`maxRequestBytes`, `images.maxPerMessage` and `images.maxPerRequest`; those are
not enforced yet.

## Prompt cache lifetimes

`promptCache` states how long the provider keeps a prompt cache entry alive
for each retention tier pidrei can request (`short` is the default tier;
`long` is used when `PIDREI_CACHE_RETENTION=long`). Values are seconds and are
estimates: providers publish ranges, so pick the conservative end.

```jsonc
{ "id": "claude-sonnet-5", "promptCache": { "short": 300, "long": 3600 } }
```

Cache warming (the `cacheWarming` setting: `"off"`, `"streaming"` — the
default — or `"idle"`, also in `/settings`) re-sends the last request with a
one-token output budget shortly before the entry expires, when the expected
avoided cache-miss cost outweighs the refresh. A model without a lifetime for
the tier a request used is never warmed. The built-in catalog fills this in
for direct Anthropic only; a `promptCache` override opts in a proxy whose
backing cache you know (merged per tier).

## Overriding a built-in model

`modelOverrides` patches without redefining:

```jsonc
{
  "providers": {
    "anthropic": {
      "modelOverrides": {
        "claude-sonnet-4-5": { "maxTokens": 16384 }
      }
    }
  }
}
```

## Tiered pricing

```jsonc
"cost": {
  "input": 1.25,
  "output": 10.0,
  "tiers": [
    { "inputTokensAbove": 200000, "input": 2.5, "output": 15.0 }
  ]
}
```

## Validating

A malformed `models.json` does not stop pidrei — it reports the failing path at
startup and continues without your custom entries. `/models` lists everything
that loaded; `--list-models` prints the same from the command line.
