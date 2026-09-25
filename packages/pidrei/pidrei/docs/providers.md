# Providers

pidrei supports subscription providers via OAuth and API-key providers via
environment variables or the auth file. Amazon Bedrock and Google Vertex AI can
also use ambient cloud credentials. `/login [provider]` shows the methods a
provider supports.

pidrei starts from the catalog bundled with the package and may overlay newer
catalog data from pi.dev for configured providers. Refreshed catalogs are
cached in `~/.pidrei/agent/models-store.json` and stay available offline;
`pidrei update --models` forces a refresh.

40 providers are built in — pi's set minus `radius`, which is a pi-specific
service that does nothing without pi's own credentials.

## Subscriptions

Run `/login` in interactive mode and pick a provider:

- ChatGPT Plus/Pro (Codex)
- Claude Pro/Max
- GitHub Copilot
- xAI (Grok/X subscription)
- Meta (Muse subscription)
- OpenRouter (OAuth mints an API key billed from OpenRouter credits)
- Kimi For Coding

Tokens live in `~/.pidrei/agent/auth.json` (mode `0600`) and refresh
automatically when they expire. OpenRouter instead mints a user-controlled API
key that does not expire on its own.

`/logout` removes the stored credential for a provider. It does not unset
environment variables, remove an `apiKey` from `models.json`, or revoke the
credential at the provider.

Claude, ChatGPT and OpenRouter sign in through a browser and a loopback
callback. On a remote or headless machine (e.g. over SSH) the browser cannot
reach that callback; paste the final redirect URL (or the authorization code)
into the login prompt instead. GitHub Copilot, xAI, Meta and Kimi use device
flows: open the shown URL on any machine and enter the code.

### Claude Pro/Max

Anthropic subscription auth works for Claude Pro/Max accounts. Third-party
harness usage draws from [extra usage](https://claude.ai/settings/usage) and is
billed per token rather than against plan limits. pidrei warns about this the
first time; the warning can be turned off in `/settings`.

### GitHub Copilot

Press Enter for github.com, or type your GitHub Enterprise Server domain. If a
model reports as unsupported, enable it once in VS Code: Copilot Chat → model
selector → the model → "Enable".

### xAI and OpenRouter

`/login xai` and `/login openrouter` each offer both a subscription flow and an
API-key path; `XAI_API_KEY` and `OPENROUTER_API_KEY` keep working either way.

### Meta (Muse subscription)

`/login meta` → **Sign in with Meta** opens a device authorization flow. The
login mints a Model API key that is re-minted automatically about once a day;
`META_API_KEY` remains available through the API-key path.

## API keys

Either store one with `/login`, or export it before starting:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pidrei
```

| Provider | Environment variable | `auth.json` key |
|----------|----------------------|-----------------|
| Anthropic | `ANTHROPIC_API_KEY` (or `ANTHROPIC_OAUTH_TOKEN`) | `anthropic` |
| Ant Ling | `ANT_LING_API_KEY` | `ant-ling` |
| OpenAI | `OPENAI_API_KEY` | `openai` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | `azure-openai-responses` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek` |
| NVIDIA NIM | `NVIDIA_API_KEY` | `nvidia` |
| Google Gemini | `GEMINI_API_KEY` | `google` |
| Google Vertex AI | `GOOGLE_CLOUD_API_KEY` | `google-vertex` |
| GitHub Copilot | `COPILOT_GITHUB_TOKEN` | `github-copilot` |
| Amazon Bedrock | `AWS_BEARER_TOKEN_BEDROCK` | `amazon-bedrock` |
| Mistral | `MISTRAL_API_KEY` | `mistral` |
| Groq | `GROQ_API_KEY` | `groq` |
| Cerebras | `CEREBRAS_API_KEY` | `cerebras` |
| Fireworks | `FIREWORKS_API_KEY` | `fireworks` |
| Together AI | `TOGETHER_API_KEY` | `together` |
| Baseten | `BASETEN_API_KEY` | `baseten` |
| Hugging Face | `HF_TOKEN` | `huggingface` |
| MiniMax | `MINIMAX_API_KEY` | `minimax` |
| MiniMax (China) | `MINIMAX_CN_API_KEY` | `minimax-cn` |
| Moonshot AI (global / China) | `MOONSHOT_API_KEY` | `moonshotai` / `moonshotai-cn` |
| Kimi For Coding | `KIMI_API_KEY` | `kimi-coding` |
| Meta | `META_API_KEY` | `meta` |
| Cloudflare AI Gateway | `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_GATEWAY_ID` | `cloudflare-ai-gateway` |
| Cloudflare Workers AI | `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` | `cloudflare-workers-ai` |
| xAI | `XAI_API_KEY` | `xai` |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter` |
| Vercel AI Gateway | `AI_GATEWAY_API_KEY` | `vercel-ai-gateway` |
| ZAI Coding Plan (global) | `ZAI_API_KEY` | `zai` |
| ZAI Coding Plan (China) | `ZAI_CODING_CN_API_KEY` | `zai-coding-cn` |
| OpenCode Zen | `OPENCODE_API_KEY` | `opencode` |
| OpenCode Go | `OPENCODE_API_KEY` | `opencode-go` |
| Qwen Token Plan | `QWEN_TOKEN_PLAN_API_KEY` | `qwen-token-plan` |
| Qwen Token Plan (Individual) | `QWEN_TOKEN_PLAN_API_KEY` | `qwen-token-plan-individual` |
| Qwen Token Plan (China) | `QWEN_TOKEN_PLAN_CN_API_KEY` | `qwen-token-plan-cn` |
| Xiaomi MiMo | `XIAOMI_API_KEY` | `xiaomi` |
| Xiaomi MiMo Token Plan (China) | `XIAOMI_TOKEN_PLAN_CN_API_KEY` | `xiaomi-token-plan-cn` |
| Xiaomi MiMo Token Plan (Amsterdam) | `XIAOMI_TOKEN_PLAN_AMS_API_KEY` | `xiaomi-token-plan-ams` |
| Xiaomi MiMo Token Plan (Singapore) | `XIAOMI_TOKEN_PLAN_SGP_API_KEY` | `xiaomi-token-plan-sgp` |

Anthropic also accepts `ANTHROPIC_AUTH_TOKEN`, sent as an
`Authorization: Bearer` header instead of an API key; it wins over the other two
Anthropic variables. `pidrei --help` prints the authoritative list.

### Azure OpenAI

Azure needs the endpoint as well as the key:

```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_BASE_URL=https://<resource>.openai.azure.com
# or AZURE_OPENAI_RESOURCE_NAME=<resource>
export AZURE_OPENAI_DEPLOYMENT_NAME_MAP="gpt-4o=my-deployment,gpt-4o-mini=my-mini"
```

`AZURE_OPENAI_API_VERSION` defaults to `v1`. Resource root URLs under
`openai.azure.com`, `cognitiveservices.azure.com` and `ai.azure.com` are
normalized to the `/openai/v1` API path.

### Amazon Bedrock

Either `AWS_BEARER_TOKEN_BEDROCK`, or the standard AWS credential chain:
`AWS_PROFILE`, or `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (plus
`AWS_SESSION_TOKEN` for temporary credentials), ECS task credentials
(`AWS_CONTAINER_CREDENTIALS_*`) or IRSA (`AWS_WEB_IDENTITY_TOKEN_FILE`). The
region comes from `AWS_REGION` or `AWS_DEFAULT_REGION` when the profile does
not supply one.

### Cloudflare

AI Gateway needs `CLOUDFLARE_API_KEY`, `CLOUDFLARE_ACCOUNT_ID` and
`CLOUDFLARE_GATEWAY_ID`; Workers AI needs the first two. The IDs can come from
the environment or the credential's `env` object (see below). The key
authenticates pidrei to the gateway only; upstream access uses Cloudflare
unified billing, keys stored in the gateway, or an `Authorization` header you
configure for the provider in [models.json](models.md).

### Google Vertex AI

Set `GOOGLE_CLOUD_API_KEY`, or use Application Default Credentials
(`gcloud auth application-default login`, or `GOOGLE_APPLICATION_CREDENTIALS`
pointing at a service-account key file) together with `GOOGLE_CLOUD_PROJECT`
(or `GCLOUD_PROJECT`) and `GOOGLE_CLOUD_LOCATION`.

## Auth file

`~/.pidrei/agent/auth.json` maps a provider key to its credential. `/login`
writes it at mode `0600`; it holds API keys and OAuth tokens, so keep it
private and never commit it.

An API-key entry may name a command instead of a literal key, so a secret
manager's output never lands on disk:

```json
{
  "anthropic": { "type": "api_key", "key": "!security find-generic-password -ws 'anthropic'" }
}
```

The command runs when the key is first needed and its stdout is cached for the
process lifetime. Empty output, a nonzero exit or the 10-second timeout leaves
the key unresolved until pidrei restarts. `$NAME` interpolation and escapes
work as in [models.json](models.md).

An API-key entry can also carry an `env` object. Its values override the
process environment for that provider only — account IDs, Azure or Vertex
settings, proxies:

```json
{
  "cloudflare-workers-ai": {
    "type": "api_key",
    "key": "...",
    "env": { "CLOUDFLARE_ACCOUNT_ID": "account-id" }
  }
}
```

## Resolution order

For a given provider, the first that exists wins:

1. `--api-key` on the command line
2. The credential stored in `auth.json` (API key or OAuth token)
3. `apiKey` from `models.json`
4. The provider's environment variables or ambient cloud credentials

A stored credential owns the provider: environment variables are not consulted
as a fallback when it fails to resolve or refresh. Environment variables must
be set in the process that starts pidrei. A model whose provider has no
credential is hidden from the model selector.

## Custom providers

To point at an OpenAI- or Anthropic-compatible endpoint, add an entry to
`models.json` — see [models.md](models.md). To implement a provider with its
own API or OAuth flow, register it from an extension — see
[custom-provider.md](custom-provider.md).

## Offline

`--offline`, or `PIDREI_OFFLINE=1`, disables automatic network activity: no
catalog refresh, no version check and no package update check. The bundled
catalog and the cached catalogs in `models-store.json` still serve every model
lookup. An explicit `pidrei update --models` still refreshes.
