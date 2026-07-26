# Providers

pidrei supports subscription providers via OAuth and API-key providers via
environment variables or the auth file. Built-in catalogs ship with the
package; configured providers may refresh newer catalogs, cached in
`~/.pidrei/agent/models-store.json` for offline use.

37 providers are built in — pi's set minus `radius`, which is a pi-specific
service that does nothing without pi's own credentials.

## Subscriptions

Run `/login` in interactive mode and pick a provider:

- ChatGPT Plus/Pro (Codex)
- Claude Pro/Max
- GitHub Copilot
- xAI (Grok/X subscription)
- OpenRouter (OAuth mints an API key billed from OpenRouter credits)
- Kimi For Coding

`/logout` clears credentials. Tokens live in `~/.pidrei/agent/auth.json` (mode
`0600`) and refresh automatically when they expire. OpenRouter instead mints a
user-controlled API key that does not expire on its own.

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
| Amazon Bedrock | `AWS_BEARER_TOKEN_BEDROCK` | `amazon-bedrock` |
| Mistral | `MISTRAL_API_KEY` | `mistral` |
| Groq | `GROQ_API_KEY` | `groq` |
| Cerebras | `CEREBRAS_API_KEY` | `cerebras` |
| Fireworks | `FIREWORKS_API_KEY` | `fireworks` |
| Together AI | `TOGETHER_API_KEY` | `together` |
| MiniMax | `MINIMAX_API_KEY` | `minimax` |
| Moonshot | `MOONSHOT_API_KEY` | `moonshot` |
| Kimi For Coding | `KIMI_API_KEY` | `kimi-coding` |
| Cloudflare AI Gateway | `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_GATEWAY_ID` | `cloudflare-ai-gateway` |
| Cloudflare Workers AI | `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` | `cloudflare-workers-ai` |
| xAI | `XAI_API_KEY` | `xai` |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter` |
| Vercel AI Gateway | `AI_GATEWAY_API_KEY` | `vercel-ai-gateway` |
| ZAI Coding Plan (global) | `ZAI_API_KEY` | `zai` |
| ZAI Coding Plan (China) | `ZAI_CODING_CN_API_KEY` | `zai-coding-cn` |
| OpenCode Zen | `OPENCODE_API_KEY` | `opencode` |
| OpenCode Go | `OPENCODE_API_KEY` | `opencode-go` |
| Qwen Token Plan | `QWEN_TOKEN_PLAN_API_KEY` / `QWEN_TOKEN_PLAN_CN_API_KEY` | `qwen-token-plan` |
| Xiaomi MiMo | `XIAOMI_API_KEY` (and the regional Token Plan keys) | `xiaomi` |

`pidrei --help` prints the authoritative list.

### Azure OpenAI

Azure needs the endpoint as well as the key:

```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_BASE_URL=https://<resource>.openai.azure.com
# or AZURE_OPENAI_RESOURCE_NAME=<resource>
export AZURE_OPENAI_DEPLOYMENT_NAME_MAP="gpt-4o=my-deployment,gpt-4o-mini=my-mini"
```

`AZURE_OPENAI_API_VERSION` defaults to `v1`.

### Amazon Bedrock

Either `AWS_BEARER_TOKEN_BEDROCK`, or the standard AWS credential chain
(`AWS_PROFILE`, or `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`), plus
`AWS_REGION`.

## Auth file

`~/.pidrei/agent/auth.json` maps a provider key to its credential. It is
written by `/login` at mode `0600` and is not meant to be hand-edited; use
`/login` and `/logout`.

## Resolution order

For a given provider, the first that exists wins:

1. `--api-key` on the command line
2. The provider's environment variable
3. `auth.json`

A model whose provider has no credential is hidden from the model selector.

## Custom providers

To point at an OpenAI- or Anthropic-compatible endpoint, add an entry to
`models.json` — see [models.md](models.md). To implement a provider with its
own API or OAuth flow, register it from an extension — see
[custom-provider.md](custom-provider.md).

## Offline

`--offline`, or `PIDREI_OFFLINE=1`, disables every startup network operation:
no catalog refresh and no update check. Cached catalogs in `models-store.json`
are still used.
