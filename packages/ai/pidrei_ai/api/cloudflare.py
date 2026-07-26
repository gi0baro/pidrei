"""Port of pi's Cloudflare endpoint constants (packages/ai/src/api/cloudflare.ts)."""

# Workers AI direct endpoint.
CLOUDFLARE_WORKERS_AI_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1"

# AI Gateway Unified API. https://developers.cloudflare.com/ai-gateway/usage/unified-api/
CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL = (
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/compat"
)

# AI Gateway -> OpenAI passthrough. Used until /compat supports /v1/responses.
CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL = (
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/openai"
)

# AI Gateway -> Anthropic passthrough.
CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL = (
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/anthropic"
)
