"""Port of pi's OAuth flow loaders (packages/ai/src/auth/oauth/load.ts).

pi imports through a variable specifier so bundlers cannot follow the import
into Node-only flow code; the deferred import inside each function is the same
trick, and matches how `api/*_lazy.py` defers the adapters.

*Deviation:* pi's `registerBundledOAuthFlowLoaders` exists for standalone Bun
binaries that statically embed the flows (`src/bun-oauth.ts`). There is no
bundler in the way here, so the registry and its Python equivalent are dropped
rather than carried as dead code.
"""

from pidrei_ai.auth.types import OAuthAuth


async def load_anthropic_oauth() -> OAuthAuth:
    from pidrei_ai.auth.oauth.anthropic import anthropic_oauth

    return anthropic_oauth


async def load_openai_codex_oauth() -> OAuthAuth:
    from pidrei_ai.auth.oauth.openai_codex import openai_codex_oauth

    return openai_codex_oauth


async def load_github_copilot_oauth() -> OAuthAuth:
    from pidrei_ai.auth.oauth.github_copilot import github_copilot_oauth

    return github_copilot_oauth


async def load_openrouter_oauth() -> OAuthAuth:
    from pidrei_ai.auth.oauth.openrouter import openrouter_oauth

    return openrouter_oauth


async def load_kimi_coding_oauth() -> OAuthAuth:
    from pidrei_ai.auth.oauth.kimi_coding import kimi_coding_oauth

    return kimi_coding_oauth


async def load_xai_oauth() -> OAuthAuth:
    from pidrei_ai.auth.oauth.xai import xai_oauth

    return xai_oauth
