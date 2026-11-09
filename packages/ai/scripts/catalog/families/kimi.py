"""Moonshot Kimi models, shared by Moonshot, Kimi Coding, and gateways
(pi: catalog/families/kimi.ts)."""

import re
from typing import Any


_KIMI_K3_FAMILY_RE = re.compile(r"^kimi-k3(?:\.\d+)?(?:-|$)")


def is_kimi_k3_family_id(model_id: str) -> bool:
    """Kimi K3 and its point releases (kimi-k3.1-pro, kimi-k3.1-pro-highspeed) share
    K3's request format. https://api.moonshot.ai/v1/models reports effort levels
    and dynamic tool loading for all of them."""
    return _KIMI_K3_FAMILY_RE.search(model_id) is not None


KIMI_K3_MAX_TOKENS = 131072
KIMI_K3_COST: dict[str, Any] = {
    "input": 3,
    "output": 15,
    "cacheRead": 0.3,
    "cacheWrite": 0,
}

# Kimi Chat Completions settings shared by Moonshot and Kimi Coding.
KIMI_OPENAI_COMPAT: dict[str, Any] = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "maxTokensField": "max_tokens",
    "supportsStrictMode": False,
    "thinkingFormat": "deepseek",
}

# Kimi K3-style models take reasoning_effort and need earlier reasoning sent back.
KIMI_K3_OPENAI_COMPAT: dict[str, Any] = {
    **KIMI_OPENAI_COMPAT,
    "requiresReasoningContentOnAssistantMessages": True,
    "thinkingFormat": "openai",
    "supportsReasoningEffort": True,
}
