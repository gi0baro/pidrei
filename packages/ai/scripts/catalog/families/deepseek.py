"""DeepSeek V4 models on OpenAI-compatible hosts (pi: catalog/families/deepseek.ts)."""

from catalog.define import ChatModel, merge_compat, merge_thinking


DEEPSEEK_COMPAT = {
    "requiresReasoningContentOnAssistantMessages": True,
    "thinkingFormat": "deepseek",
}

DEEPSEEK_V4_THINKING_LEVEL_MAP: dict[str, str | None] = {
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "max": "max",
}
DEEPSEEK_V4_FLASH_THINKING_LEVEL_MAP: dict[str, str | None] = {**DEEPSEEK_V4_THINKING_LEVEL_MAP, "low": "low"}
# Azure Foundry rejects DeepSeek's own max effort.
_AZURE_DEEPSEEK_V4_THINKING_LEVEL_MAP: dict[str, str | None] = {
    "minimal": None,
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": None,
    "max": None,
}

# Qwen Token Plan serves DeepSeek V4 through Qwen's own thinking format.
_QWEN_FORMAT_PROVIDERS = frozenset({"qwen-token-plan", "qwen-token-plan-cn", "qwen-token-plan-individual"})
# These hosts accept native reasoning efforts instead of DeepSeek's thinking toggle.
_NATIVE_EFFORT_PROVIDERS = frozenset({"openrouter", "opencode"})


def _is_deepseek_v4(model: ChatModel) -> bool:
    return model["api"] == "openai-completions" and "deepseek-v4" in model["id"]


def apply_deepseek_compat_rules(model: ChatModel) -> None:
    """DeepSeek V4 needs its reasoning content replayed on every host."""
    if not _is_deepseek_v4(model) or model["provider"] in _QWEN_FORMAT_PROVIDERS:
        return
    merge_compat(
        model,
        {"requiresReasoningContentOnAssistantMessages": DEEPSEEK_COMPAT["requiresReasoningContentOnAssistantMessages"]}
        if model["provider"] in _NATIVE_EFFORT_PROVIDERS
        else DEEPSEEK_COMPAT,
    )


def apply_deepseek_family_rules(model: ChatModel) -> None:
    """DeepSeek V4 effort levels when the host did not report its own."""
    if not _is_deepseek_v4(model) or "thinkingLevelMap" in model:
        return
    provider = model["provider"]
    if provider == "openrouter":
        merge_thinking(model, {**DEEPSEEK_V4_THINKING_LEVEL_MAP, "xhigh": "xhigh", "max": None})
    elif provider == "azure":
        merge_thinking(model, _AZURE_DEEPSEEK_V4_THINKING_LEVEL_MAP)
    elif provider in ("deepseek", "opencode", "opencode-go") and "deepseek-v4-flash" in model["id"]:
        merge_thinking(model, DEEPSEEK_V4_FLASH_THINKING_LEVEL_MAP)
    else:
        merge_thinking(model, DEEPSEEK_V4_THINKING_LEVEL_MAP)
