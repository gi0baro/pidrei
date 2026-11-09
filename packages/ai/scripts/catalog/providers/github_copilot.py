"""GitHub Copilot (pi: catalog/providers/github-copilot.ts)."""

import copy
import re

from catalog.define import (
    ChatModel,
    ProviderContext,
    ProviderDefinition,
    add_missing,
    merge_compat,
    merge_thinking,
    models_dev_models,
)
from catalog.families.anthropic import claude_55_thinking_levels, supports_claude_mid_convo_system_messages
from catalog.families.openai import (
    enable_openai_grammar_tools,
    enable_openai_mid_convo_system_messages,
    openai_standard_cost,
)
from catalog.sources.models_dev import ModelsDevModel


_BASE_URL = "https://api.individual.githubcopilot.com"
_STATIC_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}

# GitHub's "Models with extended capabilities" table lists these Copilot models as supporting
# the extended 1 million token context window.
_EXTENDED_CONTEXT_MODEL_IDS = frozenset(
    {
        "claude-fable-5",
        "claude-opus-4.8",
        "claude-opus-5",
        "claude-opus-5.5",
        "claude-sonnet-4.6",
        "claude-sonnet-5",
        "gpt-5.3-codex",
        "gpt-6-astra",
        "gpt-6-luna",
        "gpt-6-sol",
    }
)

_EAGER_TOOL_INPUT_STREAMING_UNSUPPORTED_MODEL_IDS = frozenset(
    {
        "claude-haiku-4.5",
        "claude-sonnet-4",
        "claude-sonnet-4.5",
    }
)

# Checked manually against the authenticated GitHub Copilot /models endpoint on 2026-06-15.
# Keep this to narrow corrections over models.dev metadata instead of snapshotting Copilot's catalog.
_THINKING_LEVEL_OVERRIDES: dict[str, dict[str, str | None]] = {
    "claude-opus-4.8": {"minimal": "low"},
    "claude-opus-5": {"minimal": "low"},
    "claude-sonnet-4.6": {"minimal": "low", "max": "max"},
}

_MESSAGES_RE = re.compile(r"^claude-(haiku|sonnet|opus|fable)-[45]([.-]|$)")
_RESPONSES_RE = re.compile(r"^(gpt-|grok-|oswe|mai-)")


def _copilot_api(model_id: str) -> str:
    """Claude 4.x and 5.x use Messages; GPT, Grok, OSWE, and MAI-Code are only served on /responses."""
    if _MESSAGES_RE.search(model_id) is not None:
        return "anthropic-messages"
    if _RESPONSES_RE.search(model_id) is not None:
        return "openai-responses"
    return "openai-completions"


# The authenticated Copilot catalog advertised these models on 2026-09-22,
# but models.dev did not include them yet.
_GPT_6_LUNA: ChatModel = {
    "id": "gpt-6-luna",
    "name": "GPT-6 Luna",
    "api": "openai-responses",
    "provider": "github-copilot",
    "baseUrl": _BASE_URL,
    "reasoning": True,
    "input": ["text", "image"],
    "cost": openai_standard_cost("gpt-6-luna"),
    "contextWindow": 1000000,
    "maxTokens": 128000,
    "headers": {**_STATIC_HEADERS},
}


def _map_model(model: ChatModel, _source: ModelsDevModel) -> ChatModel:
    api = _copilot_api(model["id"])
    result: ChatModel = {**model, "api": api}
    if api == "openai-completions":
        result["compat"] = {
            "supportsStore": False,
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
        }
    if model["id"] == "claude-opus-5.5":
        merge_thinking(result, claude_55_thinking_levels(model["id"]))
    return result


def _models(ctx: ProviderContext) -> list[ChatModel]:
    return [
        {**model, "contextWindow": 1000000} if model["id"] in _EXTENDED_CONTEXT_MODEL_IDS else model
        for model in add_missing(
            models_dev_models(
                ctx,
                "github-copilot",
                api="openai-completions",
                base_url=_BASE_URL,
                headers=_STATIC_HEADERS,
                skip_deprecated=True,
                default_context_window=128000,
                default_max_tokens=8192,
                map_model=_map_model,
            ),
            [copy.deepcopy(_GPT_6_LUNA)],
        )
    ]


def _finalize(model: ChatModel, _ctx: ProviderContext) -> None:
    if model["id"].startswith("gpt-5"):
        merge_thinking(model, {"minimal": "low"})
    override = _THINKING_LEVEL_OVERRIDES.get(model["id"])
    if override is not None:
        merge_thinking(model, override)
    if model["api"] == "anthropic-messages":
        # Copilot forwards mid-conversation system messages but rejects
        # `tool_addition`/`tool_removal` blocks, so tool changes stay top-level.
        if supports_claude_mid_convo_system_messages(model["id"]):
            merge_compat(model, {"supportsMidConvoSystemMessages": True})
        if model["id"] in _EAGER_TOOL_INPUT_STREAMING_UNSUPPORTED_MODEL_IDS:
            merge_compat(model, {"supportsEagerToolInputStreaming": False})
    # Copilot forwards Kimi K3's system text but silently drops its tool-bearing message.
    if model["api"] == "openai-completions" and model["id"] == "kimi-k3":
        merge_compat(model, {"supportsMidConvoSystemMessages": True})
    # Copilot passes mid-conversation developer messages and `additional_tools`
    # through to OpenAI unchanged; tool search is not verified through it.
    enable_openai_mid_convo_system_messages(model, proxy=True)
    enable_openai_grammar_tools(model)


PROVIDER = ProviderDefinition(
    id="github-copilot",
    models=_models,
    compat={"openai-completions": {"supportsStrictMode": True}},
    finalize=_finalize,
)
