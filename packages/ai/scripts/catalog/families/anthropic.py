"""Anthropic Claude models: facts that hold on every host that serves them
(pi: catalog/families/anthropic.ts).

Ids appear dashed (claude-opus-4-8), dotted (claude-opus-4.8), prefixed
(anthropic/claude-opus-4.8), and dated (claude-opus-5-20260723).
"""

import re

from catalog.define import ChatModel, ProviderContext, merge_compat, merge_thinking
from catalog.sources.models_dev import normalize_models_dev_model


def _strip_prefix(model_id: str) -> str:
    return re.sub(r"^~?anthropic/", "", model_id.lower())


_MID_CONVO_EFFORT_OPUS_RE = re.compile(r"^claude-opus-(?:5|5[.-]5)(?:-\d{8})?$")
_MID_CONVO_EFFORT_SONNET_HAIKU_RE = re.compile(r"^claude-(?:sonnet|haiku)-5[.-]5(?:-\d{8})?$")
_MID_CONVO_EFFORT_FABLE_RE = re.compile(r"^claude-(?:fable|mythos)-5[.-][12](?:-\d{8})?$")


def supports_claude_mid_convo_effort(model_id: str) -> bool:
    """Models that accept per-message `output_config` effort changes."""
    model_id = _strip_prefix(model_id)
    return bool(
        _MID_CONVO_EFFORT_OPUS_RE.search(model_id)
        or _MID_CONVO_EFFORT_SONNET_HAIKU_RE.search(model_id)
        or _MID_CONVO_EFFORT_FABLE_RE.search(model_id)
    )


_MID_CONVO_SYSTEM_OPUS_RE = re.compile(r"^claude-opus-(?:4[.-]8|5(?:[.-]5)?)(?:-\d{8})?$")
_MID_CONVO_SYSTEM_SONNET_HAIKU_RE = re.compile(r"^claude-(?:sonnet|haiku)-5[.-]5(?:-\d{8})?$")
_MID_CONVO_SYSTEM_FABLE_RE = re.compile(r"^claude-(?:fable|mythos)-5(?:[.-][12])?(?:-\d{8})?$")


def supports_claude_mid_convo_system_messages(model_id: str) -> bool:
    """Models that accept system messages and tool changes after the conversation starts."""
    return bool(
        _MID_CONVO_SYSTEM_OPUS_RE.search(model_id)
        or _MID_CONVO_SYSTEM_SONNET_HAIKU_RE.search(model_id)
        or _MID_CONVO_SYSTEM_FABLE_RE.search(model_id)
    )


def enable_claude_mid_convo_effort(model: ChatModel) -> None:
    """Enable per-message effort changes."""
    merge_compat(model, {"supportsMidConvoEffort": True})


def _id_includes_any(model_id: str, fragments: tuple[str, ...]) -> bool:
    return any(fragment in model_id for fragment in fragments)


_ADAPTIVE_THINKING = (
    "opus-4-6",
    "opus-4.6",
    "opus-4-7",
    "opus-4.7",
    "opus-4-8",
    "opus-4.8",
    "opus-5",
    "opus.5",
    "sonnet-4-6",
    "sonnet-4.6",
    "sonnet-5",
    "sonnet.5",
    "haiku-5",
    "haiku.5",
    "fable-5",
    "mythos-5",
)

_CLAUDE_LATEST_ALIAS_RE = re.compile(r"^~anthropic/claude-[a-z]+-latest$")


def _is_claude_latest_alias(model_id: str) -> bool:
    """OpenRouter's `~anthropic/claude-*-latest` aliases follow the newest model of each family,
    all of which use adaptive thinking."""
    return _CLAUDE_LATEST_ALIAS_RE.search(model_id) is not None


_TEMPERATURE_UNSUPPORTED = (
    "opus-4-7",
    "opus-4.7",
    "opus-4-8",
    "opus-4.8",
    "opus-5",
    "opus.5",
    "sonnet-5",
    "sonnet.5",
    "haiku-5-5",
    "haiku-5.5",
)
_MAX_EFFORT_ONLY = ("opus-4-6", "opus-4.6", "sonnet-4-6", "sonnet-4.6")
_XHIGH_AND_MAX_EFFORT = (
    "opus-4-7",
    "opus-4.7",
    "opus-4-8",
    "opus-4.8",
    "opus-5",
    "opus.5",
    "sonnet-5",
    "sonnet.5",
    "haiku-5",
    "haiku.5",
)


def apply_claude_family_rules(model: ChatModel, ctx: ProviderContext) -> None:
    """Adaptive thinking effort levels (per Anthropic adaptive thinking docs):
    "max" is available on all adaptive-thinking Claude models; "xhigh" only on
    Opus 4.7/4.8/5, Sonnet 5, Haiku 5.5, and Fable 5."""
    model_id = model["id"]
    if _id_includes_any(model_id, _MAX_EFFORT_ONLY):
        merge_thinking(model, {"max": "max"})
    if _id_includes_any(model_id, _XHIGH_AND_MAX_EFFORT):
        merge_thinking(model, {"xhigh": "xhigh", "max": "max"})
    if "fable-5" in model_id:
        merge_thinking(model, {"xhigh": "xhigh", "max": "max"})
    if _cannot_turn_thinking_off(model, ctx):
        merge_thinking(model, {"off": None})
    if model["api"] != "anthropic-messages":
        return
    if _is_claude_latest_alias(model_id) or _id_includes_any(model_id, _ADAPTIVE_THINKING):
        merge_compat(model, {"forceAdaptiveThinking": True})
    if _id_includes_any(model_id.lower(), _TEMPERATURE_UNSUPPORTED):
        merge_compat(model, {"supportsTemperature": False})


def _cannot_turn_thinking_off(model: ChatModel, ctx: ProviderContext) -> bool:
    """Whether Anthropic's own models.dev entry for this Claude model lists no thinking toggle. Such
    models reject `thinking: {type: "disabled"}` on every host (Opus 5.5, Sonnet 5.5, Fable 5.x),
    but most other hosts' entries list no toggle on any Claude model (#12429). Hosts built from
    models.dev name Anthropic's entry in `canonical_id`. Vercel's gateway, which has no thinking
    metadata of its own, spells Anthropic's id dotted, `anthropic/`-prefixed, and with `-batch` or
    `-fast` variants."""
    source = ctx.models_dev_source(model)
    # pi's `?.canonicalId ?? ...`: only a missing source or id falls back, not an empty string.
    canonical_id = source.canonical_id if source is not None else None
    if canonical_id is None:
        canonical_id = f"anthropic/{_anthropic_model_id(model['id'])}"
    if not canonical_id.startswith("anthropic/"):
        return False
    entry = ((ctx.sources.models_dev.get("anthropic") or {}).get("models") or {}).get(canonical_id[len("anthropic/") :])
    if not entry:
        return False
    # None when the entry does not reason: pi's `options !== undefined` keeps "off" then.
    options = normalize_models_dev_model(entry).reasoning_options
    return options is not None and not any(option.get("type") == "toggle" for option in options)


_BATCH_FAST_SUFFIX_RE = re.compile(r"[-:](?:batch|fast)\Z")
_DOTTED_VERSION_RE = re.compile(r"(\d)\.(\d)")


def _anthropic_model_id(model_id: str) -> str:
    return _DOTTED_VERSION_RE.sub(r"\1-\2", _BATCH_FAST_SUFFIX_RE.sub("", _strip_prefix(model_id)))


# models.dev may list Opus 5.5, Sonnet 5.5, and Haiku 5.5 before their effort metadata is complete.
CLAUDE_55_THINKING_LEVELS: dict[str, str | None] = {
    "minimal": None,
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}
