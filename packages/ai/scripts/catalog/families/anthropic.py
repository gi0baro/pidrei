"""Anthropic Claude models: facts that hold on every host that serves them
(pi: catalog/families/anthropic.ts).

Ids appear dashed (claude-opus-4-8), dotted (claude-opus-4.8), prefixed
(anthropic/claude-opus-4.8), and dated (claude-opus-5-20260723).
"""

import re

from catalog.define import ChatModel, merge_compat, merge_thinking


def _strip_prefix(model_id: str) -> str:
    return re.sub(r"^~?anthropic/", "", model_id.lower())


_MID_CONVO_EFFORT_OPUS_RE = re.compile(r"^claude-opus-(?:5|5[.-]5)(?:-\d{8})?$")
_MID_CONVO_EFFORT_SONNET_HAIKU_RE = re.compile(r"^claude-(?:sonnet|haiku)-5[.-]5(?:-\d{8})?$")
_MID_CONVO_EFFORT_FABLE_RE = re.compile(r"^claude-(?:fable|mythos)-5(?:[.-]1)(?:-\d{8})?$")


def supports_claude_mid_convo_effort(model_id: str) -> bool:
    """Models that accept per-message `output_config` effort changes."""
    model_id = _strip_prefix(model_id)
    return bool(
        _MID_CONVO_EFFORT_OPUS_RE.search(model_id)
        or _MID_CONVO_EFFORT_SONNET_HAIKU_RE.search(model_id)
        or _MID_CONVO_EFFORT_FABLE_RE.search(model_id)
    )


# Claude models that accept `thinking: {type: "disabled"}` alongside per-message effort.
# Opus 5.5, Sonnet 5.5, Fable 5.1, and Mythos 5.1 reject it.
# https://platform.claude.com/docs/en/build-with-claude/thinking#configuring-thinking
_DISABLED_THINKING_MODELS = frozenset({"claude-opus-5", "claude-haiku-5-5", "claude-haiku-5.5"})

_MID_CONVO_SYSTEM_OPUS_RE = re.compile(r"^claude-opus-(?:4[.-]8|5(?:[.-]5)?)(?:-\d{8})?$")
_MID_CONVO_SYSTEM_SONNET_HAIKU_RE = re.compile(r"^claude-(?:sonnet|haiku)-5[.-]5(?:-\d{8})?$")
_MID_CONVO_SYSTEM_FABLE_RE = re.compile(r"^claude-(?:fable|mythos)-5(?:[.-]1)?(?:-\d{8})?$")


def supports_claude_mid_convo_system_messages(model_id: str) -> bool:
    """Models that accept system messages and tool changes after the conversation starts."""
    return bool(
        _MID_CONVO_SYSTEM_OPUS_RE.search(model_id)
        or _MID_CONVO_SYSTEM_SONNET_HAIKU_RE.search(model_id)
        or _MID_CONVO_SYSTEM_FABLE_RE.search(model_id)
    )


def enable_claude_mid_convo_effort(model: ChatModel) -> None:
    """Enable per-message effort changes. Models that cannot disable thinking with it lose "off"."""
    merge_compat(model, {"supportsMidConvoEffort": True})
    if re.sub(r"-\d{8}$", "", _strip_prefix(model["id"])) not in _DISABLED_THINKING_MODELS:
        merge_thinking(model, {"off": None})


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
    "sonnet-5-5",
    "sonnet-5.5",
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


def apply_claude_family_rules(model: ChatModel) -> None:
    """Adaptive thinking effort levels (per Anthropic adaptive thinking docs):
    "max" is available on all adaptive-thinking Claude models; "xhigh" only on
    Opus 4.7/4.8/5, Sonnet 5, Haiku 5.5, and Fable 5."""
    model_id = model["id"]
    if _id_includes_any(model_id, _MAX_EFFORT_ONLY):
        merge_thinking(model, {"max": "max"})
    if _id_includes_any(model_id, _XHIGH_AND_MAX_EFFORT):
        merge_thinking(model, {"xhigh": "xhigh", "max": "max"})
    if "fable-5" in model_id:
        merge_thinking(model, {"off": None, "xhigh": "xhigh", "max": "max"})
    if model["api"] != "anthropic-messages":
        return
    if _is_claude_latest_alias(model_id) or _id_includes_any(model_id, _ADAPTIVE_THINKING):
        merge_compat(model, {"forceAdaptiveThinking": True})
    if _id_includes_any(model_id.lower(), _TEMPERATURE_UNSUPPORTED):
        merge_compat(model, {"supportsTemperature": False})


def claude_55_thinking_levels(model_id: str) -> dict[str, str | None]:
    """models.dev may list Opus 5.5, Sonnet 5.5, and Haiku 5.5 before their effort
    metadata is complete. Haiku 5.5 accepts disabled thinking; Opus 5.5 and
    Sonnet 5.5 reject it."""
    return {
        **({} if "haiku" in model_id else {"off": None}),
        "minimal": None,
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
        "max": "max",
    }
