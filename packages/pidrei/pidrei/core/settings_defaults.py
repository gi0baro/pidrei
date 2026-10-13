"""Mirror of pi coding-agent src/core/settings-defaults.ts.

Runtime defaults shared by settings accessors and the published JSON Schema.
PiDrei's keys: no analytics or install telemetry (recipe `no-telemetry`;
`enableProviderAttribution` replaces `enableInstallTelemetry`), no Mermaid
rendering, and codemode's `typeCheck`.
"""

from types import MappingProxyType
from typing import Any

from pidrei_ai.utils.retry import DEFAULT_MAX_AGENT_RETRY_DELAY_MS


def _frozen(value: dict[str, Any]) -> MappingProxyType:
    return MappingProxyType({key: _frozen(item) if isinstance(item, dict) else item for key, item in value.items()})


SETTINGS_DEFAULTS = _frozen(
    {
        "transport": "auto",
        "steeringMode": "one-at-a-time",
        "followUpMode": "one-at-a-time",
        "compaction": {
            "enabled": True,
            "reserveTokens": 16384,
            "keepRecentTokens": 20000,
        },
        "branchSummary": {
            "reserveTokens": 16384,
            "skipPrompt": False,
        },
        "retry": {
            "enabled": True,
            "maxRetries": 3,
            "baseDelayMs": 2000,
            "maxAgentDelayMs": DEFAULT_MAX_AGENT_RETRY_DELAY_MS,
            "provider": {
                "maxRetryDelayMs": 60000,
            },
        },
        "hideThinkingBlock": False,
        "showCacheMissNotices": False,
        "quietStartup": False,
        "defaultProjectTrust": "ask",
        "collapseChangelog": False,
        "enableProviderAttribution": True,
        "enableSkillCommands": True,
        "terminal": {
            "showImages": True,
            "imageWidthCells": 60,
            "clearOnShrink": False,
            "showTerminalProgress": False,
        },
        "images": {
            "autoResize": True,
            "blockImages": False,
        },
        "doubleEscapeAction": "tree",
        "treeFilterMode": "default",
        "editorPaddingX": 0,
        "outputPad": 1,
        "autocompleteMaxVisible": 5,
        "markdown": {
            "codeBlockIndent": "  ",
        },
        "warnings": {
            "anthropicExtraUsage": True,
        },
        "codemode": {
            "mode": "on",
            "inlineBudget": 3000,
            "typeCheck": True,
        },
        "cacheWarming": "streaming",
        "tuiMode": "fullscreen",
        "fullscreenExitOutput": "transcript",
        "fullscreenScrollbar": "auto",
        "fullscreenCopyOnSelect": True,
        "fullscreenWheelScrollLines": "auto",
    }
)
