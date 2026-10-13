"""Mirror of pi coding-agent src/core/settings-schema.ts.

The settings.json schema behind the published configuration schema. pi
builds it with TypeBox; here it is the JSON Schema TypeBox emits, as plain
dicts. The settings manager never validates against it at runtime (pi's
does not either): it documents the file for editors.

It describes PiDrei's settings, not pi's: no `npmCommand` (packages install
from git and local paths), no analytics or install telemetry (recipe
`no-telemetry`; `enableInstallTelemetry` is a legacy key migrated to
`enableProviderAttribution`), no Mermaid rendering, and codemode's
`typeCheck`.
"""

from typing import Any

from pidrei_ai.providers.model_schema import MODEL_THINKING_LEVEL_SCHEMA

from .settings_defaults import SETTINGS_DEFAULTS


_MAX_SAFE_INTEGER = 2**53 - 1


def _literals(*values: Any, **options: Any) -> dict[str, Any]:
    return {
        "anyOf": [
            {
                "type": "boolean" if isinstance(value, bool) else "number" if isinstance(value, int) else "string",
                "const": value,
            }
            for value in values
        ],
        **options,
    }


def _record(value: dict[str, Any], **options: Any) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": value, **options}


def _non_negative_safe_integer(**options: Any) -> dict[str, Any]:
    return {"type": "integer", "minimum": 0, "maximum": _MAX_SAFE_INTEGER, **options}


def _timeout_setting(**options: Any) -> dict[str, Any]:
    return {"anyOf": [{"type": "number", "minimum": 0}, {"type": "string", "const": "disabled"}], **options}


_STRING_ARRAY = {"type": "array", "items": {"type": "string"}}

_COMPACTION_MODEL_OVERRIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "reserveTokens": _non_negative_safe_integer(),
        "keepRecentTokens": _non_negative_safe_integer(),
    },
}

_COMPACTION_SETTINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "enabled": {"type": "boolean", "default": SETTINGS_DEFAULTS["compaction"]["enabled"]},
        "reserveTokens": _non_negative_safe_integer(default=SETTINGS_DEFAULTS["compaction"]["reserveTokens"]),
        "keepRecentTokens": _non_negative_safe_integer(default=SETTINGS_DEFAULTS["compaction"]["keepRecentTokens"]),
        "modelOverrides": _record(
            _COMPACTION_MODEL_OVERRIDE_SCHEMA,
            description='Per-model overrides keyed by exact "provider/modelId" strings.',
        ),
    },
}

_BRANCH_SUMMARY_SETTINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "reserveTokens": {
            "type": "number",
            "description": "Tokens reserved for the prompt and LLM response.",
            "default": SETTINGS_DEFAULTS["branchSummary"]["reserveTokens"],
        },
        "skipPrompt": {
            "type": "boolean",
            "description": 'When true, skips the "Summarize branch?" prompt and defaults to no summary.',
            "default": SETTINGS_DEFAULTS["branchSummary"]["skipPrompt"],
        },
    },
}

_PROVIDER_RETRY_SETTINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "timeoutMs": {"type": "number", "description": "SDK or provider request timeout in milliseconds."},
        "maxRetries": {"type": "number", "description": "SDK or provider retry attempts."},
        "maxRetryDelayMs": {
            "type": "number",
            "description": "Maximum server-requested delay before failing.",
            "default": SETTINGS_DEFAULTS["retry"]["provider"]["maxRetryDelayMs"],
        },
    },
}

_RETRY_SETTINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "enabled": {"type": "boolean", "default": SETTINGS_DEFAULTS["retry"]["enabled"]},
        "maxRetries": {"type": "number", "default": SETTINGS_DEFAULTS["retry"]["maxRetries"]},
        "baseDelayMs": {
            "type": "number",
            "description": "Exponential backoff base delay in milliseconds: 2s, 4s, 8s.",
            "default": SETTINGS_DEFAULTS["retry"]["baseDelayMs"],
        },
        "maxAgentDelayMs": {"type": "number", "default": SETTINGS_DEFAULTS["retry"]["maxAgentDelayMs"]},
        "provider": _PROVIDER_RETRY_SETTINGS_SCHEMA,
        "maxDelayMs": {
            "type": "number",
            "description": "Legacy retry delay setting. Use provider.maxRetryDelayMs instead.",
            "deprecated": True,
        },
    },
}

_TERMINAL_SETTINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "showImages": {
            "type": "boolean",
            "description": "Show images when the terminal supports them.",
            "default": SETTINGS_DEFAULTS["terminal"]["showImages"],
        },
        "imageWidthCells": {
            "type": "number",
            "description": "Preferred inline image width in terminal cells.",
            "default": SETTINGS_DEFAULTS["terminal"]["imageWidthCells"],
        },
        "clearOnShrink": {
            "type": "boolean",
            "description": "Clear empty rows when content shrinks.",
            "default": SETTINGS_DEFAULTS["terminal"]["clearOnShrink"],
        },
        "showTerminalProgress": {
            "type": "boolean",
            "description": "Show OSC 9;4 terminal progress indicators.",
            "default": SETTINGS_DEFAULTS["terminal"]["showTerminalProgress"],
        },
        "hyperlinks": {"anyOf": [{"type": "boolean"}, {"type": "string", "const": "auto"}]},
        "images": _literals("kitty", "iterm2", "auto", False),
        "trueColor": {"anyOf": [{"type": "boolean"}, {"type": "string", "const": "auto"}]},
    },
}

_IMAGE_SETTINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "autoResize": {
            "type": "boolean",
            "description": "Resize images to 2000x2000 maximum for better model compatibility.",
            "default": SETTINGS_DEFAULTS["images"]["autoResize"],
        },
        "blockImages": {
            "type": "boolean",
            "description": "When true, prevents all images from being sent to LLM providers.",
            "default": SETTINGS_DEFAULTS["images"]["blockImages"],
        },
    },
}

_THINKING_BUDGETS_SETTINGS_SCHEMA = {
    "type": "object",
    "properties": {level: {"type": "number"} for level in ("minimal", "low", "medium", "high")},
    "description": "Custom token budgets for thinking levels.",
}

_MARKDOWN_SETTINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "codeBlockIndent": {"type": "string", "default": SETTINGS_DEFAULTS["markdown"]["codeBlockIndent"]},
    },
}

_WARNING_SETTINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "anthropicExtraUsage": {"type": "boolean", "default": SETTINGS_DEFAULTS["warnings"]["anthropicExtraUsage"]},
    },
}

_CODEMODE_SETTINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": _literals(
            "on",
            "only",
            description=(
                'How codemode presents tools. "on" keeps direct tools declared and annotates tools callable from '
                'scripts; codemode lists only tools without direct exposure. "only" lists every script-callable '
                "tool in codemode and does not declare active direct tools."
            ),
            default=SETTINGS_DEFAULTS["codemode"]["mode"],
        ),
        "inlineBudget": {
            "type": "number",
            "minimum": 0,
            "description": "Estimated tokens available for inline codemode tool declarations.",
            "default": SETTINGS_DEFAULTS["codemode"]["inlineBudget"],
        },
        "typeCheck": {
            "type": "boolean",
            "description": "Type-check scripts before they run.",
            "default": SETTINGS_DEFAULTS["codemode"]["typeCheck"],
        },
    },
}

_PACKAGE_SOURCE_SCHEMA = {
    "anyOf": [
        {"type": "string", "description": "Load all resources from the package."},
        {
            "type": "object",
            "required": ["source"],
            "properties": {
                "source": {"type": "string"},
                "autoload": {
                    "type": "boolean",
                    "description": "When false, start empty and only apply explicit resource patterns.",
                },
                "extensions": _STRING_ARRAY,
                "skills": _STRING_ARRAY,
                "prompts": _STRING_ARRAY,
                "themes": _STRING_ARRAY,
            },
        },
    ],
    "description": (
        "Package source for git or local packages. Use the string form to load all resources or the object form "
        "to filter resources."
    ),
}

_SKILLS_INPUT_SCHEMA = {
    "anyOf": [
        _STRING_ARRAY,
        {
            "type": "object",
            "properties": {"enableSkillCommands": {"type": "boolean"}, "customDirectories": _STRING_ARRAY},
            "deprecated": True,
        },
    ],
    "description": "Local skill file paths or directories.",
}

SETTINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "$schema": {"type": "string", "description": "JSON Schema reference."},
        "lastChangelogVersion": {"type": "string"},
        "defaultProvider": {"type": "string"},
        "defaultModel": {"type": "string"},
        "defaultThinkingLevel": MODEL_THINKING_LEVEL_SCHEMA,
        "modelThinkingLevels": _record(
            MODEL_THINKING_LEVEL_SCHEMA,
            description='Per-model default thinking level overrides keyed by "provider/modelId".',
        ),
        "transport": _literals("auto", "sse", "websocket", "websocket-cached", default=SETTINGS_DEFAULTS["transport"]),
        "steeringMode": _literals("all", "one-at-a-time", default=SETTINGS_DEFAULTS["steeringMode"]),
        "followUpMode": _literals("all", "one-at-a-time", default=SETTINGS_DEFAULTS["followUpMode"]),
        "theme": {"type": "string"},
        "compaction": _COMPACTION_SETTINGS_SCHEMA,
        "branchSummary": _BRANCH_SUMMARY_SETTINGS_SCHEMA,
        "retry": _RETRY_SETTINGS_SCHEMA,
        "hideThinkingBlock": {"type": "boolean", "default": SETTINGS_DEFAULTS["hideThinkingBlock"]},
        "showCacheMissNotices": {
            "type": "boolean",
            "description": "Show cache cost and provider recovery notices.",
            "default": SETTINGS_DEFAULTS["showCacheMissNotices"],
        },
        "externalEditor": {
            "type": "string",
            "description": "Command for Ctrl+G external editor; takes precedence over VISUAL and EDITOR.",
        },
        "shellPath": {"type": "string", "description": "Custom shell path, with support for leading ~ expansion."},
        "quietStartup": {
            "anyOf": [{"type": "boolean"}, {"type": "string", "const": "header"}],
            "description": 'When true, hide all startup output. When "header", keep only the startup header.',
            "default": SETTINGS_DEFAULTS["quietStartup"],
        },
        "defaultProjectTrust": _literals(
            "ask",
            "always",
            "never",
            description="Global setting only.",
            default=SETTINGS_DEFAULTS["defaultProjectTrust"],
        ),
        "shellCommandPrefix": {
            "type": "string",
            "description": "Prefix prepended to every bash command, for example to enable shell aliases.",
        },
        "collapseChangelog": {
            "type": "boolean",
            "description": "Show the condensed changelog after update; use /changelog for the full changelog.",
            "default": SETTINGS_DEFAULTS["collapseChangelog"],
        },
        "enableProviderAttribution": {
            "type": "boolean",
            "description": "Identify PiDrei to providers that credit the calling app (OpenRouter, NVIDIA, opencode).",
            "default": SETTINGS_DEFAULTS["enableProviderAttribution"],
        },
        "packages": {
            "type": "array",
            "items": _PACKAGE_SOURCE_SCHEMA,
            "description": "git or local package sources, as strings or objects with resource filtering.",
        },
        "extensions": {**_STRING_ARRAY, "description": "Local extension file paths or directories."},
        "deviceId": {
            "type": "string",
            "description": "Stable installation UUID, generated when authentication first needs it. Global only.",
        },
        "skills": _SKILLS_INPUT_SCHEMA,
        "prompts": {**_STRING_ARRAY, "description": "Local prompt template file paths or directories."},
        "themes": {**_STRING_ARRAY, "description": "Local theme file paths or directories."},
        "enableSkillCommands": {
            "type": "boolean",
            "description": "Register skills as /skill:name commands.",
            "default": SETTINGS_DEFAULTS["enableSkillCommands"],
        },
        "terminal": _TERMINAL_SETTINGS_SCHEMA,
        "images": _IMAGE_SETTINGS_SCHEMA,
        "enabledModels": {
            **_STRING_ARRAY,
            "description": "Model patterns for cycling, in the same format as the --models CLI flag.",
        },
        "defaultTools": {
            **_STRING_ARRAY,
            "description": (
                "Initial tool selection. Plain names replace the inherited selection; +name and -name entries add "
                "or remove tools."
            ),
        },
        "doubleEscapeAction": _literals(
            "fork",
            "tree",
            "none",
            description="Action for double-escape with an empty editor.",
            default=SETTINGS_DEFAULTS["doubleEscapeAction"],
        ),
        "treeFilterMode": _literals(
            "default",
            "no-tools",
            "user-only",
            "labeled-only",
            "all",
            description="Default filter when opening /tree.",
            default=SETTINGS_DEFAULTS["treeFilterMode"],
        ),
        "thinkingBudgets": _THINKING_BUDGETS_SETTINGS_SCHEMA,
        "editorPaddingX": {
            "type": "number",
            "description": "Horizontal padding for the input editor.",
            "default": SETTINGS_DEFAULTS["editorPaddingX"],
        },
        "outputPad": _literals(
            0,
            1,
            description="Horizontal padding for transcript content.",
            default=SETTINGS_DEFAULTS["outputPad"],
        ),
        "autocompleteMaxVisible": {
            "type": "number",
            "description": "Maximum visible items in the autocomplete dropdown.",
            "default": SETTINGS_DEFAULTS["autocompleteMaxVisible"],
        },
        "showHardwareCursor": {
            "type": "boolean",
            "description": "Show the terminal cursor while still positioning it for IME.",
        },
        "markdown": _MARKDOWN_SETTINGS_SCHEMA,
        "warnings": _WARNING_SETTINGS_SCHEMA,
        "codemode": _CODEMODE_SETTINGS_SCHEMA,
        "sessionDir": {
            "type": "string",
            "description": "Custom session storage directory, in the same format as the --session-dir CLI flag.",
        },
        "httpProxy": {
            "type": "string",
            "description": "Proxy URL applied as HTTP_PROXY and HTTPS_PROXY for PiDrei-managed HTTP clients.",
        },
        "httpIdleTimeoutMs": _timeout_setting(
            description='HTTP header or body idle timeout in milliseconds; 0 or "disabled" disables it.'
        ),
        "cacheWarming": _literals(
            "off",
            "streaming",
            "idle",
            description=(
                'Cache-warming profile. "idle" also warms between agent runs. Global only because each refresh '
                "costs money."
            ),
            default=SETTINGS_DEFAULTS["cacheWarming"],
        ),
        "websocketConnectTimeoutMs": _timeout_setting(
            description='WebSocket connect or open handshake timeout in milliseconds; 0 or "disabled" disables it.'
        ),
        "tuiMode": _literals("regular", "fullscreen", default=SETTINGS_DEFAULTS["tuiMode"]),
        "fullscreenExitOutput": _literals(
            "transcript",
            "resume-hint",
            description="No effect in regular TUI mode.",
            default=SETTINGS_DEFAULTS["fullscreenExitOutput"],
        ),
        "fullscreenScrollbar": _literals(
            "auto",
            "always",
            "hidden",
            description="No effect in regular TUI mode.",
            default=SETTINGS_DEFAULTS["fullscreenScrollbar"],
        ),
        "fullscreenCopyOnSelect": {
            "type": "boolean",
            "description": "No effect in regular TUI mode.",
            "default": SETTINGS_DEFAULTS["fullscreenCopyOnSelect"],
        },
        "fullscreenWheelScrollLines": {
            "anyOf": [{"type": "number"}, {"type": "string", "const": "auto"}],
            "description": "Lines scrolled per wheel event in fullscreen mode; numeric values are clamped from 1 to 100.",
            "default": SETTINGS_DEFAULTS["fullscreenWheelScrollLines"],
        },
        "queueMode": _literals(
            "all", "one-at-a-time", description="Legacy setting migrated to steeringMode.", deprecated=True
        ),
        "websockets": {"type": "boolean", "description": "Legacy setting migrated to transport.", "deprecated": True},
        "enableInstallTelemetry": {
            "type": "boolean",
            "description": "Legacy setting migrated to enableProviderAttribution.",
            "deprecated": True,
        },
    },
    "additionalProperties": True,
}
