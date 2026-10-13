"""Mirror of pi coding-agent src/modes/interactive/theme/theme-tokens.ts.

Theme token metadata: each token's color slot, its description, and the
token it falls back to when a theme omits it. The one source for the required
color set, the fallbacks and the foreground/background split, and for the
published theme schema.
"""

from types import MappingProxyType


def _define_theme_tokens(tokens: dict[str, dict[str, str]]) -> MappingProxyType:
    for name, descriptor in tokens.items():
        fallback_name = descriptor.get("fallback")
        if fallback_name is None:
            continue
        fallback = tokens.get(fallback_name)
        if fallback is None:
            raise ValueError(f"Theme token {name} has unknown fallback {fallback_name}")
        if fallback["slot"] != descriptor["slot"]:
            raise ValueError(f"Theme token {name} has fallback {fallback_name} in a different color slot")
    return MappingProxyType({name: MappingProxyType(descriptor) for name, descriptor in tokens.items()})


# Theme token metadata in the same order used by theme files and the published JSON Schema.
THEME_TOKENS = _define_theme_tokens(
    {
        "accent": {"slot": "foreground", "description": "Primary accent color (logo, selected items, cursor)"},
        "border": {"slot": "foreground", "description": "Normal borders"},
        "borderAccent": {"slot": "foreground", "description": "Highlighted borders"},
        "borderMuted": {"slot": "foreground", "description": "Subtle borders"},
        "success": {"slot": "foreground", "description": "Success states"},
        "error": {"slot": "foreground", "description": "Error states"},
        "warning": {"slot": "foreground", "description": "Warning states"},
        "muted": {"slot": "foreground", "description": "Secondary/dimmed text"},
        "dim": {"slot": "foreground", "description": "Very dimmed text (more subtle than muted)"},
        "text": {"slot": "foreground", "description": "Default text color (usually empty string)"},
        "thinkingText": {"slot": "foreground", "description": "Thinking block text color"},
        "scrollbarTrack": {
            "slot": "foreground",
            "description": "Fullscreen scrollbar track foreground (falls back to muted when omitted)",
            "fallback": "muted",
        },
        "scrollbarThumb": {
            "slot": "foreground",
            "description": "Fullscreen scrollbar thumb foreground (falls back to text when omitted)",
            "fallback": "text",
        },
        "selectedBg": {"slot": "background", "description": "Selected item background"},
        "searchMatchBg": {
            "slot": "background",
            "description": "Transcript search match background and current-match text (falls back to selectedBg when omitted)",
            "fallback": "selectedBg",
        },
        "searchMatchText": {
            "slot": "foreground",
            "description": "Transcript search match text and current-match background (falls back to text when omitted)",
            "fallback": "text",
        },
        "userMessageBg": {"slot": "background", "description": "User message background"},
        "userMessageText": {"slot": "foreground", "description": "User message text color"},
        "customMessageBg": {"slot": "background", "description": "Custom message background (hook-injected messages)"},
        "customMessageText": {"slot": "foreground", "description": "Custom message text color"},
        "customMessageLabel": {"slot": "foreground", "description": "Custom message type label color"},
        "toolPendingBg": {"slot": "background", "description": "Tool execution box (pending state)"},
        "toolSuccessBg": {"slot": "background", "description": "Tool execution box (success state)"},
        "toolErrorBg": {"slot": "background", "description": "Tool execution box (error state)"},
        "toolTitle": {"slot": "foreground", "description": "Tool execution box title color"},
        "toolOutput": {"slot": "foreground", "description": "Tool execution box output text color"},
        "mdHeading": {"slot": "foreground", "description": "Markdown heading text"},
        "mdLink": {"slot": "foreground", "description": "Markdown link text"},
        "mdLinkUrl": {"slot": "foreground", "description": "Markdown link URL"},
        "mdCode": {"slot": "foreground", "description": "Markdown inline code"},
        "mdCodeBlock": {"slot": "foreground", "description": "Markdown code block content"},
        "mdCodeBlockBorder": {"slot": "foreground", "description": "Markdown code block fences"},
        "mdQuote": {"slot": "foreground", "description": "Markdown blockquote text"},
        "mdQuoteBorder": {"slot": "foreground", "description": "Markdown blockquote border"},
        "mdHr": {"slot": "foreground", "description": "Markdown horizontal rule"},
        "mdListBullet": {"slot": "foreground", "description": "Markdown list bullets/numbers"},
        "toolDiffAdded": {"slot": "foreground", "description": "Added lines in tool diffs"},
        "toolDiffRemoved": {"slot": "foreground", "description": "Removed lines in tool diffs"},
        "toolDiffContext": {"slot": "foreground", "description": "Context lines in tool diffs"},
        "syntaxComment": {"slot": "foreground", "description": "Syntax highlighting: comments"},
        "syntaxKeyword": {"slot": "foreground", "description": "Syntax highlighting: keywords"},
        "syntaxFunction": {"slot": "foreground", "description": "Syntax highlighting: function names"},
        "syntaxVariable": {"slot": "foreground", "description": "Syntax highlighting: variable names"},
        "syntaxString": {"slot": "foreground", "description": "Syntax highlighting: string literals"},
        "syntaxNumber": {"slot": "foreground", "description": "Syntax highlighting: number literals"},
        "syntaxType": {"slot": "foreground", "description": "Syntax highlighting: type names"},
        "syntaxOperator": {"slot": "foreground", "description": "Syntax highlighting: operators"},
        "syntaxPunctuation": {"slot": "foreground", "description": "Syntax highlighting: punctuation"},
        "thinkingOff": {"slot": "foreground", "description": "Thinking level border: off"},
        "thinkingMinimal": {"slot": "foreground", "description": "Thinking level border: minimal"},
        "thinkingLow": {"slot": "foreground", "description": "Thinking level border: low"},
        "thinkingMedium": {"slot": "foreground", "description": "Thinking level border: medium"},
        "thinkingHigh": {"slot": "foreground", "description": "Thinking level border: high"},
        "thinkingXhigh": {"slot": "foreground", "description": "Thinking level border: xhigh"},
        "thinkingMax": {
            "slot": "foreground",
            "description": "Thinking level border: max (falls back to thinkingXhigh when omitted)",
            "fallback": "thinkingXhigh",
        },
        "bashMode": {"slot": "foreground", "description": "Editor border color in bash mode"},
    }
)
