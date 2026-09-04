"""Mirror of pi coding-agent src/modes/interactive/chat-viewport.ts."""

from dataclasses import dataclass

from pidrei_tui import ScrollView, VStack


@dataclass(frozen=True, slots=True)
class ChatViewport:
    root: VStack
    transcript: ScrollView


def create_chat_viewport(
    *,
    document,
    pending_messages,
    status,
    editor,
    footer,
    widgets_above=None,
    widgets_below=None,
    scrollbar: str | None = None,
    scrollbar_style=None,
) -> ChatViewport:
    """Shared fullscreen transcript and fixed input-dock layout."""
    scroll_options = {
        "follow": "end",
        "primary": True,
        "overscroll": "chain",
        "scrollbar": scrollbar if scrollbar is not None else "auto",
    }
    if scrollbar_style is not None:
        scroll_options["scrollbarStyle"] = scrollbar_style
    transcript = ScrollView(document, scroll_options)
    dock = VStack(
        [
            {"component": pending_messages, "shrink": 1, "minSize": 0},
            {"component": status, "shrink": 1, "minSize": 0},
            *([] if widgets_above is None else [{"component": widgets_above, "shrink": 1, "minSize": 0}]),
            {"component": editor, "shrink": 1, "minSize": 3},
            *([] if widgets_below is None else [{"component": widgets_below, "shrink": 1, "minSize": 0}]),
            {"component": footer, "shrink": 1, "minSize": 1},
        ]
    )
    return ChatViewport(
        transcript=transcript,
        root=VStack(
            [
                {"component": transcript, "basis": 0, "grow": 1, "shrink": 1, "minSize": 1},
                {"component": dock, "basis": "auto", "grow": 0, "shrink": 1, "minSize": 1},
            ]
        ),
    )
