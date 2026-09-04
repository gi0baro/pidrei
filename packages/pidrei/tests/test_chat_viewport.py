"""Mirror of pi coding-agent test/chat-viewport.test.ts."""

from pidrei.modes.interactive.chat_viewport import create_chat_viewport
from pidrei_tui import Container


def test_defaults_the_transcript_scrollbar_to_auto_and_accepts_overrides():
    automatic = create_chat_viewport(
        document=Container(),
        pending_messages=Container(),
        status=Container(),
        editor=Container(),
        footer=Container(),
    )
    hidden = create_chat_viewport(
        document=Container(),
        pending_messages=Container(),
        status=Container(),
        editor=Container(),
        footer=Container(),
        scrollbar="hidden",
    )

    assert automatic.transcript.scrollbar == "auto"
    assert hidden.transcript.scrollbar == "hidden"
