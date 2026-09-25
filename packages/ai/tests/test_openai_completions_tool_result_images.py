"""Partial mirror of pi's openai-completions-tool-result-images.test.ts.

Holds the 0.87.1 empty-text-part case; the tool-result image batching cases
are a PARITY GAP. pi passes a hand-built full compat object; `detect_compat`
of the same model yields the equivalent here.
"""

import dataclasses
import time

from pidrei_ai.api.openai_completions import convert_messages, detect_compat
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import Context, ImageContent, TextContent, UserMessage
from pidrei_ai.utils.transcript import normalize_context


# Regression test for https://github.com/earendil-works/pi/issues/9797
def test_omits_empty_text_parts_from_user_messages_with_images():
    model = dataclasses.replace(
        get_builtin_model("openai", "gpt-4o-mini"),
        api="openai-completions",
        compat=None,
        input=["text", "image"],
    )
    context = normalize_context(
        Context(
            messages=[
                UserMessage(
                    content=[TextContent(text=""), ImageContent(data="ZmFrZQ==", mime_type="image/png")],
                    timestamp=int(time.time() * 1000),
                )
            ]
        )
    )

    assert convert_messages(model, context, detect_compat(model)) == [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZQ=="}}]}
    ]
