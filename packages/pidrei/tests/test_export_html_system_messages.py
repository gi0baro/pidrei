"""Partial mirror of pi coding-agent test/export-html-system-messages.test.ts.

https://github.com/earendil-works/pi/issues/11261: system messages were stored in the export but never
rendered. Not mirrored: the cases that run the template's system-message helpers in node:vm (against
pi-ai's `getSystemMessageText`/`renderSystemMessageUpdate`, the leading-entry rule, the update
summary); the tests have no JS runtime.
"""

import os
import re

from pidrei.config import get_export_template_dir


with open(os.path.join(get_export_template_dir(), "template.js"), encoding="utf-8") as _f:
    TEMPLATE_JS = _f.read()


def test_renders_system_entries_in_the_transcript():
    assert re.search(r"if \(msg\.role === 'system'\) \{\s*let html = '';", TEMPLATE_JS)
    assert "renderSystemPromptBlock('System Prompt', text)" in TEMPLATE_JS
    assert "renderToolsList('Available Tools', msg.toolsAdded)" in TEMPLATE_JS
