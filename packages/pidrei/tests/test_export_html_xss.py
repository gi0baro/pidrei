"""Mirror of pi coding-agent test/export-html-xss.test.ts."""

import os
import re

from pidrei.config import get_export_template_dir


with open(os.path.join(get_export_template_dir(), "template.js"), encoding="utf-8") as _f:
    TEMPLATE_JS = _f.read()


class TestExportHtmlMarkdownLinkSanitization:
    def test_overrides_the_marked_link_renderer_to_use_scheme_allow_list_sanitization(self):
        assert re.search(r"link\s*\(\s*token\s*\)", TEMPLATE_JS)
        assert re.search(r"sanitizeMarkdownUrl\(token\.href\)", TEMPLATE_JS)
        assert re.search(r"\^\(https\?\|mailto\|tel\|ftp\)", TEMPLATE_JS)

    def test_overrides_the_marked_image_renderer_to_use_scheme_allow_list_sanitization(self):
        assert re.search(r"image\s*\(\s*token\s*\)", TEMPLATE_JS)
        assert re.search(r"sanitizeMarkdownUrl\(token\.href\)", TEMPLATE_JS)

    def test_strips_c0_controls_before_checking_and_emitting_markdown_urls(self):
        assert "replace(/[\\x00-\\x1f\\x7f]/g, '')" in TEMPLATE_JS
        assert not re.search(r"\^\\s\*\(javascript\|vbscript\|data\):", TEMPLATE_JS, re.IGNORECASE)

    def test_escapes_href_attributes_in_the_custom_link_renderer(self):
        # The link renderer must escape href values to prevent attribute
        # breakout
        assert re.search(r"escapeHtml\(href\)", TEMPLATE_JS)

    def test_escapes_image_mime_type_attributes(self):
        # Image mimeType must be escaped to prevent attribute breakout
        assert not re.search(r"\$\{img\.mimeType\}", TEMPLATE_JS)
        assert re.search(r"escapeHtml\(img\.mimeType", TEMPLATE_JS)

    def test_escapes_image_data_attributes(self):
        # Image data is embedded in src attributes and must not allow
        # attribute breakout.
        assert not re.search(r';base64,\$\{img\.data\}"', TEMPLATE_JS)
        assert re.search(r';base64,\$\{escapeHtml\(img\.data \|\| (?:\'\'|"")\)\}"', TEMPLATE_JS)

    def test_escapes_entry_ids_before_inserting_them_into_attributes(self):
        # Session entry IDs are embedded in id and data-entry-id attributes.
        assert not re.search(r'id="\$\{entryId\}"', TEMPLATE_JS)
        assert not re.search(r'data-entry-id="\$\{entryId\}"', TEMPLATE_JS)
        assert re.search(r"entry-\$\{escapeHtml\(entry\.id\)\}", TEMPLATE_JS)
        assert re.search(r'data-entry-id="\$\{escapeHtml\(entryId\)\}"', TEMPLATE_JS)

    def test_escapes_tree_metadata_rendered_from_session_fields(self):
        # The tree renders session metadata via innerHTML, so dynamic fields
        # must be escaped.
        assert not re.search(r"\[\$\{msg\.toolName \|\| 'tool'\}\]", TEMPLATE_JS)
        assert not re.search(r"\[\$\{msg\.role\}\]", TEMPLATE_JS)
        assert not re.search(r"\[model: \$\{entry\.modelId\}\]", TEMPLATE_JS)
        assert not re.search(r"\[thinking: \$\{entry\.thinkingLevel\}\]", TEMPLATE_JS)
        assert not re.search(r"\[\$\{entry\.type\}\]", TEMPLATE_JS)
        assert re.search(r"\$\{escapeHtml\(msg\.toolName \|\| 'tool'\)\}", TEMPLATE_JS)
        assert re.search(r"\$\{escapeHtml\(msg\.role\)\}", TEMPLATE_JS)
        assert re.search(r"\$\{escapeHtml\(entry\.modelId\)\}", TEMPLATE_JS)
        assert re.search(r"\$\{escapeHtml\(entry\.thinkingLevel\)\}", TEMPLATE_JS)
        assert re.search(r"\$\{escapeHtml\(entry\.type\)\}", TEMPLATE_JS)

    def test_escapes_model_names_in_the_exported_header(self):
        # Assistant message provider/model values are collected from the
        # session and rendered with innerHTML.
        assert not re.search(r"\$\{globalStats\.models\.join\(', '\) \|\| 'unknown'\}", TEMPLATE_JS)
        assert re.search(r"\$\{escapeHtml\(globalStats\.models\.join\(', '\) \|\| 'unknown'\)\}", TEMPLATE_JS)
