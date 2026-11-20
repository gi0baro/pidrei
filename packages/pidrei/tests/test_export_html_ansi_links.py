"""Mirror of pi coding-agent test/export-html-ansi-links.test.ts."""

from pidrei.core.export_html.ansi_to_html import ansi_to_html


def _link(url: str, text: str, end: str = "\x1b\\") -> str:
    return f"\x1b]8;;{url}{end}{text}\x1b]8;;{end}"


class TestExportHtmlAnsiHyperlinks:
    def test_converts_osc_8_hyperlinks_to_anchors(self):
        assert ansi_to_html(f"see {_link('https://example.com/a?b=1&c=2', '\x1b[4mdocs\x1b[24m')}.") == (
            'see <a href="https://example.com/a?b=1&amp;c=2"><span style="text-decoration:underline">docs</span></a>.'
        )
        assert ansi_to_html(_link("mailto:me@example.com", "mail", "\x07")) == (
            '<a href="mailto:me@example.com">mail</a>'
        )

    def test_keeps_styled_text_nested_inside_the_anchor_and_closes_an_unterminated_link(self):
        assert ansi_to_html("\x1b[31mred \x1b]8;;https://example.com\x1b\\link") == (
            '<span style="color:#800000">red </span><a href="https://example.com">'
            '<span style="color:#800000">link</span></a>'
        )

    def test_drops_links_with_unsafe_schemes_and_other_escape_sequences(self):
        assert ansi_to_html(_link("javascript:alert(1)", "x")) == "x"
        assert ansi_to_html(_link("java\x01script:alert(1)", "x")) == "x"
        assert ansi_to_html("\x1b]0;title\x07a\x1b[2Kb\x1b]133;A\x1b\\c\x1b_pi:c\x07d\x1be") == "abcde"
