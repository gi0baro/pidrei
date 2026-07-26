"""Mirror of pi coding-agent test/truncate-to-width.test.ts.

Tests for truncate_to_width behavior with Unicode characters that have
different byte vs display widths.
"""

from pidrei_tui import truncate_to_width, visible_width


class TestTruncateToWidth:
    def test_should_truncate_messages_with_unicode_characters_correctly(self):
        # This message contains a checkmark (✔) which may have display width > 1 byte
        message = '✔ script to run › dev $ concurrently "vite" "node --import tsx ./'
        width = 67
        max_msg_width = width - 2  # Account for cursor

        truncated = truncate_to_width(message, max_msg_width)
        truncated_width = visible_width(truncated)

        assert truncated_width <= max_msg_width

    def test_should_handle_emoji_characters(self):
        message = "🎉 Celebration! 🚀 Launch 📦 Package ready for deployment now"
        width = 40
        max_msg_width = width - 2

        truncated = truncate_to_width(message, max_msg_width)
        truncated_width = visible_width(truncated)

        assert truncated_width <= max_msg_width

    def test_should_handle_mixed_ascii_and_wide_characters(self):
        message = "Hello 世界 Test 你好 More text here that is long"
        width = 30
        max_msg_width = width - 2

        truncated = truncate_to_width(message, max_msg_width)
        truncated_width = visible_width(truncated)

        assert truncated_width <= max_msg_width

    def test_should_not_truncate_messages_that_fit(self):
        message = "Short message"
        width = 50
        max_msg_width = width - 2

        truncated = truncate_to_width(message, max_msg_width)

        assert truncated == message
        assert visible_width(truncated) <= max_msg_width

    def test_should_add_ellipsis_when_truncating(self):
        message = "This is a very long message that needs to be truncated"
        width = 30
        max_msg_width = width - 2

        truncated = truncate_to_width(message, max_msg_width)

        assert "..." in truncated
        assert visible_width(truncated) <= max_msg_width

    def test_should_handle_the_exact_crash_case_from_issue_report(self):
        # Terminal width was 67, line had visible width 68
        # The problematic text contained "✔" and "›" characters
        message = '✔ script to run › dev $ concurrently "vite" "node --import tsx ./server.ts"'
        terminal_width = 67
        cursor_width = 2  # "› " or "  "
        max_msg_width = terminal_width - cursor_width

        truncated = truncate_to_width(message, max_msg_width)
        final_width = visible_width(truncated)

        # The final line (cursor + message) must not exceed terminal width
        assert final_width + cursor_width <= terminal_width
