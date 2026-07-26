"""Mirror of pi coding-agent test/frontmatter.test.ts."""

import pytest
import yaml

from pidrei.utils.frontmatter import parse_frontmatter, strip_frontmatter


class TestParseFrontmatter:
    def test_parses_keys_strips_quotes_and_returns_body(self):
        input = "---\nname: \"skill-name\"\ndescription: 'A desc'\nfoo-bar: value\n---\n\nBody text"
        frontmatter, body = parse_frontmatter(input)
        assert frontmatter["name"] == "skill-name"
        assert frontmatter["description"] == "A desc"
        assert frontmatter["foo-bar"] == "value"
        assert body == "Body text"

    def test_normalizes_newlines_and_handles_crlf(self):
        input = "---\r\nname: test\r\n---\r\nLine one\r\nLine two"
        _frontmatter, body = parse_frontmatter(input)
        assert body == "Line one\nLine two"

    def test_throws_on_invalid_yaml_frontmatter(self):
        # pi asserts the js-yaml message (/at line 1, column 10/); PyYAML's
        # error text differs, so only the raise is mirrored.
        input = "---\nfoo: [bar\n---\nBody"
        with pytest.raises(yaml.YAMLError):
            parse_frontmatter(input)

    def test_parses_pipe_multiline_yaml_syntax(self):
        input = "---\ndescription: |\n  Line one\n  Line two\n---\n\nBody"
        frontmatter, body = parse_frontmatter(input)
        assert frontmatter["description"] == "Line one\nLine two\n"
        assert body == "Body"

    def test_returns_original_content_when_frontmatter_is_missing_or_unterminated(self):
        no_frontmatter = "Just text\nsecond line"
        missing_end = "---\nname: test\nBody without terminator"
        assert parse_frontmatter(no_frontmatter).body == "Just text\nsecond line"
        assert parse_frontmatter(missing_end).body == "---\nname: test\nBody without terminator"

    def test_returns_empty_object_for_empty_or_comment_only_frontmatter(self):
        input = "---\n# just a comment\n---\nBody"
        frontmatter, _body = parse_frontmatter(input)
        assert frontmatter == {}


class TestStripFrontmatter:
    def test_removes_frontmatter_and_trims_body(self):
        input = "---\nkey: value\n---\n\nBody\n"
        assert strip_frontmatter(input) == "Body"

    def test_returns_body_when_no_frontmatter_present(self):
        input = "\n  No frontmatter body  \n"
        assert strip_frontmatter(input) == "\n  No frontmatter body  \n"
