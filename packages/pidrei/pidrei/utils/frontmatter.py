"""Mirror of pi coding-agent src/utils/frontmatter.ts."""

from typing import Any, NamedTuple

import yaml


class ParsedFrontmatter(NamedTuple):
    frontmatter: dict[str, Any]
    body: str


def _normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _extract_frontmatter(content: str) -> tuple[str | None, str]:
    normalized = _normalize_newlines(content)

    if not normalized.startswith("---"):
        return None, normalized

    end_index = normalized.find("\n---", 3)
    if end_index == -1:
        return None, normalized

    return normalized[4:end_index], normalized[end_index + 4 :].strip()


def parse_frontmatter(content: str) -> ParsedFrontmatter:
    yaml_string, body = _extract_frontmatter(content)
    if not yaml_string:
        return ParsedFrontmatter({}, body)
    # The slice between the `---` markers drops the newline before the closing
    # marker; restore it so PyYAML's clip chomping of `|` block scalars keeps
    # the final newline exactly like js-yaml does in pi.
    parsed = yaml.safe_load(yaml_string + "\n")
    return ParsedFrontmatter(parsed if parsed is not None else {}, body)


def strip_frontmatter(content: str) -> str:
    return parse_frontmatter(content).body
