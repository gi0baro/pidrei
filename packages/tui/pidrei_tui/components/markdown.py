"""Markdown component - renders markdown as styled terminal output.

Port of pi tui ``components/markdown.ts``. The marked token stream is
reproduced by :mod:`._marked` (markdown-it-py based); this module is the 1:1
renderer over it.

``theme`` is a record of style functions mirroring pi's ``MarkdownTheme``
(``heading``, ``link``, ``linkUrl``, ``code``, ``codeBlock``,
``codeBlockBorder``, ``quote``, ``quoteBorder``, ``hr``, ``listBullet``,
``bold``, ``italic``, ``strikethrough``, ``underline``, optional
``highlightCode`` and ``codeBlockIndent``); ``default_text_style`` mirrors
``DefaultTextStyle`` (``{"color", "bgColor", "bold", "italic",
"strikethrough", "underline"}``); ``options`` mirrors ``MarkdownOptions``
(``{"preserveOrderedListMarkers", "preserveBackslashEscapes", "transform"}``).

``transform`` is ``(markdown, available_width) -> str``, applied to the source
text before parsing. It is deliberately sync (pi's is too): it runs on every
render, including width changes and every streaming update.
"""

import re

from ..latex import render_latex
from ..terminal_image import get_capabilities, hyperlink, is_image_line
from ..utils import apply_background_to_line, visible_width, wrap_text_with_ansi
from ._marked import lex, trim_partial_closing_fences


__all__ = ["Markdown"]

_ORDERED_MARKER_RE = re.compile(r"^(?: {0,3})(\d{1,9}[.)])[ \t]+")
_UNORDERED_MARKER_RE = re.compile(r"^(?: {0,3})([-+*])(?:[ \t]+|(?=\r?\n|$))")
_FULL_RESET_RE = re.compile(r"\x1b\[0m")
_SENTINEL = "\u0000"
_MAX_UNBROKEN_WORD_WIDTH = 30


class Markdown:
    def __init__(
        self,
        text: str,
        padding_x: int,
        padding_y: int,
        theme: dict,
        default_text_style: dict | None = None,
        options: dict | None = None,
    ) -> None:
        self._text = text
        self._padding_x = padding_x  # Left/right padding
        self._padding_y = padding_y  # Top/bottom padding
        self._theme = theme
        self._default_text_style = default_text_style
        self._options = dict(options) if options is not None else {}
        self._default_style_prefix: str | None = None

        # Cache for rendered output
        self._cached_text: str | None = None
        self._cached_width: int | None = None
        self._cached_lines: list[str] | None = None

    def set_text(self, text: str) -> None:
        self._text = text
        self.invalidate()

    def invalidate(self) -> None:
        self._cached_text = None
        self._cached_width = None
        self._cached_lines = None

    def render(self, width: int) -> list[str]:
        # Check cache
        if self._cached_lines is not None and self._cached_text == self._text and self._cached_width == width:
            return self._cached_lines

        # Calculate available width for content (subtract horizontal padding)
        content_width = max(1, width - self._padding_x * 2)
        transform = self._options.get("transform")
        text = self._text if transform is None else transform(self._text, content_width)

        # Don't render anything if there's no actual text
        if not text or text.strip() == "":
            result: list[str] = []
            self._cached_text = self._text
            self._cached_width = width
            self._cached_lines = result
            return result

        # Replace tabs with 3 spaces for consistent rendering
        normalized_text = text.replace("\t", "   ")

        # Parse markdown to marked-shaped tokens
        tokens = lex(normalized_text)
        trim_partial_closing_fences(tokens)

        # Convert tokens to styled terminal output
        rendered_lines: list[str] = []

        for i, token in enumerate(tokens):
            next_token = tokens[i + 1] if i + 1 < len(tokens) else None
            token_lines = self._render_token(token, content_width, next_token["type"] if next_token else None)
            rendered_lines.extend(token_lines)

        # Wrap lines (NO padding, NO background yet)
        wrapped_lines: list[str] = []
        for line in rendered_lines:
            if is_image_line(line):
                wrapped_lines.append(line)
            else:
                wrapped_lines.extend(wrap_text_with_ansi(line, content_width))

        # Add margins and background to each wrapped line
        left_margin = " " * self._padding_x
        right_margin = " " * self._padding_x
        bg_fn = self._default_text_style.get("bgColor") if self._default_text_style else None
        content_lines: list[str] = []

        for line in wrapped_lines:
            if is_image_line(line):
                content_lines.append(line)
                continue

            line_with_margins = left_margin + line + right_margin

            if bg_fn is not None:
                content_lines.append(apply_background_to_line(line_with_margins, width, bg_fn))
            else:
                # No background - just pad to width
                visible_len = visible_width(line_with_margins)
                padding_needed = max(0, width - visible_len)
                content_lines.append(line_with_margins + " " * padding_needed)

        # Add top/bottom padding (empty lines)
        empty_line = " " * width
        empty_lines: list[str] = []
        for _ in range(self._padding_y):
            line = apply_background_to_line(empty_line, width, bg_fn) if bg_fn is not None else empty_line
            empty_lines.append(line)

        # Combine top padding, content, and bottom padding
        result = empty_lines + content_lines + empty_lines

        # Update cache
        self._cached_text = self._text
        self._cached_width = width
        self._cached_lines = result

        return result if result else [""]

    def _apply_default_style(self, text: str) -> str:
        """Apply default text style to a string.

        This is the base styling applied to all text content.
        NOTE: Background color is NOT applied here - it's applied at the padding
        stage to ensure it extends to the full line width.
        """
        style = self._default_text_style
        if not style:
            return text

        styled = text

        # Apply foreground color (NOT background - that's applied at padding stage)
        if style.get("color") is not None:
            styled = style["color"](styled)

        # Apply text decorations using the theme
        if style.get("bold"):
            styled = self._theme["bold"](styled)
        if style.get("italic"):
            styled = self._theme["italic"](styled)
        if style.get("strikethrough"):
            styled = self._theme["strikethrough"](styled)
        if style.get("underline"):
            styled = self._theme["underline"](styled)

        return styled

    def _get_default_style_prefix(self) -> str:
        style = self._default_text_style
        if not style:
            return ""

        if self._default_style_prefix is not None:
            return self._default_style_prefix

        styled = _SENTINEL

        if style.get("color") is not None:
            styled = style["color"](styled)

        if style.get("bold"):
            styled = self._theme["bold"](styled)
        if style.get("italic"):
            styled = self._theme["italic"](styled)
        if style.get("strikethrough"):
            styled = self._theme["strikethrough"](styled)
        if style.get("underline"):
            styled = self._theme["underline"](styled)

        sentinel_index = styled.find(_SENTINEL)
        self._default_style_prefix = styled[:sentinel_index] if sentinel_index >= 0 else ""
        return self._default_style_prefix

    def _get_style_prefix(self, style_fn) -> str:
        styled = style_fn(_SENTINEL)
        sentinel_index = styled.find(_SENTINEL)
        return styled[:sentinel_index] if sentinel_index >= 0 else ""

    def _get_default_inline_style_context(self) -> dict:
        return {
            "applyText": self._apply_default_style,
            "stylePrefix": self._get_default_style_prefix(),
        }

    def _render_token(
        self,
        token: dict,
        width: int,
        next_token_type: str | None = None,
        style_context: dict | None = None,
    ) -> list[str]:
        lines: list[str] = []
        token_type = token["type"]

        if token_type == "heading":
            heading_level = token["depth"]
            heading_prefix = "#" * heading_level + " "

            # Build a heading-specific style context so inline tokens (codespan,
            # bold, etc.) restore heading styling after their own ANSI resets
            # instead of falling back to the default text style.
            if heading_level == 1:

                def heading_style_fn(text: str) -> str:
                    return self._theme["heading"](self._theme["bold"](self._theme["underline"](text)))
            else:

                def heading_style_fn(text: str) -> str:
                    return self._theme["heading"](self._theme["bold"](text))

            heading_style_context = {
                "applyText": heading_style_fn,
                "stylePrefix": self._get_style_prefix(heading_style_fn),
            }

            heading_text = self._render_inline_tokens(token.get("tokens") or [], heading_style_context)
            styled_heading = heading_style_fn(heading_prefix) + heading_text if heading_level >= 3 else heading_text
            lines.append(styled_heading)
            if next_token_type and next_token_type != "space":
                lines.append("")  # Add spacing after headings (unless space token follows)

        elif token_type == "paragraph":
            paragraph_text = self._render_inline_tokens(token.get("tokens") or [], style_context)
            lines.append(paragraph_text)
            # Don't add spacing if next token is space or list
            if next_token_type and next_token_type not in ("list", "space"):
                lines.append("")

        elif token_type == "text":
            lines.append(self._render_inline_tokens([token], style_context))

        elif token_type == "latexBlock":
            if not token.get("pending") and self._options.get("renderLatex") is not False:
                rendered = render_latex(token["text"], {"display": True}) or token["raw"].strip()
            else:
                rendered = token["raw"].strip()
            for line in rendered.split("\n"):
                lines.append(self._apply_default_style(line))
            if next_token_type and next_token_type != "space":
                lines.append("")

        elif token_type == "code":
            indent = self._theme.get("codeBlockIndent", "  ")
            lines.append(self._theme["codeBlockBorder"](f"```{token.get('lang') or ''}"))
            highlight_code = self._theme.get("highlightCode")
            if highlight_code is not None:
                highlighted_lines = highlight_code(token["text"], token.get("lang"))
                for hl_line in highlighted_lines:
                    lines.append(f"{indent}{hl_line}")
            else:
                # Split code by newlines and style each line
                for code_line in token["text"].split("\n"):
                    lines.append(f"{indent}{self._theme['codeBlock'](code_line)}")
            lines.append(self._theme["codeBlockBorder"]("```"))
            if next_token_type and next_token_type != "space":
                lines.append("")  # Add spacing after code blocks (unless space token follows)

        elif token_type == "list":
            lines.extend(self._render_list(token, 0, width, style_context))
            # Don't add spacing after lists if a space token follows
            # (the space token will handle it)

        elif token_type == "table":
            lines.extend(self._render_table(token, width, next_token_type, style_context))

        elif token_type == "blockquote":

            def quote_style(text: str) -> str:
                return self._theme["quote"](self._theme["italic"](text))

            quote_style_prefix = self._get_style_prefix(quote_style)

            def apply_quote_style(line: str) -> str:
                if not quote_style_prefix:
                    return quote_style(line)
                line_with_reapplied_style = _FULL_RESET_RE.sub(f"\x1b[0m{quote_style_prefix}", line)
                return quote_style(line_with_reapplied_style)

            # Calculate available width for quote content (subtract border "│ " = 2 chars)
            quote_content_width = max(1, width - 2)

            # Blockquotes contain block-level tokens (paragraph, list, code, etc.), so
            # render children with _render_token() instead of _render_inline_tokens().
            # Default message style should not apply inside blockquotes.
            quote_inline_style_context = {
                "applyText": lambda text: text,
                "stylePrefix": quote_style_prefix,
            }
            quote_tokens = token.get("tokens") or []
            rendered_quote_lines: list[str] = []
            for i, quote_token in enumerate(quote_tokens):
                next_quote_token = quote_tokens[i + 1] if i + 1 < len(quote_tokens) else None
                rendered_quote_lines.extend(
                    self._render_token(
                        quote_token,
                        quote_content_width,
                        next_quote_token["type"] if next_quote_token else None,
                        quote_inline_style_context,
                    )
                )

            # Avoid rendering an extra empty quote line before the outer blockquote spacing.
            while rendered_quote_lines and rendered_quote_lines[-1] == "":
                rendered_quote_lines.pop()

            for quote_line in rendered_quote_lines:
                styled_line = apply_quote_style(quote_line)
                for wrapped_line in wrap_text_with_ansi(styled_line, quote_content_width):
                    lines.append(self._theme["quoteBorder"]("│ ") + wrapped_line)
            if next_token_type and next_token_type != "space":
                lines.append("")  # Add spacing after blockquotes (unless space token follows)

        elif token_type == "hr":
            lines.append(self._theme["hr"]("─" * min(width, 80)))
            if next_token_type and next_token_type != "space":
                lines.append("")  # Add spacing after horizontal rules (unless space token follows)

        elif token_type == "html":
            # Render HTML as plain text (escaped for terminal)
            raw = token.get("raw")
            if isinstance(raw, str):
                lines.append(self._apply_default_style(raw.strip()))

        elif token_type == "space":
            # Space tokens represent blank lines in markdown
            lines.append("")

        else:
            # Handle any other token types as plain text
            text = token.get("text")
            if isinstance(text, str):
                lines.append(text)

        return lines

    def _render_inline_tokens(self, tokens: list[dict], style_context: dict | None = None) -> str:
        result = ""
        resolved_style_context = (
            style_context if style_context is not None else self._get_default_inline_style_context()
        )
        apply_text = resolved_style_context["applyText"]
        style_prefix = resolved_style_context["stylePrefix"]

        def apply_text_with_newlines(text: str) -> str:
            return "\n".join(apply_text(segment) for segment in text.split("\n"))

        for token in tokens:
            token_type = token["type"]

            if token_type == "latex":
                if not token.get("pending") and self._options.get("renderLatex") is not False:
                    rendered = render_latex(token["text"]) or token["raw"]
                else:
                    rendered = token["raw"]
                result += apply_text_with_newlines(rendered)

            elif token_type == "escape":
                result += apply_text_with_newlines(
                    token["raw"] if self._options.get("preserveBackslashEscapes") else token["text"]
                )

            elif token_type == "text":
                # Text tokens in list items can have nested tokens for inline formatting
                if token.get("tokens"):
                    result += self._render_inline_tokens(token["tokens"], resolved_style_context)
                else:
                    result += apply_text_with_newlines(token.get("text") or "")

            elif token_type == "paragraph":
                # Paragraph tokens contain nested inline tokens
                result += self._render_inline_tokens(token.get("tokens") or [], resolved_style_context)

            elif token_type == "strong":
                bold_content = self._render_inline_tokens(token.get("tokens") or [], resolved_style_context)
                result += self._theme["bold"](bold_content) + style_prefix

            elif token_type == "em":
                italic_content = self._render_inline_tokens(token.get("tokens") or [], resolved_style_context)
                result += self._theme["italic"](italic_content) + style_prefix

            elif token_type == "codespan":
                result += self._theme["code"](token["text"]) + style_prefix

            elif token_type == "link":
                link_text = self._render_inline_tokens(token.get("tokens") or [], resolved_style_context)
                styled_link = self._theme["link"](self._theme["underline"](link_text))
                if get_capabilities()["hyperlinks"]:
                    # OSC 8: render as a clickable hyperlink. The URL is not printed
                    # inline, so we always show only the link text regardless of
                    # whether it matches href.
                    result += hyperlink(styled_link, token["href"]) + style_prefix
                else:
                    # Fallback: print URL in parentheses when text differs from href.
                    # Compare raw token text (not styled) against href for the equality
                    # check. For mailto: links strip the prefix (autolinked emails use
                    # text="foo@bar.com" but href="mailto:foo@bar.com").
                    href = token["href"]
                    href_for_comparison = href.removeprefix("mailto:")
                    if token["text"] in (href, href_for_comparison):
                        result += styled_link + style_prefix
                    else:
                        result += styled_link + self._theme["linkUrl"](f" ({href})") + style_prefix

            elif token_type == "br":
                result += "\n"

            elif token_type == "del":
                del_content = self._render_inline_tokens(token.get("tokens") or [], resolved_style_context)
                result += self._theme["strikethrough"](del_content) + style_prefix

            elif token_type == "html":
                # Render inline HTML as plain text
                raw = token.get("raw")
                if isinstance(raw, str):
                    result += apply_text_with_newlines(raw)

            else:
                # Handle any other inline token types as plain text
                text = token.get("text")
                if isinstance(text, str):
                    result += apply_text_with_newlines(text)

        while style_prefix and result.endswith(style_prefix):
            result = result[: -len(style_prefix)]

        return result

    def _get_ordered_list_marker(self, item: dict) -> str | None:
        match = _ORDERED_MARKER_RE.match(item.get("raw") or "")
        return f"{match.group(1)} " if match is not None else None

    def _get_unordered_list_marker(self, item: dict) -> str | None:
        match = _UNORDERED_MARKER_RE.match(item.get("raw") or "")
        return f"{match.group(1)} " if match is not None else None

    def _render_list(self, token: dict, depth: int, width: int, style_context: dict | None) -> list[str]:
        """Render a list with proper nesting support."""
        lines: list[str] = []
        indent = "    " * depth
        # Use the list's start property (defaults to 1 for ordered lists)
        start_number = token["start"] if isinstance(token["start"], int) else 1

        preserve_markers = self._options.get("preserveOrderedListMarkers")
        items = token["items"]
        for i, item in enumerate(items):
            is_last_item = i == len(items) - 1
            if token["ordered"]:
                bullet = (
                    (self._get_ordered_list_marker(item) or f"{start_number + i}. ")
                    if preserve_markers
                    else f"{start_number + i}. "
                )
            else:
                bullet = (self._get_unordered_list_marker(item) or "- ") if preserve_markers else "- "
            task_marker = f"[{'x' if item['checked'] else ' '}] " if item["task"] else ""
            marker = bullet + task_marker
            first_prefix = indent + self._theme["listBullet"](marker)
            continuation_prefix = indent + " " * visible_width(marker)
            item_width = max(1, width - visible_width(first_prefix))
            rendered_any_line = False

            for item_token in item["tokens"]:
                if item_token["type"] == "list":
                    lines.extend(self._render_list(item_token, depth + 1, width, style_context))
                    rendered_any_line = True
                    continue

                item_lines = self._render_token(item_token, item_width, None, style_context)
                for line in item_lines:
                    for wrapped_line in wrap_text_with_ansi(line, item_width):
                        line_prefix = continuation_prefix if rendered_any_line else first_prefix
                        lines.append(line_prefix + wrapped_line)
                        rendered_any_line = True

            if not rendered_any_line:
                lines.append(first_prefix)

            if token["loose"] and not is_last_item:
                lines.append("")

        return lines

    def _get_longest_word_width(self, text: str, max_width: int | None = None) -> int:
        """Get the visible width of the longest word in a string."""
        longest = 0
        for word in text.split():
            longest = max(longest, visible_width(word))
        if max_width is None:
            return longest
        return min(longest, max_width)

    def _wrap_cell_text(self, text: str, max_width: int) -> list[str]:
        """Wrap a table cell to fit into a column.

        Delegates to wrap_text_with_ansi() so ANSI codes + long tokens are
        handled consistently with the rest of the renderer.
        """
        return wrap_text_with_ansi(text, max(1, max_width))

    def _render_table(
        self,
        token: dict,
        available_width: int,
        next_token_type: str | None,
        style_context: dict | None,
    ) -> list[str]:
        """Render a table with width-aware cell wrapping.

        Cells that don't fit are wrapped to multiple lines.
        """
        lines: list[str] = []
        num_cols = len(token["header"])

        if num_cols == 0:
            return lines

        # Calculate border overhead: "│ " + (n-1) * " │ " + " │"
        # = 2 + (n-1) * 3 + 2 = 3n + 1
        border_overhead = 3 * num_cols + 1
        available_for_cells = available_width - border_overhead
        if available_for_cells < num_cols:
            # Too narrow to render a stable table. Fall back to raw markdown.
            fallback_lines = wrap_text_with_ansi(token["raw"], available_width) if token.get("raw") else []
            if next_token_type and next_token_type != "space":
                fallback_lines.append("")
            return fallback_lines

        # Calculate natural column widths (what each column needs without constraints)
        natural_widths: list[int] = [0] * num_cols
        min_word_widths: list[int] = [1] * num_cols
        for i in range(num_cols):
            header_text = self._render_inline_tokens(token["header"][i].get("tokens") or [], style_context)
            natural_widths[i] = visible_width(header_text)
            min_word_widths[i] = max(1, self._get_longest_word_width(header_text, _MAX_UNBROKEN_WORD_WIDTH))
        for row in token["rows"]:
            for i, cell in enumerate(row):
                cell_text = self._render_inline_tokens(cell.get("tokens") or [], style_context)
                natural_widths[i] = max(natural_widths[i], visible_width(cell_text))
                min_word_widths[i] = max(
                    min_word_widths[i],
                    self._get_longest_word_width(cell_text, _MAX_UNBROKEN_WORD_WIDTH),
                )

        min_column_widths = min_word_widths
        min_cells_width = sum(min_column_widths)

        if min_cells_width > available_for_cells:
            min_column_widths = [1] * num_cols
            remaining = available_for_cells - num_cols

            if remaining > 0:
                total_weight = sum(max(0, width - 1) for width in min_word_widths)
                growth = [
                    int((max(0, width - 1) / total_weight) * remaining) if total_weight > 0 else 0
                    for width in min_word_widths
                ]

                for i in range(num_cols):
                    min_column_widths[i] += growth[i]

                allocated = sum(growth)
                leftover = remaining - allocated
                for i in range(num_cols):
                    if leftover <= 0:
                        break
                    min_column_widths[i] += 1
                    leftover -= 1

            min_cells_width = sum(min_column_widths)

        # Calculate column widths that fit within available width
        total_natural_width = sum(natural_widths) + border_overhead

        if total_natural_width <= available_width:
            # Everything fits naturally
            column_widths = [max(width, min_column_widths[i]) for i, width in enumerate(natural_widths)]
        else:
            # Need to shrink columns to fit
            total_grow_potential = sum(max(0, width - min_column_widths[i]) for i, width in enumerate(natural_widths))
            extra_width = max(0, available_for_cells - min_cells_width)
            column_widths = []
            for i, min_width in enumerate(min_column_widths):
                natural_width = natural_widths[i]
                min_width_delta = max(0, natural_width - min_width)
                grow = 0
                if total_grow_potential > 0:
                    grow = int((min_width_delta / total_grow_potential) * extra_width)
                column_widths.append(min_width + grow)

            # Adjust for rounding errors - distribute remaining space
            allocated = sum(column_widths)
            remaining = available_for_cells - allocated
            while remaining > 0:
                grew = False
                for i in range(num_cols):
                    if remaining <= 0:
                        break
                    if column_widths[i] < natural_widths[i]:
                        column_widths[i] += 1
                        remaining -= 1
                        grew = True
                if not grew:
                    break

        # Render top border
        top_border_cells = ["─" * w for w in column_widths]
        lines.append(f"┌─{'─┬─'.join(top_border_cells)}─┐")

        # Render header with wrapping
        header_cell_lines = [
            self._wrap_cell_text(self._render_inline_tokens(cell.get("tokens") or [], style_context), column_widths[i])
            for i, cell in enumerate(token["header"])
        ]
        header_line_count = max(len(cell_lines) for cell_lines in header_cell_lines)

        for line_idx in range(header_line_count):
            row_parts = []
            for col_idx, cell_lines in enumerate(header_cell_lines):
                text = cell_lines[line_idx] if line_idx < len(cell_lines) else ""
                padded = text + " " * max(0, column_widths[col_idx] - visible_width(text))
                row_parts.append(self._theme["bold"](padded))
            lines.append(f"│ {' │ '.join(row_parts)} │")

        # Render separator
        separator_cells = ["─" * w for w in column_widths]
        separator_line = f"├─{'─┼─'.join(separator_cells)}─┤"
        lines.append(separator_line)

        # Render rows with wrapping
        for row_index, row in enumerate(token["rows"]):
            row_cell_lines = [
                self._wrap_cell_text(
                    self._render_inline_tokens(cell.get("tokens") or [], style_context), column_widths[i]
                )
                for i, cell in enumerate(row)
            ]
            row_line_count = max(len(cell_lines) for cell_lines in row_cell_lines)

            for line_idx in range(row_line_count):
                row_parts = []
                for col_idx, cell_lines in enumerate(row_cell_lines):
                    text = cell_lines[line_idx] if line_idx < len(cell_lines) else ""
                    row_parts.append(text + " " * max(0, column_widths[col_idx] - visible_width(text)))
                lines.append(f"│ {' │ '.join(row_parts)} │")

            if row_index < len(token["rows"]) - 1:
                lines.append(separator_line)

        # Render bottom border
        bottom_border_cells = ["─" * w for w in column_widths]
        lines.append(f"└─{'─┴─'.join(bottom_border_cells)}─┘")

        if next_token_type and next_token_type != "space":
            lines.append("")  # Add spacing after table
        return lines
