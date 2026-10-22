"""marked-shaped token stream on top of markdown-it-py.

pi's Markdown component consumes the token tree of the ``marked`` JS parser.
This module runs markdown-it-py (CommonMark + strikethrough/table/linkify,
matching pi's GFM-ish surface) and converts its flat token stream into the
nested, marked-shaped dict tokens the renderer was written against:

Block tokens: ``heading`` (depth, tokens), ``paragraph``/``text`` (tokens;
``text`` for list-item children, mirroring marked's ``state.top`` behavior),
``code`` (text, lang, raw), ``list`` (ordered, start, loose, items:
[{task, checked, raw, tokens}]), ``table`` (raw, header, rows), ``blockquote``
(tokens), ``hr``, ``html`` (raw), ``space``.

Inline tokens: ``text``, ``escape`` (raw, text), ``strong``/``em``/``del``
(tokens), ``codespan`` (text), ``link`` (href, text, tokens), ``br``,
``html`` (raw), ``image`` (text), ``latex`` (text, raw, pending).

``latexBlock`` (text, raw, pending) joins the block tokens. pi registers the
two LaTeX tokenizers as ``marked`` extensions; here they are a markdown-it
block rule (before ``fence``) and inline rule (before ``escape``), which puts
them at the same points in the scan: escaped delimiters and code spans are
consumed by their own rules first, so neither becomes math.

Deltas vs marked (documented; only the last two are observable in pi's test spec):

- ``space`` tokens are synthesized between sibling blocks whenever the line
  right before a block is blank (or ``>``-markers-only inside blockquotes).
  marked's per-rule trailing-newline consumption differs per block type, but
  the renderer's output is invariant: every block appends its own blank line
  exactly when the next token is not ``space``. Leading/trailing blank lines
  of the document produce no space tokens (marked emits them; nothing in pi's
  spec observes them).
- ``raw`` is reconstructed from source lines via token maps, so ``raw`` of
  blocks nested in blockquotes keeps the ``> `` prefixes (marked strips
  them). Only list-item marker sniffing, top-level fence trimming and the
  narrow-table fallback read ``raw``.
- Entities are kept as their source text (marked leaves them undecoded).
- A line opening display math ends the paragraph above it, as in marked, but only at the line's
  start. marked looks from the paragraph's second character, so it splits a line starting with an
  escaped ``\\\\[`` after the first backslash; here that stays an escaped bracket.
- A table needs a ``|`` in its header row (markdown-it's table rule). marked also takes a header
  without one when the delimiter row has one, which only adds single-column tables such as
  "Value" over "| --- |"; here those stay paragraph text.
"""

import re

from markdown_it import MarkdownIt
from markdown_it.common.utils import isMdAsciiPunct, isPunctChar
from markdown_it.rules_inline.state_inline import Delimiter


__all__ = ["lex", "trim_partial_closing_fences"]

_md = MarkdownIt("commonmark", {"linkify": True}).enable(["strikethrough", "table", "linkify"])
# Keep text_special tokens (escapes/entities) separate so escape tokens can be
# emitted like marked does.
_md.disable("text_join")

_TASK_MARKER_RE = re.compile(r"^\[[ xX]\] +")
_FENCE_MARKER_RE = re.compile(r"^(`{3,}|~{3,})")

# --- LaTeX math (pi registers these as `marked` tokenizer extensions) --------

_PENDING_DOLLAR_MATH_RE = re.compile(r"\\[A-Za-z]+|[_^=+*/<>()\[\]|±≤≥≠≈∈→⇒∞∫∑√-]")
_TRAILING_SPACE_RE = re.compile(r"\s$")
_LEADING_DIGIT_RE = re.compile(r"^\d")
_SHELL_VARIABLE_RE = re.compile(r"^[A-Z_][A-Z0-9_]*(?:[^A-Za-z0-9_\s])?$")
_LEADING_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*")
_DOLLAR_THEN_SPACE_RE = re.compile(r"^\$\s")
_BLOCK_DOLLAR_RE = re.compile(r"^ {0,3}\$\$[ \t]*(?:\n)?([\s\S]*?)\$\$[ \t]*(?:\n|$)")
_BLOCK_BRACKET_RE = re.compile(r"^ {0,3}\\\[[ \t]*(?:\n)?([\s\S]*?)\\\][ \t]*(?:\n|$)")
_PENDING_BLOCK_BRACKET_RE = re.compile(r"^ {0,3}\\\[[ \t]*(?:\n)?([\s\S]*)$")
_PENDING_BLOCK_DOLLAR_RE = re.compile(r"^ {0,3}\$\$[ \t]*(?:\n)?([\s\S]*)$")
_TABLE_DELIMITER_ROW_RE = re.compile(r"^ {0,3}\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$")


def _is_escaped(source: str, index: int) -> bool:
    backslashes = 0
    position = index - 1
    while position >= 0 and source[position] == "\\":
        backslashes += 1
        position -= 1
    return backslashes % 2 == 1


def _find_closing_delimiter(source: str, closing: str, start: int) -> int:
    index = source.find(closing, start)
    while index >= 0 and _is_escaped(source, index):
        index = source.find(closing, index + len(closing))
    return index


def _looks_like_pending_dollar_math(source: str) -> bool:
    return bool(_PENDING_DOLLAR_MATH_RE.search(source))


def _tokenize_inline_latex(source: str) -> dict | None:
    """Records: {"raw", "text", "pending"?} or None when this is not math."""
    if source.startswith("$$"):
        opening = closing = "$$"
    elif source.startswith("\\("):
        opening, closing = "\\(", "\\)"
    elif source.startswith("\\["):
        opening, closing = "\\[", "\\]"
    elif source.startswith("$") and not _DOLLAR_THEN_SPACE_RE.match(source):
        opening = closing = "$"
    else:
        return None

    closing_index = _find_closing_delimiter(source, closing, len(opening))
    if closing_index >= 0 and opening == "$":
        body = source[len(opening) : closing_index]
        after = source[closing_index + 1 :]
        if (
            _TRAILING_SPACE_RE.search(body)
            or _LEADING_DIGIT_RE.match(after)
            or (_SHELL_VARIABLE_RE.match(body) and _LEADING_IDENTIFIER_RE.match(after))
            or "`" in body
        ):
            return None

    if closing_index < 0:
        pending_source = source[len(opening) :]
        if opening.startswith("\\") or _looks_like_pending_dollar_math(pending_source):
            return {"raw": source, "text": pending_source, "pending": True}
        return None

    text = source[len(opening) : closing_index]
    if not text or "\n" in text:
        return None

    return {"raw": source[: closing_index + len(closing)], "text": text}


# A GFM table starts with a header row followed by a delimiter row such as "|---|---|" or "| --- |".
# The header needs a "|" only when the delimiter has none: a pipe-free delimiter without a pipe in the
# header is a thematic break, not a table.
def _contains_table_start(source: str) -> bool:
    lines = source.split("\n")
    for index in range(len(lines) - 1):
        delimiter_row = lines[index + 1]
        if ("|" in delimiter_row or "|" in lines[index]) and _TABLE_DELIMITER_ROW_RE.match(delimiter_row):
            return True
    return False


def _tokenize_block_latex(source: str) -> dict | None:
    # Display math never spans a table. Without this check, an unclosed "$$" or "\[" pairs with a later
    # opener or stays pending, swallowing the table. See https://github.com/earendil-works/pi/issues/11159.
    dollar_match = _BLOCK_DOLLAR_RE.match(source)
    if dollar_match and dollar_match.group(1) and not _contains_table_start(dollar_match.group(1)):
        return {"raw": dollar_match.group(0), "text": dollar_match.group(1).strip()}

    bracket_match = _BLOCK_BRACKET_RE.match(source)
    if bracket_match and bracket_match.group(1) and not _contains_table_start(bracket_match.group(1)):
        return {"raw": bracket_match.group(0), "text": bracket_match.group(1).strip()}

    pending_bracket = _PENDING_BLOCK_BRACKET_RE.match(source)
    if pending_bracket and pending_bracket.group(1) and not _contains_table_start(pending_bracket.group(1)):
        return {"raw": pending_bracket.group(0), "text": pending_bracket.group(1), "pending": True}
    pending_dollar = _PENDING_BLOCK_DOLLAR_RE.match(source)
    if (
        pending_dollar
        and pending_dollar.group(1)
        and _looks_like_pending_dollar_math(pending_dollar.group(1))
        and not _contains_table_start(pending_dollar.group(1))
    ):
        return {"raw": pending_dollar.group(0), "text": pending_dollar.group(1), "pending": True}
    return None


def _latex_block_rule(state, start_line: int, end_line: int, silent: bool) -> bool:
    """markdown-it block rule standing in for pi's `latexBlock` extension."""
    if state.is_code_block(start_line):
        return False
    start = state.bMarks[start_line] + state.tShift[start_line]
    indent = state.sCount[start_line]
    # Every block form opens with "$$" or "\[" at the line's start. Checking first avoids copying the
    # rest of the document for each paragraph line, where this rule runs as a terminator.
    if indent > 3 or not state.src.startswith(("$$", "\\["), start):
        return False
    source = " " * indent + state.src[start:]
    token = _tokenize_block_latex(source)
    if not token:
        return False
    if silent:
        return True

    consumed = len(token["raw"]) - indent
    line = start_line
    remaining = consumed
    while line < end_line and remaining > 0:
        line_length = state.eMarks[line] - (state.bMarks[line] + state.tShift[line]) + 1
        remaining -= line_length
        line += 1
    state.line = max(line, start_line + 1)

    md_token = state.push("latex_block", "", 0)
    md_token.content = token["text"]
    md_token.meta = {"raw": token["raw"], "pending": bool(token.get("pending"))}
    md_token.map = [start_line, state.line]
    return True


def _latex_inline_rule(state, silent: bool) -> bool:
    """markdown-it inline rule standing in for pi's `latex` extension."""
    token = _tokenize_inline_latex(state.src[state.pos :])
    if not token:
        return False
    if not silent:
        md_token = state.push("latex_inline", "", 0)
        md_token.content = token["text"]
        md_token.meta = {"raw": token["raw"], "pending": bool(token.get("pending"))}
    state.pos += len(token["raw"])
    return True


# marked cuts a paragraph where a line opens display math, so the block rule ends paragraphs too.
_md.block.ruler.before("fence", "latex_block", _latex_block_rule, {"alt": ["paragraph"]})
_md.inline.ruler.before("escape", "latex_inline", _latex_inline_rule)

# --- CJK emphasis (pi uses marked-cjk-friendly 0.1.2's emStrong tokenizer) ---

# marked-cjk-friendly's CJK set, verbatim.
_CJK_RE = re.compile(
    r"[ᄀ-ᇿ₩〈-〉☰-☷⚊-⚏⺀-⺙⺛-⻳⼀-⿕"
    r"⿰-〾ぁ-ゖ゙-ヿㄅ-ㄯㄱ-ㆎ㆐-㇥㇯-㈞"
    r"㈠-㉇㉐-ꒌ꒐-꓆ꥠ-ꥼ가-힣ힰ-ퟆퟋ-ퟻ"
    r"豈-﫿︐-︙︰-﹒﹔-﹦﹨-﹫！-ﾾￂ-ￇ"
    r"ￊ-ￏￒ-ￗￚ-ￜ￠-￦￨-￮\U00016fe0-\U00016fe4"
    r"\U00016ff0-\U00016ff6\U00017000-\U00018cda\U00018cff-\U00018d20\U00018d80-\U00018df2"
    r"\U00018e00-\U00019191\U000191a0-\U000191d2\U0001aff0-\U0001aff3\U0001aff5-\U0001affb"
    r"\U0001affd-\U0001affe\U0001b000-\U0001b128\U0001b132\U0001b150-\U0001b152\U0001b155"
    r"\U0001b164-\U0001b168\U0001b170-\U0001b2fb\U0001d300-\U0001d356\U0001d360-\U0001d376\U0001f1ae"
    r"\U0001f200\U0001f202\U0001f210-\U0001f219\U0001f21b-\U0001f22e\U0001f230-\U0001f231\U0001f237"
    r"\U0001f23b\U0001f240-\U0001f248\U0001f260-\U0001f265\U0001f7da\U00020000-\U0003fffd]"
)


def _opens_after_cjk(src: str, start: int) -> bool:
    """Whether the character before an opening run is CJK, looking through a non-emoji variation
    selector to its base; an Ideographic Variation Selector counts as CJK."""
    index = start - 1
    if "︀" <= src[index] <= "︎" and index >= 1:
        index -= 1
    return "\U000e0100" <= src[index] <= "\U000e01ef" or bool(_CJK_RE.match(src[index]))


def _cjk_emphasis_rule(state, silent: bool) -> bool:
    """markdown-it's emphasis tokenizer with marked-cjk-friendly's flanking rules.

    The CommonMark scan stands, widened next to CJK characters: a run followed by punctuation also
    opens when either neighbour is CJK, and any run next to a CJK character can close (so
    ``**这是测试。**后面`` is bold). Without a CJK neighbour nothing changes, and strikethrough keeps
    its own scan, as pi takes only marked-cjk-friendly's ``emStrong``.
    """
    start = state.pos
    marker = state.src[start]
    if silent or marker not in ("_", "*"):
        return False
    scanned = state.scanDelims(start, marker == "*")
    end = start + scanned.length
    next_char = state.src[end] if end < state.posMax else " "
    next_cjk = bool(_CJK_RE.match(next_char))
    next_punct = isMdAsciiPunct(ord(next_char)) or isPunctChar(next_char)
    can_open = scanned.can_open or (next_punct and (next_cjk or (start > 0 and _opens_after_cjk(state.src, start))))
    can_close = scanned.can_close or next_cjk or (start > 0 and bool(_CJK_RE.match(state.src[start - 1])))
    for _ in range(scanned.length):
        token = state.push("text", "", 0)
        token.content = marker
        state.delimiters.append(
            Delimiter(
                marker=ord(marker),
                length=scanned.length,
                token=len(state.tokens) - 1,
                end=-1,
                open=can_open,
                close=can_close,
            )
        )
    state.pos = end
    return True


_md.inline.ruler.at("emphasis", _cjk_emphasis_rule)


def lex(text: str) -> list[dict]:
    """Lex markdown into a marked-shaped token list."""
    lines = text.split("\n")
    nodes = _nest(_md.parse(text))
    return _convert_siblings(nodes, lines, in_list_item=False)


def trim_partial_closing_fences(tokens: list[dict]) -> None:
    """Trim streamed partial closing fences so code blocks do not shrink/flicker
    when the final fence character arrives (port of pi's helper)."""
    token = tokens[-1] if tokens else None
    if token is None:
        return
    if token["type"] == "list":
        items = token["items"]
        trim_partial_closing_fences(items[-1]["tokens"] if items else [])
        return
    if token["type"] == "blockquote":
        trim_partial_closing_fences(token.get("tokens") or [])
        return
    if token["type"] != "code":
        return

    raw = token.get("raw") or ""
    marker_match = _FENCE_MARKER_RE.match(raw)
    marker = marker_match.group(1) if marker_match is not None else None
    last_line = raw.split("\n")[-1]
    if not marker or not last_line or len(last_line) >= len(marker) or last_line != marker[0] * len(last_line):
        return

    token["text"] = re.sub(r"\n$", "", token["text"][: -len(last_line)])


# =============================================================================
# Flat markdown-it stream → nested nodes
# =============================================================================


def _nest(tokens) -> list[dict]:
    root: list[dict] = []
    stack = [root]
    for token in tokens:
        if token.nesting == 1:
            node = {"token": token, "children": []}
            stack[-1].append(node)
            stack.append(node["children"])
        elif token.nesting == -1:
            stack.pop()
        else:
            stack[-1].append({"token": token, "children": None})
    return root


# =============================================================================
# Block conversion
# =============================================================================


# A separator line between sibling blocks: blank, or blockquote markers only
# (inside blockquotes the "blank" line still carries its `>` prefixes).
_BLANK_SEPARATOR_RE = re.compile(r"^[ \t>]*$")


def _convert_siblings(nodes: list[dict], lines: list[str], *, in_list_item: bool) -> list[dict]:
    out: list[dict] = []
    has_prev = False
    for node in nodes:
        token_map = node["token"].map
        if (
            has_prev
            and token_map is not None
            and token_map[0] > 0
            and _BLANK_SEPARATOR_RE.match(lines[token_map[0] - 1]) is not None
        ):
            out.append({"type": "space"})
        converted = _convert_block(node, lines, in_list_item=in_list_item)
        if converted is not None:
            out.append(converted)
            has_prev = True
    return out


def _raw_from_map(token, lines: list[str]) -> str:
    if token.map is None:
        return ""
    return "\n".join(lines[token.map[0] : token.map[1]])


def _convert_block(node: dict, lines: list[str], *, in_list_item: bool) -> dict | None:
    token = node["token"]
    kind = token.type

    if kind == "heading_open":
        return {
            "type": "heading",
            "depth": int(token.tag[1]),
            "tokens": _inline_tokens_of(node),
        }

    if kind == "paragraph_open":
        return {
            "type": "text" if in_list_item else "paragraph",
            "tokens": _inline_tokens_of(node),
        }

    if kind in ("fence", "code_block"):
        content = token.content
        text = content.removesuffix("\n")
        code_token = {
            "type": "code",
            "text": text,
            "raw": _raw_from_map(token, lines),
        }
        if kind == "fence":
            info = token.info.strip()
            code_token["lang"] = info
        return code_token

    if kind in ("bullet_list_open", "ordered_list_open"):
        return _convert_list(node, lines, ordered=kind == "ordered_list_open")

    if kind == "blockquote_open":
        return {
            "type": "blockquote",
            "tokens": _convert_siblings(node["children"], lines, in_list_item=False),
        }

    if kind == "hr":
        return {"type": "hr"}

    if kind == "html_block":
        return {"type": "html", "raw": token.content}

    if kind == "latex_block":
        return {
            "type": "latexBlock",
            "text": token.content,
            "raw": token.meta["raw"],
            "pending": token.meta["pending"],
        }

    if kind == "table_open":
        return _convert_table(node, lines)

    if kind == "inline":
        # Shouldn't appear at block level, but keep the content visible.
        return {"type": "text", "tokens": _convert_inline(token.children or [])}

    return None


def _inline_tokens_of(node: dict) -> list[dict]:
    for child in node["children"]:
        if child["token"].type == "inline":
            return _convert_inline(child["token"].children or [])
    return []


def _convert_list(node: dict, lines: list[str], *, ordered: bool) -> dict:
    token = node["token"]

    start: int | str = ""
    if ordered:
        start_attr = token.attrGet("start")
        start = int(start_attr) if start_attr is not None else 1

    loose = False
    items: list[dict] = []
    for item_node in node["children"]:
        if item_node["token"].type != "list_item_open":
            continue
        for child in item_node["children"]:
            if child["token"].type == "paragraph_open" and not child["token"].hidden:
                loose = True
        item_tokens = _convert_siblings(item_node["children"], lines, in_list_item=True)
        item = {
            "task": False,
            "checked": False,
            "raw": _raw_from_map(item_node["token"], lines),
            "tokens": item_tokens,
        }
        _detect_task_marker(item)
        items.append(item)

    return {
        "type": "list",
        "ordered": ordered,
        "start": start,
        "loose": loose,
        "items": items,
    }


def _detect_task_marker(item: dict) -> None:
    """GFM task list detection, like marked: ``[ ] ``/``[x] `` at item start."""
    tokens = item["tokens"]
    if not tokens or tokens[0]["type"] not in ("text", "paragraph"):
        return
    inline = tokens[0].get("tokens")
    if not inline or inline[0]["type"] != "text":
        return
    match = _TASK_MARKER_RE.match(inline[0]["text"])
    if match is None:
        return
    item["task"] = True
    item["checked"] = match.group(0) != "[ ] "
    inline[0]["text"] = inline[0]["text"][match.end() :]


def _convert_table(node: dict, lines: list[str]) -> dict:
    header: list[dict] = []
    rows: list[list[dict]] = []

    def cells_of(tr_node: dict) -> list[dict]:
        cells = []
        for cell_node in tr_node["children"]:
            if cell_node["token"].type in ("th_open", "td_open"):
                cells.append({"tokens": _inline_tokens_of(cell_node)})
        return cells

    for section in node["children"]:
        section_kind = section["token"].type
        if section_kind == "thead_open":
            for tr_node in section["children"]:
                if tr_node["token"].type == "tr_open":
                    header = cells_of(tr_node)
        elif section_kind == "tbody_open":
            for tr_node in section["children"]:
                if tr_node["token"].type == "tr_open":
                    rows.append(cells_of(tr_node))

    return {
        "type": "table",
        "raw": _raw_from_map(node["token"], lines),
        "header": header,
        "rows": rows,
    }


# =============================================================================
# Inline conversion
# =============================================================================

_INLINE_WRAPPERS = {
    "strong_open": ("strong_close", "strong"),
    "em_open": ("em_close", "em"),
    "s_open": ("s_close", "del"),
}


def _convert_inline(children: list) -> list[dict]:
    out: list[dict] = []
    text_acc: list[str] = []

    def flush() -> None:
        if text_acc:
            out.append({"type": "text", "text": "".join(text_acc)})
            text_acc.clear()

    i = 0
    while i < len(children):
        token = children[i]
        kind = token.type

        if kind == "text":
            text_acc.append(token.content)
        elif kind == "softbreak":
            text_acc.append("\n")
        elif kind == "text_special":
            if token.info == "escape":
                flush()
                out.append({"type": "escape", "raw": token.markup, "text": token.content})
            else:
                # Entities stay undecoded, like marked's lexer.
                text_acc.append(token.markup or token.content)
        elif kind == "code_inline":
            flush()
            out.append({"type": "codespan", "text": token.content})
        elif kind == "latex_inline":
            flush()
            out.append(
                {
                    "type": "latex",
                    "text": token.content,
                    "raw": token.meta["raw"],
                    "pending": token.meta["pending"],
                }
            )
        elif kind == "hardbreak":
            flush()
            out.append({"type": "br"})
        elif kind == "html_inline":
            flush()
            out.append({"type": "html", "raw": token.content})
        elif kind == "image":
            flush()
            out.append({"type": "image", "text": token.content})
        elif kind in _INLINE_WRAPPERS:
            close_kind, marked_kind = _INLINE_WRAPPERS[kind]
            inner, i = _slice_to_close(children, i, close_kind)
            flush()
            out.append({"type": marked_kind, "tokens": _convert_inline(inner)})
        elif kind == "link_open":
            inner, i = _slice_to_close(children, i, "link_close")
            flush()
            inner_tokens = _convert_inline(inner)
            out.append(
                {
                    "type": "link",
                    "href": token.attrGet("href") or "",
                    "text": _plain_text(inner_tokens),
                    "tokens": inner_tokens,
                }
            )
        else:
            text_acc.append(token.content)

        i += 1

    flush()
    return out


def _slice_to_close(children: list, open_index: int, close_kind: str) -> tuple[list, int]:
    """Return the tokens between an ``*_open`` and its matching close, and the
    index of the close token."""
    depth = 0
    for j in range(open_index + 1, len(children)):
        if children[j].type == children[open_index].type:
            depth += 1
        elif children[j].type == close_kind:
            if depth == 0:
                return list(children[open_index + 1 : j]), j
            depth -= 1
    return list(children[open_index + 1 :]), len(children) - 1


def _plain_text(tokens: list[dict]) -> str:
    parts: list[str] = []
    for token in tokens:
        if token.get("tokens"):
            parts.append(_plain_text(token["tokens"]))
        elif "text" in token:
            parts.append(token["text"])
        elif "raw" in token:
            parts.append(token["raw"])
    return "".join(parts)
