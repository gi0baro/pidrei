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
``html`` (raw), ``image`` (text).

Deltas vs marked (documented, none observable in pi's test spec):

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
"""

import re

from markdown_it import MarkdownIt


__all__ = ["lex", "trim_partial_closing_fences"]

_md = MarkdownIt("commonmark", {"linkify": True}).enable(["strikethrough", "table", "linkify"])
# Keep text_special tokens (escapes/entities) separate so escape tokens can be
# emitted like marked does.
_md.disable("text_join")

_TASK_MARKER_RE = re.compile(r"^\[[ xX]\] +")
_FENCE_MARKER_RE = re.compile(r"^(`{3,}|~{3,})")


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
