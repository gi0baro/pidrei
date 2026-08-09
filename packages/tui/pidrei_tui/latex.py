"""Mirror of pi tui src/latex.ts.

Renders a supported subset of LaTeX math as terminal-friendly Unicode text,
returning None for anything unsupported or malformed so the caller can fall
back to the source. Display math stacks fractions, operator limits and
matrices; inline math stays on one line.

Port notes: JS `\\p{L}\\p{N}` property classes have no `re` equivalent, so
letters-or-digits is spelled `[^\\W_]` (word characters minus underscore) and
`\\p{N}` becomes `\\d` (Nd only, where JS also matches Nl/No — no supported
input reaches the difference). Private-use markers keep their code points.
"""

import re

from .utils import visible_width


# --- generated from pi tui src/latex.ts (tables only) ---

SYMBOLS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ϵ",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "ϑ",
    "iota": "ι",
    "kappa": "κ",
    "varkappa": "ϰ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "varpi": "ϖ",
    "rho": "ρ",
    "varrho": "ϱ",
    "sigma": "σ",
    "varsigma": "ς",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "ϕ",
    "varphi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Upsilon": "Υ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "pm": "±",
    "mp": "∓",
    "times": "×",
    "div": "÷",
    "cdot": "·",
    "ast": "∗",
    "star": "⋆",
    "circ": "∘",
    "bullet": "•",
    "oplus": "⊕",
    "ominus": "⊖",
    "otimes": "⊗",
    "oslash": "⊘",
    "odot": "⊙",
    "bigcirc": "○",
    "dagger": "†",
    "ddagger": "‡",
    "amalg": "⨿",
    "uplus": "⊎",
    "sqcap": "⊓",
    "sqcup": "⊔",
    "triangleleft": "◁",
    "triangleright": "▷",
    "wr": "≀",
    "cap": "∩",
    "cup": "∪",
    "bigcap": "⋂",
    "bigcup": "⋃",
    "bigwedge": "⋀",
    "bigvee": "⋁",
    "bigsqcup": "⨆",
    "biguplus": "⨄",
    "bigoplus": "⨁",
    "bigotimes": "⨂",
    "bigodot": "⨀",
    "setminus": "∖",
    "in": "∈",
    "notin": "∉",
    "ni": "∋",
    "subset": "⊂",
    "supset": "⊃",
    "subseteq": "⊆",
    "supseteq": "⊇",
    "sqsubset": "⊏",
    "sqsupset": "⊐",
    "sqsubseteq": "⊑",
    "sqsupseteq": "⊒",
    "prec": "≺",
    "preceq": "≼",
    "succ": "≻",
    "succeq": "≽",
    "ll": "≪",
    "gg": "≫",
    "le": "≤",
    "leq": "≤",
    "leqslant": "≤",
    "ge": "≥",
    "geq": "≥",
    "geqslant": "≥",
    "ne": "≠",
    "neq": "≠",
    "equiv": "≡",
    "approx": "≈",
    "sim": "∼",
    "simeq": "≃",
    "cong": "≅",
    "asymp": "≍",
    "doteq": "≐",
    "propto": "∝",
    "parallel": "∥",
    "perp": "⊥",
    "mid": "∣",
    "vdash": "⊢",
    "dashv": "⊣",
    "models": "⊨",
    "Vdash": "⊩",
    "Vvdash": "⊪",
    "nvdash": "⊬",
    "nvDash": "⊭",
    "forall": "∀",
    "exists": "∃",
    "nexists": "∄",
    "neg": "¬",
    "land": "∧",
    "wedge": "∧",
    "lor": "∨",
    "vee": "∨",
    "to": "→",
    "rightarrow": "→",
    "longrightarrow": "→",
    "leftarrow": "←",
    "longleftarrow": "←",
    "gets": "←",
    "leftrightarrow": "↔",
    "longleftrightarrow": "↔",
    "hookleftarrow": "↩",
    "hookrightarrow": "↪",
    "twoheadleftarrow": "↞",
    "twoheadrightarrow": "↠",
    "leftharpoonup": "↼",
    "leftharpoondown": "↽",
    "rightharpoonup": "⇀",
    "rightharpoondown": "⇁",
    "rightleftharpoons": "⇌",
    "leftrightharpoons": "⇋",
    "nearrow": "↗",
    "searrow": "↘",
    "swarrow": "↙",
    "nwarrow": "↖",
    "rightsquigarrow": "⇝",
    "leadsto": "⇝",
    "Rightarrow": "⇒",
    "Longrightarrow": "⇒",
    "Leftarrow": "⇐",
    "Longleftarrow": "⇐",
    "Leftrightarrow": "⇔",
    "Longleftrightarrow": "⇔",
    "implies": "⇒",
    "iff": "⇔",
    "mapsto": "↦",
    "longmapsto": "↦",
    "uparrow": "↑",
    "downarrow": "↓",
    "partial": "∂",
    "nabla": "∇",
    "int": "∫",
    "iint": "∬",
    "iiint": "∭",
    "oint": "∮",
    "sum": "∑",
    "prod": "∏",
    "coprod": "∐",
    "infty": "∞",
    "emptyset": "∅",
    "varnothing": "∅",
    "angle": "∠",
    "therefore": "∴",
    "because": "∵",
    "aleph": "ℵ",
    "beth": "ℶ",
    "gimel": "ℷ",
    "daleth": "ℸ",
    "top": "⊤",
    "bot": "⊥",
    "triangle": "△",
    "square": "□",
    "lozenge": "◊",
    "checkmark": "✓",
    "complement": "∁",
    "wp": "℘",
    "prime": "′",
    "ldots": "…",
    "dots": "…",
    "cdots": "⋯",
    "vdots": "⋮",
    "ddots": "⋱",
    "ell": "ℓ",
    "hbar": "ℏ",
    "Im": "ℑ",
    "Re": "ℜ",
    "langle": "⟨",
    "rangle": "⟩",
    "vert": "|",
    "lvert": "|",
    "rvert": "|",
    "Vert": "‖",
    "lVert": "‖",
    "rVert": "‖",
    "lbrace": "{",
    "rbrace": "}",
    "backslash": "\\",
    "lfloor": "⌊",
    "rfloor": "⌋",
    "lceil": "⌈",
    "rceil": "⌉",
    "colon": ":",
}

NEGATED_SYMBOLS = {
    "<": "≮",
    ">": "≯",
    "=": "≠",
    "∈": "∉",
    "∋": "∌",
    "∣": "∤",
    "∥": "∦",
    "∼": "≁",
    "≃": "≄",
    "≅": "≇",
    "≈": "≉",
    "≡": "≢",
    "≤": "≰",
    "≥": "≱",
    "≺": "⊀",
    "≻": "⊁",
    "⊂": "⊄",
    "⊃": "⊅",
    "⊆": "⊈",
    "⊇": "⊉",
    "⊢": "⊬",
    "⊨": "⊭",
    "↔": "↮",
    "←": "↚",
    "→": "↛",
    "⇒": "⇏",
    "⇐": "⇍",
    "⇔": "⇎",
    "≼": "⋠",
    "≽": "⋡",
}

BLACKBOARD = {
    "C": "ℂ",
    "H": "ℍ",
    "N": "ℕ",
    "P": "ℙ",
    "Q": "ℚ",
    "R": "ℝ",
    "Z": "ℤ",
}

SUPERSCRIPTS = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "a": "ᵃ",
    "b": "ᵇ",
    "c": "ᶜ",
    "d": "ᵈ",
    "e": "ᵉ",
    "f": "ᶠ",
    "g": "ᵍ",
    "h": "ʰ",
    "i": "ⁱ",
    "j": "ʲ",
    "k": "ᵏ",
    "l": "ˡ",
    "m": "ᵐ",
    "n": "ⁿ",
    "o": "ᵒ",
    "p": "ᵖ",
    "r": "ʳ",
    "s": "ˢ",
    "t": "ᵗ",
    "u": "ᵘ",
    "v": "ᵛ",
    "w": "ʷ",
    "x": "ˣ",
    "y": "ʸ",
    "z": "ᶻ",
}

SUBSCRIPTS = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
    "a": "ₐ",
    "e": "ₑ",
    "h": "ₕ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "l": "ₗ",
    "m": "ₘ",
    "n": "ₙ",
    "o": "ₒ",
    "p": "ₚ",
    "r": "ᵣ",
    "s": "ₛ",
    "t": "ₜ",
    "u": "ᵤ",
    "v": "ᵥ",
    "x": "ₓ",
}

ACCENTS = {
    "acute": "́",
    "bar": "̅",
    "breve": "̆",
    "check": "̌",
    "ddot": "̈",
    "dot": "̇",
    "grave": "̀",
    "hat": "̂",
    "mathring": "̊",
    "overleftarrow": "⃖",
    "overleftrightarrow": "⃡",
    "overline": "̅",
    "overrightarrow": "⃗",
    "tilde": "̃",
    "underline": "̲",
    "vec": "⃗",
    "widehat": "̂",
    "widetilde": "̃",
}

NAMED_OPERATORS = frozenset(
    {
        "arccos",
        "arcsin",
        "arctan",
        "arg",
        "cos",
        "cosh",
        "cot",
        "coth",
        "csc",
        "deg",
        "det",
        "dim",
        "exp",
        "gcd",
        "hom",
        "inf",
        "ker",
        "lg",
        "lim",
        "liminf",
        "limsup",
        "ln",
        "log",
        "max",
        "min",
        "Pr",
        "sec",
        "sin",
        "sinh",
        "sup",
        "tan",
        "tanh",
    }
)

LIMIT_OPERATORS = frozenset(
    {
        "argmax",
        "argmin",
        "inf",
        "injlim",
        "lim",
        "liminf",
        "limsup",
        "max",
        "min",
        "projlim",
        "sup",
    }
)

DISPLAY_LIMIT_SYMBOLS = frozenset(
    {
        "bigcap",
        "bigcup",
        "bigodot",
        "bigoplus",
        "bigotimes",
        "bigsqcup",
        "biguplus",
        "bigvee",
        "bigwedge",
        "coprod",
        "int",
        "iint",
        "iiint",
        "oint",
        "prod",
        "sum",
    }
)

RELATION_COMMANDS = frozenset(
    {
        "Leftarrow",
        "Leftrightarrow",
        "Longleftarrow",
        "Longleftrightarrow",
        "Longrightarrow",
        "Rightarrow",
        "Vdash",
        "Vvdash",
        "approx",
        "asymp",
        "cong",
        "dashv",
        "doteq",
        "downarrow",
        "equiv",
        "ge",
        "geq",
        "geqslant",
        "gets",
        "gg",
        "hookleftarrow",
        "hookrightarrow",
        "iff",
        "implies",
        "in",
        "leadsto",
        "le",
        "leftarrow",
        "leftharpoondown",
        "leftharpoonup",
        "leftrightarrow",
        "leftrightharpoons",
        "leq",
        "leqslant",
        "ll",
        "longleftarrow",
        "longleftrightarrow",
        "longmapsto",
        "longrightarrow",
        "mapsto",
        "mid",
        "models",
        "ne",
        "nearrow",
        "neq",
        "ni",
        "notin",
        "nvdash",
        "nvDash",
        "nwarrow",
        "parallel",
        "perp",
        "prec",
        "preceq",
        "propto",
        "rightharpoondown",
        "rightharpoonup",
        "rightleftharpoons",
        "rightarrow",
        "rightsquigarrow",
        "searrow",
        "sim",
        "simeq",
        "sqsubset",
        "sqsubseteq",
        "sqsupset",
        "sqsupseteq",
        "subset",
        "subseteq",
        "succ",
        "succeq",
        "supset",
        "supseteq",
        "swarrow",
        "to",
        "triangleleft",
        "triangleright",
        "twoheadleftarrow",
        "twoheadrightarrow",
        "uparrow",
        "vdash",
    }
)

SPACING_COMMANDS = frozenset(
    {
        ",",
        ":",
        ";",
        " ",
        ">",
        "enspace",
        "enskip",
        "medspace",
        "quad",
        "qquad",
        "thickspace",
        "thinspace",
    }
)

IGNORED_COMMANDS = frozenset(
    {
        "displaystyle",
        "limits",
        "nolimits",
        "scriptstyle",
        "scriptscriptstyle",
        "textstyle",
    }
)

SIZE_COMMANDS = frozenset(
    {
        "big",
        "Big",
        "bigg",
        "Bigg",
        "bigl",
        "Bigl",
        "biggl",
        "Biggl",
        "bigr",
        "Bigr",
        "biggr",
        "Biggr",
    }
)

PLAIN_WRAPPERS = frozenset(
    {
        "emph",
        "mathcal",
        "mathbf",
        "mathfrak",
        "mathit",
        "mathrm",
        "mathnormal",
        "mathscr",
        "mathsf",
        "mathtt",
        "mathup",
        "mbox",
        "overbrace",
        "pmb",
        "smash",
        "substack",
        "text",
        "textbf",
        "textit",
        "textmd",
        "textnormal",
        "textrm",
        "textsc",
        "textsf",
        "textsl",
        "texttt",
        "textup",
        "underbrace",
        "bm",
        "boldsymbol",
    }
)


NEGATIVE_SPACING_COMMANDS = frozenset({"!", "negmedspace", "negthickspace", "negthinspace"})
NEGATIVE_SPACE = "\x00"

_LETTERS_OR_DIGITS = r"[^\W_]"


def _replace_characters(value: str, replacements: dict) -> str | None:
    result = ""
    for character in value:
        replacement = replacements.get(character)
        if replacement is None:
            return None
        result += replacement
    return result


_SCRIPT_OPERATOR_SPACING_RE = re.compile(r"\s*([=+-])\s*")
_ALPHA_ONLY_RE = re.compile(r"^[A-Za-z]+$")


def _format_script(value: str, kind: str) -> str:
    value = value.strip()
    replacements = SUBSCRIPTS if kind == "sub" else SUPERSCRIPTS
    unicode_value = _replace_characters(_SCRIPT_OPERATOR_SPACING_RE.sub(r"\1", value), replacements)
    if unicode_value is not None:
        return unicode_value

    prefix = "_" if kind == "sub" else "^"
    if len(value) == 1 or (kind == "sub" and _ALPHA_ONLY_RE.match(value)):
        return f"{prefix}{value}"
    return f"{prefix}({value})"


_SIMPLE_TOKEN_RE = re.compile(rf"^(?:{_LETTERS_OR_DIGITS}|\.)+$")
_SIMPLE_NUMBER_RE = re.compile(r"^(?:\d|\.)+$")


def _format_fraction(numerator: str, denominator: str) -> str:
    numerator = numerator.strip()
    denominator = denominator.strip()
    simple_numerator = bool(_SIMPLE_TOKEN_RE.match(numerator))
    simple_denominator = bool(_SIMPLE_NUMBER_RE.match(denominator)) or len(denominator) == 1
    left = numerator if simple_numerator else f"({numerator})"
    right = denominator if simple_denominator else f"({denominator})"
    return f"{left}/{right}"


def _format_root(value: str, symbol: str = "√") -> str:
    value = value.strip()
    return f"{symbol}{value}" if _SIMPLE_TOKEN_RE.match(value) else f"{symbol}({value})"


NAMED_OPERATOR_START = "\U000f0004"
NAMED_OPERATOR_END = "\U000f0005"
_NAMED_OPERATOR_LEFT_SPACING_RE = re.compile(rf"(?<={_LETTERS_OR_DIGITS}|[)\]}}\U000f0001])\U000f0004")
_NAMED_OPERATOR_RIGHT_SPACING_RE = re.compile(rf"\U000f0005(?={_LETTERS_OR_DIGITS}|[√\U000f0000])")
_HORIZONTAL_WHITESPACE_RUN_RE = re.compile(r"[ \t]+")


def _normalize_output(value: str) -> str:
    value = _NAMED_OPERATOR_LEFT_SPACING_RE.sub(" ", value)
    value = value.replace(NAMED_OPERATOR_START, "")
    value = _NAMED_OPERATOR_RIGHT_SPACING_RE.sub(" ", value)
    value = value.replace(NAMED_OPERATOR_END, "")
    lines = [_HORIZONTAL_WHITESPACE_RUN_RE.sub(" ", line).strip() for line in value.split("\n")]
    kept = [line for index, line in enumerate(lines) if len(line) > 0 or (0 < index < len(lines) - 1)]
    return "\n".join(kept).strip()


# Layout nodes are records the second pass stacks vertically:
# {"type": "fraction", "numerator", "denominator"},
# {"type": "operator", "operator", "lower"?, "upper"?},
# {"type": "matrix", "lines", "baseline"}.
# A layout is {"lines", "width", "baseline"}.

LAYOUT_MARKER_START = "\U000f0000"
LAYOUT_MARKER_END = "\U000f0001"
_LAYOUT_MARKER_RE = re.compile(r"\U000f0000(\d+)\U000f0001")
_TRAILING_LAYOUT_MARKER_RE = re.compile(r"\U000f0000(\d+)\U000f0001$")
PROTECTED_SPACE = "\U000f0002"


def _pad_layout_line(line: str, width: int, centered: bool = False) -> str:
    padding = max(0, width - visible_width(line))
    left = padding // 2 if centered else 0
    return f"{' ' * left}{line}{' ' * (padding - left)}"


def _join_layouts(layouts: list[dict]) -> dict:
    if not layouts:
        return {"lines": [""], "width": 0, "baseline": 0}
    baseline = max(layout["baseline"] for layout in layouts)
    below = max(len(layout["lines"]) - layout["baseline"] - 1 for layout in layouts)
    lines: list[str] = []
    for row in range(baseline + below + 1):
        line = ""
        for layout in layouts:
            source_row = row - baseline + layout["baseline"]
            if 0 <= source_row < len(layout["lines"]):
                line += _pad_layout_line(layout["lines"][source_row], layout["width"])
            else:
                line += " " * layout["width"]
        lines.append(line.rstrip())
    return {"lines": lines, "width": sum(layout["width"] for layout in layouts), "baseline": baseline}


def _render_layout(source: str, nodes: list[dict]) -> dict:
    rendered_lines: list[str] = []
    first_baseline = 0
    for source_line in source.split("\n"):
        layouts: list[dict] = []
        position = 0
        previous_node: dict | None = None
        for match in _LAYOUT_MARKER_RE.finditer(source_line):
            index = match.start()
            node_index = int(match.group(1))
            node = nodes[node_index] if node_index < len(nodes) else None
            if not node:
                continue
            if index > position:
                sliced = source_line[position:index]
                trimmed = (sliced.lstrip() if previous_node else sliced).rstrip()
                preserve_leading_space = (
                    previous_node is not None and previous_node["type"] == "matrix" and sliced[:1].isspace()
                )
                preserve_trailing_space = node["type"] == "matrix" and sliced[-1:].isspace()
                if trimmed:
                    text = f"{' ' if preserve_leading_space else ''}{trimmed}{' ' if preserve_trailing_space else ''}"
                elif preserve_leading_space or preserve_trailing_space:
                    text = " "
                else:
                    text = ""
                layouts.append({"lines": [text], "width": visible_width(text), "baseline": 0})
            if node["type"] == "fraction":
                numerator = _render_layout(node["numerator"], nodes)
                denominator = _render_layout(node["denominator"], nodes)
                content_width = max(numerator["width"], denominator["width"], 1)
                width = content_width + 2
                layouts.append(
                    {
                        "lines": [
                            *[_pad_layout_line(line, width, True) for line in numerator["lines"]],
                            f" {'─' * content_width} ",
                            *[_pad_layout_line(line, width, True) for line in denominator["lines"]],
                        ],
                        "width": width,
                        "baseline": len(numerator["lines"]),
                    }
                )
            elif node["type"] == "operator":
                content_width = max(
                    visible_width(node["operator"]),
                    0 if node.get("lower") is None else visible_width(node["lower"]),
                    0 if node.get("upper") is None else visible_width(node["upper"]),
                )
                lines: list[str] = []
                if node.get("upper") is not None:
                    lines.append(f"{_pad_layout_line(node['upper'], content_width, True)} ")
                lines.append(f"{_pad_layout_line(node['operator'], content_width, True)} ")
                if node.get("lower") is not None:
                    lines.append(f"{_pad_layout_line(node['lower'], content_width, True)} ")
                layouts.append(
                    {
                        "lines": lines,
                        "width": content_width + 1,
                        "baseline": 0 if node.get("upper") is None else 1,
                    }
                )
            else:
                width = max([0, *[visible_width(line) for line in node["lines"]]])
                layouts.append(
                    {
                        "lines": [_pad_layout_line(line, width) for line in node["lines"]],
                        "width": width,
                        "baseline": node["baseline"],
                    }
                )
            position = index + len(match.group(0))
            previous_node = node
        if position < len(source_line):
            sliced = source_line[position:]
            trimmed = sliced.lstrip() if previous_node else sliced
            text = (
                f" {trimmed}"
                if previous_node is not None and previous_node["type"] == "matrix" and sliced[:1].isspace()
                else trimmed
            )
            layouts.append({"lines": [text], "width": visible_width(text), "baseline": 0})
        line_layout = _join_layouts(layouts)
        if not rendered_lines:
            first_baseline = line_layout["baseline"]
        rendered_lines.extend(line_layout["lines"])
    return {
        "lines": rendered_lines,
        "width": max([0, *[visible_width(line) for line in rendered_lines]]),
        "baseline": first_baseline,
    }


_COMMAND_NAME_RE = re.compile(r"[A-Za-z]")
_LIMITS_MODIFIER_RE = re.compile(r"^\\(limits|nolimits)(?![A-Za-z])")
_ENVIRONMENT_ROW_SPLIT_RE = re.compile(r"\\\\(?:\[[^\]\n]*\])?")
_ENVIRONMENT_ARGUMENT_RE = re.compile(r"^\s*\{[^}]*\}")
_TRAILING_COMMA_RE = re.compile(r",\s*$")
_CASES_CONDITION_RE = re.compile(r"^(?:if|when|for|otherwise)\b", re.IGNORECASE)

_ALIGNED_ENVIRONMENTS = frozenset(
    {
        "aligned",
        "align",
        "align*",
        "alignedat",
        "alignat",
        "alignat*",
        "gather",
        "gathered",
        "multline",
        "multline*",
        "split",
    }
)
_MATRIX_ENVIRONMENTS = frozenset(
    {"array", "matrix", "smallmatrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix"}
)
_MATRIX_DELIMITERS = {
    "pmatrix": ("⎛", "⎞", "⎜", "⎟", "⎝", "⎠"),
    "bmatrix": ("⎡", "⎤", "⎢", "⎥", "⎣", "⎦"),
    "Bmatrix": ("⎧", "⎫", "⎨", "⎬", "⎩", "⎭"),
    "vmatrix": ("│", "│", "│", "│", "│", "│"),
    "Vmatrix": ("║", "║", "║", "║", "║", "║"),
}


class _LatexParser:
    def __init__(self, source: str, layout_nodes: list[dict], display: bool) -> None:
        self._source = source
        self._layout_nodes = layout_nodes
        self._display = display
        self._position = 0
        self._supported = True
        self._stack_fractions = True

    def render(self) -> str | None:
        rendered = self._parse_sequence()
        if not self._supported or self._position != len(self._source):
            return None
        return _normalize_output(rendered)

    def _parse_sequence(self, end_character: str | None = None) -> str:
        result = ""
        while self._position < len(self._source):
            character = self._source[self._position]
            if end_character and character == end_character:
                self._position += 1
                return result

            if character == "}":
                self._supported = False
                return result

            if character == "{":
                self._position += 1
                result += self._parse_sequence("}")
                continue

            if character == "\\":
                command = self._parse_command()
                if command == NEGATIVE_SPACE:
                    result = result.rstrip()
                    result = result.removesuffix(NAMED_OPERATOR_END)
                else:
                    result += command
                continue

            if character in ("^", "_"):
                self._position += 1
                result = result.rstrip()
                script = _format_script(self._parse_required_argument(False), "sub" if character == "_" else "sup")
                if result.endswith(NAMED_OPERATOR_END):
                    result = f"{result[: -len(NAMED_OPERATOR_END)]}{script}{NAMED_OPERATOR_END}"
                else:
                    result += script
                continue

            if character.isspace():
                result += self._parse_whitespace()
                continue

            if character in ("=", "<", ">"):
                result = f"{result.rstrip()} {character} "
                self._position += 1
                continue

            if character == "&":
                self._position += 1
                continue

            if character == "~":
                self._position += 1
                result += " "
                continue

            if character == ".":
                marker = _TRAILING_LAYOUT_MARKER_RE.search(result)
                node = self._layout_nodes[int(marker.group(1))] if marker else None
                if node is not None and node["type"] == "matrix":
                    last_line = len(node["lines"]) - 1
                    node["lines"][last_line] = f"{node['lines'][last_line]}{character}"
                    self._position += 1
                    continue

            result += character
            self._position += 1

        if end_character:
            self._supported = False
        return result

    def _parse_whitespace(self) -> str:
        while self._position < len(self._source) and self._source[self._position].isspace():
            self._position += 1
        return " "

    def _parse_command(self) -> str:
        self._position += 1
        if self._position >= len(self._source):
            self._supported = False
            return ""

        first = self._source[self._position]
        if _COMMAND_NAME_RE.match(first):
            start = self._position
            while self._position < len(self._source) and _COMMAND_NAME_RE.match(self._source[self._position]):
                self._position += 1
            command = self._source[start : self._position]
        else:
            command = first
            self._position += 1

        if command == "\\":
            return "\n"
        if command in SPACING_COMMANDS:
            return " "
        if command in NEGATIVE_SPACING_COMMANDS:
            return NEGATIVE_SPACE
        if command in IGNORED_COMMANDS:
            return ""
        if command in ("{", "}", "$", "%", "#", "_", "&"):
            return command
        if command == "|":
            return "‖"
        if command == "not":
            value = self._parse_required_argument(False).strip()
            negated = NEGATED_SYMBOLS.get(value)
            if negated is not None:
                return f" {negated} "
            if len(value) == 0:
                self._supported = False
                return ""
            return f" {value[0]}̸{value[1:]} "
        if command in LIMIT_OPERATORS:
            return self._parse_operator(command, "bracket", True, True)

        symbol = SYMBOLS.get(command)
        if symbol is not None:
            if command in DISPLAY_LIMIT_SYMBOLS:
                return self._parse_operator(symbol, "script", True)
            if command in ("cdot", "times") or command in RELATION_COMMANDS:
                return f" {symbol} "
            return symbol
        if command in NAMED_OPERATORS:
            return f"{NAMED_OPERATOR_START}{command}{NAMED_OPERATOR_END}"
        if command in SIZE_COMMANDS:
            return ""
        if command in ("left", "middle", "right"):
            if self._source[self._position : self._position + 1] == ".":
                self._position += 1
            return ""
        if command in ("frac", "dfrac", "tfrac"):
            should_stack = self._display and self._stack_fractions and command != "tfrac"
            numerator = self._parse_required_argument(not should_stack)
            denominator = self._parse_required_argument(not should_stack)
            if should_stack:
                self._layout_nodes.append(
                    {
                        "type": "fraction",
                        "numerator": _normalize_output(numerator),
                        "denominator": _normalize_output(denominator),
                    }
                )
                return f"{LAYOUT_MARKER_START}{len(self._layout_nodes) - 1}{LAYOUT_MARKER_END}"
            return _format_fraction(numerator, denominator)
        if command == "sqrt":
            degree = self._parse_optional_argument()
            degree = degree.strip() if degree is not None else None
            value = self._parse_required_argument()
            if degree is None or degree == "2":
                return _format_root(value)
            if degree == "3":
                return _format_root(value, "∛")
            if degree == "4":
                return _format_root(value, "∜")
            return f"{_format_script(degree, 'sup')}{_format_root(value)}"
        if command in ("boxed", "fbox"):
            return f"[{self._parse_required_argument().strip()}]"
        if command in ("binom", "dbinom", "tbinom"):
            return f"({self._parse_required_argument()} choose {self._parse_required_argument()})"
        accent = ACCENTS.get(command)
        if accent is not None:
            value = self._parse_required_argument()
            return f"{value}{accent}" if len(value) == 1 else f"{command}({value})"
        if command == "mathbb":
            value = self._parse_required_argument()
            return "".join(BLACKBOARD.get(character, character) for character in value)
        if command == "operatorname":
            starred = self._source[self._position : self._position + 1] == "*"
            if starred:
                self._position += 1
            operator = _normalize_output(self._parse_required_argument()).strip()
            return self._parse_operator(operator, "bracket", starred, True)
        if command in ("mod", "bmod"):
            return " mod "
        if command in ("pmod", "pod"):
            value = self._parse_required_argument().strip()
            return f" (mod {value})" if command == "pmod" else f" ({value})"
        if command in ("overset", "stackrel"):
            upper = self._parse_required_argument()
            value = self._parse_required_argument().strip()
            return f"{value}{_format_script(upper, 'sup')}"
        if command == "underset":
            lower = self._parse_required_argument()
            value = self._parse_required_argument().strip()
            return f"{value}{_format_script(lower, 'sub')}"
        if command in PLAIN_WRAPPERS:
            value = self._parse_required_argument()
            return value if command.startswith("text") or command == "mbox" else value.strip()
        if command == "begin":
            return self._parse_environment()
        if command == "end":
            self._supported = False
            return ""

        self._supported = False
        return f"\\{command}"

    def _parse_operator(
        self, operator: str, inline_lower_style: str, display_limits: bool, spaced: bool = False
    ) -> str:
        use_display_limits = display_limits
        modifier_position = self._position
        while modifier_position < len(self._source) and self._source[modifier_position] in (" ", "\t"):
            modifier_position += 1
        modifier = _LIMITS_MODIFIER_RE.match(self._source[modifier_position:])
        if modifier:
            use_display_limits = modifier.group(1) == "limits"
            self._position = modifier_position + len(modifier.group(0))

        lower: str | None = None
        upper: str | None = None
        while True:
            script_position = self._position
            while script_position < len(self._source) and self._source[script_position] in (" ", "\t"):
                script_position += 1
            kind = self._source[script_position : script_position + 1]
            if kind not in ("_", "^"):
                break
            self._position = script_position + 1
            value = _normalize_output(self._parse_required_argument(False)).replace(" ", "")
            if kind == "_":
                if lower is not None:
                    self._supported = False
                lower = value
            else:
                if upper is not None:
                    self._supported = False
                upper = value

        if self._display and use_display_limits and (lower is not None or upper is not None):
            self._layout_nodes.append({"type": "operator", "operator": operator, "lower": lower, "upper": upper})
            return f"{LAYOUT_MARKER_START}{len(self._layout_nodes) - 1}{LAYOUT_MARKER_END}"

        rendered = operator
        if lower is not None:
            rendered += f"[{lower}]" if inline_lower_style == "bracket" else _format_script(lower, "sub")
        if upper is not None:
            rendered += _format_script(upper, "sup")
        return f" {rendered} " if spaced else rendered

    def _parse_required_argument(self, stack_fractions: bool = True) -> str:
        previous_stack_fractions = self._stack_fractions
        self._stack_fractions = previous_stack_fractions and stack_fractions
        value = self._parse_required_argument_value()
        self._stack_fractions = previous_stack_fractions
        return value

    def _parse_required_argument_value(self) -> str:
        while self._position < len(self._source) and self._source[self._position] in (" ", "\t"):
            self._position += 1
        if self._position >= len(self._source):
            self._supported = False
            return ""
        if self._source[self._position] == "{":
            self._position += 1
            return self._parse_sequence("}")
        if self._source[self._position] == "\\":
            return self._parse_command()
        value = self._source[self._position]
        self._position += 1
        return value

    def _parse_optional_argument(self) -> str | None:
        while self._position < len(self._source) and self._source[self._position] in (" ", "\t"):
            self._position += 1
        if self._source[self._position : self._position + 1] != "[":
            return None
        end = self._source.find("]", self._position + 1)
        if end < 0:
            self._supported = False
            return None
        value = self._source[self._position + 1 : end]
        self._position = end + 1
        return self._render_nested(value)

    def _read_raw_group(self) -> str | None:
        while self._position < len(self._source) and self._source[self._position] in (" ", "\t"):
            self._position += 1
        if self._source[self._position : self._position + 1] != "{":
            self._supported = False
            return None

        self._position += 1
        start = self._position
        depth = 1
        while self._position < len(self._source):
            character = self._source[self._position]
            if character == "\\":
                self._position += 2
                continue
            if character == "{":
                depth += 1
            if character == "}":
                depth -= 1
            if depth == 0:
                value = self._source[start : self._position]
                self._position += 1
                return value
            self._position += 1
        self._supported = False
        return None

    def _split_environment_rows(self, body: str) -> list[str]:
        return _ENVIRONMENT_ROW_SPLIT_RE.split(body)

    def _parse_environment(self) -> str:
        environment = self._read_raw_group()
        if not environment:
            return ""
        end_marker = f"\\end{{{environment}}}"
        end = self._source.find(end_marker, self._position)
        if end < 0:
            self._supported = False
            return ""
        body = self._source[self._position : end]
        self._position = end + len(end_marker)

        if environment in ("equation", "equation*", "displaymath"):
            return self._render_nested(body).strip()

        if environment in _ALIGNED_ENVIRONMENTS:
            aligned_at = environment in ("alignedat", "alignat", "alignat*")
            aligned_body = _ENVIRONMENT_ARGUMENT_RE.sub("", body, count=1) if aligned_at else body
            rows = []
            for row in self._split_environment_rows(aligned_body):
                cells = row.split("&")
                if aligned_at:
                    source = " ".join("".join(cells[index * 2 : index * 2 + 2]) for index in range(-(-len(cells) // 2)))
                else:
                    source = "".join(cells)
                rows.append(self._render_nested(source).strip())
            return "\n".join(row for row in rows if row)

        if environment in ("cases", "cases*"):
            rows = [
                [self._render_nested(cell, False).strip() for cell in row.split("&")]
                for row in self._split_environment_rows(body)
            ]
            rows = [row for row in rows if any(row)]
            lines = []
            for index, row in enumerate(rows):
                value = _TRAILING_COMMA_RE.sub("", row[0] if row else "")
                condition = row[1] if len(row) > 1 else ""
                delimiter = "⎧" if index == 0 else "⎩" if index == len(rows) - 1 else "⎨"
                condition_prefix = " " if _CASES_CONDITION_RE.match(condition) else " if "
                suffix = f"{condition_prefix}{condition}" if condition else ""
                lines.append(f"{delimiter} {value}{suffix}")
            return "\n".join(lines)

        if environment in _MATRIX_ENVIRONMENTS:
            matrix_body = _ENVIRONMENT_ARGUMENT_RE.sub("", body, count=1) if environment == "array" else body
            return self._render_matrix(environment, matrix_body)

        self._supported = False
        return body

    def _render_matrix(self, environment: str, body: str) -> str:
        matrix = [
            [self._render_nested(cell, False).strip() for cell in row.split("&")]
            for row in self._split_environment_rows(body)
        ]
        matrix = [row for row in matrix if any(row)]
        column_count = max([0, *[len(row) for row in matrix]])
        column_widths = [
            max([0, *[visible_width(row[column]) if column < len(row) else 0 for row in matrix]])
            for column in range(column_count)
        ]
        rows = []
        for row in matrix:
            cells = []
            for column in range(column_count):
                cell = row[column] if column < len(row) else ""
                cells.append(f"{cell}{PROTECTED_SPACE * max(0, column_widths[column] - visible_width(cell))}")
            rows.append(" │ ".join(cells))

        if environment in ("array", "matrix", "smallmatrix"):
            lines = rows
        else:
            delimiter = _MATRIX_DELIMITERS.get(environment)
            if not delimiter:
                self._supported = False
                return "\n".join(rows)
            lines = []
            for index, row in enumerate(rows):
                left = delimiter[0] if index == 0 else delimiter[4] if index == len(rows) - 1 else delimiter[2]
                right = delimiter[1] if index == 0 else delimiter[5] if index == len(rows) - 1 else delimiter[3]
                lines.append(f"{left} {row} {right}")

        if len(lines) <= 1:
            return lines[0] if lines else ""
        self._layout_nodes.append({"type": "matrix", "lines": lines, "baseline": 0})
        return f"{LAYOUT_MARKER_START}{len(self._layout_nodes) - 1}{LAYOUT_MARKER_END}"

    def _render_nested(self, source: str, stack_fractions: bool = True) -> str:
        rendered = _LatexParser(source, self._layout_nodes, self._display and stack_fractions).render()
        if rendered is None:
            self._supported = False
            return source
        return rendered


def render_latex(source: str, options: dict | None = None) -> str | None:
    """Render a basic LaTeX math expression as terminal-friendly Unicode text.

    Options (pi's ``RenderLatexOptions``): ``display`` stacks fractions and
    operator limits vertically for display math (default False). Returns None
    when the expression contains unsupported or malformed syntax.
    """
    options = options or {}
    layout_nodes: list[dict] = []
    rendered = _LatexParser(source, layout_nodes, options.get("display") is True).render()
    if rendered is None:
        return None
    if not layout_nodes:
        return rendered.replace(PROTECTED_SPACE, " ")
    lines = _render_layout(rendered, layout_nodes)["lines"]
    indentation = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
    return "\n".join(line[indentation:].rstrip() for line in lines).rstrip().replace(PROTECTED_SPACE, " ")
