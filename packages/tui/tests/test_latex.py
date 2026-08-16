"""Mirror of pi tui test/latex.test.ts (case tables generated from the TS)."""

import pytest

from pidrei_tui.latex import render_latex


LATEX_CASES = [
    ("\\mathbb{C}^3 \\to \\mathbb{C}^3", "ℂ³ → ℂ³"),
    (
        "\\{3x+2y,\\; 27x^2-4z-1,\\; x(x-1)(x+1)\\} \\quad\\Rightarrow\\quad x \\in \\{0, \\pm 1\\},",
        "{3x+2y, 27x²-4z-1, x(x-1)(x+1)} ⇒ x ∈ {0, ± 1},",
    ),
    ("F_1 = -\\frac{1}{4x^2}.", "F₁ = -1/(4x²)."),
    ("-2", "-2"),
    ("(0,0,-1/4)", "(0,0,-1/4)"),
    ("(1,-3/2,13/2)", "(1,-3/2,13/2)"),
    ("(1,1,1)", "(1,1,1)"),
    ("(2,1,0)", "(2,1,0)"),
    ("(-1/4, 0, 0)", "(-1/4, 0, 0)"),
    ("\\{(0,0,-1/4), (1,-3/2,13/2), (-1,3/2,13/2)\\}", "{(0,0,-1/4), (1,-3/2,13/2), (-1,3/2,13/2)}"),
    ("(2,1,1)", "(2,1,1)"),
    ("(7/3,-2/5,11/7)", "(7/3,-2/5,11/7)"),
    ("\\{y - p(x),\\; q(x)\\}", "{y - p(x), q(x)}"),
    ("\\deg q = 3", "deg q = 3"),
    ("[\\mathbb{C}(x,y,z):\\mathbb{C}(F_1,F_2,F_3)] = 3", "[ℂ(x,y,z):ℂ(F₁,F₂,F₃)] = 3"),
    ("u = 1+xy", "u = 1+xy"),
    ("G = u^2 z + y^2(4+3xy)", "G = u² z + y²(4+3xy)"),
    ("F_1 = uG", "F₁ = uG"),
    ("F_2 = y + 3xG", "F₂ = y + 3xG"),
    ("x=0", "x = 0"),
    ("F_2 = F_3 = 0", "F₂ = F₃ = 0"),
    ("xy = -3/2", "xy = -3/2"),
    ("x^2 z = 13/2", "x² z = 13/2"),
    ("\\mathbb{C}^*", "ℂ^*"),
    ("s \\mapsto (s,\\, -\\tfrac{3}{2s},\\, \\tfrac{13}{2s^2})", "s ↦ (s, -3/(2s), 13/(2s²))"),
    ("X", "X"),
    ("p_\\pm", "p_±"),
    ("F(-x,-y,z) = (F_1, -F_2, -F_3)", "F(-x,-y,z) = (F₁, -F₂, -F₃)"),
    ("p_0", "p₀"),
    ("s \\to \\infty", "s → ∞"),
    ("(0,0,0)", "(0,0,0)"),
    ("\\Rightarrow", "⇒"),
    ("\\ge 2", "≥ 2"),
    ("\\ge 3", "≥ 3"),
    ("1", "1"),
    ("\\mathrm{diag}(-1/2,1,1)", "diag(-1/2,1,1)"),
    ("4+3xy", "4+3xy"),
    (
        "E \\approx \\frac{0.1\\ \\text{lux}}{100\\ \\text{lm/W}} = 0.001\\ \\text{W/m}^2",
        "E ≈ (0.1 lux)/(100 lm/W) = 0.001 W/m²",
    ),
    ("\\boxed{1\\ \\text{milliwatt per square metre}}", "[1 milliwatt per square metre]"),
    ("5\\ \\text{km}^2 = 5{,}000{,}000\\ \\text{m}^2", "5 km² = 5,000,000 m²"),
    (
        "P_{\\text{light}} = 0.001 \\times 5{,}000{,}000\n= \\boxed{5{,}000\\ \\text{W}}",
        "P_light = 0.001 × 5,000,000 = [5,000 W]",
    ),
    (
        "P_{\\text{electric}} = 5\\ \\text{kW} \\times 0.2\n= \\boxed{1\\ \\text{kW}}",
        "P_electric = 5 kW × 0.2 = [1 kW]",
    ),
    ("\\pi(2.5\\ \\text{km})^2 = 19.6\\ \\text{km}^2", "π(2.5 km)² = 19.6 km²"),
    (
        "0.001\\ \\text{W/m}^2 \\times 19.6 \\times 10^6\\ \\text{m}^2\n\\approx \\boxed{20\\ \\text{kW optical}}",
        "0.001 W/m² × 19.6 × 10⁶ m² ≈ [20 kW optical]",
    ),
    (
        "1\\ \\text{kW} \\times \\frac{1}{3600}\\ \\text{hour}\n= \\boxed{0.28\\ \\text{Wh}}",
        "1 kW × 1/3600 hour = [0.28 Wh]",
    ),
    (
        "\\det\\!\\left(\\frac{\\partial(F_1,F_2,F_3)}{\\partial(x,y,z)}\\right)=-2.",
        "det((∂(F₁,F₂,F₃))/(∂(x,y,z))) = -2.",
    ),
    (
        "\\begin{aligned}\nF(0,0,-\\tfrac14)&=(-\\tfrac14,0,0),\\\\\nF(1,-\\tfrac32,\\tfrac{13}2)&=(-\\tfrac14,0,0),\\\\\nF(-1,\\tfrac32,\\tfrac{13}2)&=(-\\tfrac14,0,0).\n\\end{aligned}",
        "F(0,0,-1/4) = (-1/4,0,0),\nF(1,-3/2,13/2) = (-1/4,0,0),\nF(-1,3/2,13/2) = (-1/4,0,0).",
    ),
    ("F=(F_1,F_2,F_3)", "F = (F₁,F₂,F₃)"),
    ("F", "F"),
    ("3", "3"),
    (
        "J = \\begin{pmatrix}\n\\frac{\\partial f_1}{\\partial x} & \\frac{\\partial f_1}{\\partial y} & \\frac{\\partial f_1}{\\partial z} \\\\\n\\frac{\\partial f_2}{\\partial x} & \\frac{\\partial f_2}{\\partial y} & \\frac{\\partial f_2}{\\partial z} \\\\\n\\frac{\\partial f_3}{\\partial x} & \\frac{\\partial f_3}{\\partial y} & \\frac{\\partial f_3}{\\partial z}\n\\end{pmatrix}",
        "J = ⎛ (∂ f₁)/(∂ x) │ (∂ f₁)/(∂ y) │ (∂ f₁)/(∂ z) ⎞\n    ⎜ (∂ f₂)/(∂ x) │ (∂ f₂)/(∂ y) │ (∂ f₂)/(∂ z) ⎟\n    ⎝ (∂ f₃)/(∂ x) │ (∂ f₃)/(∂ y) │ (∂ f₃)/(∂ z) ⎠",
    ),
    (
        "\\begin{aligned}\nf_1 &= (1+xy)^3 z + y^2(1+xy)(4+3xy) \\\\\nf_2 &= y + 3x(1+xy)^2 z + 3xy^2(4+3xy) \\\\\nf_3 &= 2x - 3x^2y - x^3z\n\\end{aligned}",
        "f₁ = (1+xy)³ z + y²(1+xy)(4+3xy)\nf₂ = y + 3x(1+xy)² z + 3xy²(4+3xy)\nf₃ = 2x - 3x²y - x³z",
    ),
    ("x, y, z", "x, y, z"),
    ("(x, y, z)", "(x, y, z)"),
    ("(0,\\; 0,\\; -\\tfrac14)", "(0, 0, -1/4)"),
    ("(-\\tfrac14,\\; 0,\\; 0)", "(-1/4, 0, 0)"),
    ("(1,\\; -\\tfrac32,\\; \\tfrac{13}{2})", "(1, -3/2, 13/2)"),
    ("(-1,\\; \\tfrac32,\\; \\tfrac{13}{2})", "(-1, 3/2, 13/2)"),
    ("(-\\frac14, 0, 0)", "(-1/4, 0, 0)"),
    ("F: \\mathbb{C}^3 \\to \\mathbb{C}^3", "F: ℂ³ → ℂ³"),
    (
        "F(0,0,-\\tfrac14) = F(1,-\\tfrac32,\\tfrac{13}{2}) = F(-1,\\tfrac32,\\tfrac{13}{2}) = (-\\tfrac14, 0, 0)",
        "F(0,0,-1/4) = F(1,-3/2,13/2) = F(-1,3/2,13/2) = (-1/4, 0, 0)",
    ),
    ("\\mathbb{C}^3", "ℂ³"),
    (
        "\\begin{aligned}\nf_1 &= \\frac{f_1^{\\text{ut}}(u,t)}{x^2}, \\quad\nf_2 = \\frac{f_2^{\\text{ut}}(u,t)}{x}, \\quad\nf_3 = x\\,(2 - 3u - t)\n\\end{aligned}",
        "f₁ = (f₁ᵘᵗ(u,t))/(x²), f₂ = (f₂ᵘᵗ(u,t))/x, f₃ = x (2 - 3u - t)",
    ),
    ("\\det J_F", "det J_F"),
    ("(-\\tfrac14, 0, 0)", "(-1/4, 0, 0)"),
    ("u = xy", "u = xy"),
    ("t = x^2z", "t = x²z"),
    ("x \\neq 0", "x ≠ 0"),
    ("f_1^{\\text{ut}}, f_2^{\\text{ut}}", "f₁ᵘᵗ, f₂ᵘᵗ"),
    ("u,t", "u,t"),
    ("x", "x"),
    ("x, x^2", "x, x²"),
    ("\\mathbb{C}^n \\to \\mathbb{C}^n", "ℂⁿ → ℂⁿ"),
    ("n \\geq 2", "n ≥ 2"),
    ("\\mathbb{P}^3", "ℙ³"),
    ("e^{i\\pi}+1=0", "e^(iπ)+1 = 0"),
    (
        "\\boxed{\n\\mathcal{Z}(\\beta)\n=\n\\int_{\\mathcal M}\n\\exp\\!\\left(\n-\\beta\\left[\n\\frac12 g^{ij}(x)\\,\\partial_i\\phi\\,\\partial_j\\phi\n+V(\\phi)\n\\right]\\right)\n\\mathcal D\\phi\n}",
        "[Z(β) = ∫_M exp( -β[ 1/2 gⁱʲ(x) ∂ᵢϕ ∂ⱼϕ +V(ϕ) ]) Dϕ]",
    ),
    (
        "\\begin{aligned}\n\\nabla_\\mu T^{\\mu\\nu}\n&=\n\\frac{1}{\\sqrt{-g}}\n\\partial_\\mu\\!\\left(\\sqrt{-g}\\,T^{\\mu\\nu}\\right)\n+\\Gamma^\\nu_{\\mu\\lambda}T^{\\mu\\lambda}\n=0, \\\\[4pt]\nR_{\\mu\\nu}-\\frac12 Rg_{\\mu\\nu}+\\Lambda g_{\\mu\\nu}\n&=\n\\frac{8\\pi G}{c^4}T_{\\mu\\nu}.\n\\end{aligned}",
        "∇_μ T^(μν) = 1/(√(-g)) ∂_μ(√(-g) T^(μν)) +Γ^ν_(μλ)T^(μλ) = 0,\nR_(μν)-1/2 Rg_(μν)+Λ g_(μν) = (8π G)/(c⁴)T_(μν).",
    ),
    (
        "f(z)\n=\n\\frac{1}{2\\pi i}\n\\oint_{\\gamma}\n\\frac{f(\\zeta)}{\\zeta-z}\\,d\\zeta,\n\\qquad\n\\det\\!\\begin{pmatrix}\n\\lambda-a & -b & 0\\\\\n-c & \\lambda-d & -e\\\\\n0 & -f & \\lambda-g\n\\end{pmatrix}\n=0.",
        "f(z) = 1/(2π i) ∮_γ (f(ζ))/(ζ-z) dζ, det⎛ λ-a │ -b  │ 0   ⎞ = 0.\n                                        ⎜ -c  │ λ-d │ -e  ⎟\n                                        ⎝ 0   │ -f  │ λ-g ⎠",
    ),
    (
        "\\Psi(x,t)=\n\\sum_{n=1}^{\\infty}\n\\underbrace{\nc_n\n\\sqrt{\\frac{2}{L}}\n\\sin\\!\\left(\\frac{n\\pi x}{L}\\right)\n}_{\\text{spatial eigenmode}}\n\\exp\\!\\left(-\\frac{i\\hbar n^2\\pi^2}{2mL^2}t\\right),\n\\qquad\n|\\Psi(x,t)|^2\n=\n\\begin{cases}\n\\Psi^\\ast\\Psi, & 0<x<L,\\\\\n0, & \\text{otherwise}.\n\\end{cases}",
        "Ψ(x,t) = ∑ₙ₌₁^∞ cₙ √(2/L) sin((nπ x)/L)_(spatial eigenmode) exp(-(iℏ n²π²)/(2mL²)t), |Ψ(x,t)|² = ⎧ Ψ^∗Ψ if 0 < x < L,\n⎩ 0 otherwise.",
    ),
    ("x=\\frac{-b\\pm\\sqrt{b^2-4ac}}{2a}", "x = (-b±√(b²-4ac))/(2a)"),
    ("\\int_0^\\infty e^{-x^2}\\,dx=\\frac{\\sqrt{\\pi}}{2}", "∫₀^∞ e^(-x²) dx = (√π)/2"),
    ("e^{i\\theta}=\\cos\\theta+i\\sin\\theta", "e^(iθ) = cos θ+i sin θ"),
    ("\\sum_{n=1}^{\\infty}\\frac{1}{n^2}=\\frac{\\pi^2}{6}", "∑ₙ₌₁^∞1/(n²) = π²/6"),
    ("\\lim_{x\\to 0}\\frac{\\sin x}{x}=1", "lim[x→0] (sin x)/x = 1"),
    ("\\lim_{n\\to\\infty}\n\\left(1+\\frac{1}{n}\\right)^n=e", "lim[n→∞] (1+1/n)ⁿ = e"),
    ("\\int_0^1 \\frac{x^2}{1+x^3}\\,dx\n=\\frac{1}{3}\\ln 2", "∫₀¹ x²/(1+x³) dx = 1/3 ln 2"),
    ("\\sum_{k=1}^{n}\\frac{k}{k+1}\n=n+1-H_{n+1}", "∑ₖ₌₁ⁿk/(k+1) = n+1-Hₙ₊₁"),
    (
        "\\frac{\n  \\displaystyle \\frac{x^2+1}{x-1}\n  -\n  \\displaystyle \\frac{2x}{x+1}\n}{\n  \\displaystyle \\frac{x}{x^2-1}\n}",
        "((x²+1)/(x-1) - 2x/(x+1))/(x/(x²-1))",
    ),
    (
        "\\lim_{x\\to 0}\n\\frac{\n  \\displaystyle \\frac{\\sin x}{x}-1\n}{\n  \\displaystyle \\frac{e^x-1}{x}-1\n}\n=0",
        "lim[x→0] ((sin x)/x-1)/((eˣ-1)/x-1) = 0",
    ),
    (
        "\\frac{\n  1+\\displaystyle\\frac{1}{1+\\frac{1}{x}}\n}{\n  1-\\displaystyle\\frac{1}{1-\\frac{1}{x}}\n}",
        "(1+1/(1+1/x))/(1-1/(1-1/x))",
    ),
    (
        "\\sum_{n=1}^{\\infty}\n\\frac{\n  \\displaystyle \\frac{1}{n}-\\frac{1}{n+1}\n}{\n  \\displaystyle 1+\\frac{1}{n^2}\n}",
        "∑ₙ₌₁^∞ (1/n-1/(n+1))/(1+1/(n²))",
    ),
]


@pytest.mark.parametrize(("source", "expected"), LATEX_CASES)
def test_renders_latex_case(source, expected):
    assert render_latex(source) == expected


LATEX_ASSERT_CASES = [
    (
        "\\sum_{i=0}^n \\alpha_i + \\int_0^\\infty e^{-x^2}\\,dx = \\sqrt{\\pi}",
        False,
        "∑ᵢ₌₀ⁿ αᵢ + ∫₀^∞ e^(-x²) dx = √π",
    ),
    ("\\binom{n}{k}+\\vec{x}+\\hat{y}+\\overline{AB}", False, "(n choose k)+x⃗+ŷ+overline(AB)"),
    ("\\epsilon+\\varepsilon+\\varsigma+\\varkappa+\\oplus+\\otimes+\\therefore+\\because", False, "ϵ+ε+ς+ϰ+⊕+⊗+∴+∵"),
    ("A\\not\\subseteq B,\\quad x\\not\\in X", False, "A ⊈ B, x ∉ X"),
    ("\\lvert{x}\\rvert+\\lVert{v}\\rVert+\\left.\\frac{dy}{dx}\\right|_{x=0}", False, "|x|+‖v‖+dy/(dx)|ₓ₌₀"),
    ("\\left\\lbrace x \\middle| x>0 \\right\\rbrace", False, "{ x | x > 0 }"),
    ("\\operatorname*{arg\\,max}_{x\\in X} f(x)", False, "arg max[x∈X] f(x)"),
    ("a\\bmod n,\\quad a\\equiv b\\pmod n", False, "a mod n, a ≡ b (mod n)"),
    ("\\overset{!}{=}+\\underset{n}{x}+\\stackrel{def}{=}", False, "=^!+xₙ+=ᵈᵉᶠ"),
    ("\\sqrt[2]{x}+\\sqrt[3]{x}+\\sqrt[4]{x}+\\sqrt[n]{x}+\\sqrt[k]{x+1}", False, "√x+∛x+∜x+ⁿ√x+ᵏ√(x+1)"),
    ("\\acute{x}+\\grave{y}+\\widehat{xyz}+\\overrightarrow{AB}", False, "x́+ỳ+widehat(xyz)+overrightarrow(AB)"),
    ("\\textnormal{hello}+\\mbox{world}+\\boldsymbol{x}", False, "hello+world+x"),
    ("\\begin{equation}\\begin{split}a&=b\\\\&=c\\end{split}\\end{equation}", False, "a = b\n= c"),
    ("\\begin{alignedat}{2}a&=b&\\quad c&=d\\\\e&=f&g&=h\\end{alignedat}", False, "a = b c = d\ne = f g = h"),
    (
        "\\begin{cases}a & x<0 \\\\ b & \\text{if }x=0 \\\\ c & \\text{otherwise}\\end{cases}",
        False,
        "⎧ a if x < 0\n⎨ b if x = 0\n⎩ c otherwise",
    ),
    ("\\begin{pmatrix}1&200\\\\3000&4\\end{pmatrix}", False, "⎛ 1    │ 200 ⎞\n⎝ 3000 │ 4   ⎠"),
    (
        "R\\left(\\frac{\\pi}{4}\\right)\n=\n\\begin{pmatrix}\n\\frac{\\sqrt{2}}{2} & -\\frac{\\sqrt{2}}{2}\\\\\n\\frac{\\sqrt{2}}{2} & \\frac{\\sqrt{2}}{2}\n\\end{pmatrix}.",
        True,
        "   π\nR( ─ ) = ⎛ (√2)/2 │ -(√2)/2 ⎞\n   4     ⎝ (√2)/2 │ (√2)/2  ⎠.",
    ),
    (
        "\\mathbf w\n=\nR\\left(\\frac{\\pi}{4}\\right)\n\\begin{pmatrix}1\\\\0\\end{pmatrix}\n=\n\\begin{pmatrix}\\frac{\\sqrt{2}}{2}\\\\\\frac{\\sqrt{2}}{2}\\end{pmatrix}.",
        True,
        "       π\nw = R( ─ ) ⎛ 1 ⎞ = ⎛ (√2)/2 ⎞\n       4   ⎝ 0 ⎠   ⎝ (√2)/2 ⎠.",
    ),
    (
        "A\\mathbf e_1=\\begin{pmatrix}\\pi\\\\0\\end{pmatrix},\\qquad A\\mathbf e_2=\\begin{pmatrix}0\\\\\\frac{1}{\\pi}\\end{pmatrix}.",
        True,
        "Ae₁ = ⎛ π ⎞, Ae₂ = ⎛ 0   ⎞\n      ⎝ 0 ⎠        ⎝ 1/π ⎠.",
    ),
    (
        "\\sum_{i=0}^n x_i=\\begin{pmatrix}a&b\\\\c&d\\end{pmatrix}.",
        True,
        " n\n ∑  xᵢ = ⎛ a │ b ⎞\ni=0      ⎝ c │ d ⎠.",
    ),
    ("x_{i=0}", False, "xᵢ₌₀"),
    ("x\\neq0", False, "x ≠ 0"),
    ("A\\to B", False, "A → B"),
    ("\\pi\\cdot\\frac{1}{\\pi}", False, "π · 1/π"),
    ("\\sin\\theta", False, "sin θ"),
    ("\\sin^2 x", False, "sin² x"),
    ("-\\sin\\theta", False, "-sin θ"),
    ("i\\sin\\theta", False, "i sin θ"),
    ("\\det(A)", False, "det(A)"),
    # A backslash followed by a line ending is a control space.
    (
        "\\boxed{\n(1,1,1),\\ (1,1,2),\\ (1,2,5),\\ (1,5,13),\\ (2,5,29),\\\n(1,13,34),\\ (1,34,89)\n}.",
        True,
        "[(1,1,1), (1,1,2), (1,2,5), (1,5,13), (2,5,29), (1,13,34), (1,34,89)].",
    ),
    ("a\\\r\nb", False, "a b"),
    ("\\sum_{i=0}^n x_i", True, " n\n ∑  xᵢ\ni=0"),
    ("\\min_{x\\in X} f(x)", True, "min f(x)\nx∈X"),
    ("\\operatorname*{arg\\,max}_{x\\in X} f(x)", True, "arg max f(x)\n  x∈X"),
    ("\\int\\nolimits_0^1 f(x)\\,dx", True, "∫₀¹ f(x) dx"),
    ("\\int\\limits_0^1 f(x)\\,dx", True, "1\n∫ f(x) dx\n0"),
    ("\\begin{cases}a & x<0 \\\\ b & x=0 \\\\ c & x>0\\end{cases}", False, "⎧ a if x < 0\n⎨ b if x = 0\n⎩ c if x > 0"),
    ("x=\\frac{-b\\pm\\sqrt{b^2-4ac}}{2a}", True, "    -b±√(b²-4ac)\nx = ────────────\n         2a"),
    ("\\frac{x^2+1}{x-1}", True, "x²+1\n────\nx-1"),
    ("\\frac{1}\n{2}", True, "1\n─\n2"),
    ("e^{\\frac{1}{2}}", True, "e^(1/2)"),
    ("\\tfrac{1}{2}", True, "1/2"),
    ("x + \\unknown{y}", False, None),
    ("x=y", False, "x = y"),
    ("x =y", False, "x = y"),
    ("x=\ny", False, "x = y"),
    ("x\n=\ny", False, "x = y"),
    (
        "\\frac{\\frac{x^2+1}{x-1}-\\frac{2x}{x+1}}{\\frac{x}{x^2-1}}",
        True,
        "(x²+1)/(x-1)-2x/(x+1)\n─────────────────────\n      x/(x²-1)",
    ),
    (
        "\\lim_{x\\to 0}\\frac{\\frac{\\sin x}{x}-1}{\\frac{e^x-1}{x}-1}=0",
        True,
        "     (sin x)/x-1\nlim  ─────────── = 0\nx→0  (eˣ-1)/x-1",
    ),
    ("\\frac{1+\\frac{1}{1+\\frac{1}{x}}}{1-\\frac{1}{1-\\frac{1}{x}}}", True, "1+1/(1+1/x)\n───────────\n1-1/(1-1/x)"),
    ("\\frac{1}{x", False, None),
    ("x}", False, None),
    ("\\begin{matrix}1 & 2", False, None),
    ("x\\", False, None),
]


@pytest.mark.parametrize(("source", "display", "expected"), LATEX_ASSERT_CASES)
def test_renders_latex_assertion_case(source, display, expected):
    assert render_latex(source, {"display": True} if display else None) == expected
