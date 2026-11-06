"""Offline, script-free SVG rendering for supported LaTeX math expressions.

Matplotlib mathtext intentionally supports only a SUBSET of full TeX.
When rendering fails, callers should retain and display the source latex.
The original equation in paper.json is never replaced with a picture.
"""
from __future__ import annotations

import re
from pathlib import Path


def unwrap_display_latex(source: str) -> str:
    """Remove an outer display math environment when present."""
    text = source.strip()
    m = re.fullmatch(
        r"\\begin\{(?P<env>equation\*?|displaymath|math)\}\s*"
        r"(?P<body>.*?)\s*\\end\{(?P=env)\}",
        text, flags=re.DOTALL,
    )
    return m.group('body').strip() if m else text


def render_math_svg(source_latex: str, destination: Path) -> None:
    """Render supported TeX math to trusted local SVG, without browser JS.

    Raises ValueError on bad/unsupported TeX. Caller keeps a readable fallback.
    """
    from matplotlib.figure import Figure
    from matplotlib.mathtext import MathTextParser

    body = unwrap_display_latex(source_latex)
    if not body:
        raise ValueError('Empty LaTeX expression')
    # Bounds to avoid untrusted HTML producing absurdly large files.
    if len(body) > 2000:
        raise ValueError('Equation exceeds offline renderer size limit')
    math = f'${body}$'
    # Pre-validate rather than silently emitting blank SVG.
    MathTextParser('path').parse(math, dpi=140)
    width = max(6, min(28, 1.0 + len(body) * 0.095))
    fig = Figure(figsize=(width, 0.95), dpi=140)
    fig.patch.set_alpha(0)
    fig.text(0.5, 0.5, math, fontsize=16, ha='center', va='center', color='#17212f')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(destination), format='svg', bbox_inches='tight', pad_inches=0.12)