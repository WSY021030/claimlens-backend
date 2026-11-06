"""Conservative model-quote alignment to ORIGINAL text, preserving exact offsets.

The accepted span must still be a unique substring of the supplied source.  The
aligner permits only presentation-equivalent repairs (whitespace, typographic
quotes/dashes and LaTeX-adjacent spacing); it never changes letters, numbers,
word order or case and never performs semantic/fuzzy matching.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QuoteMatch:
    start: int
    end: int
    exact: str
    whitespace_repaired: bool
    formatting_repaired: bool = False


_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff"}
_TRANSLATE = {
    "\u00a0": " ",  # no-break space
    "\u202f": " ",  # narrow no-break space
    "\u2007": " ",  # figure space
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
}
_LATEX_TIGHT = set("{}[]()_^,;:")


def _canonical_with_map(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Return a formatting-only canonical form and source-coordinate map."""
    chars: list[str] = []
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _ZERO_WIDTH:
            i += 1
            continue
        if ch.isspace() or ch in {"\u00a0", "\u202f", "\u2007"}:
            start = i
            i += 1
            while i < len(text) and (text[i].isspace() or text[i] in {"\u00a0", "\u202f", "\u2007"}):
                i += 1
            chars.append(" ")
            spans.append((start, i))
            continue
        chars.append(_TRANSLATE.get(ch, ch))
        spans.append((i, i + 1))
        i += 1

    # LaTeX and mathematical HTML exports often insert/remove spaces around
    # braces, subscripts and punctuation. Removing only those adjacent spaces
    # cannot alter a word, number or operator token.
    keep = [True] * len(chars)
    for idx, ch in enumerate(chars):
        if ch != " ":
            continue
        prev = chars[idx - 1] if idx else ""
        nxt = chars[idx + 1] if idx + 1 < len(chars) else ""
        if prev in _LATEX_TIGHT or nxt in _LATEX_TIGHT:
            keep[idx] = False
    kept_chars = [ch for ch, yes in zip(chars, keep) if yes]
    kept_spans = [span for span, yes in zip(spans, keep) if yes]
    left = 0
    while left < len(kept_chars) and kept_chars[left] == " ":
        left += 1
    right = len(kept_chars)
    while right > left and kept_chars[right - 1] == " ":
        right -= 1
    return "".join(kept_chars[left:right]), kept_spans[left:right]


def _canonical(text: str) -> str:
    return _canonical_with_map(text)[0]


def align_quote(source_text: str, quoted_text: str) -> tuple[QuoteMatch | None, str | None]:
    """Find one unambiguous source span matching a model-returned quote.

    Exact content wins.  Fallbacks are deliberately limited to whitespace and
    typography/LaTeX presentation equivalence.  Returned text is always copied
    from ``source_text`` and offsets always refer to that original string.
    """
    if not quoted_text or not quoted_text.strip():
        return None, "empty_source_quote"
    quoted = quoted_text.strip()

    exact_hits = list(re.finditer(re.escape(quoted), source_text))
    if len(exact_hits) > 1:
        return None, "ambiguous_quote_location"
    if len(exact_hits) == 1:
        hit = exact_hits[0]
        return QuoteMatch(hit.start(), hit.end(), source_text[hit.start():hit.end()],
                          quoted_text != source_text[hit.start():hit.end()]), None

    # First preserve the older, stricter whitespace-only behavior.
    words = re.findall(r"\S+", quoted)
    if len(words) >= 2:
        pattern = r"\s+".join(re.escape(word) for word in words)
        matches = list(re.finditer(pattern, source_text))
        if len(matches) > 1:
            return None, "ambiguous_quote_location"
        if len(matches) == 1:
            hit = matches[0]
            return QuoteMatch(hit.start(), hit.end(), source_text[hit.start():hit.end()],
                              True, False), None

    # Final safe fallback: unique formatting-equivalent match.  No token,
    # number, punctuation class or case is fuzzily changed.
    source_canonical, source_map = _canonical_with_map(source_text)
    quoted_canonical = _canonical(quoted)
    if not quoted_canonical:
        return None, "empty_source_quote"
    starts = [m.start() for m in re.finditer(re.escape(quoted_canonical), source_canonical)]
    if len(starts) > 1:
        return None, "ambiguous_quote_location"
    if len(starts) == 1:
        start_c = starts[0]
        end_c = start_c + len(quoted_canonical)
        if end_c <= len(source_map):
            start = source_map[start_c][0]
            end = source_map[end_c - 1][1]
            exact = source_text[start:end]
            return QuoteMatch(start, end, exact, True, True), None
    return None, "quote_not_found_verbatim"


def recover_quote_boundary(source_text: str, quoted_text: str) -> tuple[QuoteMatch | None, str | None]:
    """Recover only a *contiguous, exact* excerpt after editorial edge punctuation.

    Model outputs sometimes use ``...`` before a clause or end an excerpt
    with a period when the source continues with a comma/parenthesis.  Remove
    only those presentation markers; never change words, figures, negations or
    the source coordinate space.  A unique match is required.
    """
    quote = quoted_text.strip()
    if not quote:
        return None, "empty_source_quote"
    start_trimmed = re.sub(r"^(?:\.{3}|…)+\s*", "", quote).strip()
    candidates = [start_trimmed]
    if start_trimmed.endswith((".", ";", ",")):
        candidates.append(start_trimmed[:-1].rstrip())
    if start_trimmed.endswith(("...", "…")):
        candidates.append(re.sub(r"(?:\.{3}|…)+$", "", start_trimmed).rstrip())
    for text in dict.fromkeys(candidates):
        # Do not accept trivial snippets detached from a scientific proposition.
        if len(re.findall(r"\w+", text)) < 5:
            continue
        match, _ = align_quote(source_text, text)
        if match is not None:
            return match, "quote_boundary_recovered"
    return None, "quote_not_found_verbatim"


def recover_numeric_sentence(source_text: str, quoted_text: str) -> tuple[QuoteMatch | None, str | None]:
    """Recover a source sentence only for an unambiguous multi-number result.

    This is intentionally NOT fuzzy quote acceptance: the returned Claim text
    is the exact source sentence and the numerical signature must match.  It
    handles a paraphrased reporting frame while keeping all published values.
    """
    numbers = re.findall(r"(?<!\w)[+-]?\d+(?:\.\d+)*(?:%)?", quoted_text)
    if len(set(numbers)) < 2:
        return None, "no_unique_numeric_signature"
    signatures = set(numbers)
    matches: list[QuoteMatch] = []
    # Preserve dots inside decimals and appendix designators. A simple
    # ``[^.]`` regex would silently drop the very figures we are recovering.
    spans = []
    start = 0
    for i, ch in enumerate(source_text):
        if ch == "\n" or (ch in ".!?" and not
                         (ch == "." and i + 1 < len(source_text)
                          and source_text[i + 1].isdigit())):
            spans.append((start, i + 1))
            start = i + 1
    if start < len(source_text):
        spans.append((start, len(source_text)))
    for part_start, part_end in spans:
        exact = source_text[part_start:part_end]
        if not signatures.issubset(set(re.findall(r"(?<!\w)[+-]?\d+(?:\.\d+)*(?:%)?", exact))):
            continue
        # Two matching numbers alone could be an unrelated table listing.
        # Require shared substantive words as an additional content check.
        words = lambda s: {w for w in re.findall(r"[A-Za-z]{4,}", s.lower())
                           if w not in {"this", "that", "with", "when", "which", "from",
                                        "there", "these", "those", "have", "were", "also"}}
        model_words = words(quoted_text)
        if len(model_words & words(exact)) < 2:
            continue
        lead = len(exact) - len(exact.lstrip())
        tail = len(exact.rstrip())
        start, end = part_start + lead, part_start + tail
        if start < end:
            matches.append(QuoteMatch(start, end, source_text[start:end], True))
    if len(matches) == 1:
        return matches[0], "numeric_sentence_recovered"
    return None, "numeric_sentence_ambiguous_or_missing"
