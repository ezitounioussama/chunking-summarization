"""The three chunking strategies (Steps 1-3), with no model calls.

Kept separate from the summarising so the splitting logic can be tested on its
own and so the difference between the strategies is visible in one file.
"""

import re
from typing import List

TEXT = (
    "Artificial intelligence is profoundly transforming modern businesses. "
    "It makes it possible to automate repetitive tasks, analyze customer behavior, "
    "and predict market trends. However, its integration raises ethical concerns "
    "regarding privacy, transparency, and human employment."
)


# ---------------------------------------------------------------------------
# Step 1 — fixed-size chunking
# ---------------------------------------------------------------------------


def fixed_size_chunks(text: str, size: int = 50) -> List[str]:
    """Cut the text every `size` characters, ignoring meaning entirely.

    This is the simplest possible strategy and the point of including it is to
    show the cost: the cuts land mid-word and mid-phrase, so a chunk can end
    "...raises ethical conc" and lose the idea it was carrying. Cheap, fast,
    reproducible, and semantically blind.
    """
    return [text[i : i + size] for i in range(0, len(text), size)]


# ---------------------------------------------------------------------------
# Step 2 — semantic chunking
# ---------------------------------------------------------------------------


def semantic_chunks(text: str) -> List[str]:
    """Split on sentence boundaries, so each chunk is one complete idea.

    "Semantic" here means the split follows meaning rather than a character
    count. Each sentence in this text carries exactly one idea — what AI does,
    what it enables, what it costs — so sentence boundaries are the natural
    units.

    A regex on `.!?` followed by whitespace is enough for this text. It is not a
    general-purpose sentence splitter ("Dr. Smith" or "3.5" would fool it), and a
    production system would use a proper sentence tokeniser or an embedding-based
    splitter that groups sentences by similarity.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


# ---------------------------------------------------------------------------
# Step 3 — hierarchical / recursive chunking
# ---------------------------------------------------------------------------


def hierarchical_chunks(text: str, max_chars: int = 90) -> List[dict]:
    """Split into paragraphs, then recursively split each paragraph.

    Returns a tree:

        [{"paragraph": str, "children": [str, ...]}, ...]

    The recursion tries the largest natural boundary first and only moves to a
    smaller one when a piece is still too long: paragraphs, then sentences, then
    clauses at commas. That ordering is what keeps the pieces meaningful — unlike
    Step 1, a cut is only made where the language already has a seam.

    This source text is a single paragraph, so the tree has one branch. The
    function still handles several, which is what makes it useful on a real
    document.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]

    tree = []
    for paragraph in paragraphs:
        tree.append({"paragraph": paragraph, "children": _recursive_split(paragraph, max_chars)})

    return tree


def _recursive_split(piece: str, max_chars: int) -> List[str]:
    """Split `piece` until every part is at or under max_chars.

    Separators are tried in order from the strongest boundary to the weakest.
    """
    piece = piece.strip()

    # Base case: short enough already.
    if len(piece) <= max_chars:
        return [piece]

    # Try sentence boundaries first.
    sentences = re.split(r"(?<=[.!?])\s+", piece)
    if len(sentences) > 1:
        parts = []
        for sentence in sentences:
            parts.extend(_recursive_split(sentence, max_chars))
        return parts

    # A single long sentence: fall back to clause boundaries at commas.
    clauses = [clause.strip() for clause in piece.split(",") if clause.strip()]
    if len(clauses) > 1:
        merged = []
        current = ""
        for clause in clauses:
            candidate = f"{current}, {clause}" if current else clause
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    merged.append(current)
                current = clause
        if current:
            merged.append(current)
        return merged

    # No natural seam left. Returning it whole is the honest outcome: better one
    # slightly long meaningful chunk than a cut through the middle of a word.
    return [piece]
