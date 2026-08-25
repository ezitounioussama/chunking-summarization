"""The five chunking strategies from the course, with spaCy for sentences.

    Method 1  fixed-size, with overlap
    Method 2  sentence-based        (spaCy, regex fallback)
    Method 3  recursive             (LangChain's default approach)
    Method 4  semantic              (embeddings + KMeans clustering)
    Method 5  document-structure    (headings, lists, prose treated differently)

Every function returns a list of Chunk objects so the strategies can be compared
on equal terms downstream.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from docx_loader import Document

# ---------------------------------------------------------------------------
# spaCy, loaded once
# ---------------------------------------------------------------------------

_NLP = None
_SPACY_AVAILABLE = None


def get_spacy():
    """Load en_core_web_sm once and reuse it.

    Loading a spaCy model costs a second or two, so it is cached at module level
    rather than reloaded per call.

    Only the parser's sentence boundaries are needed, so the tagger, lemmatizer
    and NER are disabled — they are the slow components and nothing here uses
    them.
    """
    global _NLP, _SPACY_AVAILABLE

    if _SPACY_AVAILABLE is False:
        return None
    if _NLP is not None:
        return _NLP

    try:
        import spacy

        _NLP = spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer", "ner", "attribute_ruler"])
        # Long documents exceed the default 1,000,000-char limit only rarely, but
        # raising it is free and avoids a surprising crash on a big file.
        _NLP.max_length = 3_000_000
        _SPACY_AVAILABLE = True
        return _NLP

    except (ImportError, OSError):
        # ImportError: spacy not installed. OSError: model not downloaded.
        _SPACY_AVAILABLE = False
        return None


def spacy_ready() -> bool:
    return get_spacy() is not None


# ---------------------------------------------------------------------------
# Chunk record
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """One chunk, plus where it came from."""

    text: str
    method: str
    index: int = 0
    section: str = ""              # the heading it sits under, when known
    kind: str = "prose"            # prose | heading | list | mixed
    parent: Optional[str] = None   # for hierarchical output
    meta: dict = field(default_factory=dict)

    @property
    def chars(self) -> int:
        return len(self.text)


def _finalise(texts, method: str, **common) -> List[Chunk]:
    """Wrap raw strings as numbered Chunks, dropping empties."""
    return [
        Chunk(text=text.strip(), method=method, index=i, **common)
        for i, text in enumerate((t for t in texts if t and t.strip()), start=1)
    ]


# ---------------------------------------------------------------------------
# Method 1 — fixed size, with overlap
# ---------------------------------------------------------------------------


def fixed_size(text: str, size: int = 500, overlap: int = 60) -> List[Chunk]:
    """Cut every `size` characters, keeping `overlap` characters of context.

    The overlap is the course's fix for the method's core weakness: a hard cut
    can land in the middle of an idea, so each chunk repeats the tail of the
    previous one. An idea straddling a boundary then survives whole in at least
    one chunk.

    The course suggests 10-15% of the chunk size, so 500/60 sits in that range.

    The step is `size - overlap`. If overlap were ever >= size the window would
    stop advancing and the loop would never end, so that is rejected outright.
    """
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    pieces = []
    start = 0
    while start < len(text):
        pieces.append(text[start : start + size])
        start += size - overlap

    return _finalise(pieces, "fixed-size", meta={"size": size, "overlap": overlap})


# ---------------------------------------------------------------------------
# Method 2 — sentence-based, spaCy preferred
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> List[str]:
    """Sentence boundaries, via spaCy when available.

    Why spaCy beats the regex: `re.split(r"(?<=[.!?])\\s+", ...)` treats every
    full stop as a sentence end, so it breaks on abbreviations and decimals.
    Measured on "Mr. Smith has arrived. He met Dr. Jones at 3.5 p.m." the regex
    produces 4 sentences ("Mr." / "Smith has arrived." / "He met Dr." / "Jones
    at 3.5 p.m.") while spaCy produces the correct 2.

    spaCy gets this right because its parser decides boundaries from the
    grammatical structure and a learned model, not from punctuation alone.
    """
    nlp = get_spacy()

    if nlp is None:
        # Fallback so the module still works without spaCy installed.
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    return [sent.text.strip() for sent in nlp(text).sents if sent.text.strip()]


def sentence_based(text: str, max_chars: int = 600) -> List[Chunk]:
    """Group whole sentences into chunks up to `max_chars`.

    Sentences are never split. Packing several into one chunk keeps chunks a
    useful size — one short sentence on its own carries too little context to
    retrieve well.
    """
    sentences = split_sentences(text)

    chunks, current = [], ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    engine = "spaCy" if spacy_ready() else "regex-fallback"
    return _finalise(chunks, "sentence", meta={"engine": engine, "max_chars": max_chars})


# ---------------------------------------------------------------------------
# Method 3 — recursive
# ---------------------------------------------------------------------------


def recursive(text: str, max_chars: int = 600,
              separators: Optional[List[str]] = None) -> List[Chunk]:
    """Split on the strongest separator that works, recursing when still too big.

    Order matters: paragraph break, then sentence boundary, then newline, then
    space. Each step gives up a little more meaning, so the coarsest separator
    that gets a piece under the limit is the one used. This is the idea behind
    LangChain's RecursiveCharacterTextSplitter, which the course recommends as
    the default starting point.

    The sentence level uses spaCy rather than a "." split, which is the
    improvement over the version in the course notes.
    """
    if separators is None:
        separators = ["\n\n", "SENTENCE", "\n", " "]

    pieces = _recurse(text, max_chars, separators)
    return _finalise(pieces, "recursive", meta={"max_chars": max_chars})


def _recurse(piece: str, max_chars: int, separators: List[str]) -> List[str]:
    piece = piece.strip()

    if len(piece) <= max_chars:
        return [piece] if piece else []

    if not separators:
        # Every natural boundary is exhausted; a hard cut is the only option
        # left. Overlap is applied so the forced break loses less.
        return [piece[i : i + max_chars] for i in range(0, len(piece), max_chars)]

    separator, rest = separators[0], separators[1:]

    if separator == "SENTENCE":
        parts = split_sentences(piece)
    else:
        parts = piece.split(separator)

    # This separator does not occur, so move straight to the next one.
    if len(parts) <= 1:
        return _recurse(piece, max_chars, rest)

    # Merge the parts back up to the limit, so the result is not needlessly
    # fragmented — splitting into 40 tiny pieces and stopping there would be
    # worse than a few well-filled chunks.
    joiner = "" if separator == "SENTENCE" else separator
    merged, current = [], ""

    for part in parts:
        candidate = f"{current}{joiner if joiner else ' '}{part}" if current else part

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                merged.append(current)
            # A single part can still be over the limit on its own.
            if len(part) > max_chars:
                merged.extend(_recurse(part, max_chars, rest))
                current = ""
            else:
                current = part

    if current:
        merged.append(current)

    return [m.strip() for m in merged if m.strip()]


# ---------------------------------------------------------------------------
# Method 4 — semantic (embeddings + KMeans)
# ---------------------------------------------------------------------------


def semantic(text: str, embed_function: Callable, n_clusters: int = 8,
             max_sentences: int = 400) -> List[Chunk]:
    """Group sentences by MEANING, not by position.

    This is the course's Method 4, and it is genuinely different from the other
    four: they all cut the text along its existing order, while this one embeds
    each sentence and clusters the vectors with KMeans. Two sentences about the
    same topic land in the same chunk even if they sit pages apart.

    The cost is one embedding per sentence before any chunking happens, which is
    why the course flags it as the expensive option.

    Sentences within a cluster are kept in document order, so each chunk still
    reads sensibly.
    """
    from sklearn.cluster import KMeans

    sentences = split_sentences(text)

    # Very short fragments (list bullets, stray labels) add noise to the
    # clustering without carrying a topic, so they are left out of it.
    usable = [s for s in sentences if len(s) > 25][:max_sentences]

    if len(usable) <= n_clusters:
        return _finalise(usable, "semantic", meta={"note": "too few sentences to cluster"})

    vectors = embed_function(usable)

    # n_init=10 runs the clustering ten times from different starts and keeps the
    # best; KMeans can otherwise settle on a poor grouping by luck of the draw.
    # random_state fixes the seed so a rerun gives the same clusters.
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(vectors)

    grouped: dict = {}
    for position, (sentence, label) in enumerate(zip(usable, labels)):
        grouped.setdefault(int(label), []).append((position, sentence))

    # Order the clusters by where they first appear, so the output roughly
    # follows the document instead of KMeans' arbitrary label numbers.
    ordered = sorted(grouped.items(), key=lambda item: min(p for p, _ in item[1]))

    chunks = []
    for label, members in ordered:
        members.sort(key=lambda item: item[0])
        chunks.append(" ".join(sentence for _, sentence in members))

    return _finalise(
        chunks,
        "semantic",
        meta={"n_clusters": n_clusters, "sentences_clustered": len(usable)},
    )


# ---------------------------------------------------------------------------
# Method 5 — document-structure aware
# ---------------------------------------------------------------------------


def structure_aware(document: Document, max_chars: int = 700) -> List[Chunk]:
    """Chunk by the document's own outline, treating each content type differently.

    This is the method that needs a real document rather than a plain string, and
    it is the reason the .docx is parsed instead of flattened to text:

      * headings become the chunk's section label, not chunks of their own —
        "Method 3 — Recursive Chunking" retrieves nothing useful alone
      * consecutive list items are kept together, because one bullet out of a
        list of eight is meaningless
      * prose paragraphs are packed up to the size limit
      * the heading is prepended to each chunk, so a chunk retrieved on its own
        still says what it is about

    That last point is what makes structure-aware chunking pay for itself in
    search: a fixed-size chunk from the middle of a section has no idea which
    section it came from.
    """
    chunks: List[Chunk] = []

    for section in document.sections():
        title = section["title"]
        prefix = f"[{title}] " if title != "(front matter)" else ""

        buffer, buffer_kind = "", "prose"

        def flush(buf, kind):
            if buf.strip():
                chunks.append(
                    Chunk(
                        text=f"{prefix}{buf.strip()}",
                        method="structure",
                        section=title,
                        kind=kind,
                    )
                )

        for block in section["blocks"]:
            kind = "list" if block.is_list_item else "prose"

            # A change of content type closes the current chunk: mixing eight
            # bullets into a prose paragraph loses the fact that it was a list.
            if kind != buffer_kind and buffer:
                flush(buffer, buffer_kind)
                buffer = ""

            buffer_kind = kind
            candidate = f"{buffer}\n{block.text}".strip() if buffer else block.text

            if len(candidate) + len(prefix) <= max_chars:
                buffer = candidate
            else:
                flush(buffer, buffer_kind)
                # A single block longer than the limit is split recursively
                # rather than cut blindly.
                if len(block.text) + len(prefix) > max_chars:
                    for piece in recursive(block.text, max_chars - len(prefix)):
                        chunks.append(
                            Chunk(text=f"{prefix}{piece.text}", method="structure",
                                  section=title, kind=kind)
                        )
                    buffer = ""
                else:
                    buffer = block.text

        flush(buffer, buffer_kind)

    for position, chunk in enumerate(chunks, start=1):
        chunk.index = position

    return chunks
