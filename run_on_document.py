"""
Five chunking strategies on a real .docx, then embeddings and semantic search.

    python run_on_document.py [path/to/document.docx]

Defaults to the NLP course document. Everything runs locally on Ollama.
"""

import sys
import time

# Preflight: this script needs spaCy and scikit-learn, which live in the project
# virtual environment. Running it with the system python gives a bare
# ModuleNotFoundError, so the check below turns that into instructions.
_MISSING = []
for _module, _package in (("spacy", "spacy"), ("sklearn", "scikit-learn"), ("numpy", "numpy")):
    try:
        __import__(_module)
    except ImportError:
        _MISSING.append(_package)

if _MISSING:
    print("Error: missing packages: " + ", ".join(_MISSING))
    print()
    print("These live in the project virtual environment, not in system Python.")
    print("Run the script through the venv instead:")
    print()
    print("    .venv/bin/python run_on_document.py")
    print()
    print("Or activate it first:")
    print()
    print("    source .venv/bin/activate")
    print("    python run_on_document.py")
    print()
    print("No venv yet? Create one (note: Arch has no standalone 'pip' — use python -m pip):")
    print()
    print("    python3 -m venv .venv")
    print("    .venv/bin/python -m pip install spacy scikit-learn numpy")
    print("    .venv/bin/python -m spacy download en_core_web_sm")
    sys.exit(1)

from chunkers import (
    fixed_size,
    recursive,
    semantic,
    sentence_based,
    spacy_ready,
    split_sentences,
    structure_aware,
)
from docx_loader import describe, load_docx
from embeddings import cosine_similarity
from ollama_client import EMBED_MODEL, LLM_MODEL, OllamaError, check_ready, embed, summarise

DEFAULT_DOC = "/home/kirito/Downloads/nlp_chunking_vectorization_EN.docx"

# Questions a student would actually ask this document. Chosen so the answers
# live in different places: one in a recommendations list, one in a comparison
# table, one in a code section.
QUERIES = [
    "Which chunking method should I start with and what overlap is recommended?",
    "What is an embedding and what does cosine similarity measure?",
    "How do I run a model locally with Ollama?",
]

LINE = "=" * 84
THIN = "-" * 84


def header(title):
    print(f"\n{LINE}\n{title}\n{LINE}")


def section(title):
    print(f"\n{title}\n{THIN}")


# ---------------------------------------------------------------------------


def show_spacy_vs_regex():
    """Prove the claim that spaCy beats the regex, rather than asserting it."""
    header("WHY spaCy AND NOT A REGEX")

    import re

    cases = [
        "Mr. Smith has arrived. He met Dr. Jones at 3.5 p.m.",
        "The U.S.A. leads in A.I. research. Costs fell 12.5% in Q3.",
        "See Fig. 2 for details. Approx. 40 items were tested (e.g. sensors).",
    ]

    print("\n  A regex on '[.!?] + whitespace' treats every full stop as a sentence end.")
    print("  spaCy decides from grammatical structure, so abbreviations survive.\n")

    for number, text in enumerate(cases, start=1):
        regex_result = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        spacy_result = split_sentences(text)

        print(f"  CASE {number}: {text}")
        print(f"    regex -> {len(regex_result)} sentences: {regex_result}")
        print(f"    spaCy -> {len(spacy_result)} sentences: {spacy_result}")
        verdict = "spaCy correct" if len(spacy_result) < len(regex_result) else "same result"
        print(f"    {verdict}\n")

    print("  Case 3 shows spaCy is better, not perfect: it still breaks after 'Approx.'")
    print("  No sentence splitter is exact — but the regex is wrong far more often.")


def build_all_strategies(document):
    """Run the five methods and return {name: [Chunk, ...]}."""
    header("THE FIVE CHUNKING STRATEGIES")

    strategies = {}
    timings = {}

    plan = [
        ("1 fixed-size", lambda: fixed_size(document.text, size=500, overlap=60)),
        ("2 sentence", lambda: sentence_based(document.text, max_chars=600)),
        ("3 recursive", lambda: recursive(document.text, max_chars=600)),
        ("4 semantic", lambda: semantic(document.text, embed, n_clusters=8)),
        ("5 structure", lambda: structure_aware(document, max_chars=700)),
    ]

    for name, build in plan:
        started = time.perf_counter()
        strategies[name] = build()
        timings[name] = time.perf_counter() - started

    section("Comparison")
    print(f"  {'method':14} {'chunks':>7} {'avg':>7} {'min':>6} {'max':>6} {'build s':>9}  notes")
    for name, chunks in strategies.items():
        lengths = [chunk.chars for chunk in chunks]
        note = ""
        if name.startswith("1"):
            note = "overlap 60 chars (12%)"
        elif name.startswith("2"):
            note = f"engine: {chunks[0].meta.get('engine', '?')}"
        elif name.startswith("4"):
            note = f"{chunks[0].meta.get('sentences_clustered', '?')} sentences clustered"
        elif name.startswith("5"):
            kinds = {chunk.kind for chunk in chunks}
            note = f"kinds: {', '.join(sorted(kinds))}"

        print(
            f"  {name:14} {len(chunks):7d} {sum(lengths) / len(lengths):7.1f} "
            f"{min(lengths):6d} {max(lengths):6d} {timings[name]:9.2f}  {note}"
        )

    section("One sample chunk from each method")
    for name, chunks in strategies.items():
        sample = chunks[len(chunks) // 2]
        preview = sample.text.replace("\n", " ")
        print(f"\n  {name} (chunk {sample.index}/{len(chunks)}, {sample.chars} chars):")
        print(f"    {preview[:200]}{'...' if len(preview) > 200 else ''}")

    return strategies


def embed_all(strategies):
    header("EMBEDDINGS")

    stores = {}
    for name, chunks in strategies.items():
        started = time.perf_counter()
        vectors = embed([chunk.text for chunk in chunks])
        elapsed = time.perf_counter() - started

        stores[name] = [
            {"chunk": chunk, "embedding": vector} for chunk, vector in zip(chunks, vectors)
        ]
        print(
            f"  {name:14} {len(vectors):3d} vectors x {len(vectors[0])} dims "
            f"in {elapsed:5.2f}s"
        )

    dims = {len(record["embedding"]) for store in stores.values() for record in store}
    print(f"\n  All strategies produce {dims.pop()}-dimension vectors: the dimension comes")
    print("  from the embedding model, not from how the text was split.")

    return stores


def run_queries(stores):
    header("SEMANTIC SEARCH — TOP-3 PER QUERY, PER STRATEGY")

    for query in QUERIES:
        query_vector = embed(query)[0]

        section(f"Query: {query}")

        for name, store in stores.items():
            scored = sorted(
                (
                    {"score": cosine_similarity(query_vector, record["embedding"]),
                     "chunk": record["chunk"]}
                    for record in store
                ),
                key=lambda item: item["score"],
                reverse=True,
            )

            best = scored[0]
            preview = best["chunk"].text.replace("\n", " ")[:118]
            print(f"\n  {name:14} top score {best['score']:.4f}")
            print(f"    {preview}...")

        # Full top-3 for the strategy the course recommends, so the output shows
        # a complete answer rather than only headline scores.
        recommended = stores["3 recursive"]
        scored = sorted(
            (
                {"score": cosine_similarity(query_vector, record["embedding"]),
                 "chunk": record["chunk"]}
                for record in recommended
            ),
            key=lambda item: item["score"],
            reverse=True,
        )[:3]

        print(f"\n  Full TOP-3 (recursive):")
        for rank, item in enumerate(scored, start=1):
            text = item["chunk"].text.replace("\n", " ")
            print(f"    {rank}) {item['score']:.4f}  {text[:200]}{'...' if len(text) > 200 else ''}")


def summarise_document(strategies):
    """Summarise the document from the recommended strategy's chunks."""
    header("SUMMARISING THE DOCUMENT (recursive chunks, map-reduce)")

    chunks = strategies["3 recursive"]

    print(f"\n  Summarising {len(chunks)} chunks, then merging.")

    partials = []
    for position, chunk in enumerate(chunks, start=1):
        partial = summarise(
            chunk.text,
            "Summarise this passage from an NLP course document in one sentence.",
            max_tokens=70,
        )
        partials.append(partial)
        if position <= 5 or position == len(chunks):
            print(f"    [{position:2d}/{len(chunks)}] {partial}")
        elif position == 6:
            print(f"    ... ({len(chunks) - 6} more)")

    # Merge in batches: 33 bullet points in one prompt exceeds what a 3B model
    # merges reliably, and the failure mode is silent truncation.
    section("Merging in batches of 8")
    batch_summaries = []
    for start in range(0, len(partials), 8):
        batch = partials[start : start + 8]
        merged = summarise(
            "\n".join(f"- {p}" for p in batch),
            "These are summaries of consecutive passages. Merge them.",
            max_tokens=200,
            merge=True,
        )
        batch_summaries.append(merged)
        print(f"  batch {start // 8 + 1}: {merged}")

    section("Final document summary")
    final = summarise(
        "\n".join(f"- {s}" for s in batch_summaries),
        "These are section summaries of one course document. Write the final overall summary.",
        max_tokens=250,
        merge=True,
    )
    print(f"  {final}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOC

    try:
        check_ready()
    except OllamaError as error:
        print(f"Error: {error}")
        sys.exit(1)

    header("SETUP")
    print(f"  Document        : {path}")
    print(f"  LLM model       : {LLM_MODEL}")
    print(f"  Embedding model : {EMBED_MODEL}")
    print(f"  spaCy available : {spacy_ready()}")

    try:
        document = load_docx(path)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error loading document: {error}")
        sys.exit(1)

    section("Document profile")
    for line in describe(document).splitlines():
        print(f"  {line}")

    try:
        show_spacy_vs_regex()
        strategies = build_all_strategies(document)
        stores = embed_all(strategies)
        run_queries(stores)
        summarise_document(strategies)
    except OllamaError as error:
        print(f"\nError talking to Ollama: {error}")
        sys.exit(1)

    print(f"\n{LINE}\nDone.\n{LINE}")


if __name__ == "__main__":
    main()
