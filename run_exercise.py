"""
Text Summarization Using Three Chunking Strategies
==================================================

Runs all six steps against a local Ollama instance.

    LLM        : llama3.2:3b
    Embeddings : nomic-embed-text

Run:
    ollama serve            # in another terminal, if not already running
    python run_exercise.py
"""

import sys

from chunking import TEXT, fixed_size_chunks, hierarchical_chunks, semantic_chunks
from embeddings import build_store, cosine_similarity, search
from ollama_client import EMBED_MODEL, LLM_MODEL, OllamaError, check_ready, embed, summarise

QUERY = "What are the ethical concerns related to artificial intelligence?"

LINE = "=" * 78
THIN = "-" * 78


def header(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def section(title: str) -> None:
    print(f"\n{title}\n{THIN}")


# ---------------------------------------------------------------------------
# Steps 1-3
# ---------------------------------------------------------------------------


def step1_fixed():
    header("STEP 1 — FIXED-SIZE CHUNKING (50 characters)")

    chunks = fixed_size_chunks(TEXT, size=50)

    section(f"Generated chunks: {len(chunks)}")
    for index, chunk in enumerate(chunks, start=1):
        print(f"  [{index}] ({len(chunk):2d} chars) {chunk!r}")

    print("\n  Note the damage: cuts land mid-word ('automa|te', 'an|d',")
    print("  'transpa|rency'), so some chunks carry a broken idea.")

    section("Summary of each chunk")
    partials = []
    for index, chunk in enumerate(chunks, start=1):
        partial = summarise(
            chunk,
            "Summarise this fragment of a longer text. It may be cut mid-sentence; "
            "summarise only what is actually there.",
            max_tokens=60,
        )
        partials.append(partial)
        print(f"  [{index}] {partial}")

    section("Combined final summary")
    final = summarise(
        "\n".join(f"- {partial}" for partial in partials),
        "These are partial summaries of one text, in order. Merge them into a "
        "single coherent summary of the whole text.",
        merge=True,
        max_tokens=180,
    )
    print(f"  {final}")

    return chunks, final


def step2_semantic():
    header("STEP 2 — SEMANTIC CHUNKING")

    chunks = semantic_chunks(TEXT)

    section(f"Generated chunks: {len(chunks)}")
    for index, chunk in enumerate(chunks, start=1):
        print(f"  [{index}] ({len(chunk):3d} chars) {chunk}")

    print("\n  Each chunk is one complete sentence, so each carries one whole idea:")
    print("  what AI does, what it enables, what it costs.")

    section("Summary of each chunk")
    partials = []
    for index, chunk in enumerate(chunks, start=1):
        partial = summarise(chunk, "Summarise this sentence in your own words.", max_tokens=60)
        partials.append(partial)
        print(f"  [{index}] {partial}")

    section("Combined final summary")
    final = summarise(
        "\n".join(f"- {partial}" for partial in partials),
        "These are summaries of consecutive parts of one text. Merge them into a "
        "single coherent summary.",
        merge=True,
        max_tokens=180,
    )
    print(f"  {final}")

    return chunks, final


def step3_hierarchical():
    header("STEP 3 — HIERARCHICAL / RECURSIVE CHUNKING")

    tree = hierarchical_chunks(TEXT, max_chars=90)

    section("Hierarchical structure")
    flat = []
    for p_index, node in enumerate(tree, start=1):
        print(f"  DOCUMENT")
        print(f"  └── PARAGRAPH {p_index} ({len(node['paragraph'])} chars)")
        for c_index, child in enumerate(node["children"], start=1):
            connector = "└──" if c_index == len(node["children"]) else "├──"
            print(f"      {connector} chunk {p_index}.{c_index} ({len(child):3d}) {child}")
            flat.append(child)

    section("Level 1 — summary of each small chunk")
    for node in tree:
        node["child_summaries"] = []
        for c_index, child in enumerate(node["children"], start=1):
            partial = summarise(child, "Summarise this fragment in one short sentence.", max_tokens=60)
            node["child_summaries"].append(partial)
            print(f"  [{c_index}] {partial}")

    section("Level 2 — summary of each paragraph, from its sub-chunks")
    paragraph_summaries = []
    for p_index, node in enumerate(tree, start=1):
        paragraph_summary = summarise(
            "\n".join(f"- {s}" for s in node["child_summaries"]),
            "These are summaries of consecutive fragments of one paragraph. "
            "Merge them into a single paragraph summary.",
        merge=True,
        max_tokens=180,
    )
        paragraph_summaries.append(paragraph_summary)
        print(f"  Paragraph {p_index}: {paragraph_summary}")

    section("Level 3 — final global summary")
    final = summarise(
        "\n".join(f"- {s}" for s in paragraph_summaries),
        "These are paragraph summaries of one document. Write the final overall summary.",
        merge=True,
        max_tokens=180,
    )
    print(f"  {final}")

    return flat, final


# ---------------------------------------------------------------------------
# Step 4
# ---------------------------------------------------------------------------


def step4_embeddings(strategies):
    header("STEP 4 — DOCUMENT EMBEDDINGS")

    stores = {}
    for name, chunks in strategies.items():
        stores[name] = build_store(chunks)

    section("Embedding dimensions")
    for name, store in stores.items():
        dims = {len(record["embedding"]) for record in store}
        print(f"  {name:14} {len(store)} chunks, vector dimension {dims.pop()}")

    section("Stored structure (one record per chunk)")
    example = stores["semantic"][0]
    print("  Each record is a dict:")
    print(f"    id        : {example['id']}")
    print(f"    text      : {example['text'][:60]}...")
    print(f"    chars     : {example['chars']}")
    print(f"    embedding : [{example['embedding'][0]:.5f}, {example['embedding'][1]:.5f}, "
          f"...]  ({len(example['embedding'])} floats)")

    section("Comparing the three strategies")
    print(f"  {'strategy':14} {'chunks':>7} {'dim':>5} {'avg chars':>10} {'min':>5} {'max':>5}")
    for name, store in stores.items():
        lengths = [record["chars"] for record in store]
        print(
            f"  {name:14} {len(store):7d} {len(store[0]['embedding']):5d} "
            f"{sum(lengths) / len(lengths):10.1f} {min(lengths):5d} {max(lengths):5d}"
        )

    print("\n  Every strategy produces vectors of the same dimension — that is set by")
    print("  the embedding model, not by how the text was split. What changes is how")
    print("  many vectors there are and how much meaning each one carries.")

    # How self-consistent is each strategy? Chunks from a coherent split should
    # relate to each other; fragments cut mid-word drift apart.
    section("Internal coherence (mean pairwise similarity between a strategy's own chunks)")
    for name, store in stores.items():
        vectors = [record["embedding"] for record in store]
        pairs = [
            cosine_similarity(vectors[i], vectors[j])
            for i in range(len(vectors))
            for j in range(i + 1, len(vectors))
        ]
        mean = sum(pairs) / len(pairs) if pairs else 0.0
        print(f"  {name:14} {mean:.4f}   ({len(pairs)} pairs)")

    return stores


# ---------------------------------------------------------------------------
# Step 5
# ---------------------------------------------------------------------------


def step5_query(stores):
    header("STEP 5 — QUERY EMBEDDING")

    print(f"\n  Query: {QUERY!r}")

    query_vector = embed(QUERY)[0]

    section("Dimension check")
    print(f"  Query embedding dimension    : {len(query_vector)}")

    for name, store in stores.items():
        document_dim = len(store[0]["embedding"])
        match = "MATCH" if document_dim == len(query_vector) else "MISMATCH"
        print(f"  {name:14} document dimension: {document_dim}  -> {match}")

    print("\n  The dimensions must match or the dot product is undefined — which is why")
    print("  the query and the documents have to go through the same embedding model.")

    return query_vector


# ---------------------------------------------------------------------------
# Step 6
# ---------------------------------------------------------------------------


def step6_search(query_vector, stores):
    header("STEP 6 — COSINE SIMILARITY SEARCH")

    for name, store in stores.items():
        section(f"Strategy: {name}")

        scored = [
            {"score": cosine_similarity(query_vector, record["embedding"]), **record}
            for record in store
        ]
        scored.sort(key=lambda record: record["score"], reverse=True)

        print("  All chunks, sorted by similarity (descending):")
        for rank, record in enumerate(scored, start=1):
            marker = " <-- TOP" if rank <= 3 else ""
            preview = record["text"] if len(record["text"]) <= 62 else record["text"][:59] + "..."
            print(f"    {rank}. {record['score']:.4f}  {preview!r}{marker}")

        print(f"\n  TOP-3 for {name}:")
        for rank, record in enumerate(search(query_vector, store, top_k=3), start=1):
            print(f"    {rank}) score {record['score']:.4f}")
            print(f"       {record['text']}")

    section("What the search results show")
    print("  The query asks about ETHICAL CONCERNS. The right answer is the third")
    print("  sentence — privacy, transparency, employment. Compare how well each")
    print("  strategy surfaces it as the single top hit.")


# ---------------------------------------------------------------------------


def main() -> None:
    try:
        check_ready()
    except OllamaError as error:
        print(f"Error: {error}")
        sys.exit(1)

    header("SETUP")
    print(f"  LLM model       : {LLM_MODEL}")
    print(f"  Embedding model : {EMBED_MODEL}")
    print(f"  Source text     : {len(TEXT)} characters")
    print(f"\n  {TEXT}")

    try:
        fixed, fixed_final = step1_fixed()
        semantic, semantic_final = step2_semantic()
        hierarchy, hierarchy_final = step3_hierarchical()

        strategies = {"fixed-size": fixed, "semantic": semantic, "hierarchical": hierarchy}

        stores = step4_embeddings(strategies)
        query_vector = step5_query(stores)
        step6_search(query_vector, stores)

        header("THE THREE FINAL SUMMARIES, SIDE BY SIDE")
        for name, summary in (
            ("Fixed-size", fixed_final),
            ("Semantic", semantic_final),
            ("Hierarchical", hierarchy_final),
        ):
            print(f"\n  {name}:\n    {summary}")

    except OllamaError as error:
        print(f"\nError talking to Ollama: {error}")
        sys.exit(1)

    print(f"\n{LINE}\nDone.\n{LINE}")


if __name__ == "__main__":
    main()
