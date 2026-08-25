"""Steps 4-6: embeddings, query embedding, cosine similarity search.

Cosine similarity is written out by hand rather than pulled from numpy or
scikit-learn, so the exercise runs with the standard library only and the maths
is visible.
"""

import math
from typing import Dict, List

from ollama_client import embed


def build_store(chunks: List[str]) -> List[Dict]:
    """Embed every chunk and keep text and vector together.

    A list of dicts is the right structure here: each record carries the chunk,
    its vector, and its length, so a similarity result can be traced straight
    back to the text that produced it. A real system would use a vector database
    for this, which is the same idea with an index on top.
    """
    vectors = embed(chunks)

    return [
        {"id": index, "text": chunk, "embedding": vector, "chars": len(chunk)}
        for index, (chunk, vector) in enumerate(zip(chunks, vectors), start=1)
    ]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine of the angle between two vectors.

        cos(a, b) = (a . b) / (|a| * |b|)

    It measures direction, not length. That is what makes it the right choice
    for text: a 20-word chunk and a 200-word chunk about the same topic point the
    same way, even though one vector may be longer. Plain Euclidean distance
    would call them far apart just because of the size difference.

    The result runs from -1 (opposite) through 0 (unrelated) to 1 (identical
    direction). Embedding models in practice return mostly positive values.
    """
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    # A zero vector has no direction, so similarity is undefined; 0.0 is the
    # safe answer and avoids a division by zero.
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def search(query_vector: List[float], store: List[Dict], top_k: int = 3) -> List[Dict]:
    """Score every chunk against the query and return the best `top_k`.

    Every chunk is scored — no filtering — then sorted descending, which is
    exactly what a vector search does before the index optimisation.
    """
    scored = [
        {"score": cosine_similarity(query_vector, record["embedding"]), **record}
        for record in store
    ]

    scored.sort(key=lambda record: record["score"], reverse=True)

    return scored[:top_k]
