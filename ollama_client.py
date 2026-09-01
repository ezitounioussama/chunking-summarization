"""Thin wrapper over the local Ollama HTTP API.

Everything runs on this machine: no API key, no network calls to a provider.
Ollama listens on http://localhost:11434 and exposes two endpoints we need:

    POST /api/generate   text generation      (qwen3:8b)
    POST /api/embed      embedding vectors    (nomic-embed-text)

Only the standard library is used for the requests, so the exercise needs no
HTTP dependency at all.
"""

import json
import urllib.error
import urllib.request

BASE_URL = "http://127.0.0.1:11434"

LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "nomic-embed-text"

# Low temperature: summarising is not a creative task, and the same chunk should
# summarise the same way every run so the three strategies can be compared fairly.
TEMPERATURE = 0.1


class OllamaError(RuntimeError):
    """Ollama is unreachable or returned something unusable."""


def _post(path: str, payload: dict, timeout: int = 180) -> dict:
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    except urllib.error.URLError as error:
        # The usual cause is that `ollama serve` is not running.
        raise OllamaError(
            f"Could not reach Ollama at {BASE_URL} ({error}). "
            "Start it with:  ollama serve"
        ) from error

    except json.JSONDecodeError as error:
        raise OllamaError(f"Ollama returned invalid JSON: {error}") from error


def summarise(text: str, instruction: str, max_tokens: int = 120,
              merge: bool = False) -> str:
    """Ask the LLM to summarise `text`.

    The prompt says "only what the text says" because a 3B model will happily
    expand a 50-character fragment into a paragraph of invention otherwise —
    which matters here, since Step 1 deliberately feeds it fragments cut
    mid-word.

    `merge=True` is for combining several partial summaries into one. It needs a
    different rule from a leaf summary: asking for "one short sentence" while
    handing the model five bullet points makes it return the first bullet and
    silently drop the rest. This was observed — the hierarchical global summary
    came back as only "Artificial intelligence is profoundly transforming modern
    businesses.", losing the ethics point entirely. Merging asks explicitly for
    every point to be covered.
    """
    if merge:
        rules = (
            "Rules: cover EVERY point listed below, none omitted. "
            "One or two sentences. Use only what the points say. "
            "Do not add facts. Reply with the summary only."
        )
    else:
        rules = (
            "Rules: one short sentence. Use only what the text says. "
            "Do not add facts. Do not explain your answer. Reply with the sentence only."
        )

    prompt = f"{instruction}\n\n{rules}\n\nTEXT:\n{text}\n\nSUMMARY:"

    data = _post(
        "/api/generate",
        {
            "model": LLM_MODEL,
            # qwen3 reasons by default and then returns an EMPTY "response",
            # with the chain of thought in a separate field. Off.
            "think": False,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": TEMPERATURE, "num_predict": max_tokens},
        },
    )

    reply = data.get("response", "").strip()
    if not reply:
        raise OllamaError("The model returned an empty summary.")

    # Models sometimes wrap the sentence in quotes; strip them for clean output.
    return reply.strip('"').strip()


def embed(texts):
    """Return one embedding vector per input string.

    /api/embed takes a list and returns a list, so all the chunks of a strategy
    are embedded in a single call rather than one request each.
    """
    if isinstance(texts, str):
        texts = [texts]

    data = _post("/api/embed", {"model": EMBED_MODEL, "input": list(texts)})

    vectors = data.get("embeddings")
    if not vectors:
        raise OllamaError("Ollama returned no embeddings.")

    return vectors


def check_ready() -> None:
    """Fail early with a readable message if Ollama or a model is missing."""
    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/tags", timeout=10) as response:
            installed = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise OllamaError(
            f"Ollama is not running at {BASE_URL} ({error}).\n"
            "Start it with:  ollama serve"
        ) from error

    names = {model["name"].split(":")[0] for model in installed.get("models", [])}

    for needed in (LLM_MODEL, EMBED_MODEL):
        if needed.split(":")[0] not in names:
            raise OllamaError(
                f"Model '{needed}' is not installed.\nPull it with:  ollama pull {needed}"
            )
