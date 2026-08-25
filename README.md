# Text Summarization Using Three Chunking Strategies

All six steps of the exercise, running **locally on Ollama** — no API key, no provider calls.

| | |
|---|---|
| LLM | `llama3.2:3b` |
| Embeddings | `nomic-embed-text` (768 dimensions) |
| Dependencies | **none** — standard library only (`urllib`, `math`, `re`) |

## Files

| File | Contents |
|---|---|
| `run_exercise.py` | Driver — runs Steps 1–6 and prints everything |
| `chunking.py` | The three strategies (Steps 1–3), no model calls |
| `embeddings.py` | Embedding store, cosine similarity, search (Steps 4–6) |
| `ollama_client.py` | Thin wrapper over the local Ollama HTTP API |
| `tests.py` | 23 tests for the splitting and the maths (no model needed) |
| `docs/output.txt` | Full captured output of a real run |

## Running it

```bash
ollama serve                      # if not already running
ollama pull llama3.2:3b
ollama pull nomic-embed-text

python3 run_exercise.py           # full exercise
python3 tests.py                  # 23 tests, no Ollama needed
```

`run_exercise.py` checks that Ollama is up and both models are installed before it starts, and
tells you the exact `ollama pull` command if one is missing.

---

## Results

### Step 1 — Fixed-size chunking (50 characters)

6 chunks. The cuts ignore meaning entirely:

```
[1] 'Artificial intelligence is profoundly transforming'
[2] ' modern businesses. It makes it possible to automa'
[3] 'te repetitive tasks, analyze customer behavior, an'
[4] 'd predict market trends. However, its integration '
[5] 'raises ethical concerns regarding privacy, transpa'
[6] 'rency, and human employment.'
```

`automa|te`, `an|d`, `transpa|rency` — words split down the middle. The per-chunk summaries show the
damage directly: chunk 2 summarised to *"It makes it possible to automate."* — the object of the
sentence was in the next chunk.

### Step 2 — Semantic chunking

3 chunks, one per sentence, each carrying one complete idea:

1. what AI does — transforming businesses
2. what it enables — automation, customer analysis, trend prediction
3. what it costs — privacy, transparency, employment

### Step 3 — Hierarchical / recursive chunking

```
DOCUMENT
└── PARAGRAPH 1 (278 chars)
    ├── chunk 1.1 ( 69) Artificial intelligence is profoundly transforming modern businesses.
    ├── chunk 1.2 ( 76) It makes it possible to automate repetitive tasks, analyze customer behavior
    ├── chunk 1.3 ( 26) and predict market trends.
    ├── chunk 1.4 ( 80) However, its integration raises ethical concerns regarding privacy, transparency
    └── chunk 1.5 ( 21) and human employment.
```

Three summary levels: each small chunk → each paragraph from its sub-chunks → one global summary.
The recursion tries the strongest boundary first (paragraphs, then sentences, then commas) and only
goes smaller when a piece is still too long, so a cut is only ever made where the language already
has a seam.

### The three final summaries

| Strategy | Final summary |
|---|---|
| **Fixed-size** | "…transforming various aspects of life, enabling automation, analysis of customer behavior, prediction of market trends, and raising ethical concerns about privacy, while also affecting human employment." |
| **Semantic** | "…changing businesses by enabling automation and analysis of various business processes, while also raising ethical concerns due to integration." |
| **Hierarchical** | "…transforming modern businesses by automating repetitive tasks and analyzing customer behavior, while also predicting market trends, and its integration raises concerns about privacy and transparency, as well as its impact on human employment." |

Hierarchical is the most complete — it keeps all three ideas *and* names the specific concerns.
Fixed-size recovers a lot but drifts ("various aspects of life" instead of *businesses*, and it
loses *transparency*). Semantic is the most concise but compresses the concerns to "ethical
concerns due to integration", dropping the specifics.

### Step 4 — Embeddings

| Strategy | Chunks | Dimension | Avg chars | Min | Max |
|---|---|---|---|---|---|
| fixed-size | 6 | 768 | 46.3 | 28 | 50 |
| semantic | 3 | 768 | 92.0 | 69 | 104 |
| hierarchical | 5 | 768 | 54.4 | 21 | 80 |

Every strategy produces **768-dimension** vectors. The dimension comes from the embedding model,
not from how the text was split — what the strategy changes is *how many* vectors there are and how
much meaning each one carries.

Each record is stored as a dict, so a similarity score can be traced back to the text that produced
it:

```python
{"id": 1, "text": "Artificial intelligence is...", "chars": 69, "embedding": [-0.02727, 0.08234, ...]}
```

Internal coherence (mean pairwise cosine between a strategy's own chunks):

```
fixed-size     0.4306   (15 pairs)
semantic       0.4943   (3 pairs)
hierarchical   0.4513   (10 pairs)
```

Semantic chunks are the most related to each other, fixed-size the least — consistent with
fragments cut mid-word carrying partial ideas that drift apart in vector space.

### Step 5 — Query embedding

```
Query: "What are the ethical concerns related to artificial intelligence?"
Query embedding dimension : 768
fixed-size / semantic / hierarchical document dimension: 768 -> MATCH
```

The dimensions must match or the dot product is undefined, which is why the query and the documents
have to go through the same embedding model.

### Step 6 — Cosine similarity search

Top-1 hit per strategy, for a query about ethical concerns:

| Strategy | Top score | Top chunk |
|---|---|---|
| fixed-size | **0.6966** | `'raises ethical concerns regarding privacy, transpa'` — truncated mid-word |
| semantic | **0.7052** | `'However, its integration raises ethical concerns regarding privacy, transparency, and human employment.'` — complete |
| hierarchical | **0.6979** | `'However, its integration raises ethical concerns regarding privacy, transparency'` — nearly complete |

All three rank the ethics content first, so all three "find" the answer. The difference is **what
you can do with the hit**:

- **Semantic** returns the whole answer. Hand it to a user or an LLM and it is usable as-is.
- **Hierarchical** returns almost all of it — "and human employment" ended up in a sibling chunk,
  though the parent node links them.
- **Fixed-size** returns a fragment ending `transpa`. The score is nearly as high, but the retrieved
  text is unusable on its own: the third concern is in a different chunk, which ranked 3rd at 0.4207.

That is the real lesson of the exercise: **a good similarity score does not mean a useful chunk.**
Retrieval quality is decided at split time, not at search time. The scores are all within 0.01 of
each other while the usefulness of the retrieved text differs completely.

---

## Comparison

| | Fixed-size | Semantic | Hierarchical |
|---|---|---|---|
| Splits on | character count | sentence boundaries | paragraphs, then sentences, then clauses |
| Respects meaning | no | yes | yes |
| Chunk count here | 6 | 3 | 5 |
| Cost to compute | lowest | low | moderate (recursive) |
| Retrieved text usable alone | **no** | **yes** | mostly |
| Summary completeness | drifts, loses detail | concise, loses specifics | most complete |
| Good for | uniform-size batching, hard token limits | Q&A and retrieval over prose | long structured documents |

**When each is the right choice.** Fixed-size is defensible when chunks must be a predictable size —
a hard token budget, or a model with a small context window — and it is the cheapest to compute.
Semantic is the default for retrieval over prose. Hierarchical earns its extra complexity on long
structured documents, where the parent level gives you a summary layer to search before drilling
into the leaves.

---

## One bug worth recording

The hierarchical global summary first came back as just:

> "Artificial intelligence is profoundly transforming modern businesses."

The ethics content had vanished — so the strategy that should perform best looked worst.

The cause was in the prompt, not the chunking. `summarise()` appended the rule *"one short
sentence"* to every call, including the merge steps. Handed five bullet points and told to produce
one short sentence, `llama3.2:3b` returned the first bullet and silently dropped the other four.

Fixed by giving merges their own rule — *"cover EVERY point listed below, none omitted, one or two
sentences"* — via a `merge=True` flag. The global summary now covers all three ideas. The failure is
worth recording because nothing errored: the output was a grammatical, plausible, badly incomplete
summary, which is the hardest kind of failure to notice.

---

Author: **Oussama Ezitouni**
