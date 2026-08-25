# Text Summarization Using Three Chunking Strategies

All six steps of the exercise, running **locally on Ollama** — no API key, no provider calls.

> **[RESULTS.md](RESULTS.md) — every captured run log, both parts.** Chunk listings, summaries,
> embedding dimensions, cosine scores and the two bugs found while building.

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
| **[`RESULTS.md`](RESULTS.md)** | **All captured results, both parts — start here** |
| `docs/output.txt` | Raw terminal log of the Part 1 run (219 lines) |

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

---

# Part 2 — spaCy sentences, and running on a real .docx

The first part above ran on a 278-character string with a regex sentence splitter. This part
answers two follow-ups: **use spaCy instead of the regex**, and **run on a real document** rather
than a hard-coded string.

Target document: `nlp_chunking_vectorization_EN.docx` — the NLP course notes, 367 paragraphs,
17,706 characters, 36 headings, 83 list items, 33 sections.

## New files

| File | Contents |
|---|---|
| `docx_loader.py` | Reads `.docx` **keeping its structure** — headings, list items, sections |
| `chunkers.py` | The course's **five** methods, with spaCy for sentence boundaries |
| `run_on_document.py` | Full pipeline on a real document |
| `docs/output_document.txt` | Raw terminal log of the Part 2 run (187 lines) — summarised in [`RESULTS.md`](RESULTS.md) |

```bash
pip install spacy scikit-learn numpy
python -m spacy download en_core_web_sm

python run_on_document.py                      # defaults to the course doc
python run_on_document.py path/to/other.docx   # any .docx
```

## Is spaCy better than the regex? Yes, measurably

The regex `re.split(r"(?<=[.!?])\s+", text)` treats **every** full stop as a sentence end. spaCy
decides from grammatical structure using a trained model, so abbreviations survive:

| Input | regex | spaCy |
|---|---|---|
| `Mr. Smith has arrived. He met Dr. Jones at 3.5 p.m.` | **4** sentences — `'Mr.'`, `'Smith has arrived.'`, `'He met Dr.'`, `'Jones at 3.5 p.m.'` | **2** — correct |
| `The U.S.A. leads in A.I. research. Costs fell 12.5% in Q3.` | **4** sentences | **2** — correct |
| `See Fig. 2 for details. Approx. 40 items were tested (e.g. sensors).` | **5** sentences | **3** |

The regex turns two sentences into four, and every broken piece becomes a chunk that carries half an
idea. That is the whole argument for spaCy here.

**Honest limit:** case 3 shows spaCy is *better, not perfect* — it still breaks after `Approx.`. No
sentence splitter is exact. The regex is simply wrong far more often, and it is wrong in a way that
scales: more abbreviations means more bad chunks.

`chunkers.py` keeps a regex fallback, so the module still runs if spaCy is not installed, and
reports which engine was used.

## The course's five methods on the real document

| Method | Chunks | Avg chars | Min | Max | Build time | Notes |
|---|---|---|---|---|---|---|
| 1 fixed-size | 41 | 489.8 | 106 | 500 | 0.00 s | overlap 60 chars (12%) |
| 2 sentence | 37 | 475.2 | 118 | 600 | 0.14 s | spaCy |
| 3 recursive | 33 | 534.6 | 153 | 600 | 0.00 s | course's recommended default |
| 4 semantic | 8 | 2186.0 | 153 | 4367 | **9.04 s** | 142 sentences clustered with KMeans |
| 5 structure | 64 | 282.8 | 27 | 664 | 0.00 s | headings + lists treated separately |

Two additions over Part 1, both taken from the course:

**Method 1 now has overlap.** Each chunk repeats the last 60 characters of the previous one, so an
idea straddling a boundary survives whole somewhere. The course suggests 10–15% of chunk size; 60/500
is 12%.

**Method 4 is real semantic chunking.** Part 1 called sentence splitting "semantic", which was
loose. The course's Method 4 embeds every sentence and clusters the vectors with **KMeans**, so
sentences about the same topic group together *even if they are pages apart* — the only method here
that ignores document order. It cost 9.04 s against ~0.00 s for the others, because it needs one
embedding per sentence before any chunking happens.

**Method 5 needs the document, not a string.** This is why `docx_loader.py` parses the XML instead
of flattening to text. It keeps headings as section labels rather than chunks of their own, keeps
consecutive list items together, and **prepends the section heading to every chunk** — so a chunk
retrieved alone still says what it is about.

## Retrieval results — three real questions about the document

Top cosine score per strategy, `*` marks the winner:

| Query | fixed | sentence | recursive | semantic | structure |
|---|---|---|---|---|---|
| Which chunking method should I start with, and what overlap? | 0.7273 | 0.7179 | 0.7184 | 0.6197 | **0.7683** * |
| What is an embedding and what does cosine similarity measure? | 0.7640 | 0.7370 | 0.7782 | 0.6257 | **0.8030** * |
| How do I run a model locally with Ollama? | 0.7468 | 0.7569 | **0.7714** * | 0.7447 | 0.7589 |

**Structure-aware won 2 of 3.** The reason is the heading prefix: its top hit for the embedding
query was `[Cosine similarity — comparing two vectors] Once embeddings are generated...`. The
heading repeats the query's own vocabulary, so the chunk matches on both its title and its body.
Cheap to compute, and the biggest single win in the whole comparison.

**Semantic (KMeans) came last on every query** — 0.6197, 0.6257, 0.7447. That is not a bug, and it
is worth understanding: with 8 clusters over 17,706 characters the chunks average **2,186 characters**
(largest 4,367). Each one mixes several topics, so its vector is an average of all of them and
matches no single question sharply. Clustering also pulls in sentences from across the document,
which dilutes it further. The fix would be far more clusters — but then it converges toward
sentence chunking at many times the cost. **The most sophisticated method was the worst performer
here**, because the document is short and already well structured.

**Fixed-size still returns unusable text.** Its top hit for query 1 begins `ed-size chunking  Take a
free-form 500-word text...` — the chunk starts mid-word ("fix|ed-size"). Score 0.7273, barely below
the winner, but you cannot show that snippet to a user. Same lesson as Part 1: **a good score does
not mean a usable chunk.**

## Document summary (map-reduce over recursive chunks)

33 chunk summaries → 5 batch summaries → 1 final:

> The course teaches two key NLP techniques, chunking and vectorization, to transform raw text into a
> usable format, focusing on solving problems like dealing with long text and converting text into
> numerical representations. It covers various methods for chunking, including sentence splitting,
> recursive chunking, and semantic chunking, and explains how to generate vectors, store them, and
> perform similarity searches using tools like Ollama and ChromaDB.

Accurate, including details only present deep in the document (ChromaDB, the pipeline shape).

**Why batches of 8 and not one merge.** Part 1 hit a silent failure where a merge prompt handed many
bullet points returned only the first. With 33 partial summaries that risk is much higher, so merging
happens in batches of 8, then once more over the batch summaries. Same lesson: the failure mode is a
fluent, plausible, incomplete answer — nothing errors.

## Which method to actually use

| If your document is… | Use |
|---|---|
| Structured — headings, sections, lists (`.docx`, Markdown, HTML) | **Structure-aware**, with the heading prefixed to each chunk |
| Unstructured prose with no headings | **Recursive** — the course's default, and a good one |
| Needs uniform chunk sizes for a hard token budget | **Fixed-size with overlap** — accept that boundaries cut words |
| Long, sprawling, poorly organised, where one topic recurs in many places | **Semantic (KMeans)** — the only method that gathers scattered mentions |
| Short, or you need a cheap sensible baseline | **Sentence (spaCy)** |

For this document, structure-aware plus recursive is the right pairing, and the 9 seconds spent on
KMeans bought nothing.

---

## Results and logs

Every number quoted in this README comes from a real run on this machine. The captured output is
collected in one place:

**→ [RESULTS.md](RESULTS.md)**

| Section | What is in it |
|---|---|
| Part 1 — six steps | Chunk listings, per-chunk summaries, embedding dimensions, coherence scores, full sorted cosine results |
| Part 2 — real `.docx` | Document profile, spaCy vs regex cases, five-method comparison, retrieval scores for three queries, map-reduce document summary |
| Test suite | 23 tests, passing |
| Two failures and their fixes | A merge prompt that silently dropped four of five points, and the batching that prevents it at 33 chunks |

Raw terminal logs, unedited: [`docs/output.txt`](docs/output.txt) (Part 1) and
[`docs/output_document.txt`](docs/output_document.txt) (Part 2).
