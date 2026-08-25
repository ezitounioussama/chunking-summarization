# Results Log

Everything on this page is captured output from real runs on this machine. Nothing is
hand-written or estimated.

**Environment**

| | |
|---|---|
| LLM | `llama3.2:3b` (via local Ollama, `ollama serve`) |
| Embeddings | `nomic-embed-text`, 768 dimensions |
| Sentence splitter | spaCy 3.8.16 with `en_core_web_sm` 3.8.0 |
| Clustering | scikit-learn 1.9.0 (`KMeans`) |
| Python | 3.14.6 |

**Raw logs**

| File | Contents |
|---|---|
| [`docs/output.txt`](docs/output.txt) | Part 1 — full run on the exercise string (219 lines) |
| [`docs/output_document.txt`](docs/output_document.txt) | Part 2 — full run on the `.docx` (187 lines) |

Reproduce with:

```bash
ollama serve
python3 run_exercise.py            # Part 1
python3 run_on_document.py         # Part 2
python3 tests.py                   # 23 tests, no Ollama needed
```

---

# Part 1 — the exercise string (six steps)

Source text: 278 characters.

> Artificial intelligence is profoundly transforming modern businesses. It makes it possible to
> automate repetitive tasks, analyze customer behavior, and predict market trends. However, its
> integration raises ethical concerns regarding privacy, transparency, and human employment.

## Step 1 — Fixed-size chunking (50 characters)

```
Generated chunks: 6
  [1] (50 chars) 'Artificial intelligence is profoundly transforming'
  [2] (50 chars) ' modern businesses. It makes it possible to automa'
  [3] (50 chars) 'te repetitive tasks, analyze customer behavior, an'
  [4] (50 chars) 'd predict market trends. However, its integration '
  [5] (50 chars) 'raises ethical concerns regarding privacy, transpa'
  [6] (28 chars) 'rency, and human employment.'
```

Cuts land mid-word: `automa|te`, `an|d`, `transpa|rency`.

Summary of each chunk:

```
  [1] Artificial intelligence is profoundly transforming
  [2] It makes it possible to automate.
  [3] Analyze customer behavior.
  [4] It predicts market trends.
  [5] Raises ethical concerns regarding privacy.
  [6] Human employment is affected by rency.
```

Chunk 2 lost its object — it was in chunk 3. Chunk 6 is worse: the model treated the fragment
`rency` as a real word and produced *"Human employment is affected by rency."* That is the
fixed-size failure mode in one line — a broken chunk produces a confidently wrong summary.

Combined final summary:

```
Artificial intelligence is profoundly transforming various aspects of life, enabling automation,
analysis of customer behavior, prediction of market trends, and raising ethical concerns about
privacy, while also affecting human employment.
```

Recovers most of the meaning, but drifted to "various aspects of life" instead of *businesses*, and
lost *transparency*.

## Step 2 — Semantic chunking (sentence level)

```
Generated chunks: 3
  [1] ( 69 chars) Artificial intelligence is profoundly transforming modern businesses.
  [2] (104 chars) It makes it possible to automate repetitive tasks, analyze customer behavior, and predict market trends.
  [3] (103 chars) However, its integration raises ethical concerns regarding privacy, transparency, and human employment.
```

One idea per chunk: what AI does, what it enables, what it costs.

Summaries, then the combination:

```
  [1] Artificial intelligence is changing businesses.
  [2] It enables automation and analysis of various business processes.
  [3] Integration raises ethical concerns.

Final: Artificial intelligence is changing businesses by enabling automation and analysis of
various business processes, while also raising ethical concerns due to integration.
```

Cleanest, but the most compressed — the specific concerns (privacy, transparency, employment)
disappeared into "ethical concerns".

## Step 3 — Hierarchical / recursive chunking

```
DOCUMENT
└── PARAGRAPH 1 (278 chars)
    ├── chunk 1.1 ( 69) Artificial intelligence is profoundly transforming modern businesses.
    ├── chunk 1.2 ( 76) It makes it possible to automate repetitive tasks, analyze customer behavior
    ├── chunk 1.3 ( 26) and predict market trends.
    ├── chunk 1.4 ( 80) However, its integration raises ethical concerns regarding privacy, transparency
    └── chunk 1.5 ( 21) and human employment.
```

Three summary levels:

```
Level 1 — per small chunk
  [1] Artificial intelligence is profoundly transforming modern businesses.
  [2] It automates repetitive tasks and analyzes customer behavior.
  [3] Predict market trends.
  [4] Its integration raises ethical concerns regarding privacy and transparency.
  [5] Human employment is included.

Level 2 — paragraph, from its sub-chunks
  Artificial intelligence is profoundly transforming modern businesses, automating repetitive
  tasks and analyzing customer behavior, while also predicting market trends, and its integration
  raises ethical concerns regarding privacy and transparency, and human employment is included.

Level 3 — final global summary
  Artificial intelligence is transforming modern businesses by automating repetitive tasks and
  analyzing customer behavior, while also predicting market trends, and its integration raises
  concerns about privacy and transparency, as well as its impact on human employment.
```

Most complete of the three: keeps all three ideas *and* names the specific concerns.

## Step 4 — Document embeddings

```
  fixed-size     6 chunks, vector dimension 768
  semantic       3 chunks, vector dimension 768
  hierarchical   5 chunks, vector dimension 768
```

Stored structure, one record per chunk:

```python
{"id": 1, "text": "Artificial intelligence is...", "chars": 69,
 "embedding": [-0.02727, 0.08234, ...]}   # 768 floats
```

Comparison:

```
  strategy        chunks   dim  avg chars   min   max
  fixed-size           6   768       46.3    28    50
  semantic             3   768       92.0    69   104
  hierarchical         5   768       54.4    21    80
```

Internal coherence (mean pairwise cosine between a strategy's own chunks):

```
  fixed-size     0.4306   (15 pairs)
  semantic       0.4943   (3 pairs)
  hierarchical   0.4513   (10 pairs)
```

Semantic chunks are most related to each other; fixed-size least — consistent with fragments cut
mid-word carrying partial ideas.

## Step 5 — Query embedding

```
Query: 'What are the ethical concerns related to artificial intelligence?'

Query embedding dimension    : 768
fixed-size     document dimension: 768  -> MATCH
semantic       document dimension: 768  -> MATCH
hierarchical   document dimension: 768  -> MATCH
```

## Step 6 — Cosine similarity search

All chunks scored and sorted, per strategy:

```
fixed-size
  1. 0.6966  'raises ethical concerns regarding privacy, transpa'   <-- TOP
  2. 0.6530  'Artificial intelligence is profoundly transforming'   <-- TOP
  3. 0.4207  'rency, and human employment.'                         <-- TOP
  4. 0.4196  ' modern businesses. It makes it possible to automa'
  5. 0.3574  'te repetitive tasks, analyze customer behavior, an'
  6. 0.3265  'd predict market trends. However, its integration '

semantic
  1. 0.7052  'However, its integration raises ethical concerns regarding privacy, transparency, and human employment.'  <-- TOP
  2. 0.6495  'Artificial intelligence is profoundly transforming modern businesses.'                                    <-- TOP
  3. 0.4358  'It makes it possible to automate repetitive tasks, analyze customer behavior, and predict market trends.' <-- TOP

hierarchical
  1. 0.6979  'However, its integration raises ethical concerns regarding privacy, transparency'  <-- TOP
  2. 0.6495  'Artificial intelligence is profoundly transforming modern businesses.'            <-- TOP
  3. 0.4605  'and human employment.'                                                            <-- TOP
  4. 0.4445  'It makes it possible to automate repetitive tasks, analyze customer behavior...'
  5. 0.3394  'and predict market trends.'
```

**Reading of the result.** All three rank the ethics content first, so all three "find" the answer,
and the top scores sit within 0.01 of each other. What differs is whether the retrieved text is
usable:

- **semantic** returns the complete sentence — usable as-is
- **hierarchical** returns nearly all of it, with "and human employment" in a sibling chunk
- **fixed-size** returns text ending `transpa` — unusable alone; the third concern is in a
  different chunk which ranked 3rd at 0.4207

A good similarity score does not mean a useful chunk. Retrieval quality is decided at split time.

---

# Part 2 — the real document

Source: `nlp_chunking_vectorization_EN.docx` (the NLP course notes).

```
blocks        : 367
  headings    : 36
  list items  : 83
  prose paras : 248
characters    : 17,706
sections      : 33
```

## spaCy versus regex — measured

```
CASE 1: Mr. Smith has arrived. He met Dr. Jones at 3.5 p.m.
  regex -> 4 sentences: ['Mr.', 'Smith has arrived.', 'He met Dr.', 'Jones at 3.5 p.m.']
  spaCy -> 2 sentences: ['Mr. Smith has arrived.', 'He met Dr. Jones at 3.5 p.m.']
  spaCy correct

CASE 2: The U.S.A. leads in A.I. research. Costs fell 12.5% in Q3.
  regex -> 4 sentences: ['The U.S.A.', 'leads in A.I.', 'research.', 'Costs fell 12.5% in Q3.']
  spaCy -> 2 sentences: ['The U.S.A. leads in A.I. research.', 'Costs fell 12.5% in Q3.']
  spaCy correct

CASE 3: See Fig. 2 for details. Approx. 40 items were tested (e.g. sensors).
  regex -> 5 sentences: ['See Fig.', '2 for details.', 'Approx.', '40 items were tested (e.g.', 'sensors).']
  spaCy -> 3 sentences: ['See Fig. 2 for details.', 'Approx.', '40 items were tested (e.g. sensors).']
  spaCy correct
```

The regex treats every full stop as a sentence end, so it turns two sentences into four and each
broken piece becomes a chunk carrying half an idea.

Case 3 shows spaCy is better but not perfect — it still breaks after `Approx.`. No splitter is
exact; the regex is simply wrong far more often, and increasingly so as abbreviations multiply.

## The course's five methods on the document

```
  method          chunks     avg    min    max   build s  notes
  1 fixed-size        41   489.8    106    500      0.00  overlap 60 chars (12%)
  2 sentence          37   475.2    118    600      0.14  engine: spaCy
  3 recursive         33   534.6    153    600      0.00
  4 semantic           8  2186.0    153   4367      9.04  142 sentences clustered
  5 structure         64   282.8     27    664      0.00  kinds: list, prose
```

Sample chunk from method 5, showing the heading prefix that only structure-aware chunking produces:

```
  5 structure (chunk 33/64, 646 chars):
    [Method 5 — Content-Aware Chunking] This method adapts the splitting strategy to the type of
    content detected in the text. It recognizes normal paragraphs, bullet lists, numbered lists,
    etc., and applies a different treatment to each...
```

## Embeddings

```
  1 fixed-size    41 vectors x 768 dims in  8.94s
  2 sentence      37 vectors x 768 dims in  8.95s
  3 recursive     33 vectors x 768 dims in  9.78s
  4 semantic       8 vectors x 768 dims in 10.92s
  5 structure     64 vectors x 768 dims in 10.54s
```

Every strategy produces 768-dimension vectors: the dimension comes from the embedding model, not
from how the text was split.

## Retrieval — three real questions about the document

Top cosine score per strategy. `*` marks the winner.

```
query                                                fixed-siz   sentence  recursive   semantic  structure
Which chunking method should I start with, overlap?     0.7273     0.7179     0.7184     0.6197    0.7683*
What is an embedding and what does cosine measure?      0.7640     0.7370     0.7782     0.6257    0.8030*
How do I run a model locally with Ollama?               0.7468     0.7569     0.7714*    0.7447    0.7589

wins: recursive 1, structure 2
```

Full top-3 for the recursive strategy on query 2:

```
1) 0.7782  Dimensionality directly affects the precision of similarity comparisons. Cosine
           similarity — comparing two vectors. Once embeddings are generated, how do you know if
           two texts are similar? Cosine sim...
2) 0.7548  Code — computing cosine similarity  import numpy as np def cosine_similarity(vec_a,
           vec_b): a = np.array(vec_a) b = np.array(vec_b) return np.dot(a, b) / ...
3) 0.7040  An embedding is a numerical vector that represents the semantic meaning of a piece of
           text (a word, sentence, or document)...
```

### Three findings

**1. Structure-aware won 2 of 3, because of the heading prefix.** Its top hit for query 2 was
`[Cosine similarity — comparing two vectors] Once embeddings are generated...`. The heading repeats
the query's vocabulary, so the chunk matches on both title and body. It costs nothing to compute and
was the largest single improvement in the comparison.

**2. Semantic (KMeans) came last on every query** — 0.6197, 0.6257, 0.7447. Not a bug: 8 clusters
over 17,706 characters gives chunks averaging **2,186 characters** (largest 4,367). Each mixes
several topics, so its vector is an average and matches no single question sharply. Clustering also
pulls sentences from across the document, diluting it further. More clusters would help, but that
converges toward sentence chunking at many times the cost — it was also the only method with a real
build cost, 9.04 s against ~0.00 s. The most sophisticated method was the worst performer on a short,
well-structured document.

**3. Fixed-size still returns unusable text.** Its top hit for query 1 begins:

```
ed-size chunking  Take a free-form 500-word text (Wikipedia, a news article…). Implement the
function fixed_chunk(tex...
```

The chunk starts mid-word (`fix|ed-size`). Score 0.7273, barely below the winner, but that snippet
cannot be shown to a user.

## Document summary — map-reduce over 33 recursive chunks

33 chunk summaries, merged in batches of 8, then merged once more:

```
Final document summary:
The course teaches two key NLP techniques, chunking and vectorization, to transform raw text into a
usable format, focusing on solving problems like dealing with long text and converting text into
numerical representations. It covers various methods for chunking, including sentence splitting,
recursive chunking, and semantic chunking, and explains how to generate vectors, store them, and
perform similarity searches using tools like Ollama and ChromaDB.
```

Accurate, including details that appear only deep in the document (ChromaDB, the pipeline shape).

---

# Test suite

```
$ python3 tests.py
Ran 23 tests in 0.001s

OK
```

23 tests covering the splitting logic and the cosine maths. No model or Ollama required — splitting
text and computing a cosine are pure functions.

---

# Two failures found while building, and their fixes

Both are recorded because in each case **nothing errored** — the output was fluent, plausible, and
wrong, which is the hardest kind of failure to notice.

## 1. A merge prompt silently dropped four of five points

The hierarchical global summary first came back as only:

> "Artificial intelligence is profoundly transforming modern businesses."

The ethics content had vanished, making the strategy that should perform best look worst.

Cause was the prompt, not the chunking. `summarise()` appended the rule *"one short sentence"* to
every call, including merges. Handed five bullet points and told to produce one short sentence,
`llama3.2:3b` returned the first bullet and dropped the rest.

Fixed by giving merges their own rule — *"cover EVERY point listed below, none omitted, one or two
sentences"* — behind a `merge=True` flag. The Level 3 summary above is the output after the fix.

## 2. The same risk at 33 chunks, handled by batching

Part 2 merges 33 partial summaries. One prompt with 33 bullets would hit the same truncation, so
merging happens in batches of 8 and then once over the batch summaries. That is why the final
document summary retains details from the end of the document rather than stopping early.

---

Author: **Oussama Ezitouni**
