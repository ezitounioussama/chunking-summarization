# Notes — file map, setup, and which method to use

The measured output lives in [RESULTS.md](RESULTS.md). This file holds the practical detail:
what each file is, how to install the Part 2 dependencies, and how the five chunking methods
compare once you have run them.

## Files

| File | Contents |
|---|---|
| `run_exercise.py` | Part 1 driver — runs Steps 1–6 and prints everything |
| `chunking.py` | The three Part 1 strategies, no model calls |
| `embeddings.py` | Embedding store, cosine similarity, search |
| `ollama_client.py` | Thin wrapper over the local Ollama HTTP API |
| `docx_loader.py` | Reads `.docx` **keeping its structure** — headings, list items, sections |
| `chunkers.py` | The course's five methods, with spaCy for sentence boundaries |
| `run_on_document.py` | The full pipeline on a real document |
| `tests.py` | 23 tests for the splitting and the maths (no model needed) |
| `docs/output.txt` | Raw terminal log of the Part 1 run (219 lines) |
| `docs/output_document.txt` | Raw terminal log of the Part 2 run (187 lines) |

## Setup

Part 1 needs nothing installed — it is standard library only, so `python3 run_exercise.py` just
works. Part 2 needs spaCy and scikit-learn, which go in a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install spacy scikit-learn numpy
.venv/bin/python -m spacy download en_core_web_sm
```

Then run it **through the venv**, not with the system python:

```bash
.venv/bin/python run_on_document.py                      # defaults to the course doc
.venv/bin/python run_on_document.py path/to/other.docx   # any .docx
```

Or activate the venv once and drop the prefix:

```bash
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python run_on_document.py
```

Two notes for Arch Linux:

- There is no standalone `pip` command. Use `python3 -m pip`, and inside a venv
  `.venv/bin/python -m pip`.
- Installing into system Python is blocked (`externally-managed-environment`) and would be the
  wrong move anyway — the venv keeps this project's packages out of the system.

`run_on_document.py` checks for the three packages before it does anything and prints these
commands if they are missing, rather than failing with a bare `ModuleNotFoundError`.

## Part 1 — the three strategies compared

| | Fixed-size | Semantic | Hierarchical |
|---|---|---|---|
| Splits on | character count | sentence boundaries | paragraphs, then sentences, then clauses |
| Respects meaning | no | yes | yes |
| Chunk count here | 6 | 3 | 5 |
| Cost to compute | lowest | low | moderate (recursive) |
| Retrieved text usable alone | **no** | **yes** | mostly |
| Summary completeness | drifts, loses detail | concise, loses specifics | most complete |
| Good for | uniform-size batching, hard token limits | Q&A and retrieval over prose | long structured documents |

Fixed-size is defensible when chunks must be a predictable size — a hard token budget, or a model
with a small context window — and it is the cheapest to compute. Semantic is the default for
retrieval over prose. Hierarchical earns its extra complexity on long structured documents, where
the parent level gives you a summary layer to search before drilling into the leaves.

The hierarchical recursion tries the strongest boundary first (paragraphs, then sentences, then
commas) and only goes smaller when a piece is still too long, so a cut is only ever made where the
language already has a seam.

## Part 2 — the course's five methods on a real document

Two additions over Part 1, both taken from the course:

**Method 1 now has overlap.** Each chunk repeats the last 60 characters of the previous one, so an
idea straddling a boundary survives whole somewhere. The course suggests 10–15% of chunk size;
60/500 is 12%.

**Method 4 is real semantic chunking.** Part 1 called sentence splitting "semantic", which was
loose. The course's Method 4 embeds every sentence and clusters the vectors with **KMeans**, so
sentences about the same topic group together *even if they are pages apart* — the only method here
that ignores document order. It cost 9.04 s against ~0.00 s for the others, because it needs one
embedding per sentence before any chunking happens.

**Method 5 needs the document, not a string.** This is why `docx_loader.py` parses the XML instead
of flattening to text. It keeps headings as section labels rather than chunks of their own, keeps
consecutive list items together, and **prepends the section heading to every chunk** — so a chunk
retrieved alone still says what it is about.

## Which method to actually use

| If your document is… | Use |
|---|---|
| Structured — headings, sections, lists (`.docx`, Markdown, HTML) | **Structure-aware**, with the heading prefixed to each chunk |
| Unstructured prose with no headings | **Recursive** — the course's default, and a good one |
| Needs uniform chunk sizes for a hard token budget | **Fixed-size with overlap** — accept that boundaries cut words |
| Long, sprawling, poorly organised, where one topic recurs in many places | **Semantic (KMeans)** — the only method that gathers scattered mentions |
| Short, or you need a cheap sensible baseline | **Sentence (spaCy)** |

For the course document, structure-aware plus recursive is the right pairing, and the 9 seconds
spent on KMeans bought nothing.

## Why the summaries merge in batches of 8

Part 1 hit a silent failure where a merge prompt handed many bullet points returned only the first
(written up in [RESULTS.md](RESULTS.md)). With 33 partial summaries that risk is much higher, so
merging happens in batches of 8, then once more over the batch summaries. The failure mode is a
fluent, plausible, incomplete answer — nothing errors, which is what makes it worth guarding
against.
