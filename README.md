# Text Summarization Using Three Chunking Strategies

Long text does not fit in a model's context window, so it has to be cut up — and cutting it in
the wrong place destroys the meaning you were trying to keep. This project splits the same text
several ways, embeds every chunk, and searches them, so the cost of a bad cut is something you
can see rather than argue about. Part 1 is the exercise itself: three strategies on a short
string, six steps. Part 2 answers two follow-ups — use spaCy instead of a regex for sentence
boundaries, and run on a real 17,706-character `.docx` instead of a hard-coded string.

The short version of what it found: a good similarity score does not mean a usable chunk.
Fixed-size chunking scored within 0.01 of the winner and handed back text starting mid-word.

Re-run on `qwen3:8b` after being built on `llama3.2:3b`, one finding softened and one held. The
summary differences mostly vanished — the stronger model reassembles what bad chunking broke, and
says *"is cut mid-sentence"* instead of inventing meaning for a fragment. The retrieval finding did
not move an inch, because it is mechanical: the top-ranked fixed-size chunk still ends `transpa`.

Everything runs locally on Ollama — `qwen3:8b` for the summaries, `nomic-embed-text` for the
embeddings — so there is no API key anywhere.

```bash
ollama serve && ollama pull qwen3:8b && ollama pull nomic-embed-text

python3 run_exercise.py                 # Part 1 — standard library only
python3 tests.py                        # 23 tests, no Ollama needed
.venv/bin/python run_on_document.py     # Part 2 — venv setup in NOTES.md
```

## Also in this repo

- **[RESULTS.md](RESULTS.md)** — every captured run log: chunk listings, summaries, embedding
  dimensions, cosine scores, and the two failures found while building
- **[NOTES.md](NOTES.md)** — file map, Part 2 setup, and how the five methods compare
- [`docs/output.txt`](docs/output.txt) and [`docs/output_document.txt`](docs/output_document.txt)
  — raw terminal logs, unedited

---

Author: **Oussama Ezitouni**
