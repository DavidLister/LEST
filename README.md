# LEST

**L**ocal **E**mbedding **S**earch **T**est — semantic search over a directory
of documents, using embedding models hosted locally with
[Ollama](https://ollama.com). Local-first and scriptable: index overnight from
cron, query from a shell, pipe into rofi/fzf.

```console
$ lest index ~/Zotero --model qwen3-embedding:8b
$ lest search ~/Zotero "perovskite solar cell degradation"
0.7231	Thermal degradation pathways in perovskite photovoltaics	/home/you/Zotero/storage/AB12CD34/paper.pdf
0.6874	Interface engineering for stable perovskite cells	/home/you/Zotero/storage/EF56GH78/main.pdf;/home/you/Zotero/storage/EF56GH78/si.pdf
...
```

## How it works

`lest index <dir>` walks a directory, extracts text (PDF via PyMuPDF, plus
`.txt`/`.md`), chunks it, embeds each chunk with a local Ollama model, and
stores everything in one SQLite database per directory (vectors via
[sqlite-vec](https://github.com/asg017/sqlite-vec)). If the directory is a
**Zotero data directory** (detected by `zotero.sqlite`), items are indexed
with their real titles, authors, year, and DOI, and each result lists all of
the item's PDF attachments. Re-runs are incremental: only new or changed
files are re-embedded, deleted ones are pruned — safe to run nightly.

`lest search <dir> <query>` embeds the query, finds the nearest chunks, and
aggregates chunk scores into a ranked list of documents. Output is TSV
(`score<TAB>title<TAB>path[;path…]`) on stdout for easy scripting; `--json`
emits JSON lines with full metadata and the best-matching chunk.

## Usage

```console
lest index <dir> [--model NAME] [--chunker NAME] [--source auto|zotero|plaindir] [-v]
lest search <dir> <query> [-n N] [--agg STRATEGY] [--json]
lest status <dir>
```

- The **first** index of a directory requires `--model` (e.g.
  `qwen3-embedding:8b`, `nomic-embed-text`). The model is recorded in that
  directory's database and reused; mixing models in one index is refused.
- `--agg` picks how chunk scores roll up into a document score:
  `max` (default, best chunk wins), `topk:K` (mean of K best),
  `softmax:T` (smooth max with temperature T), `decay` (s₁ + s₂/2 + s₃/4 …),
  `count:T` (number of chunks above similarity T).
- Environment: `OLLAMA_HOST` (default `http://localhost:11434`),
  `LEST_DATA_DIR` (where the databases live; default `./data`, gitignored).
- Exit codes: 0 ok · 1 usage/lookup errors · 2 environment errors (Ollama
  unreachable, model not pulled, Zotero DB locked).

## Install

With nix (flakes):

```console
nix run github:DavidLister/LEST -- --help   # or `nix run .` from a checkout
nix develop                                  # dev shell with all deps
```

Without nix:

```console
pip install -e .        # needs Python >= 3.11
```

Either way you need an [Ollama](https://ollama.com) server with an embedding
model pulled, e.g. `ollama pull qwen3-embedding:0.6b`.

## Development

```console
nix develop -c pytest    # test suite (no Ollama/GPU needed — fake embedder)
nix develop -c ruff check .
```

Chunkers, embedding backends, and ranking strategies are small registries
(`lest/chunkers/`, `lest/embedders/`, `lest/ranking.py`) intended to be easy
to extend — see `DESIGN.md` for the full design, including the planned
LLM-assisted chunker.

## License

MIT.
