# LEST — Design

LEST (Local Embedding Search Test) is a CLI tool for semantic search over a
directory of documents, using embedding models hosted locally with
[Ollama](https://ollama.com). It is built for local-first, scriptable use:
index overnight via cron, query from a shell or a rofi wrapper.

## Goals

- **Local-first.** No cloud calls; models run on your own hardware via Ollama.
- **Scriptable.** Plain CLI with machine-parseable output. Composes with
  cron, rofi, awk, fzf.
- **Source-aware.** A directory is the unit of indexing. Known layouts
  (Zotero) are auto-detected and yield rich per-paper metadata; anything else
  is treated as a plain tree of files.
- **Modular where it matters.** Chunking, embedding backends, and
  chunk→document ranking are strategy interfaces, because those are the axes
  of experimentation.
- **Boring where it doesn't.** SQLite for storage, flags + env vars for
  configuration, no config files, no services.

## Non-goals (for now)

- No web UI, no daemon, no HTTP API.
- No cross-directory search (each indexed directory is independent).
- No hybrid lexical (BM25) search — embedding-only for the MVP.
- No OCR; PDFs without a text layer are skipped (and reported).

## CLI

```
lest index <dir> [--model NAME] [--chunker NAME] [--verbose]
lest search <dir> <query> [-n N] [--agg STRATEGY] [--json]
lest status <dir>          # what's indexed, with which model/chunker, counts
```

- `lest index <dir>` — walk/sync the directory into its database.
  - **First run requires `--model`** (e.g. `--model qwen3-embedding:8b`).
    The model (and its embedding dimension) is recorded in the DB; later
    runs reuse it. Passing a *different* `--model` later is an error with a
    hint to delete the DB file and rebuild — vectors from different models
    must never mix.
  - Re-runs are **incremental**: unchanged files are skipped, new and
    modified files are (re-)extracted/chunked/embedded, entries for deleted
    files are removed. Safe to run nightly from cron.
- `lest search <dir> <query>` — embed the query, score chunks, aggregate to
  documents, print the top N (default 10).
- Exit codes: 0 ok, 1 usage/lookup errors (e.g. directory never indexed),
  2 environment errors (Ollama unreachable, model missing).

### Output format

Default output is TSV on stdout, one document per line, tab-separated:

```
score \t title \t path[;path…]
```

- `score` — aggregated similarity, higher is better, formatted `%.4f`.
- `title` — document title (Zotero metadata, else filename). Tabs/newlines
  in titles are replaced with spaces so the format stays line-safe.
- `path[;…]` — absolute path(s) of the document's file(s); multiple
  attachments are `;`-joined. A rofi wrapper can `cut -f3`, pick, and hand
  the path to `xdg-open`.

`--json` switches to JSON Lines with full detail per document: score, title,
metadata (authors, year, DOI when known), attachments, and the best-matching
chunk's text (useful as a snippet in richer UIs).

Progress/log messages go to stderr, never stdout.

### Configuration

Flags with sane defaults, plus environment variables for the environment-y
bits:

| Variable          | Meaning                          | Default                  |
|-------------------|----------------------------------|--------------------------|
| `OLLAMA_HOST`     | Ollama server for embeddings     | `http://localhost:11434` |
| `LEST_DATA_DIR`   | where databases live             | `./data` (repo checkout) |

Per-database facts (model, dimension, chunker used) live in the database
itself, not in config. A future LLM chunker gets its own host variable
(`LEST_CHUNKER_HOST`) so chunking and embedding can run on different
GPUs/servers (e.g. ROCm instance on one card, CUDA on another).

## Architecture

```
lest/
├── cli.py            # Typer app; thin — parses args, calls pipeline/query
├── pipeline.py       # index orchestration: source → extract → chunk → embed → store
├── query.py          # search orchestration: embed query → knn → aggregate → format
├── sources/
│   ├── base.py       # Source protocol + Document/Attachment dataclasses
│   ├── plaindir.py   # walk a directory tree; one Document per file
│   └── zotero.py     # read zotero.sqlite; one Document per item w/ attachments
├── extract.py        # file → plain text (pymupdf for PDF; .txt/.md read directly)
├── chunkers/
│   ├── base.py       # Chunker protocol + registry
│   └── paragraph.py  # default: paragraph-based with merge/split bounds
├── embedders/
│   ├── base.py       # Embedder protocol + registry
│   └── ollama.py     # Ollama /api/embed client (batched)
├── store.py          # SQLite + sqlite-vec: schema, sync bookkeeping, knn
├── ranking.py        # chunk→document aggregation strategy registry
└── output.py         # TSV / JSON-lines formatting
```

Data flows through plain dataclasses; each stage is independently testable.

### Sources

```python
class Source(Protocol):
    def documents(self) -> Iterator[Document]: ...

@dataclass
class Document:
    key: str                  # stable ID within the source (Zotero item key, or relpath)
    title: str
    attachments: list[Attachment]   # files to extract text from
    meta: dict                # authors, year, DOI, tags… (may be empty)

@dataclass
class Attachment:
    path: Path
    fingerprint: str          # mtime+size (upgradeable to content hash)
```

- **Detection:** if `<dir>/zotero.sqlite` exists → `ZoteroSource`, else
  `PlainDirSource`. Overridable with `--source {auto,zotero,plaindir}`.
- **`PlainDirSource`** walks the tree for supported extensions
  (`.pdf`, `.txt`, `.md`); each file is one Document (key = relative path,
  title = filename).
- **`ZoteroSource`** opens `zotero.sqlite` **read-only** (SQLite URI
  `mode=ro`, `immutable` off — Zotero may be running) and joins `items`,
  `itemData` (title etc.), `itemCreators`, and `itemAttachments` to produce
  one Document per bibliographic item, with all of its stored PDF
  attachments (paths resolved into `storage/<KEY>/`). Notes, snapshots, and
  link-only attachments are ignored. One paper with three PDFs is *one*
  result with three paths.

### Extraction

`extract.py` maps a file to plain text: PyMuPDF for PDFs (page texts joined
with blank lines), direct read for `.txt`/`.md`. Files yielding no text
(scanned PDFs without OCR) are recorded as skipped and listed in
`lest status` / verbose output, so failures are visible rather than silent.

### Chunking

```python
class Chunker(Protocol):
    name: str
    def chunk(self, text: str) -> list[str]: ...
```

Registry keyed by name; selected at index time with `--chunker`, recorded in
the DB (a DB has exactly one chunker, like one model).

- **`paragraph` (default, MVP):** split on blank lines; merge consecutive
  paragraphs until a minimum size (~700 chars) is reached; hard-split any
  block over a maximum (~2000 chars) at sentence boundaries. Handles the
  ragged paragraphing that PDF extraction produces while keeping chunks
  roughly semantically coherent.
- **`llm` (planned, not in MVP):** send the document text to a local chat
  model (e.g. gemma via Ollama) that returns logical section boundaries,
  identified by echoing the first words of each section (LLMs can't be
  trusted with byte offsets); boundaries are located by string search, and
  anything that fails to validate falls back to `paragraph`. Runs on a
  separate Ollama host so a second GPU can chunk paper N+1 while the first
  embeds paper N.

### Embedding

```python
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...   # documents
    def embed_query(self, text: str) -> list[float]: ...
```

- **`OllamaEmbedder`** (MVP) uses the native `/api/embed` endpoint via the
  `ollama` Python client, batching chunk requests. The interface is the seam
  for future backends (OpenAI-compatible endpoints, sentence-transformers).
- `embed_query` exists because some models (notably the Qwen3-Embedding
  family) want an instruction prefix on *queries* but not documents. The
  Ollama embedder applies a model-appropriate query prefix when one is
  known, else no prefix. Kept as a small internal table, adjustable as we
  learn what the model needs.
- Embedding dimension is taken from the first response and recorded in the
  DB; every subsequent embed is validated against it.

### Storage

One SQLite database per indexed directory, in `LEST_DATA_DIR`:

```
data/<dirname>-<8-char sha256 of absolute path>.db     e.g. data/Zotero-3f9a1c2b.db
```

Independent databases keep experiments cheap: delete the file, re-index.
**`data/` must be added to `.gitignore`** — it contains personal library
content and must never reach the public repo.

Schema (sqlite-vec loaded as an extension):

```sql
CREATE TABLE meta      (key TEXT PRIMARY KEY, value TEXT);
-- source_dir, source_type, model, dim, chunker, created, last_indexed

CREATE TABLE documents (id INTEGER PRIMARY KEY, key TEXT UNIQUE,
                        title TEXT, meta_json TEXT);

CREATE TABLE files     (id INTEGER PRIMARY KEY,
                        document_id INTEGER REFERENCES documents(id),
                        path TEXT UNIQUE, fingerprint TEXT,
                        status TEXT);        -- ok | no_text | error

CREATE TABLE chunks    (id INTEGER PRIMARY KEY,
                        file_id INTEGER REFERENCES files(id),
                        seq INTEGER, text TEXT);

CREATE VIRTUAL TABLE chunk_vectors USING vec0(
                        chunk_id INTEGER PRIMARY KEY,
                        embedding FLOAT[<dim>]);       -- dim fixed at creation
```

- **Incremental sync:** for each Document from the source, compare each
  file's `fingerprint` against `files`; unchanged → skip, changed/new →
  delete old chunks+vectors, re-extract/chunk/embed. Files in the DB but no
  longer reported by the source are removed along with orphaned documents.
- **Search:** `vec0` KNN over `chunk_vectors` (cosine distance) with an
  over-fetched k (e.g. `10 × n` chunks), joined back through
  `chunks → files → documents` for aggregation in Python.

### Ranking (chunk → document aggregation)

A registry of pure functions `scores: list[float] -> float` over a
document's chunk similarities, selected per-query with `--agg` (string
parsed as `name` or `name:param`):

| Strategy      | Score                                   | Character                          |
|---------------|-----------------------------------------|------------------------------------|
| `max` (default) | best chunk                            | needle queries, length-immune      |
| `topk:K`      | mean of K best                          | sustained relevance                |
| `softmax:T`   | temperature-weighted smooth max         | tunable between max and mean       |
| `decay`       | s₁ + s₂/2 + s₃/4 + …                    | parameter-free middle ground       |
| `count:T`     | number of chunks with similarity ≥ T    | pure topical coverage              |

All are a few lines; all ship in the MVP so ranking behavior can be compared
on real queries.

## Testing

- `pytest` unit tests per stage: chunker behavior on fixture texts, ranking
  math, sync logic against a temp DB, Zotero adapter against a tiny
  hand-built `zotero.sqlite` fixture, output escaping.
- Embedder tests use a fake deterministic embedder; no test requires Ollama
  or GPU. An optional integration test (marked, skipped by default) runs
  end-to-end against a live Ollama.
- A small fixture corpus (a few public-domain text files) lives in
  `tests/fixtures/` so the full pipeline is exercised in CI without any
  personal data.

## Packaging / environment

- **Nix flake** is the primary environment (NixOS host). All dependencies
  come from nixpkgs — verified available: `pymupdf`, `sqlite-vec`, `typer`,
  `ollama`, `numpy`, `pytest`, `ruff`. The flake provides:
  - `devShells.default` — `python3.withPackages([...])` + ruff, for hacking
    and running tests;
  - `packages.default` — `buildPythonApplication` for the `lest` binary
    (`nix run .`).
- **`pyproject.toml`** kept standard (PEP 621, `[project.scripts] lest = …`)
  so non-Nix users can `pip install -e .` from the public repo; it is also
  what the Nix build consumes.
- `ruff` for lint + format; CI (GitHub Actions) runs ruff + pytest via the
  flake.

## Milestones

1. **Skeleton** — flake, pyproject, package layout, CLI stubs, CI green.
2. **MVP pipeline** — plaindir source, paragraph chunker, Ollama embedder,
   sqlite-vec store, `max` ranking, TSV output. End-to-end on a folder of
   PDFs.
3. **Zotero source** — auto-detection, metadata, multi-attachment documents.
4. **Ranking menu + `--json` + `status`** — the full experimentation surface.
5. **Later** — LLM chunker, query-prefix tuning, rofi wrapper script in
   `contrib/`, hybrid BM25, cross-directory search.
