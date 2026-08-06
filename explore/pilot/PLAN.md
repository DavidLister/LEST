# Pilot: LLM-assisted chunking & multi-view summaries

One overnight run on ~10 real papers to settle, with data, the design questions
for LEST's gemma4 features before building them into the package. Everything
here is throwaway-quality experiment code (`explore/pilot/`), not product code.
Results land in `data/pilot/` (gitignored — contains paper text).

Infrastructure: gemma4:12B on the ROCm Ollama (RX 9070 XT, `:11435`),
qwen3-embedding:8b on the CUDA Ollama (A2000, `:11434`). Runner checkpoints
results to JSONL after every model call, so it resumes if interrupted.

## Paper selection

10 papers from the freshly indexed Zotero library, chosen for spread:
3 short (≤6 pages), 4 medium, 2 long (≥15 pages), 1 figure-heavy.
Selected automatically by page count (pymupdf) from files with status `ok`;
list frozen into `data/pilot/papers.json` so all experiments use the same set.

## Experiments

### E1 — Outline prompt refinement (the production-path candidate)

Three prompt variants for the sections+ideas outline call (text-only input):

- **P1 minimal**: "segment into sections and sequential ideas", JSON schema
  enforced via Ollama structured outputs.
- **P2 guided**: P1 + explicit granularity ("2–6 ideas per page") + anchor
  discipline ("first_words = 5–8 words copied verbatim").
- **P3 rich**: P2 + per-section 1–2 sentence context blurb.

Metrics per (paper, prompt): JSON validity; **anchor hit rate** (fraction of
first_words found verbatim / after whitespace-normalization in the source
text); ideas per page; wall time; output tokens. The winner becomes the
default prompt.

### E2 — Verbatim rewrite vs boundaries (David's original proposal, tested fairly)

Full "rewrite the paper unchanged, chunked into JSON ideas" prompt on 5 of the
10 papers (the slow arm). Metrics: wall time & output tokens vs E1;
**wording fidelity** (per-chunk normalized edit distance to best-matching
source span); **boundary agreement** with P2's cut points (± 1 sentence).
Decides whether retyping ever beats anchor-echo.

### E3 — Multimodal input: does seeing pages help?

P2 re-run with text + rastered pages (150 dpi, capped at 30 pages) on all 10
papers. Metrics: boundary agreement & anchor hit rate vs text-only P2;
**figure capture** (figures described vs figures actually present — hand-count
ground truth for the 10 papers); input token cost; VRAM (`ollama ps`) and
time. Also answers whether figure descriptions can ride along in the one
per-paper call.

### E4 — Reflection views: parallel vectors or real signal?

Second call per paper (same conversation → KV cache warm) generating the four
views: notable/memorable, main ideas, methods & techniques, problem/why-cite.

- **Redundancy test**: embed all views + the paper's abstract with qwen3-8b;
  report the within-paper pairwise cosine matrix, averaged over papers.
  Views pairing ≥ ~0.9 are duplicates; drop or merge them.
- **Utility test**: gemma4 (fresh context, high temperature) writes 3 vague
  "how I'd half-remember this paper" queries per paper. Retrieval is then run
  over the pilot corpus in three conditions: body chunks only / body + view
  chunks / views only. Metric: mean reciprocal rank of the correct paper per
  condition. Views must beat body-only to earn their place.

### E5 — Performance envelope on the 9070 XT

Recorded throughout: generation tok/s per call type; whether num_ctx=65536
loads cleanly or spills (fallback measurement at 32k); page-image token cost.
Turns the "15 min vs 30 s per paper" estimates into measured numbers.

### E6 — Freeform tag sweep (whole library, not just 10 papers)

gemma4 assigns unconstrained topic tags from title+abstract for every
bibliographic item (~1160 items ≈ 1.5–2 h on the 9070 XT, runs after E1–E5).
Output: tag → count table. Morning step: Claude reviews and pares the list
into a proposed constrained vocabulary (`data/pilot/vocab-proposed.txt`) for
David's final edit — per the agreed generate-then-curate flow.

## Morning deliverables

- `data/pilot/results.jsonl` — every raw call, timing, and metric
- `data/pilot/report.md` — tables for E1–E5, the E4 similarity matrix and MRR
  comparison, E6 tag table + proposed vocabulary, and a written
  recommendation for each design question (chunker default, multimodal or
  not, which views to keep, rewrite verdict)
