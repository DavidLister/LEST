#!/usr/bin/env python3
"""Experimental indexer: the pilot-winning LLM scheme, into a parallel DB.

Scheme (per data/pilot/report.md recommendations):
  1. text-only gemma4 outline (granularity-disciplined, section contexts)
  2. anchors = candidate cut points; mechanical merge to 700-2000 chars
  3. separate image-only gemma4 call for figure descriptions
  4. four reflection views as summary chunks
  5. tags constrained to data/pilot/vocab-proposed.txt (stored in doc meta)

Writes into data/experimental/ so A/B against the baseline is:
    LEST_DATA_DIR=data/experimental nix run . -- search ~/Zotero "..."
Checkpointing: a paper whose file is already in the experimental DB with a
matching fingerprint is skipped; safe to re-run.

Usage: python explore/pilot/exp_index.py [--limit N]   (default 100)
"""

import argparse
import json
import os
import random
import re
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "explore" / "pilot"))

os.environ["LEST_DATA_DIR"] = str(REPO / "data" / "experimental")

from pilot import (  # noqa: E402
    BIG_CTX,
    EMB_MODEL,
    GRANULARITY_FIX,
    OUTLINE_SCHEMA,
    P2B,
    PILOT,
    SMALL_CTX,
    VIEWS,
    VIEWS_PROMPT,
    VIEWS_SCHEMA,
    abstract_for,
    anchor_metrics,
    embedder,
    llm_call,
    paper_text,
    production_db,
    render_pages,
)
from pilot import (
    TAGS_SCHEMA as EXP_TAGS_SCHEMA,  # noqa: E402
)

from lest.chunkers.paragraph import MAX_CHARS, MIN_CHARS, ParagraphChunker  # noqa: E402
from lest.sources.base import fingerprint  # noqa: E402
from lest.store import Store, db_path_for  # noqa: E402

ZOTERO = Path.home() / "Zotero"
CHUNKER_NAME = "llm-v1"

OUTLINE_PROMPT = P2B.replace(
    "For every section give its title (use the paper's own headings where they exist).",
    "For every section give its title (use the paper's own headings where they "
    "exist) and a context field: 1-2 sentences saying what the section covers "
    "in the context of this specific paper (mention its actual subject, "
    "materials, or methods).",
)

FIGURES_SCHEMA = {
    "type": "object",
    "properties": {
        "figures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer"},
                    "description": {"type": "string"},
                },
                "required": ["page", "description"],
            },
        }
    },
    "required": ["figures"],
}

FIGURES_PROMPT = """These are the pages of the scientific paper titled
"{title}". Describe every figure you see (skip logos, headers, and tables of
pure numbers). For each figure give its 1-based page number and a description
of what it shows: axes, quantities, trends, and what a reader should take from
it. Mention the figure number from its caption when visible."""

TAGS_PROMPT = """Assign 1-5 topic tags to this paper. You MUST choose only from
this exact list (copy tags verbatim, lowercase):

{vocab}

TITLE: {title}
ABSTRACT: {abstract}"""


def vocab() -> list[str]:
    tags = []
    for line in (PILOT / "vocab-proposed.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            tags.append(line)
    return tags


def select_papers(limit: int) -> list[dict]:
    pilot_keys = {p["key"] for p in json.loads((PILOT / "papers.json").read_text())}
    conn = sqlite3.connect(f"file:{production_db()}?mode=ro", uri=True)
    rows = conn.execute(
        """SELECT d.key, d.title, d.meta_json, f.path,
                  (SELECT count(*) FROM chunks c WHERE c.file_id = f.id) AS n
           FROM files f JOIN documents d ON d.id = f.document_id
           WHERE f.status = 'ok' AND d.meta_json LIKE '%creators%'
           ORDER BY n DESC"""
    ).fetchall()
    conn.close()
    # one file per document (the chunk-richest), pilot papers first, then random
    seen, entries = set(), []
    for key, title, meta_json, path, _ in rows:
        if key in seen or not Path(path).exists():
            continue
        seen.add(key)
        entries.append({"key": key, "title": title, "meta": json.loads(meta_json),
                        "path": path})
    pilot_entries = [e for e in entries if e["key"] in pilot_keys]
    rest = [e for e in entries if e["key"] not in pilot_keys]
    random.seed(20260806)
    random.shuffle(rest)
    return (pilot_entries + rest)[:limit]


def cut_text(text: str, outline: dict) -> list[str]:
    """Slice at validated LLM anchors, then merge/split to the size envelope."""
    sections = []
    for section in outline.get("sections", []):
        anchors = [i["first_words"] for i in section.get("ideas", [])]
        m = anchor_metrics(anchors, text)
        sections.append({
            "title": section.get("title", ""),
            "context": section.get("context", ""),
            "positions": m["positions"],
        })
    # global, strictly increasing cut list with section attribution
    cuts = []  # (position, section_index)
    prev = -1
    for idx, s in enumerate(sections):
        for pos in s["positions"]:
            if pos > prev:
                cuts.append((pos, idx))
                prev = pos
    if not cuts or cuts[0][0] > 0:
        cuts.insert(0, (0, 0))

    chunks = []
    buffer, buffer_section = "", 0
    for (pos, idx), nxt in zip(cuts, cuts[1:] + [(len(text), None)], strict=True):
        piece = text[pos:nxt[0]].strip()
        if not piece:
            continue
        if not buffer:
            buffer_section = idx
        buffer = f"{buffer} {piece}".strip() if buffer else piece
        if len(buffer) >= MIN_CHARS:
            chunks.append((buffer_section, buffer))
            buffer = ""
    if buffer:
        if chunks and len(buffer) < MIN_CHARS // 2:
            idx, prev_text = chunks[-1]
            chunks[-1] = (idx, prev_text + " " + buffer)
        else:
            chunks.append((buffer_section, buffer))

    out = []
    for idx, chunk in chunks:
        s = sections[idx] if idx < len(sections) else {"title": "", "context": ""}
        prefix = s["title"]
        if s["context"]:
            prefix = f"{prefix} — {s['context']}" if prefix else s["context"]
        header = f"[{prefix}] " if prefix else ""
        for part in ParagraphChunker._split_long(chunk):
            out.append(header + part)
    return out


def process(entry: dict, emb, vocab_tags: list[str], store: Store) -> dict:
    text = paper_text(entry["path"])
    stats = {"key": entry["key"]}

    outline, timing = llm_call(OUTLINE_PROMPT.format(text=text), OUTLINE_SCHEMA,
                               num_predict=16384)
    stats["outline_s"] = timing["seconds"]
    if outline:
        chunks = cut_text(text, outline)
        stats["chunker"] = "llm"
    else:  # full fallback
        chunks = ParagraphChunker().chunk(text)
        stats["chunker"] = "paragraph-fallback"

    figures, timing = llm_call(
        FIGURES_PROMPT.format(title=entry["title"]), FIGURES_SCHEMA,
        images=render_pages(entry["path"]), num_predict=4096)
    stats["figures_s"] = timing["seconds"]
    for fig in (figures or {}).get("figures", []):
        chunks.append(f"[figure p.{fig['page']}] {fig['description']}")
    stats["n_figures"] = len((figures or {}).get("figures", []))

    views, timing = llm_call(VIEWS_PROMPT.format(text=text), VIEWS_SCHEMA,
                             num_predict=2048)
    stats["views_s"] = timing["seconds"]
    for view in VIEWS:
        if views and views.get(view):
            chunks.append(f"[{view}] {views[view]}")

    tags, _ = llm_call(
        TAGS_PROMPT.format(vocab="\n".join(vocab_tags), title=entry["title"],
                           abstract=abstract_for(entry["key"]) or "(none)"),
        EXP_TAGS_SCHEMA, num_ctx=SMALL_CTX, num_predict=200)
    clean_tags = [t for t in (tags or {}).get("tags", [])
                  if t.strip().lower() in {v.lower() for v in vocab_tags}]
    stats["tags"] = clean_tags

    meta = dict(entry["meta"])
    meta["gen_tags"] = clean_tags
    if views:
        meta["summary"] = views.get("main_ideas", "")

    vectors = emb.embed(chunks)
    store.ensure_vector_table(len(vectors[0]))
    doc_id = store.upsert_document(entry["key"], entry["title"], meta)
    store.replace_file(doc_id, entry["path"], fingerprint(Path(entry["path"])),
                       "ok", chunks, vectors)
    stats["n_chunks"] = len(chunks)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    store = Store(db_path_for(ZOTERO), create=True)
    store.set_meta("model", EMB_MODEL)
    store.set_meta("embedder", "ollama")
    store.set_meta("chunker", CHUNKER_NAME)
    store.set_meta("source_dir", str(ZOTERO))
    store.set_meta("source_type", "ZoteroSource(experimental)")
    known = store.file_fingerprints()

    emb = embedder()
    vocab_tags = vocab()
    entries = select_papers(args.limit)
    print(f"experimental index: {len(entries)} papers -> {store.db_path}",
          flush=True)
    for i, entry in enumerate(entries, 1):
        if known.get(entry["path"]) == fingerprint(Path(entry["path"])):
            continue
        try:
            stats = process(entry, emb, vocab_tags, store)
        except Exception as exc:  # keep the overnight run alive
            print(f"[{i}/{len(entries)}] FAILED {entry['key']}: {exc}", flush=True)
            continue
        print(f"[{i}/{len(entries)}] {stats}", flush=True)
    store.stamp_last_indexed()
    store.close()


if __name__ == "__main__":
    main()
