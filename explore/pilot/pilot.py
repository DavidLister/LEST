#!/usr/bin/env python3
"""Overnight pilot runner — see PLAN.md. Experiment code, not product code.

Usage (from repo root, inside `nix develop`):
    python explore/pilot/pilot.py select
    python explore/pilot/pilot.py e1 e2 e3 e4 e6      # any subset, any order
    python explore/pilot/pilot.py all                 # everything, resumable

Every model call is checkpointed to data/pilot/results.jsonl; re-running skips
completed work. All boundary math happens in whitespace-normalized text space.
"""

import argparse
import difflib
import glob
import json
import os
import random
import re
import sqlite3
import struct
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import ollama  # noqa: E402
import pymupdf  # noqa: E402

from lest.extract import extract  # noqa: E402

PILOT = REPO / "data" / "pilot"
RESULTS = PILOT / "results.jsonl"
PAPERS = PILOT / "papers.json"
ZOTERO = Path.home() / "Zotero"

LLM_HOST = os.environ.get("LEST_LLM_HOST", "http://localhost:11435")
EMB_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = "gemma4:12B"
EMB_MODEL = "qwen3-embedding:8b"
BIG_CTX = 65536
SMALL_CTX = 8192
MAX_TEXT_CHARS = 180_000  # ~48k tokens, leaves room for output in a 64k window
RENDER_DPI = 150
MAX_PAGES_RENDERED = 30
AGREE_TOL = 200  # chars: cut points closer than this count as "the same boundary"

llm = ollama.Client(host=LLM_HOST)

# ---------------------------------------------------------------- checkpointing


def done_keys() -> set[str]:
    keys = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            try:
                r = json.loads(line)
                keys.add(r["ckpt"])
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def record(ckpt: str, **fields):
    fields["ckpt"] = ckpt
    fields["ts"] = time.time()
    with RESULTS.open("a") as f:
        f.write(json.dumps(fields, ensure_ascii=False) + "\n")
    print(f"[done] {ckpt}", flush=True)


def load_results(stage: str) -> list[dict]:
    out = []
    for line in RESULTS.read_text().splitlines():
        try:
            r = json.loads(line)
            if r.get("stage") == stage:
                out.append(r)
        except json.JSONDecodeError:
            continue
    return out


# ---------------------------------------------------------------- text helpers


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def locate(anchor: str, norm_text: str, start: int = 0) -> int:
    """Position of anchor in normalized text, -1 if absent. Case-insensitive fallback."""
    a = normalize(anchor)
    if not a:
        return -1
    pos = norm_text.find(a, start)
    if pos < 0:
        pos = norm_text.lower().find(a.lower(), start)
    return pos


def anchor_metrics(anchors: list[str], norm_text: str) -> dict:
    positions, hits, order_ok = [], 0, 0
    prev = -1
    for a in anchors:
        pos = locate(a, norm_text)
        positions.append(pos)
        if pos >= 0:
            hits += 1
            if pos > prev:
                order_ok += 1
                prev = pos
    n = max(len(anchors), 1)
    return {
        "n_anchors": len(anchors),
        "anchor_hit_rate": hits / n,
        "order_ok_rate": order_ok / n,
        "positions": [p for p in positions if p >= 0],
    }


def agreement(cuts_a: list[int], cuts_b: list[int]) -> dict:
    """Symmetric boundary agreement between two cut-point sets (char positions)."""
    def matched(xs, ys):
        return sum(1 for x in xs if any(abs(x - y) <= AGREE_TOL for y in ys))

    pa = matched(cuts_a, cuts_b) / max(len(cuts_a), 1)
    pb = matched(cuts_b, cuts_a) / max(len(cuts_b), 1)
    f1 = 2 * pa * pb / max(pa + pb, 1e-9)
    return {"precision_a_in_b": pa, "precision_b_in_a": pb, "boundary_f1": f1}


def paper_text(path: str) -> str:
    text = normalize(extract(Path(path)) or "")
    return text[:MAX_TEXT_CHARS]


# ---------------------------------------------------------------- LLM plumbing


def llm_call(prompt: str, schema: dict, images: list[bytes] | None = None,
             num_ctx: int = BIG_CTX, temperature: float = 0.0,
             num_predict: int = 4096) -> tuple[dict | None, dict]:
    message = {"role": "user", "content": prompt}
    if images:
        message["images"] = images
    t0 = time.time()
    resp = llm.chat(
        model=LLM_MODEL,
        messages=[message],
        format=schema,
        think=False,  # gemma4 thinking tokens otherwise eat the output budget
        options={"num_ctx": num_ctx, "temperature": temperature,
                 "num_predict": num_predict},
    )
    timing = {
        "seconds": round(time.time() - t0, 1),
        "in_tokens": resp.get("prompt_eval_count"),
        "out_tokens": resp.get("eval_count"),
        "tok_per_s": round(
            resp["eval_count"] / (resp["eval_duration"] / 1e9), 1
        ) if resp.get("eval_duration") else None,
    }
    try:
        return json.loads(resp["message"]["content"]), timing
    except json.JSONDecodeError:
        return None, timing


OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "context": {"type": "string"},
                    "ideas": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"first_words": {"type": "string"}},
                            "required": ["first_words"],
                        },
                    },
                },
                "required": ["title", "ideas"],
            },
        }
    },
    "required": ["sections"],
}

OUTLINE_FIGURES_SCHEMA = json.loads(json.dumps(OUTLINE_SCHEMA))
OUTLINE_FIGURES_SCHEMA["properties"]["figures"] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"page": {"type": "integer"}, "description": {"type": "string"}},
        "required": ["page", "description"],
    },
}
OUTLINE_FIGURES_SCHEMA["required"].append("figures")

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }
    },
    "required": ["chunks"],
}

VIEWS = ["notable", "main_ideas", "methods", "why_cite"]
VIEWS_SCHEMA = {
    "type": "object",
    "properties": {v: {"type": "string"} for v in VIEWS},
    "required": VIEWS,
}

QUERIES_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
}

TAGS_SCHEMA = {
    "type": "object",
    "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    "required": ["tags"],
}

P1 = """Segment this scientific paper into its logical sections, and each section
into sequential ideas. For every section give its title; for every idea give
first_words: the words the idea starts with, copied from the text.

PAPER TEXT:
{text}"""

P2 = """Segment this scientific paper into its logical sections, and each section
into sequential ideas (one idea = one self-contained point, argument, method
step, or result — typically 2-6 ideas per page; an idea is usually one to a
few paragraphs).

Rules for first_words: copy the first 5-8 words of the idea EXACTLY as they
appear in the text, character for character. Never paraphrase them. Ideas must
appear in reading order.

For every section give its title (use the paper's own headings where they
exist).

PAPER TEXT:
{text}"""

P3 = P2.replace(
    "For every section give its title",
    "For every section give its title and a context field: 1-2 sentences "
    "saying what this section covers in the context of this specific paper "
    "(mention the paper's actual subject, materials, or methods).\n"
    "Also give its title",
)

P2M = P2.replace(
    "PAPER TEXT:",
    """Additionally, the pages of the paper are attached as images. Use them to
find section boundaries (headings, layout) and to describe every figure:
for each figure give its page number (1-based) and a description of what it
shows — axes, quantities, trends, and what a reader should take from it.

PAPER TEXT:""",
)

REWRITE_PROMPT = """Rewrite this scientific paper as JSON chunks. Copy the text
VERBATIM — do not change, drop, or add a single word — but split it so that
each chunk contains exactly one sequential idea (one self-contained point,
argument, method step, or result). The concatenation of all chunks must
reproduce the paper text exactly.

PAPER TEXT:
{text}"""

VIEWS_PROMPT = """Read this scientific paper and answer four questions about it.
Write each answer as 2-4 dense sentences that use the paper's own key
terminology (these answers will be used for search indexing).

- notable: What is most notable or memorable in this paper — the thing a
  reader would remember it by?
- main_ideas: Very concisely, what are the main ideas and conclusions?
- methods: What methods, instruments, techniques, and materials does it use?
- why_cite: What problem does this paper address, and in what situation would
  someone cite it?

PAPER TEXT:
{text}"""

QUERIES_PROMPT = """A researcher read this paper months ago and now vaguely
remembers it. Write 3 short search queries (5-12 words each) they might type
to find it again. Make them imprecise and impression-based — how people
actually half-remember papers ("that paper about X where they showed Y") —
not keyword-perfect. Do not use the paper's title.

PAPER TEXT:
{text}"""

TAGS_PROMPT = """Assign 2-6 topic tags to this paper based on its title and
abstract. Tags should be short (1-3 words, lowercase), reusable across a
physics/materials/engineering library, and describe subject matter, methods,
or materials. No tags about publication type or quality.

TITLE: {title}
ABSTRACT: {abstract}"""


# ---------------------------------------------------------------- embeddings


def embedder():
    from lest.embedders.ollama import OllamaEmbedder

    return OllamaEmbedder(EMB_MODEL, host=EMB_HOST)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / max(na * nb, 1e-12)


# ---------------------------------------------------------------- stages


def production_db() -> Path:
    matches = glob.glob(str(REPO / "data" / "Zotero-*.db"))
    if not matches:
        sys.exit("no production Zotero DB found in data/")
    return Path(matches[0])


def stage_select():
    if PAPERS.exists():
        print("papers.json exists, keeping frozen selection")
        return
    conn = sqlite3.connect(f"file:{production_db()}?mode=ro", uri=True)
    rows = conn.execute(
        """SELECT d.key, d.title, f.path FROM files f JOIN documents d
           ON d.id = f.document_id
           WHERE f.status = 'ok' AND d.meta_json LIKE '%creators%'"""
    ).fetchall()
    conn.close()
    random.seed(20260805)
    random.shuffle(rows)

    def pages(path):
        try:
            with pymupdf.open(path) as doc:
                return doc.page_count
        except Exception:
            return None

    short, medium, long_, scanned = [], [], [], []
    for key, title, path in rows:
        if not Path(path).exists():
            continue
        n = pages(path)
        if n is None:
            continue
        entry = {"key": key, "title": title, "path": path, "pages": n}
        if n <= 6 and len(short) < 3:
            short.append(entry)
        elif 7 <= n <= 14 and len(medium) < 4:
            medium.append(entry)
        elif n >= 15 and len(long_) < 2:
            long_.append(entry)
        if len(short) == 3 and len(medium) == 4 and len(long_) == 2:
            break

    # figure-heavy: most embedded images among 40 random medium-length papers
    candidates = [r for r in rows[:400] if r[2] not in {e["path"] for e in short + medium + long_}]
    best, best_n = None, -1
    for key, title, path in candidates[:40]:
        try:
            with pymupdf.open(path) as doc:
                if not 4 <= doc.page_count <= 14:
                    continue
                n_img = sum(len(page.get_images()) for page in doc)
                if n_img > best_n:
                    best, best_n = {"key": key, "title": title, "path": path,
                                    "pages": doc.page_count, "figure_heavy": True}, n_img
        except Exception:
            continue

    selection = short + medium + long_ + ([best] if best else [])
    PILOT.mkdir(parents=True, exist_ok=True)
    PAPERS.write_text(json.dumps(selection, indent=2))
    print(f"selected {len(selection)} papers:")
    for e in selection:
        print(f"  {e['pages']:>3}p  {e['title'][:70]}")


def papers() -> list[dict]:
    return json.loads(PAPERS.read_text())


def stage_e1(done):
    for paper in papers():
        text = paper_text(paper["path"])
        for variant, prompt in [("P1", P1), ("P2", P2), ("P3", P3)]:
            ckpt = f"e1/{paper['key']}/{variant}"
            if ckpt in done:
                continue
            parsed, timing = llm_call(prompt.format(text=text), OUTLINE_SCHEMA)
            fields = {"stage": "e1", "paper": paper["key"], "variant": variant,
                      "valid_json": parsed is not None, **timing}
            if parsed:
                anchors = [i["first_words"] for s in parsed.get("sections", [])
                           for i in s.get("ideas", [])]
                m = anchor_metrics(anchors, text)
                fields.update(m)
                fields["n_sections"] = len(parsed.get("sections", []))
                fields["ideas_per_page"] = round(len(anchors) / paper["pages"], 2)
                fields["has_context"] = all(
                    s.get("context") for s in parsed.get("sections", []))
                fields["outline"] = parsed
            record(ckpt, **fields)


def stage_e2(done):
    subset = sorted(papers(), key=lambda p: p["pages"])[:5]
    p2_cuts = {r["paper"]: r.get("positions", [])
               for r in load_results("e1") if r["variant"] == "P2"}
    for paper in subset:
        ckpt = f"e2/{paper['key']}"
        if ckpt in done:
            continue
        text = paper_text(paper["path"])
        parsed, timing = llm_call(REWRITE_PROMPT.format(text=text), REWRITE_SCHEMA,
                                  num_predict=-1)
        fields = {"stage": "e2", "paper": paper["key"],
                  "valid_json": parsed is not None, **timing}
        if parsed:
            chunks = [c["text"] for c in parsed.get("chunks", [])]
            rejoined = normalize(" ".join(chunks))
            fields["n_chunks"] = len(chunks)
            fields["length_ratio"] = round(len(rejoined) / max(len(text), 1), 3)
            fields["fidelity"] = round(difflib.SequenceMatcher(
                None, text[:60000], rejoined[:60000]).ratio(), 4)
            cut_anchors = [" ".join(c.split()[:8]) for c in chunks if c.strip()]
            m = anchor_metrics(cut_anchors, text)
            fields["anchor_hit_rate"] = m["anchor_hit_rate"]
            fields.update({f"vs_p2_{k}": v for k, v in
                           agreement(m["positions"], p2_cuts.get(paper["key"], [])).items()})
        record(ckpt, **fields)


def render_pages(path: str) -> list[bytes]:
    images = []
    with pymupdf.open(path) as doc:
        for page in doc[:MAX_PAGES_RENDERED]:
            pix = page.get_pixmap(dpi=RENDER_DPI)
            images.append(pix.tobytes("png"))
    return images


def stage_e3(done):
    p2_results = {r["paper"]: r for r in load_results("e1") if r["variant"] == "P2"}
    for paper in papers():
        ckpt = f"e3/{paper['key']}"
        if ckpt in done:
            continue
        text = paper_text(paper["path"])
        images = render_pages(paper["path"])
        parsed, timing = llm_call(P2M.format(text=text), OUTLINE_FIGURES_SCHEMA,
                                  images=images)
        fig_labels = len({m.group(1) for m in
                          re.finditer(r"\b(?:Figure|Fig\.?)\s*(\d+)", text)})
        fields = {"stage": "e3", "paper": paper["key"], "n_pages_sent": len(images),
                  "valid_json": parsed is not None, "fig_labels_in_text": fig_labels,
                  **timing}
        if parsed:
            anchors = [i["first_words"] for s in parsed.get("sections", [])
                       for i in s.get("ideas", [])]
            m = anchor_metrics(anchors, text)
            fields["anchor_hit_rate"] = m["anchor_hit_rate"]
            fields["n_figures_described"] = len(parsed.get("figures", []))
            fields["figures"] = parsed.get("figures", [])
            p2 = p2_results.get(paper["key"], {})
            fields.update({f"vs_p2_{k}": v for k, v in
                           agreement(m["positions"], p2.get("positions", [])).items()})
        record(ckpt, **fields)


def abstract_for(key: str) -> str | None:
    conn = sqlite3.connect(f"file:{ZOTERO / 'zotero.sqlite'}?mode=ro", uri=True)
    row = conn.execute(
        """SELECT idv.value FROM items i
           JOIN itemData id ON id.itemID = i.itemID
           JOIN itemDataValues idv USING (valueID)
           JOIN fields fl USING (fieldID)
           WHERE i.key = ? AND fl.fieldName = 'abstractNote'""", (key,)).fetchone()
    conn.close()
    return row[0] if row else None


def body_vectors(paths: list[str]) -> dict[str, list[tuple[str, list[float]]]]:
    """path -> [(chunk_text, vector)] from the production DB."""
    conn = sqlite3.connect(f"file:{production_db()}?mode=ro", uri=True)
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    (dim,) = conn.execute("SELECT value FROM meta WHERE key='dim'").fetchone()
    out = {}
    for path in paths:
        rows = conn.execute(
            """SELECT c.text, v.embedding FROM chunks c
               JOIN files f ON f.id = c.file_id
               JOIN chunk_vectors v ON v.chunk_id = c.id
               WHERE f.path = ?""", (path,)).fetchall()
        out[path] = [(t, list(struct.unpack(f"{dim}f", blob))) for t, blob in rows]
    conn.close()
    return out


def stage_e4(done):
    emb = embedder()
    for paper in papers():
        ckpt = f"e4/views/{paper['key']}"
        if ckpt not in done:
            text = paper_text(paper["path"])
            parsed, timing = llm_call(VIEWS_PROMPT.format(text=text), VIEWS_SCHEMA)
            record(ckpt, stage="e4_views", paper=paper["key"],
                   valid_json=parsed is not None, views=parsed, **timing)
        ckpt = f"e4/queries/{paper['key']}"
        if ckpt not in done:
            text = paper_text(paper["path"])
            parsed, timing = llm_call(QUERIES_PROMPT.format(text=text), QUERIES_SCHEMA,
                                      temperature=1.0)
            record(ckpt, stage="e4_queries", paper=paper["key"],
                   valid_json=parsed is not None,
                   queries=(parsed or {}).get("queries", [])[:3], **timing)

    ckpt = "e4/analysis"
    if ckpt in done:
        return
    views_by_paper = {r["paper"]: r["views"] for r in load_results("e4_views")
                      if r.get("views")}
    queries_by_paper = {r["paper"]: r["queries"] for r in load_results("e4_queries")
                        if r.get("queries")}
    plist = papers()
    bodies = body_vectors([p["path"] for p in plist])

    view_vecs, redundancy = {}, []
    for paper in plist:
        key = paper["key"]
        views = views_by_paper.get(key)
        if not views:
            continue
        texts = [views[v] for v in VIEWS]
        abstract = abstract_for(key)
        vecs = emb.embed(texts + ([abstract] if abstract else []))
        view_vecs[key] = dict(zip(VIEWS, vecs[:4]))
        matrix = {}
        labels = VIEWS + (["abstract"] if abstract else [])
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i < j:
                    matrix[f"{a}~{b}"] = round(cosine(vecs[i], vecs[j]), 4)
        redundancy.append({"paper": key, **matrix})

    conditions = {"body": [], "body+views": [], "views": []}
    per_query = []
    for paper in plist:
        for q in queries_by_paper.get(paper["key"], []):
            qv = emb.embed_query(q)
            scores = {}
            for cond in conditions:
                ranking = []
                for other in plist:
                    pools = []
                    if cond in ("body", "body+views"):
                        pools += [v for _, v in bodies.get(other["path"], [])]
                    if cond in ("views", "body+views"):
                        pools += list(view_vecs.get(other["key"], {}).values())
                    best = max((cosine(qv, v) for v in pools), default=-1)
                    ranking.append((best, other["key"]))
                ranking.sort(reverse=True)
                rank = next(i for i, (_, k) in enumerate(ranking, 1)
                            if k == paper["key"])
                conditions[cond].append(1 / rank)
                scores[cond] = rank
            per_query.append({"paper": paper["key"], "query": q, **scores})
    mrr = {c: round(sum(v) / max(len(v), 1), 3) for c, v in conditions.items()}
    record(ckpt, stage="e4_analysis", redundancy=redundancy, mrr=mrr,
           per_query=per_query)


def stage_e6(done):
    conn = sqlite3.connect(f"file:{ZOTERO / 'zotero.sqlite'}?mode=ro", uri=True)
    rows = conn.execute("""
        SELECT i.key,
          (SELECT idv.value FROM itemData id JOIN itemDataValues idv USING (valueID)
             JOIN fields f USING (fieldID)
            WHERE id.itemID = i.itemID AND f.fieldName = 'title'),
          (SELECT idv.value FROM itemData id JOIN itemDataValues idv USING (valueID)
             JOIN fields f USING (fieldID)
            WHERE id.itemID = i.itemID AND f.fieldName = 'abstractNote')
        FROM items i JOIN itemTypes it USING (itemTypeID)
        WHERE it.typeName NOT IN ('attachment', 'note', 'annotation')
          AND i.itemID NOT IN (SELECT itemID FROM deletedItems)""").fetchall()
    conn.close()
    todo = [(k, t, a) for k, t, a in rows if t and f"e6/{k}" not in done]
    print(f"e6: {len(todo)} items to tag")
    for key, title, abstract in todo:
        parsed, timing = llm_call(
            TAGS_PROMPT.format(title=title, abstract=abstract or "(none)"),
            TAGS_SCHEMA, num_ctx=SMALL_CTX, num_predict=200)
        record(f"e6/{key}", stage="e6", paper=key,
               tags=(parsed or {}).get("tags", []), **timing)


STAGES = {"select": stage_select, "e1": stage_e1, "e2": stage_e2,
          "e3": stage_e3, "e4": stage_e4, "e6": stage_e6}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="+",
                        choices=[*STAGES, "all"])
    args = parser.parse_args()
    names = list(STAGES) if "all" in args.stages else args.stages
    PILOT.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(f"=== stage {name} ===", flush=True)
        stage = STAGES[name]
        stage() if name == "select" else stage(done_keys())


if __name__ == "__main__":
    main()
