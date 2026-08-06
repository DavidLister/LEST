#!/usr/bin/env python3
"""Benchmark smart-search stages: gemma4:12B (9070XT) vs a small model on the
A2000. Compares query-parse output, listwise rerank agreement, and latency.

Run while the re-index hammers the ROCm card to see the realistic contrast.

    nix develop -c python explore/smart_bench.py [--small gemma4:e4b-it-q4_K_M]
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lest.llm import (  # noqa: E402
    QUERY_PARSE_PROMPT,
    QUERY_PARSE_SCHEMA,
    SMALL_CTX,
    LlmClient,
)
from lest.smart import rerank  # noqa: E402

PARSE_QUERIES = [
    "papers by Li about water ice on the moon",
    "only reviews of zinc oxide optical properties",
    "that MOCVD paper from around 2018, maybe by Lee",
    "distillation column design for lunar gravity",
    "gan quantum wells",
    "theses on chemical process simulation after 2015",
]

RERANK_QUERIES = [
    "water ice detection on the lunar surface",
    "mass transfer in reduced gravity distillation",
    "cubic GaN quantum well growth",
]


def parse_bench(client, label):
    print(f"\n--- parse: {label}")
    outputs = {}
    for query in PARSE_QUERIES:
        t0 = time.time()
        raw = client.call(QUERY_PARSE_PROMPT.format(query=query),
                          QUERY_PARSE_SCHEMA, num_ctx=SMALL_CTX, num_predict=512)
        dt = time.time() - t0
        outputs[query] = raw
        compact = {k: v for k, v in (raw or {}).items() if v and v != 0}
        blob = json.dumps(compact, ensure_ascii=False)[:200]
        print(f"  {dt:5.1f}s  {query!r}\n         -> {blob}")
    return outputs


def rerank_bench(client, label, shortlists):
    print(f"\n--- rerank: {label}")
    orders = {}
    for query, results in shortlists.items():
        t0 = time.time()
        reranked = rerank(client, query, list(results))
        dt = time.time() - t0
        orders[query] = [r.key for r in reranked]
        print(f"  {dt:5.1f}s  {query!r}: top3 = "
              f"{[r.title[:40] for r in reranked[:3]]}")
    return orders


def agreement(a: list, b: list) -> float:
    """Fraction of pairs ordered the same way (Kendall-tau-like, 1.0 = same)."""
    common = [k for k in a if k in b]
    pos_a = {k: i for i, k in enumerate(common)}
    pos_b = {k: i for i, k in enumerate(k for k in b if k in pos_a)}
    pairs = concordant = 0
    for i, x in enumerate(common):
        for y in common[i + 1:]:
            pairs += 1
            if (pos_a[x] - pos_a[y]) * (pos_b[x] - pos_b[y]) > 0:
                concordant += 1
    return concordant / pairs if pairs else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", default="gemma4:e4b-it-q4_K_M")
    parser.add_argument("--skip-big", action="store_true",
                        help="skip the 12B side (e.g. while it is saturated)")
    args = parser.parse_args()

    big = LlmClient()  # env-default: 12B wherever LEST_GPU_MODE points
    small = LlmClient(host="http://localhost:11434", model=args.small)
    small.ping()

    # build shortlists once from the LLM index (vector top-20 per query)
    from lest.query import search_directory  # noqa: E402

    shortlists = {}
    for query in RERANK_QUERIES:
        shortlists[query] = search_directory(
            Path.home() / "Zotero", query, n=20, db_base=REPO / "data" / "llm"
        )

    small_parse = parse_bench(small, args.small)
    small_orders = rerank_bench(small, args.small, shortlists)
    if not args.skip_big:
        big_parse = parse_bench(big, "gemma4:12B")
        big_orders = rerank_bench(big, "gemma4:12B", shortlists)
        print("\n--- rerank agreement (pairwise order match vs 12B):")
        for query in RERANK_QUERIES:
            print(f"  {agreement(big_orders[query], small_orders[query]):.2f}  {query!r}")
        print("\n--- parse diffs vs 12B:")
        for query in PARSE_QUERIES:
            b, s = big_parse.get(query) or {}, small_parse.get(query) or {}
            for field in ("authors", "tags", "doc_types", "year_from", "year_to"):
                bv, sv = b.get(field), s.get(field)
                names = lambda v: sorted(e["name"] for e in v) if isinstance(v, list) else v  # noqa: E731
                if names(bv) != names(sv):
                    print(f"  {query!r} {field}: 12B={names(bv)} small={names(sv)}")


if __name__ == "__main__":
    main()
