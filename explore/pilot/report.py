#!/usr/bin/env python3
"""Aggregate data/pilot/results.jsonl into data/pilot/report.md (+ tags.tsv)."""

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PILOT = REPO / "data" / "pilot"


def rows(stage):
    out = []
    for line in (PILOT / "results.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("stage") == stage:
            out.append(r)
    return out


def mean(values):
    values = [v for v in values if v is not None]
    return round(st.mean(values), 3) if values else None


def table(headers, data):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in data:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def main():
    papers = {p["key"]: p for p in json.loads((PILOT / "papers.json").read_text())}
    md = ["# Pilot results\n"]

    # E1
    by_variant = defaultdict(list)
    for r in rows("e1"):
        by_variant[r["variant"]].append(r)
    md.append("## E1 — Outline prompt variants (text-only)\n")
    md.append(table(
        ["variant", "valid JSON", "anchor hit", "order ok", "ideas/page",
         "sections", "context blurbs", "mean s", "mean out-tok"],
        [[v,
          f"{sum(r['valid_json'] for r in rs)}/{len(rs)}",
          mean([r.get("anchor_hit_rate") for r in rs]),
          mean([r.get("order_ok_rate") for r in rs]),
          mean([r.get("ideas_per_page") for r in rs]),
          mean([r.get("n_sections") for r in rs]),
          f"{sum(bool(r.get('has_context')) for r in rs)}/{len(rs)}",
          mean([r.get("seconds") for r in rs]),
          mean([r.get("out_tokens") for r in rs])]
         for v, rs in sorted(by_variant.items())]))

    # E2
    e2 = rows("e2")
    if e2:
        md.append("\n## E2 — Verbatim rewrite vs boundaries\n")
        md.append(table(
            ["paper", "pages", "s", "out-tok", "chunks", "fidelity",
             "len ratio", "anchor hit", "boundary F1 vs P2"],
            [[papers[r["paper"]]["title"][:40], papers[r["paper"]]["pages"],
              r.get("seconds"), r.get("out_tokens"), r.get("n_chunks"),
              r.get("fidelity"), r.get("length_ratio"),
              round(r.get("anchor_hit_rate", 0), 2),
              round(r.get("vs_p2_boundary_f1", 0), 2)] for r in e2]))
        p2_time = mean([r["seconds"] for r in by_variant.get("P2", [])])
        md.append(f"\nMean rewrite time {mean([r.get('seconds') for r in e2])}s "
                  f"vs P2 outline {p2_time}s per paper.\n")

    # E3
    e3 = rows("e3")
    if e3:
        md.append("\n## E3 — Multimodal (text + page images)\n")
        md.append(table(
            ["paper", "pages sent", "s", "anchor hit", "boundary F1 vs P2",
             "figs described", "fig labels in text"],
            [[papers[r["paper"]]["title"][:40], r.get("n_pages_sent"),
              r.get("seconds"), round(r.get("anchor_hit_rate", 0), 2),
              round(r.get("vs_p2_boundary_f1", 0), 2),
              r.get("n_figures_described"), r.get("fig_labels_in_text")]
             for r in e3]))

    # E4
    analysis = rows("e4_analysis")
    if analysis:
        a = analysis[-1]
        md.append("\n## E4 — Reflection views\n")
        pair_means = defaultdict(list)
        for row in a["redundancy"]:
            for k, v in row.items():
                if k != "paper":
                    pair_means[k].append(v)
        md.append("Mean pairwise cosine between views (within paper):\n")
        md.append(table(["pair", "mean cos"],
                        sorted(((k, mean(v)) for k, v in pair_means.items()),
                               key=lambda x: -x[1])))
        md.append("\nRetrieval (vague queries, mean reciprocal rank; "
                  "higher is better):\n")
        md.append(table(["condition", "MRR", "rank=1"],
                        [[c, a["mrr"][c],
                          sum(1 for q in a["per_query"] if q[c] == 1)]
                         for c in ("body", "body+views", "views")]))
        misses = [q for q in a["per_query"] if q["body"] > 1]
        if misses:
            md.append("\nQueries where body-only search missed rank 1:\n")
            md.append(table(["query", "body", "body+views"],
                            [[q["query"][:60], q["body"], q["body+views"]]
                             for q in misses]))

    # E5 (timing rollup across stages)
    md.append("\n## E5 — Performance (gemma4 on RX 9070 XT)\n")
    perf = []
    for stage in ("e1", "e2", "e3", "e4_views", "e4_queries", "e6"):
        rs = rows(stage)
        if rs:
            perf.append([stage, len(rs), mean([r.get("seconds") for r in rs]),
                         mean([r.get("tok_per_s") for r in rs]),
                         mean([r.get("in_tokens") for r in rs]),
                         mean([r.get("out_tokens") for r in rs])])
    md.append(table(["stage", "calls", "mean s", "tok/s", "in-tok", "out-tok"], perf))

    # E6
    e6 = rows("e6")
    if e6:
        counts = Counter(t.strip().lower() for r in e6 for t in r.get("tags", []))
        md.append(f"\n## E6 — Freeform tags ({len(e6)} items, "
                  f"{len(counts)} unique tags)\n")
        md.append(table(["tag", "count"],
                        [[t, c] for t, c in counts.most_common(40)]))
        with (PILOT / "tags.tsv").open("w") as f:
            for t, c in counts.most_common():
                f.write(f"{t}\t{c}\n")
        md.append("\nFull table: data/pilot/tags.tsv\n")

    md.append("\n## Recommendations\n\n_(filled in after review)_\n")
    (PILOT / "report.md").write_text("\n".join(md) + "\n")
    print(f"wrote {PILOT / 'report.md'}")


if __name__ == "__main__":
    sys.exit(main())
