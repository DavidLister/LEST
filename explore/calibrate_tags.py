#!/usr/bin/env python3
"""Threshold calibration for the tag catalog, against the pilot's freeform
tag dump (data/pilot/tags.tsv: 2200 unique tags over 1157 items).

Dry-run: seeds a THROWAWAY catalog from vocab-proposed.txt, embeds every
freeform tag, and reports how the similarity bands would sort them —
auto-map / gray-zone (adjudicated) / new — plus the outcomes for known
variant families (the 8-way MOCVD garble). Run inside `nix develop`:

    python explore/calibrate_tags.py [--auto 0.92] [--gray 0.80]
"""

import argparse
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lest.catalog import Catalog  # noqa: E402
from lest.embedders.ollama import OllamaEmbedder  # noqa: E402

TAGS_TSV = REPO / "data" / "pilot" / "tags.tsv"
VOCAB = REPO / "data" / "pilot" / "vocab-proposed.txt"

MOCVD_FAMILY = ["mocvd", "mcvd", "movpe", "omvpe", "mova", "mcvp", "mcvpe",
                "metalorganic vapor phase epitaxy", "metal-organic cvd"]


def load_freeform() -> Counter:
    counts = Counter()
    for line in TAGS_TSV.read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[1].isdigit():
            counts[parts[0].strip().lower()] = int(parts[1])
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", type=float, default=0.92)
    parser.add_argument("--gray", type=float, default=0.80)
    args = parser.parse_args()

    emb = OllamaEmbedder("qwen3-embedding:0.6b")
    freeform = load_freeform()
    print(f"{len(freeform)} unique freeform tags", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        cat = Catalog(path=Path(tmp) / "cal.db", embed=emb.embed)
        seeded = cat.seed_tags(VOCAB)
        vocab = cat.names("tag")
        print(f"seeded {seeded} vocab tags ({len(vocab)} total)", flush=True)

        names = list(freeform)
        vectors = emb.embed(names)
        buckets = {"exact/alias": [], "auto": [], "gray": [], "new": []}
        for name, vec in zip(names, vectors, strict=True):
            if cat._canonical("tag", name):
                buckets["exact/alias"].append((name, 1.0, name))
                continue
            neighbours = cat._nearest("tag", vec, n=1)
            best, sim = neighbours[0] if neighbours else ("", 0.0)
            if sim >= args.auto:
                buckets["auto"].append((name, sim, best))
            elif sim >= args.gray:
                buckets["gray"].append((name, sim, best))
            else:
                buckets["new"].append((name, sim, best))

        total = len(names)
        weighted = sum(freeform.values())
        print(f"\nthresholds: auto>={args.auto}  gray>={args.gray}")
        for bucket, entries in buckets.items():
            occurrences = sum(freeform[n] for n, _, _ in entries)
            print(f"{bucket:>12}: {len(entries):4d} tags ({100*len(entries)/total:4.1f}%)  "
                  f"{occurrences:5d} occurrences ({100*occurrences/weighted:4.1f}%)")

        print("\nMOCVD family outcomes:")
        by_name = {n: (s, b) for bucket in buckets.values() for n, s, b in bucket}
        for variant in MOCVD_FAMILY:
            if variant in freeform:
                sim, best = by_name.get(variant, (None, "?"))
                bucket = next(k for k, v in buckets.items()
                              if any(n == variant for n, _, _ in v))
                print(f"  {variant:>35} -> {best:<25} sim={sim:.3f}  [{bucket}]"
                      if sim else f"  {variant:>35} [exact]")

        for bucket in ("auto", "gray"):
            entries = sorted(buckets[bucket], key=lambda e: -freeform[e[0]])[:15]
            print(f"\ntop {bucket} examples (freeform -> nearest vocab):")
            for name, sim, best in entries:
                print(f"  {freeform[name]:4d}x {name:>35} -> {best:<28} {sim:.3f}")

        # boundary neighbourhoods: what flips if thresholds move +-0.02
        print("\nnear-boundary examples:")
        for name, sim, best in sorted(
            buckets["auto"] + buckets["gray"] + buckets["new"], key=lambda e: e[1]
        ):
            if abs(sim - args.auto) <= 0.015 or abs(sim - args.gray) <= 0.015:
                print(f"  {name:>35} -> {best:<28} {sim:.3f}")
        cat.close()


if __name__ == "__main__":
    main()
