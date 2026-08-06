#!/usr/bin/env python3
"""Reranker experiment: can Ollama drive Qwen3-Reranker-0.6B usefully?

Method: for each test query, take the baseline index's top-K chunks by vector
search, then score each (query, chunk) pair with the reranker via first-token
logprobs — score = P("yes") following the official Qwen3-Reranker recipe
(fixed system prompt, empty thinking, yes/no answer). Reports latency and
before/after rankings. Runs on the A2000 (:11434) next to the embedder.

    nix develop -c python explore/rerank_experiment.py [--k 50]
"""

import argparse
import glob
import math
import sys
import time
from pathlib import Path

import ollama

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lest.embedders.ollama import OllamaEmbedder  # noqa: E402
from lest.store import Store  # noqa: E402

RERANKER = "dengcao/Qwen3-Reranker-0.6B:Q8_0"
EMB_MODEL = "qwen3-embedding:8b"
HOST = "http://localhost:11434"

SYSTEM = ('Judge whether the Document meets the requirements based on the '
          'Query and the Instruct provided. Note that the answer can only be '
          '"yes" or "no".')
INSTRUCT = ("Given a web search query, retrieve relevant passages that answer "
            "the query")

QUERIES = [
    ("that paper where crystals grow on amorphous glass and the fastest "
     "grains win", "Evolutionary Selection Growth"),
    ("figure showing rocking curve comparison of GaN nucleation layers",
     "rocking curve"),
    ("review of zinc oxide fundamental properties", "ZnO"),
]


def baseline_db() -> Path:
    return Path(glob.glob(str(REPO / "data" / "Zotero-*.db"))[0])


def rerank_score(client: ollama.Client, query: str, doc: str) -> tuple[float, dict]:
    t0 = time.time()
    resp = client.chat(
        model=RERANKER,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"<Instruct>: {INSTRUCT}\n<Query>: {query}\n<Document>: {doc}"},
        ],
        think=False,
        logprobs=True,
        top_logprobs=10,
        options={"num_predict": 1, "temperature": 0.0, "num_ctx": 8192},
    )
    elapsed = time.time() - t0
    entries = resp.get("logprobs") or []
    top = entries[0].get("top_logprobs", []) if entries else []
    logp = {e["token"].strip().lower(): e["logprob"] for e in top}
    p_yes = math.exp(logp["yes"]) if "yes" in logp else 0.0
    p_no = math.exp(logp["no"]) if "no" in logp else 0.0
    denom = (p_yes + p_no) or 1.0
    return p_yes / denom, {"s": elapsed, "first_token": resp["message"]["content"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args()

    emb = OllamaEmbedder(EMB_MODEL, host=HOST)
    client = ollama.Client(host=HOST)
    store = Store(baseline_db())

    for query, marker in QUERIES:
        print(f"\n=== {query!r}")
        hits = store.knn(emb.embed_query(query), k=args.k)
        print("vector top-5:")
        for hit in hits[:5]:
            print(f"  {hit.similarity:.3f}  {hit.title[:70]}")

        t0 = time.time()
        scored = []
        per_call = []
        for hit in hits:
            score, info = rerank_score(client, query, hit.chunk_text[:4000])
            scored.append((score, hit))
            per_call.append(info["s"])
        total = time.time() - t0
        scored.sort(key=lambda pair: pair[0], reverse=True)

        print(f"reranked top-5  ({total:.1f}s for {len(hits)} chunks, "
              f"mean {sum(per_call)/len(per_call)*1000:.0f}ms, "
              f"max {max(per_call)*1000:.0f}ms):")
        for score, hit in scored[:5]:
            print(f"  {score:.3f}  {hit.title[:70]}")

        def rank_of(seq, marker=marker):
            for i, item in enumerate(seq, 1):
                title = item.title if hasattr(item, "title") else item[1].title
                chunk = item.chunk_text if hasattr(item, "chunk_text") else item[1].chunk_text
                if marker.lower() in title.lower() or marker.lower() in chunk.lower():
                    return i
            return None

        print(f"marker {marker!r}: vector rank {rank_of(hits)} -> "
              f"reranked rank {rank_of(scored)}")

    store.close()


if __name__ == "__main__":
    main()
