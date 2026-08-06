import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .embedders import get_embedder
from .ranking import parse_agg
from .store import Store, db_path_for

KNN_OVERFETCH = 10  # fetch this many chunks per requested document result
KNN_MIN = 200


@dataclass
class SearchResult:
    score: float
    title: str
    paths: list[str]
    key: str
    meta: dict
    best_chunk: str


def _open_store(directory: Path) -> Store:
    return Store(db_path_for(directory.expanduser()))


def search_directory(
    directory: Path, query: str, n: int = 10, agg_spec: str = "max"
) -> list[SearchResult]:
    aggregate = parse_agg(agg_spec)
    store = _open_store(directory)
    try:
        model = store.get_meta("model")
        embedder = get_embedder(store.get_meta("embedder") or "ollama", model)
        query_vector = embedder.embed_query(query)

        hits = store.knn(query_vector, k=max(n * KNN_OVERFETCH, KNN_MIN))
        by_document: dict[int, list] = defaultdict(list)
        for hit in hits:
            by_document[hit.document_id].append(hit)

        results = []
        for document_id, doc_hits in by_document.items():
            score = aggregate([h.similarity for h in doc_hits])
            best = max(doc_hits, key=lambda h: h.similarity)
            results.append(
                SearchResult(
                    score=score,
                    title=best.title,
                    paths=store.document_paths(document_id),
                    key=best.document_key,
                    meta=json.loads(best.meta_json),
                    best_chunk=best.chunk_text,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:n]
    finally:
        store.close()


def status_directory(directory: Path) -> str:
    store = _open_store(directory)
    try:
        lines = [f"database: {store.db_path}"]
        for key in ("source_dir", "source_type", "model", "embedder", "chunker", "dim",
                    "last_indexed"):
            value = store.get_meta(key)
            if value is not None:
                lines.append(f"{key}: {value}")
        for table, count in store.counts().items():
            lines.append(f"{table}: {count}")
        skipped = store.skipped_files()
        if skipped:
            lines.append(f"skipped files: {len(skipped)}")
            lines.extend(f"  {status}: {path}" for path, status in skipped)
        return "\n".join(lines) + "\n"
    finally:
        store.close()
