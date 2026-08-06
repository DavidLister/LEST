import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import Catalog, fold
from .embedders import get_embedder
from .ranking import parse_agg
from .store import ChunkHit, Store, db_path_for

KNN_OVERFETCH = 10  # fetch this many chunks per requested document result
KNN_MIN = 200
FILTER_OVERFETCH = 2000  # facet filters discard hits, so over-fetch harder
RRF_K = 60  # standard reciprocal-rank-fusion constant


@dataclass
class SearchResult:
    score: float
    title: str
    paths: list[str]
    key: str
    meta: dict
    best_chunk: str
    kind: str = "body"
    doc_type: str | None = None
    tags: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    duplicate_keys: list[str] = field(default_factory=list)


def _open_store(directory: Path, db_base: Path | None = None) -> Store:
    return Store(db_path_for(directory.expanduser(), base=db_base))


def _facet_groups(
    tags: list[str], authors: list[str], doc_type: str | None
) -> tuple[list[set[str]], list[set[str]], set[str] | None]:
    """Resolve CLI facet values through the catalog (alias-aware, fuzzy for
    authors). Unknown terms filter literally — an unindexed tag simply matches
    nothing rather than erroring."""
    catalog = Catalog()
    try:
        tag_groups = [catalog.lookup_term("tag", t) or {fold(t)} for t in tags]
        author_groups = [catalog.lookup_author(a) or {a} for a in authors]
        type_group = None
        if doc_type:
            type_group = catalog.lookup_term("doctype", doc_type) or {fold(doc_type)}
        return tag_groups, author_groups, type_group
    finally:
        catalog.close()


def _fuse(vec_hits: list[ChunkHit], fts_hits: list[ChunkHit]) -> list[ChunkHit]:
    """Reciprocal-rank fusion of the two ranked lists; the fused score replaces
    `similarity` (scale differs from cosine — ~0.03 max for a double #1 hit)."""
    fused: dict[int, float] = defaultdict(float)
    hits: dict[int, ChunkHit] = {}
    for ranked in (vec_hits, fts_hits):
        for rank, hit in enumerate(ranked, 1):
            fused[hit.chunk_id] += 1.0 / (RRF_K + rank)
            hits.setdefault(hit.chunk_id, hit)
    out = []
    for chunk_id, score in fused.items():
        hit = hits[chunk_id]
        hit.similarity = score
        out.append(hit)
    out.sort(key=lambda h: h.similarity, reverse=True)
    return out


def _dedup(results: list["SearchResult"]) -> list["SearchResult"]:
    """Collapse duplicate library entries (same normalized title + year)."""
    seen: dict[tuple, SearchResult] = {}
    out = []
    for result in results:  # results arrive sorted by score
        key = (fold(result.title), result.meta.get("year"))
        prior = seen.get(key)
        if prior is None:
            seen[key] = result
            out.append(result)
        else:
            prior.duplicate_keys.append(result.key)
            for path in result.paths:
                if path not in prior.paths:
                    prior.paths.append(path)
    return out


def search_directory(
    directory: Path,
    query: str,
    n: int = 10,
    agg_spec: str = "max",
    db_base: Path | None = None,
    tags: list[str] | None = None,
    authors: list[str] | None = None,
    doc_type: str | None = None,
    hybrid: bool = True,
    dedup: bool = True,
    smart: bool = False,
) -> list[SearchResult]:
    aggregate = parse_agg(agg_spec)
    parsed = client = None
    search_text = query
    if smart:
        from .llm import smart_client
        from .smart import parse_query

        client = smart_client()
        client.ping()
        parsed = parse_query(client, query)
        search_text = parsed.semantic_query
    store = _open_store(directory, db_base)
    try:
        model = store.get_meta("model")
        embedder = get_embedder(store.get_meta("embedder") or "ollama", model)
        query_vector = embedder.embed_query(search_text)

        filtering = bool(tags or authors or doc_type)
        allowed = None
        if filtering:
            tag_groups, author_groups, type_group = _facet_groups(
                tags or [], authors or [], doc_type
            )
            allowed = store.filter_documents(tag_groups, author_groups, type_group)

        k = max(n * KNN_OVERFETCH, KNN_MIN)
        if filtering:
            k = max(k, FILTER_OVERFETCH)
        vec_hits = store.knn(query_vector, k=k)
        use_hybrid = hybrid and store.has_fts
        hits = (
            _fuse(vec_hits, store.fts_search(search_text, k=k))
            if use_hybrid
            else vec_hits
        )

        by_document: dict[int, list[ChunkHit]] = defaultdict(list)
        for hit in hits:
            if allowed is not None and hit.document_id not in allowed:
                continue
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
                    kind=best.kind,
                    doc_type=best.doc_type,
                    tags=store.document_tags(document_id),
                    authors=store.document_authors(document_id),
                )
            )
        if parsed is not None:
            from .smart import facet_multiplier

            for result in results:
                result.score *= facet_multiplier(result, parsed)
        results.sort(key=lambda r: r.score, reverse=True)
        if dedup:
            results = _dedup(results)
        if parsed is not None:
            from .smart import RERANK_TOP, rerank

            results = rerank(client, query, results[: max(n, RERANK_TOP)])
        return results[:n]
    finally:
        store.close()


def status_directory(directory: Path, db_base: Path | None = None) -> str:
    store = _open_store(directory, db_base)
    try:
        lines = [f"database: {store.db_path}"]
        for key in ("schema_version", "source_dir", "source_type", "model", "embedder",
                    "chunker", "dim", "last_indexed"):
            value = store.get_meta(key)
            if value is not None:
                lines.append(f"{key}: {value}")
        for table, count in store.counts().items():
            lines.append(f"{table}: {count}")
        kinds = store.kind_counts()
        if kinds:
            lines.append(
                "chunk kinds: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
            )
        skipped = store.skipped_files()
        if skipped:
            lines.append(f"skipped files: {len(skipped)}")
            lines.extend(f"  {status}: {path}" for path, status in skipped)
        return "\n".join(lines) + "\n"
    finally:
        store.close()
