import json
import re

from .query import SearchResult

_UNSAFE = re.compile(r"[\t\n\r]+")


def _scrub(text: str) -> str:
    return _UNSAFE.sub(" ", text).strip()


def format_tsv(result: SearchResult) -> str:
    return f"{result.score:.4f}\t{_scrub(result.title)}\t{';'.join(result.paths)}"


def format_json(result: SearchResult) -> str:
    payload = {
        "score": round(result.score, 6),
        "title": result.title,
        "key": result.key,
        "paths": result.paths,
        "meta": result.meta,
        "best_chunk": result.best_chunk,
        "best_chunk_kind": result.kind,
    }
    if result.doc_type:
        payload["doc_type"] = result.doc_type
    if result.tags:
        payload["tags"] = result.tags
    if result.authors:
        payload["authors"] = result.authors
    if result.duplicate_keys:
        payload["duplicate_keys"] = result.duplicate_keys
    return json.dumps(payload, ensure_ascii=False)
