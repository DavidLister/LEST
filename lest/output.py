import json
import re

from .query import SearchResult

_UNSAFE = re.compile(r"[\t\n\r]+")


def _scrub(text: str) -> str:
    return _UNSAFE.sub(" ", text).strip()


def format_tsv(result: SearchResult) -> str:
    return f"{result.score:.4f}\t{_scrub(result.title)}\t{';'.join(result.paths)}"


def format_json(result: SearchResult) -> str:
    return json.dumps(
        {
            "score": round(result.score, 6),
            "title": result.title,
            "key": result.key,
            "paths": result.paths,
            "meta": result.meta,
            "best_chunk": result.best_chunk,
        },
        ensure_ascii=False,
    )
