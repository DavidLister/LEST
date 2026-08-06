"""Smart search (--smart): gemma parses the query into weighted facets, facet
matches boost/filter document scores continuously, and a listwise gemma pass
reranks the shortlist.

The weight semantics are one mechanism, no special cases: a facet with weight
w multiplies a non-matching document's score by (1 - w). At w=1.0 that is a
hard filter; below, a graduated nudge. Facets resolve through the catalog
(alias-aware, fuzzy for authors); a facet the catalog cannot resolve keeps its
literal spelling, and a facet matching nothing penalizes every document
equally — i.e. it is neutral, never destructive.
"""

import logging
from dataclasses import dataclass, field

from .catalog import Catalog, fold
from .llm import (
    QUERY_PARSE_PROMPT,
    QUERY_PARSE_SCHEMA,
    RERANK_PROMPT,
    RERANK_SCHEMA,
    SMALL_CTX,
    LlmClient,
)

log = logging.getLogger(__name__)

RERANK_TOP = 20
PREVIEW_CHARS = 400


@dataclass
class Facet:
    terms: set[str]  # canonical + aliases (or the literal spelling)
    weight: float
    kind: str


@dataclass
class ParsedQuery:
    semantic_query: str
    facets: list[Facet] = field(default_factory=list)
    year_from: int = 0
    year_to: int = 0


def _clamp(weight) -> float:
    try:
        return min(1.0, max(0.0, float(weight)))
    except (TypeError, ValueError):
        return 0.0


def parse_query(client: LlmClient, query: str) -> ParsedQuery:
    raw = client.call(
        QUERY_PARSE_PROMPT.format(query=query),
        QUERY_PARSE_SCHEMA,
        num_ctx=SMALL_CTX,
        num_predict=512,
    )
    if not raw:
        log.warning("query parse failed; falling back to plain search")
        return ParsedQuery(semantic_query=query)

    catalog = Catalog()
    try:
        parsed = ParsedQuery(
            semantic_query=raw.get("semantic_query", "").strip() or query,
            year_from=int(raw.get("year_from") or 0),
            year_to=int(raw.get("year_to") or 0),
        )
        for entry in raw.get("tags", []):
            weight = _clamp(entry.get("weight"))
            name = entry.get("name", "").strip()
            if name and weight > 0:
                terms = catalog.lookup_term("tag", name) or {fold(name)}
                parsed.facets.append(Facet(terms, weight, "tag"))
        for entry in raw.get("authors", []):
            weight = _clamp(entry.get("weight"))
            name = entry.get("name", "").strip()
            if name and weight > 0:
                terms = catalog.lookup_author(name) or {name}
                parsed.facets.append(Facet(terms, weight, "author"))
        for entry in raw.get("doc_types", []):
            weight = _clamp(entry.get("weight"))
            name = entry.get("name", "").strip()
            if name and weight > 0:
                terms = catalog.lookup_term("doctype", name) or {fold(name)}
                parsed.facets.append(Facet(terms, weight, "doctype"))
        return parsed
    finally:
        catalog.close()


def facet_multiplier(result, parsed: ParsedQuery) -> float:
    """Product of (1 - w) over facets the document does NOT match."""
    multiplier = 1.0
    for facet in parsed.facets:
        if facet.kind == "tag":
            matched = bool(facet.terms & set(result.tags))
        elif facet.kind == "author":
            matched = bool(facet.terms & set(result.authors))
            if not matched and not result.authors:
                # docs without ingested authors (v1 baseline): surname
                # containment against the raw creators metadata
                creators = fold(result.meta.get("creators", ""))
                matched = creators != "" and any(
                    fold(term).split(",")[0].strip() in creators
                    for term in facet.terms
                )
        else:
            matched = result.doc_type in facet.terms
        if not matched:
            multiplier *= 1.0 - facet.weight
    if parsed.year_from or parsed.year_to:
        try:
            year = int(result.meta.get("year", 0))
        except (TypeError, ValueError):
            year = 0
        in_range = year and (not parsed.year_from or year >= parsed.year_from) and (
            not parsed.year_to or year <= parsed.year_to
        )
        if not in_range:
            multiplier *= 0.4  # year memories are unreliable: nudge, never filter
    return multiplier


def rerank(client: LlmClient, query: str, results: list) -> list:
    """Listwise gemma rerank of the top results; returns the reordered list.
    Failure leaves the original order untouched."""
    head, tail = results[:RERANK_TOP], results[RERANK_TOP:]
    if len(head) < 2:
        return results
    previews = "\n".join(
        f"[{i}] {r.title} — {r.best_chunk[:PREVIEW_CHARS]}"
        for i, r in enumerate(head)
    )
    raw = client.call(
        RERANK_PROMPT.format(query=query, results=previews),
        RERANK_SCHEMA,
        num_predict=256,
    )
    ranking = [i for i in (raw or {}).get("ranking", []) if 0 <= i < len(head)]
    if not ranking:
        log.warning("rerank failed; keeping vector order")
        return results
    seen = set()
    order = [i for i in ranking if not (i in seen or seen.add(i))]
    order += [i for i in range(len(head)) if i not in seen]
    return [head[i] for i in order] + tail
