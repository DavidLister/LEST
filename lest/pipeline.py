import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunkers import get_chunker
from .chunkers.llm import LlmChunker
from .embedders import get_embedder
from .errors import LestError
from .extract import extract
from .sources import make_source
from .store import Store, db_path_for

log = logging.getLogger(__name__)

DEFAULT_CHUNKER = "llm"


@dataclass
class IndexStats:
    documents: int = 0
    files_indexed: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    files_skipped: int = 0  # no_text or error
    files_fallback: int = 0  # llm outline failed -> paragraph chunks
    chunks: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)


def _resolve_setting(store: Store, key: str, flag_value: str | None, default: str | None) -> str:
    """A DB records one value per setting; a differing explicit flag is an error."""
    recorded = store.get_meta(key)
    if recorded is None:
        value = flag_value or default
        if value is None:
            raise LestError(f"first index of this directory requires --{key}")
        store.set_meta(key, value)
        return value
    if flag_value is not None and flag_value != recorded:
        raise LestError(
            f"this index was built with {key}={recorded!r}; refusing to mix with "
            f"{flag_value!r}. Delete {store.db_path} to rebuild."
        )
    return recorded


def _make_catalog(embedder_name: str):
    """Catalog with the small embedding model for entity vectors; entity
    resolution is cheap and always lives on the default (A2000) endpoint."""
    from .catalog import Catalog
    from .embedders import get_embedder as _get

    small = _get(embedder_name, "qwen3-embedding:0.6b")
    return Catalog(embed=small.embed)


def index_directory(
    directory: Path,
    model: str | None = None,
    chunker_name: str | None = None,
    source_kind: str = "auto",
    embedder_name: str | None = None,
    db_base: Path | None = None,
    stop_at: float | None = None,
    limit: int | None = None,
) -> IndexStats:
    """Index (incrementally) DIRECTORY. `db_base` overrides the DB location
    (temporary --db A/B flag); `stop_at` is a unix timestamp and `limit` a
    files-indexed cap — after either, no new file is started (nightly budget /
    smoke tests); progress so far is kept and the next run resumes."""
    directory = directory.expanduser()
    if not directory.is_dir():
        raise LestError(f"not a directory: {directory}")

    store = Store(db_path_for(directory, base=db_base), create=True)
    catalog = None
    try:
        model = _resolve_setting(store, "model", model, default=None)
        chunker = get_chunker(_resolve_setting(store, "chunker", chunker_name, DEFAULT_CHUNKER))
        embedder_name = _resolve_setting(store, "embedder", embedder_name, "ollama")
        embedder = get_embedder(embedder_name, model)
        store.set_meta("source_dir", str(directory.resolve()))

        source = make_source(directory, source_kind)
        store.set_meta("source_type", type(source).__name__)

        llm_mode = isinstance(chunker, LlmChunker)
        enricher = None
        if llm_mode:
            chunker.client.ping()  # fail loudly up front, never degrade silently
            from .enrich import Enricher

            enricher = Enricher(chunker.client)
        catalog = _make_catalog(embedder_name)

        stats = IndexStats()
        known = store.file_states()
        seen_paths: set[str] = set()
        stopped = False

        for document in source.documents():
            if not document.attachments:
                continue
            stats.documents += 1
            document_id = store.upsert_document(document.key, document.title, document.meta)
            _ingest_authors(store, catalog, document_id, document.meta)
            enrich_this = enricher  # doc-level enrichment rides the first ok file
            for attachment in document.attachments:
                path_str = str(attachment.path)
                seen_paths.add(path_str)
                fp, status = known.get(path_str, (None, None))
                # llm_pending marks files whose LLM pass failed: always retried
                if fp == attachment.fingerprint and status != "llm_pending":
                    stats.files_unchanged += 1
                    continue
                if (stop_at is not None and time.time() >= stop_at) or (
                    limit is not None and stats.files_indexed >= limit
                ):
                    stopped = True
                    break
                status, chunks, doc_updates = _process_file(
                    attachment, document, chunker, enrich_this, catalog
                )
                vectors = embedder.embed([text for _, text in chunks]) if chunks else []
                if vectors:
                    store.ensure_vector_table(len(vectors[0]))
                store.replace_file(
                    document_id, path_str, attachment.fingerprint, status, chunks, vectors
                )
                if status == "ok":
                    stats.files_indexed += 1
                    stats.chunks += len(chunks)
                    if llm_mode and chunker.last_used_fallback:
                        stats.files_fallback += 1
                    for key, value in doc_updates.items():
                        if key == "tags":
                            store.set_doc_tags(document_id, value)
                        elif key == "doc_type":
                            store.set_doc_type(document_id, value)
                        elif key == "summary":
                            meta = dict(document.meta)
                            meta["summary"] = value
                            store.upsert_document(document.key, document.title, meta)
                    enrich_this = None  # only the first ok attachment carries doc-level chunks
                    log.info("indexed %s (%d chunks)", path_str, len(chunks))
                else:
                    stats.files_skipped += 1
                    stats.skipped.append((path_str, status))
                    log.warning("skipped %s (%s)", path_str, status)
            if stopped:
                break

        if not stopped:
            stale = set(known) - seen_paths
            if stale:
                store.remove_files(stale)
                stats.files_removed = len(stale)
                for path in sorted(stale):
                    log.info("removed %s (no longer in source)", path)
            removed_docs = store.prune_documents()
            if removed_docs:
                log.info("pruned %d empty documents", removed_docs)
        else:
            log.info("budget reached — stopping; next run resumes here")

        store.stamp_last_indexed()
        return stats
    finally:
        if catalog is not None:
            catalog.close()
        store.close()


def _ingest_authors(store: Store, catalog, document_id: int, meta: dict) -> None:
    creators = meta.get("creators")
    if not creators or store.schema_version < 2:
        return
    canonical = []
    for name in creators.split(";"):
        name = name.strip()
        if name:
            canonical.append(catalog.resolve_author(name))
    if canonical:
        store.set_doc_authors(document_id, canonical)


def _make_adjudicator(client):
    from .llm import ADJUDICATE_PROMPT, CHOICE_SCHEMA, SMALL_CTX

    def adjudicate(kind: str, proposed: str, candidates: list[str]) -> str | None:
        schema = dict(CHOICE_SCHEMA)
        schema["properties"] = {
            "choice": {"type": "string", "enum": [*candidates, "NEW"]}
        }
        label = {"tag": "topic tag", "doctype": "document type"}.get(kind, kind)
        result = client.call(
            ADJUDICATE_PROMPT.format(
                kind=label,
                proposed=proposed,
                candidates="\n".join(f"- {c}" for c in candidates),
            ),
            schema,
            num_ctx=SMALL_CTX,
            num_predict=64,
        )
        choice = (result or {}).get("choice")
        return None if not choice or choice == "NEW" else choice

    return adjudicate


def _process_file(
    attachment, document, chunker, enricher, catalog
) -> tuple[str, list[tuple[str, str]], dict]:
    """Returns (status, [(kind, text)...], doc-level updates)."""
    try:
        text = extract(attachment.path, attachment.content_type)
    except Exception as exc:  # corrupt/unreadable file — record, keep going
        log.debug("extraction failed for %s: %s", attachment.path, exc)
        return "error", [], {}
    if text is None:
        return "no_text", [], {}

    title = document.title
    if enricher is None:  # plain chunker path (or non-primary attachment)
        try:
            body = chunker.chunk(text, title=title)
        except TypeError:  # chunkers without the title kwarg (paragraph, custom)
            body = chunker.chunk(text)
        chunks = [("body", c) for c in body]
        return ("ok", chunks, {}) if chunks else ("no_text", [], {})

    from .errors import EnvironmentError_

    try:
        chunks = [("body", c) for c in chunker.chunk(text, title=title)]
        if not chunks:
            return "no_text", [], {}
        is_pdf = (attachment.content_type == "application/pdf"
                  or attachment.path.suffix.lower() == ".pdf")
        if is_pdf:
            chunks += enricher.figure_chunks(attachment.path, title)
        view_chunks, views = enricher.view_chunks(text, title)
        chunks += view_chunks
    except EnvironmentError_:
        raise  # endpoint died mid-run: abort loudly, don't mark files failed
    except Exception as exc:
        log.warning("LLM pass failed for %s: %s — will retry next run", attachment.path, exc)
        return "llm_pending", [], {}

    updates: dict = {}
    if views:
        summary = views.get("main_ideas", "")
        if summary:
            updates["summary"] = summary
        adjudicate = _make_adjudicator(enricher.client)
        raw_tags = enricher.propose_tags(title, _tag_summary(views), catalog.names("tag"))
        tags = []
        for raw in raw_tags:
            try:
                tags.append(catalog.resolve_term("tag", raw, adjudicate))
            except LestError:
                continue
        if tags:
            updates["tags"] = list(dict.fromkeys(tags))
        doc_type = _resolve_doc_type(enricher, catalog, title, text, document.meta, adjudicate)
        if doc_type:
            updates["doc_type"] = doc_type
    return "ok", chunks, updates


def _tag_summary(views: dict) -> str:
    return "\n".join(views.get(v, "") for v in ("main_ideas", "notable", "methods"))


def _resolve_doc_type(enricher, catalog, title, text, meta, adjudicate) -> str | None:
    from .catalog import GENERIC_TYPES

    vocab = catalog.names("doctype")
    hint = meta.get("type", "")
    proposal = enricher.propose_doc_type(title, text, hint, vocab)
    if proposal is None or proposal in GENERIC_TYPES:
        if vocab:  # forced choice from the existing taxonomy
            proposal = enricher.choose_doc_type(title, text, vocab)
        if proposal is None or proposal in GENERIC_TYPES:
            log.warning("no usable doc type for %r", title)
            return None
    return catalog.resolve_term("doctype", proposal, adjudicate)
