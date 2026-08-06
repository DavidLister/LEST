import logging
from dataclasses import dataclass, field
from pathlib import Path

from .chunkers import get_chunker
from .embedders import get_embedder
from .errors import LestError
from .extract import extract
from .sources import make_source
from .store import Store, db_path_for

log = logging.getLogger(__name__)


@dataclass
class IndexStats:
    documents: int = 0
    files_indexed: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    files_skipped: int = 0  # no_text or error
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


def index_directory(
    directory: Path,
    model: str | None = None,
    chunker_name: str | None = None,
    source_kind: str = "auto",
    embedder_name: str | None = None,
) -> IndexStats:
    directory = directory.expanduser()
    if not directory.is_dir():
        raise LestError(f"not a directory: {directory}")

    store = Store(db_path_for(directory), create=True)
    try:
        model = _resolve_setting(store, "model", model, default=None)
        chunker = get_chunker(_resolve_setting(store, "chunker", chunker_name, "paragraph"))
        embedder = get_embedder(_resolve_setting(store, "embedder", embedder_name, "ollama"), model)
        store.set_meta("source_dir", str(directory.resolve()))

        source = make_source(directory, source_kind)
        store.set_meta("source_type", type(source).__name__)

        stats = IndexStats()
        known = store.file_fingerprints()
        seen_paths: set[str] = set()

        for document in source.documents():
            if not document.attachments:
                continue
            stats.documents += 1
            document_id = store.upsert_document(document.key, document.title, document.meta)
            for attachment in document.attachments:
                path_str = str(attachment.path)
                seen_paths.add(path_str)
                if known.get(path_str) == attachment.fingerprint:
                    stats.files_unchanged += 1
                    continue
                status, chunks, vectors = _process_file(attachment, chunker, embedder)
                if chunks:
                    store.ensure_vector_table(len(vectors[0]))
                store.replace_file(
                    document_id, path_str, attachment.fingerprint, status, chunks, vectors
                )
                if status == "ok":
                    stats.files_indexed += 1
                    stats.chunks += len(chunks)
                    log.info("indexed %s (%d chunks)", path_str, len(chunks))
                else:
                    stats.files_skipped += 1
                    stats.skipped.append((path_str, status))
                    log.warning("skipped %s (%s)", path_str, status)

        stale = set(known) - seen_paths
        if stale:
            store.remove_files(stale)
            stats.files_removed = len(stale)
            for path in sorted(stale):
                log.info("removed %s (no longer in source)", path)
        removed_docs = store.prune_documents()
        if removed_docs:
            log.info("pruned %d empty documents", removed_docs)

        store.stamp_last_indexed()
        return stats
    finally:
        store.close()


def _process_file(attachment, chunker, embedder) -> tuple[str, list[str], list[list[float]]]:
    try:
        text = extract(attachment.path, attachment.content_type)
    except Exception as exc:  # corrupt/unreadable file — record, keep going
        log.debug("extraction failed for %s: %s", attachment.path, exc)
        return "error", [], []
    if text is None:
        return "no_text", [], []
    chunks = chunker.chunk(text)
    if not chunks:
        return "no_text", [], []
    return "ok", chunks, embedder.embed(chunks)
