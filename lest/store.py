import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import sqlite_vec

from .errors import LestError

# Schema v2: chunk kinds (body/figure/view), FTS5 mirror of chunk text, and
# queryable per-document tags / authors / doc_type. v1 databases (no `kind`
# column) remain readable untouched; they are migrated in place only when
# opened for indexing (migration is additive: existing chunks and vectors are
# never modified).
SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS documents (
    id        INTEGER PRIMARY KEY,
    key       TEXT UNIQUE NOT NULL,
    title     TEXT NOT NULL,
    meta_json TEXT NOT NULL DEFAULT '{}',
    doc_type  TEXT
);
CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    path        TEXT UNIQUE NOT NULL,
    fingerprint TEXT NOT NULL,
    status      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id      INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id),
    seq     INTEGER NOT NULL,
    text    TEXT NOT NULL,
    kind    TEXT NOT NULL DEFAULT 'body'
);
CREATE TABLE IF NOT EXISTS doc_tags (
    document_id INTEGER NOT NULL REFERENCES documents(id),
    tag         TEXT NOT NULL,
    PRIMARY KEY (document_id, tag)
);
CREATE TABLE IF NOT EXISTS doc_authors (
    document_id INTEGER NOT NULL REFERENCES documents(id),
    author      TEXT NOT NULL,
    seq         INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (document_id, author)
);
CREATE INDEX IF NOT EXISTS idx_files_document ON files(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_doc_tags_tag ON doc_tags(tag);
CREATE INDEX IF NOT EXISTS idx_doc_authors_author ON doc_authors(author);
"""

_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    text, content='chunks', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""


def data_dir() -> Path:
    return Path(os.environ.get("LEST_DATA_DIR", "data"))


def db_path_for(source_dir: Path, base: Path | None = None) -> Path:
    """DB location for a source directory; `base` overrides the data dir
    (the temporary --db A/B flag)."""
    resolved = source_dir.resolve()
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:8]
    name = resolved.name or "root"
    return (base if base is not None else data_dir()) / f"{name}-{digest}.db"


@dataclass
class ChunkHit:
    similarity: float
    chunk_id: int
    chunk_text: str
    kind: str
    document_id: int
    document_key: str
    title: str
    meta_json: str
    doc_type: str | None = None


class Store:
    """One SQLite database (with sqlite-vec) per indexed directory."""

    def __init__(self, db_path: Path, create: bool = False):
        if not create and not db_path.exists():
            raise LestError(
                f"no index found at {db_path} — run `lest index <dir> --model <model>` first"
            )
        if create:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        existing = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'documents'"
        ).fetchone()
        if not existing:
            self.conn.executescript(_SCHEMA)
            self._create_fts()
            self.conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self.conn.commit()
        elif self._detect_version() < SCHEMA_VERSION and create:
            self._migrate_v1_to_v2()
        self.schema_version = self._detect_version()

    def _detect_version(self) -> int:
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(chunks)")}
        return 2 if "kind" in columns else 1

    def _create_fts(self) -> bool:
        try:
            self.conn.executescript(_FTS)
            return True
        except sqlite3.OperationalError:  # sqlite built without FTS5
            return False

    def _migrate_v1_to_v2(self) -> None:
        """Additive in-place migration; existing chunks/vectors are untouched."""
        with self.conn:
            self.conn.execute(
                "ALTER TABLE chunks ADD COLUMN kind TEXT NOT NULL DEFAULT 'body'"
            )
            self.conn.execute("ALTER TABLE documents ADD COLUMN doc_type TEXT")
            self.conn.executescript(_SCHEMA)  # doc_tags/doc_authors/indexes
            if self._create_fts():
                self.conn.execute(
                    "INSERT INTO chunk_fts(rowid, text) SELECT id, text FROM chunks"
                )
            self.conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def close(self) -> None:
        self.conn.close()

    # -- meta ---------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def stamp_last_indexed(self) -> None:
        self.set_meta("last_indexed", datetime.now(UTC).isoformat(timespec="seconds"))

    # -- vectors ------------------------------------------------------------

    @property
    def dim(self) -> int | None:
        value = self.get_meta("dim")
        return int(value) if value else None

    def ensure_vector_table(self, dim: int) -> None:
        known = self.dim
        if known is None:
            self.conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0("
                f"chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{dim}] distance_metric=cosine)"
            )
            self.set_meta("dim", str(dim))
        elif known != dim:
            raise LestError(
                f"embedding dimension changed ({known} -> {dim}); "
                f"delete {self.db_path} and re-index"
            )

    # -- sync ---------------------------------------------------------------

    def file_fingerprints(self) -> dict[str, str]:
        return dict(self.conn.execute("SELECT path, fingerprint FROM files"))

    def file_states(self) -> dict[str, tuple[str, str]]:
        return {
            path: (fp, status)
            for path, fp, status in self.conn.execute(
                "SELECT path, fingerprint, status FROM files"
            )
        }

    def upsert_document(self, key: str, title: str, meta: dict) -> int:
        row = self.conn.execute(
            "INSERT INTO documents (key, title, meta_json) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET title = excluded.title, "
            "meta_json = excluded.meta_json RETURNING id",
            (key, title, json.dumps(meta)),
        ).fetchone()
        return row[0]

    def set_doc_type(self, document_id: int, doc_type: str) -> None:
        self.conn.execute(
            "UPDATE documents SET doc_type = ? WHERE id = ?", (doc_type, document_id)
        )
        self.conn.commit()

    def set_doc_tags(self, document_id: int, tags: list[str]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM doc_tags WHERE document_id = ?", (document_id,))
            self.conn.executemany(
                "INSERT OR IGNORE INTO doc_tags (document_id, tag) VALUES (?, ?)",
                [(document_id, tag) for tag in tags],
            )

    def set_doc_authors(self, document_id: int, authors: list[str]) -> None:
        with self.conn:
            self.conn.execute(
                "DELETE FROM doc_authors WHERE document_id = ?", (document_id,)
            )
            self.conn.executemany(
                "INSERT OR IGNORE INTO doc_authors (document_id, author, seq) "
                "VALUES (?, ?, ?)",
                [(document_id, author, seq) for seq, author in enumerate(authors)],
            )

    def replace_file(
        self,
        document_id: int,
        path: str,
        fingerprint: str,
        status: str,
        chunks: list,
        vectors: list[list[float]],
    ) -> None:
        """Replace a file's chunks and vectors atomically (one transaction per file).

        `chunks` items are (kind, text) pairs; bare strings mean kind='body'.
        """
        assert len(chunks) == len(vectors)
        normalized = [("body", c) if isinstance(c, str) else c for c in chunks]
        with self.conn:
            self._delete_file_rows(path)
            file_id = self.conn.execute(
                "INSERT INTO files (document_id, path, fingerprint, status) VALUES (?, ?, ?, ?)",
                (document_id, path, fingerprint, status),
            ).lastrowid
            for seq, ((kind, text), vector) in enumerate(
                zip(normalized, vectors, strict=True)
            ):
                if self.schema_version >= 2:
                    chunk_id = self.conn.execute(
                        "INSERT INTO chunks (file_id, seq, text, kind) VALUES (?, ?, ?, ?)",
                        (file_id, seq, text, kind),
                    ).lastrowid
                else:
                    chunk_id = self.conn.execute(
                        "INSERT INTO chunks (file_id, seq, text) VALUES (?, ?, ?)",
                        (file_id, seq, text),
                    ).lastrowid
                self.conn.execute(
                    "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, sqlite_vec.serialize_float32(vector)),
                )

    def remove_files(self, paths: set[str]) -> None:
        with self.conn:
            for path in paths:
                self._delete_file_rows(path)

    def prune_documents(self) -> int:
        """Delete documents that no longer have any files; returns count removed."""
        with self.conn:
            if self.schema_version >= 2:
                for table in ("doc_tags", "doc_authors"):
                    self.conn.execute(
                        f"DELETE FROM {table} WHERE document_id NOT IN "
                        "(SELECT DISTINCT document_id FROM files)"
                    )
            cursor = self.conn.execute(
                "DELETE FROM documents WHERE id NOT IN (SELECT DISTINCT document_id FROM files)"
            )
        return cursor.rowcount

    def _delete_file_rows(self, path: str) -> None:
        row = self.conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
        if row is None:
            return
        file_id = row[0]
        if self._has_vector_table():
            chunk_ids = self.conn.execute(
                "SELECT id FROM chunks WHERE file_id = ?", (file_id,)
            ).fetchall()
            self.conn.executemany(
                "DELETE FROM chunk_vectors WHERE chunk_id = ?", chunk_ids
            )
        self.conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        self.conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    def _has_vector_table(self) -> bool:
        return self._has_table("chunk_vectors")

    @property
    def has_fts(self) -> bool:
        return self._has_table("chunk_fts")

    def _has_table(self, name: str) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
            ).fetchone()
            is not None
        )

    # -- query --------------------------------------------------------------

    def _hit_select(self, inner: str) -> str:
        kind = "c.kind" if self.schema_version >= 2 else "'body'"
        doc_type = "d.doc_type" if self.schema_version >= 2 else "NULL"
        return f"""
            SELECT v.score, c.id, c.text, {kind}, d.id, d.key, d.title, d.meta_json,
                   {doc_type}
            FROM ({inner}) v
            JOIN chunks c ON c.id = v.chunk_id
            JOIN files f ON f.id = c.file_id
            JOIN documents d ON d.id = f.document_id
            """

    @staticmethod
    def _to_hits(rows) -> list[ChunkHit]:
        return [
            ChunkHit(
                similarity=score,
                chunk_id=chunk_id,
                chunk_text=text,
                kind=kind,
                document_id=doc_id,
                document_key=key,
                title=title,
                meta_json=meta_json,
                doc_type=doc_type,
            )
            for score, chunk_id, text, kind, doc_id, key, title, meta_json, doc_type in rows
        ]

    def knn(self, query_vector: list[float], k: int) -> list[ChunkHit]:
        if not self._has_vector_table():
            return []
        rows = self.conn.execute(
            self._hit_select(
                "SELECT chunk_id, 1.0 - distance AS score FROM chunk_vectors "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance"
            ),
            (sqlite_vec.serialize_float32(query_vector), int(k)),
        ).fetchall()
        return self._to_hits(rows)

    def fts_search(self, query: str, k: int) -> list[ChunkHit]:
        """BM25 full-text hits; `similarity` holds -bm25 (higher is better)."""
        if not self.has_fts:
            return []
        tokens = [t for t in "".join(
            ch if ch.isalnum() else " " for ch in query
        ).split() if len(t) > 1]
        if not tokens:
            return []
        match = " OR ".join(f'"{t}"' for t in tokens)
        rows = self.conn.execute(
            self._hit_select(
                "SELECT rowid AS chunk_id, -bm25(chunk_fts) AS score FROM chunk_fts "
                "WHERE chunk_fts MATCH ? ORDER BY bm25(chunk_fts) LIMIT ?"
            ),
            (match, int(k)),
        ).fetchall()
        return self._to_hits(rows)

    def filter_documents(
        self,
        tag_groups: list[set[str]] | None = None,
        author_groups: list[set[str]] | None = None,
        type_group: set[str] | None = None,
    ) -> set[int] | None:
        """Document ids matching all facets, or None when unfiltered.

        Each group is a set of equivalent strings (canonical + aliases): a
        document matches a group if it carries ANY of them; groups AND.
        """
        if not (tag_groups or author_groups or type_group):
            return None
        if self.schema_version < 2:
            raise LestError(
                "this index predates tag/author/type support — re-index it, "
                "or search without filters"
            )
        allowed: set[int] | None = None

        def restrict(sql: str, terms: set[str]) -> None:
            nonlocal allowed
            marks = ",".join("?" * len(terms))
            ids = {i for (i,) in self.conn.execute(sql.format(marks=marks), list(terms))}
            allowed = ids if allowed is None else (allowed & ids)

        for group in tag_groups or []:
            restrict("SELECT document_id FROM doc_tags WHERE tag IN ({marks})", group)
        for group in author_groups or []:
            restrict(
                "SELECT document_id FROM doc_authors WHERE author IN ({marks})", group
            )
        if type_group:
            restrict("SELECT id FROM documents WHERE doc_type IN ({marks})", type_group)
        return allowed if allowed is not None else set()

    def document_paths(self, document_id: int) -> list[str]:
        return [
            path
            for (path,) in self.conn.execute(
                "SELECT path FROM files WHERE document_id = ? ORDER BY path", (document_id,)
            )
        ]

    def document_tags(self, document_id: int) -> list[str]:
        if self.schema_version < 2:
            return []
        return [
            tag
            for (tag,) in self.conn.execute(
                "SELECT tag FROM doc_tags WHERE document_id = ? ORDER BY tag",
                (document_id,),
            )
        ]

    def document_authors(self, document_id: int) -> list[str]:
        if self.schema_version < 2:
            return []
        return [
            author
            for (author,) in self.conn.execute(
                "SELECT author FROM doc_authors WHERE document_id = ? ORDER BY seq",
                (document_id,),
            )
        ]

    # -- status -------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        counts = {}
        for table in ("documents", "files", "chunks"):
            (counts[table],) = self.conn.execute(f"SELECT count(*) FROM {table}").fetchone()
        return counts

    def kind_counts(self) -> dict[str, int]:
        if self.schema_version < 2:
            return {}
        return dict(
            self.conn.execute("SELECT kind, count(*) FROM chunks GROUP BY kind")
        )

    def skipped_files(self) -> list[tuple[str, str]]:
        return self.conn.execute(
            "SELECT path, status FROM files WHERE status != 'ok' ORDER BY path"
        ).fetchall()
