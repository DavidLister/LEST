import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import sqlite_vec

from .errors import LestError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS documents (
    id        INTEGER PRIMARY KEY,
    key       TEXT UNIQUE NOT NULL,
    title     TEXT NOT NULL,
    meta_json TEXT NOT NULL DEFAULT '{}'
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
    text    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_document ON files(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
"""


def data_dir() -> Path:
    return Path(os.environ.get("LEST_DATA_DIR", "data"))


def db_path_for(source_dir: Path) -> Path:
    resolved = source_dir.resolve()
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:8]
    name = resolved.name or "root"
    return data_dir() / f"{name}-{digest}.db"


@dataclass
class ChunkHit:
    similarity: float
    chunk_text: str
    document_id: int
    document_key: str
    title: str
    meta_json: str


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
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

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

    def upsert_document(self, key: str, title: str, meta: dict) -> int:
        row = self.conn.execute(
            "INSERT INTO documents (key, title, meta_json) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET title = excluded.title, "
            "meta_json = excluded.meta_json RETURNING id",
            (key, title, json.dumps(meta)),
        ).fetchone()
        return row[0]

    def replace_file(
        self,
        document_id: int,
        path: str,
        fingerprint: str,
        status: str,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> None:
        """Replace a file's chunks and vectors atomically (one transaction per file)."""
        assert len(chunks) == len(vectors)
        with self.conn:
            self._delete_file_rows(path)
            file_id = self.conn.execute(
                "INSERT INTO files (document_id, path, fingerprint, status) VALUES (?, ?, ?, ?)",
                (document_id, path, fingerprint, status),
            ).lastrowid
            for seq, (text, vector) in enumerate(zip(chunks, vectors, strict=True)):
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
        return (
            self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunk_vectors'"
            ).fetchone()
            is not None
        )

    # -- query --------------------------------------------------------------

    def knn(self, query_vector: list[float], k: int) -> list[ChunkHit]:
        if not self._has_vector_table():
            return []
        rows = self.conn.execute(
            """
            SELECT v.distance, c.text, d.id, d.key, d.title, d.meta_json
            FROM (SELECT chunk_id, distance FROM chunk_vectors
                  WHERE embedding MATCH ? AND k = ? ORDER BY distance) v
            JOIN chunks c ON c.id = v.chunk_id
            JOIN files f ON f.id = c.file_id
            JOIN documents d ON d.id = f.document_id
            """,
            (sqlite_vec.serialize_float32(query_vector), int(k)),
        ).fetchall()
        return [
            ChunkHit(
                similarity=1.0 - distance,
                chunk_text=text,
                document_id=doc_id,
                document_key=key,
                title=title,
                meta_json=meta_json,
            )
            for distance, text, doc_id, key, title, meta_json in rows
        ]

    def document_paths(self, document_id: int) -> list[str]:
        return [
            path
            for (path,) in self.conn.execute(
                "SELECT path FROM files WHERE document_id = ? ORDER BY path", (document_id,)
            )
        ]

    # -- status -------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        counts = {}
        for table in ("documents", "files", "chunks"):
            (counts[table],) = self.conn.execute(f"SELECT count(*) FROM {table}").fetchone()
        return counts

    def skipped_files(self) -> list[tuple[str, str]]:
        return self.conn.execute(
            "SELECT path, status FROM files WHERE status != 'ok' ORDER BY path"
        ).fetchall()
