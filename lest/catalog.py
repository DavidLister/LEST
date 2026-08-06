"""Adaptive entity catalog: tags, authors, and doc types as self-extending
vocabularies with aliases and a human-reviewed merge queue.

Lives at <data dir>/catalog.db — shared across DB variants (the --db flag
moves the chunk store, never the catalog) and never committed (data/ is
gitignored). Resolution strategy per kind:

- tags / doc types: exact & alias hits, then embedding nearest-neighbour
  (qwen3-embedding:0.6b on the A2000). High similarity auto-maps (recorded as
  an alias); a gray zone is adjudicated by the caller-supplied LLM hook; low
  similarity coins a new entity.
- authors: names are a string problem, not a semantic one — unicode-folded
  surname + initials compatibility auto-map, string-similarity gray zone.
  Gray-zone author pairs are never auto-merged (two real people may share
  similar names): the name is kept distinct and a merge proposal is queued.

Merges of existing entities are NEVER applied autonomously — they enter the
queue for `lest catalog review`. Approved merges become aliases, so they
retro-apply through alias-aware search filters without re-tagging documents.
"""

import difflib
import struct
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from sqlite3 import connect

from .errors import LestError
from .store import data_dir

KINDS = ("tag", "author", "doctype")

# Similarity bands for embedded kinds (cosine). Initial values chosen from the
# pilot's freeform-tag dump calibration; expected to be tuned as nightly runs
# accumulate.
SIM_AUTO = 0.92
SIM_GRAY = 0.80

# Author string-similarity bands (difflib ratio over folded full names).
AUTHOR_PROPOSE = 0.84

GENERIC_TYPES = {
    "misc", "miscellaneous", "other", "document", "documents", "text",
    "file", "general", "unknown", "n/a", "none",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS entities (
    id    INTEGER PRIMARY KEY,
    kind  TEXT NOT NULL,
    name  TEXT NOT NULL,
    usage INTEGER NOT NULL DEFAULT 0,
    created TEXT NOT NULL,
    UNIQUE (kind, name)
);
CREATE TABLE IF NOT EXISTS aliases (
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    kind      TEXT NOT NULL,
    alias     TEXT NOT NULL,
    UNIQUE (kind, alias)
);
CREATE TABLE IF NOT EXISTS vectors (
    entity_id INTEGER PRIMARY KEY REFERENCES entities(id),
    dim       INTEGER NOT NULL,
    vec       BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS merges (
    id        INTEGER PRIMARY KEY,
    kind      TEXT NOT NULL,
    keep_name TEXT NOT NULL,
    drop_name TEXT NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    status    TEXT NOT NULL DEFAULT 'pending',
    created   TEXT NOT NULL
);
"""

# The adjudicator receives (kind, proposed, candidate names) and returns the
# duplicated existing name, or None for "genuinely new".
Adjudicator = Callable[[str, str, list[str]], str | None]


def fold(text: str) -> str:
    """Lowercase, accent-stripped, whitespace-collapsed."""
    decomposed = unicodedata.normalize("NFKD", text)
    return " ".join(
        "".join(c for c in decomposed if not unicodedata.combining(c)).lower().split()
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Catalog:
    def __init__(self, path: Path | None = None, embed: Callable | None = None):
        """`embed(texts) -> list[vector]` is injected (None disables embedding
        resolution — exact/alias matching still works)."""
        self.path = path if path is not None else data_dir() / "catalog.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = connect(self.path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        self._embed = embed

    def close(self) -> None:
        self.conn.close()

    # -- basics -------------------------------------------------------------

    def names(self, kind: str) -> list[str]:
        return [
            name
            for (name,) in self.conn.execute(
                "SELECT name FROM entities WHERE kind = ? ORDER BY usage DESC, name",
                (kind,),
            )
        ]

    def counts(self, kind: str) -> list[tuple[str, int]]:
        return self.conn.execute(
            "SELECT name, usage FROM entities WHERE kind = ? ORDER BY usage DESC, name",
            (kind,),
        ).fetchall()

    def aliases_of(self, kind: str, name: str) -> list[str]:
        return [
            alias
            for (alias,) in self.conn.execute(
                "SELECT a.alias FROM aliases a JOIN entities e ON e.id = a.entity_id "
                "WHERE e.kind = ? AND e.name = ? ORDER BY a.alias",
                (kind, name),
            )
        ]

    def search_terms(self, kind: str, name: str) -> set[str]:
        """The canonical name plus all aliases — for alias-aware filtering."""
        return {name, *self.aliases_of(kind, name)}

    def _canonical(self, kind: str, name: str) -> str | None:
        row = self.conn.execute(
            "SELECT name FROM entities WHERE kind = ? AND name = ?", (kind, name)
        ).fetchone()
        if row:
            return row[0]
        row = self.conn.execute(
            "SELECT e.name FROM aliases a JOIN entities e ON e.id = a.entity_id "
            "WHERE a.kind = ? AND a.alias = ?",
            (kind, name),
        ).fetchone()
        return row[0] if row else None

    def _bump(self, kind: str, name: str) -> str:
        self.conn.execute(
            "UPDATE entities SET usage = usage + 1 WHERE kind = ? AND name = ?",
            (kind, name),
        )
        self.conn.commit()
        return name

    def _create(self, kind: str, name: str, vector: list[float] | None = None) -> str:
        cursor = self.conn.execute(
            "INSERT OR IGNORE INTO entities (kind, name, usage, created) "
            "VALUES (?, ?, 1, ?)",
            (kind, name, _now()),
        )
        if cursor.rowcount and vector is not None:
            entity_id = cursor.lastrowid
            self.conn.execute(
                "INSERT OR REPLACE INTO vectors (entity_id, dim, vec) VALUES (?, ?, ?)",
                (entity_id, len(vector), struct.pack(f"{len(vector)}f", *vector)),
            )
        self.conn.commit()
        return name

    def add_alias(self, kind: str, canonical: str, alias: str) -> None:
        row = self.conn.execute(
            "SELECT id FROM entities WHERE kind = ? AND name = ?", (kind, canonical)
        ).fetchone()
        if row is None:
            raise LestError(f"unknown {kind} {canonical!r}")
        self.conn.execute(
            "INSERT OR IGNORE INTO aliases (entity_id, kind, alias) VALUES (?, ?, ?)",
            (row[0], kind, alias),
        )
        self.conn.commit()

    # -- embedding NN -------------------------------------------------------

    def _vectors(self, kind: str) -> list[tuple[str, list[float]]]:
        rows = self.conn.execute(
            "SELECT e.name, v.dim, v.vec FROM entities e JOIN vectors v "
            "ON v.entity_id = e.id WHERE e.kind = ?",
            (kind,),
        ).fetchall()
        return [(name, list(struct.unpack(f"{dim}f", blob))) for name, dim, blob in rows]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / max(na * nb, 1e-12)

    def _nearest(self, kind: str, vector: list[float], n: int = 5) -> list[tuple[str, float]]:
        scored = [
            (name, self._cosine(vector, vec)) for name, vec in self._vectors(kind)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:n]

    # -- resolution: tags & doc types ---------------------------------------

    def resolve_term(
        self, kind: str, raw: str, adjudicate: Adjudicator | None = None
    ) -> str:
        """Resolve a proposed tag/doctype to a canonical entity, creating one
        when genuinely new. Never merges existing entities (that is queued)."""
        name = fold(raw)
        if not name:
            raise LestError(f"empty {kind} name")
        hit = self._canonical(kind, name)
        if hit:
            return self._bump(kind, hit)
        vector = self._embed([name])[0] if self._embed else None
        if vector is not None:
            neighbours = self._nearest(kind, vector)
            if neighbours and neighbours[0][1] >= SIM_AUTO:
                canonical = neighbours[0][0]
                self.add_alias(kind, canonical, name)
                return self._bump(kind, canonical)
            gray = [n for n, sim in neighbours if sim >= SIM_GRAY]
            if gray and adjudicate:
                verdict = adjudicate(kind, name, gray)
                if verdict and self._canonical(kind, fold(verdict)):
                    canonical = self._canonical(kind, fold(verdict))
                    self.add_alias(kind, canonical, name)
                    return self._bump(kind, canonical)
        return self._create(kind, name, vector)

    # -- read-only lookups (search side: never create) -----------------------

    def lookup_term(self, kind: str, raw: str) -> set[str] | None:
        """Alias-aware term group for filtering, or None when unknown."""
        name = fold(raw)
        canonical = self._canonical(kind, name)
        return self.search_terms(kind, canonical) if canonical else None

    def lookup_author(self, raw: str) -> set[str] | None:
        """Fuzzy author lookup: exact/alias, then surname+initials, then best
        string match. Never creates entities."""
        name = " ".join(raw.split())
        canonical = self._canonical("author", name)
        if canonical is None:  # try "Last, First" from "First Last" input
            words = name.split()
            if "," not in name and len(words) > 1:
                flipped = f"{words[-1]}, {' '.join(words[:-1])}"
                canonical = self._canonical("author", flipped)
                name = flipped if canonical else name
        if canonical is None:
            last, given = self._split_author(name)
            folded = f"{last} {given}".strip()
            best, best_ratio = None, 0.0
            for existing in self.names("author"):
                e_last, e_given = self._split_author(existing)
                if e_last == last and self._given_compatible(given, e_given):
                    canonical = existing
                    break
                ratio = difflib.SequenceMatcher(
                    None, folded, f"{e_last} {e_given}".strip()
                ).ratio()
                if ratio > best_ratio:
                    best, best_ratio = existing, ratio
            if canonical is None and best_ratio >= AUTHOR_PROPOSE:
                canonical = best
        return self.search_terms("author", canonical) if canonical else None

    # -- resolution: authors ------------------------------------------------

    @staticmethod
    def _split_author(name: str) -> tuple[str, str]:
        """'Last, First M.' -> (folded last, folded given); no-comma names are
        treated as a bare surname."""
        last, _, given = name.partition(",")
        return fold(last), fold(given.replace(".", " "))

    @staticmethod
    def _given_compatible(a: str, b: str) -> bool:
        """'j' ~ 'jorg', 'j k' ~ 'jorg k' — initials compatible with full names."""
        if not a or not b:
            return True  # missing given name never contradicts
        for wa, wb in zip(a.split(), b.split(), strict=False):
            if not (wa.startswith(wb) or wb.startswith(wa)):
                return False
        return True

    def resolve_author(self, raw: str) -> str:
        name = " ".join(raw.split())
        if not name:
            raise LestError("empty author name")
        hit = self._canonical("author", name)
        if hit:
            return self._bump("author", hit)
        last, given = self._split_author(name)
        folded = f"{last} {given}".strip()
        best, best_ratio = None, 0.0
        for existing in self.names("author"):
            e_last, e_given = self._split_author(existing)
            if e_last == last and self._given_compatible(given, e_given):
                # same surname, compatible given names: same person
                canonical = existing
                # keep the more complete spelling as canonical
                if len(given) > len(e_given):
                    self._rename("author", existing, name)
                    canonical = name
                self.add_alias("author", canonical, name if canonical != name else existing)
                return self._bump("author", canonical)
            ratio = difflib.SequenceMatcher(
                None, folded, f"{e_last} {e_given}".strip()
            ).ratio()
            if ratio > best_ratio:
                best, best_ratio = existing, ratio
        created = self._create("author", name)
        if best and best_ratio >= AUTHOR_PROPOSE:
            self.propose_merge(
                "author", best, name,
                f"string similarity {best_ratio:.2f} — possible variant spelling",
            )
        return created

    def _rename(self, kind: str, old: str, new: str) -> None:
        row = self.conn.execute(
            "SELECT id FROM entities WHERE kind = ? AND name = ?", (kind, old)
        ).fetchone()
        if row is None:
            return
        self.conn.execute(
            "UPDATE entities SET name = ? WHERE id = ?", (new, row[0])
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO aliases (entity_id, kind, alias) VALUES (?, ?, ?)",
            (row[0], kind, old),
        )
        self.conn.commit()

    # -- merge queue ---------------------------------------------------------

    def propose_merge(self, kind: str, keep: str, drop: str, rationale: str = "") -> None:
        exists = self.conn.execute(
            "SELECT 1 FROM merges WHERE kind = ? AND keep_name = ? AND drop_name = ? "
            "AND status = 'pending'",
            (kind, keep, drop),
        ).fetchone()
        if exists or keep == drop:
            return
        self.conn.execute(
            "INSERT INTO merges (kind, keep_name, drop_name, rationale, created) "
            "VALUES (?, ?, ?, ?, ?)",
            (kind, keep, drop, rationale, _now()),
        )
        self.conn.commit()

    def pending_merges(self) -> list[tuple[int, str, str, str, str]]:
        return self.conn.execute(
            "SELECT id, kind, keep_name, drop_name, rationale FROM merges "
            "WHERE status = 'pending' ORDER BY id"
        ).fetchall()

    def apply_merge(self, merge_id: int, approve: bool) -> None:
        row = self.conn.execute(
            "SELECT kind, keep_name, drop_name FROM merges WHERE id = ? "
            "AND status = 'pending'",
            (merge_id,),
        ).fetchone()
        if row is None:
            raise LestError(f"no pending merge #{merge_id}")
        kind, keep, drop = row
        if approve:
            drop_row = self.conn.execute(
                "SELECT id, usage FROM entities WHERE kind = ? AND name = ?",
                (kind, drop),
            ).fetchone()
            keep_row = self.conn.execute(
                "SELECT id FROM entities WHERE kind = ? AND name = ?", (kind, keep)
            ).fetchone()
            if drop_row and keep_row:
                drop_id, drop_usage = drop_row
                (keep_id,) = keep_row
                self.conn.execute(
                    "UPDATE OR IGNORE aliases SET entity_id = ? WHERE entity_id = ?",
                    (keep_id, drop_id),
                )
                self.conn.execute("DELETE FROM aliases WHERE entity_id = ?", (drop_id,))
                self.conn.execute("DELETE FROM vectors WHERE entity_id = ?", (drop_id,))
                self.conn.execute("DELETE FROM entities WHERE id = ?", (drop_id,))
                self.conn.execute(
                    "UPDATE entities SET usage = usage + ? WHERE id = ?",
                    (drop_usage, keep_id),
                )
                self.conn.execute(
                    "INSERT OR IGNORE INTO aliases (entity_id, kind, alias) "
                    "VALUES (?, ?, ?)",
                    (keep_id, kind, drop),
                )
        self.conn.execute(
            "UPDATE merges SET status = ? WHERE id = ?",
            ("approved" if approve else "rejected", merge_id),
        )
        self.conn.commit()

    # -- seeding ------------------------------------------------------------

    def seed_tags(self, vocab_file: Path) -> int:
        """Seed the tag vocabulary from a proposed-vocab file (idempotent).
        Lines: `tag  # = merged variants...`; variants become aliases."""
        added = 0
        for line in vocab_file.read_text().splitlines():
            body, _, comment = line.partition("#")
            name = fold(body)
            if not name:
                continue
            vector = self._embed([name])[0] if self._embed else None
            if self._canonical("tag", name) is None:
                self._create("tag", name, vector)
                self.conn.execute(
                    "UPDATE entities SET usage = 0 WHERE kind = 'tag' AND name = ?",
                    (name,),
                )
                added += 1
            variants = comment.partition("=")[2]
            for variant in variants.split(","):
                alias = fold(variant.partition("(")[0])
                if alias and alias != name and self._canonical("tag", alias) is None:
                    self.add_alias("tag", name, alias)
        self.conn.commit()
        return added
