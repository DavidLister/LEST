"""Schema v2: chunk kinds, FTS, facet filters, v1 read-compat and migration."""

import sqlite3

import pytest
import sqlite_vec

from lest.errors import LestError
from lest.store import Store

V1_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE documents (
    id INTEGER PRIMARY KEY, key TEXT UNIQUE NOT NULL, title TEXT NOT NULL,
    meta_json TEXT NOT NULL DEFAULT '{}');
CREATE TABLE files (
    id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id),
    path TEXT UNIQUE NOT NULL, fingerprint TEXT NOT NULL, status TEXT NOT NULL);
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL REFERENCES files(id),
    seq INTEGER NOT NULL, text TEXT NOT NULL);
"""


def vec(*values):
    return list(values)


def make_v1_db(path):
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(V1_SCHEMA)
    conn.execute(
        "CREATE VIRTUAL TABLE chunk_vectors USING vec0("
        "chunk_id INTEGER PRIMARY KEY, embedding FLOAT[4] distance_metric=cosine)"
    )
    conn.execute("INSERT INTO meta VALUES ('dim', '4')")
    conn.execute("INSERT INTO documents (id, key, title) VALUES (1, 'k', 'Old Doc')")
    conn.execute("INSERT INTO files VALUES (1, 1, '/tmp/old.pdf', '1:1', 'ok')")
    conn.execute("INSERT INTO chunks VALUES (1, 1, 0, 'legacy chunk about crystals')")
    conn.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (1, ?)",
        (sqlite_vec.serialize_float32(vec(1, 0, 0, 0)),),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "v2.db", create=True)
    yield s
    s.close()


def fill_kinds(store):
    doc_id = store.upsert_document("doc1", "Paper One", {"year": "2020"})
    store.ensure_vector_table(4)
    store.replace_file(
        doc_id, "/tmp/p1.pdf", "1:1", "ok",
        [("body", "crystals grow on glass substrates"),
         ("figure", "[figure p.3] SEM image of grains"),
         ("view", "[main_ideas] fastest grains win the race")],
        [vec(1, 0, 0, 0), vec(0, 1, 0, 0), vec(0, 0, 1, 0)],
    )
    return doc_id


def test_kinds_stored_and_counted(store):
    fill_kinds(store)
    assert store.kind_counts() == {"body": 1, "figure": 1, "view": 1}
    hits = store.knn(vec(0, 0, 1, 0), k=1)
    assert hits[0].kind == "view"


def test_fts_search_and_sync(store):
    doc_id = fill_kinds(store)
    hits = store.fts_search("grains SEM image", k=10)
    assert hits
    assert any("SEM" in h.chunk_text for h in hits)
    # replacing the file keeps FTS in sync (triggers)
    store.replace_file(doc_id, "/tmp/p1.pdf", "2:2", "ok",
                       [("body", "entirely new content")], [vec(1, 1, 0, 0)])
    assert store.fts_search("SEM image", k=10) == []
    assert store.fts_search("entirely new", k=10)


def test_fts_ignores_operators(store):
    fill_kinds(store)
    # quotes/AND/NOT etc. in a natural query must not crash FTS syntax
    assert store.fts_search('grains AND "glass" NOT (win)', k=5) is not None
    assert store.fts_search("...", k=5) == []


def test_facet_filters(store):
    doc_id = fill_kinds(store)
    other = store.upsert_document("doc2", "Paper Two", {})
    store.replace_file(other, "/tmp/p2.pdf", "1:1", "ok",
                       [("body", "unrelated")], [vec(0, 0, 0, 1)])
    store.set_doc_tags(doc_id, ["mocvd", "crystal growth"])
    store.set_doc_authors(doc_id, ["Neugebauer, Jörg"])
    store.set_doc_type(doc_id, "research article")

    assert store.filter_documents(tag_groups=[{"mocvd", "movpe"}]) == {doc_id}
    assert store.filter_documents(tag_groups=[{"movpe"}]) == set()
    assert store.filter_documents(
        tag_groups=[{"mocvd"}], author_groups=[{"Neugebauer, Jörg"}]
    ) == {doc_id}
    assert store.filter_documents(type_group={"research article"}) == {doc_id}
    assert store.filter_documents() is None


def test_v1_readonly_compat(tmp_path):
    db = tmp_path / "old.db"
    make_v1_db(db)
    store = Store(db)  # no create: must not migrate
    try:
        assert store.schema_version == 1
        assert not store.has_fts
        hits = store.knn(vec(1, 0, 0, 0), k=5)
        assert hits[0].chunk_text == "legacy chunk about crystals"
        assert hits[0].kind == "body"
        assert hits[0].doc_type is None
        with pytest.raises(LestError, match="predates"):
            store.filter_documents(tag_groups=[{"x"}])
    finally:
        store.close()
    # untouched: still v1 on disk
    conn = sqlite3.connect(db)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
    conn.close()
    assert "kind" not in columns


def test_v1_migrates_on_write_open(tmp_path):
    db = tmp_path / "old.db"
    make_v1_db(db)
    store = Store(db, create=True)
    try:
        assert store.schema_version == 2
        # existing chunk kept, FTS backfilled
        assert store.fts_search("legacy crystals", k=5)
        hits = store.knn(vec(1, 0, 0, 0), k=5)
        assert hits[0].chunk_text == "legacy chunk about crystals"
        # v2 writes now work
        doc_id = store.upsert_document("k2", "New Doc", {})
        store.replace_file(doc_id, "/tmp/new.pdf", "1:1", "ok",
                           [("view", "a summary")], [vec(0, 1, 0, 0)])
        assert store.kind_counts()["view"] == 1
    finally:
        store.close()
