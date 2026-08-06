import pytest

from lest.errors import LestError
from lest.store import Store, db_path_for


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db", create=True)
    yield s
    s.close()


def vec(*values):
    return list(values)


def fill(store, doc_key="doc", path="/tmp/f1", fingerprint="1:1", chunks=None, vectors=None):
    doc_id = store.upsert_document(doc_key, f"Title of {doc_key}", {"year": "2020"})
    chunks = chunks if chunks is not None else ["chunk a", "chunk b"]
    vectors = vectors if vectors is not None else [vec(1, 0, 0, 0), vec(0, 1, 0, 0)]
    store.ensure_vector_table(len(vectors[0]))
    store.replace_file(doc_id, path, fingerprint, "ok", chunks, vectors)
    return doc_id


def test_open_missing_db_errors(tmp_path):
    with pytest.raises(LestError, match="no index"):
        Store(tmp_path / "nope.db")


def test_db_path_stable_and_distinct(tmp_path, monkeypatch):
    monkeypatch.setenv("LEST_DATA_DIR", str(tmp_path))
    a, b = tmp_path / "corpusA", tmp_path / "corpusB"
    a.mkdir(), b.mkdir()
    assert db_path_for(a) == db_path_for(a)
    assert db_path_for(a) != db_path_for(b)
    assert db_path_for(a).name.startswith("corpusA-")


def test_knn_cosine_ordering(store):
    fill(store)
    hits = store.knn(vec(0.9, 0.1, 0, 0), k=10)
    assert [h.chunk_text for h in hits] == ["chunk a", "chunk b"]
    assert hits[0].similarity > hits[1].similarity
    assert hits[0].similarity == pytest.approx(1.0, abs=0.01)
    assert hits[0].title == "Title of doc"


def test_dimension_locked(store):
    fill(store)
    with pytest.raises(LestError, match="dimension changed"):
        store.ensure_vector_table(8)


def test_resync_replaces_file(store):
    doc_id = fill(store)
    store.replace_file(doc_id, "/tmp/f1", "2:2", "ok", ["new chunk"], [vec(0, 0, 1, 0)])
    hits = store.knn(vec(0, 0, 1, 0), k=10)
    assert [h.chunk_text for h in hits] == ["new chunk"]
    assert store.file_fingerprints() == {"/tmp/f1": "2:2"}


def test_remove_and_prune(store):
    fill(store, doc_key="doc1", path="/tmp/f1")
    fill(store, doc_key="doc2", path="/tmp/f2", vectors=[vec(0, 0, 1, 0), vec(0, 0, 0, 1)])
    store.remove_files({"/tmp/f1"})
    assert store.prune_documents() == 1
    hits = store.knn(vec(1, 0, 0, 0), k=10)
    assert {h.document_key for h in hits} == {"doc2"}
    assert store.counts()["documents"] == 1


def test_per_file_commit_survives_reopen(tmp_path):
    """R8: each file is committed in its own transaction — a crash keeps prior files."""
    store = Store(tmp_path / "test.db", create=True)
    fill(store, doc_key="doc1", path="/tmp/f1")
    store.conn.close()  # simulate abrupt end without explicit final commit

    reopened = Store(tmp_path / "test.db")
    assert reopened.file_fingerprints() == {"/tmp/f1": "1:1"}
    assert len(reopened.knn([1, 0, 0, 0], k=10)) == 2
    reopened.close()


def test_status_helpers(store):
    doc_id = fill(store)
    store.replace_file(doc_id, "/tmp/broken", "3:3", "error", [], [])
    counts = store.counts()
    assert (counts["documents"], counts["files"], counts["chunks"]) == (1, 2, 2)
    assert store.skipped_files() == [("/tmp/broken", "error")]
